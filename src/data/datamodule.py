import numbers
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import lightning as L
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from src import utils

log = utils.get_pylogger(__name__)

# Enforce "spawn" start method to avoid CUDA initialization in forked workers.
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class DataLoaderStop(DataLoader):
    """DataLoader that stops after ``stop_iteration`` batches."""

    def __init__(self, stop_iteration: Optional[int] = None, **kwargs):
        self.stop_iteration = stop_iteration
        super().__init__(**kwargs)

    def __len__(self):
        base_len = super().__len__()
        if self.stop_iteration is not None:
            return min(base_len, self.stop_iteration)
        return base_len


class DataModule(L.LightningDataModule):
    """Single-dataset Lightning data module.

    Builds train/val/test datasets lazily from the provided builder callables and
    serves them through :class:`DataLoaderStop` loaders, splitting the global batch
    size evenly across nodes and devices.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        global_batch_size,
        num_workers,
        num_nodes=1,
        num_devices=1,
        stop_iteration_train=None,
        stop_iteration_val=None,
        persistent_workers: bool = True,
        prefetch_factor: int = 1,
        verbose=True,
    ):
        super().__init__()
        self.verbose = verbose
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.stop_iteration_train = stop_iteration_train
        self.stop_iteration_val = stop_iteration_val
        self.batch_size = global_batch_size // (num_nodes * num_devices)
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        if self.verbose:
            print(f"Each GPU will receive {self.batch_size} images")
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        return self._builders["train"]().num_classes

    def setup(self, stage=None):
        if self.verbose:
            print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            if self.verbose:
                print(f"Train dataset size: {len(self.train_dataset)}")
                print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            if self.verbose:
                print(f"Test dataset size: {len(self.test_dataset)}")
        if self.verbose:
            print(f"Setup took {(time.time() - start_time):.2f} seconds")

    def _loader_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        if self.prefetch_factor is not None:
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def train_dataloader(self):
        return DataLoaderStop(
            dataset=self.train_dataset,
            stop_iteration=self.stop_iteration_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoaderStop(
            dataset=self.val_dataset,
            stop_iteration=self.stop_iteration_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.val_dataset.collate_fn,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoaderStop(
            dataset=self.test_dataset,
            stop_iteration=None,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.test_dataset.collate_fn,
            **self._loader_kwargs(),
        )


def _allocate_workers(
    total: int, names: List[str], weights: Optional[Dict[str, float]] = None
) -> Dict[str, int]:
    """Split ``total`` workers across ``names``, optionally proportional to ``weights``.

    Ensures every active dataset gets >= 1 worker and the sum never exceeds ``total``.
    """
    n = len(names)
    if n == 0:
        return {}
    if total <= 0:
        return {name: 0 for name in names}
    if total <= n:
        # Not enough budget to give everyone one without going over; still give 1 each.
        # (Accepting a small oversubscription is usually better than starving a loader.)
        return {name: 1 for name in names}

    if not weights or sum(max(weights.get(n_, 0), 0) for n_ in names) <= 0:
        base = total // n
        extra = total - base * n
        return {
            name: base + (1 if i < extra else 0) for i, name in enumerate(names)
        }

    raw = {name: max(float(weights.get(name, 0)), 0.0) for name in names}
    # Floor > 0 so no active dataset ever gets starved of workers.
    assigned = {name: 1 for name in names}
    remaining = total - n
    weight_sum = sum(raw.values()) or 1.0
    # Proportional share of the *remaining* budget.
    floats = {
        name: remaining * raw[name] / weight_sum for name in names
    }
    extras = {name: int(floats[name]) for name in names}
    leftover = remaining - sum(extras.values())
    # Hand out leftover workers to the largest fractional remainders.
    frac = sorted(
        names, key=lambda m: floats[m] - extras[m], reverse=True
    )
    for i in range(leftover):
        extras[frac[i % n]] += 1
    for name in names:
        assigned[name] += extras[name]
    return assigned


class DataModuleMulti(L.LightningDataModule):
    """Multi-dataset Lightning data module for joint pretraining.

    Wraps a :class:`~data.MultiDataset.MultiDataset` and serves one per-dataset
    DataLoader each, multiplexed by :class:`MultiBatchDataLoader`. Workers are
    distributed across the active datasets per epoch (honouring the curriculum
    ``weights_datasets`` ramp), and loaders are cached so persistent workers
    survive across epochs.
    """

    def __init__(
        self,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        global_batch_size: Union[int, Dict[str, int]],
        num_workers: int,
        num_nodes: int = 1,
        num_devices: int = 1,
        stop_iteration_train: Optional[int] = None,
        stop_iteration_val: Optional[int] = None,
        weights_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 1,
    ):
        super().__init__()
        self._builders = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        self.num_workers = num_workers
        self.stop_iteration_train = stop_iteration_train
        self.stop_iteration_val = stop_iteration_val
        self.batch_size = global_batch_size
        self.weights_datasets = weights_datasets
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

        # Cache per-split DataLoaders so persistent workers survive across epochs.
        # Key: dataset name; value: tuple (num_workers, DataLoader).
        self._train_cache: Dict[str, Tuple[int, DataLoader]] = {}
        self._val_cache: Dict[str, DataLoader] = {}
        self._test_cache: Dict[str, DataLoader] = {}

        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        if hasattr(self, "train_dataset"):
            return self.train_dataset.num_classes
        return self._builders["train"]().num_classes

    def setup(self, stage=None):
        print("Stage", stage)
        start_time = time.time()
        if stage == "fit" or stage is None:
            self.train_dataset = self._builders["train"]()
            self.val_dataset = self._builders["val"]()
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        else:
            self.test_dataset = self._builders["test"]()
            print(f"Test dataset size: {len(self.test_dataset)}")
        print(f"Setup took {(time.time() - start_time):.2f} seconds")

    def on_train_epoch_start(self) -> None:
        self.current_epoch = self.trainer.current_epoch

    def teardown(self, stage: Optional[str] = None) -> None:
        # Drop cached loaders so workers are cleaned up between stages.
        if stage in (None, "fit"):
            self._train_cache.clear()
            self._val_cache.clear()
        if stage in (None, "test"):
            self._test_cache.clear()

    def _make_loader(
        self,
        multi_dataset: Any,
        name: str,
        num_workers: int,
        *,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        module = multi_dataset.datasets_modules[name]
        kwargs: Dict[str, Any] = dict(
            batch_size=self.batch_size[name],
            pin_memory=True,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=multi_dataset.collate_fn[name],
            persistent_workers=self.persistent_workers and num_workers > 0,
        )
        if self.prefetch_factor is not None and num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor

        if isinstance(module, IterableDataset):
            # IterableDataset: no sampler, no shuffle kwarg allowed.
            pass
        elif dist.is_initialized():
            kwargs["sampler"] = DistributedSampler(
                module,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
            )
        else:
            kwargs["shuffle"] = shuffle
        return DataLoader(module, **kwargs)

    def _epoch_active_datasets(
        self, epoch: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """Return (active dataset names, base weights) for the given epoch.

        Base weights use the *target* ``weight`` (ignoring the curriculum ramp) so
        that a dataset ramping in doesn't start with zero workers.
        """
        available = list(self.train_dataset.datasets)
        if self.weights_datasets is None:
            return available, {name: 1.0 for name in available}

        names: List[str] = []
        base_weights: Dict[str, float] = {}
        available_set = set(available)
        for name, cfg in self.weights_datasets.items():
            if name not in available_set:
                continue
            if cfg.get("start_epoch", 0) > epoch:
                continue
            names.append(name)
            base_weights[name] = float(cfg.get("weight", 1))
        return names, base_weights

    def train_dataloader(self):
        epoch = self.trainer.current_epoch
        names, base_weights = self._epoch_active_datasets(epoch)
        allocation = _allocate_workers(self.num_workers, names, base_weights)
        log.info(
            f"Train epoch {epoch}: datasets {names} with workers {allocation} "
            f"(persistent={self.persistent_workers})."
        )

        # Drop cached loaders for datasets that are no longer active or whose
        # worker count changed; reuse the rest so their worker pools survive.
        for name in list(self._train_cache):
            if name not in allocation or self._train_cache[name][0] != allocation[name]:
                del self._train_cache[name]
        for name in names:
            if name not in self._train_cache:
                loader = self._make_loader(
                    self.train_dataset, name, allocation[name],
                    shuffle=True, drop_last=True,
                )
                self._train_cache[name] = (allocation[name], loader)

        dls = {name: self._train_cache[name][1] for name in names}
        return MultiBatchDataLoader(
            dataloaders=dls,
            cycle=True,
            scales=self.train_dataset.scales,
            stop_iteration=self.stop_iteration_train,
            weights_datasets=self.weights_datasets,
            epoch=epoch,
        )

    def val_dataloader(self):
        names = list(self.val_dataset.datasets)
        allocation = _allocate_workers(self.num_workers, names)
        missing = [n for n in names if n not in self._val_cache]
        if missing:
            log.info(
                f"Validation: building loaders for {missing} with workers "
                f"{ {n: allocation[n] for n in missing} }."
            )
            for name in missing:
                self._val_cache[name] = self._make_loader(
                    self.val_dataset, name, allocation[name],
                    shuffle=False, drop_last=True,
                )
        dls = {name: self._val_cache[name] for name in names}
        return MultiBatchDataLoader(
            dataloaders=dls,
            cycle=False,
            scales=self._strip_curiculum(self.val_dataset.scales),
            stop_iteration=self.stop_iteration_val,
            weights_datasets=None,
            epoch=30,
        )

    def test_dataloader(self):
        names = list(self.test_dataset.datasets)
        allocation = _allocate_workers(self.num_workers, names)
        missing = [n for n in names if n not in self._test_cache]
        if missing:
            log.info(
                f"Test: building loaders for {missing} with workers "
                f"{ {n: allocation[n] for n in missing} }."
            )
            for name in missing:
                self._test_cache[name] = self._make_loader(
                    self.test_dataset, name, allocation[name],
                    shuffle=False, drop_last=False,
                )
        dls = {name: self._test_cache[name] for name in names}
        return MultiBatchDataLoader(
            dataloaders=dls,
            cycle=False,
            scales=self._strip_curiculum(self.test_dataset.scales),
            stop_iteration=None,
            weights_datasets=None,
        )

    @staticmethod
    def _strip_curiculum(
        scales: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, List[int]]]:
        return {
            name: {k: cfg[k] for k in ("input_scales", "latent_scales", "output_scales")}
            for name, cfg in scales.items()
        }


class MultiBatchDataLoader(DataLoader):
    """Round-robin / weighted multiplexer over several per-dataset DataLoaders."""

    _SCALE_KEYS = ("input", "latent", "output")

    def __init__(
        self,
        dataloaders: Dict[str, DataLoader],
        scales: Dict[str, Dict[str, Any]],
        cycle: bool = True,
        stop_iteration: Optional[int] = None,
        weights_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
        epoch: int = 0,
        **kwargs,
    ):
        if not dataloaders:
            raise ValueError("dataloaders dictionary cannot be empty")
        if not all(isinstance(dl, DataLoader) for dl in dataloaders.values()):
            raise TypeError("All values in dataloaders must be DataLoader instances")
        missing = set(dataloaders).difference(scales)
        if missing:
            raise AssertionError(f"Some datasets do not have scales defined: {missing}")

        self.dataloaders = dataloaders
        self.dataset_names: List[str] = list(dataloaders.keys())
        self.scales = {name: scales[name] for name in self.dataset_names}
        self.cycle = cycle
        self.stop_iteration = stop_iteration
        self.weights_datasets = weights_datasets
        self.epoch = epoch

        total_batches = sum(len(dl) for dl in dataloaders.values())
        if cycle:
            self.length = stop_iteration if stop_iteration is not None else total_batches
        else:
            self.length = (
                total_batches if stop_iteration is None else min(total_batches, stop_iteration)
            )

        self.iterators: Dict[str, Iterator] = {}
        self.exhausted_loaders: Set[str] = set()
        self._epoch_weights: Dict[str, float] = {name: 1.0 for name in self.dataset_names}

        super().__init__(dataset=range(self.length), batch_size=1, shuffle=False)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.epoch += 1
        if dist.is_initialized():
            for dl in self.dataloaders.values():
                sampler = getattr(dl, "sampler", None)
                set_epoch = getattr(sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(self.epoch)

        self.iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
        self.exhausted_loaders = set()
        self._epoch_weights = self._compute_epoch_weights()
        return self

    def _compute_epoch_weights(self) -> Dict[str, float]:
        if not self.weights_datasets:
            return {name: 1.0 for name in self.dataset_names}

        weights: Dict[str, float] = {name: 0.0 for name in self.dataset_names}
        for name, cfg in self.weights_datasets.items():
            if name not in weights:
                continue
            start = cfg.get("start_epoch", 0)
            end = cfg.get("end_epoch", 0)
            w = float(cfg.get("weight", 1))
            if self.epoch < start:
                weights[name] = 0.0
            elif self.epoch >= end:
                weights[name] = w
            else:
                weights[name] = w * (self.epoch - start) / (end - start)

        assert sum(weights.values()) > 0, "At least one dataset must have a weight greater than 0."
        return weights

    def _sample_dataset(self):
        weights = self._epoch_weights
        available = [
            name
            for name in self.dataset_names
            if name not in self.exhausted_loaders and weights.get(name, 0) > 0
        ]
        if not available:
            if not self.cycle:
                raise StopIteration
            self.exhausted_loaders.clear()
            self.iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
            available = [n for n in self.dataset_names if weights.get(n, 0) > 0]

        if len(available) == 1:
            name = available[0]
        else:
            name = random.choices(
                available, weights=[weights[n] for n in available], k=1
            )[0]

        total = sum(weights[n] for n in available)
        weight_dict = {
            f"weight/{n}": (weights[n] / total if n in available else 0.0)
            for n in self.dataset_names
        }
        weight_dict["weight/selected_dataset"] = available.index(name)
        return name, weight_dict

    def __next__(self):
        while True:
            if not self.cycle and len(self.exhausted_loaders) == len(self.dataloaders):
                raise StopIteration
            name, weight_dict = self._sample_dataset()
            try:
                batch = next(self.iterators[name])
            except StopIteration:
                self.exhausted_loaders.add(name)
                continue
            return self.add_metadata(batch, name, weight_dict)

    def sample_scale(self, scales_list, curiculum_config):
        if not scales_list:
            return None
        if curiculum_config is None or len(scales_list) == 1:
            return random.choice(scales_list)
        start = curiculum_config["start_epoch"]
        end = curiculum_config["end_epoch"]
        if self.epoch <= start:
            p_other = 0.0
        elif self.epoch >= end:
            p_other = 1.0
        else:
            p_other = (self.epoch - start) / (end - start)
        weights = [1.0] + [p_other] * (len(scales_list) - 1)
        return random.choices(scales_list, weights=weights, k=1)[0]

    def _resolve_scales(self, dataset_name: str) -> Dict[str, Any]:
        config = self.scales[dataset_name]
        scales = {
            k: self.sample_scale(config[f"{k}_scales"], config.get(f"{k}_curiculum"))
            for k in self._SCALE_KEYS
        }
        for k in self._SCALE_KEYS:
            v = scales[k]
            if isinstance(v, str) and v in self._SCALE_KEYS:
                ref = scales[v]
                assert isinstance(ref, numbers.Number), (
                    f"{k.capitalize()} scale cannot reference '{v}' if that scale is not a number."
                )
                scales[k] = ref
        return scales

    def add_metadata(self, batch, dataset_name, weight_dict):
        scales = self._resolve_scales(dataset_name)
        if isinstance(batch, dict):
            batch["logging"] = weight_dict
            batch["dataset"] = dataset_name
            batch["input_scale"] = scales["input"]
            batch["latent_scale"] = scales["latent"]
            batch["output_scale"] = scales["output"]
        elif isinstance(batch, (list, tuple)):
            batch = (
                *batch,
                dataset_name,
                scales["input"],
                scales["latent"],
                scales["output"],
            )
        return batch
