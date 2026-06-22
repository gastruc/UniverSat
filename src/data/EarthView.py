import json
import os
from datetime import datetime
from os import environ

import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset
from tqdm import tqdm

from src import utils
from src.data.utils import load_norm, apply_norm

log = utils.get_pylogger(__name__)

os.environ["HF_DATASETS_OFFLINE"] = "1"

DATASET = "satellogic/EarthView"

sets = {
    "satellogic": {
        "shards" : 7863,
    },
    "sentinel_1": {
        "shards" : 1763,
    },
    "neon": {
        "config" : "default",
        "shards" : 607,
        "path"   : "data",
    },
    "sentinel_2": {
        "shards" : 19997,
    },
}

def convert_date(date):
    """Convert a list of date strings to day-of-year offsets."""
    to_datetime = lambda x : datetime(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:10]))
    date = list(map(to_datetime, date))
    date = list(map(lambda x : (x - datetime(x.year, 1, 1)).days + 1, date))
    return date

def get_subsets():
    return sets.keys()
def get_nshards(subset):
    return sets[subset]["shards"]
def get_path(subset):
    return sets[subset].get("path", subset)
def get_config(subset):
    return sets[subset].get("config", subset)

class _NormWorkerDataset(IterableDataset):
    """Lightweight IterableDataset for parallel normalization computation.

    Each DataLoader worker creates its own minimal HuggingFace streaming pipeline
    over a disjoint subset of parquet shards, avoiding the need to fork/pickle the
    full EarthView object (which causes OOM).
    """
    MODALITY_MAP = {"neon": "1m", "ndemneon": "chm", "rgbneon": "rgb"}

    def __init__(self, path, config, nshards, modalities):
        super().__init__()
        self.path = path
        self.config = config
        self.nshards = nshards
        self.modalities = modalities

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        all_shards = list(range(self.nshards))

        if worker_info is not None:
            # Split shards evenly across workers so each processes a disjoint set
            n_workers = worker_info.num_workers
            per_worker = (len(all_shards) + n_workers - 1) // n_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(all_shards))
            shards = all_shards[start:end]
        else:
            shards = all_shards

        if len(shards) == 0:
            return

        data_files = [
            f"{self.path}/train-{s:05d}-of-{self.nshards:05d}.parquet"
            for s in shards
        ]

        ds = load_dataset(
            path=self.path,
            name=self.config,
            split="train",
            data_files=data_files,
            streaming=True,
            token=environ.get("HF_TOKEN", None),
        )

        for sample in ds:
            output = {}
            for mod in self.modalities:
                raw_key = self.MODALITY_MAP.get(mod, mod)
                if raw_key in sample:
                    output[mod] = torch.FloatTensor(sample[raw_key])
            yield output


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name" and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["neon", "rgbneon", "ndemneon"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        assert isinstance(batch[0][key], torch.Tensor), f"Key {key} is not a tensor."
        output[key] = torch.stack([x[key] for x in batch])
    return output

class EarthView(IterableDataset):
    """Streaming EarthView (NEON) dataset served from HuggingFace parquet shards.

    Args:
        path: dataset root containing the per-subset parquet shards.
        subset: which EarthView subset to stream (e.g. ``"neon"``).
        split: kept for API parity; EarthView only ships a train split.
        modalities: modality names to expose (mapped to the raw parquet keys).
        norm_path: directory of normalisation statistics; computed on the fly if missing.
    """

    def __init__(self, path, subset="neon", split="train", modalities=[], norm_path=None):
        super().__init__()
        self.subset = subset
        if split in ["val", "validation", "test"]:
            log.warning(f"EarthView has no validation or test set, using train set instead.")
        self.path  = os.path.join(path, get_path(subset))
        self.modalities = modalities
        self.collate_fn = collate_fn

        self.norm = None
        self.norm_path = norm_path

        config = get_config(subset)
        nshards = get_nshards(subset)

        data_files = [f"{self.path}/train-{shard:05d}-of-{nshards:05d}.parquet" for shard in range(nshards)]

        ds = load_dataset(
            path=self.path,
            name=config,
            save_infos=True,
            split="train",
            data_files=data_files,
            streaming=True,
            token=environ.get("HF_TOKEN", None))

        ds = split_dataset_by_node(
            ds,
            rank=int(environ.get("RANK", 0)),
            world_size=int(environ.get("WORLD_SIZE", 1))
        )

        self.dataset =  ds.shuffle(buffer_size=100, seed=42).map(self.process_data, remove_columns=["1m", "chm", "rgb", "metadata"])

        # Load normalization values if a path is provided; compute them first if missing
        if norm_path is not None:
            missing = any(
                not os.path.exists(os.path.join(norm_path, f"NORM_{mod}_patch.json"))
                for mod in self.modalities
            )
            if missing:
                log.info("Normalization files not found, computing them from data...")
                self.compute_norm_vals(norm_path, n_samples=self.__len__())
            self.norm = load_norm(norm_path, self.modalities)

    def compute_norm_vals(self, norm_path, n_samples=10000, num_workers=8):
        """
        Computes the normalization values (mean, std per channel) for each modality
        using a lightweight parallel DataLoader and saves them to JSON files.

        Uses _NormWorkerDataset so that each DataLoader worker builds its own minimal
        streaming pipeline over a disjoint subset of shards, avoiding the OOM caused
        by forking the full EarthView state.

        Args:
            norm_path (str): Directory where NORM_<modality>_patch.json files will be saved.
            n_samples (int): Maximum number of samples to use for computing statistics.
            num_workers (int): Number of parallel DataLoader workers.
        """
        if not os.path.exists(norm_path):
            os.makedirs(norm_path)

        config = get_config(self.subset)
        nshards = get_nshards(self.subset)

        norm_ds = _NormWorkerDataset(self.path, config, nshards, self.modalities)

        loader = torch.utils.data.DataLoader(
            norm_ds,
            batch_size=1,
            num_workers=num_workers,
        )

        log.info(f"Computing normalization values for EarthView modalities: {self.modalities} "
                 f"({n_samples} samples, {num_workers} workers)...")

        means = {m: [] for m in self.modalities}
        stds = {m: [] for m in self.modalities}
        last_sample = None

        count = 0
        for batch in tqdm(loader, total=n_samples, desc="Computing Norm Stats"):
            if count >= n_samples:
                break
            last_sample = batch

            for modality in self.modalities:
                if modality not in batch or modality == 'ndemneon':
                    continue
                x = batch[modality].squeeze(0).float()  # Remove batch dimension

                if x.ndim == 4:  # (T, C, H, W)
                    means[modality].append(x.mean(dim=(0, 2, 3)).numpy())
                    stds[modality].append(x.std(dim=(0, 2, 3)).numpy())
                elif x.ndim == 3:  # (C, H, W)
                    means[modality].append(x.mean(dim=(1, 2)).numpy())
                    stds[modality].append(x.std(dim=(1, 2)).numpy())

            count += 1

        if 'ndemneon' in self.modalities and last_sample is not None and 'ndemneon' in last_sample:
            means['ndemneon'] = [torch.zeros(last_sample['ndemneon'].squeeze(0).shape[-3])]
            stds['ndemneon'] = [torch.ones(last_sample['ndemneon'].squeeze(0).shape[-3])]

        for modality in self.modalities:
            if len(means[modality]) == 0:
                log.warning(f"No data found for modality {modality}, skipping normalization computation.")
                continue

            m = np.stack(means[modality]).mean(axis=0).astype(float)
            s = np.stack(stds[modality]).mean(axis=0).astype(float)

            file_path = os.path.join(norm_path, f"NORM_{modality}_patch.json")
            with open(file_path, "w") as f:
                json.dump({"mean": list(m), "std": list(s)}, f, indent=4)

            log.info(f"Saved normalization for {modality} -> {file_path}")

    def process_data(self, sample):
        # Rename the raw parquet keys to our modality names, dropping unused ones.
        if "neon" in self.modalities:
            sample["neon"] = torch.FloatTensor(sample.pop("1m", None))
        else:
            sample.pop("1m", None)
        if "ndemneon" in self.modalities:
            sample["ndemneon"] = torch.FloatTensor(sample.pop("chm", None))
        else:
            sample.pop("chm", None)
        if "rgbneon" in self.modalities:
            sample["rgbneon"] = torch.FloatTensor(sample.pop("rgb", None))
        else:
            sample.pop("rgb", None)

        metadata = sample.pop("metadata")
        dates = convert_date(metadata["timestamp"])

        for modality in self.modalities:
            sample[f"{modality}_dates"] = torch.FloatTensor(dates)

        if self.norm is not None:
            self.norm.pop("ndemneon", None) # Don't apply normalization to ndemneon (CHM)
            sample = apply_norm(self.norm, sample)

        return sample

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return 35501 #according to paper
