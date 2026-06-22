import gc
import itertools
import json
import math
import os
import random
import tempfile
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import hydra
import numpy as np
import pyrootutils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

OmegaConf.register_new_resolver("eval", eval)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tqdm.auto import tqdm

from src import utils

torch.set_float32_matmul_precision("high")

warnings.filterwarnings(
    "ignore",
    message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors.",
    category=UserWarning,
)


def safe_print(*args, flush: bool = True, **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, flush=flush, **kwargs)


class ImageGroupedPatchBatchSampler:
    """Batch indices with image-major order: shuffle **images**, keep patch rows per image contiguous."""

    def __init__(
        self,
        dataset_size: int,
        patches_per_image: int,
        batch_rows: int,
        *,
        shuffle: bool,
        drop_last: bool,
        generator: Optional[torch.Generator] = None,
    ):
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be >= 1")
        if dataset_size % patches_per_image != 0:
            raise ValueError(
                f"dataset length {dataset_size} must be divisible by patches_per_image={patches_per_image} "
                "(same patch count per image required)."
            )
        self.dataset_size = int(dataset_size)
        self.patches_per_image = int(patches_per_image)
        self.n_img = self.dataset_size // self.patches_per_image
        self.batch_rows = max(int(batch_rows), self.patches_per_image)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator

    def _effective_rows(self) -> int:
        P = self.patches_per_image
        k = max(1, self.batch_rows // P)
        return k * P

    def __len__(self) -> int:
        eff = self._effective_rows()
        if eff <= 0:
            return 0
        n = self.dataset_size
        if self.drop_last:
            return n // eff
        return (n + eff - 1) // eff

    def __iter__(self) -> Iterator[List[int]]:
        P = self.patches_per_image
        n = self.dataset_size
        eff = self._effective_rows()
        img_order = (
            torch.randperm(self.n_img, generator=self.generator).tolist()
            if self.shuffle
            else list(range(self.n_img))
        )
        flat: List[int] = []
        for gi in img_order:
            base = gi * P
            flat.extend(range(base, base + P))

        pos = 0
        while pos + eff <= n:
            yield flat[pos : pos + eff]
            pos += eff
        if not self.drop_last and pos < n:
            yield flat[pos:n]


class InMemoryLogitsDataset(Dataset):
    """Simple in-memory dataset for logits and labels."""

    def __init__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        patches_per_image: Optional[int] = None,
    ):
        self.logits = logits
        self.labels = labels
        self.patches_per_image = patches_per_image

    def to(self, device):
        self.logits = self.logits.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)
        return self

    def __len__(self):
        return int(self.logits.shape[0])

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]


class MemMapLogitsDataset(Dataset):
    """Disk-backed dataset for logits and labels."""

    def __init__(
        self,
        logits_path: str,
        labels_path: str,
        logits_shape: tuple[int, ...],
        labels_shape: tuple[int, ...],
        logits_dtype: np.dtype,
        labels_dtype: np.dtype,
        length: int,
        patches_per_image: Optional[int] = None,
    ):
        self.logits_path = logits_path
        self.labels_path = labels_path
        self.logits = np.memmap(logits_path, mode="r+", dtype=logits_dtype, shape=logits_shape)
        self.labels = np.memmap(labels_path, mode="r+", dtype=labels_dtype, shape=labels_shape)
        self.length = int(length)
        self.patches_per_image = patches_per_image

    def to(self, device):
        raise RuntimeError(
            "MemMapLogitsDataset is disk-backed and cannot be moved as a whole to GPU. "
            "Set `param.move_logits_to_gpu=False`."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # copy=True avoids non-writable numpy warnings with torch.from_numpy
        x = torch.from_numpy(np.array(self.logits[idx], copy=True))
        y = torch.from_numpy(np.array(self.labels[idx], copy=True))
        return x, y

    def close(self):
        # Best-effort cleanup (important on some HPC filesystems).
        try:
            if hasattr(self.logits, "_mmap") and self.logits._mmap is not None:
                self.logits._mmap.close()
        except Exception:
            pass
        try:
            if hasattr(self.labels, "_mmap") and self.labels._mmap is not None:
                self.labels._mmap.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ChunkedMemmapLoader:
    """Zero-worker chunked loader for large memmap datasets.

    Motivation: for 100M+ row memmaps (>100 GB on disk), using a standard
    DataLoader with `num_workers>0` causes CPU OOM on clustered filesystems
    (Lustre + cgroup limits) because:
      - Each forked worker duplicates page tables for the full mmap region
        (~480 MB per process for a 237 GB mapping).
      - `pin_memory=True` locks prefetched batches in RAM, competing with
        mmap page cache under cgroup pressure.
      - `__getitem__` per-row access issues 50K independent 4 KB page faults
        per batch, flooding the page cache with non-contiguous pages.

    Optimizations vs. naive sync chunk reading:
      - Background prefetch thread overlaps disk I/O with GPU training,
        so the loader is never the bottleneck once the pipeline is warm.
      - chunk_size is rounded down to a multiple of batch_size, eliminating
        cross-chunk leftover concatenation copies (saves ~1.5 GB memcpy
        per chunk transition).
      - posix_fadvise(POSIX_FADV_SEQUENTIAL) hints the kernel to perform
        aggressive readahead (Linux only, fails silently elsewhere).
      - posix_fadvise(POSIX_FADV_DONTNEED) drops consumed chunk pages from
        the page cache, keeping RAM footprint bounded under cgroup limits.
      - Single contiguous numpy slice per chunk → sequential I/O (10-100x
        faster on Lustre than scattered random row-by-row reads).

    Supports the subset of DataLoader interface used by the probe training
    loops: __iter__, __len__, and yields (logits, labels) tuples.
    """

    def __init__(
        self,
        dataset: "MemMapLogitsDataset",
        batch_size: int,
        chunk_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
        prefetch: bool = True,
        drop_pages_after_chunk: bool = True,
        shuffle_within_chunk: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        # Round chunk_size down to a multiple of batch_size so each chunk
        # yields whole batches and no cross-chunk leftover concat is needed
        # (the only partial batch is the global tail, when n % chunk_size != 0).
        cs = max(int(chunk_size), int(batch_size))
        self.chunk_size = max((cs // self.batch_size) * self.batch_size, self.batch_size)
        self.shuffle = bool(shuffle)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.prefetch = bool(prefetch)
        self.drop_pages_after_chunk = bool(drop_pages_after_chunk)
        self.n = int(len(dataset))
        # Expose attributes the training code may read (DataLoader-like).
        self.num_workers = 0
        self.pin_memory = False
        # Tell the kernel we'll be reading sequentially (Linux-only hint).
        self._fadvise_sequential()

    def _arr_row_bytes(self, arr) -> int:
        if arr.ndim > 1:
            return int(arr.itemsize) * int(np.prod(arr.shape[1:]))
        return int(arr.itemsize)

    def _fadvise_sequential(self) -> None:
        if not hasattr(os, "posix_fadvise"):
            return
        try:
            for arr_name in ("logits", "labels"):
                arr = getattr(self.dataset, arr_name, None)
                mm = getattr(arr, "_mmap", None) if arr is not None else None
                if mm is not None:
                    os.posix_fadvise(mm.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except Exception:
            pass

    def _fadvise_dontneed(self, start: int, end: int) -> None:
        if not self.drop_pages_after_chunk or not hasattr(os, "posix_fadvise"):
            return
        try:
            for arr_name in ("logits", "labels"):
                arr = getattr(self.dataset, arr_name, None)
                mm = getattr(arr, "_mmap", None) if arr is not None else None
                if mm is None:
                    continue
                rb = self._arr_row_bytes(arr)
                os.posix_fadvise(
                    mm.fileno(),
                    int(start) * rb,
                    int(end - start) * rb,
                    os.POSIX_FADV_DONTNEED,
                )
        except Exception:
            pass

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def _load_chunk(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # One contiguous numpy slice; .copy() detaches from the mmap so the OS
        # can reclaim those pages (which we then explicitly hint via DONTNEED).
        x = torch.from_numpy(np.asarray(self.dataset.logits[start:end]).copy())
        y = torch.from_numpy(np.asarray(self.dataset.labels[start:end]).copy())
        return x, y

    def _yield_from_chunk(
        self, x: torch.Tensor, y: torch.Tensor, chunk_start: int, chunk_end: int
    ):
        bs = self.batch_size
        if self.shuffle and self.shuffle_within_chunk:
            perm = torch.randperm(x.shape[0], generator=self.generator)
            x = x[perm]
            y = y[perm]
        pos = 0
        total = x.shape[0]
        while pos + bs <= total:
            yield x[pos : pos + bs], y[pos : pos + bs]
            pos += bs
        if not self.drop_last and pos < total:
            yield x[pos:total], y[pos:total]
        # Drop now-unused mmap pages from page cache.
        self._fadvise_dontneed(chunk_start, chunk_end)

    def __iter__(self):
        n = self.n
        cs = self.chunk_size
        chunk_starts = list(range(0, n, cs))
        if self.shuffle:
            chunk_order = torch.randperm(
                len(chunk_starts), generator=self.generator
            ).tolist()
        else:
            chunk_order = list(range(len(chunk_starts)))

        if self.prefetch and len(chunk_order) > 1:
            return self._iter_prefetched(chunk_starts, chunk_order)
        return self._iter_sync(chunk_starts, chunk_order)

    def _iter_sync(self, chunk_starts, chunk_order):
        for ci in chunk_order:
            start = chunk_starts[ci]
            end = min(start + self.chunk_size, self.n)
            x, y = self._load_chunk(start, end)
            yield from self._yield_from_chunk(x, y, start, end)
            del x, y

    def _iter_prefetched(self, chunk_starts, chunk_order):
        import queue
        import threading

        # Producer holds at most 1 chunk ahead → peak ~2 chunks in RAM.
        q: "queue.Queue" = queue.Queue(maxsize=1)
        stop = threading.Event()

        def producer():
            try:
                for ci in chunk_order:
                    if stop.is_set():
                        return
                    start = chunk_starts[ci]
                    end = min(start + self.chunk_size, self.n)
                    chunk = self._load_chunk(start, end)
                    # Block until consumer takes; check stop periodically.
                    while not stop.is_set():
                        try:
                            q.put((start, end, chunk), timeout=0.5)
                            break
                        except queue.Full:
                            continue
                if not stop.is_set():
                    q.put(None)
            except Exception as exc:
                try:
                    q.put(exc, timeout=1.0)
                except Exception:
                    pass

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                start, end, (x, y) = item
                yield from self._yield_from_chunk(x, y, start, end)
                del x, y
        finally:
            stop.set()
            # Drain queue so the producer can exit if blocked on put().
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5.0)


def mean_iou(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_label: int = -1
):
    """
    Calculate mean IoU given prediction and label tensors, ignoring pixels with a specific label.

    Args:
    predictions (torch.Tensor): Predicted segmentation masks of shape (N, H, W)
    labels (torch.Tensor): Ground truth segmentation masks of shape (N, H, W)
    num_classes (int): Number of classes in the segmentation task
    ignore_label (int): Label value to ignore in IoU calculation (default: -1)

    Returns:
    float: Mean IoU across all classes
    """
    # Ensure inputs are on the same device
    device = predictions.device
    labels = labels.to(device)

    # Initialize tensors to store intersection and union for each class
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    # Create a mask for valid pixels (i.e., not ignore_label)
    valid_mask = labels != ignore_label

    # Iterate through each class
    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (predictions == class_id) & valid_mask
        label_mask = (labels == class_id) & valid_mask

        # Calculate intersection and union
        intersection[class_id] = (pred_mask & label_mask).sum().float()
        union[class_id] = (pred_mask | label_mask).sum().float()

    # Calculate IoU for each class
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero

    # Calculate mean IoU (excluding classes with zero union)
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()

    return mean_iou.item()

def compute_micro_iou(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_label: int = -1
):
    device = predictions.device
    labels = labels.to(device)

    total_intersection = torch.tensor(0.0, device=device)
    total_union = torch.tensor(0.0, device=device)

    valid_mask = labels != ignore_label

    for class_id in range(num_classes):
        pred_mask = (predictions == class_id) & valid_mask
        label_mask = (labels == class_id) & valid_mask

        total_intersection += (pred_mask & label_mask).sum().float()
        total_union += (pred_mask | label_mask).sum().float()

    return (total_intersection / (total_union + 1e-8)).item()

def get_logits(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    name: str,
    sensor,
    num_patches,
    res,
    scale=1,
    task="classification",
    device: str = "cpu",
    fraction=1.0,
    patch_size: Optional[int] = None,
    logits_dtype: str = "float16",
    pooling: str = "mean",
    empty_cache_every_n_batches: int = 0,
    use_memmap: bool = False,
    memmap_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get logits from model and dataloader, keeping everything in memory (RAM/VRAM)."""

    def _to_number(value, name):
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    return eval(value, {"__builtins__": {}}, {})
                except Exception as exc:
                    raise ValueError(f"{name} must be numeric, got: {value}") from exc
        raise ValueError(f"{name} must be numeric, got: {type(value)}")

    scale = _to_number(scale, "scale")
    res = float(_to_number(res, "res"))
    num_patches = int(_to_number(num_patches, "num_patches"))
    if patch_size is not None:
        patch_size = int(_to_number(patch_size, "patch_size"))

    if logits_dtype not in {"float16", "float32"}:
        raise ValueError("logits_dtype must be 'float16' or 'float32'")
    if pooling not in {"mean", "max"}:
        raise ValueError("pooling must be 'mean' or 'max'")

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    out_patch = (num_patches * res**2) // (10 * scale) ** 2 if task == "classification" else num_patches// patch_size ** 2
    model.eval()

    device_type = "cuda" if "cuda" in device else "cpu"

    logits_torch_dtype = torch.float16 if logits_dtype == "float16" else torch.float32
    labels_torch_dtype = torch.int8

    logits_storage = None
    labels_storage = None
    logits_path = None
    labels_path = None
    write_idx = 0
    samples_per_item = None  # number of rows produced per original sample (1 for cls)
    total_capacity = None
    fraction_cursor = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc=f"Getting logits (Rank {rank})",
                total=len(dataloader),
                mininterval=10,
                disable=rank != 0,
            )
        ):
            tensor_keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]
            batch_len = int(batch[tensor_keys[0]].shape[0]) if tensor_keys else 1
            mb = batch_len
            for mb_start in range(0, batch_len, mb):
                mb_end = min(batch_len, mb_start + mb)
                micro_batch = {
                    k: (v[mb_start:mb_end] if isinstance(v, torch.Tensor) and v.shape[0] == batch_len else v)
                    for k, v in batch.items()
                }
                x = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in micro_batch.items()}
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if hasattr(model, "extract_lp_logits"):
                        logit = model.extract_lp_logits(x, device=device)
                    else:
                        logit = model(
                            x,
                            sensor.wavelengths,
                            sensor.input_res,
                            scale,
                            (num_patches * res**2) // (10 * scale) ** 2,
                            out_patch,
                            sensor.subpatches,
                            keep_subpatch=False,
                        )[0]

                if task in ["segmentation", "regression"]:
                    logit = logit[:, model.n_registers:, :]
                elif task == "classification":
                    patch_tokens = logit[:, model.n_registers:, :]
                    if pooling == "mean":
                        logit = patch_tokens.mean(dim=1)
                    else:  # max
                        logit = patch_tokens.max(dim=1).values

                if task in ["segmentation", "regression"]:
                    if patch_size is None:
                        raise ValueError("patch_size must be provided for segmentation/regression")
                    batch_labels_t = micro_batch["label"]
                    if batch_labels_t.ndim == 3:
                        h, w = batch_labels_t.shape[-2:]
                        if h % patch_size != 0 or w % patch_size != 0:
                            raise ValueError("Label spatial dims must be divisible by patch_size for flattening")
                        labels_patches = rearrange(
                            batch_labels_t,
                            "b (hp p1) (wp p2) -> b (hp wp) (p1 p2)",
                            p1=patch_size,
                            p2=patch_size,
                        )
                        labels_flat = labels_patches.reshape(-1, patch_size * patch_size)
                    else:
                        labels_flat = batch_labels_t.reshape(-1, patch_size * patch_size)

                    logits_flat = logit.reshape(-1, logit.shape[-1]).to(logits_torch_dtype)
                    labels_flat = labels_flat.to(labels_torch_dtype)

                    logits_batch_cpu_full = logits_flat.cpu()
                    labels_batch_cpu_full = labels_flat.cpu()
                else:
                    logits_batch_cpu_full = logit.to(logits_torch_dtype).cpu()
                    labels_batch_cpu_full = micro_batch["label"].to(labels_torch_dtype).cpu()

                if logits_storage is None:
                    if task in ["segmentation", "regression"]:
                        samples_per_item = int(labels_batch_cpu_full.shape[0] // batch_labels_t.shape[0])
                    else:
                        samples_per_item = 1
                    if hasattr(dataloader, "dataset") and getattr(dataloader.dataset, "__len__", None):
                        try:
                            total_items = len(dataloader.dataset)
                        except Exception:
                            total_items = None
                    else:
                        total_items = None
                    if total_items is None:
                        batch_size = getattr(dataloader, "batch_size", None) or 1
                        total_items = len(dataloader) * batch_size
                    total_capacity = int(total_items * samples_per_item)
                    logits_shape = (total_capacity, logits_batch_cpu_full.shape[-1])
                    labels_shape = (total_capacity, *labels_batch_cpu_full.shape[1:]) if labels_batch_cpu_full.ndim > 1 else (total_capacity,)
                    if use_memmap:
                        target_dir = memmap_dir or tempfile.gettempdir()
                        os.makedirs(target_dir, exist_ok=True)
                        logits_path = os.path.join(target_dir, f"lp_{name}_rank{rank}_{os.getpid()}_logits.dat")
                        labels_path = os.path.join(target_dir, f"lp_{name}_rank{rank}_{os.getpid()}_labels.dat")
                        np_logits_dtype = np.float16 if logits_torch_dtype == torch.float16 else np.float32
                        np_labels_dtype = np.int8
                        logits_storage = np.memmap(logits_path, mode="w+", dtype=np_logits_dtype, shape=logits_shape)
                        labels_storage = np.memmap(labels_path, mode="w+", dtype=np_labels_dtype, shape=labels_shape)
                    else:
                        logits_storage = torch.empty(logits_shape, dtype=logits_torch_dtype)
                        labels_storage = torch.empty(labels_shape, dtype=labels_torch_dtype)

                logits_batch_cpu = logits_batch_cpu_full
                labels_batch_cpu = labels_batch_cpu_full

                if fraction < 1.0:
                    if not (0.0 < float(fraction) <= 1.0):
                        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
                    if task in ["segmentation", "regression"]:
                        n_rows = logits_batch_cpu.shape[0]
                        row_idx = torch.arange(fraction_cursor, fraction_cursor + n_rows, dtype=torch.float32)
                        keep_mask = ((row_idx * float(fraction)).floor() > ((row_idx - 1.0) * float(fraction)).floor())
                        fraction_cursor += n_rows
                        if not keep_mask.any():
                            continue
                        logits_batch_cpu = logits_batch_cpu[keep_mask]
                        labels_batch_cpu = labels_batch_cpu[keep_mask]

                if logits_batch_cpu.shape[0] != labels_batch_cpu.shape[0]:
                    raise AssertionError(
                        f"Logits/labels count mismatch after preprocessing: logits={logits_batch_cpu.shape}, labels={labels_batch_cpu.shape}."
                    )

                end = write_idx + logits_batch_cpu.shape[0]
                if total_capacity is not None and end > total_capacity:
                    raise RuntimeError("Preallocated logits buffer exceeded; check dataset length estimation.")

                if use_memmap:
                    logits_storage[write_idx:end] = logits_batch_cpu.numpy()
                    labels_storage[write_idx:end] = labels_batch_cpu.numpy()
                else:
                    logits_storage[write_idx:end].copy_(logits_batch_cpu)
                    labels_storage[write_idx:end].copy_(labels_batch_cpu)
                write_idx = end

                del x, logit, logits_batch_cpu_full, labels_batch_cpu_full, logits_batch_cpu, labels_batch_cpu, micro_batch
            if (
                "cuda" in str(device)
                and empty_cache_every_n_batches > 0
                and (batch_idx + 1) % int(empty_cache_every_n_batches) == 0
            ):
                torch.cuda.empty_cache()

    if write_idx == 0:
        raise RuntimeError("No batches were processed; check dataloader or fraction setting.")

    pp_meta = int(samples_per_item) if task in ["segmentation", "regression"] and samples_per_item is not None else None
    if use_memmap:
        logits_storage.flush()
        labels_storage.flush()
        logits_shape_final = (write_idx, logits_storage.shape[1]) if len(logits_storage.shape) == 2 else (write_idx,)
        labels_shape_final = (write_idx, *labels_storage.shape[1:]) if len(labels_storage.shape) > 1 else (write_idx,)
        return MemMapLogitsDataset(
            logits_path=logits_path,
            labels_path=labels_path,
            logits_shape=logits_shape_final,
            labels_shape=labels_shape_final,
            logits_dtype=logits_storage.dtype,
            labels_dtype=labels_storage.dtype,
            length=write_idx,
            patches_per_image=pp_meta,
        )
    logits_all = logits_storage[:write_idx]
    labels_all = labels_storage[:write_idx]
    return InMemoryLogitsDataset(logits_all, labels_all, patches_per_image=pp_meta)

def load_model_from_checkpoint(ckpt_path: str) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        ckpt_path (str): Path to checkpoint.
    """

    config_path = "/".join(ckpt_path.split("/")[:-2]) + "/.hydra/config.yaml"

    cfg = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(cfg.model.network.encoder)

    ckpt = torch.load(ckpt_path, weights_only=False)
    weight = {k.replace("model.encoder.", ""): v for k, v in ckpt["state_dict"].items() if "model.encoder" in k}
    model.load_state_dict(weight)

    path_sensor = "configs/model/network/sensor/default.yaml"
    if os.path.exists(path_sensor):
        safe_print(f"Loading sensor config from {path_sensor}")
        sensor_dict = OmegaConf.load(path_sensor)
    else:
        safe_print(f"No sensor config found at {path_sensor}, using sensor from model training")
        sensor_dict = cfg.model.network.sensor

    return model,sensor_dict


def load_model_from_hf(repo_id: str):
    """Load the released UniverSat encoder from the HuggingFace Hub.

    Uses the hubconf ``from_pretrained`` entrypoint (config.json + model.safetensors)
    and returns the bare encoder -- the same object ``load_model_from_checkpoint``
    yields (forward keeps the register tokens, exposes ``n_registers``) -- paired
    with the default sensor metadata.

    Args:
        repo_id (str): HuggingFace Hub repo id, e.g. "g-astruc/UniverSat".
    """
    import hubconf

    safe_print(f"Loading released UniverSat weights from HuggingFace Hub: {repo_id}")
    model = hubconf.UniverSat.from_pretrained(repo_id).model

    path_sensor = "configs/model/network/sensor/default.yaml"
    safe_print(f"Loading sensor config from {path_sensor}")
    sensor_dict = OmegaConf.load(path_sensor)
    return model, sensor_dict


def load_feature_model(cfg: DictConfig):
    """Load the encoder + sensor metadata.

    Priority: a foundation-model adapter, then a local Lightning checkpoint
    (``cfg.ckpt_path``), else the released weights from the HuggingFace Hub
    (``cfg.hf_repo_id``, default "g-astruc/UniverSat").
    """
    fm_cfg = cfg.get("foundation_model", None)
    if fm_cfg and fm_cfg.get("enabled", False):
        model = hydra.utils.instantiate(fm_cfg.model)
        sensor_dict = OmegaConf.create(fm_cfg.get("sensor", {}))
        safe_print(f"Loaded foundation model adapter: {fm_cfg.get('name', 'unknown')}")
        return model, sensor_dict

    if cfg.get("ckpt_path", ""):
        return load_model_from_checkpoint(cfg.ckpt_path)

    return load_model_from_hf(cfg.get("hf_repo_id", "g-astruc/UniverSat"))


def resolve_output_dir(cfg: DictConfig) -> str:
    """Compute the results output dir for checkpoint, foundation-model, or HF runs."""
    fm_cfg = cfg.get("foundation_model", None)
    if fm_cfg and fm_cfg.get("enabled", False):
        base_dir = Path(str(cfg.get("output_base_dir", ".")))
        run_name = str(fm_cfg.get("name", "foundation_model"))
        return str(base_dir / run_name / cfg.output_dir)

    if cfg.get("ckpt_path", ""):
        return os.path.join(os.path.dirname(os.path.dirname(cfg.ckpt_path)), cfg.output_dir)

    # HuggingFace released weights -> outputs/<repo>/<output_dir>.
    repo = str(cfg.get("hf_repo_id", "g-astruc/UniverSat")).replace("/", "_")
    base_dir = Path(str(cfg.get("output_base_dir", "outputs")))
    return str(base_dir / repo / cfg.output_dir)

def _set_lr_schedule_all_groups(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> float:
    """Cosine schedule (no warmup, decays to 0), applied per param group via `base_lr`."""
    cos_term = math.cos(math.pi * step / max(1, total_steps))
    rel_lr = 0.5 * (1.0 + cos_term)
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group.get("lr", 0.0))
        group["lr"] = base_lr * rel_lr
    return rel_lr


def _try_make_fused_adamw(params_groups: list[dict[str, Any]]):
    """Create AdamW with `fused=True` when supported."""
    try:
        return torch.optim.AdamW(
            params_groups,
            lr=0.0,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(
            params_groups,
            lr=0.0,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )


def _make_probe_optimizer(
    params_groups: list[dict[str, Any]],
    kind: str,
    *,
    momentum: float = 0.9,
):
    """Build the probe-sweep optimizer.

    ``kind="adamw"`` -> fused AdamW (default). ``kind="sgd"`` -> SGD with the given
    ``momentum`` (DINOv2 linear-eval style: momentum=0.9). Per-group ``lr`` (set to 0
    here and driven by the scheduler) and ``weight_decay`` are honored by both.
    """
    k = str(kind).strip().lower()
    if k in ("adamw", "adam"):
        return _try_make_fused_adamw(params_groups)
    if k == "sgd":
        return torch.optim.SGD(params_groups, lr=0.0, momentum=float(momentum))
    raise ValueError(f"Unknown probe optimizer={kind!r}; use 'adamw' or 'sgd'.")


class LayerNormLinearClassifier(nn.Module):
    """CAPI-style probe head: LayerNorm + Linear."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


class BatchedLayerNormLinearProbes(nn.Module):
    """Vectorized forward over many probe heads.

    Returns stacked logits: [H, B, C] where H = number of heads.
    """

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        if len(heads) == 0:
            raise ValueError("heads must be non-empty")
        self.heads = nn.ModuleList(heads)
        # Assumes heads are LayerNormLinearClassifier-like.
        self.in_features = self.heads[0].linear.in_features
        self.out_features = self.heads[0].linear.out_features
        self.ln_eps = self.heads[0].ln.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized over heads -> logits [H, B, C].
        x_fp32 = x.float()
        W = torch.stack([h.linear.weight for h in self.heads], dim=0)  # [H, C, D]
        b = torch.stack([h.linear.bias for h in self.heads], dim=0)  # [H, C]

        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, unbiased=False, keepdim=True)
        x_hat = (x_fp32 - mean) / torch.sqrt(var + self.ln_eps)
        ln_weight = torch.stack([h.ln.weight for h in self.heads], dim=0)  # [H, D]
        ln_bias = torch.stack([h.ln.bias for h in self.heads], dim=0)  # [H, D]
        x_affine = x_hat.unsqueeze(0) * ln_weight[:, None, :] + ln_bias[:, None, :]  # [H, B, D]
        return torch.bmm(x_affine, W.transpose(1, 2)) + b[:, None, :]  # [H, B, C]


@torch.no_grad()
def _evaluate_batched_probe_cls(
    data_loader,
    probe_container: nn.Module,
    *,
    is_multilabel: bool,
    device: torch.device,
    heads: int,
    num_classes: int,
) -> dict[str, list[float]]:
    """Evaluate batched probe heads with rank0 metric computation."""
    probe_container.eval()

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    # We flatten in sample-major order to preserve head slices after gather:
    # [B, H] -> flatten -> [B*H]
    all_preds_flat = []
    all_labels_flat = []

    device_type = "cuda" if "cuda" in str(device) else "cpu"

    for batch in data_loader:
        batch_emb, batch_labels = batch  # (bsz, dim), (bsz)
        batch_emb = batch_emb.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_all = probe_container(batch_emb)  # [H, B, C]

        if is_multilabel:
            # Sample-major flatten: [H, B, C] -> [B, H, C] -> [B*H, C]
            # so we can reshape gathered tensors back to [N, H, C].
            preds_all = (logits_all.sigmoid() > 0.5).to(torch.int64)  # [H, B, C]
            preds_bh = preds_all.transpose(0, 1).reshape(-1, num_classes).cpu()  # [B*H, C]

            labels_bh = (
                batch_labels.to(torch.int64)
                .unsqueeze(1)
                .expand(-1, heads, -1)
                .reshape(-1, num_classes)
                .cpu()
            )  # [B*H, C]

            all_preds_flat.append(preds_bh)
            all_labels_flat.append(labels_bh)
            continue

        preds = logits_all.argmax(dim=-1)  # [H, B]
        preds_bh = preds.transpose(0, 1).reshape(-1).to(torch.int64)  # [B*H]

        labels_bh = batch_labels.to(torch.int64).unsqueeze(1).expand(-1, heads).reshape(-1)  # [B*H]
        all_preds_flat.append(preds_bh.cpu())
        all_labels_flat.append(labels_bh.cpu())

    if is_multilabel:
        # Gather and compute head-wise multilabel metrics.
        if dist.is_available() and dist.is_initialized():
            gathered_preds = _gather_to_rank0(torch.cat(all_preds_flat, dim=0))
            gathered_labels = _gather_to_rank0(torch.cat(all_labels_flat, dim=0))
            if rank != 0:
                return {}
            preds_np = gathered_preds.numpy()  # [N*H, C]
            labels_np = gathered_labels.numpy()
        else:
            preds_np = torch.cat(all_preds_flat, dim=0).numpy()
            labels_np = torch.cat(all_labels_flat, dim=0).numpy()

        preds_np = preds_np.astype(bool)
        labels_np = labels_np.astype(int)

        total_rows = int(preds_np.shape[0])  # N*H
        if total_rows % heads != 0:
            raise RuntimeError(f"Gathered rows {total_rows} not divisible by heads={heads}")
        n_total = total_rows // heads  # N

        preds_mat = preds_np.reshape(n_total, heads, num_classes)  # [N, H, C]
        labels_mat = labels_np.reshape(n_total, heads, num_classes)

        accs: list[float] = []
        f1_macros: list[float] = []
        f1_micros: list[float] = []
        for h in range(heads):
            p_h = preds_mat[:, h, :]
            y_h = labels_mat[:, h, :]
            accs.append(float(accuracy_score(y_h, p_h)))
            f1_macros.append(float(f1_score(y_h, p_h, average="macro")))
            f1_micros.append(float(f1_score(y_h, p_h, average="micro")))

        return {"accuracy": accs, "f1_macro": f1_macros, "f1_micro": f1_micros}

    # Non-multilabel: fast flattened gather + head-wise metric slicing.
    all_preds_flat = torch.cat(all_preds_flat, dim=0) if len(all_preds_flat) else torch.empty(0, dtype=torch.int64)
    all_labels_flat = torch.cat(all_labels_flat, dim=0) if len(all_labels_flat) else torch.empty(0, dtype=torch.int64)

    if dist.is_available() and dist.is_initialized():
        gathered_preds = _gather_to_rank0(all_preds_flat)
        gathered_labels = _gather_to_rank0(all_labels_flat)
        if rank != 0:
            return {}
        all_preds_flat = gathered_preds
        all_labels_flat = gathered_labels

    if not dist.is_available() or not dist.is_initialized() or rank == 0:
        if all_preds_flat.numel() == 0:
            return {"accuracy": [0.0] * heads}

        total_len = int(all_labels_flat.numel())
        if total_len % heads != 0:
            raise RuntimeError(f"Gathered label length {total_len} not divisible by heads={heads}")
        n_total = total_len // heads  # samples across the whole (gathered) eval set

        # Flatten order is sample-major: for each sample, we store preds for all heads.
        labels_mat = all_labels_flat.view(n_total, heads)  # [N, H]
        preds_mat = all_preds_flat.view(n_total, heads)  # [N, H]

        accs: list[float] = []
        f1_macros: list[float] = []
        f1_micros: list[float] = []
        for h in range(heads):
            y = labels_mat[:, h].cpu().numpy()
            p = preds_mat[:, h].cpu().numpy()
            accs.append(float(accuracy_score(y, p)))
            f1_macros.append(float(f1_score(y, p, average="macro")))
            f1_micros.append(float(f1_score(y, p, average="micro")))

        metrics = {"accuracy": accs, "f1_macro": f1_macros, "f1_micro": f1_micros}
        return metrics

    # Should never reach here, but keep for type clarity.
    return {}


def _broadcast_batched_cls_metrics(metrics: dict[str, list[float]], *, heads: int, device: torch.device) -> dict[str, list[float]]:
    """Broadcast list-of-floats metrics from rank0 to all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return metrics

    keys = list(metrics.keys()) if dist.get_rank() == 0 else []
    if dist.get_rank() != 0:
        keys = ["accuracy", "f1_macro", "f1_micro"]

    out: dict[str, list[float]] = {}
    for k in keys:
        if dist.get_rank() == 0:
            t = torch.tensor(metrics[k], dtype=torch.float32, device=device)
        else:
            t = torch.zeros(heads, dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        out[k] = [float(x) for x in t.detach().cpu().tolist()]
    return out


def train_sweep_batched_probe_cls(
    train_loader,
    val_loader,
    test_loader,
    *,
    lr_list: list[float],
    weight_decay_list: list[float],
    max_epochs: int,
    in_features: int,
    num_classes: int,
    is_multilabel: bool,
    device: torch.device,
    rank: int,
    world_size: int,
    optimizer_name: str = "adamw",
    sgd_momentum: float = 0.9,
):
    """Train a batched set of linear probe heads in one optimizer step.

    Returns a list of per-head result dicts matching your existing output schema.
    """
    if len(lr_list) == 0 or len(weight_decay_list) == 0:
        raise ValueError("lr_list and weight_decay_list must be non-empty")

    specs = list(itertools.product(lr_list, weight_decay_list))
    heads = len(specs)

    # Create per-(lr,wd) heads.
    probes = [LayerNormLinearClassifier(in_features, num_classes) for _ in range(heads)]

    container = BatchedLayerNormLinearProbes(probes).to(device)
    if world_size == 1 and torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            container = torch.compile(container)
        except Exception:
            pass
    if world_size > 1:
        local_rank = rank % torch.cuda.device_count()
        container = DDP(container, device_ids=[local_rank])

    # Build optimizer param groups: keep different weight_decay and base_lr per head.
    # For weight decay, exclude biases as in the example.
    params_groups: dict[tuple[float, float], dict[str, Any]] = defaultdict(lambda: {"params": [], "base_lr": 0.0})
    for (lr, wd), head in zip(specs, probes):
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            group_wd = 0.0 if "bias" in name else float(wd)
            key = (float(lr), float(group_wd))
            params_groups[key]["params"].append(p)
            params_groups[key]["base_lr"] = float(lr)

    optim_groups = []
    for (lr, wd_bias), g in params_groups.items():
        optim_groups.append({
            "params": g["params"],
            "lr": 0.0,
            "weight_decay": wd_bias,
            "base_lr": g["base_lr"],
        })

    optimizer = _make_probe_optimizer(optim_groups, optimizer_name, momentum=sgd_momentum)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    metric_name = "f1_macro" if is_multilabel else "accuracy"
    maximize = True  # for accuracy and f1_macro

    best_scores = torch.full((heads,), -float("inf"), dtype=torch.float32, device="cpu")
    best_params: list[dict[str, torch.Tensor] | None] = [None for _ in range(heads)]

    loss_fn_ml = nn.MultiLabelSoftMarginLoss()
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    data_iter = iter(train_loader)
    for _epoch in range(max_epochs):
        ep = _epoch + 1
        if ep == 1 or ep % 10 == 0 or ep == max_epochs:
            safe_print(f"[LP CLS sweep] Epoch {ep}/{max_epochs}")
        container.train()
        for _step in range(steps_per_epoch):
            try:
                batch_emb, batch_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_emb, batch_labels = next(data_iter)

            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits_all = container(batch_emb)  # [H, B, C]

                if is_multilabel:
                    # Safe fallback: loop over heads for multilabel loss.
                    losses = []
                    for h in range(heads):
                        losses.append(loss_fn_ml(logits_all[h], batch_labels.float()))
                    loss = torch.stack(losses).mean()
                else:
                    # Vectorized CE: input [B, C, H], target [B, H]
                    stack = logits_all.permute(1, 2, 0)  # [B, C, H]
                    target_expanded = batch_labels.long().unsqueeze(1).expand(-1, heads)  # [B, H]
                    loss = F.cross_entropy(
                        stack,
                        target_expanded,
                        reduction="mean",
                    )

            loss.backward()

            _set_lr_schedule_all_groups(optimizer, global_step, total_steps)
            optimizer.step()
            global_step += 1

        # ---- Validation: update best head params per-head
        metrics = _evaluate_batched_probe_cls(
            val_loader,
            container,
            is_multilabel=is_multilabel,
            device=device,
            heads=heads,
            num_classes=num_classes,
        )
        metrics = _broadcast_batched_cls_metrics(metrics, heads=heads, device=device)

        scores = torch.tensor(metrics[metric_name], dtype=torch.float32)
        improved = (scores > best_scores) if maximize else (scores < best_scores)
        if improved.any():
            # Clone best parameters for improved heads.
            base_container = container.module if isinstance(container, DDP) else container
            for h in range(heads):
                if not bool(improved[h].item()):
                    continue
                best_scores[h] = scores[h]
                head = base_container.heads[h]
                best_params[h] = {
                    "linear_weight": head.linear.weight.detach().cpu().clone(),
                    "linear_bias": head.linear.bias.detach().cpu().clone(),
                    "ln_weight": head.ln.weight.detach().cpu().clone(),
                    "ln_bias": head.ln.bias.detach().cpu().clone(),
                }

    # Load best params before test evaluation.
    base_container = container.module if isinstance(container, DDP) else container
    for h in range(heads):
        bp = best_params[h]
        if bp is None:
            continue
        head = base_container.heads[h]
        head.linear.weight.data.copy_(bp["linear_weight"].to(device))
        head.linear.bias.data.copy_(bp["linear_bias"].to(device))
        head.ln.weight.data.copy_(bp["ln_weight"].to(device))
        head.ln.bias.data.copy_(bp["ln_bias"].to(device))

    # Re-evaluate on val with the best params so the JSON reflects the selected heads.
    val_metrics_best = _evaluate_batched_probe_cls(
        val_loader,
        container,
        is_multilabel=is_multilabel,
        device=device,
        heads=heads,
        num_classes=num_classes,
    )
    val_metrics_best = _broadcast_batched_cls_metrics(val_metrics_best, heads=heads, device=device)

    # Test evaluation
    test_metrics = _evaluate_batched_probe_cls(
        test_loader,
        container,
        is_multilabel=is_multilabel,
        device=device,
        heads=heads,
        num_classes=num_classes,
    )
    test_metrics = _broadcast_batched_cls_metrics(test_metrics, heads=heads, device=device)

    results = []
    for h, (lr, wd) in enumerate(specs):
        val_entry = {k: float(val_metrics_best[k][h]) for k in val_metrics_best.keys()}
        test_entry = {k: float(test_metrics[k][h]) for k in test_metrics.keys()}
        results.append({
            "lr": float(lr),
            "weight_decay": float(wd),
            "val": val_entry,
            "test": test_entry,
        })

    return results


@torch.no_grad()
def _evaluate_batched_probe_seg(
    data_loader,
    probe_container: nn.Module,
    *,
    task: str,
    device: torch.device,
    heads: int,
    num_classes: int,
    patch_size: int,
) -> dict[str, list[float]]:
    """Evaluate batched LP heads for segmentation (mIoU/micro IoU) or regression (MSE)."""
    if task not in {"segmentation", "regression"}:
        raise ValueError(f"Unsupported task={task!r}")

    probe_container.eval()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    patch_area = patch_size * patch_size
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    if task == "segmentation":
        # [H, C, C]: rows=true class, cols=pred class
        conf = torch.zeros((heads, num_classes, num_classes), dtype=torch.float64, device=device)
    else:
        sse = torch.zeros((heads,), dtype=torch.float64, device=device)
        count = torch.zeros((heads,), dtype=torch.float64, device=device)

    for batch_emb, batch_labels in data_loader:
        batch_emb = batch_emb.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_all = probe_container(batch_emb)  # [H, B, out_dim]

        if task == "segmentation":
            logits_all = logits_all.view(heads, batch_emb.shape[0], num_classes, patch_area)
            preds_all = logits_all.argmax(dim=2)  # [H, B, patch_area]
            labels = batch_labels.long()  # [B, patch_area]
            for h in range(heads):
                pred_h = preds_all[h].reshape(-1).long()
                label_h = labels.reshape(-1)
                valid = label_h != -1
                if not valid.any():
                    continue
                y = label_h[valid]
                p = pred_h[valid].clamp(min=0, max=num_classes - 1)
                bins = torch.bincount(
                    y * num_classes + p,
                    minlength=num_classes * num_classes,
                ).to(torch.float64)
                conf[h] += bins.view(num_classes, num_classes)
        else:
            # Regression (MSE): assume out_dim matches batch_labels last dimension.
            out_dim = logits_all.shape[-1]
            labels_flat = batch_labels
            if labels_flat.shape[-1] != out_dim:
                if labels_flat.shape[-1] != patch_area:
                    raise RuntimeError(
                        f"Regression label width {labels_flat.shape[-1]} doesn't match out_dim={out_dim} or patch_area={patch_area}."
                    )
                labels_flat = labels_flat.unsqueeze(1).expand(-1, num_classes, -1).reshape(labels_flat.shape[0], -1)

            diff = logits_all.float() - labels_flat.unsqueeze(0).float()  # [H, B, out_dim]
            sse += diff.pow(2).sum(dim=(1, 2)).to(torch.float64)
            count += float(diff.shape[1] * diff.shape[2])

    if dist.is_available() and dist.is_initialized():
        if task == "segmentation":
            dist.all_reduce(conf, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(sse, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

    if task == "segmentation":
        miou_list: list[float] = []
        micro_iou_list: list[float] = []
        for h in range(heads):
            cm = conf[h]
            inter = torch.diag(cm)
            union = cm.sum(dim=1) + cm.sum(dim=0) - inter
            valid = union > 0
            miou_h = (inter[valid] / (union[valid] + 1e-8)).mean() if valid.any() else torch.tensor(0.0, device=device)
            micro_h = inter.sum() / (union.sum() + 1e-8)
            miou_list.append(float(miou_h.item()))
            micro_iou_list.append(float(micro_h.item()))
        return {"miou": miou_list, "micro_iou": micro_iou_list}

    mse = sse / torch.clamp(count, min=1.0)
    return {"mse": [float(x.item()) for x in mse]}


def _broadcast_batched_seg_metrics(
    metrics: dict[str, list[float]],
    *,
    heads: int,
    device: torch.device,
    task: str,
) -> dict[str, list[float]]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics

    keys = ["miou", "micro_iou"] if task == "segmentation" else ["mse"]
    out: dict[str, list[float]] = {}
    for key in keys:
        if dist.get_rank() == 0:
            t = torch.tensor(metrics[key], dtype=torch.float32, device=device)
        else:
            t = torch.zeros(heads, dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        out[key] = [float(x) for x in t.detach().cpu().tolist()]
    return out


def train_sweep_batched_probe_seg(
    train_loader,
    val_loader,
    test_loader,
    *,
    lr_list: list[float],
    weight_decay_list: list[float],
    max_epochs: int,
    in_features: int,
    num_classes: int,
    patch_size: int,
    task: str,
    device: torch.device,
    rank: int,
    world_size: int,
    optimizer_name: str = "adamw",
    sgd_momentum: float = 0.9,
) -> list[dict[str, Any]]:
    """Train batched LP heads for segmentation/regression with one optimizer."""
    if task not in {"segmentation", "regression"}:
        raise ValueError(f"Unsupported task={task!r}")

    if len(lr_list) == 0 or len(weight_decay_list) == 0:
        raise ValueError("lr_list and weight_decay_list must be non-empty")

    patch_area = patch_size * patch_size
    out_dim = num_classes * patch_area

    specs = list(itertools.product(lr_list, weight_decay_list))
    heads = len(specs)

    probes = [LayerNormLinearClassifier(in_features, out_dim) for _ in range(heads)]
    container = BatchedLayerNormLinearProbes(probes).to(device)
    if world_size == 1 and torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            container = torch.compile(container)
        except Exception:
            pass
    if world_size > 1:
        local_rank = rank % torch.cuda.device_count()
        container = DDP(container, device_ids=[local_rank])

    params_groups: dict[tuple[float, float], dict[str, Any]] = defaultdict(lambda: {"params": [], "base_lr": 0.0})
    for (lr, wd), head in zip(specs, probes):
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            group_wd = 0.0 if "bias" in name else float(wd)
            key = (float(lr), float(group_wd))
            params_groups[key]["params"].append(p)
            params_groups[key]["base_lr"] = float(lr)

    optim_groups = []
    for (_lr, wd_bias), g in params_groups.items():
        optim_groups.append({
            "params": g["params"],
            "lr": 0.0,
            "weight_decay": wd_bias,
            "base_lr": g["base_lr"],
        })
    optimizer = _make_probe_optimizer(optim_groups, optimizer_name, momentum=sgd_momentum)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    metric_name = "miou" if task == "segmentation" else "mse"
    maximize = True if task == "segmentation" else False

    best_scores = torch.full((heads,), -float("inf") if maximize else float("inf"), dtype=torch.float32, device="cpu")
    best_params: list[dict[str, torch.Tensor] | None] = [None for _ in range(heads)]

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    data_iter = iter(train_loader)
    for _epoch in range(max_epochs):
        ep = _epoch + 1
        if ep == 1 or ep % 10 == 0 or ep == max_epochs:
            safe_print(f"[LP SEG/REG sweep] Epoch {ep}/{max_epochs}")
        container.train()
        for _step in range(steps_per_epoch):
            try:
                batch_emb, batch_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_emb, batch_labels = next(data_iter)

            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits_all = container(batch_emb)  # [H, B, out_dim]

                if task == "segmentation":
                    losses = []
                    for h in range(heads):
                        logits_h = logits_all[h].view(batch_emb.shape[0], num_classes, patch_area)
                        losses.append(ce_loss(logits_h, batch_labels.long()))
                    loss = torch.stack(losses).mean()
                else:
                    # MSE
                    diff = logits_all.float() - batch_labels.unsqueeze(0).float()
                    loss = diff.pow(2).mean()

            loss.backward()
            _set_lr_schedule_all_groups(optimizer, global_step, total_steps)
            optimizer.step()
            global_step += 1

        metrics = _evaluate_batched_probe_seg(
            val_loader,
            container,
            task=task,
            device=device,
            heads=heads,
            num_classes=num_classes,
            patch_size=patch_size,
        )
        metrics = _broadcast_batched_seg_metrics(metrics, heads=heads, device=device, task=task)

        scores = torch.tensor(metrics[metric_name], dtype=torch.float32)
        improved = (scores > best_scores) if maximize else (scores < best_scores)
        if improved.any():
            base_container = container.module if isinstance(container, DDP) else container
            for h in range(heads):
                if not bool(improved[h].item()):
                    continue
                best_scores[h] = scores[h]
                head = base_container.heads[h]
                best_params[h] = {
                    "linear_weight": head.linear.weight.detach().cpu().clone(),
                    "linear_bias": head.linear.bias.detach().cpu().clone(),
                    "ln_weight": head.ln.weight.detach().cpu().clone(),
                    "ln_bias": head.ln.bias.detach().cpu().clone(),
                }

    base_container = container.module if isinstance(container, DDP) else container
    for h in range(heads):
        bp = best_params[h]
        if bp is None:
            continue
        head = base_container.heads[h]
        head.linear.weight.data.copy_(bp["linear_weight"].to(device))
        head.linear.bias.data.copy_(bp["linear_bias"].to(device))
        head.ln.weight.data.copy_(bp["ln_weight"].to(device))
        head.ln.bias.data.copy_(bp["ln_bias"].to(device))

    val_metrics_best = _evaluate_batched_probe_seg(
        val_loader,
        container,
        task=task,
        device=device,
        heads=heads,
        num_classes=num_classes,
        patch_size=patch_size,
    )
    val_metrics_best = _broadcast_batched_seg_metrics(val_metrics_best, heads=heads, device=device, task=task)

    test_metrics = _evaluate_batched_probe_seg(
        test_loader,
        container,
        task=task,
        device=device,
        heads=heads,
        num_classes=num_classes,
        patch_size=patch_size,
    )
    test_metrics = _broadcast_batched_seg_metrics(test_metrics, heads=heads, device=device, task=task)

    results: list[dict[str, Any]] = []
    for h, (lr, wd) in enumerate(specs):
        results.append({
            "lr": float(lr),
            "weight_decay": float(wd),
            "val": {k: float(v[h]) for k, v in val_metrics_best.items()},
            "test": {k: float(v[h]) for k, v in test_metrics.items()},
        })
    return results


_STD_CENTER_MODES = {"center"}
_STD_ZSCORE_MODES = {"standard", "standardscaler", "standard_scaler", "zscore", "z-score"}


def _compute_feature_mean_std(
    train_ds,
    *,
    device: torch.device,
    world_size: int,
    chunk_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over TRAIN features (fp64 streaming accumulation, DDP-safe)."""
    logits = train_ds.logits
    n = int(len(train_ds))
    D = int(logits.shape[1])
    s1 = torch.zeros(D, dtype=torch.float64)  # sum x
    s2 = torch.zeros(D, dtype=torch.float64)  # sum x^2
    count = 0
    cs = max(int(chunk_size), 1)
    for st in range(0, n, cs):
        en = min(n, st + cs)
        x = logits[st:en]
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        x = x.detach().cpu().to(torch.float64)
        s1 += x.sum(dim=0)
        s2 += (x * x).sum(dim=0)
        count += int(x.shape[0])

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        packed = torch.cat([s1, s2, torch.tensor([float(count)], dtype=torch.float64)]).to(device)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed = packed.cpu()
        s1, s2, cnt = packed[:D], packed[D : 2 * D], float(packed[-1].item())
    else:
        cnt = float(count)

    cnt = max(cnt, 1.0)
    mu = s1 / cnt
    var = (s2 / cnt - mu * mu).clamp(min=0.0)
    sigma = torch.sqrt(var + eps)
    return mu.to(torch.float32), sigma.to(torch.float32)


def _apply_standardizer(ds, mu: torch.Tensor, sigma: torch.Tensor, *, zscore: bool, chunk_size: int) -> None:
    """Apply (x - mu)[/ sigma] in place to a logits dataset (tensor or memmap), in chunks."""
    logits = ds.logits
    n = int(len(ds))
    is_memmap = not torch.is_tensor(logits)
    inv = (1.0 / sigma) if zscore else torch.ones_like(sigma)
    cs = max(int(chunk_size), 1)
    # Features are extracted under torch.inference_mode(), so the in-memory tensor
    # is an "inference tensor" that can only be mutated in place from within an
    # inference-mode context. (The memmap path writes through numpy, unaffected.)
    inplace_ctx = torch.inference_mode() if (not is_memmap and logits.is_inference()) else nullcontext()
    with inplace_ctx:
        for st in range(0, n, cs):
            en = min(n, st + cs)
            if is_memmap:
                chunk = torch.from_numpy(np.asarray(logits[st:en])).float()
                chunk = (chunk - mu) * inv
                logits[st:en] = chunk.numpy().astype(logits.dtype)
            else:
                dev = logits.device
                chunk = (logits[st:en].float() - mu.to(dev)) * inv.to(dev)
                logits[st:en] = chunk.to(logits.dtype)
    if is_memmap and hasattr(logits, "flush"):
        logits.flush()


def _standardize_feature_datasets(
    train_ds,
    val_ds,
    test_ds,
    *,
    mode: str,
    device: torch.device,
    world_size: int,
    chunk_size: int,
) -> None:
    """Fit a standardizer on TRAIN and apply it to all three splits, in place."""
    mode_l = str(mode).strip().lower()
    if mode_l in ("", "none", "null"):
        return
    if mode_l not in (_STD_CENTER_MODES | _STD_ZSCORE_MODES):
        raise ValueError(
            f"Unsupported standardization={mode!r}. Use 'standard'/'StandardScaler', 'center', or null."
        )
    zscore = mode_l in _STD_ZSCORE_MODES
    mu, sigma = _compute_feature_mean_std(
        train_ds, device=device, world_size=world_size, chunk_size=chunk_size,
    )
    safe_print(
        f"[LP standardize] mode={mode_l} fit on train (D={mu.numel()}): "
        f"mean(mu)={mu.mean().item():.4g}, mean(sigma)={sigma.mean().item():.4g}; "
        "applying to train/val/test."
    )
    for ds in (train_ds, val_ds, test_ds):
        _apply_standardizer(ds, mu, sigma, zscore=zscore, chunk_size=chunk_size)


def _gather_to_rank0(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Capture original state
    orig_dtype = tensor.dtype
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # 1. NCCL SUPPORT CHECK (The "Whitelist" approach)
    # NCCL generally supports: float32, float16, bfloat16, double, long, int32, uint8
    # It DOES NOT support: uint16, int16, bool
    supported_types = [torch.float32, torch.float16, torch.bfloat16,
                       torch.float64, torch.long, torch.int32, torch.uint8]

    working_tensor = tensor.to(device)

    if orig_dtype not in supported_types:
        # Cast to Long (Int64) as it's the safest bet for labels/indices
        working_tensor = working_tensor.to(torch.long)

    working_tensor = working_tensor.contiguous()

    # 2. Metadata Gather
    local_size = torch.tensor([working_tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = int(max(s.item() for s in all_sizes))

    # 3. Padding
    if working_tensor.shape[0] < max_size:
        pad_shape = list(working_tensor.shape)
        pad_shape[0] = max_size - working_tensor.shape[0]
        padding = torch.zeros(pad_shape, dtype=working_tensor.dtype, device=device)
        working_tensor = torch.cat([working_tensor, padding])

    # 4. Main Gather
    gathered = [torch.zeros(list(working_tensor.shape), dtype=working_tensor.dtype, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, working_tensor)

    if rank == 0:
        # Concatenate, trim padding, cast back to original dtype, and move to CPU
        res = torch.cat([g[: int(s.item())] for g, s in zip(gathered, all_sizes)])
        return res.to(orig_dtype).cpu()

    return None


def evaluate_knn_cls(
    train_dataset: "InMemoryLogitsDataset",
    val_dataset: "InMemoryLogitsDataset",
    test_dataset: "InMemoryLogitsDataset",
    k_list: list,
) -> Optional[dict]:

    if dist.is_available() and dist.is_initialized():
        train_logits = _gather_to_rank0(train_dataset.logits)
        train_labels = _gather_to_rank0(train_dataset.labels)
        val_logits = _gather_to_rank0(val_dataset.logits)
        val_labels = _gather_to_rank0(val_dataset.labels)
        test_logits = _gather_to_rank0(test_dataset.logits)
        test_labels = _gather_to_rank0(test_dataset.labels)

        if dist.get_rank() != 0:
            return None
    else:
        train_logits, train_labels = train_dataset.logits, train_dataset.labels
        val_logits, val_labels = val_dataset.logits, val_dataset.labels
        test_logits, test_labels = test_dataset.logits, test_dataset.labels

    def torch_to_clean_numpy(logits, labels, name="dataset"):
        # 1. Ensure we are working with floats for logits
        logits = logits.float()

        # 2. Find valid indices (no NaN and no Inf)
        # We check across the feature dimension (dim=1)
        is_nan = torch.isnan(logits).any(dim=1)
        is_inf = torch.isinf(logits).any(dim=1)
        invalid_mask = is_nan | is_inf
        valid_mask = ~invalid_mask

        num_invalid = invalid_mask.sum().item()
        if num_invalid > 0:
            print(f"!!! Warning: Found {num_invalid} NaNs/Infs in {name}. Removing them.")
            logits = logits[valid_mask]
            labels = labels[valid_mask]

        if logits.shape[0] == 0:
            raise ValueError(f"CRITICAL: {name} is empty after NaN removal! Check your model output.")

        return logits.cpu().numpy(), labels.cpu().numpy().astype(int)

    # Convert and Clean
    train_X, train_y = torch_to_clean_numpy(train_logits, train_labels, "train")
    val_X, val_y = torch_to_clean_numpy(val_logits, val_labels, "val")
    test_X, test_y = torch_to_clean_numpy(test_logits, test_labels, "test")

    # --- Standard KNN Loop ---
    best_val_acc = -float("inf")
    best_result = None
    all_k_results = []

    for k in k_list:
        # Safety: Ensure k isn't larger than the cleaned dataset
        current_k = min(k, len(train_X))
        safe_print(f"  KNN k={current_k}: fitting...")

        knn = KNeighborsClassifier(n_neighbors=current_k, n_jobs=-1)
        knn.fit(train_X, train_y)

        val_preds = knn.predict(val_X)
        val_metrics = {
            "accuracy": float(accuracy_score(val_y, val_preds)),
            "f1_macro": float(f1_score(val_y, val_preds, average="macro")),
            "f1_micro": float(f1_score(val_y, val_preds, average="micro")),
        }

        test_preds = knn.predict(test_X)
        test_metrics = {
            "accuracy": float(accuracy_score(test_y, test_preds)),
            "f1_macro": float(f1_score(test_y, test_preds, average="macro")),
            "f1_micro": float(f1_score(test_y, test_preds, average="micro")),
        }

        k_result = {"k": k, "val": val_metrics, "test": test_metrics, "cosine": False}
        all_k_results.append(k_result)
        safe_print(
            f"  KNN k={k}: val_acc={val_metrics['accuracy']:.4f}, "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_result = k_result


    train_X = normalize(train_X, axis=1, norm='l2')
    val_X = normalize(val_X, axis=1, norm='l2')
    test_X = normalize(test_X, axis=1, norm='l2')

    for k in k_list:
        safe_print(f"  KNN cosine k={k}: fitting...")
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(train_X, train_y)

        val_preds = knn.predict(val_X)
        val_metrics = {
            "accuracy": float(accuracy_score(val_y, val_preds)),
            "f1_macro": float(f1_score(val_y, val_preds, average="macro")),
            "f1_micro": float(f1_score(val_y, val_preds, average="micro")),
        }

        test_preds = knn.predict(test_X)
        test_metrics = {
            "accuracy": float(accuracy_score(test_y, test_preds)),
            "f1_macro": float(f1_score(test_y, test_preds, average="macro")),
            "f1_micro": float(f1_score(test_y, test_preds, average="micro")),
        }

        k_result = {"k": k, "val": val_metrics, "test": test_metrics, "cosine": True}
        all_k_results.append(k_result)
        safe_print(
            f"  KNN cosine k={k}: val_acc={val_metrics['accuracy']:.4f}, "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_result = k_result

    return {
        "best": best_result,
        "all_k": all_k_results,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="LP_eval.yaml")
def main(cfg: DictConfig) -> None:
    # Setup DDP
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    rank = dist.get_rank() if dist.is_initialized() else 0

    utils.extras(cfg)

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # grid search parameters
    pooling_list = getattr(cfg.param.classification, "pooling_list", ["mean"])
    if cfg.dataset.task == 'segmentation':
        pooling_list = ['mean']
    # Allow configs without param.scale; fallback to dataset.scale_base
    scale_list = getattr(cfg.param, "scale_list", [1])
    seed_list = getattr(cfg.param, "seed_list", [1])
    norm_path_list = getattr(cfg.param, "norm_path_list", [None])

    # Segmentation/regression-specific patch grid options; classification ignores
    if cfg.dataset.task in ["segmentation", "regression"]:
        out_patch_list = getattr(cfg.param.segmentation, "out_patch_size_list", [getattr(cfg.dataset, "out_patch_size", None)])
    else:
        out_patch_list = [None]

    original_norm_path = getattr(cfg.dataset, "norm_path", None)
    all_results = []
    all_knn_results = []

    for norm_override in norm_path_list:
        cfg.dataset.norm_path = original_norm_path if norm_override is None else norm_override

        model, sensor_dict = load_feature_model(cfg)
        model = model.to(cfg.device)
        model.eval()

        for scale_value in scale_list:
            for out_patch_value in out_patch_list:
                patch_val = out_patch_value if out_patch_value is not None else getattr(cfg.dataset, "out_patch_size", None)
                if cfg.dataset.task in ["segmentation", "regression"] and patch_val is None:
                    raise ValueError("out_patch_size must be set for segmentation/regression tasks")

                for pooling_value in pooling_list:
                    # Fixed seed for embedding extraction
                    set_seed(seed_list[0] if seed_list else 1)

                    datamodule = hydra.utils.instantiate(cfg.datamodule)
                    num_patches = cfg.dataset.num_patches
                    datamodule.setup("fit")
                    train_loader = datamodule.train_dataloader()
                    val_loader = datamodule.val_dataloader()
                    datamodule.setup("test")
                    test_loader = datamodule.test_dataloader()

                    if dist.is_initialized():
                        train_sampler = DistributedSampler(train_loader.dataset, shuffle=False)
                        train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, sampler=train_sampler, num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory)

                        val_sampler = DistributedSampler(val_loader.dataset, shuffle=False)
                        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, sampler=val_sampler, num_workers=val_loader.num_workers, pin_memory=val_loader.pin_memory)

                        test_sampler = DistributedSampler(test_loader.dataset, shuffle=False)
                        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_loader.batch_size, sampler=test_sampler, num_workers=test_loader.num_workers, pin_memory=test_loader.pin_memory)

                    fraction = cfg.dataset.get("fraction", 1.0)

                    LP_train_dataset = get_logits(
                        model,
                        train_loader,
                        "train",
                        sensor_dict,
                        num_patches=num_patches,
                        res=cfg.dataset.res,
                        scale=scale_value,
                        task=cfg.dataset.task,
                        device=cfg.device,
                        fraction=fraction,
                        patch_size=patch_val,
                        logits_dtype=getattr(cfg.dataset, "logits_dtype", "float32"),
                        pooling=pooling_value,
                        use_memmap=bool(getattr(cfg.dataset, "use_memmap", False)),
                        memmap_dir=getattr(cfg.dataset, "memmap_dir", None),
                    )
                    del train_loader
                    gc.collect()
                    torch.cuda.empty_cache()
                    LP_val_dataset = get_logits(
                        model,
                        val_loader,
                        "val",
                        sensor_dict,
                        num_patches=num_patches,
                        res=cfg.dataset.res,
                        scale=scale_value,
                        task=cfg.dataset.task,
                        device=cfg.device,
                        fraction=fraction,
                        patch_size=patch_val,
                        logits_dtype=getattr(cfg.dataset, "logits_dtype", "float32"),
                        pooling=pooling_value,
                        use_memmap=bool(getattr(cfg.dataset, "use_memmap", False)),
                        memmap_dir=getattr(cfg.dataset, "memmap_dir", None),
                    )
                    del val_loader
                    gc.collect()
                    torch.cuda.empty_cache()
                    LP_test_dataset = get_logits(
                        model,
                        test_loader,
                        "test",
                        sensor_dict,
                        num_patches=num_patches,
                        res=cfg.dataset.res,
                        scale=scale_value,
                        task=cfg.dataset.task,
                        device=cfg.device,
                        patch_size=patch_val,
                        logits_dtype=getattr(cfg.dataset, "logits_dtype", "float32"),
                        pooling=pooling_value,
                        use_memmap=bool(getattr(cfg.dataset, "use_memmap", False)),
                        memmap_dir=getattr(cfg.dataset, "memmap_dir", None),
                    )

                    del test_loader
                    gc.collect()
                    torch.cuda.empty_cache()

                    safe_print(f"Train logits shape: {LP_train_dataset.logits.shape}, labels shape: {LP_train_dataset.labels.shape}")
                    safe_print(f"Val logits shape: {LP_val_dataset.logits.shape}, labels shape: {LP_val_dataset.labels.shape}")
                    safe_print(f"Test logits shape: {LP_test_dataset.logits.shape}, labels shape: {LP_test_dataset.labels.shape}")

                    def _get_bytes(arr):
                        if hasattr(arr, "numel") and hasattr(arr, "element_size"):
                            return arr.numel() * arr.element_size()
                        if hasattr(arr, "nbytes"):
                            return arr.nbytes
                        if hasattr(arr, "size") and hasattr(arr, "itemsize"):
                            return int(np.prod(arr.size)) * int(arr.itemsize)
                        return np.inf

                    size_bytes_train = _get_bytes(LP_train_dataset.logits) + _get_bytes(LP_train_dataset.labels)
                    size_bytes_val = _get_bytes(LP_val_dataset.logits) + _get_bytes(LP_val_dataset.labels)
                    size_bytes_test = _get_bytes(LP_test_dataset.logits) + _get_bytes(LP_test_dataset.labels)
                    safe_print(f"Logits train dataset size: {size_bytes_train/1024**3:.2f} GB / val dataset size: {size_bytes_val/1024**3:.2f} GB / test dataset size: {size_bytes_test/1024**3:.2f} GB")

                    # --- Feature standardization (fit on train, applied to all splits) ---
                    standardization = getattr(cfg.param, "standardization", None)
                    if standardization is not None and str(standardization).strip().lower() not in ("", "none", "null"):
                        _standardize_feature_datasets(
                            LP_train_dataset,
                            LP_val_dataset,
                            LP_test_dataset,
                            mode=str(standardization),
                            device=cfg.device,
                            world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                            chunk_size=int(getattr(cfg.param, "standardize_chunk_size", 1_000_000)),
                        )

                    # --- KNN evaluation (classification only, before GPU move) ---
                    is_corine = "corine" in str(getattr(cfg.dataset, "name", "")).lower() or "corine" in str(getattr(cfg.dataset, "product", "")).lower() or "corine" in str(getattr(cfg.dataset, "norm_path", "")).lower()
                    if cfg.dataset.task == "classification" and not is_corine:
                        knn_k_list = list(getattr(cfg.param.classification, "knn_k_list", [1, 3, 5, 10, 20]))
                        if isinstance(LP_train_dataset, MemMapLogitsDataset):
                            safe_print("Skipping KNN evaluation in memmap mode to avoid full in-memory materialization.")
                        else:
                            safe_print(f"Running KNN evaluation with k={knn_k_list}")
                            knn_result = evaluate_knn_cls(
                                LP_train_dataset, LP_val_dataset, LP_test_dataset,
                                k_list=knn_k_list,
                            )
                            if knn_result is not None:
                                knn_entry = {
                                    "scale": scale_value,
                                    "pooling": pooling_value,
                                    "norm_path": cfg.dataset.norm_path,
                                    **knn_result,
                                }
                                all_knn_results.append(knn_entry)
                                safe_print(
                                    f"KNN best: k={knn_result['best']['k']}, "
                                    f"test_acc={knn_result['best']['test']['accuracy']:.4f}"
                                )
                    elif cfg.dataset.task == "classification" and is_corine:
                        safe_print("Skipping KNN evaluation for EnmapCorine dataset.")

                    move_logits_to_gpu = bool(getattr(cfg.param, "move_logits_to_gpu", True))
                    _is_memmap = isinstance(LP_train_dataset, MemMapLogitsDataset)
                    # Disk-backed datasets cannot be moved to GPU; keep on CPU and use workers.
                    can_move_to_gpu = torch.cuda.is_available() and move_logits_to_gpu and not _is_memmap
                    if _is_memmap:
                        safe_print("Keeping extracted logits/labels on disk (memmap mode).")
                        cpu_dataset = True
                    elif can_move_to_gpu:
                        total_logits_bytes = size_bytes_train + size_bytes_val + size_bytes_test
                        # Keep headroom for model + activations during probe training.
                        gpu_memory_fraction = float(getattr(cfg.param, "logits_gpu_memory_fraction", 0.8))
                        gpu_memory_fraction = min(max(gpu_memory_fraction, 0.1), 0.95)
                        free_bytes, _ = torch.cuda.mem_get_info(device=cfg.device)
                        allowed_bytes = int(free_bytes * gpu_memory_fraction)
                        if total_logits_bytes <= allowed_bytes:
                            safe_print(
                                "Moving logits/labels to VRAM "
                                f"(need {total_logits_bytes/1024**3:.2f} GB, "
                                f"allowing {allowed_bytes/1024**3:.2f} GB at fraction={gpu_memory_fraction:.2f})"
                            )
                            LP_train_dataset.to(cfg.device)
                            LP_val_dataset.to(cfg.device)
                            LP_test_dataset.to(cfg.device)
                            cpu_dataset = False
                        else:
                            safe_print(
                                "Keeping extracted logits/labels on CPU: "
                                f"need {total_logits_bytes/1024**3:.2f} GB > allowed {allowed_bytes/1024**3:.2f} GB "
                                f"(free={free_bytes/1024**3:.2f} GB, fraction={gpu_memory_fraction:.2f})"
                            )
                            cpu_dataset = True
                    else:
                        if torch.cuda.is_available() and not move_logits_to_gpu:
                            safe_print("Keeping extracted logits/labels on CPU (cfg.param.move_logits_to_gpu=False)")
                        cpu_dataset = True

                    if cfg.dataset.task == "classification":
                        parameter = cfg.param.classification
                    else:
                        parameter = cfg.param.segmentation

                    for seed_value in seed_list:
                        set_seed(seed_value)

                        safe_print("Creating dataloaders for probes")
                        gen = torch.Generator()
                        gen.manual_seed(seed_value)
                        batch_size = parameter.batch_size
                        # Optimal batch size to target <=500 steps/epoch (can be huge on large datasets).
                        min_batch_for_500_steps = max(1, math.ceil(len(LP_train_dataset) / 500))
                        opt_batch_size = 2 ** math.ceil(math.log2(min_batch_for_500_steps))
                        batch_size = max(batch_size, opt_batch_size)

                        # Ensure at least 16 iterations per epoch
                        max_batch_for_16_iters = math.floor(len(LP_train_dataset) / 16)
                        max_batch_for_16_iters = 2 ** math.floor(math.log2(max(1, max_batch_for_16_iters)))
                        batch_size = min(batch_size, max_batch_for_16_iters)

                        max_probe_batch = int(
                            getattr(parameter, "max_probe_batch_size", None)
                            or getattr(cfg.dataset, "max_probe_batch_size", 200000)
                        )
                        if max_probe_batch < 1:
                            max_probe_batch = 50000
                        batch_size = min(batch_size, max_probe_batch)
                        safe_print(
                            f"Using batch size {batch_size} for probe training with {len(LP_train_dataset)} samples "
                            f"(capped at max_probe_batch_size={max_probe_batch}; more steps/epoch is OK)."
                        )

                        probe_num_workers = int(
                            getattr(
                                parameter,
                                "probe_num_workers",
                                getattr(cfg.dataset, "probe_num_workers", 8 if cpu_dataset else 0),
                            )
                        )
                        probe_pin_memory = bool(
                            getattr(
                                parameter,
                                "probe_pin_memory",
                                getattr(cfg.dataset, "probe_pin_memory", cpu_dataset),
                            )
                        )

                        probe_group_by_image = cfg.dataset.task in ("segmentation", "regression") and bool(
                            getattr(parameter, "probe_shuffle_group_by_image", False)
                        )
                        P_train = getattr(LP_train_dataset, "patches_per_image", None)
                        can_group_train = (
                            probe_group_by_image
                            and P_train is not None
                            and int(P_train) > 0
                            and len(LP_train_dataset) % int(P_train) == 0
                        )
                        if probe_group_by_image and P_train and not can_group_train:
                            safe_print(
                                "probe_shuffle_group_by_image: dataset length is not divisible by "
                                f"patches_per_image={P_train} (e.g. fractional subsampling); using flat shuffle."
                            )

                        can_group_for_train = can_group_train

                        if _is_memmap:
                            # ChunkedMemmapLoader for train (sequential I/O). Val/test always chunked.
                            _memmap_chunk_size = int(getattr(cfg.dataset, "probe_memmap_chunk_size", 1_000_000))
                            _memmap_chunk_aligned = _memmap_chunk_size
                            _shuffle_within_chunk = True
                            if can_group_train:
                                P = int(P_train)
                                _memmap_chunk_aligned = max(
                                    batch_size,
                                    (_memmap_chunk_size // P) * P,
                                )
                                _shuffle_within_chunk = False
                                safe_print(
                                    f"[ChunkedMemmapLoader] image-grouped patches P={P}: "
                                    f"chunk_size {_memmap_chunk_size:,} -> {_memmap_chunk_aligned:,}; "
                                    "within-chunk shuffle disabled (pixels stay image-local)."
                                )
                            _memmap_prefetch = bool(getattr(cfg.dataset, "probe_memmap_prefetch", True))
                            _memmap_drop_pages = bool(getattr(cfg.dataset, "probe_memmap_drop_pages", True))
                            safe_print(
                                f"[ChunkedMemmapLoader] train={len(LP_train_dataset):,} "
                                f"val={len(LP_val_dataset):,} test={len(LP_test_dataset):,} rows, "
                                f"chunk_size={_memmap_chunk_aligned:,}, prefetch={_memmap_prefetch}, "
                                f"drop_pages={_memmap_drop_pages}. Train loader: chunked sequential shuffle."
                            )
                            train_loader_lp = ChunkedMemmapLoader(
                                LP_train_dataset,
                                batch_size=batch_size,
                                chunk_size=_memmap_chunk_aligned,
                                shuffle=True,
                                drop_last=True,
                                generator=gen,
                                prefetch=_memmap_prefetch,
                                drop_pages_after_chunk=_memmap_drop_pages,
                                shuffle_within_chunk=_shuffle_within_chunk,
                            )
                            val_loader_lp = ChunkedMemmapLoader(
                                LP_val_dataset,
                                batch_size=batch_size,
                                chunk_size=_memmap_chunk_aligned,
                                shuffle=False,
                                drop_last=False,
                                prefetch=_memmap_prefetch,
                                drop_pages_after_chunk=_memmap_drop_pages,
                            )
                            test_loader_lp = ChunkedMemmapLoader(
                                LP_test_dataset,
                                batch_size=batch_size,
                                chunk_size=_memmap_chunk_aligned,
                                shuffle=False,
                                drop_last=False,
                                prefetch=_memmap_prefetch,
                                drop_pages_after_chunk=_memmap_drop_pages,
                            )
                        else:
                            if cpu_dataset and probe_num_workers > 0:
                                # In-memory tensors are shared via /dev/shm; kill workers when too large.
                                torch.multiprocessing.set_sharing_strategy("file_system")
                                shm_risky_threshold = int(getattr(cfg.dataset, "probe_shm_risky_threshold", 20_000_000))
                                if len(LP_train_dataset) >= shm_risky_threshold:
                                    safe_print(
                                        f"Setting probe_num_workers=0 to avoid shared-memory OOM "
                                        f"(len={len(LP_train_dataset)} >= probe_shm_risky_threshold={shm_risky_threshold})."
                                    )
                                    probe_num_workers = 0

                            _probe_persistent = probe_num_workers > 0
                            _probe_prefetch = int(getattr(cfg.dataset, "probe_prefetch_factor", 2)) if probe_num_workers > 0 else None

                            if can_group_for_train:
                                P = int(P_train)
                                eff_rows = max(P, (batch_size // P) * P)
                                safe_print(
                                    f"Probe train: image-grouped shuffle (P={P} patches/image, "
                                    f"~{eff_rows} patch-rows per batch)."
                                )
                                train_loader_lp = torch.utils.data.DataLoader(
                                    LP_train_dataset,
                                    batch_sampler=ImageGroupedPatchBatchSampler(
                                        len(LP_train_dataset),
                                        P,
                                        batch_size,
                                        shuffle=True,
                                        drop_last=not cpu_dataset,
                                        generator=gen,
                                    ),
                                    num_workers=probe_num_workers,
                                    pin_memory=probe_pin_memory,
                                    persistent_workers=_probe_persistent,
                                    prefetch_factor=_probe_prefetch,
                                )
                            else:
                                train_loader_lp = torch.utils.data.DataLoader(
                                    LP_train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=probe_num_workers,
                                    pin_memory=probe_pin_memory,
                                    drop_last=not cpu_dataset,
                                    generator=gen,
                                    persistent_workers=_probe_persistent,
                                    prefetch_factor=_probe_prefetch,
                                )
                            val_loader_lp = torch.utils.data.DataLoader(
                                LP_val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=probe_num_workers,
                                pin_memory=probe_pin_memory,
                                persistent_workers=_probe_persistent,
                                prefetch_factor=_probe_prefetch,
                            )
                            test_loader_lp = torch.utils.data.DataLoader(
                                LP_test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=probe_num_workers,
                                pin_memory=probe_pin_memory,
                                persistent_workers=_probe_persistent,
                                prefetch_factor=_probe_prefetch,
                            )

                        results = []
                        world_size = dist.get_world_size() if dist.is_initialized() else 1

                        solver = str(getattr(parameter, "solver", "adamw")).strip().lower()
                        sweep_optim_kwargs = dict(
                            optimizer_name=solver,
                            sgd_momentum=float(getattr(parameter, "sgd_momentum", 0.9)),
                        )
                        _n_lr = len(list(getattr(parameter, "lr_list", [])))
                        _n_wd = len(list(getattr(parameter, "weight_decay_list", [])))
                        safe_print(
                            f"[LP_eval] probe sweep: optimizer={solver}"
                            + (f" sgd_momentum={sweep_optim_kwargs['sgd_momentum']:g}" if solver == "sgd" else "")
                            + f" | grid: {_n_lr} lr x {_n_wd} wd = {_n_lr * _n_wd} parallel heads"
                        )

                        if cfg.dataset.task == "classification":
                            is_multilabel = cfg.dataset.get("is_multilabel", False)
                            max_epochs = getattr(parameter, "max_epochs", getattr(parameter, "max_steps", None))
                            if max_epochs is None:
                                raise ValueError("param.classification.max_epochs is required")

                            sweep_results = train_sweep_batched_probe_cls(
                                train_loader_lp,
                                val_loader_lp,
                                test_loader_lp,
                                lr_list=list(parameter.lr_list),
                                weight_decay_list=list(parameter.weight_decay_list),
                                max_epochs=int(max_epochs),
                                in_features=int(LP_train_dataset.logits.shape[-1]),
                                num_classes=int(cfg.dataset.num_classes),
                                is_multilabel=bool(is_multilabel),
                                device=cfg.device,
                                rank=rank,
                                world_size=world_size,
                                **sweep_optim_kwargs,
                            )
                        elif cfg.dataset.task in ["segmentation", "regression"]:
                            max_epochs = getattr(parameter, "max_epochs", getattr(parameter, "max_steps", None))
                            if max_epochs is None:
                                raise ValueError("param.segmentation.max_epochs is required")

                            sweep_results = train_sweep_batched_probe_seg(
                                train_loader_lp,
                                val_loader_lp,
                                test_loader_lp,
                                lr_list=list(parameter.lr_list),
                                weight_decay_list=list(parameter.weight_decay_list),
                                max_epochs=int(max_epochs),
                                in_features=int(LP_train_dataset.logits.shape[-1]),
                                num_classes=int(cfg.dataset.num_classes),
                                patch_size=int(patch_val),
                                task=str(cfg.dataset.task),
                                device=cfg.device,
                                rank=rank,
                                world_size=world_size,
                                **sweep_optim_kwargs,
                            )
                        else:
                            raise ValueError(f"Unsupported task={cfg.dataset.task!r}")

                        for r in sweep_results:
                            lr = r["lr"]
                            weight_decay = r["weight_decay"]
                            result = {
                                "lr": lr,
                                "weight_decay": weight_decay,
                                "val": r["val"],
                                "test": r["test"],
                                "scale": scale_value,
                                "out_patch_size": patch_val,
                                "pooling": pooling_value,
                                "seed": seed_value,
                                "norm_path": cfg.dataset.norm_path,
                            }
                            results.append(result)
                            all_results.append(result)
                            safe_print(
                                f"Linear Run finished | val={r['val']} | test={r['test']}"
                            )

                        del train_loader_lp, val_loader_lp, test_loader_lp
                        gc.collect()
                        torch.cuda.empty_cache()

                    del LP_train_dataset, LP_val_dataset, LP_test_dataset
                    gc.collect()
                    torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if rank == 0 and all_results:
        best_val = {}
        metric_names = sorted({m for r in all_results for m in r.get("val", {}).keys()})
        minimize_metrics = {"mse"}

        for metric in metric_names:
            minimize = metric in minimize_metrics
            best_score = float("inf") if minimize else -float("inf")
            best_config = None
            best_val_score = None
            best_test_score = None

            for res in all_results:
                if metric not in res["val"]:
                    continue
                score = res["val"][metric]
                is_better = score < best_score if minimize else score > best_score
                if is_better:
                    best_score = score
                    best_config = {
                        "lr": res["lr"],
                        "weight_decay": res["weight_decay"],
                        "scale": res["scale"],
                        "out_patch_size": res["out_patch_size"],
                        "pooling": res["pooling"],
                        "seed": res["seed"],
                        "norm_path": res["norm_path"],
                    }
                    best_val_score = res["val"]
                    best_test_score = res["test"]

            if best_config is not None:
                best_val[metric] = {
                    "config": best_config,
                    "val": best_val_score,
                    "test": best_test_score,
                }

        output_data = {}
        if len(all_knn_results) > 0:
            max_knn_acc = -float("inf")
            for knn_result in all_knn_results:
                if knn_result["best"]["test"]["accuracy"] > max_knn_acc:
                    max_knn_acc = knn_result["best"]["test"]["accuracy"]
                    best_knn = knn_result["best"]
            output_data["best_knn"] = best_knn
            output_data["best_LP"] = best_val
            output_data["knn_results"] = all_knn_results
            output_data["LP_results"] = all_results
        else:
            output_data["best_LP"] = best_val
            output_data["LP_results"] = all_results

        output_dir = resolve_output_dir(cfg)
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"{getattr(cfg.dataset, 'name', 'dataset')}_f{getattr(cfg.dataset, 'fraction', '1')}_p{getattr(cfg.dataset, 'partition', '1')}.json"
        output_path = os.path.join(output_dir, output_name)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        safe_print(f"Results saved to {output_path}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        safe_print(f"Error occurred: {e}")
