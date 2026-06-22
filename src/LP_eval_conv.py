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
from typing import Any, Optional, Tuple

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
from torch.utils.data import Dataset

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


class InMemoryLogitsDataset(Dataset):
    """Simple in-memory dataset for logits and labels."""

    def __init__(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits = logits
        self.labels = labels

    def to(self, device):
        self.logits = self.logits.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)
        return self

    def __len__(self):
        return int(self.logits.shape[0])

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx]


class InMemorySpatialSegLogitsDataset(Dataset):
    """Per-image patch-token maps for full-image conv probes.

    ``logits`` are encoder tokens on a 2D grid ``(N, D, ht, wt)``.
    ``labels`` are dense maps ``(N, H, W)`` with pixel-level annotations.
    All samples in a split must share the same shapes.
    """

    def __init__(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.ndim != 4:
            raise ValueError(f"logits must be (N, D, ht, wt), got shape {tuple(logits.shape)}")
        if labels.ndim != 3:
            raise ValueError(f"labels must be (N, H, W), got shape {tuple(labels.shape)}")
        self.logits = logits
        self.labels = labels

    def to(self, device):
        self.logits = self.logits.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)
        return self

    def __len__(self) -> int:
        return int(self.logits.shape[0])

    def __getitem__(self, idx: int):
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
    ):
        self.logits_path = logits_path
        self.labels_path = labels_path
        self.logits = np.memmap(logits_path, mode="r+", dtype=logits_dtype, shape=logits_shape)
        self.labels = np.memmap(labels_path, mode="r+", dtype=labels_dtype, shape=labels_shape)
        self.length = int(length)

    def to(self, device):
        raise RuntimeError(
            "MemMapLogitsDataset is disk-backed and cannot be moved as a whole to GPU. "
            "Set `param.move_logits_to_gpu=False`."
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.logits[idx], copy=True))
        y = torch.from_numpy(np.array(self.labels[idx], copy=True))
        return x, y

    def close(self):
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
    """Zero-worker chunked loader for large memmap datasets."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        chunk_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
        prefetch: bool = True,
        drop_pages_after_chunk: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        cs = max(int(chunk_size), int(batch_size))
        self.chunk_size = max((cs // self.batch_size) * self.batch_size, self.batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        self.prefetch = bool(prefetch)
        self.drop_pages_after_chunk = bool(drop_pages_after_chunk)
        self.n = int(len(dataset))
        self.num_workers = 0
        self.pin_memory = False
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
        x = torch.from_numpy(np.asarray(self.dataset.logits[start:end]).copy())
        y = torch.from_numpy(np.asarray(self.dataset.labels[start:end]).copy())
        return x, y

    def _yield_from_chunk(self, x, y, chunk_start, chunk_end):
        bs = self.batch_size
        if self.shuffle:
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
        self._fadvise_dontneed(chunk_start, chunk_end)

    def __iter__(self):
        n = self.n
        cs = self.chunk_size
        chunk_starts = list(range(0, n, cs))
        if self.shuffle:
            chunk_order = torch.randperm(len(chunk_starts), generator=self.generator).tolist()
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
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
            t.join(timeout=5.0)


def mean_iou(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_label: int = -1
):
    device = predictions.device
    labels = labels.to(device)
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    valid_mask = labels != ignore_label
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id) & valid_mask
        label_mask = (labels == class_id) & valid_mask
        intersection[class_id] = (pred_mask & label_mask).sum().float()
        union[class_id] = (pred_mask | label_mask).sum().float()
    iou = intersection / (union + 1e-8)
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
) -> Any:
    """Extract encoder features into a cached dataset for probe training.

    For **classification**: returns flat (N, D) features via mean/max pooling over tokens.
    For **segmentation / regression**: returns spatial token maps (N, D, ht, wt) paired
    with dense pixel labels (N, H, W). The probe then applies convolutions on this token
    grid and upsamples its output to (H, W) for loss/metric computation.

    Key design decision: ``out_patch`` passed to the model forward is always set to
    ``num_patches`` for segmentation/regression so the model produces one token per ViT
    patch (matching the spatial grid derived from label dims / patch_size).
    """

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

    cls_out_patch = int((num_patches * res**2) // (10 * scale) ** 2)
    if task == "classification":
        out_patch = cls_out_patch
    else:
        if patch_size is None:
            raise ValueError("patch_size (out_patch_size) must be provided for segmentation/regression")
        out_patch = num_patches // (patch_size ** 2)

    model.eval()
    device_type = "cuda" if "cuda" in device else "cpu"

    logits_torch_dtype = torch.float16 if logits_dtype == "float16" else torch.float32
    labels_torch_dtype = torch.int8

    logits_storage = None
    labels_storage = None
    logits_path = None
    labels_path = None
    write_idx = 0
    total_capacity = None
    fraction_cursor = 0
    spatial_mode = task in ("segmentation", "regression")
    spatial_logits_chunks: list[torch.Tensor] = []
    spatial_labels_chunks: list[torch.Tensor] = []
    image_idx_global = 0
    spatial_ref: Optional[tuple] = None  # (hp, wp, H, W, D, kind)

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
                x = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in micro_batch.items()
                }
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    if hasattr(model, "extract_lp_logits"):
                        logit = model.extract_lp_logits(x, device=device)
                    else:
                        logit = model(
                            x,
                            sensor.wavelengths,
                            sensor.input_res,
                            scale,
                            cls_out_patch,   # always pass the classification token count
                            out_patch,       # this now correctly == num_patches for seg/reg
                            sensor.subpatches,
                            keep_subpatch=False,
                        )[0]

                # Strip register tokens; logit is now (B, num_spatial_tokens, D)
                logit = logit[:, model.n_registers:, :]

                if task == "classification":
                    if pooling == "mean":
                        logit = logit.mean(dim=1)   # (B, D)
                    else:
                        logit = logit.max(dim=1).values  # (B, D)

                if spatial_mode:
                    if patch_size is None:
                        raise ValueError("patch_size must be provided for segmentation/regression")

                    batch_labels_t = micro_batch["label"]

                    if task == "segmentation":
                        if batch_labels_t.ndim != 3:
                            raise ValueError(
                                "Segmentation requires dense label maps shaped (B, H, W); "
                                f"got ndim={batch_labels_t.ndim}."
                            )
                        h, w = int(batch_labels_t.shape[1]), int(batch_labels_t.shape[2])
                    else:
                        if batch_labels_t.ndim == 3:
                            h, w = int(batch_labels_t.shape[1]), int(batch_labels_t.shape[2])
                        elif batch_labels_t.ndim == 4:
                            h, w = int(batch_labels_t.shape[2]), int(batch_labels_t.shape[3])
                        else:
                            raise ValueError(
                                "Regression requires labels shaped (B, H, W) or (B, C, H, W); "
                                f"got ndim={batch_labels_t.ndim}."
                            )

                    if h % patch_size != 0 or w % patch_size != 0:
                        raise ValueError(
                            f"Label spatial dims must be divisible by patch_size "
                            f"(got H,W={h},{w}, patch_size={patch_size})."
                        )

                    # hp, wp: token grid dimensions derived from label resolution
                    hp, wp = h // patch_size, w // patch_size
                    num_tok = int(logit.shape[1])

                    if hp * wp != num_tok:
                        raise RuntimeError(
                            f"Token count mismatch: model produced {num_tok} tokens, "
                            f"but label grid needs hp*wp={hp}*{wp}={hp*wp} "
                            f"(label {h}x{w}, patch_size={patch_size}). "
                            f"Check that num_patches in cfg matches the ViT token count "
                            f"for this image size."
                        )

                    d_ch = int(logit.shape[2])
                    kind = "seg" if task == "segmentation" else "reg"

                    # Rearrange flat token sequence → 2D spatial grid (B, D, hp, wp)
                    logit_spatial = (
                        rearrange(logit, "b (hp wp) d -> b d hp wp", hp=hp, wp=wp)
                        .to(logits_torch_dtype)
                        .cpu()
                    )

                    if task == "segmentation":
                        labs_cpu = batch_labels_t.to(labels_torch_dtype).cpu()
                    else:
                        labs_cpu = batch_labels_t.float().cpu()

                    is_first_spatial = spatial_ref is None
                    if spatial_ref is None:
                        spatial_ref = (hp, wp, h, w, d_ch, kind)
                    else:
                        ref_hp, ref_wp, ref_h, ref_w, ref_d, ref_kind = spatial_ref
                        if (hp, wp, h, w, d_ch, kind) != (ref_hp, ref_wp, ref_h, ref_w, ref_d, ref_kind):
                            raise RuntimeError(
                                "Spatial cache requires fixed label resolution and token grid across batches "
                                f"(expected hp,wp={ref_hp}x{ref_wp}, labels {ref_h}x{ref_w}, D={ref_d}, kind={ref_kind}; "
                                f"got {hp}x{wp}, {h}x{w}, D={d_ch}, kind={kind})."
                            )

                    if use_memmap and is_first_spatial:
                        try:
                            total_items_sp = len(dataloader.dataset)
                        except Exception:
                            total_items_sp = len(dataloader) * (getattr(dataloader, "batch_size", None) or 1)
                        total_capacity_sp = max(1, int(total_items_sp))
                        target_dir = memmap_dir or tempfile.gettempdir()
                        os.makedirs(target_dir, exist_ok=True)
                        logits_path = os.path.join(
                            target_dir, f"lp_spatial_{name}_rank{rank}_{os.getpid()}_logits.dat"
                        )
                        labels_path = os.path.join(
                            target_dir, f"lp_spatial_{name}_rank{rank}_{os.getpid()}_labels.dat"
                        )
                        np_logits_dtype = np.float16 if logits_torch_dtype == torch.float16 else np.float32
                        np_labels_dtype = np.int8 if kind == "seg" else np.float32
                        logits_storage = np.memmap(
                            logits_path, mode="w+", dtype=np_logits_dtype,
                            shape=(total_capacity_sp, d_ch, hp, wp),
                        )
                        labels_storage = np.memmap(
                            labels_path, mode="w+", dtype=np_labels_dtype,
                            shape=(total_capacity_sp,) + tuple(labs_cpu.shape[1:]),
                        )

                    for bi in range(int(batch_labels_t.shape[0])):
                        if fraction < 1.0:
                            gi = float(image_idx_global)
                            keep = (gi * float(fraction)).floor() > ((gi - 1.0) * float(fraction)).floor()
                            image_idx_global += 1
                            if not keep:
                                continue
                        if use_memmap:
                            if logits_storage is None:
                                raise RuntimeError("Spatial memmap was not initialized (internal error).")
                            logits_storage[write_idx] = np.asarray(logit_spatial[bi].numpy(), dtype=logits_storage.dtype)
                            labels_storage[write_idx] = np.asarray(labs_cpu[bi].numpy(), dtype=labels_storage.dtype)
                            write_idx += 1
                        else:
                            spatial_logits_chunks.append(logit_spatial[bi : bi + 1])
                            spatial_labels_chunks.append(labs_cpu[bi : bi + 1])

                    if fraction >= 1.0:
                        image_idx_global += int(batch_labels_t.shape[0])

                    del x, logit, logit_spatial, micro_batch
                    continue

                logits_batch_cpu_full = logit.to(logits_torch_dtype).cpu()
                labels_batch_cpu_full = micro_batch["label"].to(labels_torch_dtype).cpu()

                if logits_storage is None:
                    try:
                        total_items = len(dataloader.dataset)
                    except Exception:
                        total_items = len(dataloader) * (getattr(dataloader, "batch_size", None) or 1)
                    total_capacity = int(total_items)
                    logits_shape = (total_capacity, logits_batch_cpu_full.shape[-1])
                    labels_shape = (
                        (total_capacity, *labels_batch_cpu_full.shape[1:])
                        if labels_batch_cpu_full.ndim > 1
                        else (total_capacity,)
                    )
                    if use_memmap:
                        target_dir = memmap_dir or tempfile.gettempdir()
                        os.makedirs(target_dir, exist_ok=True)
                        logits_path = os.path.join(target_dir, f"lp_{name}_rank{rank}_{os.getpid()}_logits.dat")
                        labels_path = os.path.join(target_dir, f"lp_{name}_rank{rank}_{os.getpid()}_labels.dat")
                        np_logits_dtype = np.float16 if logits_torch_dtype == torch.float16 else np.float32
                        logits_storage = np.memmap(logits_path, mode="w+", dtype=np_logits_dtype, shape=logits_shape)
                        labels_storage = np.memmap(labels_path, mode="w+", dtype=np.int8, shape=labels_shape)
                    else:
                        logits_storage = torch.empty(logits_shape, dtype=logits_torch_dtype)
                        labels_storage = torch.empty(labels_shape, dtype=labels_torch_dtype)

                logits_batch_cpu = logits_batch_cpu_full
                labels_batch_cpu = labels_batch_cpu_full

                if fraction < 1.0:
                    n_rows = logits_batch_cpu.shape[0]
                    row_idx = torch.arange(fraction_cursor, fraction_cursor + n_rows, dtype=torch.float32)
                    keep_mask = (row_idx * float(fraction)).floor() > ((row_idx - 1.0) * float(fraction)).floor()
                    fraction_cursor += n_rows
                    if not keep_mask.any():
                        continue
                    logits_batch_cpu = logits_batch_cpu[keep_mask]
                    labels_batch_cpu = labels_batch_cpu[keep_mask]

                end = write_idx + logits_batch_cpu.shape[0]
                if total_capacity is not None and end > total_capacity:
                    raise RuntimeError("Preallocated logits buffer exceeded.")

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

    # --- Return spatial dataset ---
    if spatial_mode:
        if use_memmap:
            if write_idx == 0:
                raise RuntimeError("No batches processed (spatial memmap); check dataloader or fraction.")
            logits_storage.flush()
            labels_storage.flush()
            ref_hp, ref_wp, ref_h, ref_w, ref_d, _ = spatial_ref
            return MemMapLogitsDataset(
                logits_path=logits_path,
                labels_path=labels_path,
                logits_shape=(write_idx, ref_d, ref_hp, ref_wp),
                labels_shape=(write_idx,) + tuple(labels_storage.shape[1:]),
                logits_dtype=logits_storage.dtype,
                labels_dtype=labels_storage.dtype,
                length=write_idx,
            )
        if not spatial_logits_chunks:
            raise RuntimeError("No batches processed (spatial); check dataloader or fraction.")
        return InMemorySpatialSegLogitsDataset(
            torch.cat(spatial_logits_chunks, dim=0),
            torch.cat(spatial_labels_chunks, dim=0),
        )

    # --- Return classification dataset ---
    if write_idx == 0:
        raise RuntimeError("No batches processed; check dataloader or fraction.")

    if use_memmap:
        logits_storage.flush()
        labels_storage.flush()
        logits_shape_final = (write_idx,) + tuple(logits_storage.shape[1:])
        labels_shape_final = (write_idx,) + tuple(labels_storage.shape[1:]) if labels_storage.ndim > 1 else (write_idx,)
        return MemMapLogitsDataset(
            logits_path=logits_path,
            labels_path=labels_path,
            logits_shape=logits_shape_final,
            labels_shape=labels_shape_final,
            logits_dtype=logits_storage.dtype,
            labels_dtype=labels_storage.dtype,
            length=write_idx,
        )

    return InMemoryLogitsDataset(logits_storage[:write_idx], labels_storage[:write_idx])


def load_model_from_checkpoint(ckpt_path: str) -> torch.nn.Module:
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
    return model, sensor_dict


def load_model_from_hf(repo_id: str):
    """Load the released UniverSat encoder from the HuggingFace Hub.

    Uses the hubconf ``from_pretrained`` entrypoint (config.json + model.safetensors)
    and returns the bare encoder -- the same object ``load_model_from_checkpoint``
    yields (forward keeps the register tokens, exposes ``n_registers``) -- paired
    with the default sensor metadata.
    """
    import hubconf

    safe_print(f"Loading released UniverSat weights from HuggingFace Hub: {repo_id}")
    model = hubconf.UniverSat.from_pretrained(repo_id).model

    path_sensor = "configs/model/network/sensor/default.yaml"
    safe_print(f"Loading sensor config from {path_sensor}")
    sensor_dict = OmegaConf.load(path_sensor)
    return model, sensor_dict


def load_feature_model(cfg: DictConfig):
    """Priority: foundation-model adapter, then local ckpt (cfg.ckpt_path), else HF Hub."""
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


def _set_lr_schedule_all_groups(optimizer, step, total_steps):
    cos_term = math.cos(math.pi * step / max(1, total_steps))
    rel_lr = 0.5 * (1.0 + cos_term)
    for group in optimizer.param_groups:
        base_lr = group.get("base_lr", group.get("lr", 0.0))
        group["lr"] = base_lr * rel_lr
    return rel_lr


def _try_make_fused_adamw(params_groups):
    try:
        return torch.optim.AdamW(params_groups, lr=0.0, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    except TypeError:
        return torch.optim.AdamW(params_groups, lr=0.0, weight_decay=0.0, betas=(0.9, 0.95))


def _make_probe_optimizer(params_groups, kind, *, momentum=0.9):
    """Build the probe-sweep optimizer.

    ``kind="adamw"`` -> fused AdamW (default). ``kind="sgd"`` -> SGD(momentum) DINOv2 linear-eval
    style. Per-group ``lr`` (set to 0 here, driven by the scheduler) and ``weight_decay`` honored.
    """
    k = str(kind).strip().lower()
    if k in ("adamw", "adam"):
        return _try_make_fused_adamw(params_groups)
    if k == "sgd":
        return torch.optim.SGD(params_groups, lr=0.0, momentum=float(momentum))
    raise ValueError(f"Unknown probe optimizer={kind!r}; use 'adamw' or 'sgd'.")


_STD_CENTER_MODES = {"center"}
_STD_ZSCORE_MODES = {"standard", "standardscaler", "standard_scaler", "zscore", "z-score"}


def _rows_per_chunk(logits, chunk_size, *, bytes_per_elem, max_chunk_bytes=1_000_000_000):
    """Rows (dim 0) per streaming chunk, capped so one working copy fits a byte budget.

    For flat (N, D) caches a row is tiny, so ``chunk_size`` is used as-is. For 4D conv
    caches (N, D, ht, wt) a single row can be GBs, so this shrinks the chunk (down to 1
    row) to avoid materializing the whole tensor at higher precision (the cause of OOM).
    """
    row_elems = 1
    for s in tuple(logits.shape)[1:]:
        row_elems *= int(s)
    row_elems = max(row_elems, 1)
    max_rows = max(1, int(max_chunk_bytes // (row_elems * int(bytes_per_elem))))
    return max(1, min(int(chunk_size), max_rows))


def _compute_feature_mean_std(train_ds, *, device, world_size, chunk_size, eps=1e-6):
    """Per-channel (dim 1) mean/std over TRAIN features (fp64 streaming, DDP-safe)."""
    logits = train_ds.logits
    n = int(len(train_ds))
    D = int(logits.shape[1])
    nd = len(logits.shape)
    reduce_dims = tuple(d for d in range(nd) if d != 1)
    s1 = torch.zeros(D, dtype=torch.float64)  # sum x
    s2 = torch.zeros(D, dtype=torch.float64)  # sum x^2
    count = 0
    # chunk_size counts ROWS (dim 0). For 4D conv caches a "row" is a whole (D, ht, wt)
    # image — possibly GBs — so cap rows so the fp64 working copy stays within budget.
    cs = _rows_per_chunk(logits, chunk_size, bytes_per_elem=8)
    for st in range(0, n, cs):
        en = min(n, st + cs)
        x = logits[st:en]
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.asarray(x))
        x = x.detach().cpu().to(torch.float64)
        s1 += x.sum(dim=reduce_dims)
        s2 += (x * x).sum(dim=reduce_dims)
        count += int(x.numel() // D)  # scalar positions contributed per channel

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


def _apply_standardizer(ds, mu, sigma, *, zscore, chunk_size):
    """Apply (x - mu)[/ sigma] in place per-channel (dim 1), tensor or memmap, in chunks."""
    logits = ds.logits
    n = int(len(ds))
    is_memmap = not torch.is_tensor(logits)
    inv = (1.0 / sigma) if zscore else torch.ones_like(sigma)
    nd = len(logits.shape)
    view = [1] * nd
    view[1] = -1                       # broadcast stats over the channel axis (dim 1)
    mu_b = mu.view(view)
    inv_b = inv.view(view)
    cs = _rows_per_chunk(logits, chunk_size, bytes_per_elem=4)  # fp32 working copy
    inplace_ctx = torch.inference_mode() if (not is_memmap and logits.is_inference()) else nullcontext()
    with inplace_ctx:
        for st in range(0, n, cs):
            en = min(n, st + cs)
            if is_memmap:
                chunk = torch.from_numpy(np.asarray(logits[st:en])).float()
                chunk = (chunk - mu_b) * inv_b
                logits[st:en] = chunk.numpy().astype(logits.dtype)
            else:
                dev = logits.device
                chunk = (logits[st:en].float() - mu_b.to(dev)) * inv_b.to(dev)
                logits[st:en] = chunk.to(logits.dtype)
    if is_memmap and hasattr(logits, "flush"):
        logits.flush()


def _standardize_feature_datasets(train_ds, val_ds, test_ds, *, mode, device, world_size, chunk_size):
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


def _probe_in_features_from_logits_ds(ds: Any) -> int:
    """Channel dim D for linear inputs (N, D) or spatial caches (N, D, ht, wt)."""
    if isinstance(ds, (InMemorySpatialSegLogitsDataset,)):
        return int(ds.logits.shape[1])
    if isinstance(ds, MemMapLogitsDataset) and int(getattr(ds.logits, "ndim", 0)) == 4:
        return int(ds.logits.shape[1])
    return int(ds.logits.shape[-1])


class LayerNormLinearClassifier(nn.Module):
    """CAPI-style classification probe head with one hidden layer.

    Mirrors the segmentation ``ConvHead`` scheme (in → 128 → BN → ReLU → C), using
    Linear/BatchNorm1d instead of Conv2d/BatchNorm2d since classification features are
    flat ``(B, D)`` with no spatial grid. A LayerNorm normalizes the input features
    (CAPI-style) before the MLP.

    Input:  (B, D)  — pooled encoder features.
    Output: (B, C)  — per-sample class logits.
    """

    def __init__(self, in_features: int, out_features: int, hidden_features: int = 128):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        # in → 128 → BN → ReLU → C, the flat-feature analog of ConvHead.
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)  →  out: (B, C)
        return self.head(self.ln(x))


class BatchedLayerNormLinearProbes(nn.Module):
    """Batched wrapper over many classification probe heads. Returns [H, B, C].

    Loops over heads and stacks (mirroring ``BatchedConvHeads``); the hidden-layer
    BatchNorm means heads cannot share a single vectorized forward, so each head keeps
    its own running statistics.
    """

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        if len(heads) == 0:
            raise ValueError("heads must be non-empty")
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        # Each head returns (B, C); stack over H dim.
        return torch.stack([h(x) for h in self.heads], dim=0)  # [H, B, C]


class ConvHead(nn.Module):
    """Segmentation/regression probe operating on the ViT token grid.

    Input:  (B, D, ht, wt)  — spatial token map from the encoder cache.
    Output: (B, C, ht, wt)  — per-token class logits on the same grid.

    The caller (train/eval loops) upsamples this to (B, C, H, W) for
    loss and metric computation. No spatial flattening happens inside this module.
    """

    def __init__(self, embedding_size: int = 384, num_classes: int = 5):
        super().__init__()
        # 3×3 conv with padding=1 preserves (ht, wt) so the output stays on the token grid.
        self.head = nn.Sequential(
            nn.Conv2d(embedding_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.ln = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, ht, wt)  →  out: (B, C, ht, wt)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return self.head(x)


class BatchedConvHeads(nn.Module):
    """Batched wrapper: returns [H, B, C, ht, wt] preserving spatial dims."""

    def __init__(self, heads: list[nn.Module]):
        super().__init__()
        if len(heads) == 0:
            raise ValueError("heads must be non-empty")
        self.heads = nn.ModuleList(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, ht, wt)
        # Each head returns (B, C, ht, wt); stack over H dim.
        return torch.stack([h(x) for h in self.heads], dim=0)  # [H, B, C, ht, wt]


def _upsample_logits_to_label_grid(
    logits_all: torch.Tensor,
    *,
    label_hw: tuple[int, int],
) -> torch.Tensor:
    """Upsample [H, B, C, ht, wt] → [H*B, C, H_lab, W_lab].

    Encoder features stay on (ht, wt); only probe outputs are interpolated.
    Labels are never resized.
    """
    H_lab, W_lab = int(label_hw[0]), int(label_hw[1])
    # Merge H and B for a single interpolate call
    hb, c, ht, wt = logits_all.shape[0] * logits_all.shape[1], logits_all.shape[2], logits_all.shape[3], logits_all.shape[4]
    x = logits_all.reshape(hb, c, ht, wt).float()
    return F.interpolate(x, size=(H_lab, W_lab), mode="bilinear", align_corners=False)  # [H*B, C, H_lab, W_lab]


def _multiclass_metrics_from_confusion(cm: torch.Tensor) -> tuple[float, float, float]:
    eps = torch.finfo(cm.dtype).eps
    diag = torch.diagonal(cm, 0)
    row_sum = cm.sum(dim=1)
    col_sum = cm.sum(dim=0)
    precision = diag / (col_sum + eps)
    recall = diag / (row_sum + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    active = (row_sum > 0) | (col_sum > 0)
    f1_macro = f1[active].mean() if bool(active.any()) else torch.zeros((), device=cm.device, dtype=cm.dtype)
    total = cm.sum()
    acc = diag.sum() / (total + eps)
    return float(acc.item()), float(f1_macro.item()), float(acc.item())


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
    probe_container.eval()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    if is_multilabel:
        all_preds_flat = []
        all_labels_flat = []
        for batch_emb, batch_labels in data_loader:
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits_all = probe_container(batch_emb)  # [H, B, C]
            preds_all = (logits_all.sigmoid() > 0.5).to(torch.int64)
            preds_bh = preds_all.transpose(0, 1).reshape(-1, num_classes).cpu()
            labels_bh = batch_labels.to(torch.int64).unsqueeze(1).expand(-1, heads, -1).reshape(-1, num_classes).cpu()
            all_preds_flat.append(preds_bh)
            all_labels_flat.append(labels_bh)
        preds_np = torch.cat(all_preds_flat, dim=0).numpy().astype(bool)
        labels_np = torch.cat(all_labels_flat, dim=0).numpy().astype(int)
        n_total = preds_np.shape[0] // heads
        preds_mat = preds_np.reshape(n_total, heads, num_classes)
        labels_mat = labels_np.reshape(n_total, heads, num_classes)
        accs, f1_macros, f1_micros = [], [], []
        for h in range(heads):
            accs.append(float(accuracy_score(labels_mat[:, h, :], preds_mat[:, h, :])))
            f1_macros.append(float(f1_score(labels_mat[:, h, :], preds_mat[:, h, :], average="macro")))
            f1_micros.append(float(f1_score(labels_mat[:, h, :], preds_mat[:, h, :], average="micro")))
        return {"accuracy": accs, "f1_macro": f1_macros, "f1_micro": f1_micros}

    else:
        conf = torch.zeros((heads, num_classes, num_classes), dtype=torch.float64, device=device)
        for batch_emb, batch_labels in data_loader:
            batch_emb = batch_emb.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits_all = probe_container(batch_emb)  # [H, B, C]
            preds = logits_all.argmax(dim=-1).long().clamp(0, num_classes - 1)  # [H, B]
            labels = batch_labels.long().clamp(0, num_classes - 1)  # [B]
            if labels.numel() == 0:
                continue
            h_dim, b_dim = preds.shape
            hh = torch.arange(h_dim, device=device).view(-1, 1).expand(-1, b_dim).reshape(-1)
            lab = labels.view(1, -1).expand(h_dim, -1).reshape(-1)
            pr = preds.reshape(-1)
            conf.view(-1).index_add_(0, hh * num_classes * num_classes + lab * num_classes + pr,
                                     torch.ones(h_dim * b_dim, device=device, dtype=torch.float64))
        accs, f1_macros, f1_micros = [], [], []
        for h in range(heads):
            a, fm, fmi = _multiclass_metrics_from_confusion(conf[h])
            accs.append(a)
            f1_macros.append(fm)
            f1_micros.append(fmi)
        return {"accuracy": accs, "f1_macro": f1_macros, "f1_micro": f1_micros}


def train_sweep_batched_probe_cls(
    train_loader, val_loader, test_loader,
    *, lr_list, weight_decay_list, max_epochs, in_features, num_classes,
    is_multilabel, device, rank, world_size, val_every_n_epochs=1,
    optimizer_name="adamw", sgd_momentum=0.9,
):
    specs = list(itertools.product(lr_list, weight_decay_list))
    heads = len(specs)
    probes = [LayerNormLinearClassifier(in_features, num_classes) for _ in range(heads)]
    container = BatchedLayerNormLinearProbes(probes).to(device)

    params_groups: dict = defaultdict(lambda: {"params": [], "base_lr": 0.0})
    for (lr, wd), head in zip(specs, probes):
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            group_wd = 0.0 if "bias" in name else float(wd)
            key = (float(lr), float(group_wd))
            params_groups[key]["params"].append(p)
            params_groups[key]["base_lr"] = float(lr)

    optim_groups = [{"params": g["params"], "lr": 0.0, "weight_decay": wd, "base_lr": g["base_lr"]}
                    for (_, wd), g in params_groups.items()]
    optimizer = _make_probe_optimizer(optim_groups, optimizer_name, momentum=sgd_momentum)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    metric_name = "f1_macro" if is_multilabel else "accuracy"
    best_scores = torch.full((heads,), -float("inf"))
    best_params = [None] * heads
    loss_fn_ml = nn.MultiLabelSoftMarginLoss()
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    data_iter = iter(train_loader)
    for _epoch in range(max_epochs):
        ep = _epoch + 1
        if heads > 1 and (ep == 1 or ep % 10 == 0 or ep == max_epochs):
            safe_print(f"[LP CLS sweep] Epoch {ep}/{max_epochs}")
        container.train()
        for _ in range(steps_per_epoch):
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
                    loss = torch.stack([loss_fn_ml(logits_all[h], batch_labels.float()) for h in range(heads)]).mean()
                else:
                    stack = logits_all.permute(1, 2, 0)  # [B, C, H]
                    target = batch_labels.long().unsqueeze(1).expand(-1, heads)
                    loss = F.cross_entropy(stack, target)
            loss.backward()
            _set_lr_schedule_all_groups(optimizer, global_step, total_steps)
            optimizer.step()
            global_step += 1

        should_val = (ep % max(1, val_every_n_epochs) == 0) or ep == max_epochs

        if heads == 1:
            metrics_train = _evaluate_batched_probe_cls(
                train_loader, container, is_multilabel=is_multilabel,
                device=device, heads=heads, num_classes=num_classes,
            )
            metrics_val = _evaluate_batched_probe_cls(
                val_loader, container, is_multilabel=is_multilabel,
                device=device, heads=heads, num_classes=num_classes,
            )
            ts = float(metrics_train[metric_name][0])
            vs = float(metrics_val[metric_name][0])
            safe_print(
                f"[LP CLS sweep] Epoch {ep}/{max_epochs} train {metric_name}={ts:.4f} "
                f"val {metric_name}={vs:.4f}"
            )
            if should_val:
                scores = torch.tensor(metrics_val[metric_name])
                improved = scores > best_scores
                if improved.any():
                    for h in range(heads):
                        if not improved[h]:
                            continue
                        best_scores[h] = scores[h]
                        head = container.heads[h]
                        best_params[h] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            continue

        if not should_val:
            continue

        metrics = _evaluate_batched_probe_cls(val_loader, container, is_multilabel=is_multilabel,
                                               device=device, heads=heads, num_classes=num_classes)
        scores = torch.tensor(metrics[metric_name])
        improved = scores > best_scores
        if improved.any():
            for h in range(heads):
                if not improved[h]:
                    continue
                best_scores[h] = scores[h]
                head = container.heads[h]
                best_params[h] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    for h in range(heads):
        if best_params[h] is not None:
            container.heads[h].load_state_dict({k: v.to(device) for k, v in best_params[h].items()})

    val_metrics = _evaluate_batched_probe_cls(val_loader, container, is_multilabel=is_multilabel,
                                               device=device, heads=heads, num_classes=num_classes)
    test_metrics = _evaluate_batched_probe_cls(test_loader, container, is_multilabel=is_multilabel,
                                                device=device, heads=heads, num_classes=num_classes)
    return [{"lr": float(lr), "weight_decay": float(wd),
             "val": {k: float(val_metrics[k][h]) for k in val_metrics},
             "test": {k: float(test_metrics[k][h]) for k in test_metrics}}
            for h, (lr, wd) in enumerate(specs)]


@torch.no_grad()
def _evaluate_batched_probe_seg(
    data_loader,
    probe_container: nn.Module,
    *,
    task: str,
    device: torch.device,
    heads: int,
    num_classes: int,
    label_hw: tuple[int, int],
) -> dict[str, list[float]]:
    """Evaluate batched segmentation/regression heads.

    ``data_loader`` yields ``(batch_emb, batch_labels)`` where:
      - ``batch_emb``:   (B, D, ht, wt)   — spatial token maps
      - ``batch_labels``: (B, H, W)         — dense pixel labels

    ``probe_container`` returns ``[H, B, C, ht, wt]`` (no flattening).
    Logits are upsampled to (H, W) before loss/metric computation.
    """
    if task not in {"segmentation", "regression"}:
        raise ValueError(f"Unsupported task={task!r}")

    probe_container.eval()
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    H_lab, W_lab = label_hw

    if task == "segmentation":
        conf = torch.zeros((heads, num_classes, num_classes), dtype=torch.float64, device=device)
    else:
        sse = torch.zeros((heads,), dtype=torch.float64, device=device)
        count = torch.zeros((heads,), dtype=torch.float64, device=device)

    for batch_emb, batch_labels in data_loader:
        batch_emb = batch_emb.to(device, non_blocking=True)    # (B, D, ht, wt)
        batch_labels = batch_labels.to(device, non_blocking=True)  # (B, H, W)
        bsz = batch_emb.shape[0]

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_all = probe_container(batch_emb)  # [H, B, C, ht, wt]

        # Upsample probe outputs from token grid to label resolution
        logits_up = _upsample_logits_to_label_grid(logits_all, label_hw=(H_lab, W_lab))
        # logits_up: [H*B, C, H_lab, W_lab]

        if task == "segmentation":
            preds = logits_up.argmax(dim=1)  # [H*B, H_lab, W_lab]
            preds = preds.reshape(heads, bsz, H_lab, W_lab)   # [H, B, H_lab, W_lab]
            labels = batch_labels.long()  # (B, H, W)

            lab = labels.unsqueeze(0).expand(heads, -1, -1, -1)  # [H, B, H, W]
            valid = lab != -1
            if valid.any():
                y = lab[valid].clamp(0, num_classes - 1)
                p = preds[valid].clamp(0, num_classes - 1)
                h_idx = (
                    torch.arange(heads, device=device).view(-1, 1, 1, 1)
                    .expand(heads, bsz, H_lab, W_lab)[valid]
                )
                conf.view(-1).index_add_(
                    0,
                    h_idx * num_classes * num_classes + y * num_classes + p,
                    torch.ones(y.shape[0], device=device, dtype=torch.float64),
                )
        else:
            # Regression: MSE between upsampled predictions and labels
            # logits_up: [H*B, C, H_lab, W_lab] → [H, B, C, H_lab, W_lab]
            pred_5 = logits_up.reshape(heads, bsz, num_classes, H_lab, W_lab).float()
            if batch_labels.ndim == 3:
                tgt = batch_labels.float().unsqueeze(0).unsqueeze(2).expand(heads, bsz, num_classes, H_lab, W_lab)
            else:
                tgt = batch_labels.float().unsqueeze(0).expand(heads, bsz, num_classes, H_lab, W_lab)
            diff = pred_5 - tgt
            sse += diff.pow(2).sum(dim=(1, 2, 3, 4)).to(torch.float64)
            count += float(diff[0].numel())

    if task == "segmentation":
        miou_list, micro_iou_list = [], []
        for h in range(heads):
            cm = conf[h]
            inter = torch.diag(cm)
            union = cm.sum(1) + cm.sum(0) - inter
            valid = union > 0
            miou_h = (inter[valid] / (union[valid] + 1e-8)).mean() if valid.any() else torch.tensor(0.0)
            micro_h = inter.sum() / (union.sum() + 1e-8)
            miou_list.append(float(miou_h.item()))
            micro_iou_list.append(float(micro_h.item()))
        return {"miou": miou_list, "micro_iou": micro_iou_list}

    mse = sse / torch.clamp(count, min=1.0)
    return {"mse": [float(x.item()) for x in mse]}


def train_sweep_batched_probe_seg(
    train_loader, val_loader, test_loader,
    *, lr_list, weight_decay_list, max_epochs, in_features, num_classes,
    patch_size, task, device, rank, world_size, val_every_n_epochs=1,
    label_hw: tuple[int, int],
    optimizer_name="adamw", sgd_momentum=0.9,
) -> list[dict[str, Any]]:
    """Train batched conv probe heads for segmentation/regression.

    Data flow:
      cache:       (N, D, ht, wt)  token maps  +  (N, H, W) pixel labels
      ConvHead:    (B, D, ht, wt) → (B, C, ht, wt)     [token-grid logits]
      upsample:    (B, C, ht, wt) → (B, C, H, W)        [pixel-grid logits]
      loss/metric: pixel-grid logits  vs  (B, H, W) labels
    """
    if task not in {"segmentation", "regression"}:
        raise ValueError(f"Unsupported task={task!r}")

    specs = list(itertools.product(lr_list, weight_decay_list))
    heads = len(specs)
    H_lab, W_lab = label_hw

    # ConvHead no longer needs patch_size — it always operates on the token grid
    # and the caller handles upsampling.
    probes = [ConvHead(embedding_size=in_features, num_classes=num_classes) for _ in range(heads)]
    container = BatchedConvHeads(probes).to(device)

    params_groups: dict = defaultdict(lambda: {"params": [], "base_lr": 0.0})
    for (lr, wd), head in zip(specs, probes):
        for name, p in head.named_parameters():
            if not p.requires_grad:
                continue
            group_wd = 0.0 if "bias" in name else float(wd)
            key = (float(lr), float(group_wd))
            params_groups[key]["params"].append(p)
            params_groups[key]["base_lr"] = float(lr)

    optim_groups = [{"params": g["params"], "lr": 0.0, "weight_decay": wd, "base_lr": g["base_lr"]}
                    for (_, wd), g in params_groups.items()]
    optimizer = _make_probe_optimizer(optim_groups, optimizer_name, momentum=sgd_momentum)

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max_epochs * steps_per_epoch
    global_step = 0

    metric_name = "miou" if task == "segmentation" else "mse"
    maximize = task == "segmentation"
    best_scores = torch.full((heads,), -float("inf") if maximize else float("inf"))
    best_params = [None] * heads
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    data_iter = iter(train_loader)
    for _epoch in range(max_epochs):
        ep = _epoch + 1
        if heads > 1 and (ep == 1 or ep % 10 == 0 or ep == max_epochs):
            safe_print(f"[LP SEG/REG sweep] Epoch {ep}/{max_epochs}")
        container.train()
        for _ in range(steps_per_epoch):
            try:
                batch_emb, batch_labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_emb, batch_labels = next(data_iter)

            batch_emb = batch_emb.to(device, non_blocking=True)    # (B, D, ht, wt)
            batch_labels = batch_labels.to(device, non_blocking=True)  # (B, H, W)
            bsz = batch_emb.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits_all = container(batch_emb)  # [H, B, C, ht, wt]

                # Upsample to label resolution
                logits_up = _upsample_logits_to_label_grid(logits_all, label_hw=(H_lab, W_lab))
                # logits_up: [H*B, C, H_lab, W_lab]

                if task == "segmentation":
                    labels_rep = (
                        batch_labels.long()
                        .unsqueeze(0).expand(heads, bsz, H_lab, W_lab)
                        .reshape(heads * bsz, H_lab, W_lab)
                    )
                    loss = ce_loss(logits_up, labels_rep)
                else:
                    pred_5 = logits_up.reshape(heads, bsz, num_classes, H_lab, W_lab).float()
                    if batch_labels.ndim == 3:
                        tgt = batch_labels.float().unsqueeze(0).unsqueeze(2).expand(heads, bsz, num_classes, H_lab, W_lab)
                    else:
                        tgt = batch_labels.float().unsqueeze(0).expand(heads, bsz, num_classes, H_lab, W_lab)
                    tgt = tgt.reshape(heads * bsz, num_classes, H_lab, W_lab)
                    loss = F.mse_loss(logits_up.float(), tgt)

            loss.backward()
            _set_lr_schedule_all_groups(optimizer, global_step, total_steps)
            optimizer.step()
            global_step += 1

        should_val = (ep % max(1, val_every_n_epochs) == 0) or ep == max_epochs

        if heads == 1:
            mt = _evaluate_batched_probe_seg(
                train_loader, container, task=task, device=device,
                heads=heads, num_classes=num_classes, label_hw=label_hw,
            )
            mv = _evaluate_batched_probe_seg(
                val_loader, container, task=task, device=device,
                heads=heads, num_classes=num_classes, label_hw=label_hw,
            )
            ts = float(mt[metric_name][0])
            vs = float(mv[metric_name][0])
            if metric_name == "mse":
                safe_print(
                    f"[LP SEG/REG sweep] Epoch {ep}/{max_epochs} train {metric_name}={ts:.6f} "
                    f"val {metric_name}={vs:.6f}"
                )
            else:
                safe_print(
                    f"[LP SEG/REG sweep] Epoch {ep}/{max_epochs} train {metric_name}={ts:.4f} "
                    f"val {metric_name}={vs:.4f}"
                )
            if should_val:
                scores = torch.tensor(mv[metric_name])
                improved = (scores > best_scores) if maximize else (scores < best_scores)
                if improved.any():
                    for h in range(heads):
                        if not improved[h]:
                            continue
                        best_scores[h] = scores[h]
                        best_params[h] = {k: v.detach().cpu().clone() for k, v in container.heads[h].state_dict().items()}
            continue

        if not should_val:
            continue

        metrics = _evaluate_batched_probe_seg(
            val_loader, container, task=task, device=device,
            heads=heads, num_classes=num_classes, label_hw=label_hw,
        )
        scores = torch.tensor(metrics[metric_name])
        improved = (scores > best_scores) if maximize else (scores < best_scores)
        if improved.any():
            for h in range(heads):
                if not improved[h]:
                    continue
                best_scores[h] = scores[h]
                best_params[h] = {k: v.detach().cpu().clone() for k, v in container.heads[h].state_dict().items()}

    for h in range(heads):
        if best_params[h] is not None:
            container.heads[h].load_state_dict({k: v.to(device) for k, v in best_params[h].items()})

    val_metrics = _evaluate_batched_probe_seg(val_loader, container, task=task, device=device,
                                               heads=heads, num_classes=num_classes, label_hw=label_hw)
    test_metrics = _evaluate_batched_probe_seg(test_loader, container, task=task, device=device,
                                                heads=heads, num_classes=num_classes, label_hw=label_hw)

    return [{"lr": float(lr), "weight_decay": float(wd),
             "val": {k: float(v[h]) for k, v in val_metrics.items()},
             "test": {k: float(v[h]) for k, v in test_metrics.items()}}
            for h, (lr, wd) in enumerate(specs)]


def _gather_to_rank0(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    supported_types = [torch.float32, torch.float16, torch.bfloat16,
                       torch.float64, torch.long, torch.int32, torch.uint8]
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    working_tensor = tensor.to(device)
    orig_dtype = tensor.dtype
    if orig_dtype not in supported_types:
        working_tensor = working_tensor.to(torch.long)
    working_tensor = working_tensor.contiguous()
    local_size = torch.tensor([working_tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = int(max(s.item() for s in all_sizes))
    if working_tensor.shape[0] < max_size:
        pad_shape = list(working_tensor.shape)
        pad_shape[0] = max_size - working_tensor.shape[0]
        working_tensor = torch.cat([working_tensor, torch.zeros(pad_shape, dtype=working_tensor.dtype, device=device)])
    gathered = [torch.zeros_like(working_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, working_tensor)
    if rank == 0:
        return torch.cat([g[:int(s.item())] for g, s in zip(gathered, all_sizes)]).to(orig_dtype).cpu()
    return None


def evaluate_knn_cls(train_dataset, val_dataset, test_dataset, k_list):
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
        logits = logits.float()
        invalid_mask = torch.isnan(logits).any(1) | torch.isinf(logits).any(1)
        if invalid_mask.any():
            safe_print(f"Warning: {invalid_mask.sum().item()} NaN/Inf rows removed from {name}.")
            logits = logits[~invalid_mask]
            labels = labels[~invalid_mask]
        if logits.shape[0] == 0:
            raise ValueError(f"{name} is empty after NaN removal.")
        return logits.cpu().numpy(), labels.cpu().numpy().astype(int)

    train_X, train_y = torch_to_clean_numpy(train_logits, train_labels, "train")
    val_X, val_y = torch_to_clean_numpy(val_logits, val_labels, "val")
    test_X, test_y = torch_to_clean_numpy(test_logits, test_labels, "test")

    best_val_acc = -float("inf")
    best_result = None
    all_k_results = []

    for cosine in [False, True]:
        if cosine:
            train_X = normalize(train_X, axis=1, norm="l2")
            val_X = normalize(val_X, axis=1, norm="l2")
            test_X = normalize(test_X, axis=1, norm="l2")
        for k in k_list:
            current_k = min(k, len(train_X))
            safe_print(f"  KNN {'cosine ' if cosine else ''}k={current_k}: fitting...")
            knn = KNeighborsClassifier(n_neighbors=current_k, n_jobs=-1)
            knn.fit(train_X, train_y)
            val_preds = knn.predict(val_X)
            test_preds = knn.predict(test_X)
            val_m = {"accuracy": float(accuracy_score(val_y, val_preds)),
                     "f1_macro": float(f1_score(val_y, val_preds, average="macro")),
                     "f1_micro": float(f1_score(val_y, val_preds, average="micro"))}
            test_m = {"accuracy": float(accuracy_score(test_y, test_preds)),
                      "f1_macro": float(f1_score(test_y, test_preds, average="macro")),
                      "f1_micro": float(f1_score(test_y, test_preds, average="micro"))}
            k_result = {"k": k, "val": val_m, "test": test_m, "cosine": cosine}
            all_k_results.append(k_result)
            safe_print(f"  KNN {'cosine ' if cosine else ''}k={k}: val_acc={val_m['accuracy']:.4f}, test_acc={test_m['accuracy']:.4f}")
            if val_m["accuracy"] > best_val_acc:
                best_val_acc = val_m["accuracy"]
                best_result = k_result

    return {"best": best_result, "all_k": all_k_results}


@hydra.main(version_base="1.3", config_path="../configs", config_name="LP_eval.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    rank = dist.get_rank() if dist.is_initialized() else 0

    utils.extras(cfg)

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pooling_list = getattr(cfg.param.classification, "pooling_list", ["mean"])
    if cfg.dataset.task == "segmentation":
        pooling_list = ["mean"]
    scale_list = getattr(cfg.param, "scale_list", [1])
    seed_list = getattr(cfg.param, "seed_list", [1])
    norm_path_list = getattr(cfg.param, "norm_path_list", [None])

    if cfg.dataset.task in ["segmentation", "regression"]:
        out_patch_list = getattr(cfg.param.segmentation, "out_patch_size_list",
                                 [getattr(cfg.dataset, "out_patch_size", None)])
    else:
        out_patch_list = [None]

    original_norm_path = getattr(cfg.dataset, "norm_path", None)
    all_results = []
    all_knn_results = []

    for norm_override in norm_path_list:
        cfg.dataset.norm_path = original_norm_path if norm_override is None else norm_override

        # Loaded lazily per (scale, out_patch, pooling) pass — released after logits are cached so LP probes
        # do not share VRAM with the foundation encoder (would otherwise OOM during BatchedConvHeads + full-res CE).
        model = None
        sensor_dict = None

        for scale_value in scale_list:
            for out_patch_value in out_patch_list:
                patch_val = out_patch_value if out_patch_value is not None else getattr(cfg.dataset, "out_patch_size", None)
                if cfg.dataset.task in ["segmentation", "regression"] and patch_val is None:
                    raise ValueError("out_patch_size must be set for segmentation/regression tasks")

                for pooling_value in pooling_list:
                    if model is None:
                        model, sensor_dict = load_feature_model(cfg)
                        model = model.to(cfg.device)
                        model.eval()

                    set_seed(seed_list[0] if seed_list else 1)

                    datamodule = hydra.utils.instantiate(cfg.datamodule)
                    num_patches = cfg.dataset.num_patches
                    datamodule.setup("fit")
                    train_loader = datamodule.train_dataloader()
                    val_loader = datamodule.val_dataloader()
                    datamodule.setup("test")
                    test_loader = datamodule.test_dataloader()

                    fraction = cfg.dataset.get("fraction", 1.0)

                    def _extract(loader, split_name):
                        ds = get_logits(
                            model, loader, split_name, sensor_dict,
                            num_patches=num_patches,
                            res=cfg.dataset.res,
                            scale=scale_value,
                            task=cfg.dataset.task,
                            device=cfg.device,
                            fraction=fraction if split_name == "train" else 1.0,
                            patch_size=patch_val,
                            logits_dtype=getattr(cfg.dataset, "logits_dtype", "float16"),
                            pooling=pooling_value,
                            use_memmap=bool(getattr(cfg.dataset, "use_memmap", False)),
                            memmap_dir=getattr(cfg.dataset, "memmap_dir", None),
                        )
                        return ds

                    LP_train_dataset = _extract(train_loader, "train")
                    del train_loader; gc.collect(); torch.cuda.empty_cache()

                    LP_val_dataset = _extract(val_loader, "val")
                    del val_loader; gc.collect(); torch.cuda.empty_cache()

                    LP_test_dataset = _extract(test_loader, "test")
                    del test_loader; gc.collect(); torch.cuda.empty_cache()

                    del model
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    safe_print("Released foundation model from GPU before LP probe training.")

                    safe_print(f"Train logits: {LP_train_dataset.logits.shape}, labels: {LP_train_dataset.labels.shape}")
                    safe_print(f"Val   logits: {LP_val_dataset.logits.shape},   labels: {LP_val_dataset.labels.shape}")
                    safe_print(f"Test  logits: {LP_test_dataset.logits.shape},  labels: {LP_test_dataset.labels.shape}")

                    # --- Feature standardization (fit on train, applied to all splits) ---
                    # Per-channel z-score/center on the extracted features, before KNN/probe.
                    # Handles both (N, D) classification caches and (N, D, ht, wt) conv caches.
                    _std_mode = getattr(cfg.param, "standardization", None)
                    if _std_mode is not None and str(_std_mode).strip().lower() not in ("", "none", "null"):
                        _standardize_feature_datasets(
                            LP_train_dataset, LP_val_dataset, LP_test_dataset,
                            mode=str(_std_mode),
                            device=cfg.device,
                            world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                            chunk_size=int(getattr(cfg.param, "standardize_chunk_size", 1_000_000)),
                        )

                    # Derive label_hw once from the cached dataset for seg/reg
                    if cfg.dataset.task in ["segmentation", "regression"]:
                        # labels shape: (N, H, W) — take H, W from train
                        _lbl = LP_train_dataset.labels
                        label_hw = (int(_lbl.shape[-2]), int(_lbl.shape[-1]))
                        safe_print(f"Label resolution for probe training: {label_hw}")

                    def _get_bytes(arr):
                        if hasattr(arr, "numel") and hasattr(arr, "element_size"):
                            return arr.numel() * arr.element_size()
                        if hasattr(arr, "nbytes"):
                            return arr.nbytes
                        return np.inf

                    size_bytes = (_get_bytes(LP_train_dataset.logits) + _get_bytes(LP_train_dataset.labels)
                                  + _get_bytes(LP_val_dataset.logits) + _get_bytes(LP_val_dataset.labels)
                                  + _get_bytes(LP_test_dataset.logits) + _get_bytes(LP_test_dataset.labels))

                    # KNN (classification only)
                    is_corine = any("corine" in str(getattr(cfg.dataset, k, "")).lower() for k in ["name", "product", "norm_path"])
                    if cfg.dataset.task == "classification" and not is_corine:
                        knn_k_list = list(getattr(cfg.param.classification, "knn_k_list", [1, 3, 5, 10, 20]))
                        if not isinstance(LP_train_dataset, MemMapLogitsDataset):
                            knn_result = evaluate_knn_cls(LP_train_dataset, LP_val_dataset, LP_test_dataset, knn_k_list)
                            if knn_result is not None:
                                all_knn_results.append({"scale": scale_value, "pooling": pooling_value,
                                                        "norm_path": cfg.dataset.norm_path, **knn_result})
                                safe_print(f"KNN best: k={knn_result['best']['k']}, test_acc={knn_result['best']['test']['accuracy']:.4f}")

                    # GPU move decision
                    _is_memmap = isinstance(LP_train_dataset, MemMapLogitsDataset)
                    move_to_gpu = bool(getattr(cfg.param, "move_logits_to_gpu", True))
                    cpu_dataset = True
                    if not _is_memmap and move_to_gpu and torch.cuda.is_available():
                        gpu_frac = float(getattr(cfg.param, "logits_gpu_memory_fraction", 0.8))
                        free_bytes, _ = torch.cuda.mem_get_info(device=cfg.device)
                        if size_bytes <= int(free_bytes * gpu_frac):
                            safe_print(f"Moving logits to VRAM ({size_bytes/1024**3:.2f} GB).")
                            LP_train_dataset.to(cfg.device)
                            LP_val_dataset.to(cfg.device)
                            LP_test_dataset.to(cfg.device)
                            cpu_dataset = False
                        else:
                            safe_print(f"Keeping logits on CPU (need {size_bytes/1024**3:.2f} GB, free {free_bytes/1024**3:.2f} GB).")

                    if cfg.dataset.task == "classification":
                        parameter = cfg.param.classification
                    else:
                        parameter = cfg.param.segmentation

                    for seed_value in seed_list:
                        set_seed(seed_value)
                        gen = torch.Generator()
                        gen.manual_seed(seed_value)

                        batch_size = int(parameter.batch_size)
                        min_bs = max(1, math.ceil(len(LP_train_dataset) / 500))
                        batch_size = max(batch_size, 2 ** math.ceil(math.log2(min_bs)))
                        max_bs_16 = 2 ** math.floor(math.log2(max(1, len(LP_train_dataset) // 64)))
                        batch_size = min(batch_size, max_bs_16)
                        max_probe_batch = int(getattr(parameter, "max_probe_batch_size", None) or getattr(cfg.dataset, "max_probe_batch_size", 50000))
                        batch_size = min(batch_size, max(max_probe_batch, 1))
                        safe_print(f"Probe batch size: {batch_size} ({len(LP_train_dataset)} train samples)")

                        probe_num_workers = int(getattr(parameter, "probe_num_workers",
                                                        getattr(cfg.dataset, "probe_num_workers", 12 if cpu_dataset else 0)))
                        probe_pin_memory = bool(getattr(parameter, "probe_pin_memory",
                                                        getattr(cfg.dataset, "probe_pin_memory", cpu_dataset)))

                        # Build the three loaders. Wrap the whole branch in a
                        # helper called *unconditionally* so no control-flow path
                        # (continue, try/except, indentation slip) can leave the
                        # loaders unset between iterations.
                        def _build_loaders():
                            if _is_memmap:
                                chunk_size = int(getattr(cfg.dataset, "probe_memmap_chunk_size", 1_000_000))
                                prefetch = bool(getattr(cfg.dataset, "probe_memmap_prefetch", True))
                                drop_pages = bool(getattr(cfg.dataset, "probe_memmap_drop_pages", True))

                                if cfg.dataset.task in ("segmentation", "regression"):
                                    lt = LP_train_dataset.logits
                                    lb = LP_train_dataset.labels
                                    row_bytes = int(np.prod(lt.shape[1:]) * np.dtype(lt.dtype).itemsize)
                                    row_bytes += int(np.prod(lb.shape[1:]) * np.dtype(lb.dtype).itemsize)
                                    max_gb_cfg = float(getattr(cfg.dataset, "probe_memmap_max_chunk_gb", 24.0))
                                    max_gb = max_gb_cfg
                                    if prefetch:
                                        max_gb = max_gb / 2.0
                                    max_rows_by_ram = int((max_gb * (1024**3)) / max(row_bytes, 1))
                                    old_cs = chunk_size
                                    chunk_size = min(chunk_size, max_rows_by_ram, len(LP_train_dataset))
                                    aligned = (chunk_size // int(batch_size)) * int(batch_size)
                                    if aligned == 0 and len(LP_train_dataset) > 0:
                                        one = min(int(batch_size), len(LP_train_dataset))
                                        safe_print(
                                            f"[ChunkedMemmapLoader] WARNING: RAM budget (~{max_gb_cfg} GiB config, "
                                            f"prefetch→{max_gb:.1f} GiB per chunk) fits <1 full batch of spatial rows "
                                            f"(~{row_bytes / 1024**2:.1f} MiB/row). Using chunk_size={one} "
                                            f"(~{one * row_bytes / 1024**3:.1f} GiB); raise "
                                            f"dataset.probe_memmap_max_chunk_gb or set probe_memmap_prefetch=false if OOM."
                                        )
                                        chunk_size = one
                                    else:
                                        chunk_size = min(aligned, len(LP_train_dataset))
                                    safe_print(
                                        f"[ChunkedMemmapLoader] spatial memmap: ~{row_bytes / 1024**2:.1f} MiB/row, "
                                        f"chunk_size {old_cs} -> {chunk_size}"
                                        + (
                                            f" (cap probe_memmap_max_chunk_gb={max_gb_cfg}, prefetch uses half the budget)"
                                            if prefetch
                                            else f" (cap probe_memmap_max_chunk_gb={max_gb_cfg})"
                                        )
                                        + "."
                                    )

                                return (
                                    ChunkedMemmapLoader(LP_train_dataset, batch_size, chunk_size, shuffle=True,  drop_last=True,  generator=gen, prefetch=prefetch, drop_pages_after_chunk=drop_pages),
                                    ChunkedMemmapLoader(LP_val_dataset,   batch_size, chunk_size, shuffle=False, drop_last=False, prefetch=prefetch, drop_pages_after_chunk=drop_pages),
                                    ChunkedMemmapLoader(LP_test_dataset,  batch_size, chunk_size, shuffle=False, drop_last=False, prefetch=prefetch, drop_pages_after_chunk=drop_pages),
                                )

                            num_workers = probe_num_workers
                            if cpu_dataset and num_workers > 0:
                                torch.multiprocessing.set_sharing_strategy("file_system")
                                shm_threshold = int(getattr(cfg.dataset, "probe_shm_risky_threshold", 20_000_000))
                                if len(LP_train_dataset) >= shm_threshold:
                                    safe_print(f"Setting probe_num_workers=0 (len={len(LP_train_dataset)} >= {shm_threshold}).")
                                    num_workers = 12
                            _persistent = num_workers > 0
                            _prefetch = int(getattr(cfg.dataset, "probe_prefetch_factor", 2)) if num_workers > 0 else None
                            return (
                                torch.utils.data.DataLoader(LP_train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=probe_pin_memory, drop_last=not cpu_dataset, generator=gen, persistent_workers=_persistent, prefetch_factor=_prefetch),
                                torch.utils.data.DataLoader(LP_val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=probe_pin_memory, persistent_workers=_persistent, prefetch_factor=_prefetch),
                                torch.utils.data.DataLoader(LP_test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=probe_pin_memory, persistent_workers=_persistent, prefetch_factor=_prefetch),
                            )

                        train_loader_lp, val_loader_lp, test_loader_lp = _build_loaders()

                        world_size = dist.get_world_size() if dist.is_initialized() else 1
                        val_every = max(1, int(getattr(cfg.param, "probe_val_every_n_epochs", 1)))

                        solver = str(getattr(parameter, "solver", "adamw")).strip().lower()
                        sweep_optim_kwargs = dict(
                            optimizer_name=solver,
                            sgd_momentum=float(getattr(parameter, "sgd_momentum", 0.9)),
                        )
                        _n_lr = len(list(getattr(parameter, "lr_list", [])))
                        _n_wd = len(list(getattr(parameter, "weight_decay_list", [])))
                        safe_print(
                            f"[LP_eval_conv] probe sweep: optimizer={solver}"
                            + (f" sgd_momentum={sweep_optim_kwargs['sgd_momentum']:g}" if solver == "sgd" else "")
                            + f" | grid: {_n_lr} lr x {_n_wd} wd = {_n_lr * _n_wd} parallel heads"
                        )

                        if cfg.dataset.task == "classification":
                            is_multilabel = cfg.dataset.get("is_multilabel", False)
                            max_epochs = getattr(parameter, "max_epochs", None) or getattr(parameter, "max_steps", None)
                            if max_epochs is None:
                                raise ValueError("param.classification.max_epochs is required")
                            sweep_results = train_sweep_batched_probe_cls(
                                train_loader_lp, val_loader_lp, test_loader_lp,
                                lr_list=list(parameter.lr_list),
                                weight_decay_list=list(parameter.weight_decay_list),
                                max_epochs=int(max_epochs),
                                in_features=int(LP_train_dataset.logits.shape[-1]),
                                num_classes=int(cfg.dataset.num_classes),
                                is_multilabel=bool(is_multilabel),
                                device=cfg.device, rank=rank, world_size=world_size,
                                val_every_n_epochs=val_every,
                                **sweep_optim_kwargs,
                            )
                        elif cfg.dataset.task in ["segmentation", "regression"]:
                            max_epochs = getattr(parameter, "max_epochs", None) or getattr(parameter, "max_steps", None)
                            if max_epochs is None:
                                raise ValueError("param.segmentation.max_epochs is required")
                            sweep_results = train_sweep_batched_probe_seg(
                                train_loader_lp, val_loader_lp, test_loader_lp,
                                lr_list=list(parameter.lr_list),
                                weight_decay_list=list(parameter.weight_decay_list),
                                max_epochs=int(max_epochs),
                                in_features=_probe_in_features_from_logits_ds(LP_train_dataset),
                                num_classes=int(cfg.dataset.num_classes),
                                patch_size=int(patch_val),
                                task=str(cfg.dataset.task),
                                device=cfg.device, rank=rank, world_size=world_size,
                                val_every_n_epochs=val_every,
                                label_hw=label_hw,
                                **sweep_optim_kwargs,
                            )
                        else:
                            raise ValueError(f"Unsupported task={cfg.dataset.task!r}")

                        for r in sweep_results:
                            result = {
                                "lr": r["lr"], "weight_decay": r["weight_decay"],
                                "val": r["val"], "test": r["test"],
                                "scale": scale_value, "out_patch_size": patch_val,
                                "pooling": pooling_value, "seed": seed_value,
                                "norm_path": cfg.dataset.norm_path,
                            }
                            all_results.append(result)
                            safe_print(f"Run | val={r['val']} | test={r['test']}")

                        # Drop refs so DataLoader workers / memmap pages get
                        # cleaned up. The loaders are unconditionally rebuilt
                        # by ``_build_loaders()`` at the top of the next
                        # iteration, so this is safe.
                        del train_loader_lp, val_loader_lp, test_loader_lp
                        gc.collect(); torch.cuda.empty_cache()

                    del LP_train_dataset, LP_val_dataset, LP_test_dataset
                    gc.collect(); torch.cuda.empty_cache()

    if rank == 0 and all_results:
        minimize_metrics = {"mse"}
        metric_names = sorted({m for r in all_results for m in r.get("val", {}).keys()})
        best_val = {}
        for metric in metric_names:
            minimize = metric in minimize_metrics
            best_score = float("inf") if minimize else -float("inf")
            best_config = best_val_score = best_test_score = None
            for res in all_results:
                if metric not in res["val"]:
                    continue
                score = res["val"][metric]
                if (score < best_score) if minimize else (score > best_score):
                    best_score = score
                    best_config = {k: res[k] for k in ("lr", "weight_decay", "scale", "out_patch_size", "pooling", "seed", "norm_path")}
                    best_val_score = res["val"]
                    best_test_score = res["test"]
            if best_config is not None:
                best_val[metric] = {"config": best_config, "val": best_val_score, "test": best_test_score}

        output_data = {}
        if all_knn_results:
            best_knn = max(all_knn_results, key=lambda x: x["best"]["test"]["accuracy"])["best"]
            output_data["best_knn"] = best_knn
            output_data["knn_results"] = all_knn_results
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
        raise
