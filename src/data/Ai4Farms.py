# AI4Farms / AI4SmallFarms-style field-boundary data.
#
# Image/mask pairs are resized to ``height``×``width``: bilinear for imagery, nearest for masks.
# Labels are then subsampled to the token grid (see ``num_patches``).
# Default on-disk layout (Sentinel-2 Asia release):
#   {path}/{region}/train/images/*.tif
#   {path}/{region}/train/masks/*.tif
# Example: AI4Farms/sentinel-2-asia/train/images/
#
# Legacy pipeline layout (optional): set subdir="patches" to use
#   {path}/{region}/{split}/patches/images|masks/

import math
import os
from glob import glob
from typing import Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import apply_norm, load_norm


def _align_mask_to_image(img_chw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    _, h, w = img_chw.shape
    mh, mw = mask_hw.shape
    if (mh, mw) == (h, w):
        return mask_hw
    m = torch.from_numpy(mask_hw.astype(np.float32)).view(1, 1, mh, mw)
    m = F.interpolate(m, size=(h, w), mode="nearest").squeeze().numpy()
    return (m > 0.5).astype(np.float32)


def _resize_image_and_mask(
    img_chw: np.ndarray,
    mask_hw: np.ndarray,
    th: int,
    tw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image and mask to ``(th, tw)``: bilinear for ``img_chw``, nearest for ``mask_hw``."""
    mask_hw = _align_mask_to_image(img_chw, mask_hw)
    im = torch.from_numpy(img_chw.astype(np.float32)).unsqueeze(0)
    m = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im, size=(th, tw), mode="bilinear", align_corners=False)
    m = F.interpolate(m, size=(th, tw), mode="nearest")
    img_out = im.squeeze(0).numpy().astype(np.float32)
    mask_out = (m.squeeze().numpy() > 0.5).astype(np.float32)
    return img_out, mask_out


def collate_fn(batch):
    keys = list(batch[0].keys())
    output = {}
    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


def _split_dir(split: str) -> str:
    m = {"train": "train", "val": "validate", "test": "test"}
    if split not in m:
        raise ValueError(f"split must be one of {list(m.keys())}, got {split!r}")
    return m[split]


def _resolve_split_folder(path: str, region: str, split: str) -> str:
    """Prefer AI4SmallFarms ``validate``; fall back to ``val`` if present."""
    sd = _split_dir(split)
    cand = os.path.join(path, region, sd, "images")
    if os.path.isdir(cand):
        return sd
    if split == "val":
        alt = os.path.join(path, region, "val", "images")
        if os.path.isdir(alt):
            return "val"
    return sd


def _gather_rasters(img_dir: str):
    out = []
    for pat in ("*.tif", "*.tiff", "*.TIF"):
        out.extend(glob(os.path.join(img_dir, pat)))
    return sorted(set(out))


def _find_mask_path(msk_dir: str, image_path: str) -> Optional[str]:
    base = os.path.basename(image_path)
    stem, ext = os.path.splitext(base)
    candidates = [
        os.path.join(msk_dir, base),
        os.path.join(msk_dir, "mask_" + base),
        os.path.join(msk_dir, base.replace("image_", "mask_", 1)),
        os.path.join(msk_dir, stem + "_mask" + ext),
        os.path.join(msk_dir, "mask_" + stem + ext),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


class Ai4Farms(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path=None,
        region: str = "sentinel-2-asia",
        subdir: str = "",
        height: int = 496,
        width: int = 496,
        num_patches: int = 1024,
        ignore_index: int = -1,
    ):
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split
        self.region = region
        self.height = int(height)
        self.width = int(width)
        self.num_patches = int(num_patches)
        self.ignore_index = int(ignore_index)
        gh = int(math.isqrt(self.num_patches))
        if gh * gh != self.num_patches:
            raise ValueError(
                f"num_patches must be a perfect square (e.g. 1024 → 32×32), got {self.num_patches}"
            )
        self._label_gh = gh
        self._label_gw = gh
        if subdir is None:
            subdir = ""

        sd = _resolve_split_folder(path, region, split)
        parts = [path, region, sd]
        if subdir:
            parts.append(subdir)
        img_dir = os.path.join(*parts, "images")
        msk_dir = os.path.join(*parts, "masks")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image directory: {img_dir}")
        if not os.path.isdir(msk_dir):
            raise FileNotFoundError(f"Missing mask directory: {msk_dir}")

        self.pairs = []
        for ip in _gather_rasters(img_dir):
            mp = _find_mask_path(msk_dir, ip)
            if mp is not None:
                self.pairs.append((ip, mp))

        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs found under {img_dir} / {msk_dir}")

        self.collate_fn = collate_fn
        self.norm = load_norm(norm_path, self.modalities, self)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ip, mp = self.pairs[index]
        with rasterio.open(ip) as src:
            img = src.read(out_dtype=np.float32)
        with rasterio.open(mp) as src:
            mask = src.read(1, out_dtype=np.float32)

        label_np = (mask > 0).astype(np.float32)
        img, label_np = _resize_image_and_mask(img, label_np, self.height, self.width)

        valid_mask = torch.ones((self.height, self.width), dtype=torch.bool)
        invalid_mask = torch.zeros((self.height, self.width), dtype=torch.bool)

        lab_long = torch.from_numpy(label_np).long()
        label_full = torch.where(
            valid_mask, lab_long, torch.full_like(lab_long, self.ignore_index)
        )

        gh, gw = self._label_gh, self._label_gw
        invalid_f = (label_full == self.ignore_index).float().view(1, 1, self.height, self.width)
        fg_f = (label_full == 1).float().view(1, 1, self.height, self.width)
        inv_d = F.interpolate(invalid_f, size=(gh, gw), mode="nearest").squeeze() > 0.5
        fg_d = F.interpolate(fg_f, size=(gh, gw), mode="nearest").squeeze() > 0.5
        label = torch.where(
            inv_d,
            torch.full_like(fg_d, self.ignore_index, dtype=torch.long),
            fg_d.long(),
        )

        x = torch.from_numpy(img).unsqueeze(0)

        out = {
            "name": ip,
            "label": label,
            "invalid_mask": invalid_mask,
        }
        for m in self.modalities:
            out[m] = x
            out[f"{m}_dates"] = torch.tensor([0])

        out = apply_norm(self.norm, out)
        if invalid_mask.any():
            for m in self.modalities:
                out[m][0][:, invalid_mask] = 0.0
        return self.transform(out)
