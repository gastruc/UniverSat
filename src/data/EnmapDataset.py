import json
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .utils import apply_norm, load_norm


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name" and modalities
    Returns:
        dict: batched dictionary
    """
    keys = list(batch[0].keys())
    output = {}
    if "name" in keys:
        output["name"] = [x["name"] for x in batch]
        keys.remove("name")
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output


class EnMAPDataset(Dataset):
    """Flexible EnMAP segmentation dataset with optional class remapping."""

    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        path: str,
        split: str = "train",
        transform: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        classes: Optional[List[int]] = None,
        raw_mask: bool = False,
        norm_path: Optional[str] = None,
        num_bands: int = 202,
        split_root: Optional[str] = None,
        image_root: str = "enmap",
        mask_root: Optional[str] = None,
        product: Optional[str] = None,
        classes_path: Optional[str] = None,
        split_subdir: Optional[str] = None,
        ignore_index: int = -1,
        class_mapping: Optional[dict[int, int]] = None,
        background_class: Optional[int] = 0,
        classif: bool = False,
    ) -> None:
        """
        Args:
            path (str): dataset root (also used to infer the product name).
            split (str): one of "train", "val", "test".
            transform (callable): transform applied to each sample.
            classes (list[int], optional): label codes to keep; inferred from
                disk (and cached to ``classes_path``) when omitted.
            raw_mask (bool): if True, return the mask unremapped.
            norm_path (str, optional): directory holding normalisation statistics.
            num_bands (int): number of spectral bands to keep per image.
            split_root / split_subdir / image_root / mask_root / product /
                classes_path (str, optional): override the on-disk layout;
                sensible defaults are inferred from ``path``.
            ignore_index (int): label assigned to unmapped pixels.
            class_mapping (dict[int, int], optional): raw-code -> target-code remap.
            background_class (int, optional): code excluded from foreground classes.
            classif (bool): if True, build a multi-label tile target instead of
                a dense segmentation mask.
        """
        if split not in self.valid_splits:
            raise ValueError(f"Split '{split}' not one of {self.valid_splits}.")

        self.path = path
        self.split = split
        self.transform = transform
        self.raw_mask = raw_mask
        self.num_bands = num_bands
        self.split_root = split_root
        self.image_root = image_root

        path_name = os.path.basename(os.path.normpath(self.path))
        inferred_product = product or path_name
        self.product = inferred_product

        inferred_mask_root = inferred_product.replace("enmap_", "")
        self.mask_root = mask_root or inferred_mask_root

        self.split_subdir = split_subdir or os.path.join("splits", self.product)
        self.class_mapping = (
            {int(raw_code): int(target_code) for raw_code, target_code in class_mapping.items()}
            if class_mapping is not None
            else None
        )
        self.background_class = background_class

        default_classes_path = os.path.join(self.path, f"{self.product}_classes.json")
        self.classes_path = classes_path or default_classes_path
        self.classif = classif

        self.collate_fn = collate_fn

        self.split_file = self._resolve_split_file()
        self.sample_collection = self._read_split_file()

        if classes is None and not raw_mask:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, "r") as f:
                    classes = json.load(f)
            else:
                classes = self._infer_all_classes()
                with open(self.classes_path, "w") as f:
                    json.dump(classes, f, indent=2)

        if classes is None:
            classes = [0]

        if 0 not in classes:
            classes = [0] + list(classes)

        self.classes = list(classes)
        if self.background_class is None:
            self.foreground_classes = list(self.classes)
        else:
            self.foreground_classes = [
                class_code for class_code in self.classes if class_code != self.background_class
            ]
        self.ignore_index = ignore_index
        self.num_classes = len(self.foreground_classes)
        self.mapping = {
            class_code: ordinal
            for ordinal, class_code in enumerate(self.foreground_classes)
        }

        self.norm = load_norm(norm_path, ["enmap"], self)

    def _resolve_split_file(self) -> str:
        candidate_paths = []

        if self.split_root is not None:
            candidate_paths.append(os.path.join(self.split_root, f"{self.split}.txt"))
        else:
            candidate_paths.extend(
                [
                    os.path.join(self.path, self.split_subdir, f"{self.split}.txt"),
                    os.path.join(os.path.dirname(self.path), self.split_subdir, f"{self.split}.txt"),
                    os.path.join("data", self.split_subdir, f"{self.split}.txt"),
                ]
            )

        for split_file in candidate_paths:
            if os.path.exists(split_file):
                return split_file

        search_locations = "\n- ".join(candidate_paths)
        raise FileNotFoundError(
            f"Official split file for '{self.product}' not found. Looked in:\n"
            f"- {search_locations}"
        )

    def _read_split_file(self) -> list:
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines() if x.strip()]

        sample_collection = [
            (
                os.path.join(self.path, self.image_root, sample_id),
                os.path.join(self.path, self.mask_root, sample_id),
            )
            for sample_id in sample_ids
        ]

        for img_path, mask_path in sample_collection:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        return sample_collection

    def _infer_all_classes(self) -> List[int]:
        mask_root = os.path.join(self.path, self.mask_root)
        mask_files = sorted(Path(mask_root).rglob("*.tif"))
        if len(mask_files) == 0:
            return [0]

        unique_codes = set([0])
        for mask_path in mask_files:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            unique_codes.update(np.unique(mask).tolist())

        return sorted(int(code) for code in unique_codes)

    def __len__(self) -> int:
        return len(self.sample_collection)

    def _load_image(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read()).float()
        return image[: self.num_bands]

    def _load_mask(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            return torch.from_numpy(src.read(1)).long()

    def _remap_mask(self, mask: Tensor) -> Tensor:
        mask_to_encode = mask

        if self.class_mapping is not None:
            converted_mask = torch.full_like(mask, fill_value=self.ignore_index)
            for raw_code, target_code in self.class_mapping.items():
                converted_mask[mask == raw_code] = target_code
            mask_to_encode = converted_mask

        new_mask = torch.full_like(mask, fill_value=self.ignore_index)
        for class_code, ordinal in self.mapping.items():
            new_mask[mask_to_encode == class_code] = ordinal
        return new_mask

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_path, mask_path = self.sample_collection[index]

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if not self.raw_mask:
            mask = self._remap_mask(mask)

        if self.classif:
            indices = torch.unique(mask)
            if self.background_class is not None and self.background_class in indices:
                indices = indices[indices != self.background_class]

            target = torch.zeros(self.num_classes, dtype=torch.int)
            target[indices] = 1
            mask = target

        sample = {
            "name": os.path.relpath(img_path, os.path.join(self.path, self.image_root)),
            "enmap": image,
            "label": mask,
        }

        sample = apply_norm(self.norm, sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class EnMAPEurocropsDataset(EnMAPDataset):
    """Backward-compatible alias for EnMAP EuroCrops dataset."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("product", "enmap_eurocrops")
        kwargs.setdefault("mask_root", "eurocrops")
        super().__init__(*args, **kwargs)
