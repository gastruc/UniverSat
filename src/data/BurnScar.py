import os
from glob import glob

import numpy as np
import tifffile as tiff
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import apply_norm, load_norm


def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

class BurnScar(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
    ):
        """Initialize the HLS Burn Scars dataset.

        Link: https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars

        The upstream dataset only ships two on-disk folders, ``training/`` and
        ``validation/``. We treat ``validation/`` as our ``test`` split, and
        derive a 90/10 train/val split from ``training/`` with a fixed seed
        (see :py:meth:`get_train_val_split`) so results are reproducible.

        Args:
            path (str): Path to the dataset root (containing ``training/``
                and ``validation/`` folders).
            modalities (list): List of modalities to use. Only ``"hls"``
                is currently supported by this dataset.
            transform (callable): A function/transform to apply to the
                fully-loaded sample dict.
            split (str, optional): Which split to expose -- one of
                ``"train"``, ``"val"``, ``"test"``. Defaults to ``"train"``.
            norm_path (str, optional): Directory where ``NORM_<modality>.json``
                files live. If ``None``, no normalization is applied.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split

        self.split_mapping = {
            "train": "training",
            "val": "training",
            "test": "validation",
        }

        all_files = sorted(
            glob(
                os.path.join(
                    self.path, self.split_mapping[self.split], "*merged.tif"
                )
            )
        )
        all_targets = sorted(
            glob(
                os.path.join(
                    self.path, self.split_mapping[self.split], "*mask.tif"
                )
            )
        )

        if self.split != "test":
            split_indices = self.get_train_val_split(all_files)
            if self.split == "train":
                indices = split_indices["train"]
            else:
                indices = split_indices["val"]
            self.image_list = [all_files[i] for i in indices]
            self.target_list = [all_targets[i] for i in indices]
        else:
            self.image_list = all_files
            self.target_list = all_targets

        self.collate_fn = collate_fn
        self.norm = load_norm(norm_path, self.modalities, self)

    @staticmethod
    def get_train_val_split(all_files):
        # Fixed stratified sample to split data into train/val.
        # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set.
        train_idxs, val_idxs = train_test_split(
            np.arange(len(all_files)),
            test_size=0.1,
            random_state=23,
        )
        return {"train": train_idxs, "val": val_idxs}

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = tiff.imread(self.image_list[index])
        image = image.astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)

        target = tiff.imread(self.target_list[index])
        target = target.astype(np.int64)
        target = torch.from_numpy(target).long()

        invalid_mask = image == 9999          # (C, H, W)
        invalid_2d = invalid_mask.any(dim=0)  # (H, W)
        image[invalid_mask] = 0

        target[invalid_2d] = -1

        output = {
            "name": self.image_list[index],
            "hls": image.unsqueeze(0),
            "hls_dates": torch.tensor([0]),
            "label": target,
        }

        output = apply_norm(self.norm, output)

        # Reset invalid pixels to 0 *after* normalization so they land on the
        # per-channel neutral value instead of -mean/std outliers.
        if invalid_mask.any():
            output['hls'][0][invalid_mask] = 0.0

        return self.transform(output)

