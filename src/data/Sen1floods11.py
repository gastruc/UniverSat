# Source: https://github.com/cloudtostreet/Sen1Floods11

import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
import torch
from .utils import load_norm, apply_norm

from torch.utils.data import Dataset

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

class Sen1Floods11(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        norm_path = None,
    ):
        """Initialize the Sen1Floods11 dataset.
        Link: https://github.com/cloudtostreet/Sen1Floods11

        Args:
            path (str): Path to the dataset.
            modalities (list): List of modalities to use.
            transform (callable): A function/transform to apply to the data.
            split (str, optional): Split of the dataset ('train', 'val', 'test'). Defaults to 'train'.
            norm_path (str, optional): Path for normalization data. Defaults to None.
        """
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.split = split

        self.split_mapping = {"train": "train", "val": "valid", "test": "test"}

        split_file = os.path.join(
            self.path,
            "v1.1",
            f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv",
        )
        metadata_file = os.path.join(
            self.path, "v1.1", "Sen1Floods11_Metadata.geojson"
        )
        data_root = os.path.join(
            self.path, "v1.1", "data/flood_events/HandLabeled/"
        )

        self.metadata = geopandas.read_file(metadata_file)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[1]) for f in file_list
        ]

        self.collate_fn = collate_fn
        self.norm = load_norm(norm_path, self.modalities, self)

    def __len__(self):
        return len(self.s1_image_list)

    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            s2_date = pd.to_datetime("01-01-1998", dayfirst=True)
            s1_date = pd.to_datetime("01-01-1998", dayfirst=True)
        else:
            s2_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s2_date"].item()
            )
            s1_date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s1_date"].item()
            )
        return torch.tensor([s2_date.dayofyear]), torch.tensor([s1_date.dayofyear])

    def __getitem__(self, index):
        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()

        s1_image = np.nan_to_num(s1_image, nan=0.0, posinf=0.0, neginf=0.0)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)

        timestamp = self._get_date(index)

        s1_image = torch.from_numpy(s1_image).float()
        ratio_band = s1_image[:1, :, :] / (s1_image[1:, :, :] + 1e-10)
        ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4)
        s1_image = torch.cat((s1_image[:2, :, :], ratio_band), dim=0)

        target = torch.from_numpy(target).long()

        output = {
            "name": str(index),
            "s1": s1_image.unsqueeze(0),
            "label": target,
            "s1_dates": timestamp[1],
        }

        output = apply_norm(self.norm, output)

        return self.transform(output)
