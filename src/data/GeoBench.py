import ast
import io
import json
import os
import pickle
from typing import Dict, List

import h5py
import numpy as np
import torch
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

class IgnoreGeoBenchUnpickler(pickle.Unpickler):
    """Unpickler that swaps ``geobench`` classes for a dummy, so attribute
    dicts can be read without importing the original geobench package."""

    def find_class(self, module, name):
        if module.startswith("geobench"):
            class Dummy:
                def __new__(cls, *_args, **_kwargs):
                    return super(Dummy, cls).__new__(cls)
            return Dummy
        return super().find_class(module, name)

class GeoBench(Dataset):
    """GeoBench benchmark tiles stored as per-sample HDF5 files.

    Each sample is one ``.hdf5`` file; the split-to-samples mapping is read from
    ``default_partition.json`` at the dataset root.
    """

    def __init__(
        self,
        path,
        modalities: Dict[str, List[str]],
        transform,
        split: str = "train",
        norm_path: str = None,
        normalize=True,
    ):
        """
        Args:
            path (str): path to the dataset
            modalities (Dict[str, List[str]]): keys are modality names, values the list of bands to load
            transform (callable): transform to apply to each sample
            split (str): one of "train", "val"/"valid" or "test"
            norm_path (str): directory holding per-modality normalisation statistics
            normalize (bool): whether to apply normalisation to the loaded modalities
        """
        super(GeoBench, self).__init__()
        self.path = path
        self.modalities = modalities
        self.transform = transform
        self.norm_path = norm_path
        # Accept "val" as an alias for "valid"
        if split == "val":
            split = "valid"
        assert split in ["train", "valid", "test"], "split must be one of 'train', 'valid' or 'test'"
        self.split = split

        with open(os.path.join(path, "default_partition.json"), "r") as f:
            split_index = json.load(f)
            assert self.split in split_index.keys(), f"split {self.split} not found in partition file"
            self.sample_ids = split_index[self.split]

        self.norm = None
        if normalize:
            self.modalities_to_norm = [m for m in self.modalities if m != 'label']
            self.norm = load_norm(self.norm_path, self.modalities_to_norm, self)

        self.collate_fn = collate_fn

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        name = self.sample_ids[idx] + ".hdf5"

        out = {}
        with h5py.File(os.path.join(self.path, name), 'r') as f:

            raw = ast.literal_eval(f.attrs["pickle"])
            attr_dict = IgnoreGeoBenchUnpickler(io.BytesIO(raw)).load()

            for modality_name, modality_band in self.modalities.items():
                # Map each requested band to its actual HDF5 key (matched by prefix).
                mapping = {band: [key for key in f.keys() if key.startswith(band)] for band in modality_band}
                mapping = {k: v[0] for k, v in mapping.items()}
                modality_data = [f[mapping[band]] for band in modality_band]
                modality_data = np.stack(modality_data, axis=0)
                if modality_name == 'neon':  # stored HWC -> CHW
                    modality_data = modality_data[0].transpose((2, 0, 1))
                elif modality_name == 'dsm-dtm':  # drop trailing singleton channel
                    modality_data = modality_data[..., 0]
                elif modality_name == 's1':  # append VV/VH ratio as a third band
                    ratio_band = modality_data[0] / (modality_data[1] + 1e-6)
                    ratio_band = np.clip(ratio_band, a_min=-1e4, a_max=1e4)
                    modality_data = np.stack([modality_data[0], modality_data[1], ratio_band], axis=0)
                out[modality_name] = torch.from_numpy(modality_data).float()

            if 'label' not in self.modalities.keys():
                label = attr_dict['label']
                out['label'] = torch.tensor(label).long()
            else:
                out['label'] = out['label'].squeeze().long()

            if self.norm is not None:
                norm = self.norm
                if 'lidar' in self.modalities and 'lidar' in norm.keys():
                    mean, std = norm['lidar']
                    mean = mean.mean()
                    std = std.mean()
                    out['lidar'] = (out['lidar'] - mean) / std
                    norm.drop('lidar')

                out = apply_norm(norm, out)

        return out

