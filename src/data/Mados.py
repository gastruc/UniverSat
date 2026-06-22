import os
import warnings
from glob import glob

import numpy as np
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
from torch.utils.data import Dataset
from .utils import load_norm, apply_norm


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

class MADOS(Dataset):
    def __init__(
        self,
        split: str,
        path: str,
        transform = None,
        norm_path: str = None,
    ):
        """Initialize the MADOS dataset.
        Link: https://marine-pollution.github.io/index.html

        Args:
            split (str): split of the dataset (train, val, test).
            path (str): root path of the dataset.
            transform (callable, optional): transform applied to each sample.
            norm_path (str, optional): directory holding normalisation statistics.
        """
        super(MADOS, self).__init__()

        self.root_path = path
        self.split = split
        self.transform = transform
        self.norm_path = norm_path

        self.collate_fn = collate_fn

        self.ROIs_split = np.genfromtxt(os.path.join(self.root_path, 'splits', f'{split}_X.txt'), dtype='str')

        self.image_list = []
        self.target_list = []

        self.tiles = sorted(glob(os.path.join(self.root_path, '*')))

        for tile in self.tiles:
            splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]

            for crop in splits:
                crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]

                if crop_name in self.ROIs_split:
                    all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_' + crop))
                    all_bands = sorted(all_bands, key=self.get_band)

                    self.image_list.append(all_bands)

                    cl_path = os.path.join(tile, '10', os.path.basename(tile) + '_L2R_cl_' + crop)
                    self.target_list.append(cl_path)

        self.norm = load_norm(norm_path, ["s2"], self)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        all_bands = self.image_list[index]
        current_image = []
        for band in all_bands:
            upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
            with rasterio.open(band, mode='r') as src:
                this_band = src.read(1,
                                     out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)),
                                     resampling=rasterio.enums.Resampling.nearest
                                     )
                this_band = torch.from_numpy(this_band)
                current_image.append(this_band)

        image = torch.stack(current_image)[1:]  # (C, H, W)  drop the first band which is coastal aerosol
        invalid_mask = torch.isnan(image)            # (C, H, W) per-channel
        invalid_2d = invalid_mask.any(dim=0)         # (H, W)  any-channel invalid
        image[invalid_mask] = 0

        with rasterio.open(self.target_list[index], mode='r') as src:
            target = src.read(1)
        target = torch.from_numpy(target.astype(np.int64))
        target = target - 1

        target[invalid_2d] = -1

        output = {
            's2': image,
            'label': target,
        }

        output = apply_norm(self.norm, output)

        if invalid_mask.any():
            output['s2'][invalid_mask] = 0.0

        if self.transform is not None:
            output = self.transform(output)

        return output

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

