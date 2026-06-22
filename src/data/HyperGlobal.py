import os
import random

import rasterio
import scipy.io as scio
import torch
from torch.utils import data

from .utils import apply_norm, load_norm

# EO-1 Hyperion band reference:
# https://developers.google.com/earth-engine/datasets/catalog/EO1_HYPERION#bands


def read(file):
    with rasterio.open(file) as src:
        return src.read(1)

def collate_fn(batch):
    """Collate function for Hyperion batches (single 'hyperspectral' modality).
    - Stacks 3D tensors [C,H,W].
    - Pads and stacks 4D tensors [T,C,H,W] along T to the max length in batch.
    - Aggregates 'name' as a list.
    """
    keys = set()
    for x in batch:
        keys.update(x.keys())
    output = {}

    for key in ["hyperspectral"]:
        if key in keys:
            elems = [x[key] for x in batch if key in x]
            if len(elems) == 0:
                continue
            if elems[0].dim() == 4:
                # [T, C, H, W] -> pad T
                max_t = max(t.size(0) for t in elems)
                stacked = torch.stack([
                    torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 0, 0, max_t - t.size(0)))
                    for t in elems
                ], dim=0)
            else:
                stacked = torch.stack(elems, dim=0)
            output[key] = stacked

    if 'name' in keys:
        output['name'] = [x.get('name') for x in batch]

    # Stack any other common tensor keys present in all samples
    other_keys = [k for k in keys if k not in {"hyperspectral", "name"}]
    for key in other_keys:
        output[key] = torch.stack([x[key] for x in batch])

    return output

class HyperionDataset(data.Dataset):
    """EO-1 Hyperion hyperspectral tiles stored as MATLAB ``.mat`` files.

    Args:
        path: root directory searched recursively for ``.mat`` files.
        norm_path: directory holding per-modality normalisation statistics.
        transform: callable applied to each sample dict.
        modalities: sensor codes to load; filenames must start with one of them.
    """

    def __init__(self, path, norm_path, transform=None, modalities=['EO1']):
        self.root = path
        self.modalities = modalities

        self.ids = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.endswith('.mat') and any(filename.startswith(mod) for mod in self.modalities):
                    self.ids.append(os.path.join(dirpath, filename))
        self.transform = transform

        self.collate_fn = collate_fn
        self.norm = load_norm(norm_path, self.modalities, self)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        output = {}

        img_path = self.ids[index]
        img_name = os.path.basename(img_path)

        try:
            data = scio.loadmat(img_path)
            tmp = torch.FloatTensor(data['img'])
            modality = [mod for mod in self.modalities if img_name.startswith(mod)][0]
            output[modality] = tmp
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Skip corrupted file by picking another random one
            new_index = random.randint(0, len(self.ids) - 1)
            return self.__getitem__(new_index)
        output['name'] = os.path.basename(img_path)


        output = apply_norm(self.norm, output)

        return self.transform(output, dataset_name='hyperglobal')
