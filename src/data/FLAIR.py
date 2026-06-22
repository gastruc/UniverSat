import glob
import json
import os
import sys
from datetime import datetime
from random import shuffle

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
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
    for key in ["s2flair", "s1-asc", "s1-des", "s1flair"]:
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
    if 's2_cloud_density' in keys:
        idx = [x['s2_cloud_density'] for x in batch]
        max_size_0 = max(tensor.size(0) for tensor in idx)
        stacked_tensor = torch.stack([
                torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                for tensor in idx
            ], dim=0)
        output['s2_cloud_density'] = stacked_tensor.long()
        keys.remove('s2_cloud_density')
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def convert_date(date:str):
    """Parse a JSON date dict into day-of-year offsets."""
    json_date = json.loads(date)
    date = json_date.values()
    to_datetime = lambda x : datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
    date = list(map(to_datetime, date))
    date = list(map(lambda x : (x - datetime(x.year, 1, 1)).days + 1, date))
    return date

class FLAIR(Dataset):

    official_split_fold1 = {
    "train": ["D005", "D006", "D007", "D008", "D009", "D010", "D011", "D013", "D016", "D017", "D018", "D020", "D021", "D023", "D024047", "D025039", "D030", "D032", "D033", "D034", "D035", "D037", "D038", "D040", "D041", "D044", "D045", "D046", "D049", "D051", "D052", "D054057", "D055", "D056", "D059062", "D060", "D063", "D065", "D070", "D072", "D074", "D078", "D080", "D081", "D086", "D091"],
    "val": ["D004", "D014", "D029", "D031", "D058", "D066", "D067", "D077"],
    "test": ["D012", "D015", "D022", "D026", "D036", "D061", "D064", "D068", "D069", "D071", "D073", "D075", "D076", "D083", "D084", "D085"]
    }

    flair_name_to_modality = {
        "AERIAL_RGBI": "aerialflair",
        "SENTINEL1-ASC_TS": "s1-asc",
        "SENTINEL1-DESC_TS": "s1-des",
        "SENTINEL2_TS": "s2flair",
        "DEM_ELEV": "lidar",
        "SPOT_RGBI": "spotRGBN",
        "DEM_ELEV": "dem",
        "AERIAL_LABEL-COSIA": "label"
    }

    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        num_classes: int = 18,
        norm_path = None,
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torchvision.transforms): transform to apply to the data
            split (str): split to use (train, val, test)
            num_classes (int): number of classes
            norm_path (str): directory holding per-modality normalisation statistics
        """
        self.path = path
        self.transform = transform
        self.modalities = modalities
        self.num_classes = num_classes
        self.split = split
        self.collate_fn = collate_fn
        self.norm = None

        self.ids, self.data, self.s2_dates, self.s1_asc_dates, self.s1_des_dates = self.load_metadata(split)

        self.norm = load_norm(norm_path, self.modalities, self)

    def get_folder_modality(self, folder):
        """
        Returns the modality of the folder.
        Args:
            folder (str): path to the folder
        Returns:
            str: modality of the folder
        """
        for modality in self.flair_name_to_modality.keys():
            if modality in folder:
                return self.flair_name_to_modality[modality]
        return None

    def get_data_path (self, split):
        """ Returns a list of dixtionaries with the paths to the data files fore each modality of a training sample.
        Args:
            split (str): split to use (train, val, test)
        """
        domains = FLAIR.official_split_fold1[split]

        id_to_path_dict = {}
        modalities = self.modalities + ["label"]
        if 's1flair' in modalities:
            modalities.remove('s1flair')
            modalities += ['s1-asc', 's1-des']

        for domain in domains:
            folders = glob.glob(os.path.join(self.path, domain + "*"))
            for folder in folders:
                modality  = self.get_folder_modality(folder)
                if modality in modalities:
                    # Get all files in the folder
                    files = glob.glob(os.path.join(folder, "*/*.tif"), recursive=True)
                    for file in files:
                        id = os.path.basename(file).split(".")[0].split("_")
                        id = "_".join(id[:1] + id[-2:])
                        if id not in id_to_path_dict:
                            id_to_path_dict[id] = {}
                        id_to_path_dict[id][modality] = file

        #assert all modalities in all samples
        for id, paths in id_to_path_dict.items():
            for modality in modalities:
                if modality not in paths:
                    raise ValueError(f"Missing modality {modality} for sample {id}")

        list_of_ids = list(id_to_path_dict.keys())
        shuffle(list_of_ids)

        return list_of_ids, id_to_path_dict

    def load_metadata(self, split, memory_map=True):
        # Define file paths for memory-mapped storage
        ids_file = os.path.join(self.path, f"ids_{split}.parquet")
        data_file = os.path.join(self.path, f"data_{split}.parquet")
        s2_dates_file = os.path.join(self.path, f"s2_dates_{split}.parquet")
        s1_asc_dates_file = os.path.join(self.path, f"s1_asc_dates_{split}.parquet")
        s1_des_dates_file = os.path.join(self.path, f"s1_des_dates_{split}.parquet")

        # Load or compute ids and data
        if os.path.exists(ids_file) and os.path.exists(data_file):
            # Load ids
            ids_table = pq.read_table(ids_file, memory_map=memory_map)
            ids = ids_table.to_pandas()['id'].tolist()
            # Load data
            data_table = pq.read_table(data_file, memory_map=memory_map)
            df = data_table.to_pandas()
            data = {}
            for row in df.itertuples(index=False):
                id_val, mod, path = row.id, row.modality, row.path
                if id_val not in data:
                    data[id_val] = {}
                data[id_val][mod] = path
        else:
            ids, data = self.get_data_path(split)
            # Save ids
            ids_df = pd.DataFrame({'id': ids})
            pq.write_table(pa.Table.from_pandas(ids_df), ids_file)
            # Save data
            df_data = pd.DataFrame([(id_val, mod, path) for id_val, mods in data.items() for mod, path in mods.items()], columns=['id', 'modality', 'path'])
            pq.write_table(pa.Table.from_pandas(df_data), data_file)

        # Load or compute dates
        s2_dates = None
        s1_asc_dates = None
        s1_des_dates = None
        if 's2flair' in self.modalities:
            if os.path.exists(s2_dates_file):
                s2_dates = pq.read_table(s2_dates_file, memory_map=memory_map)
            else:
                s2_dates_full = gpd.read_file(os.path.join(self.path, "GLOBAL_ALL_MTD/GLOBAL_SENTINEL2_MTD_DATES.gpkg"), columns=['patch_id', 'acquisition_dates']).drop(columns='geometry')
                s2_dates_df = s2_dates_full[s2_dates_full['patch_id'].isin(ids)].reset_index(drop=True)
                missing_ids = set(ids) - set(s2_dates_df['patch_id'].tolist())
                if missing_ids:
                    print(f"Warning: {len(missing_ids)} s2 date entries missing for split '{split}' (kept {len(s2_dates_df)} records)")
                pq.write_table(pa.Table.from_pandas(s2_dates_df), s2_dates_file)
                s2_dates = pq.read_table(s2_dates_file, memory_map=memory_map)

        if 's1flair' in self.modalities or 's1-asc' in self.modalities:
            if os.path.exists(s1_asc_dates_file):
                s1_asc_dates = pq.read_table(s1_asc_dates_file, memory_map=memory_map)
            else:
                s1_asc_dates_full = gpd.read_file(os.path.join(self.path, "GLOBAL_ALL_MTD/GLOBAL_SENTINEL1-ASC_MTD_DATES.gpkg"), columns=['patch_id', 'acquisition_dates']).drop(columns='geometry')
                s1_asc_dates_df = s1_asc_dates_full[s1_asc_dates_full['patch_id'].isin(ids)].reset_index(drop=True)
                pq.write_table(pa.Table.from_pandas(s1_asc_dates_df), s1_asc_dates_file)
                s1_asc_dates = pq.read_table(s1_asc_dates_file, memory_map=memory_map)

        if 's1flair' in self.modalities or 's1-des' in self.modalities:
            if os.path.exists(s1_des_dates_file):
                s1_des_dates = pq.read_table(s1_des_dates_file, memory_map=memory_map)
            else:
                s1_des_dates_full = gpd.read_file(os.path.join(self.path, "GLOBAL_ALL_MTD/GLOBAL_SENTINEL1-DESC_MTD_DATES.gpkg"), columns=['patch_id', 'acquisition_dates']).drop(columns='geometry')
                s1_des_dates_df = s1_des_dates_full[s1_des_dates_full['patch_id'].isin(ids)].reset_index(drop=True)
                pq.write_table(pa.Table.from_pandas(s1_des_dates_df), s1_des_dates_file)
                s1_des_dates = pq.read_table(s1_des_dates_file, memory_map=memory_map)

        return ids, data, s2_dates, s1_asc_dates, s1_des_dates

    def __getitem__(self, i):
        max_try=3
        for attempt in range(max_try):
            try:
                return self._getitem(i+attempt)
            except Exception as e:
                print(f"Error loading sample {i} ({self.ids[i]}): {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to load sample {i} ({self.ids[i]}) after {max_try} attempts")


    def _getitem(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        id = self.ids[i]
        line = self.data[id]
        output = {'name': id}

        with rasterio.open(line['label']) as f:
            labels = f.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            output["label"] = torch.LongTensor(labels)

        if 'aerialflair' in self.modalities:
            with rasterio.open(line['aerialflair']) as f:
                output["aerialflair"] = torch.FloatTensor(f.read())

        if 'spotRGBN' in self.modalities:
            with rasterio.open(line['spotRGBN']) as f:
                output["spotRGBN"] = torch.FloatTensor(f.read())

        if 's2flair' in self.modalities:
            with rasterio.open(line['s2flair']) as f:
                arr = torch.FloatTensor(f.read())
                output["s2flair"] = arr.reshape((arr.shape[0]//10, 10, *arr.shape[1:]))
            #dates
            filtered = self.s2_dates.filter(pa.compute.field('patch_id') == id)
            assert filtered.num_rows == 1, f"Patch {id} not found in s2_dates"
            s2_dates = filtered['acquisition_dates'][0].as_py()
            output["s2flair_dates"] = torch.FloatTensor(convert_date(s2_dates))

        if 's1-asc' in self.modalities:
            with rasterio.open(line['s1-asc']) as f:
                arr = torch.FloatTensor(f.read())
                output["s1_asc"] = arr.reshape((arr.shape[0]//2, 2, *arr.shape[1:]))
            #dates
            filtered = self.s1_asc_dates.filter(pa.compute.field('patch_id') == id)
            assert filtered.num_rows == 1, f"Patch {id} not found in s1_asc_dates"
            s1_asc_dates = filtered['acquisition_dates'][0].as_py()
            output["s1_asc_dates"] = torch.FloatTensor(convert_date(s1_asc_dates))

        if 's1-des' in self.modalities:
            with rasterio.open(line['s1-des']) as f:
                arr = torch.FloatTensor(f.read())
                output["s1_des"] = arr.reshape((arr.shape[0]//2, 2, *arr.shape[1:]))
            #dates
            filtered = self.s1_des_dates.filter(pa.compute.field('patch_id') == id)
            assert filtered.num_rows == 1, f"Patch {id} not found in s1_des_dates"
            s1_des_dates = filtered['acquisition_dates'][0].as_py()
            output["s1_des_dates"] = torch.FloatTensor(convert_date(s1_des_dates))

        if 's1flair' in self.modalities:
            with rasterio.open(line['s1-asc']) as f:
                s1_asc = torch.FloatTensor(f.read())
                s1_asc = s1_asc.reshape((s1_asc.shape[0]//2, 2, *s1_asc.shape[1:]))
            with rasterio.open(line['s1-des']) as f:
                s1_des = torch.FloatTensor(f.read())
                s1_des = s1_des.reshape((s1_des.shape[0]//2, 2, *s1_des.shape[1:]))
            # dates
            filtered_asc = self.s1_asc_dates.filter(pa.compute.field('patch_id') == id)
            assert filtered_asc.num_rows == 1, f"Patch {id} not found in s1_dates"
            s1_asc_dates_str = filtered_asc['acquisition_dates'][0].as_py()
            s1_asc_dates = torch.FloatTensor(convert_date(s1_asc_dates_str))
            filtered_des = self.s1_des_dates.filter(pa.compute.field('patch_id') == id)
            assert filtered_des.num_rows == 1, f"Patch {id} not found in s1_dates"
            s1_des_dates_str = filtered_des['acquisition_dates'][0].as_py()
            s1_des_dates = torch.FloatTensor(convert_date(s1_des_dates_str))
            #merge
            output["s1flair"] = torch.cat([s1_asc, s1_des], dim=0)
            output["s1flair_dates"] = torch.cat([s1_asc_dates, s1_des_dates], dim=0)

        if 'dem' in self.modalities:
            with rasterio.open(line['dem']) as f:
                output["dem"] = torch.FloatTensor(f.read())

        for modality in ['s2flair', 's1flair', 's1-asc', 's1-des']:
            if modality in self.modalities:
                #rescale 10x10 to 8x8
                output[modality] = torch.nn.functional.interpolate(output[modality], size=(8, 8), mode='bilinear', align_corners=False)

        for modality in ['s1flair', 's1-asc', 's1-des']:
            if modality in self.modalities:
                #add ratio band
                ratio_band = output[modality][:, 0, :, :] / (output[modality][:, 1, :, :] + 1e-10)
                ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4).unsqueeze(1)
                output[modality] = torch.cat((output[modality][:, :2, :, :], ratio_band), dim=1)

        if 's2flair' in output:
            s2 = output['s2flair'] #TCHW
            b2 = s2[:, 0, :, :]
            b3 = s2[:, 1, :, :]
            b4 = s2[:, 2, :, :]
            b8 = s2[:, 5, :, :]
            cloud_mask = ((b2 + b3 + b4) / 3 > 3200) & (b8 > 2500) #THW
            cloud_density = cloud_mask.flatten(-2).float().mean(-1)  #T
            output['s2_cloud_density'] = cloud_density

        if self.norm is not None:
            norm = self.norm.copy()
            if 'dem' in self.modalities:
                dsm = output["dem"][0]
                dtm = output["dem"][1]
                ndem = (dsm - dtm) / 10
                dsm = (dsm-dsm.mean()) / 10
                output["dem"] = torch.stack([dsm, ndem], dim=0)
                norm.pop('dem', None)

            output  = apply_norm(norm, output)

        return self.transform(output, dataset_name='flair')

    def __len__(self):
        return len(self.data)

