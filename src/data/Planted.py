import json
import os
from datetime import datetime

import numpy as np
import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from torch.utils.data import Dataset, IterableDataset

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

def milliseconds_to_datetime(milliseconds):
    """Converts milliseconds since the Unix epoch to a datetime object.

    Args:
        milliseconds (int): The number of milliseconds since the Unix epoch.

    Returns:
        datetime.datetime: A datetime object representing the corresponding date and time.
    """
    return [datetime.fromtimestamp(int(m) / 1000).timetuple().tm_yday for m in milliseconds]

class Planted(IterableDataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        classes: list = [],
        norm_path = None,
        temporal_dropout = 0,
        density_sampling: bool = False,
        ):
        """
        Initializes the dataset using MultiTFRecordDataset as an iterable.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torch module): transform to apply to the data
            split (str): split to use (train, val, test)
            classes (list): name of the different classes
            norm_path (str): path to normalization values
            temporal_dropout (float): probability of dropping a temporal observation
            density_sampling (bool): not supported, included for compatibility
        """
        self.path = path
        self.transform = transform
        self.modalities = modalities
        self.split = split if not split =="val" else "validation"
        self.temporal_dropout = temporal_dropout
        self.classes = classes
        assert density_sampling is False, "Density sampling is not supported for Planted dataset."
        assert len(classes) > 0, "Classes list must be provided for species name mapping."

        # Try to load available modalities from features.json
        self.available_modalities = self._get_available_modalities()

        # TFRecord patterns - format: planted_public-{split}.tfrecord-{shard}-of-{total}
        tfrecord_pattern, index_pattern = self._get_pattern_for_split(path, self.split)

        # If no modalities specified and available_modalities is not empty, use all available modalities
        if not self.modalities and self.available_modalities:
            self.modalities = self.available_modalities
            print(f"Using all available modalities: {self.modalities}")

        # Define splits for MultiTFRecordDataset (assuming equal weight for all shards)
        num_shards = self._get_num_shards(path, self.split)
        splits = {}
        for i in range(num_shards):
            splits[str(i).zfill(5)] = 1.0 / num_shards

        # Define the description for each field in the TFRecord
        self.tf_description, self.tf_shapes = self._get_tfrecord_description(modalities)

        # Create the MultiTFRecordDataset
        self.dataset = MultiTFRecordDataset(
            tfrecord_pattern,
            index_pattern,
            splits,
            self.tf_description,
            transform=self._transform_data,
            shuffle_queue_size=1024,
        )

        self.collate_fn = collate_fn
        self.norm = None
        if norm_path is not None:
            missing = [m for m in self.modalities if not os.path.exists(os.path.join(norm_path, f"NORM_{m}_patch.json"))]
            if missing:
                for m in missing:
                    self._compute_norm_vals_from_sample(m, norm_path)

            self.norm = load_norm(norm_path, self.modalities)

    def _get_available_modalities(self):
        """
        Get the list of available modalities from features.json.

        Returns:
            list: List of available modality names
        """
        features_path = os.path.join(os.path.dirname(self.path), "features.json")
        available_modalities = []

        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    features_data = json.load(f)

                features_dict = features_data.get("featuresDict", {}).get("features", {})

                # Get all features that don't end with _mask or _timestamps
                for feature_name in features_dict.keys():
                    if not (feature_name.endswith("_mask") or
                            feature_name.endswith("_timestamps") or
                            feature_name.endswith("_dates")):
                        # Check if it's a tensor (not a scalar)
                        feature_info = features_dict[feature_name]
                        if feature_info.get("pythonClassName", "").endswith("tensor_feature.Tensor"):
                            available_modalities.append(feature_name)
            except Exception as e:
                print(f"Warning: Could not load available modalities from features.json: {e}")

        return available_modalities

    def _get_num_shards(self, path, split):
        """
        Get the number of TFRecord shards available for a given split.
        """
        import glob
        import re

        # Match pattern like: planted_public-test.tfrecord-00000-of-00256
        pattern = os.path.join(path, f"planted_public-{split}.tfrecord-*-of-*")
        files = glob.glob(pattern)

        # Get the total number of shards from the last file name
        # Format: planted_public-test.tfrecord-00000-of-00256
        if files:
            last_file = os.path.basename(files[0])
            match = re.search(r'-of-(\d+)', last_file)
            if match:
                return int(match.group(1))

        return len(files)

    def _get_pattern_for_split(self, path, split):
        """
        Get the TFRecord pattern for a specific split.
        Args:
            split (str): The split to get the pattern for (train, validation, test)
        Returns:
            str: The TFRecord pattern for the specified split
        """
        nb_shard = self._get_num_shards(path, split)
        tfrecord_pattern = os.path.join(path, "planted_public-{}.tfrecord-{{}}-of-{}".format(
            split, str(nb_shard).zfill(5)))
        index_pattern = tfrecord_pattern+ ".idx"
        return tfrecord_pattern, index_pattern

    def _get_tfrecord_description(self, modalities):
        """
        Create description dictionary for the TFRecord format based on features.json.

        This method reads the features.json file to determine the correct data types
        for each field in the TFRecord.
        """
        # Default description with required fields
        description = {
            "species": "byte",  # Species name (string encoded as bytes)
            "id": "byte",       # Sample ID
        }
        shape_dict = {}

        # Try to load features.json to get the actual data types
        features_path = os.path.join(os.path.dirname(self.path), "features.json")
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    features_data = json.load(f)

                features_dict = features_data.get("featuresDict", {}).get("features", {})

                # Map tensorflow_datasets dtype to tfrecord format
                dtype_mapping = {
                    "float32": "float",
                    "float64": "float",
                    "int32": "int",
                    "int64": "int",
                    "string": "byte",
                    "uint8": "byte",
                }

                # Add descriptions for requested modalities based on features.json
                for modality in modalities:
                    # Check if modality exists in features.json
                    if modality in features_dict:
                        modality_dtype = features_dict[modality]["tensor"]["dtype"]
                        description[modality] = dtype_mapping.get(modality_dtype, "byte")
                        shape = features_dict[modality]["tensor"]["shape"]["dimensions"]
                        shape_dict[modality] = [int(dim) for dim in shape]  # Convert to int
                    else:
                        raise ValueError(f"Modality '{modality}' not found in features.json")

                    # Mask field
                    mask_field = f"{modality}_mask"
                    if mask_field in features_dict:
                        description[mask_field] = "int"  # Masks are typically int data
                        shape = features_dict[mask_field]["tensor"]["shape"]["dimensions"]
                        shape_dict[mask_field] = [int(dim) for dim in shape]  # Convert to int
                    else:
                        pass # modality masks are not always present, so we can skip adding it

                    # Timestamps/dates field
                    dates_field = f"{modality}_timestamps"
                    if dates_field in features_dict:
                        dates_dtype = features_dict[dates_field].get("tensor", {}).get("dtype", "float32")
                        description[dates_field] = dtype_mapping.get(dates_dtype, "int")
                        shape = features_dict[dates_field]["tensor"]["shape"]["dimensions"]
                        shape_dict[dates_field] = [int(dim) for dim in shape]  # Convert to int
                    else:
                        pass # dates field is not always present, so we can skip adding it

            except Exception as e:
                raise FileNotFoundError(f"Warning: Could not load features.json: {e}. ")
        else:
            raise FileNotFoundError(f"features.json not found at {features_path}")

        return description, shape_dict

    def _compute_norm_vals_from_sample(self, modality, folder):
        """
        Compute normalization values for a given modality using a small sample.

        Args:
            modality (str): The modality to compute normalization values for
            folder (str): The folder to save the normalization values to

        Returns:
            dict: The normalization values
        """
        means = []
        stds = []
        medians = []

        # Get pattern for the tfrecord files
        tfrecord_pattern, index_pattern = self._get_pattern_for_split(self.path, self.split)

        # Create a temporary dataset just for computing norm values
        splits = {str(i).zfill(5): 1.0 for i in range(100)}  # Use the first 10 shards
        sample_dataset = MultiTFRecordDataset(
            tfrecord_pattern,
            index_pattern,
            splits,
            self.tf_description,
            transform=self._extract_modality_data(modality)
        )

        # Process a few samples (maximum 10000)
        sample_count = 0
        for data in sample_dataset:
            if sample_count >= 10000:
                break

            if len(data.shape) == 4:  # (T, C, H, W)
                data = data.permute(1, 0, 2, 3)  # (C, T, H, W)
                means.append(data.to(torch.float32).mean(dim=(1, 2, 3)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2, 3)).numpy())
                # compute per-channel median by moving channel to dim 0 and flattening the rest
                channels = data.shape[0]
                medians.append(data.reshape(channels, -1).median(dim=1).values.numpy())
            else:  # (C, H, W)
                means.append(data.to(torch.float32).mean(dim=(1, 2)).numpy())
                stds.append(data.to(torch.float32).std(dim=(1, 2)).numpy())
                # flatten spatial dims and compute median per channel
                tmp = data.flatten(1, 2)
                medians.append(tmp.median(dim=1).values.numpy())

            sample_count += 1

        if len(means) == 0:
            print(f"Warning: No samples found to compute normalization for {modality}")
            return None

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)
        median = np.median(np.stack(medians), axis=0).astype(float)

        norm_vals = dict(mean=list(mean), std=list(std), median=list(median))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(modality)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))

        return norm_vals

    def _extract_modality_data(self, target_modality):
        """
        Helper function to extract only data for a specific modality.

        Args:
            target_modality (str): The modality to extract

        Returns:
            function: A transform function that extracts only the specified modality
        """
        def transform(data):
            modality_data = data[target_modality] #already a numpy array
            tensor_data = torch.tensor(modality_data).view(
                *self.tf_shapes[target_modality]  # Use the shape from features.json
            ) # T,H,W,C
            tensor_data = tensor_data.permute(0, 3, 1, 2)  # Convert to (T, C, H, W)

            # Process special cases for alos and s1
            if target_modality == "alos" or target_modality == "s1":
                ratio_band = tensor_data[:, 0, :, :] / (tensor_data[:, 1, :, :] + 1e-10)
                ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4).unsqueeze(1)  # Add channel dimension
                tensor_data = torch.cat((tensor_data[:, :2, :, :], ratio_band), dim=1)

            return tensor_data
        return transform

    def _transform_data(self, data):
        """
        Transform the raw data from TFRecord format.
        """
        output = {}

        # Process label from species field
        species_name = data.get("species", None)

        # Convert species string to class index
        if species_name is None:
            # Default to first class if species is not available
            label_idx = 0
        else:
            # Convert bytes to string if needed
            if isinstance(species_name, bytes):
                species_name = species_name.decode('utf-8')

            # Find the index of the species name in the classes list
            if species_name in self.classes:
                label_idx = self.classes.index(species_name)
            else:
                # Default to first class if species name not in classes list
                label_idx = len(self.classes)

        output["label"] = torch.LongTensor([label_idx])

        # Process each modality
        for modality in self.modalities:
            # Load and convert the byte arrays to tensors
            modality_data = data[modality]
            output[modality] = torch.tensor(modality_data).view(*self.tf_shapes[modality])  # (T, H, W, C)
            output[modality] = output[modality].permute(0, 3, 1, 2)  # Convert to (T, C, H, W)

            # Process special cases for alos and s1
            if modality == "alos" or modality == "s1":
                #remove angle and instead use ratio band HH/HV
                ratio_band = output[modality][:, 0, :, :] / (output[modality][:, 1, :, :] + 1e-10)
                ratio_band = torch.clamp(ratio_band, max=1e4, min=-1e4).unsqueeze(1)
                output[modality] = torch.cat((output[modality][:, :2, :, :], ratio_band), dim=1)

            # Load and process masks
            mask_data = data[f"{modality}_mask"]
            output[f"{modality}_mask"] = torch.tensor(mask_data).view(*self.tf_shapes[modality]).permute(0, 3, 1, 2)

            # Process dates
            dates = data[f"{modality}_timestamps"]
            output[f"{modality}_dates"] = torch.tensor(milliseconds_to_datetime(dates))

            # Apply temporal dropout if needed
            N = len(output[f"{modality}_dates"])
            if self.split == "train" and self.temporal_dropout > 0:
                if modality == "alos":
                    random_indices = torch.randperm(N)[:int(N * 0.9)]
                else:
                    random_indices = torch.randperm(N)[:int(N * (1 - self.temporal_dropout))]

                output[modality] = output[modality][random_indices]
                output[f"{modality}_mask"] = output[f"{modality}_mask"][random_indices]
                output[f"{modality}_dates"] = output[f"{modality}_dates"][random_indices]

        # Apply normalization if available
        output = apply_norm(self.norm, output)

        return self.transform(output, dataset_name='planted')

    def __iter__(self):
        """
        Returns an iterator over the dataset.

        Returns:
            iterator: An iterator that yields transformed samples
        """
        return iter(self.dataset)

    def __len__(self):
        """
        Returns the length of the dataset based on dataset_info.json.

        Returns:
            int: The total number of samples in this split
        """
        # First try to get the length from dataset_info.json
        try:
            dataset_info_path = os.path.join(os.path.dirname(self.path), "dataset_info.json")
            if os.path.exists(dataset_info_path):
                with open(dataset_info_path, 'r') as f:
                    dataset_info = json.load(f)

                for split_info in dataset_info.get("splits", []):
                    if split_info.get("name") == self.split:
                        shard_lengths = split_info.get("shardLengths", [])
                        if shard_lengths:
                            return sum(int(length) for length in shard_lengths)
        except Exception as e:
            print(f"Warning: Could not load dataset length from dataset_info.json: {e}")

        # Fallback: count files in the split or use a rough estimate
        try:
            num_shards = self._get_num_shards(self.path, self.split)
            if self.split == "train":
                # Average shard size from dataset_info.json is ~1750 samples
                return num_shards * 1750
            elif self.split == "validation" or self.split == "test":
                # Average shard size from dataset_info.json is ~900 samples
                return num_shards * 900
            else:
                return num_shards * 1000  # Generic estimate
        except:
            # Last resort
            return 1000000  # Just return a large number
