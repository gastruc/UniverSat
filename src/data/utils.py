import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm


def subset_dict_by_filename(files_to_subset, dictionary):
    return {file : dictionary[file] for file in files_to_subset}

def filter_labels_by_threshold(labels_dict, area_threshold = 0.07):
    """
    Parameters
    ----------
    labels_dict: dict, {filename1: [(label, area)],
                        filename2: [(label, area), (label, area)],
                        ...
                        filenameN: [(label, area), (label, area)]}
    area_threshold: float

    Returns
    -------
    filtered: dict, {filename1: [label],
                     filename2: [label, label],
                     ...
                     filenameN: [label, label]}
    """
    filtered = {}

    for img in labels_dict:
        for lbl, area in labels_dict[img]:
            # if area greater than threshold we keep the label
            if area > area_threshold:
                # init the list of labels for the image
                if img not in filtered:
                    filtered[img] = []
                # add only the label, since we won't use area information further
                filtered[img].append(lbl)

    return filtered

def filter_labels_by_max(labels_dict):
    """
    Parameters
    ----------
    labels_dict: dict, {filename1: [(label, area)],
                        filename2: [(label, area), (label, area)],
                        ...
                        filenameN: [(label, area), (label, area)]}

    Returns
    -------
    filtered: dict, {filename1: [label],
                     filename2: [label, label],
                     ...
                     filenameN: [label, label]}
    """
    filtered = {}

    for img in labels_dict:
        # find the label with the maximum area
        max_area = 0
        max_label = None
        for lbl, area in labels_dict[img]:
            if area > max_area:
                max_area = area
                max_label = lbl
        # add the label with the maximum area to the filtered dict
        if img not in filtered:
            filtered[img] = []
        filtered[img].append(max_label)

    return filtered

def _dist_is_initialized():
    return dist.is_available() and dist.is_initialized()

def _dist_rank():
    return dist.get_rank() if _dist_is_initialized() else 0

def _dist_world_size():
    return dist.get_world_size() if _dist_is_initialized() else 1

def _dist_barrier():
    if _dist_is_initialized():
        dist.barrier()

class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks


    def _sample_block_size(self, scale, aspect_ratio_scale):
        _rand = torch.rand(1).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep

        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        p_size = self._sample_block_size(
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            if self.allow_overlap:
                acceptable_regions= None

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

def compute_norm_vals(dataset, norm_path, modalities, n_samples=10000):
    """Estimate per-channel mean/std for each modality over a random sample of
    the dataset and write them to ``NORM_<modality>_patch.json`` under
    ``norm_path``. Distributed-aware: work is split across ranks and reduced on
    rank 0. Returns the computed ``{modality: {"mean", "std"}}`` dict.
    """
    rank = _dist_rank()
    world_size = _dist_world_size()

    if rank == 0:
        os.makedirs(norm_path, exist_ok=True)
    _dist_barrier()

    dataset.norm = None

    if rank == 0:
        print(f"Computing normalization values for {modalities}...")

    stats = {m: None for m in modalities}

    n_samples = min(n_samples, len(dataset))

    rng = np.random.default_rng(0)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    rank_indices = indices[rank::world_size]

    for i in tqdm(rank_indices, desc=f"Computing Norm Stats (rank {rank})", disable=rank != 0):
        try:
            item = dataset[i]
        except Exception as e:
            if rank == 0:
                print(f"Error loading sample {i}: {e}")
            continue

        for mod in modalities:
            if mod not in item:
                continue

            data = item[mod]
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().double()
            else:
                data = torch.from_numpy(np.asarray(data)).double()

            if data.ndim == 3:        # (C, H, W)
                reduce_dims = (1, 2)
                C = data.shape[0]
            elif data.ndim == 4:      # (T, C, H, W)
                reduce_dims = (0, 2, 3)
                C = data.shape[1]
            else:
                reduce_dims = tuple(range(1, data.ndim))
                C = data.shape[0]

            s = data.sum(dim=reduce_dims)
            s2 = (data * data).sum(dim=reduce_dims)
            n = 1
            for d in reduce_dims:
                n *= data.shape[d]

            if stats[mod] is None:
                stats[mod] = {
                    "count": 0,
                    "sum": torch.zeros(C, dtype=torch.float64),
                    "sum_sq": torch.zeros(C, dtype=torch.float64),
                }
            stats[mod]["count"] += n
            stats[mod]["sum"] += s
            stats[mod]["sum_sq"] += s2

    if _dist_is_initialized():
        gathered_stats = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_stats, stats)
    else:
        gathered_stats = [stats]

    norm_vals = {}
    if rank == 0:
        for mod in modalities:
            total_count = 0
            total_sum = None
            total_sum_sq = None

            for rank_stats in gathered_stats:
                mod_stats = rank_stats.get(mod) if rank_stats is not None else None
                if mod_stats is None or mod_stats["count"] == 0:
                    continue

                if total_sum is None:
                    total_sum = mod_stats["sum"].clone()
                    total_sum_sq = mod_stats["sum_sq"].clone()
                else:
                    total_sum += mod_stats["sum"]
                    total_sum_sq += mod_stats["sum_sq"]
                total_count += mod_stats["count"]

            if total_count == 0:
                print(f"Warning: No data found for modality {mod}")
                continue

            mu = total_sum / total_count
            var = total_sum_sq / total_count - mu * mu
            var = var.clamp(min=0.0)
            std = var.sqrt()

            norm_vals[mod] = {"mean": mu.tolist(), "std": std.tolist()}

            file_path = os.path.join(norm_path, f"NORM_{mod}_patch.json")
            with open(file_path, "w") as f:
                json.dump(norm_vals[mod], f, indent=4)

    if _dist_is_initialized():
        norm_vals_list = [norm_vals]
        dist.broadcast_object_list(norm_vals_list, src=0)
        norm_vals = norm_vals_list[0]
        dist.barrier()

    return norm_vals

def load_norm(norm_path, modalities, dataset=None, max_samples=10000):
    """Load per-modality normalisation tensors from ``norm_path``.

    Missing statistics are computed first (via :func:`compute_norm_vals`) when a
    ``dataset`` is supplied. Returns ``{modality: (mean, std)}`` or ``None`` when
    ``norm_path`` is ``None``.
    """
    if norm_path is None:
        return None

    missing = [
        mod
        for mod in modalities
        if not os.path.exists(os.path.join(norm_path, f"NORM_{mod}_patch.json"))
    ]
    if _dist_is_initialized():
        missing_by_rank = [False for _ in range(_dist_world_size())]
        dist.all_gather_object(missing_by_rank, bool(missing))
        missing_anywhere = any(missing_by_rank)
    else:
        missing_anywhere = bool(missing)

    if missing_anywhere:
        if dataset is not None:
            compute_norm_vals(dataset, norm_path, modalities, n_samples=max_samples)
        elif _dist_rank() == 0:
            for mod in missing:
                print(f"Warning: Normalization file for {mod} not found and no dataset provided for computation.")

    norm = {}
    for mod in modalities:
        file_path = os.path.join(norm_path, f"NORM_{mod}_patch.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                vals = json.load(f)
            norm[mod] = (
                torch.tensor(vals["mean"]).float(),
                torch.tensor(vals["std"]).float()
            )
    return norm

def apply_norm(norm, output):
    """Standardise each modality tensor in ``output`` in place using the
    ``{modality: (mean, std)}`` mapping from :func:`load_norm`.
    """
    if norm is None:
        return output

    for modality, (mean, std) in norm.items():
        if modality in output:
            data = output[modality]
            if data.ndim == 3: # (C, H, W)
                m = mean.view(-1, 1, 1)
                s = std.view(-1, 1, 1)
            elif data.ndim == 4: # (T, C, H, W)
                m = mean.view(1, -1, 1, 1)
                s = std.view(1, -1, 1, 1)
            else:
                continue

            # Floor the std so a (near-)constant channel can't blow up the output.
            output[modality] = (data - m) / s.clamp_min(1e-6)

    return output
