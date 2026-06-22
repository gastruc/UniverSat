import math

import numpy as np
import torch

from .mask import Mask, MaskSpatial


class Masker(object):
    """Mask base class."""
    def __init__(self,
                 input_size=(6, 6)
                 ):
        """Initialize the Masker
        :param input_size: size of the input image in patches
        """
        self.patched_input_size = input_size

    def _get_modalities(self, batch_keys):
        """Get the modalities from the batch
        :param batch: batch of data
        :return: list of modalities
        """
        modalities = [mod for mod in batch_keys if not (mod.endswith('_dates') or mod.endswith('_mask') or mod.endswith('_density') or mod.endswith('cloud_density'))]
        modalities = [mod for mod in modalities if mod not in ['label', 'name', 'dataset', 'scale', 'modis', 'input_scale', 'latent_scale', 'output_scale']]
        return modalities

    def __call__(self, batch):
        """Generate mask for batch
        :param batch: batch of data
        :return: Mask dictionary (for each modality a dictionnary {S:mask, T: mask, C: mask})
        """

        raise NotImplementedError("Masker is an abstract class. Please implement the __call__ method.")

class RandomMasker(Masker):
    """Spatially coherent random Masker
    """
    def __init__(self,
                 input_size=(6, 6),
                 fraction_drop_time=0.5,
                 proba_monodate=0.0,
                 alpha_drop_channel=0.5,
                 fraction_drop_spatial=0.5,
                 crop_percentage_axes=0.8,
                 mask_C_with_blocks=False,
                 ):
        """
        :param mask_C_with_blocks: spectral masking strategy.
            - ``False`` (default): fully-random per-channel masking (legacy).
            - ``True``: keep a single contiguous block of visible channels.
            - ``int > 0``: keep that many contiguous blocks of visible
              channels (with randomly-sized gaps in between). Useful for
              spectral-aware SSL where neighbouring bands are correlated.
            Blocks are laid out on a circular view of the channel axis, so
            block boundaries can wrap past channel ``C-1``.
        """
        super(RandomMasker, self).__init__(input_size)
        self.fraction_drop_time = fraction_drop_time
        self.proba_monodate = proba_monodate
        self.alpha_drop_channel = alpha_drop_channel
        self.fraction_drop_spatial = fraction_drop_spatial
        self.crop_percentage_axes = crop_percentage_axes
        self.mask_C_with_blocks = mask_C_with_blocks

    def _sample_partition(self, B, n_parts, total, device='cpu'):
        """Sample non-negative integer partitions with fixed row sum (zeros allowed)."""
        if n_parts <= 0:
            raise ValueError(f"Invalid n_parts={n_parts}. Must be > 0")
        if total < 0:
            raise ValueError(f"Invalid total={total}. Must be >= 0")
        if total == 0:
            return torch.zeros((B, n_parts), dtype=torch.long, device=device)

        picks = torch.randint(0, n_parts, (B, total), device=device)
        counts = torch.zeros((B, n_parts), dtype=torch.long, device=device)
        counts.scatter_add_(1, picks, torch.ones_like(picks, dtype=torch.long))
        return counts

    def _mask_Spatial(self, B, patched_input_size, device='cpu'):
        assert isinstance(patched_input_size, tuple)
        # generating crop
        if self.crop_percentage_axes < 1.0:
            H, W = patched_input_size
            h_crop, w_crop = (math.floor(patched_input_size[0] * self.crop_percentage_axes), math.floor(patched_input_size[1] * self.crop_percentage_axes))
            start_x = torch.randint(0, H - h_crop + 1, (B,), device=device)
            start_y = torch.randint(0, W - w_crop + 1, (B,), device=device)
            # per-batch row and column offsets for the crop
            row_offsets = start_x.unsqueeze(1) + torch.arange(h_crop, device=device).unsqueeze(0)  # (B, h_crop)
            col_offsets = start_y.unsqueeze(1) + torch.arange(w_crop, device=device).unsqueeze(0)  # (B, w_crop)
            # compute linear indices (row * W + col) and flatten per sample
            all_indices = row_offsets.unsqueeze(2) * W + col_offsets.unsqueeze(1)  # (B, h_crop, w_crop)
            all_indices = all_indices.reshape(B, -1)

            patch_size = h_crop * w_crop
        else:
            patch_size = patched_input_size[0] * patched_input_size[1]
        #random masking
        value = torch.rand((B, patch_size), device=device)
        indices = torch.argsort(value, dim=1)
        cutoff = math.ceil(patch_size * (1-self.fraction_drop_spatial))
        mask_input, mask_output = torch.sort(indices[:, :cutoff], dim=1)[0], torch.sort(indices[:, cutoff:], dim=1)[0]
        if self.crop_percentage_axes < 1.0:
            # if we cropped the patch, we need to gather the indices from the original indices
            final_input, final_output = torch.gather(all_indices, dim=1, index=mask_input), torch.gather(all_indices, dim=1, index=mask_output)
            return final_input, final_output
        else:
            return mask_input, mask_output

    def _mask_Temporal(self, B, T, device='cpu'):
        if T == 1:
            return None, None

        if torch.rand(1).item() < self.proba_monodate:
            # all dates but one are masked
            cutoff = 1
        else: #bernoulli masking
            cutoff = 1 + np.random.binomial(T-1, 1 - self.fraction_drop_time)

        value = torch.rand((B, T), device=device)
        indices = torch.argsort(value, dim=1)
        return torch.sort(indices[:, :cutoff], dim=1)[0], torch.sort(indices[:, cutoff:], dim=1)[0]

    def _mask_Channel(self, B, C, device='cpu'):
        proba = 1/(1 + math.sqrt(self.alpha_drop_channel/C))
        cutoff = 1+ np.random.binomial(C-1, 1-proba)

        # Fully random per-channel masking (legacy behaviour).
        if self.mask_C_with_blocks is False:
            value = torch.rand((B, C), device=device)
            indices = torch.argsort(value, dim=1)
            return torch.sort(indices[:, :cutoff], dim=1)[0], torch.sort(indices[:, cutoff:], dim=1)[0]

        # Block-based spectral masking: keep ``n_blocks`` contiguous blocks of
        # visible channels (with random gaps in between), laid out on a
        # circular view of the channel axis.
        if self.mask_C_with_blocks is True:
            requested_blocks = 1
        elif isinstance(self.mask_C_with_blocks, int):
            if self.mask_C_with_blocks <= 0:
                raise ValueError(
                    f"Invalid mask_C_with_blocks={self.mask_C_with_blocks}. "
                    f"Use False for fully random or a strictly positive integer for block masking."
                )
            requested_blocks = int(self.mask_C_with_blocks)
        else:
            raise ValueError(
                f"Invalid mask_C_with_blocks type {type(self.mask_C_with_blocks)}. Expected bool or int."
            )

        n_blocks = max(1, min(requested_blocks, C))

        # Zeros are allowed so some sampled blocks can be empty.
        block_lengths = self._sample_partition(B, n_blocks, cutoff, device=device)
        gap_lengths = self._sample_partition(B, n_blocks, C - cutoff, device=device)

        # Build non-overlapping circular blocks by alternating gaps and visible blocks.
        offset = torch.randint(0, C, (B,), device=device)
        block_starts = (
            offset.unsqueeze(1)
            + torch.cumsum(gap_lengths, dim=1)
            + (torch.cumsum(block_lengths, dim=1) - block_lengths)
        ) % C

        visible = torch.zeros((B, C), dtype=torch.bool, device=device)
        max_len = int(block_lengths.max().item())
        if max_len > 0:
            step = torch.arange(max_len, device=device).view(1, 1, -1)
            valid = step < block_lengths.unsqueeze(-1)
            indices = (block_starts.unsqueeze(-1) + step) % C
            rows = torch.arange(B, device=device).view(B, 1, 1).expand_as(indices)
            visible[rows[valid], indices[valid]] = True

        all_indices = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
        indices_visible = all_indices[visible].view(B, cutoff)
        indices_masked = all_indices[~visible].view(B, C - cutoff)
        return indices_visible, indices_masked

    def __call__(self, batch):
        """Generate mask for batch
        :param batch: batch of data
        :return: Mask dictionary (for each modality a dictionnary {S:mask, T: mask, C: mask})
        """
        #Spatial masking is shared across modalities
        modalities = self._get_modalities(batch.keys())
        B = batch[modalities[0]].shape[0]
        device = batch[modalities[0]].device

        # IMPORTANT: build spatial masks at the *input scale*.
        # This guarantees that `Mask.S_length` matches the encoder spatial grid `S`
        # (which is computed from `input_scale`), even when `output_scale` is finer.
        mask_scale = float(batch['input_scale'])
        patched_input_size = (
            max(int(self.patched_input_size[0] / mask_scale + 1e-6), 1),
            max(int(self.patched_input_size[1] / mask_scale + 1e-6), 1),
        )

        mask_enc_S, mask_pred_S = self._mask_Spatial(B, patched_input_size, device=device)
        assert mask_enc_S.numel() > 0 , f"mask_enc_S is empty. {mask_enc_S} {patched_input_size}"

        #Temporal and channel masking is done for each modality
        masks_enc = {}
        masks_pred = {}
        for modality in modalities:
            shape = batch[modality].shape
            if len(shape) == 4:
                B,C,H,W = shape
                T=1
            elif len(shape) == 5:
                B,T,C,H,W = shape
            else:
                raise ValueError(f"{modality} has an invalid shape {shape}. Must be 4 or 5")
            masks_enc_C, mask_pred_C = self._mask_Channel(B, C, device=device)
            masks_enc_T, mask_pred_T = self._mask_Temporal(B, T, device=device)

            masks_enc[modality] = Mask(S=mask_enc_S, T=masks_enc_T, C=masks_enc_C, S_length=patched_input_size[0]*patched_input_size[1], T_length=T, C_length=C)

        return [masks_enc], [MaskSpatial(S=mask_pred_S, S_length=patched_input_size[0]*patched_input_size[1])]