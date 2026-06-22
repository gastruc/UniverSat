from typing import List

import torch
from torch import nn

from models.networks.encoder.utils.pos_embed import get_coords
from models.networks.encoder.utils.utils import _init_weights
from models.networks.masking import (MaskSpatial, ModalityMaskCollection,
                                     extract_non_empty_spatial_masks)


class SSLUniverSat(nn.Module):
    """Self-supervised learning wrapper around a UniverSat encoder.

    The module couples a UniverSat encoder with a predictor head used during
    masked/self-supervised pretraining. It exposes helper methods to encode
    visible tokens and predict masked target tokens while carrying the static
    modality and dataset metadata required by the encoder and predictor.

    Args:
        encoder: UniverSat encoder module. It receives the multimodal input
            batch, modality metadata, scale information, and encoder/predictor
            masks, and returns encoded tokens plus intermediate outputs.
        predictor: Predictor module that maps encoder tokens and masks to
            predicted target tokens and associated prediction metadata.
        wavelengths: Mapping from modality name to channel wavelengths. Passed
            to the encoder to describe the spectral layout of each modality.
        input_res: Mapping from modality name to native input resolution. Values
            are converted to tensors and passed to the encoder for scale-aware
            positional/modality processing.
        n_registers: Number of register tokens prepended by the encoder and
            preserved when computing predictor input/output coordinates.
        output_grid: Mapping from dataset name to the number of spatial tokens
            in that dataset at output scale 1. Also reused as ``latent_grid``;
            both are divided by ``output_scale ** 2`` or ``latent_scale ** 2``
            at runtime.
        subpatches: Mapping from modality name to subpatch configuration used
            by the encoder's patch embedding logic.
        keep_subpatch: Whether the encoder should keep subpatch-level tokens
            instead of aggregating them in the patch encoder.
        num_patches: Mapping from dataset name to its base number of spatial
            patches. Stored for configuration compatibility with predictor and
            dataset metadata.
    """
    def __init__(self,
        encoder,
        predictor,
        wavelengths: dict = {},
        input_res: dict = {},
        n_registers: int = 0,
        output_grid: dict = {},
        subpatches: dict = {},
        keep_subpatch: bool = False,
        num_patches: dict = {},
        ):
        super().__init__()
        self.wavelengths = wavelengths
        self.input_res = {k: torch.tensor(v) for k, v in input_res.items()}
        self.n_registers = n_registers
        self.latent_grid =  output_grid
        self.output_grid = output_grid
        self.subpatches = subpatches
        self.keep_subpatch = keep_subpatch
        self.num_patches = num_patches

        self.encoder = encoder
        self.predictor = predictor
        self.apply(_init_weights)

    def forward_encoder(self, x, mask_enc: ModalityMaskCollection, mask_pred: List[MaskSpatial], dataset: str):
        tokens, intermediate = self.encoder(x,
                              self.wavelengths,
                              self.input_res, x['input_scale'],
                              self.latent_grid[x['dataset']]//x['latent_scale']**2,
                              self.output_grid[x['dataset']]//x['output_scale']**2,
                              self.subpatches,
                              self.keep_subpatch,
                              keep_intermediate=True,
                              mask_in=mask_enc,
                              mask_out=mask_pred,
                              dataset=dataset
                              )
        return tokens, intermediate

    def forward_decoder(self, out, tokens, mask_enc: List[MaskSpatial], mask_pred: List[MaskSpatial], modalities=None, dates=None, cloud_density=None):
        # embed tokens
        output_grid = self.output_grid[out['dataset']] // out['output_scale']**2
        coords_out = get_coords(tokens, int(output_grid ** 0.5), 1, self.n_registers)
        latent_grid = self.latent_grid[out['dataset']] // out['latent_scale']**2
        coords_in = get_coords(tokens, int(latent_grid ** 0.5), 1, self.n_registers)
        return self.predictor(tokens, mask_enc, mask_pred, out['dataset'], out['latent_scale'], modalities=modalities, dates=dates, cloud_density=cloud_density, coords_out=coords_out, coords_in=coords_in)

    def forward(self, imgs, mask_enc, mask_pred):
        out = {'dataset': imgs['dataset'], 'output_scale': imgs['output_scale'], 'latent_scale': imgs['latent_scale']}
        modalities = sorted(list(mask_enc[0].keys()))
        dates = {modality + '_dates' : imgs[modality + '_dates'] for modality in modalities if modality + '_dates' in imgs}
        # to get the global mask, we need to find the first non-empty mask
        spatial_mask = extract_non_empty_spatial_masks(mask_enc, modalities)

        target_out_S = int(self.output_grid[out['dataset']] // out['output_scale']**2)
        spatial_mask = [
            (m.upsample_mask(target_out_S) if getattr(m, "S_length", None) != target_out_S else m)
            for m in spatial_mask
        ]

        # Hard alignment checks: if these fail, learning signal will be corrupted.
        for m in spatial_mask:
            if getattr(m, "S", None) is not None and m.S.numel() > 0:
                assert int(m.S.max()) < target_out_S, f"spatial_mask index {int(m.S.max())} >= target_out_S {target_out_S}"

        tokens, intermediate = self.forward_encoder(imgs, mask_enc, spatial_mask, imgs['dataset'])
        out['encoder_tokens'] = tokens[:, self.encoder.n_registers:, :] # Exclude register tokens if any
        out['encoder_register'] = tokens[:, :self.encoder.n_registers, :] # Include only register tokens
        out['output_mask'] = spatial_mask
        out.update(intermediate)
        out['predicted_tokens'], out['predicted_meta']=  self.forward_decoder(out, tokens, spatial_mask, mask_pred, modalities=modalities, dates=dates, cloud_density={k: v for k, v in imgs.items() if 'cloud_density' in k})
        return out
