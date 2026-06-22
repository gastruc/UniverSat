"""Multi-modality SSL predictor.

Used by ``models.networks.SSLUniverSat.SSLUniverSat`` during pretraining to
predict masked targets from visible context tokens. Not loaded by the
``torch.hub`` release path; an ``einops`` dependency is fine here.
"""

from functools import partial

import einops
import torch
from torch import nn

from models.networks.encoder.UniversalPatchEncoder import repeat_interleave_batch
from models.networks.encoder.utils.patch_embeddings import MPFourier
from models.networks.encoder.utils.pos_embed import get_2d_sincos_pos_embed_with_scale
from models.networks.encoder.utils.utils import DynamicTanh, RMSNorm
from models.networks.encoder.utils.utils_ViT import Block
from models.networks.masking import apply_spatial_masks


class Predictor(nn.Module):
    """ Predictor """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        out_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(DynamicTanh, channels_last=True),
        init_std=0.02,
        scales={},
        n_registers=0,
        pos_encoding_type: str = 'rope2d',
        modality_wise=False,
        modalities_list=None,
        time_wise=False,
        dates_to_select=-1,
        gating=False,
    ):
        super().__init__()
        self.n_registers = n_registers
        assert pos_encoding_type.lower() in ['rope2d', 'sincos'], f"Unknown pos_encoding_type: {pos_encoding_type}"
        self.pos_encoding_type = pos_encoding_type.lower()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.mask_token, std=init_std)

        norm_layer = partial(RMSNorm)

        self.cloud_density_threshold = 0.3

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.datasets = list(scales.keys())
        self.predictor_pos_embed = {}
        self.num_patches = num_patches
        for dataset in self.datasets:
            scales_list = scales[dataset]["output_scales"]
            if "input" in scales_list:
                scales_list.remove("input")
                scales_list.extend(scales[dataset]["input_scales"])
            if "latent" in scales_list:
                scales_list.remove("latent")
                scales_list.extend(scales[dataset]["latent_scales"])
            scales_list = list(set(scales_list))
            for scale in scales_list:
                num_p = num_patches[dataset] // (scale * scale)
                self.predictor_pos_embed['_'.join([dataset, str(scale)])] = get_2d_sincos_pos_embed_with_scale(embed_dim,
                                                            int(num_p ** .5), scale, n_registers=0)
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, gating=gating)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, out_dim, bias=True)

        self.modality_wise = modality_wise
        if self.modality_wise:
            assert modalities_list is not None, "modalities_list must be provided when modality_wise is True"
            if type(modalities_list) == dict:
                modalities_list = list(modalities_list.keys())
            self.modalities_list = modalities_list
            self.modality_embeddings = nn.ParameterDict()
            for modality in self.modalities_list:
                self.modality_embeddings[modality] = nn.Parameter(torch.randn(predictor_embed_dim), requires_grad=True)

        self.time_wise = time_wise
        if self.time_wise:
            assert modality_wise, "time_wise can only be True if modality_wise is also True"
            self.time_embedding = MPFourier(predictor_embed_dim)
            self.date_to_select = dates_to_select

    def _select_time_idx(self, pred_tokens, dates, modality, cloud_density, masks_out, masks_x, device):
        if self.date_to_select == -1:  # Randomly select a date for each patch in the batch
            if cloud_density and modality + '_cloud_density' in cloud_density:  # sample timestep without clouds
                cloud_density_modality = einops.repeat(
                    cloud_density[modality + '_cloud_density'],
                    'b t -> (r b) n t',
                    r=len(masks_out) * len(masks_x),
                    n=pred_tokens.size(1),
                )  # B, T -> len(masks_out)*len(masks_x)*B, N_out, T
                proba = torch.randn((pred_tokens.size(0), pred_tokens.size(1), dates[modality + '_dates'].shape[1]), device=device)  # len(masks_out)* len(masks_x)*B, N_out, T
                proba[cloud_density_modality > self.cloud_density_threshold] = -1e6  # avoid cloudy timesteps
                time_idx = torch.argmax(proba, dim=-1)  # len(masks_out)*len(masks_x)*B, N_out
            else:  # sample a random time for all patches
                time_idx = torch.randint(0, dates[modality + '_dates'].shape[1], (pred_tokens.size(0), pred_tokens.size(1)), device=device)  # len(masks_out)*len(masks_x)*B, N_out

        else:  # select a few date for all patches
            T = dates[modality + '_dates'].shape[1]
            n_select = min(self.date_to_select, T)
            if cloud_density and modality + '_cloud_density' in cloud_density:  # sample timestep without clouds
                cloud_density_modality = einops.repeat(
                    cloud_density[modality + '_cloud_density'],
                    'b t -> (r b) t',
                    r=len(masks_out) * len(masks_x),
                )  # B, T -> len(masks_out)*len(masks_x)*B, T
                proba = torch.randn((pred_tokens.size(0), T), device=device)  # len(masks_out)* len(masks_x)*B, T
                proba[cloud_density_modality > self.cloud_density_threshold] = -1e6  # avoid cloudy timesteps
                selected_time_idx = torch.argsort(proba, dim=-1)[:, :n_select]  # len(masks_out)*len(masks_x)*B, date_to_select
            else:  # sample a random time for all patches
                selected_time_idx = torch.argsort(torch.randn((pred_tokens.size(0), T), device=device), dim=-1)[:, :n_select]  # len(masks_out)*len(masks_x)*B, date_to_select

            time_idx = torch.randint(0, n_select, (pred_tokens.size(0), pred_tokens.size(1)), device=device)  # len(masks_out)*len(masks_x)*B, N_out
            time_idx = selected_time_idx.gather(1, time_idx)  # len(masks_out)*len(masks_x)*B, N_out

        return time_idx

    @torch.compile()
    def _predict_mask_tokens(self, x, coords, n_context_tokens):
        for blk in self.predictor_blocks:
            x = blk(x, coords)

        x = x[:, (n_context_tokens + self.n_registers):]
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x

    def forward(self, x, masks_x, masks_out, dataset, scale, modalities=None, dates=None, cloud_density=None, coords_out=None, coords_in=None):
        assert (masks_out is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks_out, list):
            masks_out = [masks_out]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        x, registers = x[:, self.n_registers:, :], x[:, :self.n_registers, :]

        _, N_ctxt, D = x.shape

        # -- add positional embedding to x tokens
        if self.pos_encoding_type == 'rope2d':
            pred_tokens = self.mask_token.repeat(len(masks_out) * len(masks_x) * B, masks_out[0].S.shape[1], 1)
            x = x.repeat(len(masks_out), 1, 1)
            registers = registers.repeat(len(masks_out), 1, 1)

            coords_register = coords_out[:, :self.n_registers].repeat(len(masks_out) * len(masks_x), 1, 1)                                   #len(masks_out)*len(masks_x)*B, N_reg, D
            coords_context = apply_spatial_masks(coords_out[:, self.n_registers:], masks_x).repeat(len(masks_out), 1, 1)    
            coords_pred = repeat_interleave_batch(apply_spatial_masks(coords_in[:, self.n_registers:], masks_out), B, repeat=len(masks_x))  #len(masks_out)*len(masks_x)*B, N_out, D
        else: #sincos
            coords = None

            pos_emb = self.predictor_pos_embed['_'.join([dataset, str(scale)])].to(x.device)
            pos_embs = pos_emb.repeat(B, 1, 1)
            x = x + apply_spatial_masks(pos_embs, masks_x)                          # len(masks_x)*B, N_ctxt, D
            pred_tokens = self.mask_token.repeat(B, masks_out[0].S.shape[1], 1)
            pred_tokens = pred_tokens + apply_spatial_masks(pos_embs, masks_out)    # len(masks_out)*B, N_out, D

            x = x.repeat(len(masks_out), 1, 1)                                              # len(masks_out)*len(masks_x)*B, N_ctxt, D
            pred_tokens = repeat_interleave_batch(pred_tokens, B, repeat=len(masks_x))      # len(masks_out)*len(masks_x)*B, N_out, D
            registers = registers.repeat(len(masks_out), 1, 1)                              # len(masks_out)*len(masks_x)*B, N_reg, D

        meta_out = {}
        if self.modality_wise:
            assert modalities is not None, "modalities must be provided when modality_wise is True"
            if self.pos_encoding_type == 'rope2d':
                coords_pred = coords_pred.repeat(1,len(modalities), 1)
            pred_tokens_modality=[]
            meta_out['modality_list'] = modalities

            for modality in modalities:
                modality_emb = self.modality_embeddings[modality].view(1, 1, -1)
                modality_emb = modality_emb.repeat(pred_tokens.size(0), pred_tokens.size(1), 1)
                if self.time_wise and dates is not None and modality + '_dates' in dates.keys():
                    # for each patch in the batch, get a random time embedding
                    time_idx = self._select_time_idx(
                        pred_tokens=pred_tokens,
                        dates=dates,
                        modality=modality,
                        cloud_density=cloud_density,
                        masks_out=masks_out,
                        masks_x=masks_x,
                        device=x.device,
                    )

                    time_values = dates[modality + '_dates'].repeat(len(masks_out)*len(masks_x), 1).gather(1, time_idx)  #len(masks_out)*len(masks_x)*B, N_out
                    time_emb = self.time_embedding((time_values.float()/365).unsqueeze(-1))
                    modality_emb = modality_emb + time_emb
                    meta_out[modality + '_dates'] = time_idx
                pred_tokens_modality.append(pred_tokens + modality_emb)

            pred_tokens = torch.cat(pred_tokens_modality, dim=1)  # len(masks_out)*len(masks_x)*B, len(modalities)*N_out, D

        # -- concat mask tokens to x
        x = torch.cat([registers, x, pred_tokens], dim=1)
        if self.pos_encoding_type == 'rope2d':
            coords = torch.cat([coords_register, coords_context, coords_pred], dim=1)

        x = self._predict_mask_tokens(x, coords, N_ctxt)
        return x, meta_out
