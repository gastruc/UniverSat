from functools import partial
from typing import List

import torch
from torch import nn

from models.networks.encoder.utils.patch_embeddings import MPFourier
from models.networks.encoder.utils.pos_embed import get_coords
from models.networks.encoder.utils.utils import RMSNorm
from models.networks.encoder.utils.utils_ViT import ACABlock
from models.networks.masking import Mask

class UniversalPatchEncoder(nn.Module):
    """Per-modality patch encoder used inside :class:`UniverSat`.

    Runs a configurable sequence of axis-wise attention layers
    (:class:`~models.networks.encoder.utils.utils_ViT.ACABlock`) over a
    6-D tensor ``(B, T, C, S, s, D)`` where ``T`` is time, ``C`` is
    spectral channels, ``S`` is the patch grid and ``s`` is the sub-patch
    grid. The order of axes attended to is given by ``order``:

    - ``"S1"`` — sub-patch spatial pooling (2D RoPE).
    - ``"C"``  — spectral attention. Continuous wavelengths are embedded
      with :class:`MPFourier`; named channels (``VV``, ``VH``,
      ``Ratio_VV_VH``, ``HH``, ``HV``, ``Ratio_HH_HV``, ``nDEM``,
      ``DSM``) use learned per-channel embeddings.
    - ``"T"``  — temporal attention (1D RoPE). Two extra register tokens
      are prepended on this axis.
    - ``"S"``  — patch spatial attention (2D RoPE). Four extra register
      tokens are prepended; a final ``Linear(embed_dim, final_dim)`` head
      projects the per-sub-patch features to ``final_dim`` for the trunk.

    Args:
        embed_dim: working channel dimension (per axis). Note this is
            multiplied along the way by ``expand_dim[i]`` so the final
            channel size of each output is ``embed_dim * prod(expand_dim)``.
        final_dim: output dimension of the projection head used for
            sub-patch features (must match the trunk's ``embed_dim``).
        num_heads: attention heads per block, scaled per axis: ``//4`` for
            spectral ``C``, ``//2`` for temporal ``T``, ``//6`` for sub-patch
            ``S1``; unchanged for patch spatial ``S``.
        mlp_ratio: hidden-to-input ratio of each block's MLP.
        qkv_bias: whether QKV projections carry a bias term.
        attn_drop_rate: attention-map dropout shared by every block.
        norm_layer: kept for backward compatibility; RMSNorm is forced.
        n_queries: number of output tokens along the attended axis (one
            entry per ``order`` step).
        order: sequence of axis labels — ``"S1"``, ``"C"``, ``"T"``, ``"S"``.
        expand_dim: multiplier on ``embed_dim`` after each step (one
            entry per ``order`` step).
        gating: enable the sigmoid-gated attention output in the
            ``ACABlock`` layers (used by Tiny / Base v2 configs).
    """

    def __init__(
        self,
        embed_dim=64,
        final_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        norm_layer=partial(RMSNorm),
        n_queries=[],
        order=[],
        expand_dim=[],
        gating=False,
    ):
        super().__init__()
        assert len(n_queries) == len(order), "n_queries and order must have the same length"
        assert len(n_queries) == len(expand_dim), "n_queries and expand_dim must have the same length"

        self.spectral_embed = MPFourier(embed_dim)
        self.embed = MPFourier(embed_dim, bandwidth=2.5)

        for modality in ['VV', 'VH', 'Ratio_VV_VH', 'HH', 'HV', 'Ratio_HH_HV', 'nDEM', 'DSM']:
            setattr(self, '_'.join(['Encoding', modality]), nn.Parameter(torch.randn(embed_dim), requires_grad=True))

        blocks = []

        spectral, temporal, spatial = True, True, True
        max_seq_len = 12
        for i in range(len(order)):
            num_heads_i = num_heads
            if order[i] == 'C' and spectral:
                RoPe = None
                max_seq_len = 250
                spectral = False
                num_heads_i = num_heads_i // 4
            elif order[i] == 'T' and temporal:
                RoPe = '1D'
                max_seq_len = 750
                temporal = False
                self.registers_temporal = nn.Parameter(torch.empty(1, 2, embed_dim))
                nn.init.normal_(self.registers_temporal, std=0.02)
                num_heads_i = num_heads_i // 2
            elif order[i] == 'S1':
                RoPe = '2D'
                num_heads_i = 1 if num_heads_i < 3 else num_heads_i//6
            elif order[i] == 'S' and spatial:
                RoPe = '2D'
                spatial = False
                self.registers_spatial = nn.Parameter(torch.empty(1, 4, embed_dim))
                self.spatial_proj = nn.Linear(embed_dim, final_dim, bias=True)
                nn.init.normal_(self.registers_spatial, std=0.02)
            else:
                RoPe = '1D'
                max_seq_len = max(n_queries)
            blocks.append(ACABlock(dim=embed_dim, num_heads=num_heads_i, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                               attn_drop=attn_drop_rate, n_queries=n_queries[i], expand_dim=expand_dim[i], max_seq_len=max_seq_len,
                               RoPe=RoPe, gating=gating))
            embed_dim = embed_dim * expand_dim[i]

        self.predictor_blocks = nn.ModuleList(blocks)
        self.order = order
        self.n_queries = n_queries
        self.expand_dim = expand_dim

        self.predictor_norm = norm_layer(embed_dim)
        self.final_proj = nn.Linear(embed_dim, final_dim, bias=True)

    def forward(self, x, modality, wavelengths, scale, dates, subpatch=1, res=1, masks:List[Mask]=None):
        B_o = x.shape[0]
        x = x.unsqueeze(-1)

        coords_spectral_tensors = []
        for i, wv in enumerate(wavelengths):
            if type(wv) == str:
                # Named channels (VV/VH/HH/.../DSM): a learned unit-variance code,
                # matching the scale of the MPFourier wavelength embedding below.
                coords_spectral_tensors.append(getattr(self, f'Encoding_{wv}').view(1, 1, -1))
            else:
                coords_spectral_tensors.append(self.spectral_embed(torch.tensor(wv, device=x.device, dtype=x.dtype).view(1, 1, 1)))

        coords_spectral = torch.cat(coords_spectral_tensors, dim=1)
        coords_spectral = coords_spectral.repeat(x.shape[0], 1, 1)

        x = self.embed(x)
        x = to_scale(x, scale, res, subpatch)

        B, T, C, S, s, D = x.shape
        P = B // B_o

        coords_spectral = coords_spectral.repeat_interleave(P, dim=0)
        coords_spatial = get_coords(x, int(S ** 0.5), 1, 4, res=1) * (scale*int(S ** 0.5)/100) #in km
        coords_sub_spatial = get_coords(x, int(s ** 0.5), 1, 0, res=1) * (subpatch * res)/10


        if masks is not None:
            x = [mask.apply(x.view(B_o, P, T, C, S, s, D), axis='SCT', current_shape="BSTCXXD").flatten(0,1) for mask in masks]
            x = torch.cat(x, dim=0) #len(masks) * B, T, C, S, s, D
            coords_spectral = [mask.apply(coords_spectral.view(B_o, P, C, D), axis='SC', current_shape="BSCD").flatten(0,1) for mask in masks]
            coords_spectral = torch.cat(coords_spectral, dim=0) #len(masks) * B, C, D
            coords_spatial = coords_spatial[:x.shape[0]] #it is the same for all masks
            coords_sub_spatial = coords_sub_spatial[:x.shape[0]] #it is the same for all masks
            dates = [mask.apply(dates, axis='T', current_shape="BT") for mask in masks]
            dates = torch.cat(dates, dim=0) #len(masks) * B, T

        B, T, C, S, s, D = x.shape #now masked

        coords_temporal = torch.cat([torch.zeros(x.shape[0], self.registers_temporal.shape[1], device=x.device), dates.repeat_interleave(x.shape[0]//dates.shape[0], dim=0)], dim=1).int()

        out = {"coords_spatial": coords_spatial, "N_masked": B//B_o}
        if B == 0:
            return out

        D_orig = D
        for i in range(len(self.order)):
            blk = self.predictor_blocks[i]
            if self.order[i] == 'C':
                # Attention on Spectral Axis
                if coords_spectral is not None:
                    D_expand = D // D_orig
                    cs = coords_spectral.repeat(1, 1, D_expand) if D_expand > 1 else coords_spectral
                    x = x + cs.unsqueeze(1).unsqueeze(3) #len(masks) * B, T, C, S, D
                x = x.permute(0, 1, 3, 2, 4)
                x = x.flatten(0, 2)
                x = blk(x)
                coords_spectral = None
                C = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, T, S, C, D).permute(0, 1, 3, 2, 4)

            elif self.order[i] == 'T':
                # Attention on Temporal Axis
                out['temporal'] = x
                x = x.permute(0, 2, 3, 1, 4)
                x = x.flatten(0, 2)
                if T > 1:
                    x = torch.cat([self.registers_temporal.repeat(B * C * S, 1, 1), x], dim=1)
                else:
                    coords_temporal = coords_temporal[:, self.registers_temporal.shape[1]:]
                if coords_temporal is not None:
                    x = blk(x, coords=coords_temporal.repeat_interleave(C * S, dim=0))
                else:
                    x = blk(x)
                coords_temporal = None
                T = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, C, S, T, D).permute(0, 3, 1, 2, 4)

            elif self.order[i] == 'S':
                # Attention on Spatial Axis
                out['spatial'] = self.spatial_proj(x.flatten(1, 3))
                x = x.flatten(0, 2)
                x = torch.cat([self.registers_spatial.repeat(B * T * C, 1, 1), x], dim=1)
                if coords_spatial is not None:
                    x = blk(x, coords=coords_spatial.repeat_interleave(T * C, dim=0))
                else:
                    x = blk(x)
                coords_spatial = None
                S = self.n_queries[i]
                D = D * self.expand_dim[i]
                x = x.view(B, T, C, S, D)

            elif self.order[i] == 'S1':
                x = x.flatten(0, 3)
                x = blk(x, coords=coords_sub_spatial.repeat_interleave(S * T * C, dim=0))
                D = D * self.expand_dim[i]
                x = x.view(B, T, C, S, 1, D).squeeze(4)

        x = self.predictor_norm(x)
        x = self.final_proj(x)
        out['tokens'] = x
        return out

def to_scale(x, scale, res, subpatch=1):
    """Tile a per-pixel feature map into (patch, sub-patch) grids.

    Given ``x`` of shape ``(B, T, C, H, W, D)`` (per-pixel features for
    every modality timestep) and a physical resolution ``res`` (meters
    per pixel), unfold to a 7-D tensor of shape
    ``(B * S, T, C, S, s, D)`` where:

    - ``patch_size = max(int(scale * 10 / res), 1)`` pixels per patch,
    - ``S`` is the number of patches per image
      (``(H // patch_size) * (W // patch_size)``),
    - ``s = (patch_size // subpatch) ** 2`` is the number of sub-patches
      per patch.

    Args:
        x: input feature map, shape ``(B, T, C, H, W, D)``.
        scale: target patch scale in units of 10 m.
        res: physical resolution of ``x`` in meters per pixel.
        subpatch: sub-patch stride (1 for no sub-patching).

    Returns:
        Tensor of shape ``(B * S, T, C, S, s, D)``.
    """
    grid_size = max(int(scale * 10 / res), 1)
    B, T, C, H, W, D = x.shape

    x = x.permute(0, 1, 2, 5, 3, 4)  # B, T, C, D, H, W
    x = x.unfold(4, grid_size, grid_size).unfold(5, grid_size, grid_size)

    x = x.unfold(6, subpatch, subpatch).unfold(7, subpatch, subpatch)
    x = x.flatten(4, 5).flatten(5, 6).flatten(6, 7)
    x = x.permute(0, 4, 1, 2, 5, 6, 3).flatten(0, 1)
    return x



def repeat_interleave_batch(x, B, repeat):
    """Repeat each of the ``N`` groups (size B) ``repeat`` times along dim 0.

    Equivalent to::

        einops.rearrange(x, '(n b) ... -> n b ...', n=N, b=B)
        einops.repeat(_, 'n b ... -> (n r b) ...', r=repeat)

    Implemented with torch primitives so the hub entrypoint has no einops dep.
    """
    N = len(x) // B
    x = x.reshape(N, B, *x.shape[1:])
    # n b ... -> n r b ... -> (n r b) ...
    x = x.unsqueeze(1).expand(N, repeat, B, *x.shape[2:]).reshape(N * repeat * B, *x.shape[2:])
    return x
