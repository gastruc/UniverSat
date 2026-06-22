from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn

from models.networks.encoder.utils.pos_embed import get_coords
from models.networks.encoder.utils.utils import RMSNorm
from models.networks.encoder.utils.utils_ViT import (
    Block,
    ACABlock,
    CrossBlock,
    CrossBlockLearned,
)
from models.networks.masking import (
    MaskSpatial,
    ModalityMaskCollection,
    apply_spatial_masks,
    extract_non_empty_spatial_masks,
    merge_Mask,
)


def unroll_block_list(blocks: List[str]) -> List[str]:
    """Expand block repeat shorthand such as ``"SAx3"``.

    Args:
        blocks: Ordered block names. Entries containing ``x`` are interpreted
            as ``"<block_name>x<count>"`` and expanded in place.

    Returns:
        A flat block-name list.
    """
    unrolled_blocks = []
    for block in blocks:
        if "x" in block:
            n = int(block.split("x")[1])
            block_type = block.split("x")[0]
            unrolled_blocks.extend([block_type] * n)
        else:
            unrolled_blocks.append(block)
    return unrolled_blocks


class UniverSat(nn.Module):
    """Top-level multimodal, multi-resolution, multi-scale encoder.

    Each call goes through two stages:

    1. ``UPE_forward`` — per-modality patch encoding via ``spatial_encoder``
       (a :class:`~models.networks.encoder.UniversalPatchEncoder.UniversalPatchEncoder`)
       plus per-(dataset, modality) MLP projector heads used during SSL.
    2. ``ViT_forward`` — a stack of mixed attention blocks (selected by
       ``block_type``) over the concatenated tokens, with optional masking
       and bilinear up/down-sampling between ``latent_grid`` and
       ``output_grid``.

    ``norm_layer`` (default ``RMSNorm``) is the single source of normalization
    and is **propagated to every submodule** — the trunk blocks (``Block`` /
    ``ACABlock`` / ``CrossBlock`` / ``CrossBlockLearned``), the
    optional pre-trunk norm, the per-(dataset, modality) SSL projector heads,
    **and the patch encoder**. To propagate it into the encoder, ``spatial_encoder``
    is supplied as a *partial* (Hydra ``_partial_: true`` / ``functools.partial``)
    and finished here with ``norm_layer``; an already-built module is accepted
    as-is (and then keeps whatever norm it was built with).

    Args:
        spatial_encoder: per-modality patch encoder, normally a *partial* of
            :class:`UniversalPatchEncoder` (finished here with ``norm_layer``).
            A pre-built instance is also accepted.
        block_type: ordered list of block kinds composing the trunk. Each
            entry is one of ``"SA"`` (self-attention), ``"CA_In"`` /
            ``"CA_Out"`` / ``"CA_Sub"`` (sparse cross-attention),
            ``"QCA_In"`` / ``"QCA_Out"`` (learned-query cross-attention),
            ``"Bi_ACA_in"`` (mean/max-pooled cross-attention input fusion),
            or ``"Bilinear_out"`` (interpolation-only resize). Suffix
            ``xN`` repeats the block ``N`` times (e.g. ``"SAx6"``).
        embed_dim: token embedding dimension.
        num_heads: attention heads per block.
        mlp_ratio: hidden-to-input ratio in each block's MLP.
        qkv_bias: whether QKV projections carry a bias term.
        qk_scale: legacy unused override for the QK scaling factor. Kept
            for state-dict compatibility; ignored at runtime (blocks use
            ``head_dim ** -0.5``).
        n_registers: number of learnable register / "[CLS]"-like tokens
            prepended before the spatial trunk.
        pre_norm: apply RMSNorm to tokens before entering the trunk.
        drop_rate: legacy / unused dropout knob kept for config-compat.
        drop_path_rate: stochastic depth — interpolated linearly across
            ``len(block_type)`` blocks.
        attn_drop_rate: attention-map dropout used inside each block.
        norm_layer: normalization factory (default ``partial(RMSNorm)``) shared
            by the trunk, projectors, and the patch encoder (see above).
        gating: when True, attention layers add a sigmoid-gated output
            (used for the v2 Tiny / Base configs; off for Large v2).
        proba_drop_modalities: probability of dropping each non-MODIS
            modality at SSL pre-training time. Set ``0.0`` for inference.
        modalities_dict: ``{dataset_name: [modality_name, ...]}``. One MLP
            projector head ``self.projector__<dataset>_<modality>`` is
            created for each non-``modis`` entry. Pretrained checkpoints
            expect the exact same set of (dataset, modality) keys.
    """

    def __init__(
        self,
        spatial_encoder: nn.Module,
        block_type: List[str] = None,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        n_registers: int = 1,
        pre_norm: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=partial(RMSNorm),
        gating: bool = False,
        proba_drop_modalities: float = 0.3,
        modalities_dict: Dict[str, List[str]] = None,
    ):

        super().__init__()
        block_type = unroll_block_list(block_type or [])
        modalities_dict = modalities_dict or {}
        _ = qk_scale, drop_rate  # accepted for config compatibility

        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

        self.n_registers = n_registers
        if n_registers > 0:
            self.registers = nn.Parameter(torch.empty(1, n_registers, embed_dim))
            nn.init.normal_(self.registers, std=0.02)
        else:
            self.registers = None

        self.proba_drop_modalities = proba_drop_modalities

        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        # Propagate our norm layer into the patch encoder so every submodule
        # shares one normalization. ``spatial_encoder`` is supplied as a partial
        # (Hydra ``_partial_: true`` / ``functools.partial``) and finished here;
        # an already-built module is accepted as-is for backward compatibility.
        if isinstance(spatial_encoder, nn.Module):
            self.spatial_encoder = spatial_encoder
        else:
            self.spatial_encoder = spatial_encoder(norm_layer=norm_layer)

        for dataset in sorted(list(modalities_dict.keys())):
            for modality in modalities_dict[dataset]:
                if modality != "modis":
                    setattr(
                        self,
                        f"projector__{dataset}_{modality}",
                        nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            norm_layer(embed_dim * 2),
                            nn.Linear(embed_dim * 2, embed_dim),
                        ),
                    )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(block_type))]

        self.block_type = block_type
        blocks = []
        for i in range(len(block_type)):
            if block_type[i] == "SA":
                blocks.append(
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        gating=gating,
                    )
                )
            elif block_type[i] == "QCA_In" or block_type[i] == "QCA_Out":
                blocks.append(
                    CrossBlockLearned(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        n_registers=n_registers,
                    )
                )
            elif (
                block_type[i] == "CA_Sub"
                or block_type[i] == "CA_In"
                or block_type[i] == "CA_Out"
            ):
                blocks.append(
                    CrossBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        gating=gating,
                    )
                )
            elif block_type[i] == "Bilinear_out":
                blocks.append(nn.Identity())
            elif block_type[i] == "Bi_ACA_in":
                blocks.append(
                    ACABlock(
                        dim=embed_dim,
                        num_heads=8,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        expand_dim=1,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        n_queries=1,
                        RoPe=None,
                        gating=gating,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown block type: {block_type[i]}. Block type must be one of 'SA', 'CA_In', 'CA_Out', 'CA_Sub', 'QCA_In', 'QCA_Out', 'Bilinear_out', or 'Bi_ACA_in'."
                )

        self.blocks = nn.ModuleList(blocks)

    @torch.compile()
    def UPE_forward(
        self,
        x,
        modalities,
        wavelengths,
        input_res,
        scale,
        subpatches,
        mask_in: ModalityMaskCollection = None,
        dataset: str = "",
    ):
        """Encode each input modality into patch tokens and sub-patch tokens.

        Returns modality token tensors, their coordinates and grid sizes,
        flattened sub-patch tokens, optional SSL projector outputs, and the
        modalities kept after stochastic modality dropout.
        """
        tokens, spatial, coords_spatial, coords_in = [], [], [], []
        token_sizes = []
        kept_modalities = []
        intermediate_tokens = {}

        B = x[modalities[0]].shape[0]

        # Move modis first BEFORE computing proba_drop so indices stay aligned
        if "modis" in modalities:
            modalities = [m for m in modalities if m != "modis"]
            modalities.insert(0, "modis")

        if mask_in is not None:
            proba_drop = torch.rand(len(modalities))
            # Force at least one non-modis modality to not be dropped
            # so that `spatial` is never empty
            non_modis_indices = [i for i, m in enumerate(modalities) if m != "modis"]
            if non_modis_indices:
                non_modis_indices_tensor = torch.tensor(
                    non_modis_indices, device=proba_drop.device
                )
                forced_idx = non_modis_indices_tensor[
                    torch.randint(
                        0,
                        non_modis_indices_tensor.numel(),
                        (1,),
                        device=proba_drop.device,
                    )
                ]
                proba_drop[forced_idx] = 1.0
            else:
                proba_drop[torch.randint(0, len(modalities), (1,))] = 1.0
        else:
            proba_drop = torch.ones(len(modalities))

        for count, modality in enumerate(modalities):
            if mask_in is not None and modality != "modis":
                masks = [mask_in[i][modality] for i in range(len(mask_in))]
            else:
                masks = None

            token = x[modality]
            size = token.shape[-1] // max(int(scale * 10 / input_res[modality]), 1)

            if "_".join([modality, "dates"]) in list(x.keys()):
                dates = x["_".join([modality, "dates"])]
            else:
                dates = torch.zeros(
                    x[modality].shape[0], 1, device=x[modality].device
                ).int()
                token = token.unsqueeze(1)

            out = self.spatial_encoder(
                token,
                modality,
                wavelengths[modality],
                scale,
                dates,
                subpatches[modality],
                input_res[modality],
                masks,
            )
            if out["N_masked"] == 0:
                continue

            if modality == "modis":
                token = out["tokens"].view(B, 1, self.embed_dim)
                tokens.append(token)
                token_sizes.append(size)  # <-- move token_sizes.append here
                kept_modalities.append(modality)
                coords_in.append(get_coords(token, 1, 1, 0))
            else:
                N = out["N_masked"]
                token = out["tokens"].view(B, N, self.embed_dim)
                if dataset != "":
                    intermediate_tokens[f"tokens_{modality}"] = getattr(
                        self, f"projector__{dataset}_{modality}"
                    )(token)
                else:
                    intermediate_tokens[f"tokens_{modality}"] = token

                if proba_drop[count] < self.proba_drop_modalities:
                    continue

                tokens.append(token)
                token_sizes.append(size)
                kept_modalities.append(modality)
                sub = out["spatial"].view(B, N, -1, self.embed_dim)
                B, N, S, D = sub.shape
                S = int(S ** (1 / 2))
                full_size = size * S
                sub = sub.flatten(1, 2)

                coord_token = get_coords(token, size, 1, 0, 1)
                coord_spatial = (
                    get_coords(sub, full_size, 1, 0, 1)
                    .reshape(B, size, S, size, S, 2)
                    .permute(0, 1, 3, 2, 4, 5)
                    .reshape(B, size**2, S**2, 2)
                )

                if masks is not None:
                    coord_token = [
                        masks[i].apply(coord_token, axis="S", current_shape="BSX")
                        for i in range(len(masks))
                    ]
                    coord_token = torch.cat(coord_token, dim=0)
                    coord_spatial = [
                        masks[i].apply(coord_spatial, axis="S", current_shape="BSXX")
                        for i in range(len(masks))
                    ]
                    coord_spatial = torch.cat(coord_spatial, dim=0)

                coords_in.append(coord_token)
                spatial.append(sub)
                coords_spatial.append(coord_spatial.flatten(1, 2))

        return (
            tokens,
            coords_in,
            token_sizes,
            spatial,
            coords_spatial,
            intermediate_tokens,
            kept_modalities,
        )

    @torch.compile(dynamic=True)
    def ViT_forward(
        self,
        tokens,
        spatial,
        coords_spatial,
        coords_in,
        token_sizes,
        latent_grid,
        output_grid,
        modalities,
        keep_subpatch=False,
        mask_in: ModalityMaskCollection = None,
        mask_out: List[MaskSpatial] = None,
    ):
        """Run the transformer trunk over encoded modality tokens.

        The block sequence in ``self.block_type`` controls fusion, resizing,
        self-attention, and sparse cross-attention. ``mask_in`` masks input
        spatial tokens during pretraining; ``mask_out`` selects output spatial
        locations for masked reconstruction.
        """
        B = tokens[0].shape[0]
        n_modalities = len(spatial)

        if mask_in is not None:
            mask_in_spatial = extract_non_empty_spatial_masks(mask_in, modalities)
            mask_in_spatial_merged = merge_Mask(
                [mask_in_spatial[i] for i in range(len(mask_in_spatial))]
            )
        else:
            mask_in_spatial_merged = None

        spatial = torch.cat(spatial, dim=1).detach()
        coords_spatial = torch.cat(coords_spatial, dim=1)
        coords_in = torch.cat(coords_in, dim=1)

        coords = get_coords(tokens[0], int(latent_grid**0.5), 1, self.n_registers, 1)

        if mask_in is not None:
            coords_tocken, coords_registers = (
                coords[:, self.n_registers :],
                coords[:, : self.n_registers],
            )
            coords_tocken = mask_in_spatial_merged.apply(
                coords_tocken, axis="S", current_shape="BSX"
            )
            coords = torch.cat((coords_registers, coords_tocken), dim=1)

        coords_out = get_coords(
            tokens[0], int(output_grid**0.5), 1, self.n_registers, res=1
        )
        if mask_out is not None:
            coords_out_tocken = coords_out[:, self.n_registers :]

            coords_out_tocken = apply_spatial_masks(coords_out_tocken, mask_out)
            coords_out = torch.cat(
                (
                    coords_out[:, : self.n_registers].repeat(len(mask_out), 1, 1),
                    coords_out_tocken,
                ),
                dim=1,
            )

        tokens_in = torch.cat(tokens, dim=1)

        for i, blk in enumerate(self.blocks):
            if self.block_type[i] == "SA":
                tokens = blk(tokens, coords)
            elif self.block_type[i] == "QCA_In":
                tokens = tokens_in
                tokens = blk(
                    tokens,
                    coords_in,
                    n_registers=0,
                    n_modalities=n_modalities,
                    grid_size=latent_grid,
                )
            elif self.block_type[i] == "QCA_Out":
                tokens_latent = tokens
                tokens = blk(
                    tokens,
                    coords,
                    n_registers=self.n_registers,
                    n_modalities=1,
                    grid_size=output_grid,
                    mask_out=mask_out,
                )
                if mask_out is not None:
                    tokens = torch.cat(
                        [
                            tokens[:, : self.n_registers].repeat(len(mask_out), 1, 1),
                            apply_spatial_masks(
                                tokens[:, self.n_registers :], mask_out
                            ),
                        ],
                        dim=1,
                    )
                    spatial = spatial.repeat(len(mask_out), 1, 1)
                    coords_spatial = coords_spatial.repeat(len(mask_out), 1, 1)
            elif self.block_type[i] == "CA_Sub":
                tokens = blk(
                    tokens,
                    coords_out,
                    spatial,
                    coords_spatial,
                    n_registers=0,
                    n_registers_q=self.n_registers,
                    n_modalities=n_modalities,
                )
            elif self.block_type[i] == "CA_In":
                tokens = blk(
                    tokens,
                    coords,
                    tokens_in,
                    coords_in,
                    n_registers=0,
                    n_registers_q=self.n_registers,
                    n_modalities=n_modalities,
                )
            elif self.block_type[i] == "CA_Out":
                tokens = blk(
                    tokens,
                    coords_out,
                    tokens_latent,
                    coords,
                    n_registers=0,
                    n_registers_q=self.n_registers,
                    n_modalities=n_modalities,
                )
            elif self.block_type[i] == "Bilinear_out":
                tokens_latent = tokens
                registers, tokens = (
                    tokens[:, : self.n_registers],
                    tokens[:, self.n_registers :],
                )
                if mask_in is not None:  # revert mask
                    tokens = mask_in_spatial_merged.revert(
                        tokens,
                        axis="S",
                        current_shape="BSX",
                        target_S_length=int(latent_grid),
                    )
                if latent_grid != output_grid:
                    tokens = tokens.view(
                        B, int(latent_grid**0.5), int(latent_grid**0.5), self.embed_dim
                    ).permute(0, 3, 1, 2)  # B D H W
                    # bilinear out with normalized interpolation to ignore zero-filled positions
                    out_size = (int(output_grid**0.5), int(output_grid**0.5))
                    if mask_in is not None:
                        # build binary mask: 1 at visible positions, 0 at masked (zero-filled)
                        visibility = mask_in_spatial_merged.revert(
                            torch.ones(
                                B,
                                mask_in_spatial_merged.S.shape[1],
                                1,
                                device=tokens.device,
                                dtype=tokens.dtype,
                            ),
                            axis="S",
                            current_shape="BSX",
                            target_S_length=int(latent_grid),
                        )  # B, latent_grid, 1
                        visibility = visibility.view(
                            B, int(latent_grid**0.5), int(latent_grid**0.5), 1
                        ).permute(0, 3, 1, 2)  # B 1 H W
                        # interpolate values and mask, then normalize
                        tokens = torch.nn.functional.interpolate(
                            tokens, size=out_size, mode="bilinear", align_corners=True
                        )
                        visibility = torch.nn.functional.interpolate(
                            visibility,
                            size=out_size,
                            mode="bilinear",
                            align_corners=True,
                        )
                        tokens = tokens / visibility.clamp(min=1e-6)
                    else:
                        tokens = torch.nn.functional.interpolate(
                            tokens, size=out_size, mode="bilinear", align_corners=True
                        )
                    tokens = tokens.permute(0, 2, 3, 1).flatten(1, 2)  # B N D
                if mask_out is not None:
                    tokens = apply_spatial_masks(
                        tokens, mask_out, upsample_S_to=output_grid
                    )
                    spatial = spatial.repeat(len(mask_out), 1, 1)
                    coords_spatial = coords_spatial.repeat(len(mask_out), 1, 1)
                tokens = torch.cat((registers, tokens), dim=1)
            elif self.block_type[i] == "Bi_ACA_in":
                tokens_out = []
                for t, m, s in zip(tokens, modalities, token_sizes):
                    if s != int(latent_grid**0.5):
                        if mask_in is not None and m != "modis":
                            mask = merge_Mask(
                                [mask_in[i][m] for i in range(len(mask_in))]
                            )
                            t = mask.revert(
                                t, axis="S", current_shape="BSX", target_S_length=s**2
                            )
                        t = t.view(B, s, s, self.embed_dim).permute(
                            0, 3, 1, 2
                        )  # B D H W
                        # bilinear interpolation with visibility mask normalization
                        out_size = (int(latent_grid**0.5), int(latent_grid**0.5))
                        if mask_in is not None and m != "modis":
                            # build binary mask: 1 at visible positions, 0 at masked (zero-filled)
                            visibility = mask.revert(
                                torch.ones(
                                    B,
                                    mask.S.shape[1],
                                    1,
                                    device=t.device,
                                    dtype=t.dtype,
                                ),
                                axis="S",
                                current_shape="BSX",
                                target_S_length=s**2,
                            )  # B, s**2, 1
                            visibility = visibility.view(B, s, s, 1).permute(
                                0, 3, 1, 2
                            )  # B 1 H W
                            t = torch.nn.functional.interpolate(
                                t, size=out_size, mode="bilinear", align_corners=True
                            )
                            visibility = torch.nn.functional.interpolate(
                                visibility,
                                size=out_size,
                                mode="bilinear",
                                align_corners=True,
                            )
                            t = t / visibility.clamp(min=1e-6)
                        else:
                            t = torch.nn.functional.interpolate(
                                t, size=out_size, mode="bilinear", align_corners=True
                            )
                        t = t.permute(0, 2, 3, 1).flatten(1, 2)  # B N D
                        if mask_in is not None:
                            t = mask_in_spatial_merged.apply(
                                t,
                                axis="S",
                                current_shape="BSX",
                                upsample_S_to=latent_grid,
                            )
                    tokens_out.append(t)
                tokens = torch.stack(tokens_out, dim=2)
                B, N, M, D = tokens.shape
                tokens = tokens.flatten(0, 1)

                tokens = blk(tokens)
                tokens = tokens.view(B, N, D)

                if self.registers is not None:
                    registers = self.registers.expand(tokens.shape[0], -1, -1)
                    tokens = torch.cat((registers, tokens), dim=1)
            else:
                raise ValueError(
                    f"Unknown block type: {self.block_type[i]}. Block type must be one of 'SA', 'CA_In', 'CA_Out', 'CA_Sub', 'QCA_In', 'QCA_Out', 'Bilinear_out', or 'Bi_ACA_in'."
                )

        out = {}
        if keep_subpatch:
            out.update({"subpatches": spatial})

        return tokens, out

    def forward(
        self,
        x,
        wavelengths,
        input_res,
        scale,
        latent_grid,
        output_grid,
        subpatches,
        keep_subpatch=False,
        keep_intermediate=False,
        mask_in: ModalityMaskCollection = None,
        mask_out: List[MaskSpatial] = None,
        dataset: str = "",
    ):
        """Encode a batch of modality-keyed tensors.

        Args:
            x: ``{modality_name -> tensor}``. Spatial-only modalities use
                shape ``(B, C, H, W)``; time-series modalities additionally
                expect ``x[<mod>_dates]`` of shape ``(B, T)`` and use
                ``(B, T, C, H, W)`` for the data tensor.
            wavelengths: ``{modality_name -> list}`` of either floats
                (continuous wavelengths, embedded by MP-Fourier) or sensor
                codes such as ``"VV"`` / ``"VH"`` / ``"DSM"`` (looked up via
                learned per-channel embeddings in the spatial encoder).
            input_res: ``{modality_name -> meters/pixel}`` physical
                resolution. Used to align modalities of different GSDs on a
                common positional grid.
            scale: patch scale in units of 10 m.
            latent_grid: number of latent tokens after the trunk (a perfect
                square — e.g. ``16`` means a 4×4 grid).
            output_grid: number of output tokens. May differ from
                ``latent_grid``; ``Bilinear_out`` / ``CA_Sub`` blocks resize
                between the two.
            subpatches: ``{modality_name -> int}``. Per-modality sub-patch
                factor (1 means no sub-patching).
            keep_subpatch: when True, attach the per-modality sub-patch
                token grid to the returned ``out`` dict under ``"subpatches"``.
            keep_intermediate: when True, attach the post-projector
                per-modality tokens to ``out`` under
                ``f"tokens_{modality}"``. Used by SSL losses.
            mask_in: ``ModalityMaskCollection`` for input-side masking
                (MAE-style). ``None`` at inference.
            mask_out: list of ``MaskSpatial`` for output-side spatial
                masking. ``None`` at inference.
            dataset: dataset key used to select the per-(dataset, modality)
                projector head. Empty string ``""`` (the default and the
                hub-inference setting) skips projection — useful when no
                checkpoint-time dataset matches.

        Returns:
            Tuple ``(tokens, out)``:

            - ``tokens``: ``(B*, output_grid + n_registers, embed_dim)``
              latent tokens at the output grid (``B*`` is
              ``len(mask_in)*B`` if masking is active, else ``B``).
            - ``out``: dict with optional extras (``"subpatches"``,
              ``"tokens_<modality>"``) controlled by the flags above.
        """
        modalities = list(wavelengths.keys())
        modalities = [modality for modality in modalities if modality in x.keys()]

        (
            tokens,
            coords_in,
            token_sizes,
            spatial,
            coords_spatial,
            intermediate_tokens,
            kept_modalities,
        ) = self.UPE_forward(
            x, modalities, wavelengths, input_res, scale, subpatches, mask_in, dataset
        )
        tokens, out = self.ViT_forward(
            tokens,
            spatial,
            coords_spatial,
            coords_in,
            token_sizes,
            latent_grid,
            output_grid,
            kept_modalities,
            keep_subpatch,
            mask_in,
            mask_out,
        )

        if keep_intermediate:
            out.update(intermediate_tokens)

        return tokens, out
