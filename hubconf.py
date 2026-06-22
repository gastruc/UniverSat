"""Torch Hub entrypoint for UniverSat (AnySat v2).

Only the **Base** model is released, so this is the single size the hub
entrypoint builds.

Usage::

    # via the HuggingFace Hub (this counts towards the model's download metric)
    from hubconf import UniverSat
    model = UniverSat.from_pretrained("g-astruc/UniverSat").eval()

    # or via Torch Hub (same weights, same tracked download)
    import torch
    model = torch.hub.load("gastruc/UniverSat", "universat", pretrained=True).eval()

The model is exposed through ``huggingface_hub.PyTorchModelHubMixin`` so that
``from_pretrained`` / ``push_to_hub`` work and HF can track downloads (it counts
the ``config.json`` the mixin fetches via ``hf_hub_download``). Building an
*untrained* model needs only ``torch``; loading the pretrained weights also needs
``huggingface_hub``. Hydra, Lightning, einops and the rest of the training stack
are **not** imported here — the training code lives unchanged under ``src/``.

Forward signature (inference mode, ``mask_in=mask_out=None``)::

    features, extras = model(
        x,                # dict: modality name -> tensor
        wavelengths,      # dict: modality name -> list of wavelengths (or sensor codes)
        input_res,        # dict: modality name -> physical resolution (m)
        scale,            # int, patch scale in units of 10 m
        latent_grid,      # int, number of latent tokens (squared)
        output_grid,      # int, number of output tokens (squared)
        subpatches,       # dict: modality name -> sub-patch factor (default 1)
    )
"""

import os
import sys

from torch import nn

# UniverSat's encoder wraps two hot paths in ``@torch.compile`` to speed up
# large-scale training (see ``src/models/networks/encoder/UniverSat.py``). On the
# hub / inference path the one-off compile cost never amortises over a handful of
# forward passes, and varying input shapes can trip dynamo's recompile limit — so
# importing this entrypoint turns compilation into a no-op. This is a global,
# process-wide switch and is more robust than ``TORCH_COMPILE_DISABLE=1`` (which
# torch only reads at import). The training stack under ``src/`` never imports this
# module, so it keeps its ``torch.compile`` speedups.
import torch._dynamo

torch._dynamo.config.disable = True

dependencies = ["torch"]
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_BLOCK_ORDER = ["S1", "C", "T", "S"]
_DEFAULT_N_QUERIES = [1, 1, 1, 1]
_DEFAULT_EXPAND_DIM = [2, 2, 2, 2]

# (dataset, modality) pairs the model was pre-trained with. The constructor
# instantiates one MLP projector per non-``modis`` entry, so the set must
# match the checkpoint or ``load_state_dict`` will complain.
DEFAULT_MODALITIES_DICT = {
    "flair":      ["spotRGBN", "aerialflair", "s2flair", "s1flair", "dem"],
    "pastishd":   ["spot", "s2", "s1"],
    "planted":    ["s2", "s1", "l7", "alos", "modis"],
    "tsaits":     ["aerial", "s2", "s1"],
    "s2naip":     ["naip", "l8", "s2", "s1"],
    "hyperglobal": ["EO1"],
    "earthview":  ["rgbneon", "ndemneon", "neon"],
    "spectralearth": ["enmap"],
}

MODEL_CONFIGS = {
    "base": dict(
        embed_dim=768,
        num_heads=12,
        block_type=["Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"],
        gating=True,
        spatial_encoder_div=8,
    ),
}

# Released weights live on the HuggingFace Hub. The model is exposed via
# ``PyTorchModelHubMixin`` (below), so ``from_pretrained``/``push_to_hub`` use the
# HF-native ``config.json`` + ``model.safetensors`` layout and HF can count
# downloads. Produce the weights with ``scripts/clean_checkpoint.py`` and publish
# them with ``scripts/push_to_hub.py`` (which calls ``model.push_to_hub``).
_HF_REPO_ID = "g-astruc/UniverSat"

# Optional dependency — only needed to *load*/export pretrained weights. Building
# an untrained model works with torch alone, so the import is guarded.
try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # pragma: no cover - exercised only without huggingface_hub
    PyTorchModelHubMixin = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_model(size: str, modalities_dict, **overrides):
    """Instantiate ``UniverSat`` at the requested size."""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown size {size!r}. Available: {sorted(MODEL_CONFIGS)}.")
    cfg = {**MODEL_CONFIGS[size], **overrides}

    # Imports deferred so the module can be inspected without torch.
    from functools import partial

    from models.networks.encoder.UniverSat import UniverSat as _UniverSatModel
    from models.networks.encoder.UniversalPatchEncoder import UniversalPatchEncoder

    embed_dim = cfg["embed_dim"]
    spatial_embed = embed_dim // cfg["spatial_encoder_div"]
    # Passed as a partial so UniverSat finishes it with its (propagated) norm_layer.
    spatial_encoder = partial(
        UniversalPatchEncoder,
        embed_dim=spatial_embed,
        final_dim=embed_dim,
        n_queries=_DEFAULT_N_QUERIES,
        expand_dim=_DEFAULT_EXPAND_DIM,
        order=_DEFAULT_BLOCK_ORDER,
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        attn_drop_rate=0.0,
        gating=cfg["gating"],
    )
    model = _UniverSatModel(
        spatial_encoder=spatial_encoder,
        block_type=cfg["block_type"],
        embed_dim=embed_dim,
        num_heads=cfg["num_heads"],
        mlp_ratio=4.0,
        qkv_bias=False,
        n_registers=4,
        pre_norm=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        gating=cfg["gating"],
        proba_drop_modalities=0.0,  # inference default
        modalities_dict=modalities_dict,
    )
    _attach_encode(model)
    return model


# ---------------------------------------------------------------------------
# Convenience ``encode`` method — auto-fills modality metadata
# ---------------------------------------------------------------------------


def _attach_encode(model):
    """Bind a high-level :meth:`encode` method that looks up modality metadata.

    The bare :meth:`UniverSat.forward` requires the caller to provide a dict of
    wavelengths, physical resolutions, sub-patch factors, etc. for every
    modality. For all known sensors those values are static — this method
    fills them in from :data:`modality_registry.WAVELENGTHS`,
    :data:`modality_registry.INPUT_RES`, :data:`modality_registry.SUBPATCHES`,
    so callers only need to pass ``{modality_name: tensor}``::

        features = model.encode(
            {"s2": s2_batch, "s1": s1_batch, "s2_dates": s2_dates,
             "s1_dates": s1_dates},
            patch_size=40,           # 40 m patches (scale = 40 / 10 = 4)
            output_grid=4,           # 4×4 output grid (side G = 4)
        )

    Any key in the input dict that ends in ``_dates`` (or otherwise doesn't
    match a modality in the registry) is passed through unchanged — useful
    for time-series sensors that expect a ``<modality>_dates`` tensor.
    """
    import torch
    from modality_registry import INPUT_RES, SUBPATCHES, WAVELENGTHS

    def encode(
        x,
        patch_size: float = 40,
        latent_grid=None,
        output_grid=None,
        wavelengths=None,
        input_res=None,
        subpatches=None,
        **forward_kwargs,
    ):
        """Run :meth:`UniverSat.forward` with auto-filled modality metadata.

        Args:
            x: ``{modality_name -> tensor}``. Time-series modalities should
                also include ``<modality>_dates`` entries. Any key starting
                with ``modality_name`` + ``"_"`` is treated as auxiliary
                metadata for that modality and ignored by metadata lookup.
            patch_size: patch size in **metres**. Default 40 m. Converted to
                the encoder's 10 m units internally as ``scale = patch_size / 10``.
            latent_grid: number of latent tokens (perfect square). Inferred
                from the input shapes if omitted.
            output_grid: **side length** ``G`` of the desired ``G×G`` output
                feature map — ``G²`` tokens are requested from the encoder.
                Defaults to the **sub-patch resolution** side
                ``sqrt(latent_grid) × max_subpatch``, where ``max_subpatch`` is
                the largest sub-patch factor across the supplied modalities —
                the finest spatial output UniverSat can produce. Pass the latent
                side (``int(latent_grid ** 0.5)``) for patch-level features.
            wavelengths / input_res / subpatches: optional dict overrides
                for unknown sensors or to inject custom metadata. Anything
                set here wins over the registry lookup.
            **forward_kwargs: forwarded verbatim to
                :meth:`UniverSat.forward` (e.g. ``keep_subpatch=True``).

        Returns:
            Same as :meth:`UniverSat.forward`: ``(features, extras)``.
        """
        # ``patch_size`` is exposed in metres; the encoder works in units of 10 m.
        scale = patch_size / 10

        # Identify the *actual* modalities (skip auxiliary keys like "_dates").
        # We treat a key as a modality if it's in the registry or in the
        # user-supplied override dicts. Auxiliary keys (e.g. "s2_dates",
        # "s1_cloud_density") are left in ``x`` for the encoder to consume
        # but not used for metadata lookup.
        known = set(WAVELENGTHS) | set(wavelengths or ()) | set(input_res or ()) | set(subpatches or ())
        modalities = [k for k in x if k in known]
        if not modalities:
            raise ValueError(
                "encode(): no recognised modality keys found in x. "
                f"Got {list(x)}; known modalities: {sorted(known)}."
            )

        wl_out = {}
        res_out = {}
        sub_out = {}
        for name in modalities:
            user_wl = (wavelengths or {}).get(name)
            user_res = (input_res or {}).get(name)
            user_sub = (subpatches or {}).get(name)
            if name in WAVELENGTHS or user_wl is not None:
                wl_out[name] = list(user_wl) if user_wl is not None else list(WAVELENGTHS[name])
                res_out[name] = float(user_res) if user_res is not None else float(INPUT_RES[name])
                sub_out[name] = int(user_sub) if user_sub is not None else int(SUBPATCHES.get(name, 1))
            else:
                raise KeyError(
                    f"Modality {name!r} is not in the registry and no "
                    "wavelengths / input_res / subpatches override was "
                    "provided. Either add it to modality_registry.py or "
                    "pass the values explicitly."
                )

        # Default the latent grid from the input shapes when we can.
        if latent_grid is None:
            # Pick the first modality with a square H, W layout.
            for name in modalities:
                t = x[name]
                if t.ndim >= 4:
                    h = t.shape[-2]
                    res = res_out[name]
                    # number of patches along one side = h / (scale * 10 / res)
                    patch_px = max(int(scale * 10 / res), 1)
                    side = max(h // patch_px, 1)
                    latent_grid = side * side
                    break
            if latent_grid is None:
                raise ValueError("encode(): cannot infer latent_grid; pass it explicitly.")

        # ``output_grid`` is the *side* G of the requested square. Default to the
        # sub-patch-resolution side, then square it to a token count for forward.
        if output_grid is None:
            max_sub = max(sub_out.values()) if sub_out else 1
            output_side = int(round(latent_grid ** 0.5)) * max_sub
        else:
            output_side = output_grid
        output_grid_tokens = output_side ** 2

        return _strip_registers(
            model.forward(
                x,
                wavelengths=wl_out,
                input_res=res_out,
                scale=scale,
                latent_grid=latent_grid,
                output_grid=output_grid_tokens,
                subpatches=sub_out,
                **forward_kwargs,
            ),
            model.n_registers,
        )

    # Bind as a method; not an nn.Module so plain attribute assignment is fine.
    model.encode = encode


# ---------------------------------------------------------------------------
# HuggingFace-Hub-enabled model wrapper
# ---------------------------------------------------------------------------
#
# ``PyTorchModelHubMixin`` needs a class whose ``__init__`` takes only
# JSON-serialisable arguments (it stores them in ``config.json`` and replays them
# on ``from_pretrained``). The underlying ``UniverSat`` constructor takes a
# ``functools.partial`` spatial encoder, so we wrap it in a thin module whose
# config is just ``(size, modalities_dict)`` and delegate ``forward``/``encode``.


def _strip_registers(result, n_registers):
    """Drop the prepended register tokens from a ``(tokens, out)`` forward result.

    ``UniverSat.forward`` returns tokens of shape
    ``(B*, output_grid + n_registers, embed_dim)`` with the ``n_registers``
    register / "[CLS]"-like tokens prepended. Hub callers want the spatial
    output tokens only, so we slice them off before returning.
    """
    if not n_registers:
        return result
    tokens, out = result
    return tokens[:, n_registers:], out


class _UniverSatHub(nn.Module):
    """Thin wrapper around the built ``UniverSat`` with a serialisable config."""

    def __init__(self, size: str = "base", modalities_dict=None):
        super().__init__()
        self.size = size
        self.modalities_dict = modalities_dict or DEFAULT_MODALITIES_DICT
        # The released encoder lives at ``.model`` (so its state_dict keys are
        # ``model.*`` — what push_to_hub serialises and from_pretrained reloads).
        self.model = _build_model(size=size, modalities_dict=self.modalities_dict)

    def forward(self, *args, **kwargs):
        return _strip_registers(self.model(*args, **kwargs), self.model.n_registers)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


if PyTorchModelHubMixin is not None:
    class UniverSat(
        _UniverSatHub,
        PyTorchModelHubMixin,
        repo_url="https://github.com/gastruc/UniverSat",
        pipeline_tag="image-feature-extraction",
        tags=["image-feature-extraction", "remote-sensing", "earth-observation"],
        license="mit",
    ):
        """UniverSat (AnySat v2) — multimodal, multi-resolution EO encoder.

        Load the released weights with download tracking::

            model = UniverSat.from_pretrained("g-astruc/UniverSat").eval()

        Publish weights with :meth:`push_to_hub` (see ``scripts/push_to_hub.py``).
        """
else:  # huggingface_hub not installed: building still works, loading does not.
    UniverSat = _UniverSatHub


# ---------------------------------------------------------------------------
# Hub entrypoints
# ---------------------------------------------------------------------------


def universat(pretrained: bool = False, size: str = "base", modalities_dict=None, **kwargs):
    """UniverSat (AnySat v2) — multimodal, multi-resolution EO encoder.

    Args:
        pretrained: load the released weights from the HuggingFace Hub
            (``g-astruc/UniverSat``) via the mixin — this is the tracked path
            and requires ``huggingface_hub``.
        size: only ``"base"`` is supported — the single released model.
        modalities_dict: mapping of dataset -> list[modality] used to build the
            per-(dataset, modality) projectors. Defaults to the GeoPlex
            pretraining recipe — change only if your checkpoint used a different
            set. Ignored when ``pretrained=True`` (the published ``config.json``
            decides).
        **kwargs: forwarded to :meth:`UniverSat.from_pretrained` when
            ``pretrained`` (e.g. ``revision=``, ``cache_dir=``).
    """
    if pretrained:
        return from_pretrained(repo_id=_HF_REPO_ID, size=size, **kwargs)
    return UniverSat(size=size, modalities_dict=modalities_dict)


def from_pretrained(repo_id: str = _HF_REPO_ID, **kwargs):
    """Load released UniverSat weights from the HuggingFace Hub (tracked).

    Convenience for ``UniverSat.from_pretrained`` usable through Torch Hub::

        import torch
        model = torch.hub.load("gastruc/UniverSat", "from_pretrained").eval()

    Requires ``huggingface_hub`` (``pip install huggingface_hub``).
    """
    if PyTorchModelHubMixin is None:
        raise RuntimeError(
            "Loading pretrained UniverSat weights requires huggingface_hub. "
            "Install it (`pip install huggingface_hub`), or build an untrained "
            "model with universat(pretrained=False)."
        )
    return UniverSat.from_pretrained(repo_id, **kwargs)
