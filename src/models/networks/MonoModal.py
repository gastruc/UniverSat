import torch
from torch import nn

from models.networks.encoder.utils.utils import _init_weights


class MonoModalUniv2(nn.Module):
    """Supervised / linear-probe head on top of a :class:`UniverSat` encoder.

    Used by ``configs/exp/UniverSat_Pastis_FT.yaml`` and
    ``configs/exp/UniverSat_Pastis_LP.yaml``: the encoder is either trained
    from scratch (``load_path=None``) or initialized from an SSL checkpoint
    and optionally frozen (``freeze_backbone=True``). An ``MLPSemSeg`` head
    on top projects tokens to per-pixel logits.

    Args:
        encoder: a :class:`UniverSat` instance.
        mlp: hydra-instantiated MLP head config (its ``.instance`` is used).
        classif: if True, keep only the first register token (CLS-style)
            before the head — for classification rather than segmentation.
        wavelengths: ``{modality -> list}`` of wavelengths or sensor codes.
        input_res: ``{modality -> meters/pixel}`` physical resolution.
        scale: patch scale in units of 10 m.
        n_registers: number of register tokens prepended by the encoder.
        latent_grid: encoder latent grid (number of tokens).
        output_grid: encoder output grid (number of tokens).
        subpatches: ``{modality -> sub-patch factor}``.
        keep_subpatch: if True, the encoder returns sub-patch features and
            the head operates on them.
        load_path: optional Lightning checkpoint to initialize the encoder
            from. Only weights prefixed by ``target_encoder.`` are loaded
            (the EMA encoder from SSL training).
        freeze_backbone: freeze ``encoder`` parameters after loading.
    """

    def __init__(
        self,
        encoder,
        mlp,
        classif: bool = False,
        wavelengths: dict = {},
        input_res: dict = {},
        scale: int = 1,
        n_registers: int = 0,
        latent_grid: int = 1,
        output_grid: int = 1,
        subpatches: dict = {},
        keep_subpatch: bool = False,
        load_path: str = None,
        freeze_backbone: bool = False,
        ):
        super().__init__()
        self.encoder = encoder
        self.classif = classif
        self.mlp = mlp.instance
        self.wavelengths = wavelengths
        self.input_res = {k: torch.tensor(v) for k, v in input_res.items()}
        self.scale = scale
        self.n_registers = n_registers
        self.latent_grid = latent_grid
        self.output_grid = output_grid
        self.subpatches = subpatches
        self.keep_subpatch = keep_subpatch
        self.freeze_backbone = freeze_backbone

        if load_path is not None:
            print(f"Loading weights from {load_path}")
            state_dict = torch.load(load_path, map_location='cpu', weights_only=False)['state_dict']
            updated_state_dict = {}
            for k,v in state_dict.items():
                if 'target_encoder' in k:
                    k = k.split('target_encoder.')[1]
                    updated_state_dict[k] = v

            self.encoder.load_state_dict(updated_state_dict, strict=True)
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            assert self.freeze_backbone == False, "If you want to freeze the backbone, please provide a load_path"
            self.apply(_init_weights)

    def forward(self, x):
        """
        Forward pass of the network
        """
        out = self.encoder(x, self.wavelengths, self.input_res, self.scale, self.latent_grid,
                           self.output_grid, self.subpatches, self.keep_subpatch)
        if self.classif:
            out = out[:, 0]
        elif not(self.keep_subpatch):
            out = out[:, self.n_registers:]
        out = self.mlp(out)
        return out

