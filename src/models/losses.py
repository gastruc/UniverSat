import math

import einops
import torch
import torch.nn.functional as F
from torch import nn

from models.networks.encoder.utils.flexiVit import (FlexiViTLinear,
                                                    FlexiViTTemporel)
from utils.data import parse_scale
from models.networks.masking import Mask, apply_spatial_masks


def resize_labels(labels: torch.Tensor, mask_pred: Mask, nb_classes: int, ignore_index: int = -1, apply_mask: bool = True) -> torch.Tensor:
    """
    Resize labels to match the spatial dimensions of the mask and apply it.
    :param labels: tensor of shape (B, C, H, W)
    :param mask_pred: list of MaskSpatial objects
    :param nb_classes: number of classes for one-hot encoding
    :param ignore_index: index to ignore in the labels
    :param apply_mask: whether to apply the mask to the labels
    :return: resized labels tensor
    """
    if len(mask_pred) == 0:
        return labels

    S_length = mask_pred[0].S_length
    side_size = int(math.sqrt(S_length))
    pooling_factor = int(labels.shape[-1] / side_size)

    raise NotImplementedError("It was working, but there may be a bug in the way ignore labels are handled. Please check the implementation before using this function.")

    if ignore_index:
        nb_classes += 1
        mask = labels == ignore_index
        labels[mask] = nb_classes - 1

    labels = F.one_hot(labels.to(dtype=torch.long), num_classes=nb_classes).float()

    if ignore_index:
        labels = labels[..., :-1]  # Exclude the last channel if ignore_index is used

    if pooling_factor > 1:
        labels = einops.rearrange(labels, 'B H W C -> B C H W')
        labels = F.avg_pool2d(labels, kernel_size=pooling_factor, stride=pooling_factor)
        labels = einops.rearrange(labels, 'B C H W -> B H W C')

    labels = einops.rearrange(labels, 'B H W C -> B (H W) C')
    if apply_mask:
        labels = apply_spatial_masks(labels, mask_pred)

    if ignore_index:
        #normalize the labels across C dimension
        labels = labels / (labels.sum(dim=-1, keepdim=True) + 1e-8)

    return labels

class CrossEntropyWeighted(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyWeighted, self).__init__()
        self.weights = torch.ones(num_classes).float()
        self.weights[-1] = 0

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while having a 0 weight
        """
        self.weights = self.weights.to(x.device)
        return {"cross_entropy_loss": nn.functional.cross_entropy(x.flatten(2, 3), y["label"].flatten(1, 2).long(), weight=self.weights)}

class CrossEntropyIgnore(nn.Module):
    def __init__(self):
        super(CrossEntropyIgnore, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B]) while ignoring -1 index
        """
        if len(y["label"].shape) > 1:
            x = x.flatten(2, 3)
            label = y["label"].flatten(1, 2)
        else:
            label = y["label"]
        return {"cross_entropy_loss": nn.functional.cross_entropy(x, label.long(), ignore_index=-1)}

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        x_value = x
        y_value = y["label"]

        return {"cross_entropy_loss": nn.functional.cross_entropy(x_value, y_value)}

class MILNCE(nn.Module):
    """Multiple Instance Learning Noise Contrastive Estimation (MIL-NCE) loss.

    This loss function implements a contrastive learning approach that handles multiple modalities
    and patches within each modality. It computes similarities between different modality features
    while handling potential masks for valid/invalid patches. The modalities present in each batch
    are read from the ``tokens_<modality>`` keys at forward time.

    Args:
        tau (float, optional): Temperature parameter for scaling the logits. Defaults to 0.1.
            Lower values make the model more confident about its predictions.
    """

    def __init__(self, tau=0.1):
        super(MILNCE, self).__init__()
        self.tau = tau

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        x = input
        modalities = [item.split('_')[1] for item in list(x.keys()) if item.startswith('tokens_')]

        features = [x[f'tokens_{modality}'] for modality in modalities]
        n_patches = features[0].shape[1]
        n_tokens = n_patches * features[0].shape[0]
        features = torch.cat(features, dim=0).flatten(0, 1)

        # Compute similarity matrix
        logits = self.cosine_similarity(features, features, normalize=True)

        # Set diagonal blocks to -inf efficiently
        diag_mask = torch.block_diag(*[torch.ones(n_patches, n_patches) for _ in range(len(logits)//n_patches)])
        logits.masked_fill_(diag_mask.bool().to(logits.device), float('-inf'))

        # Handle masks if present
        masks = [item.split('_')[1] for item in list(x.keys()) if item.startswith('masks')]
        if masks:
            # Combine all masks efficiently
            mask = torch.cat([x[f'masks_{modality}'] for modality in modalities],
                           dim=1).flatten(0, 1).float()

            # Create mask matrix in one operation
            mask_matrix = mask.unsqueeze(-1) @ mask.unsqueeze(0)

            # Apply mask
            logits.masked_fill_(~mask_matrix.bool(), float('-inf'))
            valid_entries = mask_matrix.bool()

            # Compute loss only on valid entries
            loss = torch.logsumexp(logits[valid_entries].view(-1, valid_entries.sum(1).max()) / self.tau, dim=1).sum()
        else:
            loss = torch.logsumexp(logits / self.tau, dim=1).sum()

        if len(modalities) > 1:
            # Compute positive examples efficiently
            idx = torch.tensor([[i + j * n_tokens for j in range(len(modalities)) if j != k]
                            for k in range(len(modalities)) for i in range(n_tokens)],
                            device=logits.device)
            pos_logits = torch.gather(logits, 1, idx)


            if masks:
                valid_pos = pos_logits > float('-inf')
                pos_logits = pos_logits[valid_pos.any(dim=1)]

            loss += -torch.logsumexp(pos_logits / self.tau, dim=1).sum()
        else:
            loss = torch.zeros_like(loss)
        assert torch.isfinite(loss).all(), f"Loss contains NaN or Inf values. Loss: {loss}, n modalities: {len(modalities)}"
        loss = loss / len(features)
        return {
            "contrastive_loss": loss,
            "logits": logits
        }

def avg_pairwise_sim(q,k):
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)

    if len(q.shape) == 3 and len(k.shape) == 3:
        return torch.einsum('npd,nqd->npq', q, k).mean()
    else:
        return torch.einsum('nhpd,nhqd->nhpq', q, k).mean()


class MILNCEPatch(nn.Module):
    """Multiple Instance Learning Noise Contrastive Estimation (MIL-NCE) loss for patches.

    See Latent MIM paper : https://arxiv.org/pdf/2407.15837

    """

    def __init__(self, tau=0.2, sim_target=0.75, avg_sim_coeff=0.0):
        super(MILNCEPatch, self).__init__()
        self.tau = tau
        self.sim_target = sim_target
        self.avg_sim_coeff = avg_sim_coeff


    def forward(self, x, y):
        pred = x["predicted_tokens"]
        target = y["target"]

        bs, nt, d = pred.shape

        pred_magnitude = pred.norm(p=2, dim=-1).mean()
        pred_sim = avg_pairwise_sim(pred, pred)
        if "encoder_tokens" in x:
            visible_sim = avg_pairwise_sim(x["encoder_tokens"], x["encoder_tokens"])

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        scores = torch.einsum("npd,nqd->npq", pred, target) / self.tau
        labels = torch.arange(nt, dtype=torch.long, device=pred.device).unsqueeze(0).repeat(
                bs, 1
            )  # BxN
        loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1)) * (
            self.tau * 2
        )

        if self.avg_sim_coeff > 0:
            sim_reg_loss = self.avg_sim_coeff * (pred_sim-self.sim_target).pow(2)
            if "encoder_tokens" in x:
                sim_reg_loss += self.avg_sim_coeff * (visible_sim-self.sim_target).pow(2)
        else:
            sim_reg_loss = torch.zeros_like(loss)

        if "encoder_tokens" in x:
            return {"latentMIM_loss": loss, "pred_token_similarity": pred_sim, "visible_token_similarity": visible_sim, "pred_magnitude": pred_magnitude, "sim_reg_loss": sim_reg_loss}
        return {"latentMIM_loss": loss, "pred_token_similarity": pred_sim, "pred_magnitude": pred_magnitude, "sim_reg_loss": sim_reg_loss}


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN
            y: torch.Tensor BxN
        Returns:
            torch.Tensor: MSE loss between x and y: torch.Tensor([B, N])
        """
        return {"mse_loss": F.smooth_l1_loss(x['predicted_tokens'], y['target'])}

class MILNCEPatchSimple(nn.Module):
    """Simplified Multiple Instance Learning Noise Contrastive Estimation (MIL-NCE) loss for patches.

    See Latent MIM paper : https://arxiv.org/pdf/2407.15837

    """

    def __init__(self, tau=0.05):
        super(MILNCEPatchSimple, self).__init__()
        self.tau = tau

    def forward(self, x, y):
        """InfoNCE between predicted (``x``) and target (``y``) tokens, where each
        token's positive is the target at the same position. Both are
        instance-normalised then L2-normalised before the dot-product scores."""
        y = (y - y.mean(dim=(1,2), keepdim=True)) / (y.std(dim=(1,2), keepdim=True) + 1e-6)

        pred = x.flatten(0, 1)
        target = y.flatten(0, 1)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum("pd,qd->pq", pred, target) / self.tau
        labels = torch.arange(pred.shape[0], dtype=torch.long, device=pred.device)
        loss = F.cross_entropy(scores, labels)

        return loss
    
class MAEMLPLoss(nn.Module):
    """Latent masked-modeling loss (LM³).

    Per (modality, latent scale) the masked-patch target is projected into a
    target space and compared with the model's predicted tokens. Two target
    spaces are supported via ``space``: ``"pixel"`` (a learned MLP regresses the
    raw patch, trained with MSE) and ``"randomlinear"`` (a frozen random/Flexi
    linear projection, trained with an InfoNCE/MSE objective). Optionally selects
    low-cloud timesteps for optical time series before building the target.
    """

    def __init__(self, dim, wavelength, resolution, scales, modalities_dict=None, space="pixel", depth=2, use_contrastif=False, share_mlp=False, flexiVit=False, flexiVit_timestep=0, flexiVit_excluded_modalities=['s1', 'alos', 's1flair'], cloud_threshold=0.3, dim_expansion=1, tau=0.05):
        """
        Args:
            dim (int): token embedding dimension.
            wavelength (dict): modality -> list of channel descriptors (sets channel count).
            resolution (dict): modality -> physical resolution (m/pixel).
            scales (dict): per-dataset scale config (latent scales drive the projector set).
            modalities_dict (dict, optional): dataset -> modalities; defaults to all
                modalities in ``resolution`` for every dataset.
            space (str): target space, ``"pixel"`` or ``"randomlinear"``.
            depth (int): number of layers in the per-modality predictor MLP (0 = identity).
            use_contrastif (bool): use InfoNCE (``MILNCEPatchSimple``) instead of MSE;
                requires ``space="randomlinear"``.
            share_mlp (bool): share one predictor MLP across modalities (adds
                modality/scale/dataset embeddings); ``randomlinear`` only.
            flexiVit (bool|int): use FlexiViT projections; int caps the kernel size.
            flexiVit_timestep (int): if >0, use a temporal FlexiViT kernel with this many anchors.
            flexiVit_excluded_modalities (list): modalities that never use FlexiViT.
            cloud_threshold (float): cloud-density cutoff for timestep selection.
            dim_expansion (int): widening factor of the target/predictor output dim.
            tau (float): InfoNCE temperature (contrastive only).
        """
        super(MAEMLPLoss, self).__init__()
        assert space in ["pixel", "randomlinear"]
        self.space = space
        self.cloud_threshold = cloud_threshold
        self.use_contrastif = use_contrastif

        if not use_contrastif:
            self.loss = nn.MSELoss(reduction='mean')
        else:
            assert space == "randomlinear", "Contrastive loss only implemented for randomlinear space"
            self.loss = MILNCEPatchSimple(tau=tau)

        if space == "pixel":
            assert share_mlp == False, "Shared MLP only implemented for randomlinear space"
            assert flexiVit == False, "FlexiVit spatial only implemented for randomlinear space"
        self.share_mlp = share_mlp
        self.flexiVit = flexiVit
        self.flexiVit_timestep = flexiVit_timestep
        self.flexiVit_excluded_modalities = set(flexiVit_excluded_modalities or [])

        self.mlp_dict = nn.ModuleDict()
        self.linear_dict = nn.ModuleDict()

        datasets_l = scales.keys()

        if modalities_dict is None:
            modalities_dict = {dataset: resolution.keys() for dataset in datasets_l}

        if self.space == "pixel":
            print("Using pixel space for MAE MLP loss")
            # each modality and scale has its own MLP
            self.mlp_dict = self._generate_mlp_dict(dim, wavelength, resolution, scales, modalities_dict, depth, outsize='patch')
        elif self.space == "randomlinear":
            print("Using random linear space for MAE MLP loss")
            self.linear_dict = self._generate_randomlinear_dict(
                dim,
                wavelength,
                resolution,
                scales,
                modalities_dict,
                dim_expansion=dim_expansion,
                flexiVit=flexiVit,
                flexiVit_timestep=flexiVit_timestep,
                flexiVit_excluded_modalities=self.flexiVit_excluded_modalities,
            )
            if not share_mlp:
                self.mlp_dict = self._generate_mlp_dict(dim, wavelength, resolution, scales, modalities_dict, depth, outsize='dim', dim_expansion=dim_expansion)
            else:
                if depth > 0:
                    self.mlp = nn.Sequential(
                        *(f for d in range(depth-1) for f in (nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))),
                        nn.Linear(dim, dim*dim_expansion)
                    )
                else:
                    self.mlp = nn.Identity()
                self.modality_embedding_dict, self.scale_embedding_dict, self.dataset_embedding_dict = self._generate_emb_dicts(dim, datasets_l, scales, modalities_dict)

        print(f"all config : {self.mlp_dict.keys()}, {self.linear_dict.keys()}")

    def _use_flexivit_for_modality(self, modality):
        return bool(self.flexiVit) and modality not in self.flexiVit_excluded_modalities

    def _select_low_cloud_timesteps(self, target, cloud_density, dates=None):
        """
        Select timesteps with the lowest cloud density, enforcing a common
        sequence length across the batch.

        Args:
            target: Tensor with time dimension at dim=2 (B, N, T, ...)
            cloud_density: Tensor of shape (B, T)
            dates: Optional tensor with time dimension at dim=2 (B, N, T)

        Returns:
            target_sel: Tensor with time dimension reduced to K (B, N, K, ...)
            dates_sel: Optional tensor with time dimension reduced to K (B, N, K)
        """
        if cloud_density is None:
            return target, dates

        cloud_density = cloud_density.to(target.device)
        B, T = cloud_density.shape

        valid_mask = cloud_density < self.cloud_threshold
        valid_counts = valid_mask.sum(dim=1)
        min_count = int(valid_counts.min().item())
        k = max(1, min_count)

        if k >= T:
            return target, dates

        sel_idx = torch.topk(cloud_density, k=k, largest=False).indices  # (B, K)
        sel_idx_exp = sel_idx.view(B, 1, k, *([1] * (target.dim() - 3))).expand(
            B, target.shape[1], k, *target.shape[3:]
        )
        target_sel = target.gather(2, sel_idx_exp)

        if dates is None:
            return target_sel, None

        if dates.dim() == 2:
            dates_sel = dates.gather(1, sel_idx)
        else:
            dates_idx_exp = sel_idx.view(B, 1, k).expand(B, dates.shape[1], k)
            dates_sel = dates.gather(2, dates_idx_exp)
        return target_sel, dates_sel

    def _generate_mlp_dict(self, dim, wavelength, resolution, scales, modalities_dict, depth, outsize='patch', dim_expansion=1):
        assert outsize in ['patch', 'dim']
        mlp_dict = nn.ModuleDict()
        datasets_l = sorted(scales.keys())  # Sort for deterministic order across ranks
        for dataset in datasets_l:
            scales_l = sorted(parse_scale(scales[dataset])['latent_scales'])  # Sort for deterministic order
            modalities = sorted(modalities_dict[dataset])  # Sort for deterministic order
            for modality in modalities:
                for scale in scales_l:
                    scale_str = str(scale).replace('.', '_')
                    if outsize == 'patch':
                        C = len(wavelength[modality])
                        HW = int(10 * scale/resolution[modality])
                        out_dim = C*HW*HW
                    elif outsize == 'dim':
                        out_dim = dim*dim_expansion

                    if f"{modality}_{scale_str}" in mlp_dict:
                        continue

                    if depth > 0:
                        mlp = nn.Sequential(
                            *(f for d in range(depth-1) for f in (nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))),
                            nn.Linear(dim, out_dim)
                        )
                    else:
                        mlp = nn.Identity()
                    mlp_dict[f"{modality}_{scale_str}"] = mlp

        return mlp_dict

    def _generate_randomlinear_dict(self, dim, wavelength, resolution, scales, modalities_dict, dim_expansion=1, flexiVit=False, flexiVit_timestep=0, flexiVit_excluded_modalities=None):
        linear_dict = nn.ModuleDict()
        datasets_l = sorted(scales.keys())  # Sort for deterministic order across ranks
        flexiVit_excluded_modalities = set(flexiVit_excluded_modalities or [])

        modalities_l = set()
        scale_dict = {}
        for dataset in datasets_l:
            modalities = modalities_dict[dataset]
            for modality in modalities:
                if modality not in scale_dict:
                    scale_dict[modality] = set()
                modalities_l.add(modality)
                for scale in parse_scale(scales[dataset])['latent_scales']:
                    scale_dict[modality].add(scale)

        for modality in sorted(modalities_l):  # Sort for deterministic order across ranks
            scales_l = sorted(scale_dict[modality])  # Sort for deterministic order across ranks
            use_flexivit = bool(flexiVit) and modality not in flexiVit_excluded_modalities
            if use_flexivit:
                max_HW = int(10 * max(scales_l)/resolution[modality])

                if isinstance(flexiVit, int):
                    # limit max_HW to flexiVit_spatial
                    max_HW = min(max_HW, flexiVit)

                if flexiVit_timestep > 0:
                    flexiVit_linear = FlexiViTTemporel(max_HW, len(wavelength[modality]), dim_expansion*dim, flexiVit_timestep)
                    print(f"FlexiVit timestep linear for modality {modality} with max HW {max_HW}, timesteps {flexiVit_timestep} and parameters {sum(p.numel() for p in flexiVit_linear.parameters())}")
                else:
                    flexiVit_linear = FlexiViTLinear(max_HW, len(wavelength[modality]), dim_expansion*dim)
                    print(f"FlexiVit random linear for modality {modality} with max HW {max_HW} and parameters {sum(p.numel() for p in flexiVit_linear.parameters())}")

            for scale in scales_l:
                scale_str = str(scale).replace('.', '_')
                C = len(wavelength[modality])
                HW = int(10 * scale/resolution[modality])


                if f"{modality}_{scale_str}" in linear_dict:
                    continue
                if use_flexivit:
                    linear = flexiVit_linear
                else:
                    linear = nn.Linear(C*HW*HW, dim_expansion*dim, bias=False)
                    nn.init.orthogonal_(linear.weight)
                for param in linear.parameters():
                    param.requires_grad = False
                linear_dict[f"{modality}_{scale_str}"] = linear

        return linear_dict

    def _generate_emb_dicts(self, dim, datasets_l, scales, modalities_dict):
        modality_embedding_dict = nn.ParameterDict()
        scale_embedding_dict = nn.ParameterDict()
        dataset_embedding_dict = nn.ParameterDict()
        for dataset in sorted(datasets_l):  # Sort for deterministic order across ranks
            scales_l = sorted(parse_scale(scales[dataset])['latent_scales'])  # Sort for deterministic order
            modalities = sorted(modalities_dict[dataset])  # Sort for deterministic order
            dataset_embedding_dict[dataset] = nn.Parameter(torch.randn((1,1,dim)))
            for modality in modalities:
                if not modality in modality_embedding_dict:
                    modality_embedding_dict[modality] = nn.Parameter(torch.randn((1,1,dim)))
            for scale in scales_l:
                if not str(scale) in scale_embedding_dict:
                    scale_embedding_dict[str(scale)] = nn.Parameter(torch.randn((1,1,dim)))
        return modality_embedding_dict, scale_embedding_dict, dataset_embedding_dict

    def forward(self, x, y):
        """Compute the latent masked-modeling loss summed over modalities.

        ``x`` holds ``predicted_tokens`` (B, N, D), ``predicted_meta`` and the
        ``dataset`` key; ``y["target"]`` maps modality -> target tensor
        (B, N, T, C, H*W) plus the requested ``scale``. Returns the loss and, when
        the contrastive objective is active, per-batch retrieval accuracies.
        """
        total_loss = torch.tensor(0.0, device=x['predicted_tokens'].device)
        total_patch_acc = torch.tensor(0.0, device=x['predicted_tokens'].device)
        total_patch_acc_tie_aware = torch.tensor(0.0, device=x['predicted_tokens'].device)
        n_contrastive = 0
        patch_acc_per_modality = {}
        meta = x['predicted_meta']
        dataset = x['dataset']
        scale = y["target"]["scale"]

        if 'modality_list' in meta:
            modalities = sorted(meta['modality_list'])
            logits = x['predicted_tokens'].view(x['predicted_tokens'].shape[0], len(modalities), -1, x['predicted_tokens'].shape[-1]) #B,M,N,D
            logits = {modality: logits[:, i] for i, modality in enumerate(modalities)}
        else:
            modalities = y["target"].keys()
            logits = {modality: x['predicted_tokens'] for modality in modalities}

        for modality in modalities:
            key = f"{modality}_{str(scale).replace('.', '_')}"

            if modality + '_dates' in meta:
                assert not (self._use_flexivit_for_modality(modality) and self.flexiVit_timestep > 0), "Recovering specific dates not implemented with FlexiVit temporal"
                dates = meta[modality + '_dates'] #list of date for each sample in batch B,N
                dates = dates.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,*y["target"][modality].shape[-2:]) #B,N,T,c,HW
                target = y["target"][modality].gather(2, dates) #B,N,c,HW
                target = target.flatten(2)
            elif self._use_flexivit_for_modality(modality) and self.flexiVit_timestep > 0:
                target = y["target"][modality].flatten(3) #B,N,T,CHW
                dates = y.get(modality + '_dates', None) #B, N, T
                if f"{modality}_cloud_density" in y:
                    target, dates = self._select_low_cloud_timesteps(
                        target,
                        y[f"{modality}_cloud_density"],
                        dates,
                    )
            else:
                if f"{modality}_cloud_density" in y:
                    target = y["target"][modality].flatten(3)  # B,N,T,CHW
                    target, _ = self._select_low_cloud_timesteps(
                        target,
                        y[f"{modality}_cloud_density"],
                        None,
                    )
                    target = torch.median(target, dim=2).values  # B,N,CHW
                else:
                    target = torch.median(y["target"][modality], dim=2).values
                target = target.flatten(2) #B,N,CHW

            if self.space == "pixel":
                pred = self.mlp_dict[key](logits[modality])
                loss = self.loss(pred, target)

            elif self.space == "randomlinear":
                if self._use_flexivit_for_modality(modality) and self.flexiVit_timestep > 0:
                    target = self.linear_dict[key](target, dates)
                else:
                    target = self.linear_dict[key](target)

                if not self.share_mlp:
                    pred = self.mlp_dict[key](logits[modality])
                else:
                    pred = self.mlp(logits[modality] +
                                    self.modality_embedding_dict[modality] +
                                    self.scale_embedding_dict[str(scale)] +
                                    self.dataset_embedding_dict[dataset])

                loss = self.loss(pred, target)

            total_loss += loss

            if self.use_contrastif:
                with torch.no_grad():
                    pred_n   = F.normalize(pred, p=2, dim=-1)
                    target_n = F.normalize(target, p=2, dim=-1)
                    with torch.cuda.amp.autocast(enabled=False):
                        scores = torch.bmm(pred_n.to(torch.float32), target_n.transpose(1, 2).to(torch.float32))

                    labels   = torch.arange(scores.shape[1], device=scores.device) \
                                    .unsqueeze(0).expand(scores.shape[0], -1)
                    acc = (scores.flatten(0,1).argmax(dim=-1) == labels.flatten(0,1)).float().mean()
                    total_patch_acc += acc
                    
                    flat_scores = scores.flatten(0, 1)          # (B*N, N)
                    flat_labels = labels.flatten(0, 1)          # (B*N,)
                    
                    max_vals = flat_scores.max(dim=-1, keepdim=True).values
                    is_max   = (flat_scores == max_vals)                             # (B*N, N) bool
                    label_is_max    = is_max[torch.arange(len(flat_labels)), flat_labels]  # (B*N,) bool
                    acc_tie_aware   = label_is_max.float().mean()
                    total_patch_acc_tie_aware += acc_tie_aware
                    
                    n_contrastive += 1
                    patch_acc_per_modality[modality] = acc

        output = {"mae_mlp_loss": total_loss}
        if n_contrastive > 0:
            output["mae_patch_acc"] = total_patch_acc / n_contrastive
            output["mae_patch_acc_tie_aware"] = total_patch_acc_tie_aware / n_contrastive
            for modality, acc in patch_acc_per_modality.items():
                output[f"mae_patch_acc_{modality}"] = acc
        return output


AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, losses, weight):
        """Initializes the Losses object.
        Args:
            losses (dict): mapping ``loss_name -> loss module``.
            weight (dict): mapping ``loss_name -> scalar weight`` (same keys as ``losses``).
        """
        super(Losses, self).__init__()
        assert len(losses)
        assert losses.keys() == weight.keys()
        self.loss = nn.ModuleDict(losses)
        self.weight = weight

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            y: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        output = {"loss": 0}
        for loss_name, loss in self.loss.items():
            weight = self.weight[loss_name]
            if weight == 0:
                continue
            loss_output = loss(x, y)
            for k, v in loss_output.items():
                if k.endswith("_loss"):
                    v = AVERAGE[average](v)
                    output["loss"] += weight * v
                output[k] = v

        return output
