import datetime
import math

import einops
import numpy as np
import PIL
import torch
import wandb
from lightning import LightningModule

from src import utils
from models.networks.masking import Masker, MaskSpatial

log = utils.get_pylogger(__name__)


class ModuleMultiMAE(LightningModule):
    """SSL Lightning module driving the UniverSat masked-modeling pretraining loop.

    For each batch:

    1. ``masker`` produces ``mask_in`` (visible context) and ``mask_out``
       (predicted) masks drawn from the configured distribution
       (modalities / time / channels / spatial cropping).
    2. ``self.model`` encodes the visible context tokens and its predictor
       predicts the masked target tokens.
    3. Reconstruction targets are the raw inputs patchified at the masked
       positions (:func:`patchify_and_apply_mask`).
    4. ``loss`` composes the MAE-MLP (LM³) reconstruction loss and the
       contrastive MILNCE loss on the predictor outputs.

    Validation metrics are computed here; linear-probe evaluation is handled
    by ``LinearProbeEvalCallback``. Used by ``configs/model/UniverSat.yaml``
    (the final UniverSat pretraining recipe).
    """

    def __init__(self,
                 network,
                 loss,
                 val_metrics,
                 test_metrics,
                 scheduler,
                 optimizer,
                 target_head_lr,
                 masker:Masker,
                 scales,
                 shapes,
                 warmup_epochs=0,
                 ):
        super().__init__()
        self.model = network.instance
        self.loss = loss
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.optimizer = optimizer
        self.target_head_lr = target_head_lr
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

        self.mask_collator = {}
        for dataset in scales.keys():
            self.mask_collator[dataset] = masker(input_size=(shapes[dataset], shapes[dataset]))

        self.step_time = []

    def forward(self, x):
        x.pop('logging', None)  # Remove weight information from the batch if present

        mask_enc, mask_pred = self.mask_collator[x['dataset']](x)

        prediction = self.model(x, mask_enc, mask_pred)

        if "classif" in self.loss.loss.keys():
            mask_enc2, mask_pred2 = self.mask_collator[x['dataset']](x)

            prediction["encoder_tokens2"] = self.model(x, mask_enc2, mask_pred2)["encoder_tokens"]

        assert len(mask_pred) == 1, "Only one mask_pred supported for now"

        target = {"scale": x['latent_scale']}
        for modality in self.model.wavelengths.keys():
            if modality in x.keys() and modality != "modis":
                target[modality] = patchify_and_apply_mask(x[modality], mask_pred[0])

        return prediction, {"target": target}, mask_enc, mask_pred

    def training_step(self, batch, batch_idx):
        t_start = datetime.datetime.now()
        to_log = batch.get("logging", {})
        pred, target, mask_enc, mask_pred = self.forward(batch)
        batch.update(target)
        batch['mask_enc'], batch['mask_pred'] = mask_enc, mask_pred

        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        if "target_logits" in loss.keys():
            target_logits = loss.pop("target_logits")
            batch["target_logits"] = target_logits

        for metric_name, metric_value in loss.items():
            self.log(
                f"train/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )
        for metric_name, metric_value in to_log.items():
            self.log(
                f"{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
            )

        self.step_time.append((datetime.datetime.now() - t_start).total_seconds())
        return loss

    def on_train_epoch_start(self):
        self.step_time = []
        return super().on_train_epoch_start()

    def on_train_epoch_end(self):
        log.info("Training ended.")
        log.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB/ reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        log.info(f"Average step time: {np.median(self.step_time):.2f} seconds")
        return super().on_train_epoch_end()

    def on_validation_start(self):
        log.info("Starting validation epoch, resetting metrics...")
        torch.cuda.empty_cache()
        return super().on_validation_start()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pred, target, mask_enc, mask_pred = self.forward(batch)
        batch.update(target)
        batch['mask_enc'], batch['mask_pred'] = mask_enc, mask_pred

        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        if "target_logits" in loss.keys():
            target_logits = loss.pop("target_logits")
            batch["target_logits"] = target_logits

        self.val_metrics.update(pred, batch)
        for metric_name, metric_value in loss.items():
            self.log(
                f"val/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        log.info("Finished validation epoch, computing metrics...")
        metrics = self.val_metrics.compute()

        # Log if not doing validation sanity steps
        if self.trainer.sanity_checking:
            return

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor) and metric_value.numel() > 1:
                #Not a scalar
                if "assignment" in metric_name:
                    #log with wandb as an histogram
                    value = metric_value.cpu().numpy()
                    epoch_column = np.full((value.shape[0]), self.current_epoch)
                    value = np.stack([epoch_column,np.arange(value.shape[0]), value], axis=-1)
                    table = wandb.Table(data=value.tolist(), columns=["epoch", "cluster_id", "cluster_assignment"])
                    if self.global_rank == 0:
                        wandb.log({f"val/{metric_name}": table, "epoch": self.current_epoch})
            elif isinstance(metric_value, PIL.Image.Image):
                #log with wandb as an image
                if self.global_rank == 0:
                    wandb.log({f"val/{metric_name}": wandb.Image(metric_value), "epoch": self.current_epoch})
            else:
                self.log(
                    f"val/{metric_name}",
                    metric_value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )
        self.val_metrics.reset()


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pred, target, mask_enc, mask_pred = self.forward(batch)
        batch.update(target)
        batch['mask_enc'], batch['mask_pred'] = mask_enc, mask_pred

        loss = self.loss(pred, batch, average=True)
        if "logits" in loss.keys():
            loss.pop("logits")
        if "target_logits" in loss.keys():
            target_logits = loss.pop("target_logits")
            batch["target_logits"] = target_logits

        self.test_metrics.update(pred, batch)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.test_metrics.reset()
        for metric_name, metric_value in metrics.items():
            self.log(
                f"test/{metric_name}",
                metric_value,
                sync_dist=True,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        parameters_list = [
                {'params': self.model.parameters()},
                {'params': self.loss.parameters(), 'lr': self.target_head_lr},
        ]
        optimizer = self.optimizer(params=parameters_list)
        if self.scheduler is not None:
            import functools
            scheduler_cls = self.scheduler.func if isinstance(self.scheduler, functools.partial) else self.scheduler
            if scheduler_cls.__name__ == "CosineAnnealingLR":
                warmup_epochs = int(self.warmup_epochs or 0)
                if warmup_epochs > 0:
                    num_epochs = int(self.scheduler.keywords.get("T_max", warmup_epochs + 1)) \
                        if isinstance(self.scheduler, functools.partial) else warmup_epochs + 1
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[
                            torch.optim.lr_scheduler.LinearLR(
                                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
                            ),
                            self.scheduler(optimizer=optimizer, T_max=max(num_epochs - warmup_epochs, 1)),
                        ],
                        milestones=[warmup_epochs],
                    )
                else:
                    scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


def patchify_and_apply_mask(x: torch.Tensor, mask: MaskSpatial) -> torch.Tensor:
    """
    :param x: tensor of shape [B, T, C, H, W]
    :param mask: MaskSpatial object containing indices of patches in [N] to keep
    :return: tensor of shape [B, N_keep, T, C, P*P] where P is the patch size
    """
    if len(x.shape) ==4:
        x = x.unsqueeze(1)  # Add temporal dimension if missing
    B, T, C, H, W = x.shape
    assert math.sqrt(mask.S_length).is_integer(), 'Mask S_length must be a perfect square.'
    P = H // int(math.sqrt(mask.S_length))
    assert H % P == 0 and W % P == 0, 'Image dimensions must be divisible by the patch size.'
    grid_size_h = H // P
    grid_size_w = W // P
    x = einops.rearrange(x, 'B T C (gh ph) (gw pw) -> B (gh gw) T C (ph pw)', gh=grid_size_h, gw=grid_size_w, ph=P, pw=P)
    x = mask.apply(x, axis='S', current_shape='BSTCD')  # Apply mask
    return x

