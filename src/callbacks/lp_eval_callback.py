import gc
import os
from pathlib import Path
from typing import Any

import PIL
import torch
import wandb
from lightning import Callback, LightningModule, Trainer

from src import utils
from src.models.metrics.LP_SSL import (eval_model_FT, get_dataset_config,
                                       resolve_lp_eval_model)
from src.utils.slurm_submit import create_and_run_script

log = utils.get_pylogger(__name__)


class LinearProbeEvalCallback(Callback):
    def __init__(
        self,
        enabled: bool = True,
        eval_every: int = 5,
        submit_on_cluster: bool = True,
        wandb_group: str | None = None,
        slurm_account: str | None = None,
        slurm_gpu: str | None = None,
        slurm_qos: str = "t3",
        slurm_nodes: int = 1,
        slurm_num_gpus: int = 1,
        slurm_walltime: str = "2:00:00",
        slurm_env_path: str | None = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.eval_every = eval_every
        self.submit_on_cluster = submit_on_cluster
        self.wandb_group = wandb_group
        self.slurm_account = slurm_account
        self.slurm_gpu = slurm_gpu
        self.slurm_qos = slurm_qos
        self.slurm_nodes = slurm_nodes
        self.slurm_num_gpus = slurm_num_gpus
        self.slurm_walltime = slurm_walltime
        self.slurm_env_path = slurm_env_path
        self._dataset_config_cache: dict[str, list[dict[str, Any]]] = {}
        self._last_trigger_epoch: int | None = None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log.info("Finnished validation epoch, checking if we should run LP evaluation...")
        if not self.enabled or trainer.sanity_checking:
            return

        data_dir = getattr(pl_module, "path_to_data", None)
        if not self.eval_every or self.eval_every <= 0 or not data_dir:
            return

        if trainer.current_epoch % self.eval_every != 0:
            return

        if self._last_trigger_epoch == trainer.current_epoch:
            return
        self._last_trigger_epoch = trainer.current_epoch

        if self._should_submit_on_cluster():
            self._submit_cluster_eval(trainer)
            return

        self._run_local_eval(trainer, pl_module)

    def _run_local_eval(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log.info("Running LP evaluation locally")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dataset_configs = self._get_dataset_config(getattr(pl_module, "path_to_data"))
        world_size = trainer.world_size if trainer.world_size is not None else 1
        global_rank = trainer.global_rank if trainer.global_rank is not None else 0
        rank_dataset_config = [
            config for i, config in enumerate(dataset_configs) if i % world_size == global_rank
        ]

        eval_model = resolve_lp_eval_model(pl_module)
        metrics = eval_model_FT(rank_dataset_config, eval_model, centroid=None, device=pl_module.device, verbose=False)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            metrics_all = [None] * world_size
            torch.distributed.all_gather_object(object_list=metrics_all, obj=metrics)
        else:
            metrics_all = [metrics]

        merged_metrics = {}
        for rank_metrics in metrics_all:
            if rank_metrics:
                merged_metrics.update(rank_metrics)

        self._log_metrics(trainer, merged_metrics)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _submit_cluster_eval(self, trainer: Trainer) -> None:
        log.info("Running LP evaluation on cluster")
        checkpoint_path, config_path, output_dir = self._build_eval_paths(trainer)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        trainer.save_checkpoint(str(checkpoint_path), weights_only=True)
        if not trainer.is_global_zero:
            return

        run_file = Path(__file__).resolve().parents[1] / "lp_cluster_eval.py"
        group = self._get_wandb_group(trainer)
        project = self._get_wandb_project(trainer)
        run_name = self._get_wandb_name(trainer)
        run_id = self._get_wandb_run_id(trainer)

        current_name = (run_name or Path(output_dir).name).replace("/", "_").replace(" ", "_")
        job_name = f"{current_name}_eval_e{trainer.current_epoch:03d}"[:120]
        sbatch_dir = Path(output_dir) / "lp_eval" / "sbatch"
        cache_dir = Path(output_dir) / "lp_eval" / "logs"
        result_path = Path(output_dir) / "lp_eval" / f"epoch_{trainer.current_epoch:03d}_step_{trainer.global_step}_metrics.json"
        artifact_dir = Path(output_dir) / "lp_eval" / f"epoch_{trainer.current_epoch:03d}_step_{trainer.global_step}_artifacts"

        _, submit_output = create_and_run_script(
            sbatch_file_save_dir=str(sbatch_dir),
            cache_dir=str(cache_dir),
            work_dir=str(Path(__file__).resolve().parents[2]),
            job_name=job_name,
            run_file=str(run_file),
            account=self.slurm_account or os.getenv("ACCOUNT_NAME"),
            gpu=(self.slurm_gpu or os.getenv("GPU_TYPE") or os.getenv("SLURM_JOB_GRES", "h100")).split(":")[-1].lower(),
            qos=self.slurm_qos,
            nodes=self.slurm_nodes,
            num_gpus=self.slurm_num_gpus,
            walltime=self.slurm_walltime,
            env_path=self.slurm_env_path,
            use_srun=False,
            launch_sbatch=True,
            unset_env=True,
            additional_args={
                "--checkpoint-path": str(checkpoint_path),
                "--config-path": str(config_path),
                "--output-path": str(result_path),
                "--artifact-dir": str(artifact_dir),
                "--epoch": trainer.current_epoch,
                "--global-step": trainer.global_step,
                "--wandb-group": group or "",
                "--wandb-project": project or "",
                "--wandb-name": run_name or job_name,
                "--wandb-run-id": run_id or "",
                "--delete-checkpoint": True,
            },
        )
        log.info(f"Submitted LP eval job: {submit_output}")

    def _log_metrics(self, trainer: Trainer, metrics: dict[str, Any]) -> None:
        if not trainer.is_global_zero:
            return

        scalar_metrics = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor) and metric_value.numel() > 1:
                if "assignment" in metric_name and self._get_wandb_run(trainer) is not None:
                    value = metric_value.detach().cpu().numpy()
                    epoch_column = torch.full((value.shape[0],), trainer.current_epoch).numpy()
                    table_value = torch.tensor(value).cpu().numpy()
                    stacked = list(zip(epoch_column.tolist(), list(range(len(table_value))), table_value.tolist()))
                    wandb.log({f"val/{metric_name}": wandb.Table(data=stacked, columns=["epoch", "cluster_id", "cluster_assignment"]), "epoch": trainer.current_epoch}, step=trainer.global_step)
            elif isinstance(metric_value, PIL.Image.Image):
                if self._get_wandb_run(trainer) is not None:
                    wandb.log({f"val/{metric_name}": wandb.Image(metric_value), "epoch": trainer.current_epoch}, step=trainer.global_step)
            else:
                scalar_metrics[f"val/{metric_name}"] = float(metric_value)

        log.info(scalar_metrics)
        if scalar_metrics:
            scalar_metrics["epoch"] = trainer.current_epoch
            for logger in trainer.loggers or []:
                logger.log_metrics(scalar_metrics, step=trainer.global_step)

    def _get_dataset_config(self, data_dir: str) -> list[dict[str, Any]]:
        if data_dir not in self._dataset_config_cache:
            self._dataset_config_cache[data_dir] = get_dataset_config(data_dir)
        return self._dataset_config_cache[data_dir]

    def _build_eval_paths(self, trainer: Trainer) -> tuple[Path, Path, Path]:
        output_dir = Path(trainer.default_root_dir)
        checkpoint_path = output_dir / "lp_eval" / f"epoch_{trainer.current_epoch:03d}_step_{trainer.global_step}_weights.ckpt"
        config_path = output_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            config_path = Path.cwd() / ".hydra" / "config.yaml"
        return checkpoint_path, config_path, output_dir

    def _should_submit_on_cluster(self) -> bool:
        if not self.submit_on_cluster:
            return False
        return bool(os.getenv("SLURM_JOB_ID")) or os.getenv("IS_CLUSTER", "").lower() == "true"

    def _get_wandb_logger(self, trainer: Trainer):
        for logger in trainer.loggers or []:
            if logger.__class__.__name__ == "WandbLogger":
                return logger
        return None

    def _get_wandb_run(self, trainer: Trainer):
        logger = self._get_wandb_logger(trainer)
        if logger is None:
            return None
        try:
            return logger.experiment
        except Exception:
            return None

    def _get_wandb_group(self, trainer: Trainer) -> str | None:
        run = self._get_wandb_run(trainer)
        if run is not None and getattr(run, "group", None):
            return run.group
        logger = self._get_wandb_logger(trainer)
        if logger is not None and getattr(logger, "_wandb_init", None):
            return logger._wandb_init.get("group")
        return self.wandb_group

    def _get_wandb_project(self, trainer: Trainer) -> str | None:
        run = self._get_wandb_run(trainer)
        if run is not None and getattr(run, "project", None):
            return run.project
        logger = self._get_wandb_logger(trainer)
        if logger is not None and getattr(logger, "_wandb_init", None):
            return logger._wandb_init.get("project")
        return None

    def _get_wandb_name(self, trainer: Trainer) -> str | None:
        run = self._get_wandb_run(trainer)
        if run is not None and getattr(run, "name", None):
            return run.name
        logger = self._get_wandb_logger(trainer)
        return getattr(logger, "name", None) if logger is not None else None

    def _get_wandb_run_id(self, trainer: Trainer) -> str | None:
        run = self._get_wandb_run(trainer)
        if run is not None and getattr(run, "id", None):
            return run.id
        return None
