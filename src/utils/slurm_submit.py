import os
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any


def _quote_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    return shlex.quote(str(value))


def _default_activate_path() -> str:
    if os.getenv("VENV"):
        return os.getenv("VENV")
    if os.getenv("VIRTUAL_ENV"):
        return str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "activate")
    if os.getenv("CONDA_PREFIX"):
        return os.environ["CONDA_PREFIX"]
    return ""


def _activation_command(env_path: str) -> str:
    if not env_path:
        return ""
    if Path(env_path).name == "activate":
        return f"source {env_path}"
    return f"conda activate {env_path} \nmicromamba activate {env_path}"


def _gpu_defaults(gpu: str) -> tuple[int, str, str, str]:
    gpu = gpu.lower()
    if gpu == "h100":
        return 24, f"#SBATCH --account={{account}}@{gpu}", f"#SBATCH --qos=qos_gpu_{gpu}-{{qos}}", f"#SBATCH -C {gpu}"
    if gpu == "a100":
        return 8, f"#SBATCH --account={{account}}@{gpu}", f"#SBATCH --qos=qos_gpu_{gpu}-{{qos}}", f"#SBATCH -C {gpu}"
    if gpu == "v100":
        return 10, f"#SBATCH --account={{account}}@{gpu}", "#SBATCH --qos=qos_gpu-{qos}", "#SBATCH -C v100-32g"
    if gpu == "mi300":
        return 24, "#SBATCH -A {account}", "", "#SBATCH --constraint=MI300"
    raise NotImplementedError(f"Unsupported gpu type: {gpu}")


def create_and_run_script(
    sbatch_file_save_dir: str = "artifacts",
    cache_dir: str = "artifacts/logs",
    work_dir: str | None = None,
    job_name: str = "lp_eval",
    run_file: str = "src/lp_cluster_eval.py",
    account: str | None = None,
    gpu: str = "h100",
    qos: str = "t3",
    nodes: int = 1,
    num_gpus: int = 1,
    walltime: str | None = None,
    env_path: str | None = None,
    job_array: bool = False,
    job_array_start_idx: int | None = None,
    num_jobs_in_array: int | None = None,
    use_srun: bool = False,
    relaunch: bool = False,
    launch_sbatch: bool = False,
    additional_args: dict[str, Any] | None = None,
    unset_env: bool = False,
):
    os.makedirs(sbatch_file_save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    gpu = gpu.lower()
    additional_args = additional_args or {}
    env_path = env_path or _default_activate_path()
    work_dir = work_dir or os.getcwd()

    num_cpus, account_directive, qos_directive, constraint_directive = _gpu_defaults(gpu)

    if walltime is None:
        if qos == "dev":
            walltime = "2:00:00"
        elif qos == "t3":
            walltime = "20:00:00"
        elif qos == "t4":
            walltime = "100:00:00"
        else:
            walltime = "20:00:00"

    account = account or os.getenv("ACCOUNT_NAME") or os.getenv("SLURM_ACCOUNT") or "qmd"
    array_directive = ""
    if job_array and job_array_start_idx is not None and num_jobs_in_array is not None:
        array_directive = f"#SBATCH --array={job_array_start_idx}-{job_array_start_idx + num_jobs_in_array - 1}"

    module_load = f"module load arch/{gpu}" if gpu in {"h100", "a100"} else ""
    extra_gpu_lines = "#SBATCH --exclusive" if gpu == "mi300" else ""
    qos_text = qos_directive.format(qos=qos) if qos_directive else ""
    account_text = account_directive.format(account=account)

    script = f"""\
#!/bin/bash
{account_text}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
{qos_text}
{constraint_directive}
#SBATCH --time={walltime}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --hint=nomultithread
{f'#SBATCH --error={cache_dir}/%a_%x_%j.err' if job_array else f'#SBATCH --error={cache_dir}/%x_%j.out'}
{f'#SBATCH --output={cache_dir}/%a_%x_%j.out' if job_array else f'#SBATCH --output={cache_dir}/%x_%j.out'}
{f'#SBATCH --signal=SIGUSR1@900' if relaunch else ''}
{f'#SBATCH --export=NONE' if unset_env else ''}
{array_directive}
{extra_gpu_lines}

{f'unset SLURM_EXPORT_ENV' if unset_env else ''}

[ -f ~/.bashrc ] && source ~/.bashrc
module purge

{module_load}
{_activation_command(env_path)}
cd {work_dir}

export HYDRA_FULL_ERROR=1
export WANDB_MODE=${{WANDB_MODE:-offline}}
export HF_HUB_OFFLINE=${{HF_HUB_OFFLINE:-1}}
export CUDA_LAUNCH_BLOCKING=${{CUDA_LAUNCH_BLOCKING:-1}}
export EXP_NAME={job_name}
export VENV={env_path}
export ACCOUNT_NAME={account}

set -x
    """
    script = textwrap.dedent(script)

    python_cmd = f"srun python {shlex.quote(run_file)}" if use_srun else f"python {shlex.quote(run_file)}"
    cli_args = []
    for key, value in additional_args.items():
        if key.startswith("--"):
            cli_args.append(f"{key}={_quote_cli_value(value)}")
        else:
            cli_args.append(f"{key}={_quote_cli_value(value)}")
    separator = " \\" + "\n\t"
    run_command = separator.join([python_cmd, *cli_args]) if cli_args else python_cmd
    script += run_command + "\n"

    script_path = Path(sbatch_file_save_dir) / f"{job_name}.sh"
    script_path.write_text(textwrap.dedent(script))

    if launch_sbatch:
        result = subprocess.check_output(["sbatch", str(script_path)], text=True).strip()
        return str(script_path), result
    return str(script_path), None
