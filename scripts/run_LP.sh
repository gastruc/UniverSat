#!/usr/bin/env bash
# Linear-probe evaluations with the released UniverSat weights, pulled from the
# HuggingFace Hub (g-astruc/UniverSat). Run from the repo root:
#     ./scripts/run_LP.sh
#
#   LP_eval.py       -> all datasets
#   LP_eval_conv.py  -> EnMAP datasets only (separate output_dir so it doesn't
#                       overwrite the LP_eval results)
#
# All datasets use the standard standardizer except m-brick-kiln, and
# Sen1floods11 uses the SGD probe solver -- both handled on their own lines below.
#
# Any extra args are forwarded as Hydra overrides to every run, e.g.:
#     ./scripts/run_LP.sh ckpt_path=/path/to/local.ckpt   # use a local checkpoint
#     ./scripts/run_LP.sh hf_repo_id=other/repo device=cpu

# Disable torch.compile for the whole sweep.
export TORCH_COMPILE_DISABLE=1

# --- GeoBench ---
python src/LP_eval.py dataset/geobench_dataset=m-brick-kiln "$@"
for d in m-pv4ger m-forestnet m-chesapeake m-NeonTree; do
    python src/LP_eval.py dataset/geobench_dataset=$d param.standardization=standard "$@"
done

# --- Other benchmarks ---
for d in Ai4Farms BurnScars Mados PastisLP; do
    python src/LP_eval.py dataset=$d param.standardization=standard "$@"
done
python src/LP_eval.py dataset=Sen1floods11 param.standardization=standard param.segmentation.solver=sgd "$@"

# --- EnMAP (linear probe + conv probe) ---
for d in EnmapBdforet EnmapBnetd EnmapCdl EnmapCorine EnmapEurocrops EnmapNlcd EnmapTreemap; do
    python src/LP_eval.py      dataset=$d param.standardization=standard "$@"
    python src/LP_eval_conv.py dataset=$d param.standardization=standard param.segmentation.batch_size=8 param.segmentation.max_epochs=200 output_dir=LP_eval_conv "$@"
done
