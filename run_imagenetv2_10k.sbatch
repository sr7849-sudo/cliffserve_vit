#!/bin/bash
#SBATCH --job-name=cliff10k
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/cliff10k_%j.out
#SBATCH --error=results/cliff10k_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

export IMAGENETV2="$PWD/data/imagenetv2/imagenetv2-matched-frequency-format-val"

mkdir -p results/imagenetv2_10k

echo "========================================"
echo "CliffServe ImageNetV2 10k"
echo "Job:  $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"

python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA runtime:", torch.version.cuda)
PY

echo
echo "Dataset images:"
find "$IMAGENETV2" -type f | wc -l
echo

python cliffserve.py accuracy-only \
    --config imagenetv2_10k.yaml \
    --dataset-root "$IMAGENETV2" \
    --outdir results/imagenetv2_10k

echo
echo "========================================"
echo "RUN COMPLETE"
echo "End: $(date)"
echo "========================================"
