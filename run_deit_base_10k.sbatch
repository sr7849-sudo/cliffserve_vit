#!/bin/bash
#SBATCH --job-name=deitB10k
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=results/deitB_10k_%j.out
#SBATCH --error=results/deitB_10k_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

export IMAGENETV2="$PWD/data/imagenetv2/imagenetv2-matched-frequency-format-val"

mkdir -p results/deit_base_10k

python cliffserve.py accuracy-only \
    --config deit_base_10k.yaml \
    --dataset-root "$IMAGENETV2" \
    --outdir results/deit_base_10k
