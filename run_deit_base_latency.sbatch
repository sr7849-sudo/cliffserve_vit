#!/bin/bash
#SBATCH --job-name=deitB_lat
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/deitB_latency_%j.out
#SBATCH --error=results/deitB_latency_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

mkdir -p results/deit_base_latency

python cliffserve.py latency-only \
    --config deit_base_latency.yaml \
    --outdir results/deit_base_latency
