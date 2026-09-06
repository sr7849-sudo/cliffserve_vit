#!/bin/bash
#SBATCH --job-name=deitB_sched
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/deitB_scheduler_%j.out
#SBATCH --error=results/deitB_scheduler_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

mkdir -p results/deit_base_scheduler

python final_scheduler.py \
  --config final_scheduler.yaml \
  --surface results/deit_base_final/quality_latency_surface.csv \
  --latency-samples results/deit_base_latency/latency_samples.npz \
  --outdir results/deit_base_scheduler
