#!/bin/bash
#SBATCH --job-name=cliffvar
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/latency_variability_%j.out
#SBATCH --error=results/latency_variability_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

python repeat_latency.py
