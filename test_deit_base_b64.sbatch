#!/bin/bash
#SBATCH --job-name=deitBtest
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/deitB_test_%j.out
#SBATCH --error=results/deitB_test_%j.err

set -euo pipefail

cd /scratch/sr7849/cliffserve_vit
source .venv/bin/activate

python - <<'PY'
import torch
import timm

device = "cuda:0"

model = timm.create_model(
    "deit_base_patch16_224",
    pretrained=True,
).eval().to(device)

x = torch.randn(
    64,
    3,
    224,
    224,
    device=device,
)

with torch.inference_mode():
    y = model(x)

torch.cuda.synchronize()

print("Output:", y.shape)

print(
    "Peak allocated GB:",
    torch.cuda.max_memory_allocated()
    / 1024**3
)

print("B=64 FITS")
PY
