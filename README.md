# CliffServe-ViT

CliffServe-ViT studies joint batch-size and token-pruning decisions for
deadline- and quality-constrained Vision Transformer inference on GPUs.

## Core idea

Token pruning does not provide uniform hardware benefit across batch sizes.
On real GPUs, pruning can be slower than dense inference at small batches but
become increasingly profitable at larger batches because of nonlinear
latency/workload behavior.

CliffServe-ViT profiles the hardware latency surface

\[
L_h(B,R)
\]

over batch size \(B\) and token-pruning amount \(R\), then selects a
quality-feasible operating point subject to deadline slack.

## Models

Current evaluation includes:

- DeiT-S
- DeiT-B

Token pruning is based on the POMT implementation, pinned in:

- `third_party/POMT_COMMIT.txt`
- `third_party/POMT_LOCAL_PATCH.diff`

The upstream POMT repository itself is not vendored in this repository.

## Dataset

ImageNetV2 matched-frequency, 10,000 images.

The dataset is intentionally excluded from this repository.

## Main experiments

### Hardware characterization

Batch sizes:

`1, 2, 4, 8, 16, 24, 32, 48, 64`

Representative pruning amounts:

`0, 48, 64, 80, 96, 112, 128, 144`

### Quality budgets

Accuracy-loss budgets:

- 0.5 percentage points
- 1.0 percentage point
- 2.0 percentage points

### Serving policies

- dense
- static pruning
- decoupled batch/pruning selection
- joint full search
- CliffServe

Workloads include Poisson and bursty ON/OFF request traces.

## Key current results

For DeiT-S on ImageNetV2:

- pruning is not latency-profitable at batches up to 8;
- under a 2 pp accuracy-loss budget, throughput improvement increases from
  13.7% at batch 16 to 28.6% at batch 64.

For DeiT-B:

- under a 1 pp accuracy-loss budget, quality-feasible pruning gives up to
  about 9% throughput improvement;
- under a 2 pp budget, gains reach about 31% for representative batch sizes.

See the `results/` directory for full tables and figures.

## Main files

- `cliffserve.py` — profiling, accuracy evaluation, surface construction, replay
- `final_scheduler.py` — optimized checkpointable scheduler evaluation
- `repeat_latency.py` — repeated latency variability experiment
- `plot_final_scheduler.py` — final scheduler plots

## Reproducibility

Large artifacts are intentionally excluded:

- ImageNetV2 dataset
- Python virtual environment
- raw request/batch traces
- latency-sample NPZ archives
- Slurm logs

Compact result summaries, configuration files, provenance information, and
figures are retained.
