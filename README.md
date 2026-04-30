# Thesis Assyrian Relief

Codebase for the thesis project:

**Neo-Assyrian Relief Completion: A Deep Learning Approach for Artistic Style and Archaeological Reconstruction**

Current implemented focus:
- style classification
- relief-level aggregation
- centroid-based attribution
- retrieval analysis
- UMAP visualization

## Project structure

- `src/thesis_assyrian_relief/` — reusable package code
- `scripts/` — runnable entry-point scripts
- `configs/` — YAML configs
- `data/splits/` — split CSV files
- `outputs/` — checkpoints, metrics, plots

## Environment setup

Using `uv`:

```bash
uv sync
```

If needed, activate the environment:

```bash
source .venv/bin/activate
```

## Main Config

Current main config:
```bash
configs/style_dinov2.yaml
```

Update the dataset image root there to match your machine.

## Train

```bash
uv run python scripts/train_style.py \
  --config configs/style_dinov2.yaml
```

Example quick smoke run:
```bash
uv run python scripts/train_style.py \
  --config configs/style_dinov2.yaml \
  --num-epochs 1 \
  --batch-size 4 \
  --num-workers 0
```
## Evaluate

```bash
uv run python scripts/eval_style.py \
  --config configs/style_dinov2.yaml \
  --eval-split test
```
Example quick smoke run:

```bash
uv run python scripts/eval_style.py \
  --config configs/style_dinov2.yaml \
  --eval-split val \
  --batch-size 4 \
  --num-workers 0
```

## UMAP visualization


```bash
uv run python scripts/umap_style.py \
  --config configs/style_dinov2.yaml \
  --eval-split test
```
This saves:

- interactive HTML UMAP
- CSV with UMAP coordinates

## Retrieval analysis

```bash
uv run python scripts/retrieval_analysis.py \
  --config configs/style_dinov2.yaml \
  --eval-split test
```
This saves:
- retrieval metrics JSON
- top-1 retrieval CSV
- top-k inspection CSV
- failure CSV

## Data assumptions

The image-level CSV is expected to contain:

- `Relief_ID`
- `Authority`
- `split`
- `view_index`
- `suffix`

Images are resolved at runtime using:

```bash
<Relief_ID><filename_sep><view_index><suffix>
```
Example:

```bash
BM 124773-1.jpg
```

## Current best style model
Current strongest result so far:
- frozen DINOv2 backbone
- learned embedding/classification probe
- relief-level multi-view aggregation


## Notes
- Local CPU smoke tests are supported.
- Heavy training is intended to run on Google Colab or another GPU environment.
- Local GPU support may require matching PyTorch CUDA build and NVIDIA driver versions.