# DiT (Diffusion Transformer) for MNIST — `msml612_dit_qga`

A compact PyTorch implementation of a **class-conditioned Diffusion Transformer (DiT)** trained on **MNIST**.  
Includes **DDIM sampling**, optional **multi-GPU DistributedDataParallel (DDP)** training, and an **FID** computation using **Inception-v3 features** (no `torchmetrics`).

## Repository contents (actual)

- `train.py` — main training script (single GPU/CPU or DDP), checkpointing, sample grid export, and FID evaluation each epoch.
- `model.py` — DiT model:
  - sinusoidal timestep embeddings
  - class embedding (0–9)
  - AdaLN(-Zero style) conditioning blocks
  - Grouped Query Attention (GQA)
- `diffusion.py` — diffusion utilities:
  - forward process sampling `q_sample`
  - training objective `p_losses` (predict noise with MSE)
  - DDIM sampler `ddim_sample`
- `utils.py` — helpers:
  - `is_main_process()` for rank-0 logging/saving
  - `save_grid()` for saving generated samples

Directories created during runs:
- `./data/` — MNIST download/cache (default)
- `./outputs/dit/` — sample grids saved as `samples_e{epoch}.png`
- `./checkpoints/dit/` — checkpoints (`last.pt`, `best.pt`) + `logs/train_metrics.csv`

---

## Setup

### Requirements
You’ll need Python 3 + the following packages (typical versions that work):
- `torch`, `torchvision`
- `numpy`
- `scipy` (used for `sqrtm` in FID)
  
Note: this repo does **not** include a `requirements.txt` currently; install manually, e.g.:

```bash
pip install torch torchvision numpy scipy
```

If you use CUDA, install the CUDA-enabled PyTorch build from the official PyTorch instructions.

---

## Train (single process)

```bash
python train.py
```

Common overrides:

```bash
python train.py \
  --epochs 30 \
  --batch_size 256 \
  --lr 2e-4 \
  --timesteps 1000 \
  --ddim_steps 50 \
  --dim 512 \
  --depth 6 \
  --num_heads 8 \
  --num_kv_heads 2
```

### What training does
Each epoch (rank 0 only):
- prints timing/throughput and loss
- generates a **10×10 grid** (100 samples; balanced labels 0–9 repeated)
- computes **FID** between a batch of real MNIST and generated samples
- saves checkpoints:
  - `checkpoints/dit/last.pt`
  - `checkpoints/dit/best.pt` (lowest FID so far)
- appends metrics to `checkpoints/dit/logs/train_metrics.csv`

---

## Multi-GPU (DDP)

This script enables DDP automatically if `RANK`/`WORLD_SIZE` env vars are set (i.e., when launched with `torchrun`).

Example (2 GPUs):

```bash
torchrun --nproc_per_node=2 train.py
```

Notes:
- Backend: `nccl` (so multi-GPU typically requires Linux + CUDA).
- Logging/saving happens on rank 0 only (`utils.is_main_process()`).

---

## Outputs

### Sample grids
Saved to:

```text
outputs/dit/samples_e{epoch}.png
```

### Checkpoints
Saved to:

```text
checkpoints/dit/last.pt
checkpoints/dit/best.pt
```

### Metrics CSV
Saved to:

```text
checkpoints/dit/logs/train_metrics.csv
```

Columns:
- `epoch`
- `loss`
- `epoch_time_s`
- `step_time_ms`
- `peak_vram_mb`
- `imgs_per_sec`
- `fid`

---

## FID evaluation details

FID is computed inside `train.py` using:
- `torchvision.models.inception_v3(weights="DEFAULT")` with the final FC replaced by identity to extract **2048-d features**
- grayscale MNIST samples are converted to 3-channel and resized to **299×299**
- covariance square root computed with `scipy.linalg.sqrtm`

Controls (to manage memory/time):
- `--fid_n` number of samples used for FID each epoch (default: `100`)
- `--fid_bs` batch size used when generating FID samples via DDIM (default: `25`)

Example:

```bash
python train.py --fid_n 500 --fid_bs 25
```

---

## Model overview (high level)

- **Backbone:** Transformer over image patches
- **Conditioning:** timestep embedding + class embedding combined, applied via **adaptive LayerNorm** (AdaLN) style blocks
- **Attention:** **Grouped Query Attention (GQA)** to reduce KV cost (`num_kv_heads <= num_heads`)
- **Objective:** predict noise `ε` with MSE (standard DDPM training)
- **Sampler:** DDIM (`Diffusion.ddim_sample`)

---

## Known limitations / mismatches to older README text

The previous README referenced files/folders like `configs/`, `run.py`, `inference.py`, and `evaluate.py`—these are **not present** in the current repository. This README reflects the current code layout (`train.py`, `model.py`, `diffusion.py`, `utils.py`).

---

## Citation / reference

DiT concept: *Scalable Diffusion Models with Transformers* (DiT). This repo is a small educational implementation for MNIST.
