# Diffusion Models on MNIST: UNet + MHA vs DiT + GQA

This repository provides a **controlled, implementation-level comparison** between two diffusion model denoisers trained on **MNIST** under aligned training and evaluation settings:

- **Baseline:** Stable-Diffusion–style **UNet** with spatial **Multi-Head Attention (MHA)** (`nn.MultiheadAttention`)
- **Proposed:** **Diffusion Transformer (DiT)** with **Grouped Query Attention (GQA)**

We compare:
- **Sample quality** (FID ↓)
- **GPU memory usage** (peak VRAM ↓)
- **Training speed** (throughput ↑, time/epoch ↓)

---

## Repository structure (actual)

```
.
├── train.py                 # Train DiT + GQA (DDP-capable) + sampling + FID
├── model.py                 # DiT architecture with GQA + AdaLN-style conditioning
├── diffusion.py             # DDPM training loss + DDIM sampling (used by DiT)
├── utils.py                 # DDP helpers + image grid saving
├── train_unet.py            # Train UNet + MHA baseline + sampling + FID
├── data/                    # MNIST dataset (auto-downloaded)
├── outputs/                 # Created during training
│   ├── dit/
│   │   └── samples_e*.png
│   └── unet/
│       └── samples_e*.png
└── checkpoints/             # Created during training
    ├── dit/
    │   ├── best.pt
    │   ├── last.pt
    │   └── logs/train_metrics.csv
    └── unet/
        ├── best.pt
        ├── last.pt
        └── logs/train_metrics.csv
```

---

## Common features (both pipelines)

- **DDPM training** with **1000 diffusion timesteps**
- **DDIM sampling** with **50 steps** (η = 0)
- **Class-conditional generation** (digits 0–9)
- **Per-epoch sampling grids** (100 samples, `nrow=10`)
- **FID computation using Inception-v3** (no `torchmetrics`)
- **Automatic CSV logging** each epoch:
  - loss
  - FID
  - peak GPU memory usage (MB)
  - epoch time / step time
  - throughput (images/sec)

---

## Installation

```bash
git clone https://github.com/heemalsic/msml612_dit_qga.git
cd msml612_dit_qga

pip install torch torchvision numpy scipy tqdm
```

CUDA GPU is recommended.

---

## Training: DiT + GQA (proposed)

```bash
python train.py
```

Outputs:
- samples: `outputs/dit/samples_e{epoch}.png`
- checkpoints: `checkpoints/dit/{last.pt,best.pt}`
- metrics: `checkpoints/dit/logs/train_metrics.csv`

### Key arguments (`train.py`)
- `--epochs` (default: 30)
- `--batch_size` (default: 256)
- `--lr` (default: 2e-4)
- `--timesteps` (default: 1000)
- `--ddim_steps` (default: 50)
- `--dim` (default: 512)
- `--depth` (default: 6)
- `--num_heads` (default: 8)
- `--num_kv_heads` (default: 2)
- `--fid_n` (default: 100)
- `--fid_bs` (default: 25)

### Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=2 train.py
```

---

## Training: UNet + MHA (baseline)

```bash
python train_unet.py
```

Outputs:
- samples: `outputs/unet/samples_e{epoch}.png`
- checkpoints: `checkpoints/unet/{last.pt,best.pt}`
- metrics: `checkpoints/unet/logs/train_metrics.csv`

UNet baseline highlights (see `train_unet.py`):
- convolutional residual blocks
- **spatial self-attention** via `nn.MultiheadAttention` over flattened HW tokens
- class embedding injected at the input feature map

---

## Logged metrics (CSV)

Both scripts produce:

`checkpoints/<dit|unet>/logs/train_metrics.csv`

Columns:
`epoch, loss, epoch_time_s, step_time_ms, peak_vram_mb, imgs_per_sec, fid`

These metrics allow direct comparison of:
- **quality** (FID ↓)
- **memory** (peak_vram_mb ↓)
- **speed** (imgs_per_sec ↑)

---

## Results (Tables 3–6)

### Table 3 — FID across training epochs (↓ better)

| Epoch | DiT (GQA) | UNet (MHA) |
|------:|----------:|-----------:|
| 15 | 58.5715 | 121.0554 |
| 20 | 62.6925 | 80.0348 |
| 25 | 67.0684 | 119.2162 |
| 30 | 57.1900 | 116.8445 |

### Table 4 — Peak GPU VRAM usage across epochs (MB; ↓ better)

| Epoch | DiT (GQA) | UNet (MHA) |
|------:|----------:|-----------:|
| 15 | 2006.990 | 8037.323 |
| 20 | 2006.990 | 8037.323 |
| 25 | 2006.990 | 8037.323 |
| 30 | 2006.990 | 8037.323 |

### Table 5 — Training time per epoch (s; ↓ better)

| Epoch | DiT (GQA) | UNet (MHA) |
|------:|----------:|-----------:|
| 15 | 104.4946 | 108.0396 |
| 20 | 105.4056 | 108.0608 |
| 25 | 104.9518 | 108.2394 |
| 30 | 105.2851 | 108.1569 |

### Table 6 — Training throughput (imgs/sec; ↑ better)

| Epoch | DiT (GQA) | UNet (MHA) |
|------:|----------:|-----------:|
| 15 | 573.2734 | 554.4628 |
| 20 | 568.3188 | 554.3545 |
| 25 | 570.7760 | 553.4394 |
| 30 | 568.9691 | 553.8615 |

---

## Key findings (summary)

Based on the tables above:

- **DiT + GQA achieves lower FID** than the UNet + MHA baseline across the reported epochs.
- **DiT + GQA uses substantially less peak VRAM** (≈ 2007 MB vs ≈ 8037 MB), i.e. ~4× lower peak allocated GPU memory.
- **DiT + GQA is slightly faster**: marginally lower time/epoch and higher throughput (imgs/sec).

---

## Novelty statement

This work explores integrating **Grouped Query Attention (GQA)** into a diffusion transformer denoiser and provides a **controlled empirical comparison** against a Stable-Diffusion–style UNet baseline with spatial MHA under aligned training and evaluation settings.

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022
- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023
- Shazeer, *Multi-Query Attention*, 2019
