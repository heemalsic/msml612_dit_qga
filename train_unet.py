import os
import time
import csv
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import linalg

# ============================================================
# Attention Block
# ============================================================

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=2):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        tokens = h.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + attn_out


# ============================================================
# Residual Block
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


# ============================================================
# UNet (Class-conditional)
# ============================================================

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_classes=10):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, base_channels)

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels)
        self.attn1 = AttentionBlock(base_channels)

        self.down2 = ResBlock(base_channels, base_channels * 2)
        self.attn2 = AttentionBlock(base_channels * 2)

        self.mid = ResBlock(base_channels * 2, base_channels * 2)

        self.up1 = ResBlock(base_channels * 3, base_channels)
        self.attn3 = AttentionBlock(base_channels)

        self.conv_out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x, y):
        emb = self.class_emb(y)[:, :, None, None]  # [B, C, 1, 1]
        x = self.conv_in(x) + emb

        d1 = self.attn1(self.down1(x))
        d2 = self.attn2(self.down2(F.avg_pool2d(d1, 2)))

        mid = self.mid(d2)

        u1 = F.interpolate(mid, scale_factor=2, mode="nearest")
        u1 = torch.cat([u1, d1], dim=1)  # [B, 2C + C] = 3C
        u1 = self.attn3(self.up1(u1))

        return self.conv_out(u1)


# ============================================================
# Diffusion (DDPM train + DDIM sample)
# ============================================================

class Diffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.T = timesteps

        # buffers start on CPU, but will move correctly with .to(device)
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphabars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphabars", alphabars)

    # ---- DDPM loss ----
    def forward(self, x0, y):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        a_bar = self.alphabars[t][:, None, None, None]  # now same device as t
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        pred = self.model(xt, y)
        return F.mse_loss(pred, noise)

    # ---- DDIM sampling ----
    @torch.no_grad()
    def ddim_sample(self, shape, y, steps=50, eta=0.0):
        device = y.device
        times = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(shape, device=device)

        for i in range(len(times) - 1):
            t = times[i]
            t_next = times[i + 1]

            a_bar = self.alphabars[t]
            a_bar_next = self.alphabars[t_next]

            eps = self.model(x, y)
            x0 = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)

            sigma = eta * torch.sqrt(
                (1 - a_bar_next) / (1 - a_bar) * (1 - a_bar / a_bar_next)
            )

            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = (
                torch.sqrt(a_bar_next) * x0
                + torch.sqrt(1 - a_bar_next - sigma**2) * eps
                + sigma * noise
            )

        return x


# ============================================================
# FID (Inception-v3, no torchmetrics)
# ============================================================

@torch.no_grad()
def compute_fid(real_01, fake_01, device):
    inception = models.inception_v3(weights="DEFAULT", transform_input=False)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    def feats(x):
        x = x.to(device, non_blocking=True)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        f = inception(x)
        return f.detach().cpu().numpy()

    f1 = feats(real_01)
    f2 = feats(fake_01)

    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)

    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))


def to_01(x_m11):
    return ((x_m11 + 1.0) * 0.5).clamp(0.0, 1.0)


# ============================================================
# Training
# ============================================================

def train():
    # ---- match your DiT defaults ----
    epochs = 30
    batch_size = 256
    lr = 2e-4
    timesteps = 1000
    ddim_steps = 50
    fid_n = 100
    fid_bs = 25
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- dirs like your DiT layout ----
    out_dir = "outputs/unet"
    ckpt_dir = "checkpoints/unet"
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    metrics_csv = os.path.join(log_dir, "train_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "epoch_time_s", "step_time_ms",
                        "peak_vram_mb", "imgs_per_sec", "fid"])

    # [-1, 1] like your DiT
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])

    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # model + diffusion
    model = AttentionUNet().to(device)
    diff = Diffusion(model, timesteps=timesteps).to(device)  # ✅ FIX: move buffers to device

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    best_fid = float("inf")

    # real batch loader for FID
    fid_real_loader = DataLoader(train_ds, batch_size=fid_n, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_t0 = time.time()
        step_times = []
        running = 0.0

        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            t0 = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss = diff(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            step_times.append(time.time() - t0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_t0
        epoch_loss = running / max(1, len(train_loader))

        total_imgs = len(train_loader) * batch_size
        imgs_per_sec = total_imgs / max(epoch_time, 1e-9)

        step_ms = (sum(step_times) / max(len(step_times), 1)) * 1000.0
        peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={epoch_loss:.5f} | "
            f"epoch_time={epoch_time:.2f}s | step_time={step_ms:.2f}ms | "
            f"peak_vram={peak_vram_mb:.0f}MB | imgs/s={imgs_per_sec:.1f}"
        )

        # ---- sample grid: row i is digit i ----
        model.eval()
        with torch.no_grad():
            y_grid = torch.arange(0, 10, device=device).repeat_interleave(10)[:100]  # 0..9 rows
            samples_100 = diff.ddim_sample((100, 1, 28, 28), y_grid, steps=ddim_steps, eta=0.0)
            save_image(samples_100, os.path.join(out_dir, f"samples_e{epoch+1}.png"),
                       nrow=10, normalize=True)

        # ---- FID: sample in mini-batches (balanced labels) ----
        with torch.no_grad():
            reps = math.ceil(fid_n / 10)
            labels = torch.arange(0, 10, device=device).repeat(reps)[:fid_n]

            fake_chunks = []
            for i in range(0, fid_n, fid_bs):
                yb = labels[i:i + fid_bs]
                xb = diff.ddim_sample((len(yb), 1, 28, 28), yb, steps=ddim_steps, eta=0.0)
                fake_chunks.append(xb.detach().cpu())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            fake_m11 = torch.cat(fake_chunks, dim=0)  # CPU [-1,1]
            real_m11, _ = next(iter(fid_real_loader))  # CPU [-1,1]

            fid = compute_fid(to_01(real_m11), to_01(fake_m11), device)

        # ---- checkpoints ----
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        if fid < best_fid:
            best_fid = fid
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

        # ---- log ----
        with open(metrics_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch + 1, epoch_loss, epoch_time, step_ms, peak_vram_mb, imgs_per_sec, fid])

        print(f"Epoch {epoch+1} | FID={fid:.2f} | best_FID={best_fid:.2f}")

        # cleanup
        del samples_100, fake_m11, real_m11
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    train()
