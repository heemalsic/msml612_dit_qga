import os
import time
import csv
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
        tokens = h.view(B, C, H * W).permute(0, 2, 1)
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
# UNet with Class Conditioning
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
        emb = self.class_emb(y)[:, :, None, None]
        x = self.conv_in(x) + emb

        d1 = self.attn1(self.down1(x))
        d2 = self.attn2(self.down2(F.avg_pool2d(d1, 2)))

        mid = self.mid(d2)

        u1 = F.interpolate(mid, scale_factor=2, mode="nearest")
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.attn3(self.up1(u1))

        return self.conv_out(u1)


# ============================================================
# DDPM
# ============================================================

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.T = timesteps

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphabars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphabars", alphabars)

    def forward(self, x0, y):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        a_bar = self.alphabars[t][:, None, None, None]
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        pred = self.model(xt, y)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, device):
        x = torch.randn(n, 1, 28, 28, device=device)

        # ✅ FIX: labels ALWAYS match batch size
        y = torch.arange(n, device=device) % 10

        for t in reversed(range(self.T)):
            beta = self.betas[t]
            alpha = self.alphas[t]
            a_bar = self.alphabars[t]

            eps = self.model(x, y)
            z = torch.randn_like(x) if t > 0 else 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - a_bar) * eps
            ) + torch.sqrt(beta) * z

        return x


# ============================================================
# Batched Sampling (OOM-safe)
# ============================================================

@torch.no_grad()
def sample_in_batches(ddpm, total, batch_size, device):
    samples = []
    for i in range(0, total, batch_size):
        bs = min(batch_size, total - i)
        samples.append(ddpm.sample(bs, device).cpu())
        torch.cuda.empty_cache()
    return torch.cat(samples, dim=0)


# ============================================================
# FID (NO torchmetrics)
# ============================================================

@torch.no_grad()
def compute_fid(real, fake, device):
    inception = models.inception_v3(weights="DEFAULT", transform_input=False)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    def feats(x):
        x = x.to(device, non_blocking=True)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        f = inception(x)
        return f.detach().cpu().numpy()

    f1 = feats(real)
    f2 = feats(fake)

    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)

    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    del inception
    torch.cuda.empty_cache()

    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))


# ============================================================
# Training
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints/unet", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    model = AttentionUNet().to(device)
    ddpm = DDPM(model).to(device)
    opt = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    csv_path = "logs/unet_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "loss", "epoch_time_s", "step_time_ms", "peak_vram_mb", "imgs_per_sec", "fid"]
        )

    best_fid = float("inf")
    epochs = 30

    for epoch in range(epochs):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        losses, step_times = [], []
        ddpm.train()

        for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
            t0 = time.time()
            x, y = x.to(device), y.to(device)

            loss = ddpm(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            step_times.append(time.time() - t0)

        epoch_time = time.time() - start
        imgs_per_sec = len(dataset) / epoch_time
        step_ms = np.mean(step_times) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0

        ddpm.eval()
        fake = sample_in_batches(ddpm, total=100, batch_size=25, device=device)
        real = next(iter(DataLoader(dataset, batch_size=100)))[0]

        fid = compute_fid(real, fake, device)

        save_image(fake[:100], f"samples/epoch_{epoch}.png", nrow=10, normalize=True)

        torch.save(ddpm.state_dict(), "checkpoints/unet/last.pt")
        if fid < best_fid:
            best_fid = fid
            torch.save(ddpm.state_dict(), "checkpoints/unet/best.pt")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, np.mean(losses), epoch_time, step_ms, peak_vram, imgs_per_sec, fid]
            )

        print(f"Epoch {epoch} | Loss {np.mean(losses):.4f} | FID {fid:.2f}")

if __name__ == "__main__":
    train()
