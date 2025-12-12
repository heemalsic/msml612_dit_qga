import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ============================================================
# Multi-Head Self-Attention Block (Standard MHA)
# ============================================================

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        tokens = x_norm.view(B, C, H * W).permute(0, 2, 1)
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
# Stable-Diffusion-Style U-Net with MHA
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
# DDPM Wrapper
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
        alpha_bar = self.alphabars[t][:, None, None, None]
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        noise_pred = self.model(xt, y)
        return F.mse_loss(noise_pred, noise)


# ============================================================
# Training Loop with PROFILING
# ============================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints/unet", exist_ok=True)

    model = DDPM(AttentionUNet()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    epochs = 30
    best_loss = float("inf")

    print("\nEpoch | Loss | Epoch Time (s) | Step Time (ms) | Peak VRAM (MB) | imgs/s")
    print("-" * 85)

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        step_times = []
        epoch_loss = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()

            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()

            step_times.append((t1 - t0) * 1000)
            epoch_loss += loss.item()

        epoch_time = time.perf_counter() - epoch_start
        avg_step = sum(step_times) / len(step_times)
        avg_loss = epoch_loss / len(step_times)

        peak_mem = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )

        imgs_per_sec = len(dataset) / epoch_time

        print(
            f"{epoch:>5} | {avg_loss:.4f} | {epoch_time:>13.2f} | "
            f"{avg_step:>14.2f} | {peak_mem:>14.0f} | {imgs_per_sec:>6.1f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/ddpm_unet_best.pt")

        

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
