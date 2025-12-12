# train.py
import os
import argparse
import time
import csv
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import DiT
from diffusion import Diffusion
from utils import is_main_process, save_grid

# ---- FID deps (NO torchmetrics) ----
from scipy import linalg


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


@torch.no_grad()
def compute_fid_inception(real_01: torch.Tensor, fake_01: torch.Tensor, device: torch.device) -> float:
    """
    real_01, fake_01: [N, 1, H, W] in [0, 1]
    FID uses Inception-v3 features (2048-d).
    """
    inception = models.inception_v3(weights="DEFAULT", transform_input=False)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    def feats(x_01: torch.Tensor) -> np.ndarray:
        # grayscale -> 3ch, resize -> 299
        x = x_01.to(device, non_blocking=True)
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        f = inception(x)  # [N, 2048]
        return f.detach().cpu().numpy()

    f1 = feats(real_01)
    f2 = feats(fake_01)

    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)

    covmean = linalg.sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2.0 * covmean)

    del inception
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float(fid)


@torch.no_grad()
def sample_in_batches_ddim(diff: Diffusion, model, total: int, batch_size: int,
                           image_shape: tuple, device: torch.device,
                           steps: int, eta: float = 0.0) -> torch.Tensor:
    """
    Returns samples in [-1, 1] as [total, C, H, W].
    Uses Diffusion.ddim_sample in mini-batches to avoid OOM.
    """
    assert total > 0 and batch_size > 0
    C, H, W = image_shape

    all_samples = []
    # balanced labels 0..9
    reps = math.ceil(total / 10)
    labels = torch.arange(0, 10, device=device).repeat(reps)[:total]

    num_batches = math.ceil(total / batch_size)
    for i in range(num_batches):
        s = i * batch_size
        e = min((i + 1) * batch_size, total)
        bs = e - s
        y = labels[s:e]

        x = diff.ddim_sample(
            model,
            (bs, C, H, W),
            y,
            steps=steps,
            eta=eta
        )
        all_samples.append(x.detach().cpu())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(all_samples, dim=0)


def to_01(x_m11: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] -> [0, 1] and clamp."""
    return ((x_m11 + 1.0) * 0.5).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=28)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_kv_heads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    # FID controls (safe defaults to avoid OOM)
    ap.add_argument("--fid_n", type=int, default=100, help="How many samples to use for FID each epoch.")
    ap.add_argument("--fid_bs", type=int, default=25, help="Batch size for sampling during FID (avoid OOM).")
    args = ap.parse_args()

    ddp = ddp_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- dirs: put dit stuff in subfolders ----
    out_dir = os.path.join(args.out_dir, "dit")
    ckpt_dir = os.path.join(args.ckpt_dir, "dit")
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    metrics_csv = os.path.join(log_dir, "train_metrics.csv")
    if is_main_process() and (not os.path.exists(metrics_csv)):
        with open(metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "epoch_time_s", "step_time_ms",
                        "peak_vram_mb", "imgs_per_sec", "fid"])

    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1, 1]
    ])

    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=tfm)

    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = DiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_ch=1,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_classes=10,
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)

    diff = Diffusion(timesteps=args.timesteps, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    global_step = 0
    best_fid = float("inf")

    # real batch for FID (rank-0 only)
    fid_real_loader = None
    if is_main_process():
        fid_real_loader = DataLoader(train_ds, batch_size=args.fid_n, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(args.epochs):
        if ddp:
            sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_t0 = time.time()
        step_times = []
        running = 0.0

        model.train()
        for x, y in train_loader:
            t0 = time.time()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device)

            loss = diff.p_losses(model.module if ddp else model, x, t, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item()
            global_step += 1

            step_times.append(time.time() - t0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_t0

        # average epoch loss across processes
        epoch_loss = running / max(1, len(train_loader))
        if ddp:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / dist.get_world_size()

        # Throughput (global)
        world = dist.get_world_size() if ddp else 1
        total_imgs = len(train_loader) * args.batch_size * world
        imgs_per_sec = total_imgs / max(epoch_time, 1e-9)

        step_ms = (sum(step_times) / max(len(step_times), 1)) * 1000.0
        peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

        # ---- rank-0: sample + FID + save ----
        if is_main_process():
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"loss={epoch_loss:.5f} | "
                f"epoch_time={epoch_time:.2f}s | step_time={step_ms:.2f}ms | "
                f"peak_vram={peak_vram_mb:.0f}MB | imgs/s={imgs_per_sec:.1f}"
            )

            model_eval = model.module if ddp else model
            model_eval.eval()

            # quick sample (grid)
            y_s = torch.arange(0, 10, device=device).repeat_interleave(10)[:100]
            samples_100 = diff.ddim_sample(
                model_eval,
                (100, 1, args.image_size, args.image_size),
                y_s,
                steps=args.ddim_steps,
                eta=0.0
            )
            save_grid(samples_100, os.path.join(out_dir, f"samples_e{epoch+1}.png"), nrow=10)

            # FID sampling in batches to avoid OOM
            fake_m11 = sample_in_batches_ddim(
                diff=diff,
                model=model_eval,
                total=args.fid_n,
                batch_size=args.fid_bs,
                image_shape=(1, args.image_size, args.image_size),
                device=device,
                steps=args.ddim_steps,
                eta=0.0
            )  # CPU [-1,1]

            # real batch
            real_m11, _ = next(iter(fid_real_loader))  # CPU [-1,1]

            # map to [0,1]
            real_01 = to_01(real_m11)
            fake_01 = to_01(fake_m11)

            if device.type == "cuda":
                torch.cuda.empty_cache()

            fid = compute_fid_inception(real_01, fake_01, device)

            # checkpoints (best by FID, like before)
            torch.save(model_eval.state_dict(), os.path.join(ckpt_dir, "last.pt"))
            if fid < best_fid:
                best_fid = fid
                torch.save(model_eval.state_dict(), os.path.join(ckpt_dir, "best.pt"))

            # append CSV
            with open(metrics_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    epoch + 1,
                    epoch_loss,
                    epoch_time,
                    step_ms,
                    peak_vram_mb,
                    imgs_per_sec,
                    fid
                ])

            print(f"Epoch {epoch+1} | FID={fid:.2f} | best_FID={best_fid:.2f}")

            # cleanup
            del samples_100, fake_m11, real_m11, real_01, fake_01
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
