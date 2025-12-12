# train.py
import os
import argparse
import time
import csv

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model import DiT
from diffusion import Diffusion
from utils import is_main_process, save_grid


def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)  # per-GPU batch in DDP
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
    args = ap.parse_args()

    ddp = ddp_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # -------------------------
    # CSV metrics logger (rank0)
    # -------------------------
    csv_file = None
    csv_writer = None
    metrics_path = os.path.join(args.ckpt_dir, "dit_gqa_training_metrics.csv")
    if is_main_process():
        csv_file = open(metrics_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch",
            "avg_loss",
            "epoch_time_sec",
            "avg_step_time_ms",
            "peak_vram_mb",
            "images_per_sec",
            "world_size",
            "batch_size_per_gpu",
        ])
        print("\nEpoch | Loss | Epoch Time (s) | Step Time (ms) | Peak VRAM (MB) | imgs/s")
        print("-" * 85)

    # -------------------------
    # Data
    # -------------------------
    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),  # [-1,1]
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

    # -------------------------
    # Model
    # -------------------------
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
        model = DDP(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=False
        )

    diff = Diffusion(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    best_loss = 1e9

    # For global throughput computation
    world_size = dist.get_world_size() if ddp else 1

    for epoch in range(args.epochs):
        if ddp:
            sampler.set_epoch(epoch)

        # Reset peak memory each epoch (so peak is per-epoch, not cumulative)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running = 0.0
        step_times_ms = []

        # Start epoch timer
        sync_cuda()
        epoch_start = time.perf_counter()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device)

            sync_cuda()
            t0 = time.perf_counter()

            loss = diff.p_losses(model.module if ddp else model, x, t, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sync_cuda()
            t1 = time.perf_counter()

            step_times_ms.append((t1 - t0) * 1000.0)
            running += loss.item()

        # End epoch timer
        sync_cuda()
        epoch_time = time.perf_counter() - epoch_start

        # Average epoch loss (local)
        epoch_loss = running / max(1, len(train_loader))

        # Average loss across processes
        if ddp:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = loss_tensor.item() / world_size

        avg_step_ms = sum(step_times_ms) / max(1, len(step_times_ms))

        # Peak VRAM (rank-local; good enough for same-hardware comparisons)
        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )

        # Throughput: global images/sec (all GPUs)
        # Each process runs len(train_loader) steps of batch_size images (drop_last=True).
        images_per_sec = (args.batch_size * len(train_loader) * world_size) / max(epoch_time, 1e-9)

        if is_main_process():
            print(
                f"{epoch+1:>5} | {epoch_loss:.4f} | {epoch_time:>13.2f} | "
                f"{avg_step_ms:>14.2f} | {peak_mem_mb:>14.0f} | {images_per_sec:>6.1f}"
            )

            # Save metrics row
            csv_writer.writerow([
                epoch + 1,
                epoch_loss,
                epoch_time,
                avg_step_ms,
                peak_mem_mb,
                images_per_sec,
                world_size,
                args.batch_size,
            ])
            csv_file.flush()

            # quick sample (DDIM) — runs only on rank0
            model_eval = model.module if ddp else model
            model_eval.eval()
            y_s = torch.arange(0, 10, device=device).repeat_interleave(10)[:100]
            samples = diff.ddim_sample(
                model_eval,
                (100, 1, args.image_size, args.image_size),
                y_s,
                steps=args.ddim_steps,
                eta=0.0
            )
            save_grid(samples, os.path.join(args.out_dir, f"samples_e{epoch+1}.png"), nrow=10)

            # Save best / last
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model_eval.state_dict(), os.path.join(args.ckpt_dir, "best.pt"))
            torch.save(model_eval.state_dict(), os.path.join(args.ckpt_dir, "last.pt"))

    if is_main_process() and csv_file is not None:
        csv_file.close()
        print(f"\nSaved metrics CSV: {metrics_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
