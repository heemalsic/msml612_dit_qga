# train.py
import os
import argparse
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
    args = ap.parse_args()

    ddp = ddp_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

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
    os.makedirs(args.ckpt_dir, exist_ok=True)

    global_step = 0
    best_loss = 1e9

    for epoch in range(args.epochs):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        running = 0.0
        for x, y in train_loader:
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

        # average epoch loss across processes
        epoch_loss = running / max(1, len(train_loader))
        if ddp:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = (loss_tensor.item() / dist.get_world_size())

        if is_main_process():
            print(f"Epoch {epoch+1}/{args.epochs} | loss={epoch_loss:.5f}")

            # quick sample
            model_eval = model.module if ddp else model
            model_eval.eval()
            y_s = torch.arange(0, 10, device=device).repeat_interleave(10)[:100]
            samples = diff.ddim_sample(model_eval, (100, 1, args.image_size, args.image_size), y_s, steps=args.ddim_steps, eta=0.0)
            save_grid(samples, os.path.join(args.out_dir, f"samples_e{epoch+1}.png"), nrow=10)

            # save best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model_eval.state_dict(), os.path.join(args.ckpt_dir, "best.pt"))

            torch.save(model_eval.state_dict(), os.path.join(args.ckpt_dir, "last.pt"))

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
