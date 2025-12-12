# sample.py
import argparse
import torch

from model import DiT
from diffusion import Diffusion
from utils import save_grid


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="./outputs/generated.png")
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=28)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_kv_heads", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

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
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    diff = Diffusion(timesteps=args.timesteps, device=device)

    # balanced labels
    rows = 10
    cols = args.num // rows

    assert rows * cols == args.num, "num must be divisible by 10 for row-wise grid"

    y = torch.arange(0, 10, device=device).repeat_interleave(cols)

    samples = diff.ddim_sample(model, (args.num, 1, args.image_size, args.image_size), y, steps=args.ddim_steps, eta=0.0)

    save_grid(samples, args.out, nrow=10)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
