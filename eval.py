# eval.py
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader

from model import DiT
from diffusion import Diffusion
from mnist_classifier import SmallMNISTClassifier


def inception_features(x, inception):
    """
    x: (B,1,28,28) in [-1,1]
    convert to (B,3,299,299) in [0,1] for inception
    """
    x = (x + 1) / 2  # [0,1]
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    # inception expects normalized roughly like ImageNet; this is approximate but common for FID codepaths
    x = (x - 0.5) / 0.5
    feats = inception(x)
    return feats


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID = ||mu1-mu2||^2 + Tr(sigma1+sigma2 - 2*sqrt(sigma1*sigma2))
    We use torch.linalg for stability; for MNIST this is typically fine.
    """
    diff = mu1 - mu2
    covmean = torch.linalg.sqrtm((sigma1 @ sigma2).cpu().double()).to(mu1.device).float()
    if torch.isfinite(covmean).all() is False:
        offset = torch.eye(sigma1.size(0), device=mu1.device) * eps
        covmean = torch.linalg.sqrtm(((sigma1 + offset) @ (sigma2 + offset)).cpu().double()).to(mu1.device).float()
    fid = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.item()


@torch.no_grad()
def compute_stats(loader, inception, device, max_items):
    feats_list = []
    n = 0
    for x, _ in loader:
        x = x.to(device)
        f = inception_features(x, inception)  # (B, 2048)
        feats_list.append(f)
        n += x.size(0)
        if n >= max_items:
            break
    feats = torch.cat(feats_list, dim=0)[:max_items]
    mu = feats.mean(dim=0)
    xc = feats - mu
    sigma = (xc.t() @ xc) / (feats.size(0) - 1)
    return mu, sigma


@torch.no_grad()
def classifier_score(samples, clf):
    """
    % classified as their conditioning label, averaged.
    """
    logits = clf(samples)
    pred = logits.argmax(dim=1)
    return pred


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--num_gen", type=int, default=10000)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--image_size", type=int, default=28)
    ap.add_argument("--patch_size", type=int, default=4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--num_kv_heads", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DiT
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

    # Real data loader
    tfm = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Inception (pool3 features via fc layer disabled)
    inception = inception_v3(weights="DEFAULT", transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    # Compute real stats (cap at num_gen for fairness)
    mu_r, sig_r = compute_stats(test_loader, inception, device, max_items=min(args.num_gen, len(test_ds)))

    # Generate samples and compute stats
    feats_list = []
    total = 0

    # Optional classifier score
    clf = SmallMNISTClassifier().to(device)
    # NOTE: This classifier is untrained unless you train it separately.
    # If you want a trained one, tell me and I’ll add a quick trainer + checkpoint.
    clf.eval()

    correct = 0
    counted = 0

    while total < args.num_gen:
        b = min(args.batch, args.num_gen - total)
        y = torch.arange(0, 10, device=device).repeat((b + 9) // 10)[:b]

        samples = diff.ddim_sample(model, (b, 1, args.image_size, args.image_size), y, steps=args.ddim_steps, eta=0.0)
        f = inception_features(samples, inception)
        feats_list.append(f)
        total += b

        # classifier score (meaningful only if clf trained)
        pred = clf(samples).argmax(dim=1)
        correct += (pred == y).sum().item()
        counted += b

    feats = torch.cat(feats_list, dim=0)[:args.num_gen]
    mu_g = feats.mean(dim=0)
    xc = feats - mu_g
    sig_g = (xc.t() @ xc) / (feats.size(0) - 1)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    print(f"FID (Inception-v3 features): {fid:.4f}")
    print(f"Classifier match (UNTRAINED clf placeholder): {correct / max(1, counted):.4f}")


if __name__ == "__main__":
    main()
