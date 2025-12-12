# utils.py
import os
import torch
import torch.distributed as dist
from torchvision.utils import save_image


def setup_ddp():
    if dist.is_available() and dist.is_initialized():
        return True
    return False


def is_main_process():
    return (not setup_ddp()) or dist.get_rank() == 0


def save_grid(images, path, nrow=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # images assumed in [-1,1]
    imgs = (images + 1) / 2
    save_image(imgs, path, nrow=nrow)
