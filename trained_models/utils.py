import requests
from tqdm import tqdm
from pathlib import Path
from math import ceil


class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


import torch

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def download_file(url: str, filename: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in tqdm(
                r.iter_content(chunk_size=8192),
                total=int(ceil(int(r.headers["Content-Length"]) / 8192)),
                desc=f"Downloading {filename.name}",
            ):
                f.write(chunk)
