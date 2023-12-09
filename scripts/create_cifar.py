"""
To download CIFAR-100 run 

`wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`

then extract the tar and run this script
"""

import os
import einops
from PIL import Image
from tqdm import tqdm


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def prepare_subset(subset="test"):
    dataset = unpickle(f"/home/gaber/Downloads/cifar-100-python/{subset}")
    
    for k, v in dataset.items():
        print(f"{k} {len(v)} {type(v)}")
    
    data = dataset[b"data"]
    fine_labels = dataset[b"fine_labels"]
    coarse_labels = dataset[b"coarse_labels"]
    print(data.shape)
    
    print(len(set(fine_labels)))
    
    print(data.min())
    print(data.max())
    
    images = einops.rearrange(data, "b (c h w) -> b h w c", c=3, h=32, w=32)
    print(images.shape)
    
    for idx, (img, label) in enumerate(zip(tqdm(images), fine_labels)):
        folder = f"cifar/cifar-100-{subset}/{label:02d}"
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(f"{folder}/{idx:04d}.png")
    
    for idx, (img, label) in enumerate(zip(tqdm(images), coarse_labels)):
        folder = f"cifar/cifar-10-{subset}/{label}"
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img).save(f"{folder}/{idx:04d}.png")


prepare_subset("test")
prepare_subset("train")

