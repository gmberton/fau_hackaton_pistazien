import torch
import torchvision
import requests

from trained_models import mixvpr
from trained_models.deepmad import get_backbone
from pathlib import Path
import tarfile


def get_model(method, backbone=None, descriptors_dimension=None):
    weights_dir = Path(__file__).parent.parent.joinpath("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    if method == "resnet18":
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512

    if method == "deepmad-89m":
        structure = {
            "best_structures": [
                [
                    {"class": "ConvKXBNRELU", "in": 3, "k": 3, "out": 56, "s": 2},
                    {
                        "L": 10,
                        "btn": 816,
                        "class": "SuperResK1DWSEK1",
                        "in": 56,
                        "inner_class": "ResK1DWSEK1",
                        "k": 5,
                        "out": 136,
                        "s": 2,
                    },
                    {
                        "L": 9,
                        "btn": 1224,
                        "class": "SuperResK1DWSEK1",
                        "in": 136,
                        "inner_class": "ResK1DWSEK1",
                        "k": 5,
                        "out": 204,
                        "s": 2,
                    },
                    {
                        "L": 7,
                        "btn": 1952,
                        "class": "SuperResK1DWSEK1",
                        "in": 204,
                        "inner_class": "ResK1DWSEK1",
                        "k": 5,
                        "out": 326,
                        "s": 2,
                    },
                    {
                        "L": 4,
                        "btn": 2944,
                        "class": "SuperResK1DWSEK1",
                        "in": 326,
                        "inner_class": "ResK1DWSEK1",
                        "k": 5,
                        "out": 490,
                        "s": 2,
                    },
                    {
                        "L": 1,
                        "btn": 4464,
                        "class": "SuperResK1DWSEK1",
                        "in": 490,
                        "inner_class": "ResK1DWSEK1",
                        "k": 3,
                        "out": 744,
                        "s": 1,
                    },
                    {"class": "ConvKXBNRELU", "in": 744, "k": 1, "out": 2560, "s": 1},
                ]
            ],
            "space_arch": "CnnNet",
        }

        weights_url = "http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DeepMAD/DeepMAD-89M/DeepMAD-89M-Res224-84.0acc.pth.tar"

        weights_file = (
            weights_dir.joinpath("DeepMAD-89M-Res224-84.0acc.pth.tar")
            .absolute()
            .resolve()
        )

        # Download the weights file
        if not weights_file.exists():
            response = requests.get(weights_url, stream=True)
            if response.status_code == 200:
                with open(weights_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

        model, network_arch = get_backbone(
            structure,
            str(weights_file),
            classification=True,
            descriptors_dimension=1000,
        )

        model.fc_linear = torch.nn.Identity()
        descriptors_dimension = 2560

    elif method == "mixvpr":
        model = mixvpr.MixVPR()
        descriptors_dimension = 4096

    # TODO add models here

    return model, descriptors_dimension
