
import os
import gdown
import torch
import torchvision
import requests

from trained_models import mixvpr
from .fasternet.model_api import LitModel as FasterNet
from .fasternet.parser import parse_fasternet_args
from .utils import download_file
from trained_models import models_vit_mage
from util_mage.pos_embed import interpolate_pos_embed
from trained_models import clip_backbone, g2sd_backbone
from trained_models.deepmad import get_backbone
from pathlib import Path


def get_model(method, backbone=None, descriptors_dimension=None):
    weights_dir = Path(__file__).parent.parent.joinpath("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(__file__).parent.parent
    if method == "resnet18":
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512
    
    elif method == "clip":
        model = clip_backbone.image_encoder()
        descriptors_dimension = 512

    elif method == "g2sd":
        model = g2sd_backbone.image_encoder()
        descriptors_dimension = 384


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

    elif method == "fasternet":
        cfg_path = base_path / "trained_models" / "fasternet" / "fasternet_l.yaml"
        fasternet_checkpoint_path = base_path / "weights" / "fasternet_l.pth"
        checkpoint_url = "https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_l-epoch.299-val_acc1.83.5060.pth"

        download_file(checkpoint_url, fasternet_checkpoint_path)

        args = parse_fasternet_args(
            [
                "-c",
                str(cfg_path.absolute().resolve()),
                "--checkpoint_path",
                str(fasternet_checkpoint_path.absolute().resolve()),
            ]
        )
        model = FasterNet(num_classes=1000, hparams=args)
        missing_keys, unexpected_keys = model.model.load_state_dict(
            torch.load(fasternet_checkpoint_path), False
        )
        model.model.head = torch.nn.Identity()
        descriptors_dimension = 1280

    
    elif method == "mage":
        # To download the models run this
        url = "https://drive.google.com/file/d/1Q6tbt3vF0bSrv5sPrjpFu8ksG3vTsVX2/view"
        if not os.path.exists("mage-vitb-1600.pth"):
            gdown.download(url=url, output="mage-vitb-1600.pth", fuzzy=True)
        url = "https://drive.google.com/file/d/13S_unB87n6KKuuMdyMnyExW0G1kplTbP/view"
        if not os.path.exists("vqgan_jax_strongaug.ckpt"):
            gdown.download(url=url, output="vqgan_jax_strongaug.ckpt", fuzzy=True)
        
        model = models_vit_mage.vit_base_patch16()
        checkpoint_model = torch.load("mage-vitb-1600.pth")["model"]
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        descriptors_dimension = 768
    

    # TODO add models here

    return model, descriptors_dimension
