import torch
import torchvision

from trained_models import mixvpr
from .fasternet.model_api import LitModel as FasterNet
from .fasternet.parser import parse_fasternet_args
from .utils import download_file
from trained_models import models_vit_mage
from util_mage.pos_embed import interpolate_pos_embed

from pathlib import Path


def get_model(method, backbone=None, descriptors_dimension=None):
    base_path = Path(__file__).parent.parent
    if method == "resnet18":
        model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512

    elif method == "mixvpr":
        model = mixvpr.MixVPR()
        descriptors_dimension = 4096

    elif method == "fasternet":
        cfg_path = base_path / "trained_models" / "fasternet" / "fasternet_l.yaml"
        checkpoint_path = base_path / "weights" / "fasternet_l.pth"
        checkpoint_url = "https://github.com/JierunChen/FasterNet/releases/download/v1.0/fasternet_l-epoch.299-val_acc1.83.5060.pth"

        download_file(checkpoint_url, checkpoint_path)

        args = parse_fasternet_args(
            [
                "-c",
                str(cfg_path.absolute().resolve()),
                "--checkpoint_path",
                str(checkpoint_path.absolute().resolve()),
            ]
        )
        model = FasterNet(num_classes=1000, hparams=args)
        missing_keys, unexpected_keys = model.model.load_state_dict(
            torch.load(checkpoint_path), False
        )
        model.model.head = torch.nn.Identity()
        descriptors_dimension = 1280

    
    elif method == "mage":
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
        descriptors_dimension = 1000
    
    # TODO add models here

    return model, descriptors_dimension
