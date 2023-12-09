
import torch
import torchvision

from trained_models import mixvpr
from trained_models import models_vit_mage
from util_mage.pos_embed import interpolate_pos_embed


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512
    
    elif method == "mixvpr":
        model = mixvpr.MixVPR()
        descriptors_dimension = 4096
    
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

