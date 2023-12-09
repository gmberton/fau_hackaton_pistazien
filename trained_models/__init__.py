import sys
import torch
import torchvision

from trained_models import mixvpr, clip_backbone, iptn_backbone


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512
    
    elif method == "clip":
        model = clip_backbone.image_encoder()
        descriptors_dimension = 512

    elif method == "itpn":
        model = iptn_backbone.image_encoder()
        descriptors_dimension = 512
        sys.exit("kcd here")

    elif method == "mixvpr":
        model = mixvpr.MixVPR()
        descriptors_dimension = 4096
    
    # TODO add models here
    
    return model, descriptors_dimension

