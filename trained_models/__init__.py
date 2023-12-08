
import torch
import torchvision

from trained_models import mixvpr


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "resnet18":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512
    
    elif method == "mixvpr":
        model = mixvpr.MixVPR()
        descriptors_dimension = 4096
    
    # TODO add models here
    
    return model, descriptors_dimension

