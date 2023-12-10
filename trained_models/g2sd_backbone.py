import os
import torch
import gdown
import torch.nn as nn
import timm.models.vision_transformer

from dis import dis
from functools import partial
from torchvision.transforms import Resize
from timm.models.layers import trunc_normal_


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, distillation=False,use_distoken= False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.distillation = distillation
        self.use_distoken = use_distoken
        if distillation:
            
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            trunc_normal_(self.dist_token, std=.02)
            self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.distillation:
            x = torch.cat((x, self.dist_token.expand(B, -1, -1)),dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        if self.distillation:
            return outcome, x[: ,-1,:]
        return outcome, None
    

class image_encoder(torch.nn.Module):
    def __init__(self, n_classes = 1000):
        super().__init__()
        
        # Download Model
        model_path = os.path.join(os.getcwd(), "trained_models", "g2sd_weights")
        model_name = os.path.join(model_path, "checkpoint-last.pth")
        
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            gdown.download(id = "1bICQeHUXWvu5HQ3cVtqfbtFS4Cke36Aq", 
                        output = model_name, 
                        quiet = False)

        # Setup Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.student_model = vit_small_patch16()
        self.student_model.to(self.device)
        checkpoint = torch.load(model_name, map_location='cpu')
        checkpoint_model = checkpoint['model']
        self.student_model.load_state_dict(checkpoint_model, strict = False)
        
        # Image resizer
        self.resizer = Resize((224, 224))
        
    def forward(self, x):
        x = self.resizer(x)
        x, _ = self.student_model.forward_features(x)
        return x