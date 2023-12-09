import clip
import torch
import torchvision

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    

class image_encoder(torch.nn.Module):
    def __init__(self, version = "ViT-B/32", dimension = 512):
        super().__init__()
        
        self.dimension = dimension
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unnormalize = UnNormalize(mean = (0.485, 0.456, 0.406), 
                                        std = (0.229, 0.224, 0.225))
        self.model, self.preprocess = clip.load(version, 
                                                device = self.device)

    def forward(self, x):
        # Un-normalize images
        x = self.unnormalize(x)

        # Convert to PIL
        images = list()
        for i in range(x.shape[0]):
            im = torchvision.transforms.ToPILImage()(x[i, :, :,:])
            images.append(self.preprocess(im).unsqueeze(0).to(self.device))

        # Extract Features
        image_features = torch.zeros(x.shape[0], self.dimension)
        with torch.no_grad():
            for i, im in enumerate(images):
                image_features[i, :] = self.model.encode_image(im)

        return image_features