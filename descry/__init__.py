__version__ = '0.1.0'

import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor
from transformers import ViTConfig, ViTFeatureExtractor, ViTModel


class FashionDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 1000 # TODO generalize

    def __getitem__(self, n):
        raw_image = Image.open(os.path.join(self.path, f"png_images/IMAGES/img_{n:04}.png"))
        raw_mask = Image.open(os.path.join(self.path, f"png_masks/MASKS/seg_{n:04}.png"))
        image = ToTensor()(raw_image)
        image = nn.functional.pad(image, (237, 237, 100, 99), "constant", 0)
        mask = ToTensor()(raw_mask)
        mask = nn.functional.pad(mask, (237, 237, 100, 99), "constant", 0)
        return (image, mask)

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.head = nn.Sequential(            
        #     # nn.Unflatten(1, torch.Size([1, 64, 64])),
        #     # nn.ConvTranspose2d(1, 8, 8),
        #     # nn.ConvTranspose2d(8, 1, 16),
        #     # nn.Upsample(size=(512, 512)),
        #     # nn.Conv2d(1, 1, (4, 4)),
        #     # nn.ReLU(),            
        #     # nn.Upsample(size=(512, 512)),
        #     # nn.Conv2d(1, 1, (4, 4)),
        #     # nn.ReLU(),
        #     nn.Upsample(size=(1024, 1024)),
        #     nn.ReLU(),            
        # )

    def forward(self, image, mask):
        inputs = self.feature_extractor(image, return_tensors="pt")
        out = self.model(**inputs).last_hidden_state
        out = out.view([1, 1, 197, 768])
        out = nn.functional.interpolate(out, (1024, 1024))
        # out = torch.reshape(out, (197, 768))
        # out = self.head(out)        
        return nn.functional.relu(out)
    # out -= out.min(1, keepdim=True)[0]
        # out /= out.max(1, keepdim=True)[0]
        # out *= 58

    # 550x825
# 453750
