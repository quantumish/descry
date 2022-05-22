__version__ = '0.1.0'

import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import ViTConfig, ViTFeatureExtractor, ViTModel


class FashionDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 1000 # TODO generalize

    def __getitem__(self, n):
        raw_image = Image.open(os.path.join(self.path, f"png_images/IMAGES/img_{n:04}.png"))
        raw_mask = Image.open(os.path.join(self.path, f"png_masks/MASKS/seg_{n:04}.png"))
        return (raw_image, raw_mask)

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.head = nn.Sequential(
            nn.Conv2d(1, 8, (8, 8), stride=2),
            nn.MaxPool2d(4),
            nn.Conv2d(8, 1, (4, 4)),
            nn.ReLU(),
            # nn.Flatten(1),
            # nn.Linear(1840, 512),
            # nn.Linear(512, 128),
            # nn.Linear(128, 512),
            # nn.Linear(512, 1000),
            nn.ConvTranspose2d(1, 8, 16),
            nn.ConvTranspose2d(8, 1, 32),
            nn.ConvTranspose2d(1, 1, 128),
            # nn.Conv2d(8, 1, (4, 4)),
        )

    def forward(self, image, mask):
        inputs = self.feature_extractor(image, return_tensors="pt")
        out = self.model(**inputs).last_hidden_state
        # out = torch.reshape(out, (197, 768))
        print(self.head(out).shape)

    # 550x825
# 453750
