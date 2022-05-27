__version__ = '0.1.0'

import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, ToTensor
from transformers import (
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    SegformerModel,
    ViTConfig,
    ViTFeatureExtractor,
    ViTModel,
)


class FashionDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.NUM_CLASSES = 58

    def __len__(self):
        return 1000 # TODO generalize

    def __getitem__(self, n):
        return (
            torch.load(os.path.join(self.path, f"tensor_images/img_down_{n}.pt")),
            torch.load(os.path.join(self.path, f"tensor_masks/_down_{n}.pt")),
        )

class VisionTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = SegformerFeatureExtractor()
        self.feature_extractor.do_resize = False
        self.feature_extractor.do_normalize = False
        self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=58, **kwargs))
        self.model.train() 
        self.head = nn.Sequential(            
            nn.Upsample(size=(64, 64)),
            nn.Conv2d(58, 58, 4),
            nn.ReLU(),
            nn.Upsample(size=(128, 128)),
            nn.Softmax(dim=1)
        )

    def forward(self, image):
        inputs = self.feature_extractor(list(image.numpy()), return_tensors="pt")
        out = self.model(inputs.data["pixel_values"].cuda()).logits
        out = self.head(out)
        return out
