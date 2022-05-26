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
            torch.load(os.path.join(self.path, f"tensor_images/img_{n}.pt")),
            torch.load(os.path.join(self.path, f"tensor_masks/_{n}.pt")),
        )

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.feature_extractor.do_resize = False
        self.feature_extractor.do_normalize = False
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.train() 
        self.head = nn.Sequential(
            nn.Sigmoid(),
            nn.Upsample(size=(1024, 1024)),
        )

    def forward(self, image):
        inputs = self.feature_extractor(image.numpy(), return_tensors="pt")
        out = self.model(inputs.data["pixel_values"].cuda()).logits
        out = self.head(torch.narrow(out, 1, 0, 58))
        return out
