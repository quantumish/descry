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

    def __to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

        return zeros.scatter(scatter_dim, y_tensor, 1)


    def __len__(self):
        return 1000 # TODO generalize

    def __getitem__(self, n):
        raw_image = Image.open(os.path.join(self.path, f"png_images/IMAGES/img_{n:04}.png"))
        raw_mask = Image.open(os.path.join(self.path, f"png_masks/MASKS/seg_{n:04}.png"))
        image = ToTensor()(raw_image)
        image = nn.functional.pad(image, (237, 237, 100, 99), "constant", 0)
        mask = ToTensor()(raw_mask)
        mask = nn.functional.pad(mask, (237, 237, 100, 99), "constant", 0)
        mask = self.__to_one_hot(torch.round(mask*58).long(), 58)
        mask = mask.transpose(3, 1).transpose(2,3)
        target = torch.zeros(1, 150, 1024, 1024)
        target[0, :58, :, :] = mask

        return (image, target)

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")        
        self.model =  SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.train() 
        self.head = nn.Sequential(
            nn.Sigmoid(),
            nn.Upsample(size=(1024, 1024)),
        )

    def forward(self, image, mask):
        inputs = self.feature_extractor(image, return_tensors="pt")
        out = self.model(**inputs).logits
        out = self.head(out)
        return out
