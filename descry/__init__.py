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
        # sep_mask = torch.zeros(1, self.NUM_CLASSES, 
        # for i in range(1024):
        #     for j in range(1024):

        return (image, mask.transpose(3, 1).transpose(2,3))

class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")        
        self.model =  SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        # self.head = nn.Sequential(
        #     nn.Conv2d(1,1,8),
        #     nn.MaxPool2d(4),
        #     nn.Flatten(1),
        #     nn.Linear(8930, 4096),
        #     nn.Unflatten(1, (1,64,64)),
        #     # nn.ReLU(),
        #     # # nn.MaxPool2d(4),
        #     # # nn.Flatten(1),
        #     # nn.Linear(2046, 4096),
        #     # nn.(),
        #     # nn.Linear(1024, 4096),
        #     # nn.ReLU(),
        #     # nn.Unflatten(1, (1, 64, 64)),            
        #     nn.Upsample(size=(128, 128)),        
        #     nn.Conv2d(1, 1, 4),
        #     nn.ReLU(),
        #     nn.Upsample(size=(1024, 1024)),
        #     #nn.Conv2d(1, 1, 1),
        # ) 

    def forward(self, image, mask):
        inputs = self.feature_extractor(image, return_tensors="pt")
        out = self.model(**inputs).logits
        #out = nn.Upsample(size=(1024,1024))(out)
        #print(out.shape)
        # out = nn.functional.interpolate(out, (197, 768))
        # out = torch.reshape(out, (197, 768))
        #out = self.head(out)
        return out
    # out -= out.min(1, keepdim=True)[0]
        # out /= out.max(1, keepdim=True)[0]
        # out *= 58

    # 550x825
# 453750
