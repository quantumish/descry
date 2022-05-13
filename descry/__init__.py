__version__ = '0.1.0'

import torch
from transformers import ViTConfig, ViTFeatureExtractor, ViTModel


class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, image):
        inputs = self.feature_extractor(image, return_tensors="pt")
        return self.model(**inputs)
