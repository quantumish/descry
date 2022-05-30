"""Library for clothing segmentation."""
__version__ = '0.1.0'

import os

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    SegformerConfig,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
)


class FashionDataset(Dataset):
    """Implementation of PyTorch Dataset for People Clothing Segmentation dataset."""

    def __init__(self, path):
        """Initialize the dataset.

        Arguments:
        - `path`: a string representing the absolute path to your dataset.
        """
        self.path = path

    def __len__(self):
        """Get the size of the dataset."""
        return 8279 # TODO generalize

    def __getitem__(self, n):
        """Get the n-th item from the dataset.

        Arguments:
        - `n`: integer representing index of item to fetch.

        Returns: tuple of tensor containing image and tensor containing 13 one-hot encoded masks.
        """
        return (
            torch.load(os.path.join(self.path, f"tensor_images/{n+1:04}.pt")),
            torch.load(os.path.join(self.path, f"tensor_masks/{n+1:04}.pt"))[0],
        )


class VisionTransformer(nn.Module):
    """Model for clothing segmentation."""

    def __init__(self, **kwargs):
        """Initialize the VisionTransformer.

        Arguments:
        - `**kwargs`: keyword arguments corresponding to the arguments of a `SegformerConfig`

        See HuggingFace documentation for more.
        """
        super(VisionTransformer, self).__init__()
        self.feature_extractor = SegformerFeatureExtractor()
        self.feature_extractor.do_resize = False
        self.feature_extractor.do_normalize = False
        self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=13, **kwargs))
        self.model.train()
        self.head = nn.Sequential(
            nn.Upsample(size=(64, 64)),
            nn.Conv2d(13, 13, 4),
            nn.ReLU(),
            nn.Upsample(size=(128, 128)),
            nn.Softmax(dim=1)
        )

    def forward(self, image):
        """Run inference on an image.

        Arguments:
        - `image`: a (3,128,128) RGB image in the form of a torch.tensor.

        Returns: a (1,13,128,128) tensor containing 13 one-hot encoded masks for each clothing type.
        """
        inputs = self.feature_extractor(list(image.numpy()), return_tensors="pt")
        out = self.model(inputs.data["pixel_values"].cuda()).logits
        out = self.head(out)
        return out
