import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor

from descry import FashionDataset, VisionTransformer

vt = VisionTransformer()
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print(image.shape)
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
image = data[3][0]
image = ToTensor()(image)
print(image.shape)
# print(1024-image.shape[1], 1024-image.shape[2])
image = nn.functional.pad(image, (237, 237, 100, 99), mode="replicate")

print(type(image), image.size)
vt.forward(image, image)
