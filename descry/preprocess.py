import os

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor


def __to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)

def serialize(path):
    for n in range(1000):
        raw_image = Image.open(os.path.join(path, f"png_images/IMAGES/img_{n+1:04}.png"))
        raw_mask = Image.open(os.path.join(path, f"png_masks/MASKS/seg_{n+1:04}.png"))
        image = ToTensor()(raw_image)
        image = nn.functional.pad(image, (237, 237, 100, 99), "constant", 0)
        image = nn.functional.interpolate(image.reshape(1,1,3,1024,1024), (3,128,128)).reshape(3,128,128)
        
        mask = torch.tensor(np.array(raw_mask.getdata()).reshape(1,1024,1024))
        mask = nn.functional.pad(mask, (237, 237, 100, 99), "constant", 0)
        mask = __to_one_hot(torch.round(mask*58).long(), 58)
        mask = mask.transpose(3, 1).transpose(2, 3)
        mask = nn.functional.interpolate(mask.reshape(1,1,58,1024,1024).float(), (58,128,128)).reshape(1,58,128,128)
        
        if not os.path.isdir(os.path.join(path, "tensor_images")):
           os.mkdir(os.path.join(path, "tensor_images"))
        if not os.path.isdir(os.path.join(path, "tensor_masks")):
           os.mkdir(os.path.join(path, "tensor_masks"))
        
        torch.save(image, os.path.join(path, f"tensor_images/img_down_{n}.pt"))
        torch.save(mask, os.path.join(path, f"tensor_masks/_down_{n}.pt"))

serialize("/home/quantumish/aux/fashion-seg/")
