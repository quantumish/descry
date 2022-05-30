import os
import warnings
from collections import Counter
from pprint import pprint

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor

warnings.filterwarnings("ignore")

def __to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)

collapse_map = {
    "shirt": [38,49,51,5,48,54],
    "glasses": [17,47],
    "shoes": [58,43,36,32,28,7,16,21,39],
    "pants": [25,31],
    "jacket": [24,11,4,13,55],
    "bag": [2,33],
    "accessories": [1,9,29,34,15,56,57],
    "blacklist": [45,40,6,46,61,8,53,58,35,37,10,27,22,18,26,44,52,53,42],
    "plainlist": [0,3,14,19,20,41]
}

seq = iaa.SomeOf((1,3), [
    iaa.GaussianBlur((0, 3.0)),
    iaa.Affine(translate_px=({"x": (-100, 100)})),
    iaa.Affine(rotate=(-30, 30)),
    iaa.Sharpen(alpha=(0,1.0), lightness=(0.4, 1.5)),
    iaa.Invert(1, per_channel=0.4),
    iaa.imgcorruptlike.GaussianBlur(severity=2),
    iaa.Emboss(alpha=(0.0,1.0), strength=(0.5,1.5)),
    iaa.Fliplr(),
    iaa.Canny(alpha=(0.0,0.5)),
    iaa.Dropout(p=(0,0.1)),
    iaa.MultiplySaturation((0.5,1.5)),
    iaa.ElasticTransformation(alpha=(0,5.0), sigma=0.25),
])

def serialize(path):
    for n in range(1000):
        skip = False
        png_mask = Image.open(os.path.join(path, f"png_masks/MASKS/seg_{n+1:04}.png"))
        colors = [i[1] for i in png_mask.getcolors()]
        raw_mask = np.array(png_mask)
        png_mask.close()
        for i in colors:
            if i in collapse_map["plainlist"]:
                raw_mask[raw_mask == i] = collapse_map["plainlist"].index(i)
            elif i in collapse_map["shirt"]:
                raw_mask[raw_mask == i] = 6
            elif i in collapse_map["pants"]:
                raw_mask[raw_mask == i] = 7
            elif i in collapse_map["accessories"]:
                raw_mask[raw_mask == i] = 8
            elif i in collapse_map["bag"]:
                raw_mask[raw_mask == i] = 9
            elif i in collapse_map["shoes"]:
                raw_mask[raw_mask == i] = 10
            elif i in collapse_map["glasses"]:
                raw_mask[raw_mask == i] = 11
            elif i in collapse_map["jacket"]:
                raw_mask[raw_mask == i] = 12
            if i in collapse_map["blacklist"]:
                skip = True
                break

        if skip:
            continue

        print(n)
        raw_image = Image.open(os.path.join(path, f"png_images/IMAGES/img_{n+1:04}.png"))
        image = np.array(raw_image)

        for i in range(17):
            vimage, vmask = seq(images=image.reshape(1,825,550,3), segmentation_maps=raw_mask.reshape(1,825,550,1))
            vimage = torch.tensor(vimage.transpose()[:, :, :, 0]).transpose(1,2)
            vimage = nn.functional.pad(vimage, (237, 237, 100, 99), "constant", 0)
            vimage = nn.functional.interpolate(vimage.reshape(1,1,3,1024,1024), (3,128,128)).reshape(3,128,128)

            vmask = nn.functional.pad(torch.tensor(vmask[0,:,:,0]), (237, 237, 100, 99), "constant", 0)
            vmask = __to_one_hot(vmask.long(), 13).transpose(0,2)
            vmask = nn.functional.interpolate(vmask.reshape(1,1,13,1024,1024).float(), (13,128,128)).reshape(1,13,128,128)

            if not os.path.isdir(os.path.join(path, "tensor_images")):
                os.mkdir(os.path.join(path, "tensor_images"))
            if not os.path.isdir(os.path.join(path, "tensor_masks")):
                os.mkdir(os.path.join(path, "tensor_masks"))

            torch.save(vimage.clone(), os.path.join(path, f"tensor_images/img_down_{n*17+i}.pt"))
            torch.save(vmask.clone(), os.path.join(path, f"tensor_masks/_down_{n*17+i}.pt"))



#sanitize("/home/quantumish/aux/fashion-seg/")
serialize("/home/quantumish/aux/fashion-seg/")
