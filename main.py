import matplotlib.pyplot as plt
import torch
from PIL import Image

from descry import FashionDataset, VisionTransformer

classes = {
    0:"null",
    1:"accessories",
    2:"bag",
    3:"belt",
    4:"blazer",
    5:"blouse",
    6:"bodysuit",
    7:"boots",
    8:"bra",
    9:"bracelet",
    10:"cape",
    11:"cardigan",
    12:"clogs",
    13:"coat",
    14:"dress",
    15:"earrings",
    16:"flats",
    17:"glasses",
    18:"gloves",
    19:"hair",
    20:"hat",
    21:"heels",
    22:"hoodie",
    23:"intimate",
    24:"jacket",
    25:"jeans",
    26:"jumper",
    27:"leggings",
    28:"loafers",
    29:"necklace",
    30:"panties",
    31:"pants",
    32:"pumps",
    33:"purse",
    34:"ring",
    35:"romper",
    36:"sandals",
    37:"scarf",
    38:"shirt",
    39:"shoes",
    40:"shorts",
    41:"skin",
    42:"skirt",
    43:"sneakers",
    44:"socks",
    45:"stockings",
    46:"suit",
    47:"sunglasses",
    48:"sweater",
    49:"sweatshirt",
    50:"swimwear",
    51:"t-shirt",
    52:"tie",
    53:"tights",
    54:"top",
    55:"vest",
    56:"wallet",
    57:"watch",
    58:"wedges",
}

vt = VisionTransformer()
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print(image.shape)
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vt.parameters(), lr=3e-3)
for i in range(1000):
    datapoint = data[3]
    out = vt.forward(datapoint[0], datapoint[1])
    print(out.shape)
    # for c in range(58):
    #     ax = plt.subplot()
    #     im = plt.imshow(out.detach().numpy()[0, c, :, :])
    #     plt.colorbar(im)
    #     plt.title(classes[c])
    #     plt.show()
    full = torch.zeros(1, 1, 128, 128)
    for a in range(128):
        for b in range(128):
            maxc = -100000
            for c in range(58):
                if out[0][c][a][b].item() > maxc:
                    maxc = c
            full[0][0][a][b] = maxc
    full = torch.floor(torch.nn.Upsample(size=(1024, 1024))(full))
    print(full.dtype, datapoint[1].dtype)
    loss = criterion(full, datapoint[1].long())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i % 10 == 0 and i != 0):
        plt.imshow(out.detach().numpy()[0, 0, :, :])
        plt.show()
        plt.imshow(datapoint[1][0, :, :])
        plt.show()
        plt.imshow(datapoint[0][0, :, :])
        plt.show()
    print(loss.item())
