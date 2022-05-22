import matplotlib.pyplot as plt
import torch
from PIL import Image

from descry import FashionDataset, VisionTransformer

vt = VisionTransformer()
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print(image.shape)
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
datapoint = data[3]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vt.parameters(), lr=8e-2)
for i in range(1000):
    out = vt.forward(datapoint[0], datapoint[1])
    loss = criterion(out[0], datapoint[1])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())
    if (i % 20 == 0 and i != 0):
        plt.imshow(out.detach().numpy()[0, 0, :, :])
        plt.show()
        plt.imshow(datapoint[1][0, :, :])
        plt.show()
