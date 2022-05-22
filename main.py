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
optimizer = torch.optim.Adam(vt.parameters(), lr=3e-3)
datapoint = (
    torch.nn.functional.interpolate(torch.reshape(datapoint[0], (1,3,1024,1024)), (197, 768)),
    torch.nn.functional.interpolate(torch.reshape(datapoint[1], (1,1,1024,1024)), (197, 768))
)
datapoint = (
    torch.reshape(datapoint[0], (3,197,768)),
    torch.reshape(datapoint[1], (1,197,768)),
)
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
        plt.imshow(datapoint[0][0, :, :])
        plt.show()
