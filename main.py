import matplotlib.pyplot as plt
import torch
from PIL import Image

from descry import FashionDataset, VisionTransformer

vt = VisionTransformer()
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print(image.shape)
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vt.parameters(), lr=3e-1)
for epoch in range(1000):
    datapoint = data[3]
    out = vt.forward(datapoint[0], datapoint[1])
    
    # for c in range(58):
    #     ax = plt.subplot()
    #     im = plt.imshow(out.detach().numpy()[0, c, :, :])
    #     plt.colorbar(im)
    #     plt.title(classes[c])
    #     plt.show()
    # full = torch.zeros(1, 1, 128, 128) 
    # for a in range(128):
    #     for b in range(128):
    #         maxc = -100000
    #         for c in range(58):
    #             if out[0][c][a][b].item() > maxc:
    #                 maxc = c
    #         full[0][0][a][b] = maxc
    full = torch.reshape(
        torch.floor(torch.nn.Upsample(size=(1024, 1024))(out))[0, :58],
        [1, 58, 1024, 1024]
    )
    # for i in range(10):
    #     ax = plt.subplot()
    #     qim = plt.imshow(out.detach().numpy()[0][i])
    #     plt.colorbar(im)
    #     plt.show()
    # plt.imshow(full.detach().numpy()[0, :, :])
    # plt.show()
    # plt.imshow(datapoint[1][0, :, :])
    # plt.show()
    #print(full.shape)
    loss = criterion(full, datapoint[1].float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # if (i % 10 == 0 and i != 0):
    #     plt.imshow(out.detach().numpy()[0, 0, :, :])
    #     plt.show()
    #     plt.imshow(datapoint[1][0, :, :])
    #     plt.show()
    #     plt.imshow(datapoint[0][0, :, :])
    #     plt.show()
    print(loss.item())

