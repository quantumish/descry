import matplotlib.pyplot as plt
import torch
from PIL import Image

from descry import FashionDataset, VisionTransformer


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

vt = VisionTransformer()
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# print(image.shape)
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vt.parameters(), lr=3e-2)
datapoint = data[3]
for epoch in range(1000):
    out = vt.forward(datapoint[0], datapoint[1])
    if (epoch != 0 and epoch % 5 == 0):
        for c in range(5):
            ax = plt.subplot()
            im = plt.imshow(out.detach().numpy()[0, c, :, :])
            plt.colorbar(im)
            plt.show()
            ax = plt.subplot()
            im = plt.imshow(datapoint[1][0, c, :, :])
            plt.colorbar(im)
            plt.title("(actual)")
            plt.show()
    
    loss = criterion(out, datapoint[1].float())    
    loss.backward()
    optimizer.step()
    #plot_grad_flow(vt.named_parameters())
    #plt.show()
    optimizer.zero_grad()
    print(loss.item())

