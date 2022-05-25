import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from descry import FashionDataset, VisionTransformer

wandb.init(project="descry", entity="quantumish")

# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
# }

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

torch.backends.cudnn.benchmark = True  # Performance tweak for GPUs
cuda = torch.device('cuda')
vt = VisionTransformer().to(cuda)

data = FashionDataset("/home/quantumish/aux/fashion-seg/")
train, test = torch.utils.data.random_split(
    data,
    [int(0.7 * (len(data))), (len(data))-int(0.7 * len(data))]
)
batch_sz = 1
#trainloader = DataLoader(train, batch_size=batch_sz, num_workers=1, pin_memory=True, shuffle=True)
#testloader = DataLoader(test, batch_size=batch_sz, num_workers=1, pin_memory=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(vt.parameters(), lr=1e-3)
for epoch in range(1000):
    avg_loss = 0
    i = 0
    for t in range(1):
        batch = data[2]
        label = batch[1].float().cuda()
        out = vt.forward(batch[0], batch[1])
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        optimizer.zero_grad()
        i += 1
        wandb.log({"null": [wandb.Image(out.cpu().detach().numpy()[0, 0, :, :], caption="Null")]})
        wandb.log({"shirt": [wandb.Image(out.cpu().detach().numpy()[0, 3, :, :], caption="shirt")]})
        wandb.log({"boots": [wandb.Image(out.cpu().detach().numpy()[0, 2, :, :], caption="boots")]})
        wandb.log({"acc": [wandb.Image(out.cpu().detach().numpy()[0, 1, :, :], caption="accessories")]})
    #avg_loss /= len(data)
    #exit(0)
    wandb.log({"loss": avg_loss})

    # val_loss = 0
    # for batch in testloader:
    #     label = batch[1][0].float().cuda()
    #     out = vt.forward(batch[0][0], batch[1][0])            
    #     loss = criterion(out, label)    
    #     val_loss += loss.item()
    # val_loss /= len(data)
    # wandb.log({"val_loss": avg_loss})        
    #plot_grad_flow(vt.named_parameters())
    #plt.show()
    # print(loss.item())

