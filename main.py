import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from descry import FashionDataset, VisionTransformer

hyper_defaults = dict(
    learning_rate=0.001,
    lr_decay=0,
    lr_decay_type="none",
    weight_decay=0,
    optimizer="adam",
    epochs=30,
)

wandb.init(project="descry", entity="quantumish", config=hyper_defaults)
config = wandb.config

torch.backends.cudnn.benchmark = True  # Performance tweak for GPUs
cuda = torch.device('cuda')
vt = VisionTransformer().to(cuda)

optim_map = dict(
    adam=torch.optim.Adam,
    nadam=torch.optim.NAdam,
    adagrad=torch.optim.Adagrad,
    sgd=torch.optim.SGD,
    rmsprop=torch.optim.RMSprop
)

data = FashionDataset("/home/quantumish/aux/fashion-seg/")
train, test = torch.utils.data.random_split(
    data,
    [int(0.7 * (len(data))), (len(data))-int(0.7 * len(data))]
)
batch_sz = 4
trainloader = DataLoader(train, batch_size=batch_sz, num_workers=0, shuffle=True)
testloader = DataLoader(test, batch_size=batch_sz, num_workers=0)

criterion = torch.nn.MSELoss()
optimizer = optim_map[config["optimizer"]](
    vt.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],    
)
for epoch in range(config["epochs"]):
    avg_loss = 0
    for batch in trainloader:
        # print(batch[1].shape)
        # label = batch[1][0].float().cuda()
        out = vt.forward(batch[0].reshape(batch_sz, 3, 128, 128).float())
        loss = criterion(out, batch[1].reshape(batch_sz, 58, 128, 128).float().cuda())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        optimizer.zero_grad()
    avg_loss /= len(train)
    wandb.log({"loss": avg_loss})

    # batch = data[2]
    # out = vt.forward(batch[0].reshape(1, 3, 128, 128).float())
    # wandb.log({"null": [wandb.Image(out.cpu().detach().numpy()[0, 0, :, :], caption="Null")]})
    # wandb.log({"shirt": [wandb.Image(out.cpu().detach().numpy()[0, 3, :, :], caption="shirt")]})
    # wandb.log({"boots": [wandb.Image(out.cpu().detach().numpy()[0, 2, :, :], caption="boots")]})
    # wandb.log({"acc": [wandb.Image(out.cpu().detach().numpy()[0, 1, :, :], caption="accessories")]})
    #exit(0)

    val_loss = 0
    for batch in testloader:
        out = vt.forward(batch[0].reshape(batch_sz, 3, 128, 128).float())
        loss = criterion(out, batch[1].reshape(batch_sz, 58, 128, 128).float().cuda())
        val_loss += loss.item()
    val_loss /= len(test)
    # print(epoch)
    # if epoch == 16:
    #     batch = test[3]
    #     out = vt.forward(batch[0].reshape(1, 3, 128, 128).float())
    #     for j in range(58):
    #         f, axarr = plt.subplots(1,2)
    #         axarr[0].imshow(out.cpu().detach()[0,j,:,:])
    #         axarr[1].imshow(batch[1][0,j,:,:])
    #         plt.savefig("./check_{}.png".format(j))
    #     exit()
        
    wandb.log({"val_loss": val_loss})
    #plot_grad_flow(vt.named_parameters())
    #plt.show()
    #print(loss.item())
