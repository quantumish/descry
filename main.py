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
    dr_hidden=0.0,
    dr_classifier=0.1,
    dr_attention=0.0,
    dr_blocks=0.1,
    encoder_blocks=4,
)

wandb.init(project="descry", entity="quantumish", config=hyper_defaults)
config = wandb.config

torch.backends.cudnn.benchmark = True  # Performance tweak for GPUs
cuda = torch.device('cuda')
vt = VisionTransformer(
    hidden_dropout_prob=config["dr_hidden"],
    drop_path_rate=config["dr_blocks"],
    attention_probs_dropout_prob=config["dr_attention"],
    classifier_dropout_prob=config["dr_classifier"],
    num_encoder_blocks=config["encoder_blocks"],
).to(cuda)

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
    out = 0
    # for batch in trainloader:
    for batch in trainloader:
        out = vt.forward(batch[0].reshape(batch_sz, 3, 128, 128).float())
        loss = criterion(out, batch[1].reshape(batch_sz, 58, 128, 128).float().cuda())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        optimizer.zero_grad()    
    avg_loss /= len(train)
    val_loss = 0
    for batch in testloader:
        out = vt.forward(batch[0].reshape(batch_sz, 3, 128, 128).float())
        loss = criterion(out, batch[1].reshape(batch_sz, 58, 128, 128).float().cuda())
        val_loss += loss.item()
    val_loss /= len(test)
    bench_thing = vt.forward(data[2][0].reshape(1, 3, 128, 128).float()).cpu().detach().numpy()[0]
    wandb.log({
        "loss": avg_loss,
        "val_loss": val_loss,
        "outputs": [wandb.Image(i) for i in bench_thing], 
        # "val_outputs": [wandb.Image(i) for i in val_last], 
    })
