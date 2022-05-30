"""Training script for `descry`."""

import torch
from torch.utils.data import DataLoader

import wandb
from descry import FashionDataset, VisionTransformer

# set hyperparameter defaults for W&B
hyper_defaults = dict(
    learning_rate=0.0001,
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

# wandb initialization
wandb.init(project="descry", entity="quantumish", config=hyper_defaults)
config = wandb.config

torch.backends.cudnn.benchmark = True  # Performance tweak for GPUs
# disable a variety of things that slow down runtime
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# initialize and copy model to gpu
cuda = torch.device('cuda')
vt = VisionTransformer(
    hidden_dropout_prob=config["dr_hidden"],
    drop_path_rate=config["dr_blocks"],
    attention_probs_dropout_prob=config["dr_attention"],
    classifier_dropout_prob=config["dr_classifier"],
    num_encoder_blocks=config["encoder_blocks"],
).to(cuda)

# map the wandb config optimizer string to an actual function
optim_map = dict(
    adam=torch.optim.Adam,
    nadam=torch.optim.NAdam,
    adagrad=torch.optim.Adagrad,
    sgd=torch.optim.SGD,
    rmsprop=torch.optim.RMSprop
)

# split the dataset and initialize batched data loaders
data = FashionDataset("/home/quantumish/aux/fashion-seg/")
train, test = torch.utils.data.random_split(
    data,
    [int(0.7 * (len(data))), (len(data))-int(0.7 * len(data))]
)
batch_sz = 8
trainloader = DataLoader(train, batch_size=batch_sz, num_workers=4, shuffle=True)
testloader = DataLoader(test, batch_size=batch_sz, num_workers=4)

criterion = torch.nn.MSELoss()
optimizer = optim_map[config["optimizer"]](
    vt.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
for epoch in range(config["epochs"]):
    avg_loss = 0
    for batch in trainloader:
        out = vt.forward(batch[0].float())
        loss = criterion(out, batch[1].float().cuda())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)

    avg_loss /= len(train)
    val_loss = 0
    vt.eval() # put network in eval mode just in case
    with torch.no_grad(): # disable grads during validation
        for batch in testloader:
            out = vt.forward(batch[0].float())
            loss = criterion(out, batch[1].float().cuda())
            val_loss += loss.item()
        val_loss /= len(test)
        # preview model outputs 
        # bench_thing = vt.forward(data[2][0].reshape(1, 3, 128, 128).float()).cpu().numpy()[0]
        wandb.log({
            "loss": avg_loss,
            "val_loss": val_loss,
            # "outputs": [wandb.Image(i) for i in bench_thing],
            # "val_outputs": [wandb.Image(i) for i in val_last],
        })

    print("Epoch {} val: {}".format(avg_loss, val_loss))
    vt.train()
