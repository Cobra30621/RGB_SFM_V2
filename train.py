import os
import torch
import wandb

from torch import nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader

from models import SOMNetwork, CNN, ResNet, AlexNet, LeNet, GoogLeNet, MLP
from load_data import load_data

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
root = os.path.dirname(__file__)
current_model = 'SFM' # SFM, mlp, cnn, resnet50, alexnet, lenet, googlenet
dataset = 'malaria' # mnist, fashion, cifar10, malaria
input_size = (28, 28)
kernel_size = (5, 5)
kernels = [(10, 10), (15, 15), (25, 25), (35, 35)]
in_channels = 3 # 1, 3
rbf = 'gauss' # gauss, triangle

batch_size = 256
epoch = 200
lr = 1e-3
layer = 4
stride = 4
description = f"test"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="paper experiment",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": current_model,
    "dataset": dataset,
    "batch_size": batch_size,
    "layer": layer,
    "stride": stride,
    "epochs": epoch,
    }
)


if current_model == 'SFM': 
    model = SOMNetwork(stride=stride, in_channels=in_channels, input_size=input_size, kernel_size=kernel_size, kernels=kernels, rbf=rbf).to(device)
elif current_model == 'cnn':
    model = CNN(in_channels=in_channels).to(device)
elif current_model == 'mlp':
    model = MLP().to(device)
elif current_model == 'resnet18':
    model = ResNet().to(device)
elif current_model == 'resnet34':
    model = ResNet(layers=34).to(device)
elif current_model == 'resnet50':
    model = ResNet(layers=50).to(device)
elif current_model == 'resnet101':
    model = ResNet(layers=101).to(device)
elif current_model == 'resnet152':
    model = ResNet(layers=152).to(device)
elif current_model == 'alexnet':
    model = AlexNet().to(device)
elif current_model == 'lenet':
    model = LeNet().to(device)
elif current_model == 'googlenet':
    model = GoogLeNet().to(device)
    
# model.load_state_dict(torch.load(f'{root}/result_pth/SFM_2000_85.44728724369442_malaria_4layer_4stride_22133111_15253045.pth'))
# model = torch.load(f'{root}/result_pth/SFM_2000_85.44728724369442_malaria_4layer_4stride_22133111_15253045.pth').to(device)

run_name = f"{dataset}_{layer}layer_{stride}stride_{''.join(str(x) + str(y) for x, y in model.filters)}_{rbf}" if current_model == 'SFM' else f"{dataset}_{current_model}"
if not os.path.exists(f"{root}/result/{run_name}"):
   os.makedirs(f"{root}/result/{run_name}")

def train(train_dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, epoch):
    example_ct = 0 # number of examples seen
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epoch):
            print(f"------------------------------EPOCH {e}------------------------------")
            model.train()
            progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
            losses = 0
            for batch, (X, y) in progress:
                X = X.to(device); y= y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.item()
                example_ct += len(X)
                progress.set_description("Loss: {:.7f}".format(losses/(batch+1)))
            metrics = {
                "train/loss": losses/(batch+1),
                "train/epoch": e,
            }
            wandb.log(metrics, step=e)
            
    
    
def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    table = []
    for X, y in dataloader:
        X = X.to(device); y= y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        size += len(X)

        # tensor to cpu
        loss = loss_fn(pred, y).item()
        c = (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)
        X = X.cpu()
        y = y.cpu()
        
        # sample first image in batch 
        table.append([wandb.Image(np.array(X[0])), y[0], pred[0].argmax(), loss, c])

    test_loss /= num_batches
    correct = (correct / size) * 100
    return correct, test_loss, table


train_dataloader, test_dataloader = load_data(dataset=dataset, root=root, batch_size=batch_size, input_size=input_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

wandb.watch(model, loss_fn, log="all", log_freq=1)
train(train_dataloader, model, loss_fn, optimizer, epoch)

# Train model
train_acc, train_loss, train_table = test(train_dataloader, model, loss_fn)
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))

# Test model
test_acc, test_loss, test_table = test(test_dataloader, model, loss_fn)
print("Test: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['train_accuracy'] = train_acc
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['test_accuracy'] = test_acc
wandb.summary['test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})

# test_aug_dataloader = load_data(dataset='mnist_aug', root=root, batch_size=batch_size, input_size=input_size)
# test_aug_acc, test_aug_loss = test(test_aug_dataloader, model, loss_fn)
# print("Test(AUG): \n\tAccuracy: {}, Avg loss: {} \n".format(test_aug_acc, test_aug_loss))

torch.save(model.state_dict(), f'{root}/result/{run_name}/{current_model}_{epoch}_{test_acc}_{run_name}_final.pth')
art = wandb.Artifact(f"{current_model}_{run_name}", type="model")
art.add_file(f'{root}/result/{run_name}/{current_model}_{epoch}_{test_acc}_{run_name}_final.pth')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()