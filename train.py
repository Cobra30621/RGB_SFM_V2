import os
import torch
import wandb
import numpy as np
import time
from pathlib import Path
import copy
import shutil

from torch import nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

from utils import increment_path
from models.basemodels import CNN, ResNet, AlexNet, LeNet, GoogLeNet, MLP
from models.RGB_Plan_v9 import SOMNetwork
from load_data import load_data
from test import eval

def choose_model(current_model):
    if current_model == 'SFM': 
        model = SOMNetwork(input_shape=input_size, out_channels=out_channels, stride = stride).to(device)
    elif current_model == 'cnn':
        model = CNN(in_channels=input_size[0], out_channels = out_channels).to(device)
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
    return model

def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, scheduler, epoch, device):
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    count = 0
    patience = 20
    checkpoint = {}
    with torch.autograd.set_detect_anomaly(True):
        for e in range(epoch):
            print(f"------------------------------EPOCH {e}------------------------------")
            model.train()
            progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
            losses = 0
            correct = 0
            size = 0
            for batch, (X, y) in progress:
                X = X.to(device); y= y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # model.RGB_preprocess[0].rgb_weight.data = model.RGB_preprocess[0].rgb_weight.data.clamp(0.0, 1.0)
                # if count == patience - 1:
                #     print(model.RGB_preprocess[0].rgb_weight.data)
                
                losses += loss.detach().item()
                size += len(X)

                pred = nn.Softmax(dim=-1)(pred)
                _, maxk = torch.topk(pred, 1, dim = -1, sorted = False)
                _, y = torch.topk(y, 1, dim=-1, sorted = False)
                correct += torch.eq(maxk, y).sum().detach().item()
                train_loss = losses/(batch+1)
                train_acc = correct/size
                progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))

            valid_acc, valid_loss, _ = eval(valid_dataloader, model, loss_fn, False, device = device)
            print(f"Test Loss: {valid_loss}, Test Accuracy: {valid_acc}")
            if scheduler:
                scheduler.step(valid_loss)

            metrics = {
                "train/loss": train_loss,
                "train/epoch": e,
                "train/accuracy": train_acc,
                "train/learnrate": optimizer.param_groups[0]['lr'],
                "test/loss": valid_loss,
                "test/accuracy": valid_acc
            }
            wandb.log(metrics, step=e)

            #early stopping
            if valid_acc < best_valid_acc:
                count += 1
                # if count >= patience:
                #     break
            else:
                count = 0
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                cur_train_loss = train_loss
                cur_train_acc = train_acc
                print(f'best epoch: {e}')
                checkpoint['model_weights'] = model.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['scheduler'] = scheduler.state_dict()
                    
    return cur_train_loss, cur_train_acc, best_valid_loss, best_valid_acc, checkpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
root = os.path.dirname(__file__)
current_model = 'SFM' # SFM, mlp, cnn, resnet50, alexnet, lenet, googlenet
dataset = 'rgb_simple_shape' # mnist, fashion, cifar10, malaria, malaria_split, rgb_simple_shape
input_size = (3, 30, 30)
rbf = 'triangle' # gauss, triangle
batch_size = 32
epoch = 2000
lr = 0.001
layer = 4
stride = 3
out_channels = 15
description = f"Conv2d_kernel = [(3, 3), (5, 5), (7, 7), (9, 9), (35, 35)]\n SFM_combine_filters = [(2, 2),  (1, 5), (5, 1), (1, 1)]"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="paper experiment",

    name = f"RGB_Plan_v9_change_Conv_filter",

    notes = description,
    
    tags = ["RGB_Plan_v9", "rgb-simple-shape-multiclass"],

    group = "RGB_Simple_shape_multiclass",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "layer": layer,
    "stride": stride,
    "epochs": epoch,
    "architecture": current_model,
    "dataset": dataset,
    # "train_data_num": len(train_dataloader.sampler),
    # "test_data_num": len(test_dataloader.sampler),
    # "total_data_num": len(train_dataloader.sampler) + len(test_dataloader.sampler),
    "batch_size": batch_size,
    "input shape": input_size,
    "out_channels": out_channels,
    "SFM filter": "(2, 2)",
    "lr scheduler": "ReduceLROnPlateau",
    "optimizer": "Adam",
    "loss_fn": "CrossEntropyLoss"
    }
)

save_dir = increment_path('./runs/train/exp', exist_ok = False)
Path(save_dir).mkdir(parents=True, exist_ok=True)
shutil.copyfile('./models/RGB_Plan_v9.py', f'{save_dir}/RGB_Plan_v9.py')
print(save_dir)

model = choose_model(current_model)

train_dataloader, test_dataloader = load_data(dataset=dataset, root=root, batch_size=batch_size, input_size=(input_size[1], input_size[2]))

print(model)
summary(model, input_size = input_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, patience=10)

# wandb.watch(model, loss_fn, log="all", log_freq=1)
train_loss, train_acc, valid_loss, valid_acc, checkpoint = train(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, epoch, device = device)
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))
print("Valid: \n\tAccuracy: {}, Avg loss: {} \n".format(valid_acc, valid_loss))

# Test model
model.load_state_dict(checkpoint['model_weights'])
test_acc, test_loss, test_table = eval(test_dataloader, model, loss_fn, device = device)
print("Test: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['train_accuracy'] = train_acc
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['test_accuracy'] = test_acc
wandb.summary['test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})
print(f'checkpoint keys: {checkpoint.keys()}')

torch.save(checkpoint, f'{save_dir}/RGB_Plan_v9_epochs{epoch}.pth')
# m = torch.jit.script(model)
# script = torch.jit.save(m, f'{save_dir}/RGB_Plan_v9_epochs{epoch}_entire_model.pth')
art = wandb.Artifact(f"RGB_Plan_v9_{dataset}", type="model")
art.add_file(f'{save_dir}/RGB_Plan_v9_epochs{epoch}.pth')
art.add_file(f'{save_dir}/RGB_Plan_v9.py')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()