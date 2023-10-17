import os
import torch
import wandb
import numpy as np
import time

from torch import nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

from models import CNN, ResNet, AlexNet, LeNet, GoogLeNet, MLP
from RGB_Plan_v1 import SOMNetwork
from load_data import load_data

def train(train_dataloader: DataLoader, test_dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, scheduler, epoch, device):
    min_loss = float('inf')
    count = 0
    patience = 20
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
                # time_start = time.time()
                # print(pred[0])
                # print(y[0])
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses += loss.detach().item()
                size += len(X)

                _, maxk = torch.topk(pred, 2, dim = -1, sorted = False)
                _, y = torch.topk(y, 2, dim=-1, sorted = False)
                correct += torch.eq(maxk, y).all(dim=-1).sum().detach().item()
                torch.cuda.synchronize()
                # print(f'cal loss and accuracy: {time.time() - time_start}')
                progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(losses/(batch+1), correct/size))

            test_acc, test_loss, _ = test(test_dataloader, model, loss_fn, False, device = device)
            print(f"Test Accuracy: {test_acc}%, Test Loss: {test_loss}")
            if scheduler:
                scheduler.step(test_loss)

            metrics = {
                "train/loss": losses/(batch+1),
                "train/epoch": e,
                "train/accuracy": correct/size,
                "train/learnrate": optimizer.param_groups[0]['lr'],
                "test/loss": test_loss,
                "test/accuracy": test_acc
            }
            wandb.log(metrics, step=e)

            #early stopping
            # if e > 50:
            #     if test_loss >= min_loss:
            #         count += 1
            #         if count >= patience:
            #             break
            #     else:
            #         count = 0
            #         min_loss = test_loss
            
def test(dataloader: DataLoader, model: nn.Module, loss_fn, need_table = True, device=None):
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    table = []
    for X, y in dataloader:
        X = X.to(device); y= y.to(device)
        pred = model(X)
        
        loss = loss_fn(pred, y)
        test_loss += loss

        _, maxk = torch.topk(pred, 2, dim = -1, sorted = False)
        _, y = torch.topk(y, 2, dim=-1, sorted = False)
        batch_correct = torch.eq(maxk, y).all(dim=-1).sum().item()
        correct += batch_correct
        size += len(X)

        if need_table:
            X = X.cpu()
            y = y.cpu()
            
            # sample first image in batch 
            if X[0].shape[0] == 3:
                X = np.transpose(np.array(X[0]), (1, 2, 0))
            else:
                X = np.array(X[0])
            table.append([wandb.Image(X), y[0], maxk[0], loss, batch_correct])

    test_loss /= num_batches
    correct = (correct / size) * 100
    return correct, test_loss, table

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
root = os.path.dirname(__file__)
current_model = 'SFM' # SFM, mlp, cnn, resnet50, alexnet, lenet, googlenet
dataset = 'rgb_simple_shape' # mnist, fashion, cifar10, malaria, malaria_split, rgb_simple_shape
input_size = (64, 64)
in_channels = 3 # 1, 3
rbf = 'triangle' # gauss, triangle
batch_size = 32
epoch = 200
lr = 0.001
layer = 4
stride = 4
out_channels = 8
description = f""

if current_model == 'SFM': 
    model = SOMNetwork(in_channels=in_channels, out_channels=out_channels, ).to(device)
elif current_model == 'cnn':
    model = CNN(in_channels=in_channels, out_channels = out_channels).to(device)
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
    
run_name = f"{dataset}_{layer}layer_{stride}stride_{rbf}" if current_model == 'SFM' else f"{dataset}_{current_model}"
if not os.path.exists(f"{root}/result/{run_name}"):
   os.makedirs(f"{root}/result/{run_name}")

train_dataloader, test_dataloader = load_data(dataset=dataset, root=root, batch_size=batch_size, input_size=input_size)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="paper experiment",

    name = f"RGB_Plan_v1",

    notes = description,

    group = "RGB_Simple_shape_experiment",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "layer": layer,
    "stride": stride,
    "epochs": epoch,
    "architecture": current_model,
    "dataset": dataset,
    "train_data_num": len(train_dataloader.sampler),
    "test_data_num": len(test_dataloader.sampler),
    "total_data_num": len(train_dataloader.sampler) + len(test_dataloader.sampler),
    "batch_size": batch_size,
    "input shape": (in_channels, *input_size),
    "out_channels": out_channels,
    "SFM filter": "(2, 2)",
    "lr scheduler": "ReduceLROnPlateau",
    "optimizer": "Adam",
    "loss_fn": "BCELoss"
    }
)

print(model)
summary(model, input_size = (in_channels, *input_size))

loss_fn = nn.BCELoss()
# loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer)

wandb.watch(model, loss_fn, log="all", log_freq=1)
train(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, epoch, device = device)

# Train model
train_acc, train_loss, train_table = test(train_dataloader, model, loss_fn, device = device)
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))

# Test model
test_acc, test_loss, test_table = test(test_dataloader, model, loss_fn, device = device)
print("Test: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['train_accuracy'] = train_acc
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['test_accuracy'] = test_acc
wandb.summary['test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})

checkpoint = {'model': SOMNetwork(in_channels=in_channels, out_channels=out_channels),
          'state_dict': model.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'scheduler': scheduler.state_dict()}

torch.save(checkpoint, f'{root}/result/{run_name}/{current_model}_{epoch}_{run_name}_final.pth')
art = wandb.Artifact(f"{current_model}_{run_name}", type="model")
art.add_file(f'{root}/result/{run_name}/{current_model}_{epoch}_{run_name}_final.pth')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()