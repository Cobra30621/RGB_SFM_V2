import torch
import wandb
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from load_data import load_data
from config import *
import models
from tqdm import tqdm

def test(dataloader: DataLoader, model: nn.Module, loss_fn, device=None):
    progress = tqdm(enumerate(dataloader), desc="Loss: ", total=len(dataloader))
    model.eval()
    losses = 0
    correct = 0
    size = 0
    for batch, (X, y) in progress:
        X = X.to(device); y= y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        losses += loss.detach().item()
        size += len(X)
        
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss = losses/(batch+1)
        test_acc = correct/size
        progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(test_loss, test_acc))

    return test_acc, test_loss

checkpoint = torch.load('pth/face_dataset_pth/0424_SFMCNN_best_jq0g56bj.pth')

model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
model = model.to(config['device'])
model.load_state_dict(checkpoint['model_weights'])
print(model)

train_dataloader, test_dataloader = load_data(dataset=config['dataset'], root=config['root'], batch_size=config['batch_size'], input_size=config['input_shape'])
loss_fn = getattr(nn, config['loss_fn'])()
test_acc, test_loss = test(test_dataloader, model, loss_fn, device = config['device'])

print(test_acc, test_loss)