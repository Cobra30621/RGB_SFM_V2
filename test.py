from load_data import load_data

from utils import *
import matplotlib.pyplot as plt
import torch
import os
import wandb

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale
from RGB_Plan_v8 import SOMNetwork, Visualize

def eval(dataloader: DataLoader, model: nn.Module, loss_fn, need_table = True, device=None):
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

        _, maxk = torch.topk(pred, 1, dim = -1, sorted = False)
        _, y = torch.topk(y, 1, dim=-1, sorted = False)
        batch_correct = torch.eq(maxk, y).sum().detach().item()
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
    correct = (correct / size)
    return correct, test_loss, table

def showlayer(model:nn.Module, X, y, train_data, save_dir):
    model.eval()
    visualize = Visualize(model)
    FMs = visualize.get_FM_img()
    RMs = visualize.get_RM_img(X)
    CIs = visualize.get_CI_img(train_data)

    if save_dir == None:
        return

    for key in FMs:
        print(f'FMs[{key}] = {FMs[key].shape}')
        plot_map(FMs[key].detach().cpu().numpy(), path = save_dir + f'/FMs_{key}.png')

    in_channels = 3
    for key in CIs:
        print(f'CIs[{key}] = {CIs[key].shape}')
        print(CIs[key].reshape(int(len(model.layer2[3 * (key - 1)].weight)**0.5), int(len(model.layer2[3 * (key - 1)].weight)**0.5), CIs[key].shape[1], CIs[key].shape[2], CIs[key].shape[3]).permute(0, 1, 3, 4, 2).shape)
        plot_map(CIs[key].reshape(int(len(model.layer2[3 * (key - 1)].weight)**0.5), int(len(model.layer2[3 * (key - 1)].weight)**0.5), CIs[key].shape[1], CIs[key].shape[2], CIs[key].shape[3]).permute(0, 1, 3, 4, 2), path = save_dir + f'/CI_{key+1}.png')

        
    pred = model(X)
    _, maxk = torch.topk(pred, 1, dim = -1, sorted = False)
    _, y = torch.topk(y, 1, dim=-1, sorted = False)

    for target in range(15):
        #儲存y = target的正確案例
        if len(X[(torch.concat((y == target, maxk == target), dim=-1).all(dim=-1))]) != 0:
            RM_save_dir = save_dir + f'/correct_{y[(torch.concat((y == target, maxk == target), dim=-1).all(dim=-1))][0].detach().cpu().item()}'
            visualize.save_RM_CI(X, torch.concat((y == target, maxk == target), dim=-1).all(dim=-1), RMs, CIs, RM_save_dir)

        #儲存y = target的錯誤案例
        if len(X[(torch.concat((y == target, maxk != target), dim=-1).all(dim=-1))]) != 0:
            RM_save_dir = save_dir + f'/incorrect_{target}_{maxk[torch.concat((y == target, maxk != target), dim=-1).all(dim=-1)][0].detach().cpu().item()}'
            visualize.save_RM_CI(X, torch.concat((y == target, maxk != target), dim=-1).all(dim=-1), RMs, CIs, RM_save_dir)

    # #儲存正確的案例
    # RM_save_dir = save_dir + f'/correct_{y[(maxk == y)][0]}'
    # save_RM_CI(model, X, (maxk == y).reshape(-1), RMs, CIs, RM_save_dir)

    # #儲存錯誤的案例
    # RM_save_dir = save_dir + f'/incorrect_{y[(maxk != y)][0]}_{maxk[(maxk != y)][0]}'
    # save_RM_CI(model, X, (maxk != y).reshape(-1), RMs, CIs, RM_save_dir)


if __name__ == '__main__':
    path = './runs/11_06/RGB_Plan_v8_rgbBlock_initial/RGB_Plan_v8_epochs200.pth'
    checkpoint = torch.load(path)
    model = SOMNetwork(3,15,4)
    model.load_state_dict(checkpoint['model_weights'])

    dataset = 'rgb_simple_shape'
    batch_size = 32
    root = os.path.dirname(__file__)
    input_size = (28, 28)
    train_dataloader, test_dataloader = load_data(dataset=dataset, root=root, batch_size=batch_size, input_size=input_size)
    train_data, train_label = next(iter(train_dataloader))
    data, label = next(iter(test_dataloader))
    for X, y in train_dataloader:
        train_data = torch.concat((train_data, X), dim=0)
        train_label = torch.concat((train_label, y), dim=0)
    train_data = train_data.to('cuda')

    for X, y in test_dataloader:
        data = torch.concat((data, X), dim=0)
        label = torch.concat((label, y), dim=0)
    X, y = data.to('cuda'), label.to('cuda')
    
    save_dir = increment_path('./runs/detect/exp', exist_ok = False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(save_dir)
    showlayer(model, X, y, train_data,save_dir)

    




