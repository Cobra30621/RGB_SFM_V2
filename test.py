from load_data import load_data
from utils import *

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Grayscale

import matplotlib.pyplot as plt
import torch
import os

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
    correct = (correct / size) * 100
    return correct, test_loss, table

def showlayer(model:nn.Module, X, y, save_dir):
    model.eval()
    print(model)
    FMs = get_FM_img(model)
    RMs = get_RM_img(model, X)
    CIs = get_CI_img(model, X)

    if save_dir == None:
        return

    for key in FMs:
        print(f'FMs[{key}] saving ...')
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
        RM_save_dir = save_dir + f'/correct_{y[(torch.concat((y == target, maxk == target), dim=-1).all(dim=-1))][0].detach().cpu().item()}'
        save_RM_CI(model, X, torch.concat((y == target, maxk == target), dim=-1).all(dim=-1), RMs, CIs, RM_save_dir)

        #儲存y = target的錯誤案例
        RM_save_dir = save_dir + f'/incorrect_{target}_{maxk[torch.concat((y == target, maxk != target), dim=-1).all(dim=-1)][0].detach().cpu().item()}'
        save_RM_CI(model, X, torch.concat((y == target, maxk != target), dim=-1).all(dim=-1), RMs, CIs, RM_save_dir)

    # #儲存正確的案例
    # RM_save_dir = save_dir + f'/correct_{y[(maxk == y)][0]}'
    # save_RM_CI(model, X, (maxk == y).reshape(-1), RMs, CIs, RM_save_dir)

    # #儲存錯誤的案例
    # RM_save_dir = save_dir + f'/incorrect_{y[(maxk != y)][0]}_{maxk[(maxk != y)][0]}'
    # save_RM_CI(model, X, (maxk != y).reshape(-1), RMs, CIs, RM_save_dir)

def get_FM_img(model:nn.Module):
    FMs={}
    FMs['rgb'] = torch.repeat_interleave(torch.repeat_interleave(model.RGB_preprocess[0].rgb_weight.reshape(36, 3, 1, 1), model.RGB_preprocess[0].kernel_size[0], dim=2), model.RGB_preprocess[0].kernel_size[1], dim=3)
    FMs['rgb'] = FMs['rgb'].reshape(6, 6, 3, 5, 5).permute(0, 1, 3, 4, 2)
    FMs['gray'] = model.GRAY_preprocess[0].weight.reshape(8, 8, 5, 5, 1)
    FMs[1] = model.layer2[0].weight.permute(0, 2, 3, 1).reshape(15, 15, 10, 10, 1)
    FMs[2] = model.layer2[3].weight.permute(0, 2, 3, 1).reshape(25, 25, 15, 15, 1)
    FMs[3] = model.layer2[6].weight.permute(0, 2, 3, 1).reshape(35, 35, 25, 25, 1)

    return FMs

def get_RM_img(model:nn.Module, X):
    RGB_output = model.RGB_preprocess(X)
    GRAY_output = model.GRAY_preprocess(Grayscale()(X))
    input = torch.concat((RGB_output, GRAY_output), dim=1)

    RMs={}
    RMs['rgb'] = get_RM(input[:, :36, :, :], (6, 6, 6, 6, 1))
    RMs['gray'] = get_RM(input[:, 36:, :, :], (6, 6, 8, 8, 1))
    RMs[0] = get_RM(input[:, :, :, :], (6, 6, 10, 10, 1))
    RMs[1] = get_RM(torch.nn.Sequential(model.layer1 + model.layer2[0:2])(input), (3, 3, 15, 15, 1))
    RMs[2] = get_RM(torch.nn.Sequential(model.layer1 + model.layer2[0:5])(input), (3, 1, 25, 25, 1))
    RMs[3] = get_RM(torch.nn.Sequential(model.layer1 + model.layer2)(input), (1, 1, 35, 35, 1))
    return RMs

def get_CI_img(model:nn.Module, X):
    RGB_output = model.RGB_preprocess(X)
    GRAY_output = model.GRAY_preprocess(Grayscale()(X))
    input = torch.concat((RGB_output, GRAY_output), dim=1)

    CIs = {}
    pred = torch.nn.Sequential(*(list(model.layer1)+list(model.layer2[:1])))(input)
    CIs[1] = get_ci(X, pred, sfm_filter=model.layer1[0].filter, n_filters = model.layer2[0].weight.shape[0])
    pred = torch.nn.Sequential(*(list(model.layer1)+list(model.layer2[:4])))(input)
    CIs[2] = get_ci(X, pred, sfm_filter=tuple(np.multiply(model.layer1[0].filter, model.layer2[2].filter)), n_filters = model.layer2[3].weight.shape[0])
    pred = torch.nn.Sequential(*(list(model.layer1)+list(model.layer2[:7])))(input)
    CIs[3] = get_ci(X, pred, sfm_filter=tuple(np.multiply(np.multiply(model.layer1[0].filter, model.layer2[2].filter), model.layer2[5].filter)), n_filters = model.layer2[6].weight.shape[0])
    return CIs

def save_RM_CI(model:nn.Module, X, filter, RMs, CIs, RM_save_dir):
    if len(X[filter]) != 0: 
        Path(RM_save_dir).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.imshow(X[filter][0].permute(1, 2, 0).detach().cpu().numpy())
        plt.axis('off')
        plt.savefig(RM_save_dir + '/input.png', bbox_inches='tight')

        plt.clf()
        plt.imshow(Grayscale()(X)[filter][0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(RM_save_dir + '/input_Gray.png', bbox_inches='tight')
        
        for key in RMs:
            print(f'{RM_save_dir} \t RMs[{key}] saving\t{RMs[key][filter][0].shape}')
            plot_map(RMs[key][filter][0].detach().cpu().numpy(), path = RM_save_dir + f'/RMs_{key}.png')
        
        for key in CIs:
            _, tmp = torch.topk(RMs[key][filter][0].reshape(RMs[key][filter][0].shape[0] * RMs[key][filter][0].shape[1], -1), k=1, dim=1)
            print(f'{RM_save_dir} \t CIs[{key}] saving\t{CIs[key][tmp.cpu()].reshape(*model.layer2[2].shape, CIs[key].shape[-3], 10, 10).permute(0, 1, 3, 4, 2).shape}')
            plot_map(CIs[key][tmp.cpu()].reshape(*model.shape[key], CIs[key].shape[-3], CIs[key].shape[-2], CIs[key].shape[-1]).permute(0, 1, 3, 4, 2), path = RM_save_dir + f'/CIs_{key}.png')
    else:
        print(f"This batch don't have label {y[filter]}")

if __name__ == '__main__':
    path = './runs/train/exp/SFM_200_original_RGB_Plan_v1.pth'
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_weights'])

    dataset = 'rgb_simple_shape'
    batch_size = 32
    root = os.path.dirname(__file__)
    input_size = (28, 28)
    train_dataloader, test_dataloader = load_data(dataset=dataset, root=root, batch_size=batch_size, input_size=input_size)
    data, label = next(iter(test_dataloader))
    for X, y in test_dataloader:
        data = torch.concat((data, X), dim=0)
        label = torch.concat((label, y), dim=0)
    X, y = data.to('cuda'), label.to('cuda')
    
    save_dir = increment_path('./runs/detect/exp', exist_ok = False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    showlayer(model, X, y, save_dir)

    # showlayer(model, X, y, None)
    




