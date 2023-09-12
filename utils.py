import os
import torch
import collections
import numpy as np
from itertools import repeat
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

root = os.path.dirname(__file__)


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")


def get_rbf(rbf):
    if rbf == 'gauss':
        return gaussian
    elif rbf == 'triangle':
        return triangle
    
def gaussian(d, std):
    return torch.exp(d.pow(2) / (-2 * torch.pow(std, 2)))

def triangle(d, std):
    return torch.ones_like(d) - torch.minimum(d, std)


def test(dataloader: DataLoader, model: nn.Module, loss_fn, device=torch.device('cpu')):
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device); y= y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            size += len(X)

    test_loss /= num_batches
    correct = (correct / size) * 100
    return correct, test_loss


def plot_map(rm, grid_size=None, rowspan=None, colspan = None, cmap='viridis', path=None):
    rows, cols, e_h, e_w, _ = rm.shape
    if rowspan is None:
        rowspan = int(e_h / min(e_h, e_w))
    if colspan is None:
        colspan = int(e_w / min(e_h, e_w))
    if grid_size is None:
        grid_size = (rows*rowspan, cols*colspan)
    fig = plt.figure(figsize=(grid_size[1], grid_size[0]))
    for row in range(rows):
        for col in range(cols):
            ax = plt.subplot2grid(grid_size, (row*rowspan, col*colspan), rowspan=rowspan, colspan=colspan)
            ax.imshow(rm[row][col], cmap=cmap)
            ax.axis('off')
    
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
    
    
def split(input, kernel_size = (5, 5), stride = 4):
    batch, channel, _, _ = input.size()
    split_size = np.subtract(input.size()[2:], kernel_size) // stride + 1
    segments = F.unfold(input, kernel_size=kernel_size, stride=stride).reshape(batch, channel, *kernel_size, *split_size ).permute(0, 1, 4, 5, 2, 3)
    return segments




def infer_data(image, model, rms, save=None):
    paths = [None] * 16
    save_root = f"{root}/{save}"
    if save and not os.path.exists(f"{save_root}"):
        os.makedirs(f"{save_root}")
    if save:
        paths = [
            f"{save_root}/{save}-input.png",
            f"{save_root}/{save}-rm1-crelu.png",
            f"{save_root}/{save}-rm1-max-fm.png",
            f"{save_root}/{save}-rm1-ci-max-1.png",
            f"{save_root}/{save}-rm2-crelu.png",
            f"{save_root}/{save}-rm2-ci-max-1.png",
            f"{save_root}/{save}-rm2-ci-max-5.png",
            f"{save_root}/{save}-rm2-ci-max-10.png",
            f"{save_root}/{save}-rm3-crelu.png",
            f"{save_root}/{save}-rm3-ci-max-1.png",
            f"{save_root}/{save}-rm3-ci-max-5.png",
            f"{save_root}/{save}-rm3-ci-max-10.png",
            f"{save_root}/{save}-rm4-crelu.png",
            f"{save_root}/{save}-rm4-ci-max-1.png",
            f"{save_root}/{save}-rm4-ci-max-5.png",
            f"{save_root}/{save}-rm4-ci-max-10.png",
        ]
    
    
    plt.imshow(image[0].permute(1, 2, 0))
    plt.axis('off')
    if save:
        plt.savefig(paths[0])
    plt.show()
    plt.close()
    
    with torch.no_grad():
        print('----------layer1----------')
        layer1_crelu = model.layer1[0:2](image)
        layer1_crelu_reshape = layer1_crelu.permute(0, 2, 3, 1).reshape(model.layer1[2].shape[0], model.layer1[2].shape[1], *model.layer1[2].kernel_size, 1)
        plot_map(layer1_crelu_reshape.detach().numpy(), path=paths[1])
        plot_map(model.layer1[0].weight[layer1_crelu_reshape.reshape(*model.layer1[2].shape, -1).argmax(-1)].permute(0, 1, 3, 4, 2).detach().numpy(), grid_size=model.layer1[2].shape, path=paths[2])
        print(rms[0][layer1_crelu_reshape.reshape(*model.layer1[2].shape, -1).argmax(-1)][:, :, :, :, None].shape)
        plot_map(rms[0][layer1_crelu_reshape.reshape(*model.layer1[2].shape, -1).argmax(-1)][:, :, :, :, None].detach().numpy(), grid_size=model.layer1[2].shape, path=paths[3])
        print('----------layer2----------')
        layer2_crelu = model.layer2[0:2](model.layer1[2](layer1_crelu))
        layer2_crelu_reshape = layer2_crelu.permute(0, 2, 3, 1).reshape(model.layer2[2].shape[0], model.layer2[2].shape[1], *model.layer2[2].kernel_size, 1)
        plot_map(layer2_crelu_reshape.detach().numpy(), path=paths[4])
        plot_map(rms[1][layer2_crelu_reshape.reshape(*model.layer2[2].shape, -1).argmax(-1)][:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[5])
        plot_map(rms[1][torch.topk(layer2_crelu_reshape.reshape(*model.layer2[2].shape, -1), 5, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[6])
        plot_map(rms[1][torch.topk(layer2_crelu_reshape.reshape(*model.layer2[2].shape, -1), 10, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[7])
        print('----------layer3----------')
        layer3_crelu = model.layer3[0:2](model.layer2[2](layer2_crelu))
        layer3_crelu_reshape = layer3_crelu.permute(0, 2, 3, 1).reshape(model.layer3[2].shape[0], model.layer3[2].shape[1], *model.layer3[2].kernel_size, 1)
        plot_map(layer3_crelu_reshape.detach().numpy(), path=paths[8])
        plot_map(rms[2][layer3_crelu_reshape.reshape(*model.layer3[2].shape, -1).argmax(-1)][:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[9])
        plot_map(rms[2][torch.topk(layer3_crelu_reshape.reshape(*model.layer3[2].shape, -1), 5, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[10])
        plot_map(rms[2][torch.topk(layer3_crelu_reshape.reshape(*model.layer3[2].shape, -1), 10, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[11])
        
        print('----------layer4----------')
        layer4_crelu = model.layer4[0:2](model.layer3[2](layer3_crelu))
        layer4_crelu_reshape = layer4_crelu.permute(0, 2, 3, 1).reshape(1, 1, 35, 35, 1)
        plot_map(layer4_crelu_reshape.detach().numpy(), path=paths[12])
        plot_map(rms[3][layer4_crelu_reshape.reshape(1, 1, -1).argmax(-1)][:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[13])
        plot_map(rms[3][torch.topk(layer4_crelu_reshape.reshape(1, 1, -1), 5, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[14])
        plot_map(rms[3][torch.topk(layer4_crelu_reshape.reshape(1, 1, -1), 10, dim=-1)[1]].mean(2)[:, :, :, :, None], grid_size=model.layer1[2].shape, path=paths[15])
        

def get_ci(input, layer, sfm_filter=(1, 1), n_filters=100):
    output: Tensor
    segments = split(input)
    with torch.no_grad():
        output = layer(input)
        rm_h, rm_w, ci_h, ci_w = (int(segments.shape[2]/sfm_filter[0]), int(segments.shape[3]/sfm_filter[1]), int(segments.shape[4]*sfm_filter[0]), int(segments.shape[5]*sfm_filter[1]))
        segments = segments.reshape(-1, input.shape[1], rm_h, sfm_filter[0], rm_w, sfm_filter[1], segments.shape[4], segments.shape[5]).permute(0, 2, 4, 3, 6, 5, 7, 1).reshape(-1, ci_h, ci_w, input.shape[1])
        segments = segments.permute(0, 3, 1, 2)
        output = output.permute(0, 2, 3, 1).reshape(-1, n_filters)
        CI = torch.empty(n_filters, input.shape[1], ci_h, ci_w)
        for i in range(n_filters):
            CI[i] = segments[output[:, i:i+1].argmax()]
    return CI