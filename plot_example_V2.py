from PIL import Image
import torch
import torchvision
import random
import torch.nn.functional as F
import numpy as np

from torchsummary import summary
from torch import nn

from config import *
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from models.RGB_SFMCNN_V2 import RGB_SFMCNN_V2
from dataloader import get_dataloader

import matplotlib

matplotlib.use('Agg')

'''
	對某個資料集產生RM，RM-CI
'''

with torch.no_grad():
    # Load Dataset
    train_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], root=config['root'] + '/data/',
                                                       batch_size=config['batch_size'],
                                                       input_size=config['input_shape'])
    images, labels = torch.tensor([]), torch.tensor([])
    for batch in test_dataloader:
        imgs, lbls = batch
        images = torch.cat((images, imgs))
        labels = torch.cat((labels, lbls))
    print(images.shape, labels.shape)

    # Load Model
    models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN': RGB_SFMCNN, 'RGB_SFMCNN_V2': RGB_SFMCNN_V2}
    checkpoint_filename = 'RGB_SFMCNN_V2_best'
    checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth', weights_only=True)
    model = models[arch['name']](**dict(config['model']['args']))
    model.load_state_dict(checkpoint['model_weights'])
    model.cpu()
    model.eval()
    summary(model, input_size=(config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
    print(model)

    # Test Model
    batch_num = 1000
    pred = model(images[:batch_num])
    y = labels[:batch_num]
    correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    print("Test Accuracy: " + str(correct / len(pred)))
    # input()

FMs = {}
if arch['args']['in_channels'] == 1:
    FMs[0] = model.convs[0][0].weight.reshape(-1, *arch['args']['Conv2d_kernel'][0], 1)
    print(f'FM[0] shape: {FMs[0].shape}')
    FMs[1] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1] ** 0.5),
                                              int(model.convs[1][0].weight.shape[1] ** 0.5), 1)
    print(f'FM[1] shape: {FMs[1].shape}')
    FMs[2] = model.convs[2][0].weight.reshape(-1, int(model.convs[2][0].weight.shape[1] ** 0.5),
                                              int(model.convs[2][0].weight.shape[1] ** 0.5), 1)
    print(f'FM[2] shape: {FMs[2].shape}')
    FMs[3] = model.convs[3][0].weight.reshape(-1, int(model.convs[3][0].weight.shape[1] ** 0.5),
                                              int(model.convs[3][0].weight.shape[1] ** 0.5), 1)
    print(f'FM[3] shape: {FMs[3].shape}')
else:
    # 平行架構
    kernel_size = arch['args']['Conv2d_kernel'][0]
    weights = torch.concat([model.RGB_convs[0][0].transform_weights()])
    weights = weights.reshape(arch['args']['channels'][0][0], arch['args']['in_channels'], 1, 1)
    weights = weights.repeat(1, 1, *kernel_size)
    FMs['RGB_convs_0'] = weights
    print(f'FM[RGB_convs_0] shape: {FMs["RGB_convs_0"].shape}')

    FMs['RGB_convs_1'] = model.RGB_convs[2][0].weight.reshape(-1, 2, 15, 1)
    print(f'FM[RGB_convs_1] shape: {FMs["RGB_convs_1"].shape}')

    FMs['RGB_convs_2'] = model.RGB_convs[3][0].weight.reshape(-1, int(model.RGB_convs[3][0].weight.shape[1] ** 0.5),
                                                              int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), 1)
    print(f'FM[RGB_convs_2] shape: {FMs["RGB_convs_2"].shape}')

    FMs['Gray_convs_0'] = model.Gray_convs[0][0].weight.reshape(arch['args']['channels'][1][0], 1, *kernel_size)
    print(f'FM[Gray_convs_0] shape: {FMs["Gray_convs_0"].shape}')
    FMs['Gray_convs_1'] = model.Gray_convs[2][0].weight.reshape(-1, 7, 10, 1)
    print(f'FM[Gray_convs_1] shape: {FMs["Gray_convs_1"].shape}')
    FMs['Gray_convs_2'] = model.Gray_convs[3][0].weight.reshape(-1, int(model.Gray_convs[3][0].weight.shape[1] ** 0.5),
                                                                int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), 1)
    print(f'FM[Gray_convs_2] shape: {FMs["Gray_convs_2"].shape}')

layers = get_layers(model)

CIs = {}
kernel_size = arch['args']['Conv2d_kernel'][0]
stride = (arch['args']['strides'][0], arch['args']['strides'][0])
if arch['args']['in_channels'] == 1:
    CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
    CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride,
                                       sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
    CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride,
                                       sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
    CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride,
                                       sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
else:

    CIs["RGB_convs_0"], CI_idx, CI_values = get_ci(images, layers['RGB_convs_0'], kernel_size, stride=stride)
    CIs["RGB_convs_1"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_1"], kernel_size=kernel_size,
                                                   stride=stride,
                                                   sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]),
                                                                         dim=0))
    CIs["RGB_convs_2"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_2"], kernel_size=kernel_size,
                                                   stride=stride,
                                                   sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]),
                                                                         dim=0))

    CIs["Gray_convs_0"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_convs_0'], kernel_size,
                                                    stride=stride)
    CIs["Gray_convs_1"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_1"],
                                                    kernel_size=kernel_size, stride=stride,
                                                    sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]),
                                                                          dim=0))
    CIs["Gray_convs_2"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_2"],
                                                    kernel_size=kernel_size, stride=stride,
                                                    sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]),
                                                                          dim=0))

if config['dataset'] == 'Colored_MNIST' or config['dataset'] == 'Colored_FashionMNIST':
    label_to_idx = {}
    i = 0
    for c in ['red', 'green', 'blue']:
        for n in range(10):
            label_to_idx[c + '_' + str(n)] = i
            i += 1
    idx_to_label = {value: key for key, value in label_to_idx.items()}
elif config['dataset'] == 'AnotherColored_MNIST' or config['dataset'] == 'AnotherColored_FashionMNIST':
    label_to_idx = {}
    colors = {
        'brown': [151, 74, 0],
        'light_blue': [121, 196, 208],
        'light_pink': [221, 180, 212]
    }
    i = 0
    for c in colors.keys():
        for n in range(10):
            label_to_idx[c + '_' + str(n)] = i
            i += 1
    idx_to_label = {value: key for key, value in label_to_idx.items()}

example_num = 450


# 繪製反應 RM 圖，並存到陣列中
def plot_RM_then_save(layer_num, plot_shape, img, save_path, is_gray = False, figs = None, titles = None):
    fig = plot_RM_map(layer_num, plot_shape, img, save_path, is_gray)

    if figs is not None:
        figs.append(fig)
    if titles is not None:
        titles.append(f'Layer{layer_num}_RM')

# 繪製反應 RM 圖
def plot_RM_map(layer_num, plot_shape, img, save_path, is_gray = False):

    if is_gray:
        RM = layers[layer_num](model.gray_transform(img.unsqueeze(0)))[0]
    else:
        RM = layers[layer_num](img.unsqueeze(0))[0]

    # print(f"Layer{layer_num}_RM: {RM.shape}")

    RM_H, RM_W = RM.shape[1], RM.shape[2]
    return plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
             path=save_path + f'{layer_num}_RM')


save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example'
if os.path.exists(save_path):
    shutil.rmtree(save_path)  # 刪除資料夾及其內容
    os.makedirs(save_path)  # 重新建立資料夾


def process_image(image, label, test_id):
    print(test_id)
    save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example/{label.argmax().item()}/example_{test_id}/'
    RM_save_path = f'{save_path}/RMs/'
    RM_CI_save_path = f'{save_path}/RM_CIs/'
    os.makedirs(RM_save_path, exist_ok=True)
    os.makedirs(RM_CI_save_path, exist_ok=True)

    if arch['args']['in_channels'] == 1:
        torchvision.utils.save_image(image, save_path + f'origin_{test_id}.png')
    else:
        plt.imsave(save_path + f'origin_{test_id}.png', image.permute(1, 2, 0).detach().numpy())

    segments = split(image.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0],
                     stride=(arch['args']['strides'][0], arch['args']['strides'][0]))[0]

    RM_CIs = {}
    RM_figs = []
    RM_title = []

    fig = plot_map(segments.permute(1, 2, 3, 4, 0), path=save_path + f'origin_split_{test_id}.png')
    RM_figs.append(fig)
    RM_title.append('Origin_Split')

    if arch['args']['in_channels'] == 1:
        # Layer 0
        layer_num = 0
        RM = layers[layer_num](image.unsqueeze(0))[0]
        print(f"Layer{layer_num}_RM: {RM.shape}")
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5),
                                             1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, 1)
        plot_map(RM_CI.detach().numpy(), path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
        RM_CIs[layer_num] = RM_CI

        # Layer 1
        layer_num = 1
        RM = layers[layer_num](image.unsqueeze(0))[0]
        print(f"Layer{layer_num}_RM: {RM.shape}")
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5),
                                             1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI.detach().numpy(), path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
        RM_CIs[layer_num] = RM_CI

        # Layer 2
        layer_num = 2
        RM = layers[layer_num](image.unsqueeze(0))[0]
        print(f"Layer{layer_num}_RM: {RM.shape}")
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5),
                                             1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI.detach().numpy(), path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
        RM_CIs[layer_num] = RM_CI

        # Layer 3
        layer_num = 3
        RM = layers[layer_num](image.unsqueeze(0))[0]
        print(f"Layer{layer_num}_RM: {RM.shape}")
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5),
                                             1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI.detach().numpy(), path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
        RM_CIs[layer_num] = RM_CI


    else:
        ################################### RGB ###################################
        ### RGB_convs_0 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'RGB_convs_0'
        plot_shape = (5, 6)

        # 繪製只跑 Conv(卷積) 的 RM
        plot_RM_then_save('RGB_convs_0_Conv', plot_shape, image, RM_save_path,
                          False, RM_figs, RM_title)

        RM = layers[layer_num](image.unsqueeze(0))[0]
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
        RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0, 2, 3, 1).reshape(
            RM_H, RM_W, FM_H, FM_W, arch['args']['in_channels'])
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
        RM_CI = RM_CI.permute(0, 1, 4, 2, 3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(
            dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
        RM_CIs[layer_num] = RM_CI
        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        RM_figs.append(fig)
        RM_title.append(layer_num)

        plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')


        # 繪製 SFM 合併後的 RM
        plot_RM_then_save('RGB_convs_0_SFM', plot_shape, image, RM_save_path,
                          False, RM_figs, RM_title)

        ### RGB_convs_1 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'RGB_convs_1'
        RM = layers[layer_num](image.unsqueeze(0))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        # 繪製 只跑 Conv(卷積) 的 RM
        plot_RM_then_save('RGB_convs_1_Conv', plot_shape, image, RM_save_path,
                          False, RM_figs, RM_title)

        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
        # 將RM_CI取每個小圖的代表色塊後合併成為新的RM_CI
        RM_CI = RM_CI.reshape(RM_H, RM_W, CI_H // RM_CIs['RGB_convs_0'].shape[2], RM_CIs['RGB_convs_0'].shape[2],
                              CI_W // RM_CIs['RGB_convs_0'].shape[3], RM_CIs['RGB_convs_0'].shape[3],
                              arch['args']['in_channels'])
        RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6)
        RM_CI = RM_CI.reshape(RM_CIs['RGB_convs_0'].shape)
        RM_CI = RM_CI.permute(0, 1, 4, 2, 3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(
            dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
        RM_CI = RM_CI.reshape(RM_H, RM_CI.shape[0] // RM_H, RM_W, RM_CI.shape[1] // RM_W, *RM_CI.shape[2:])
        RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6).reshape(RM_H, RM_W, CI_H, CI_W, 3)
        RM_CIs[layer_num] = RM_CI
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')
        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        RM_figs.append(fig)
        RM_title.append(layer_num)


        # 繪製 SFM 合併後的 RM
        plot_RM_then_save('RGB_convs_1_SFM', plot_shape, image, RM_save_path,
                          False, RM_figs, RM_title)

        ### RGB_convs_2
        # 跑完響應模組，SFM 合併前
        layer_num = 'RGB_convs_2'
        RM = layers[layer_num](image.unsqueeze(0))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 繪製只跑 Conv(卷積) 的 RM
        plot_RM_then_save('RGB_convs_2_Conv', plot_shape, image, RM_save_path,
                          False, RM_figs, RM_title)

        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
        # 將RM_CI取每個小圖的代表色塊後合併成為新的RM_CI
        RM_CI = RM_CI.reshape(RM_H, RM_W, CI_H // RM_CIs['RGB_convs_0'].shape[2], RM_CIs['RGB_convs_0'].shape[2],
                              CI_W // RM_CIs['RGB_convs_0'].shape[3], RM_CIs['RGB_convs_0'].shape[3],
                              arch['args']['in_channels'])
        RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6)
        RM_CI = RM_CI.reshape(RM_CIs['RGB_convs_0'].shape)
        RM_CI = RM_CI.permute(0, 1, 4, 2, 3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(
            dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
        RM_CI = RM_CI.reshape(RM_H, RM_CI.shape[0] // RM_H, RM_W, RM_CI.shape[1] // RM_W, *RM_CI.shape[2:])
        RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6).reshape(RM_H, RM_W, CI_H, CI_W, 3)
        RM_CIs[layer_num] = RM_CI
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')
        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                       path=RM_save_path + f'{layer_num}_RM')
        RM_figs.append(fig)
        RM_title.append(layer_num)


        combine_images(RM_figs, RM_title, RM_save_path + f'RGB_combine')


        ################################### Gray ###################################
        ### Gray_convs_0 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'Gray_convs_0'
        plot_shape = (7, 10)
        RM = layers[layer_num](model.gray_transform(image.unsqueeze(0)))[0]
        print(f"{layer_num}_RM: {RM.shape}")
        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, 1)
        RM_CIs[layer_num] = RM_CI
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

        # 只跑 Conv(卷積)
        plot_RM_map('Gray_convs_0_Conv', plot_shape, image, RM_save_path, is_gray = True)
        # SFM 合併後
        plot_RM_map('Gray_convs_0_SFM', plot_shape, image, RM_save_path, is_gray = True)


        ### Gray_convs_1 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'Gray_convs_1'
        RM = layers[layer_num](model.gray_transform(image.unsqueeze(0)))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, 1)
        RM_CIs[layer_num] = RM_CI
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

        # 只跑 Conv(卷積)
        plot_RM_map('Gray_convs_1_Conv', plot_shape, image, RM_save_path, is_gray=True)
        # SFM 合併後
        plot_RM_map('Gray_convs_1_SFM', plot_shape, image, RM_save_path, is_gray=True)


        ### Gray_convs_2 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'Gray_convs_2'
        RM = layers[layer_num](model.gray_transform(image.unsqueeze(0)))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, 1)
        RM_CIs[layer_num] = RM_CI
        plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

        # 只跑 Conv(卷積)
        plot_RM_map('Gray_convs_2_Conv', plot_shape, image, RM_save_path, is_gray=True)

    plt.close('all')



# 新增的迴圈來讀取指定資料夾中的圖片
# input_folder = f'./detect/{config["dataset"]}_{checkpoint_filename}/test_images/'  # 指定資料夾路徑
# image_files = os.listdir(input_folder)
#
# for image_file in image_files:
#     if image_file.endswith(('.png', '.jpg', '.jpeg')):  # 檢查檔案類型
#         # 讀取圖片
#         image_path = os.path.join(input_folder, image_file)
#         test_image = Image.open(image_path).convert('RGB')  # 轉換為 RGB 格式
#         test_image = torchvision.transforms.ToTensor()(test_image)  # 轉換為 Tensor
#
#
#         # 提取標籤，假設標籤在檔名中
#         label = torch.zeros(10)  # 假設有 10 個類別
#         label[9] = 1  # 將對應的標籤設為 1
#
#
#         # 執行 process_image 函數
#         process_image(test_image, label, image_file)

# # 針對整個資料集
for test_id in range(example_num):
    process_image(images[test_id], labels[test_id], test_id)

