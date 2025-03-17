from PIL import Image
import torch
import torchvision
import random
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import ( GradCAM, HiResCAM, GradCAMPlusPlus,
                              GradCAMElementWise, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                              LayerCAM, KPCA_CAM)

from torchsummary import summary
from torch import nn

from config import *
from plot_heatmap import get_cam_target_layers, get_each_layers_cam, plot_RM_CI_with_cam_mask, \
    plot_cams_on_image, get_reduced_cam, plot_reduced_cam
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from models.RGB_SFMCNN_V2 import RGB_SFMCNN_V2
from dataloader import get_dataloader

import matplotlib

# 繪製熱圖
PLOT_HEATMAP = True
heatmap_methods = [GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, AblationCAM,
                   ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, KPCA_CAM]


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
def plot_RM_then_save(layer_num, plot_shape, img, save_path, is_gray = False, figs = None):
    fig = plot_RM_map(layer_num, plot_shape, img, save_path, is_gray)

    if figs is not None:
        figs[layer_num] = fig

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

    RM_figs = {}

    RM_CI_figs = {}

    origin_img = plot_map(segments.permute(1, 2, 3, 4, 0), path=save_path + f'origin_split_{test_id}.png')
    RM_figs['Origin_Split'] = origin_img
    RM_CI_figs['Origin_Split'] = origin_img

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
                          False, RM_figs)

        RM = layers[layer_num](image.unsqueeze(0))[0]
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
        RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0, 2, 3, 1).reshape(
            RM_H, RM_W, FM_H, FM_W, arch['args']['in_channels'])
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        RM_CI_figs[layer_num] = plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
        RM_CI = RM_CI.permute(0, 1, 4, 2, 3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(
            dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)


        RM_CIs[layer_num] = RM_CI
        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        RM_figs[layer_num] = fig


        plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')


        # 繪製 SFM 合併後的 RM
        plot_RM_then_save('RGB_convs_0_SFM', plot_shape, image, RM_save_path,
                          False, RM_figs)

        ### RGB_convs_1 ###
        # 跑完響應模組，SFM 合併前
        layer_num = 'RGB_convs_1'
        RM = layers[layer_num](image.unsqueeze(0))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        # 繪製 只跑 Conv(卷積) 的 RM
        plot_RM_then_save('RGB_convs_1_Conv', plot_shape, image, RM_save_path,
                          False, RM_figs)

        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        RM_CIs[layer_num] = RM_CI
        RM_CI_figs[layer_num] = plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')


        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                 path=RM_save_path + f'{layer_num}_RM')
        RM_figs[layer_num] = fig


        # 繪製 SFM 合併後的 RM
        plot_RM_then_save('RGB_convs_1_SFM', plot_shape, image, RM_save_path,
                          False, RM_figs)

        ### RGB_convs_2
        # 跑完響應模組，SFM 合併前
        layer_num = 'RGB_convs_2'
        RM = layers[layer_num](image.unsqueeze(0))[0]
        plot_shape = (int(RM.shape[0] ** 0.5), int(RM.shape[0] ** 0.5))
        print(f"Layer{layer_num}_RM: {RM.shape}")
        # 繪製只跑 Conv(卷積) 的 RM
        plot_RM_then_save('RGB_convs_2_Conv', plot_shape, image, RM_save_path,
                          False, RM_figs)

        # 存RM_FM、RM_CI
        RM_H, RM_W = RM.shape[1], RM.shape[2]
        CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
        RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H, RM_W, CI_H,
                                                                                                   CI_W, arch['args'][
                                                                                                       'in_channels'])
        RM_CIs[layer_num] = RM_CI
        RM_CI_figs[layer_num] = plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
        plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')
        # 繪製 RM
        fig = plot_map(RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
                       path=RM_save_path + f'{layer_num}_RM')
        RM_figs[layer_num] = fig

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
        RM_CI_figs[layer_num] = plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

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
        RM_CI_figs[layer_num] = plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

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
        RM_CI_figs[layer_num] =plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

        # 只跑 Conv(卷積)
        plot_RM_map('Gray_convs_2_Conv', plot_shape, image, RM_save_path, is_gray=True)

        # 繪製 RM 的合併圖
        plot_combine_images(RM_figs, RM_save_path + f'RGB_combine')
        # 繪製 RM_CI 的合併圖
        plot_combine_images(RM_CI_figs, RM_CI_save_path + f'combine')

        ################################### heatmap ###################################

        # 創建保存熱圖的目錄
        cam_save_pth = os.path.join(save_path, 'heatmap')
        os.makedirs(cam_save_pth, exist_ok=True)

        # 獲取標籤的索引值
        label_arg = label.argmax().item()
        # 獲取需要生成 CAM 的目標層
        heatmap_layers = get_cam_target_layers(model)
        # 設定使用 GradCAM 方法
        method = GradCAM

        # 對每一層生成 CAM (Class Activation Map)
        cams = get_each_layers_cam(
            model=model,
            target_layers=heatmap_layers,
            label=label_arg,
            input_tensor=image,
            cam_method=method)


        # 初始化存儲 CAM 遮罩後的 RM_CI 圖和縮減後的 CAM 圖的字典
        RM_CI_cams = {}
        reduced_cams = {}
        # 將原始分割圖加入字典
        RM_CI_cams['Origin_Split'] = origin_img
        reduced_cams['Origin_Split'] = origin_img

        # 對每一層的 RM_CI 進行處理
        for layer_name, RM_CI in RM_CIs.items():
            print(f"{layer_name}, {RM_CI.shape}")

            # 設定輸出形狀為 RM_CI 的高寬
            output_shape = (RM_CI.shape[0], RM_CI.shape[1])
            # 將 CAM 縮減到指定大小
            reduced_cam = get_reduced_cam(cams[layer_name], output_shape)

            # 繪製並保存縮減後的 CAM
            reduced_cams[layer_name] = plot_reduced_cam(reduced_cam)
            # 繪製並保存使用 CAM 遮罩後的 RM_CI 圖
            RM_CI_cams[layer_name] = plot_RM_CI_with_cam_mask(RM_CI, reduced_cam, RM_CI_save_path)

        # 將所有使用 CAM 遮罩後的 RM_CI 圖組合成一張圖並保存
        plot_combine_images(RM_CI_cams, RM_CI_save_path + f'/cam_RM_CI_combine')
        # 將所有縮減後的 CAM 圖組合成一張圖並保存
        plot_combine_images(reduced_cams, cam_save_pth + f'/reduced_cam_combine')
        # 在原始圖像上繪製 CAM 並保存
        plot_cams_on_image(image, cams, cam_save_pth, method.__name__)
 

    plt.close('all')



# # 針對整個資料集
for test_id in range(example_num):
    process_image(images[test_id], labels[test_id], test_id)

