from pytorch_grad_cam import ( GradCAM, HiResCAM, GradCAMPlusPlus,
                              GradCAMElementWise, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                              LayerCAM, KPCA_CAM)

from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, display_image_comparison
from load_tools import load_model_and_data
from plot_cam import  generate_cam_visualizations
from utils import *

import matplotlib


PLOT_CAM = False
# PLOT_CAM = True
# 使用影像前處理
use_preprocessed_image= config['use_preprocessed_image']

matplotlib.use('Agg')

'''
	對某個資料集產生RM，RM-CI
'''

checkpoint_filename = 'RGB_SFMCNN_V2_best'
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename)

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


    RM_CIs = {}
    RM_figs = {}

    RM_CI_figs = {}

    # 繪製原始圖片
    fig_origin = plt.figure(figsize=(5, 5), facecolor="white")
    plt.imshow(image.permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.savefig(save_path + f'origin_{test_id}.png', bbox_inches='tight', pad_inches=0)
    origin_img = fig_origin
    RM_CI_figs['Origin'] = origin_img

    if use_preprocessed_image:
        scaled_img, blurred_img, high_contrast_img, final_img = preprocess_retinal_tensor_image(image)
        display_image_comparison(save_path= save_path + f'preprocess.png', origin_img=image,  final_img=final_img)
        # 繪製前處理後圖片
        fig_origin = plt.figure(figsize=(5, 5), facecolor="white")
        plt.imshow(final_img.permute(1, 2, 0).detach().numpy())
        plt.axis('off')
        plt.savefig(save_path + f'preprocess_{test_id}.png', bbox_inches='tight', pad_inches=0)
        origin_img = fig_origin

        RM_CI_figs['Preprocess'] = origin_img

    # 繪製灰階圖片
    fig_gray = plt.figure(figsize=(5, 5), facecolor="white")
    gray_image = model.gray_transform(image.unsqueeze(0))[0]  # 使用模型中的灰階轉換
    plt.imshow(gray_image.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(save_path + f'gray_{test_id}.png', bbox_inches='tight', pad_inches=0)
    gray_img = fig_gray

    segments = split(image.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0],
                     stride=(arch['args']['strides'][0], arch['args']['strides'][0]))[0]



    origin_split_img = plot_map(segments.permute(1, 2, 3, 4, 0), path=save_path + f'origin_split_{test_id}.png')
    RM_figs['Origin_Split'] = origin_split_img


    RM_CI_figs['Origin_Split'] = origin_split_img

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

        RM_CI_figs['Gray'] = gray_img

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
        RM_CI_combine_fig = plot_combine_images(RM_CI_figs, RM_CI_save_path + f'combine')

        ################################### CAM ###################################

        if PLOT_CAM:
            # 定義所有要使用的 CAM 方法
            cam_methods = [GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, AblationCAM,
                               ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, KPCA_CAM]
            # 如果只想測試部分方法,可以使用下面這行
            # cam_methods = [GradCAM, HiResCAM, GradCAMPlusPlus]

            # 創建兩個字典來存儲不同方法生成的圖像
            cam_figs = {}  # 存儲 CAM 可視化結果
            RM_CI_figs = {}  # 存儲 RM-CI 可視化結果
            RM_CI_figs['raw'] = RM_CI_combine_fig  # 保存原始的 RM-CI 組合圖

            # 對每個 CAM 方法進行處理
            for method in cam_methods:
                print(f"drawing {method.__name__}")
                # 生成 CAM 可視化結果
                cam_fig, RM_CI_fig = generate_cam_visualizations(
                    model=model,          # 模型
                    label=label.argmax().item(),  # 預測標籤
                    image=image,          # 輸入圖像
                    origin_img=origin_split_img,  # 原始圖像
                    RM_CIs=RM_CIs,         # RM-CI 數據
                    save_path=RM_CI_save_path,  # 保存路徑
                    method=method         # CAM 方法
                )
                # 將結果保存到對應的字典中
                cam_figs[f'{method.__name__}'] = cam_fig
                RM_CI_figs[f'{method.__name__}'] = RM_CI_fig


            # 將所有 CAM 結果垂直組合並保存
            plot_combine_images_vertical(cam_figs, RM_CI_save_path + f'cam/cams_combine')
            # 將所有 RM-CI 結果垂直組合並保存
            plot_combine_images_vertical(RM_CI_figs, RM_CI_save_path + f'/{method.__name__}_combine')

    plt.close('all')



# # 針對整個資料集
for test_id in range(example_num):
    process_image(images[test_id], labels[test_id], test_id)

