from pytorch_grad_cam import ( GradCAM, HiResCAM, GradCAMPlusPlus,
                              GradCAMElementWise, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM,
                              LayerCAM, KPCA_CAM)

from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, display_image_comparison, \
    check_then_preprocess_images
from load_tools import load_model_and_data
from models.RGB_SFMCNN_V2 import get_feature_extraction_layers
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

checkpoint_filename = config["load_model_name"]
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename)


rgb_layers, gray_layers = get_feature_extraction_layers(model)
layers = rgb_layers

# 使用影像前處理，如果有需要(視網膜資料集)
preprocess_images = check_then_preprocess_images(images)

# 獲得每一層的CIs
CIs, CI_values = get_CIs(model, preprocess_images)

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

save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example'
if os.path.exists(save_path):
    shutil.rmtree(save_path)  # 刪除資料夾及其內容
    os.makedirs(save_path)  # 重新建立資料夾


gray_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        # Sobel_Conv2d(),
        # Renormalize(),
        # NormalizeToRange(),
    ])
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


def process_layer(
    image: torch.Tensor,
    layer_name: str,
    use_gray: bool,
    model,
    layers,
    plot_shape,
    CIs,
    arch_args,
    RM_save_path: str,
    RM_CI_save_path: str,
    RM_figs: dict,
    RM_CI_figs: dict,
) -> None:
    """
    處理單層的 RM, FM, CI 可視化並儲存圖檔。
    """
    print(f"Processing {layer_name}...")

    in_channels = 1 if use_gray else arch_args['in_channels']
    input_image = model.gray_transform(image.unsqueeze(0)) if use_gray else image.unsqueeze(0)

    # 得到該層的 RM（Response Map）
    RM = layers[layer_name](input_image)[0]
    RM_H, RM_W = RM.shape[1], RM.shape[2]

    # 繪製 RM 圖
    fig_rm = plot_map(
        RM.permute(1, 2, 0).reshape(RM_H, RM_W, *plot_shape, 1).detach().numpy(),
        path=f'{RM_save_path}/{layer_name}_RM'
    )
    RM_figs[layer_name] = fig_rm

    # Top-1 index 取出對應的 CI
    top_idx = torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()
    CI_H, CI_W = CIs[layer_name].shape[2], CIs[layer_name].shape[3]
    RM_CI = CIs[layer_name][top_idx].reshape(RM_H, RM_W, CI_H, CI_W, in_channels)
    RM_CI_figs[layer_name] = plot_map(RM_CI, path=f'{RM_CI_save_path}/{layer_name}_RM_CI', cmap='gray' if use_gray else None)


    print(f"Finished {layer_name}.")


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

    # 原始圖
    fig_origin = plt.figure(figsize=(5, 5), facecolor="white")
    plt.imshow(image.permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.savefig(save_path + f'origin_{test_id}.png', bbox_inches='tight', pad_inches=0)
    RM_CI_figs['Origin'] = fig_origin

    if use_preprocessed_image:
        scaled_img, blurred_img, high_contrast_img, final_img = preprocess_retinal_tensor_image(image)
        display_image_comparison(save_path=save_path + f'preprocess.png', origin_img=image, final_img=final_img)
        fig_pre = plt.figure(figsize=(5, 5), facecolor="white")
        plt.imshow(final_img.permute(1, 2, 0).detach().numpy())
        plt.axis('off')
        plt.savefig(save_path + f'preprocess_{test_id}.png', bbox_inches='tight', pad_inches=0)
        RM_CI_figs['Preprocess'] = fig_pre
        image = final_img

    # 灰階圖
    fig_gray = plt.figure(figsize=(5, 5), facecolor="white")
    gray_image = gray_transform(image.unsqueeze(0))[0]
    plt.imshow(gray_image.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(save_path + f'gray_{test_id}.png', bbox_inches='tight', pad_inches=0)
    RM_CI_figs['Gray'] = fig_gray

    # 原圖切割視覺化
    segments = split(image.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride=(arch['args']['strides'][0], arch['args']['strides'][0]))[0]
    origin_split_img = plot_map(segments.permute(1, 2, 3, 4, 0), path=save_path + f'origin_split_{test_id}.png')
    RM_figs['Origin_Split'] = origin_split_img

    channels = arch['args']['channels']

    # 處理 RGB 分支所有層
    for i in range(len(model.RGB_convs)):
        layer_name = f'RGB_convs_{i}'
        plot_shape = channels[0][i]
        process_layer(image, layer_name, use_gray=False, model=model, layers=rgb_layers, plot_shape= plot_shape,CIs=CIs,
                      arch_args=arch['args'], RM_save_path=RM_save_path, RM_CI_save_path=RM_CI_save_path,
                      RM_figs=RM_figs, RM_CI_figs=RM_CI_figs)

    # 處理 Gray 分支所有層
    for i in range(len(model.Gray_convs)):
        layer_name = f'Gray_convs_{i}'
        plot_shape = channels[1][i]
        process_layer(image, layer_name, use_gray=True, model=model, layers=gray_layers, plot_shape= plot_shape, CIs=CIs,
                      arch_args=arch['args'], RM_save_path=RM_save_path, RM_CI_save_path=RM_CI_save_path,
                      RM_figs=RM_figs, RM_CI_figs=RM_CI_figs)

    # 合併圖
    plot_combine_images(RM_figs, RM_save_path + f'RGB_combine')
    RM_CI_combine_fig = plot_combine_images(RM_CI_figs, RM_CI_save_path + f'combine')

    if PLOT_CAM:
        cam_methods = [GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, AblationCAM,
                       ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, KPCA_CAM]

        cam_figs = {}
        RM_CI_figs = {'raw': RM_CI_combine_fig}

        for method in cam_methods:
            print(f"drawing {method.__name__}")
            cam_fig, RM_CI_fig = generate_cam_visualizations(
                model=model,
                label=label.argmax().item(),
                image=image,
                origin_img=origin_split_img,
                RM_CIs=RM_CIs,
                save_path=RM_CI_save_path,
                method=method
            )
            cam_figs[method.__name__] = cam_fig
            RM_CI_figs[method.__name__] = RM_CI_fig

        plot_combine_images_vertical(cam_figs, RM_CI_save_path + f'cam/cams_combine')
        plot_combine_images_vertical(RM_CI_figs, RM_CI_save_path + f'/{method.__name__}_combine')

    plt.close('all')


# # 針對整個資料集
for test_id in range(example_num):
    process_image(images[test_id], labels[test_id], test_id)

