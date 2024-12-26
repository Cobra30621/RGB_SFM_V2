from pathlib import Path
import glob
import re
import torchvision
import matplotlib.pyplot as plt
import os
import torch
import math
import torch.nn.functional as F
from config import *
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from collections import defaultdict
import numpy as np
'''
    將 image 按照 kernel size 進行切割
'''
def show_Image_split(test_img, save_path = ""):
    segments = split(test_img.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride = (arch['args']['strides'][0], arch['args']['strides'][0]))[0]
    plot_map(segments.permute(1,2,3,4,0), vmax=1, vmin=0, path='origin_split_0.png', cmap='gray')

    segments = segments.permute(1,2,3,4,0)
    print(segments.shape)
    segments = segments.reshape(segments.shape[0]//2, 2, segments.shape[1]//2, 2, 5, 5, 1)
    segments = segments.permute(0,2,1,4,3,5,6).reshape(segments.shape[0], segments.shape[2], 10, 10, 1)
    if save_path == "":
        plot_map(segments, vmax=1, vmin=0, path=save_path)
    else:
        plot_map(segments, vmax=1, vmin=0)

'''
    讀取圖片
'''
def read_Image(path = 'D:/Project/paper/RGB_SFM/showout/Colored_MNIST_0610_RGB_SFMCNN_best_t1np8eon_LAB/example/0/example_240/origin_240.png'):
    test_img = torchvision.io.read_image(path)
    test_img = test_img.to(torch.float32)
    test_img /= 255
    test_img = test_img[:3, :, :]
    return test_img

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def plot_map(rm, grid_size=None, rowspan=None, colspan=None, path=None, **kwargs):
    # 新增：計算 rm 的全局最大值和最小值
    global_max = rm.max()
    global_min = rm.min()

    rows, cols, e_h, e_w, _ = rm.shape
    if rowspan is None:
        rowspan = int(e_h / min(e_h, e_w))
    if colspan is None:
        colspan = int(e_w / min(e_h, e_w))
    if grid_size is None:
        grid_size = (rows * rowspan, cols * colspan)

    char_space = cols * 0.4
    fig = plt.figure(figsize=(grid_size[1] + char_space, grid_size[0]), facecolor="white")

    # 繪製圖像
    for row in range(rows):
        for col in range(cols):
            ax = plt.subplot2grid(grid_size, (row * rowspan, col * colspan), rowspan=rowspan, colspan=colspan)
            im = ax.imshow(rm[row][col], vmin=global_min, vmax=global_max, **kwargs)  # 使用全局最大值和最小值
            ax.axis('off')
    
    # 新增：調整 subplot 位置，讓每個 subplot 靠左
    plt.subplots_adjust(left=0, right=0.7, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)  # 調整邊距

    # 添加色彩對應數值的長條圖，與整張圖片等高
    cbar_ax = fig.add_axes([0.75, 0.1, 0.02, 0.8])  # 自定義長條圖的位置和大小
    cbar = plt.colorbar(im, cax=cbar_ax)  # 使用自定義的長條圖軸
    cbar.ax.tick_params(labelsize=2 * rows)  # 調整長條圖文字大小

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        plt.close()
    
    return fig  # 新增：回傳繪製的圖像
        


import matplotlib.pyplot as plt


def combine_images(images, titles=None, save_path=None, spacing=0.05, fixed_width=5, fixed_height=5):
    import matplotlib.pyplot as plt

    num_images = len(images)
    fig_width = num_images * fixed_width + (num_images - 1) * spacing
    fig_height = fixed_height + (0.5 if titles else 0)

    # 創建畫布，啟用 constrained_layout
    fig, axes = plt.subplots(
        1, num_images,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1] * num_images, 'wspace': spacing / fixed_width},
        constrained_layout=True
    )

    # 確保 axes 是列表
    if num_images == 1:
        axes = [axes]

    for i, (fig_source, ax) in enumerate(zip(images, axes)):
        # 將 `fig_source` 等比例縮小 90%
        fig_source.set_size_inches(fig_source.get_size_inches() * 0.8)
        ax.imshow(fig_source.canvas.buffer_rgba())
        ax.axis('off')

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10, pad=10)

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_heatmap(CI_values, save_path, width=15, height=15):
    """
    Generate and save a heatmap from the given values.

    Parameters:
    CI_values (list of list of float): The matrix values for the heatmap.
    save_path (str): The path to save the heatmap image.
    width (int): The width of the heatmap.
    height (int): The height of the heatmap.
    """
    # Convert the input list to a NumPy array and reshape to the specified width and height
    reshaped_CI_values = np.array(CI_values).reshape(height, width)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 9))  # Enlarge by 1.5 times

    # Display the heatmap
    cax = ax.matshow(reshaped_CI_values, cmap='viridis')

    # Add color bar
    fig.colorbar(cax)

    # Set up ticks and labels
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels([''] * width)
    ax.set_yticklabels([''] * height)

    # Render the values in the matrix
    for i in range(height):
        for j in range(width):
            ax.text(j, i, f"{reshaped_CI_values[i][j]:.2f}", va='center', ha='center', color='white', fontsize=7)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def split(input, kernel_size = (5, 5), stride = (5,5)):
    batch, channel, h, w = input.shape
    output_height = math.floor((h  - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    output_width = math.floor((w  - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    input = torch.tensor(input)
    segments = F.unfold(input, kernel_size=kernel_size, stride=stride).reshape(batch, channel, *kernel_size, -1).permute(0,1,4,2,3)
    segments = segments.reshape(batch, channel, output_height, output_width, *kernel_size) 
    return segments

def get_ci(input, layer, kernel_size = (5,5), stride= (5,5), sfm_filter = (1,1)):
    segments = split(input, kernel_size, stride)
    combine_h, combine_w, ci_h, ci_w = (int(segments.shape[2]/sfm_filter[0]), int(segments.shape[3]/sfm_filter[1]), int(segments.shape[4]*sfm_filter[0]), int(segments.shape[5]*sfm_filter[1]))
    segments = segments.reshape(-1, input.shape[1], combine_h, sfm_filter[0], combine_w, sfm_filter[1], segments.shape[4], segments.shape[5])
    segments = segments.permute(0, 2, 4, 3, 6, 5, 7, 1)
    segments = segments.reshape(-1, ci_h, ci_w, input.shape[1])
    print(f"segments shape: {segments.shape}")
    
    with torch.no_grad():
        outputs = layer(input)
        n_filters = outputs.shape[1]
        outputs = outputs.permute(0,2,3,1).reshape(-1, n_filters)
        print(f"output shape: {outputs.shape}")

    k = 50
    CI = torch.empty(n_filters, k, ci_h, ci_w, input.shape[1])
    CI_values = torch.empty(n_filters, k) 
    CI_idx = torch.empty(n_filters, k)    
    for i in range(n_filters):
        values, indices = torch.topk(outputs[:, i], k=k, largest=True)
        CI_idx[i] = indices
        CI_values[i] = values
        CI[i] = segments[indices.tolist()]

    # 只保留 CI_values 中 dim=1 的第一个元素
    CI_values = CI_values[:, :1]  # 只保留第一个元素
    # 根据 k 对 CI 进行平均，最终形状为 [70, 1, 5, 5, 1]
    CI = CI.mean(dim=1, keepdim=True)  # 在 k 维度上取平均
    print(f"CI shape: {CI.shape}")
    return CI, CI_idx, CI_values

'''
    使用於讀取cifar10
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

    

