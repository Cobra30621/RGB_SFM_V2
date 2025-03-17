# 可解釋性圖繪製方法
import os
import warnings

from utils import plot_combine_images, plot_map

warnings.filterwarnings('ignore')
from torchvision import transforms
import seaborn as sns
from pytorch_grad_cam import run_dff_on_image, GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, \
    AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization, KPCA_CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional, Dict
import matplotlib.pyplot as plt

""" Model wrapper to return a tensor"""


class ModelWrapper(torch.nn.Module):
    """將 Huggingface 模型包裝為返回張量的模型
    
    這個包裝器類用於確保模型輸出為張量格式，主要用於配合 CAM 可視化方法的使用。
    
    Args:
        model (torch.nn.Module): 要包裝的原始模型
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def get_each_layers_cam(
        model: torch.nn.Module,
        target_layers: dict,
        label: int,
        input_tensor: torch.Tensor,
        cam_method: Callable = GradCAM
):
    """為每個目標層生成 CAM 熱力圖
    
    使用指定的 CAM 方法，為模型中的每個目標層生成對應的熱力圖。
    
    Args:
        model (torch.nn.Module): 目標模型
        target_layers (dict): 目標層字典
        label (int): 目標標籤
        input_tensor (torch.Tensor): 輸入圖像張量
        cam_method (Callable): CAM 方法，默認使用 GradCAM
        
    Returns:
        Dict[str, torch.Tensor]: 每個層的名稱及其對應的 CAM 熱力圖
    """
    wrapped_model = ModelWrapper(model)
    cam_target = ClassifierOutputTarget(label)

    cams = {}
    for layer_name in target_layers:
        layer = target_layers[layer_name]

        with cam_method(
                model=wrapped_model,
                target_layers=[layer]
        ) as cam:
            # 移除批次處理邏輯，直接使用單個輸入
            input_batch = input_tensor.unsqueeze(0)
            grayscale_cam = cam(input_tensor=input_batch, targets=[cam_target])[0]
            cams[layer_name] = grayscale_cam

    return cams


def plot_cams_on_image(
        input_image: torch.Tensor,
        cams: Dict[str, torch.Tensor],
        save_path: str,
        cam_method_name: str
):
    """將 CAM 熱力圖疊加在原始圖像上並繪製
    
    將生成的 CAM 熱力圖與原始圖像進行融合，並保存可視化結果。
    
    Args:
        input_image (torch.Tensor): 原始輸入圖像
        cams (Dict[str, torch.Tensor]): 每個層的 CAM 熱力圖
        save_path (str): 保存路徑
        cam_method_name (str): CAM 方法名稱
    """
    pil_image = transforms.ToPILImage()(input_image)

    imgs = {}
    # 添加原始圖像的figure
    fig_raw = plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)  # 確保數據是numpy數組
    plt.axis('off')
    plt.draw()
    plt.close(fig_raw)
    imgs['raw'] = fig_raw

    for layer, cam in cams.items():
        cam_on_image = show_cam_on_image(
            np.float32(pil_image) / 255,
            cam,
            use_rgb=True
        )

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(cam_on_image)
        plt.axis('off')
        plt.draw()
        plt.close(fig)

        imgs[layer] = fig

    output_path = os.path.join(save_path, cam_method_name)
    plot_combine_images(imgs, output_path, show=True)


def get_cam_target_layers(model: torch.nn.Module) -> dict:
    """獲取模型中需要分析的目標層
    
    返回模型中所有需要進行 CAM 分析的層級。
    
    Args:
        model (torch.nn.Module): 目標模型
        
    Returns:
        dict: 包含所有目標層的字典，鍵為層名稱，值為層物件
    """
    return {
        'RGB_convs_0': model.RGB_convs[1],
        'RGB_convs_1': model.RGB_convs[2][1],
        'RGB_convs_2': model.RGB_convs[3],
        'Gray_convs_0': model.Gray_convs[0][1],
        'Gray_convs_1': model.Gray_convs[2][1],
        'Gray_convs_2': model.Gray_convs[3]
    }


def get_reduced_cam(cam, output_shape):
    """將 CAM 熱力圖縮減到指定大小
    
    通過區塊平均的方式將 CAM 熱力圖縮減到目標尺寸。
    
    Args:
        cam (numpy.ndarray): 原始 CAM 熱力圖
        output_shape (tuple): 目標輸出形狀 (高度, 寬度)
        
    Returns:
        numpy.ndarray: 縮減後的 CAM 熱力圖
    """
    n, m = cam.shape
    target_n, target_m = output_shape

    # 沿著兩個維度進行不均勻分割
    split_n = np.array_split(np.arange(n), target_n)
    split_m = np.array_split(np.arange(m), target_m)

    # 對每個區塊取平均
    reduced_cam = np.array([
        np.mean(cam[np.ix_(rows, cols)])
        for rows in split_n
        for cols in split_m
    ]).reshape(output_shape)

    return reduced_cam


def plot_RM_CI_with_cam_mask(RM_CI, reduced_cam, save_path = None):
    """使用 CAM 遮罩繪製 RM_CI 圖
    
    根據 CAM 熱力圖的閾值，遮罩 RM_CI 圖像中的特定區域。
    
    Args:
        RM_CI (numpy.ndarray): RM_CI 數據
        reduced_cam (numpy.ndarray): 縮減後的 CAM 熱力圖
        save_path (str): 保存路徑
        
    Returns:
        matplotlib.figure.Figure: 生成的圖形物件
    """
    Threshold = 0.2
    # 找出 cam 中小於 Threshold 的位置
    mask = reduced_cam < Threshold

    # 將對應的 RM_CI 區塊全部設為 0
    RM_CI[mask] = 0

    fig = plot_map(RM_CI, path = save_path)

    return fig


def plot_reduced_cam(resize_cam,  title="Heatmap of resize_cam", save_path=None):
    """繪製縮減後的 CAM 熱力圖
    
    將縮減後的 CAM 熱力圖以熱力圖形式可視化。
    
    Args:
        resize_cam (numpy.ndarray): 縮減後的 CAM 數據
        save_path (str): 保存路徑
        title (str): 圖表標題，默認為 "Heatmap of resize_cam"
        
    Returns:
        matplotlib.figure.Figure: 生成的圖形物件
    """
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(resize_cam, annot=True, cmap="viridis", cbar=True, fmt=".2f")
    plt.title(title)
    plt.xlabel("m-axis")
    plt.ylabel("n-axis")

    # 儲存圖像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()  # 關閉圖形，節省內存

    return fig



