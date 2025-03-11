# 可解釋性圖繪製方法
import os
import warnings

from utils import plot_combine_images

warnings.filterwarnings('ignore')
from torchvision import transforms

from pytorch_grad_cam import run_dff_on_image, GradCAM, HiResCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, \
    AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad, DeepFeatureFactorization, KPCA_CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
import matplotlib.pyplot as plt

""" Model wrapper to return a tensor"""


class ModelWrapper(torch.nn.Module):
    """將 Huggingface 模型包裝為返回張量的模型"""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.

"""
def generate_cam_visualization(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    cam_targets: List[Callable],
    transform: Optional[Callable],
    input_tensor: torch.Tensor,
    input_image: Image.Image,
    cam_method: Callable = GradCAM
) -> np.ndarray:
    """生成 CAM 可視化結果
    
    Args:
        model: 目標模型
        target_layer: 目標層
        cam_targets: CAM 目標列表
        transform: 形狀轉換函數
        input_tensor: 輸入張量
        input_image: 輸入圖像
        cam_method: CAM 方法
        
    Returns:
        包含所有目標的 CAM 可視化結果
    """
    wrapped_model = ModelWrapper(model)
    
    with cam_method(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=transform
    ) as cam:
        batch_tensor = input_tensor[None, :].repeat(len(cam_targets), 1, 1, 1)
        cam_results = cam(input_tensor=batch_tensor, targets=cam_targets)
        
        visualizations = []
        for grayscale_cam in cam_results:
            vis = show_cam_on_image(
                np.float32(input_image) / 255,
                grayscale_cam,
                use_rgb=True
            )
            vis = cv2.resize(vis, (vis.shape[1] * 2, vis.shape[0] * 2))
            visualizations.append(vis)
            
        return np.hstack(visualizations)


def get_target_layers(model: torch.nn.Module) -> dict:
    """獲取需要分析的目標層
    
    Args:
        model: 目標模型
        
    Returns:
        包含所有目標層的字典
    """
    return {
        'RGB_convs_0': model.RGB_convs[1],
        'RGB_convs_1': model.RGB_convs[2][1],
        'RGB_convs_2': model.RGB_convs[3],
        'Gray_convs_0': model.Gray_convs[0][1],
        'Gray_convs_1': model.Gray_convs[2][1],
        'Gray_convs_2': model.Gray_convs[3]
    }



def visualize_heatmap(model, layers, image, img_tensor, label, method: Callable = GradCAM):
    """生成可視化結果

        Args:
            model: 目標模型
            layers: 目標層字典
            image: 輸入圖像
            img_tensor: 輸入圖片 tensor
            label: 目標標籤
            method: CAM 方法
        """
    imgs = {}
    # 添加原始圖像的figure
    fig_raw = plt.figure(figsize=(6, 6))
    plt.imshow(img_tensor.permute(1, 2, 0).cpu().numpy())  # 確保數據是numpy數組
    plt.axis('off')
    plt.draw()
    plt.close(fig_raw)
    imgs['raw'] = fig_raw

    targets_for_gradcam = [ClassifierOutputTarget(label)]
    for layer in layers:
        target_layer = layers[layer]

        cam_img = generate_cam_visualization(model, target_layer, targets_for_gradcam, None, img_tensor, image, method)

        # 確保cam_img是正確的numpy數組
        if isinstance(cam_img, torch.Tensor):
            cam_img = cam_img.cpu().numpy()

        fig = plt.figure(figsize=(6, 6))

        plt.imshow(cam_img)
        plt.axis('off')
        plt.draw()
        plt.close(fig)

        imgs[layer] = fig

    return imgs


def visualize_all_heatmap(
    model: torch.nn.Module,
    target_layers: dict,
    image: torch.Tensor,
    label: int,
    output_dir: str,
    cam_methods: List[Callable]
) -> None:
    """為所有 CAM 方法生成可視化結果
    
    Args:
        model: 目標模型
        target_layers: 目標層字典
        image: 輸入圖像
        label: 目標標籤
        output_dir: 輸出目錄
        cam_methods: CAM 方法列表
    """
    heatmap_dir = os.path.join(output_dir, 'heatmap')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    pil_image = transforms.ToPILImage()(image)
    img_tensor = transforms.ToTensor()(pil_image)
    
    for method in cam_methods:
        try:
            print(f"\n使用方法: {method.__name__}")
            visualizations = visualize_heatmap(
                model=model,
                layers=target_layers,
                image=pil_image,
                img_tensor=img_tensor,
                label=label,
                method=method
            )
            
            output_path = os.path.join(heatmap_dir, method.__name__)
            plot_combine_images(visualizations, output_path, show=True)
            
        except Exception as e:
            print(f"方法 {method.__name__} 執行失敗: {str(e)}")