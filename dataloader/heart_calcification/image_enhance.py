import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Dict, Tuple
import os



def enhance_image_with_contrast(img: np.ndarray, contrast_factor: float = 2.0) -> np.ndarray:
    """
    提高图像的对比度。

    参数:
    img (np.ndarray): 输入图像（NumPy 数组格式）

    返回:
    np.ndarray: 对比度增强后的图像（NumPy 数组格式）
    """
    # 确保输入图像是灰度图像或RGB图像
    if len(img.shape) not in [2, 3]:
        raise ValueError("Input image must be a 2D (grayscale) or 3D (RGB) array.")

    # 将 NumPy 数组转换为 PIL.Image 以进行对比度增强
    pil_img = Image.fromarray(img)

    # 创建对比度增强器
    enhancer = ImageEnhance.Contrast(pil_img)

    # 增强对比度
    enhanced_img = enhancer.enhance(contrast_factor)

    # 将增强后的 PIL.Image 转换回 NumPy 数组
    return np.array(enhanced_img)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    對圖像進行正規化處理，使其像素值範圍在 [0, 1] 之間。

    參數：
    - image: 輸入的圖像數據（NumPy.ndarray）

    返回：
    - normalized_image: 正規化後的圖像數據（NumPy.ndarray）
    """
    # 确保输入图像是有效的 NumPy 数组
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # 计算最小值和最大值
    min_val = np.min(image)
    max_val = np.max(image)

    # 正规化处理，避免除以零
    if max_val > min_val:
        normalized_image_np = (image - min_val) / (max_val - min_val)
    else:
        normalized_image_np = np.zeros_like(image)  # 如果所有像素值相同，返回全零图像

    return (normalized_image_np * 255).astype('uint8')  # 返回 uint8 格式的图像


def enhance_with_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    使用直方图均衡化提高图像对比度。

    参数:
    image (np.ndarray): 输入图像（灰度图像）

    返回:
    np.ndarray: 对比度增强后的图像
    """
    # 如果输入图像不是灰度图像，则转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def enhance_with_scale_and_offset(image: np.ndarray, alpha: float = 2, beta: float = -255) -> np.ndarray:
    """
    使用缩放和偏移提高图像对比度。

    参数:
    image (np.ndarray): 输入图像（可以是灰度或RGB图像）
    alpha (float): 缩放因子
    beta (float): 偏移量

    返回:
    np.ndarray: 对比度增强后的图像
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def enhance_with_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    使用自适应直方图均衡化（CLAHE）提高图像对比度。

    参数:
    image (np.ndarray): 输入图像（可以是灰度或RGB图像）
    clip_limit (float): 剪切限制
    tile_grid_size (Tuple[int, int]): 网格大小

    返回:
    np.ndarray: 对比度增强后的图像
    """
    # 如果输入图像不是灰度图像，则转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    return clahe_image

def none(image: np.ndarray):
    return image

# 定义增强函数字典
ENHANCE_FUNCTIONS = {
    'contrast': enhance_image_with_contrast,
    'normalize': normalize_image,
    'histogram_equalization': enhance_with_histogram_equalization,
    'scale_and_offset': enhance_with_scale_and_offset,
    'clahe': enhance_with_clahe,
    'none' : none
}
