from PIL import Image, ImageEnhance
import numpy as np


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    将 PIL.Image 转换为 NumPy 数组。

    参数:
    image (Image.Image): 输入的 PIL 图像

    返回:
    np.ndarray: 转换后的 NumPy 数组
    """
    return np.array(image)

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """
    将 NumPy 数组转换为 PIL.Image。

    参数:
    image (np.ndarray): 输入的 NumPy 数组（应为 uint8 格式）

    返回:
    Image.Image: 转换后的 PIL 图像
    """
    return Image.fromarray(image.astype('uint8'))




