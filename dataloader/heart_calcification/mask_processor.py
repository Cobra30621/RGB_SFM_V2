import cv2
import numpy as np

from typing import List, Dict, Tuple





def mask_image_with_polygon(img: np.ndarray, mask_file: str) -> np.ndarray:
    """
    使用掩码文件创建多边形，并根据多边形遮罩处理图像。

    参数:
    img (np.ndarray): 输入图像（NumPy 数组格式）
    mask_file (str): 掩码文件路径

    返回:
    np.ndarray: 处理后的图像，仅显示遮罩区域
    """
    # 获取图像的高度和宽度
    img_height, img_width = img.shape[:2]
    with open(mask_file, 'r') as f:
        lines = f.readlines()

    polygons = []
    for line in lines:
        class_id, *polygon = line.strip().split()
        if class_id == '0':
            polygon = create_polygon(polygon, img_width, img_height, scale=1.0)
            polygons.append(polygon)

    # 创建遮罩
    mask = np.zeros((img_height, img_width), dtype=np.uint8)  # 创建黑色遮罩
    for polygon in polygons:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)  # 填充多边形区域

    # 将遮罩应用于图像
    masked_img = img[:, :, np.newaxis] * mask[:, :, np.newaxis]  # 确保 img 变为三维数组
    return masked_img


def create_polygon(polygon_data: List[str], img_width: int, img_height: int, scale: float) -> List[
    Tuple[float, float]]:
    """
    创建多边形并根据中心缩放。

    参数:
    polygon_data (List[str]): 多边形数据
    img_width (int): 图像宽度
    img_height (int): 图像高度
    scale (float): 多边形缩放比例

    返回:
    List[Tuple[float, float]]: 处理后的多边形坐标列表
    """
    polygon = [(float(polygon_data[i]) * img_width, float(polygon_data[i + 1]) * img_height)
               for i in range(0, len(polygon_data), 2)]

    # 计算多边形的中心
    center_x = sum(p[0] for p in polygon) / len(polygon)
    center_y = sum(p[1] for p in polygon) / len(polygon)

    # 根据中心缩放多边形
    polygon = [
        (center_x + (p[0] - center_x) * scale, center_y + (p[1] - center_y) * scale)
        for p in polygon
    ]

    polygon.append(polygon[0])
    return polygon
