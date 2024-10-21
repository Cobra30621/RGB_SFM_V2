import cv2
import numpy as np

from typing import List, Dict, Tuple







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

    # 計算多邊形的重心
    A = 0  # 面積
    Cx = 0  # 重心 x 坐標
    Cy = 0  # 重心 y 坐標

    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]  # 確保閉合
        A += x0 * y1 - x1 * y0
        Cx += (x0 + x1) * (x0 * y1 - x1 * y0)
        Cy += (y0 + y1) * (x0 * y1 - x1 * y0)

    A *= 0.5
    Cx /= (6 * A)
    Cy /= (6 * A)

    # 根據重心縮放多邊形
    polygon = [
        (Cx + (p[0] - Cx) * scale, Cy + (p[1] - Cy) * scale)
        for p in polygon
    ]

    polygon.append(polygon[0])
    return polygon

def draw_polygons_on_image(img: np.ndarray, mask_file: str, color: Tuple[int, int, int] = (0, 255, 255), scale: float = 1.0) -> np.ndarray:
    """
    在图像上绘制多边形，使用指定颜色和缩放比例。

    参数:
    img (np.ndarray): 输入图像（NumPy 数组格式）
    mask_file (str): 掩码文件路径
    color (Tuple[int, int, int]): 多边形颜色，默认为半透明黄色
    scale (float): 多边形缩放比例，默认为 1.0

    返回:
    np.ndarray: 绘制了多边形的图像
    """
    # 获取图像的高度和宽度
    img_height, img_width = img.shape[:2]
    with open(mask_file, 'r') as f:
        lines = f.readlines()

    polygons = []
    for line in lines:
        class_id, *polygon = line.strip().split()
        if class_id == '0':
            polygon = create_polygon(polygon, img_width, img_height, scale=scale)  # 使用指定的 scale
            polygons.append(polygon)

    # 在图像上绘制多边形
    overlay = img.copy()  # 创建图像副本用于绘制
    for polygon in polygons:
        cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], color)  # 使用指定的颜色填充

    # 创建半透明效果
    alpha = 0.5  # 透明度
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)  # 合并图像

    return img

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
    mask = mask[:, :, np.newaxis]  # 增加一个维度，使 mask 变为 (567, 376, 1)
    masked_img = img * mask  # 确保 img 和 mask 的形状匹配
    return masked_img
