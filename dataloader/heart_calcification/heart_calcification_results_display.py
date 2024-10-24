from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from dataloader.heart_calcification.image_enhance import normalize_image, \
    enhance_image_with_contrast, ENHANCE_FUNCTIONS
from dataloader.heart_calcification.mask_processor import *
from .image_split_data import ImageSplitData

from .image_tool import resize_image



def visualize_predict(result:List[Tuple[ImageSplitData, Dict[tuple, int], Dict[tuple, int]]],
                      data_dir: str, save_dir: str, grid_size: int, resize_height: int, mask_with_vessel: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_data, true_labels, predicted_labels in result:
        save_path = os.path.join(save_dir, f'{image_data.image_name}_dataset.png')

        # 读取图像
        img = load_image(image_data.image_path, resize_height)

        # 使用血管遮罩
        if mask_with_vessel:
            vessel_mask = os.path.join(data_dir, image_data.vessel_mask_file)
            img = mask_image_with_polygon(img, vessel_mask)

        _visualize_and_save_dataset_image(img, predicted_labels, image_data.split_count, grid_size, save_path)


def visualize_dataset(data_dict: Dict[str, ImageSplitData], save_dir: str, data_dir : str,
                    enhance_method: str, grid_size: int, resize_height: int,
                      draw_calcification: bool = False, mask_with_vessel: bool = False):
    """
    可視化數據集並保存為圖片。

    參數:
    data_dict: 字典, 鍵為圖像名稱, 值為 ImageSplitData 對象
    save_dir: 保存圖片的目錄
    enhance_method: 增强方法的名稱
    grid_size: 繪製網格的區間大小
    resize_height: 縮放後的高度
    draw_calcification: 是否繪製鈣化點的標誌
    mask_with_vessel: 是否使用血管遮罩
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for image_name, image_data in data_dict.items():

        # 處理路徑
        vessel_mask = os.path.join(data_dir, image_data.vessel_mask_file)
        calcification_path = os.path.join(data_dir, image_data.calcification_path)
        save_path = os.path.join(save_dir, f'{image_name}_dataset.png')

        # 读取图像
        img = load_image(image_data.image_path, resize_height)

        # 劃出鈣化點
        if draw_calcification:
            # 畫出完整的框框
            img = draw_polygons_on_image(img, calcification_path)
            # 劃出縮放後的框框
            img = draw_polygons_on_image(img, calcification_path, (255, 0, 0), 0.5)
        # 根据增强方法对图像进行增强
        enhance_func = ENHANCE_FUNCTIONS.get(enhance_method)
        if enhance_func:
            img = enhance_func(img)
        else:
            print(f"未知的增强方法: {enhance_method}")

        # 使用血管遮罩
        if mask_with_vessel:
            img = mask_image_with_polygon(img, vessel_mask)

        _visualize_and_save_dataset_image(img, image_data.labels, image_data.split_count, grid_size, save_path)



def _visualize_and_save_dataset_image(img: np.ndarray, labels: Dict[tuple, int], split_count : tuple,
                                      grid_size:int, save_path: str):
    """
    可視化單個數據集圖像並保存。

    參數:
    img: 圖像數組
    label: 標籤數組
    grid_size: 圖片切割大小
    save_path: 保存路徑
    """
    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    num_blocks_h, num_blocks_w = split_count

    # 繪製網格線
    for i in range(1, num_blocks_h):
        plt.axhline(y=i * grid_size, color='w', linestyle='-', linewidth=1)
    for j in range(1, num_blocks_w + 1):
        plt.axvline(x=j * grid_size, color='w', linestyle='-', linewidth=1)

    # 在標籤為1或0的格子中繪製不同顏色的 'O'
    for key, label in labels.items():
        i, j = key
        if label == 1:
            color = 'r'  # 紅色
        elif label == 0:
            color = 'y' # 藍色
        else:
            continue  # 如果標籤為0,不繪製任何內容

        plt.text((j + 0.5) * grid_size ,
                 (i + 0.5)* grid_size , 'O',
                 color=color, fontsize=16, ha='center', va='center')

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"數據集圖像已保存到: {save_path}")



def load_image(image_path: str, resize_height: int):
    # 读取原始图像
    img = Image.open(image_path)
    img_array = np.array(img)
    # 确保图像是 3 通道 RGB
    if len(img_array.shape) == 2:  # 如果是灰度图
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # 如果是 RGBA
        img_array = img_array[:, :, :3]
    img = img_array

    # 根據 resize_height 等比例縮放圖像
    img = resize_image(img, resize_height)

    return img


