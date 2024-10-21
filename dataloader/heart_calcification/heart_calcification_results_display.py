from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance

from dataloader.heart_calcification.image_enhance import normalize_image, \
    enhance_image_with_contrast, ENHANCE_FUNCTIONS
from dataloader.heart_calcification.mask_processor import *

from .image_tool import resize_image

class HeartCalcificationResultsDisplay:
    """
    心臟鈣化結果處理器類別。
    用於處理和分析心臟鈣化相關的訓練和預測結果。
    """


    def visualize_dataset(self, images: Dict[str, np.ndarray], labels: Dict[str, np.ndarray],
                          vessel_masks: Dict[str, str], calcifications: Dict[str, str],
                          save_dir: str, enhance_method: str, grid_size: int, resize_height: int):
        """
        可視化數據集並保存為圖片。

        參數:
        images: 字典,鍵為圖像名稱,值為圖像數組
        labels: 字典,鍵為圖像名稱,值為標籤數組
        save_dir: 保存圖片的目錄
        enhance_method: 增强方法的名称
        grid_size: 繪製網格的區間大小
        resize_height: 縮放後的高度
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for image_name, img in images.items():
            label = labels[image_name]
            vessel_mask = vessel_masks[image_name]
            calcification_path = calcifications[image_name]

            save_path = os.path.join(save_dir, f'{image_name}_dataset.png')

            # 根據 resize_height 等比例縮放圖像
            img = resize_image(img, resize_height)

            # 畫出完整的框框
            img = draw_polygons_on_image(img, calcification_path)
            # 劃出縮放後的框框
            img = draw_polygons_on_image(img, calcification_path, (255, 0 ,0), 0.5)
            # img = mask_image_with_polygon(img, vessel_mask)

            # 根据增强方法对图像进行增强
            enhance_func = ENHANCE_FUNCTIONS.get(enhance_method)
            if enhance_func:
                img = enhance_func(img)
            else:
                print(f"未知的增强方法: {enhance_method}")
            
            self._visualize_and_save_dataset_image(img, label, grid_size, save_path)

    def _visualize_and_save_dataset_image(self, img: np.ndarray, label: np.ndarray, grid_size:int, save_path: str):
        """
        可視化單個數據集圖像並保存。

        參數:
        img: 圖像數組
        label: 標籤數組
        save_path: 保存路徑
        resize_height: 縮放後的高度
        """
        plt.figure(figsize=(9, 9))
        plt.imshow(img)
        num_blocks_h, num_blocks_w = label.shape

        # 繪製網格線
        for i in range(1, num_blocks_h):
            plt.axhline(y=i * grid_size, color='w', linestyle='-', linewidth=1)
        for j in range(1, num_blocks_w + 1):
            plt.axvline(x=j * grid_size, color='w', linestyle='-', linewidth=1)

        # 在標籤為1或0的格子中繪製不同顏色的 'O'
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if label[i, j] == 1:
                    color = 'r'  # 紅色
                elif label[i, j] == 0:
                    color = 'b'  # 藍色
                else:
                    continue  # 如果標籤為0,不繪製任何內容

                plt.text((j + 0.5) * grid_size ,
                         (i + 0.5)* grid_size , 'O',
                         color=color, fontsize=12, ha='center', va='center')

        plt.axis('off')
        plt.tight_layout()

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"數據集圖像已保存到: {save_path}")

