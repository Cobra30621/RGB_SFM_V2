from typing import List, Tuple, Dict

import numpy as np

class ImageSplitData:
    """
    圖像分割數據類，用於存儲和管理分割後的圖像數據。
    """

    def __init__(self, image_name: str, image_path: str, split_count: tuple, labels: dict,
                 split_images: List[np.ndarray], vessel_mask_file: str, calcification_path: str):
        """
        初始化 ImageSplitData 對象。

        參數:
        image_name (str): 圖像名稱
        image_path (str): 圖像路徑
        split_count (tuple): 分割數量，格式為 (行數, 列數)
        labels (dict): 標籤字典，鍵為 (行, 列) 元組，值為對應的標籤
        split_images (List[Image.Image]): 切割後的圖像列表
        vessel_mask_file (str): 血管掩碼文件的路徑
        calcification_path (str): calcification_path 路徑
        """
        self.image_name = image_name
        self.image_path = image_path
        self.split_count = split_count
        self.labels = labels
        self.split_images = split_images  # 新增 img 屬性
        self.vessel_mask_file = vessel_mask_file  # 新增 vessel_mask_file 屬性
        self.calcification_path = calcification_path  # 新增 calcification_path 屬性

    @property
    def split_count(self):
        """
        獲取分割數量。

        返回:
        tuple: 分割數量 (行數, 列數)
        """
        return self._split_count

    @split_count.setter
    def split_count(self, value):
        """
        設置分割數量。

        參數:
        value (tuple): 分割數量，格式為 (行數, 列數)

        異常:
        ValueError: 如果輸入的值不符合要求
        """
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("split_count 必须是一个包含两个整数的元组 (m, n)")
        if not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError("split_count 的值必须是正整数")
        self._split_count = value

    @property
    def labels(self):
        """
        獲取標籤字典。

        返回:
        dict: 標籤字典
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        設置標籤字典。

        參數:
        value (dict): 標籤字典，鍵為 (行, 列) 元組，值為對應的標籤

        異常:
        ValueError: 如果輸入的值不符合要求
        """
        if not isinstance(value, dict):
            raise ValueError("labels 必须是一个字典")
        if not all(isinstance(k, tuple) and len(k) == 2 and
                   all(isinstance(i, int) and i >= 0 for i in k) and
                   isinstance(v, int) for k, v in value.items()):
            raise ValueError("labels 字典的键必须是非负整数元组 (row, col)，值必须是整数")
        self._labels = value

    def get_vessel_split_images(self) -> Dict[tuple, np.ndarray]:
        """
        獲取血管分割圖像的字典。

        返回:
        dict: 鍵為標籤，值為對應的分割圖像
        """
        vessel_images = {}
        for (i, j), label in self.labels.items():
            if label != -1:
                index = i * self.split_count[1] + j
                vessel_images[(i,j)] = self.split_images[index]
        return vessel_images

    
    def get_vessel_labels(self) -> Dict[tuple, int]:
        """
        獲取非 -1 的標籤字典。

        返回:
        dict: 鍵為 (行, 列) 元組，值為對應的標籤
        """
        vessel_labels = {k: v for k, v in self.labels.items() if v != -1}
        return vessel_labels