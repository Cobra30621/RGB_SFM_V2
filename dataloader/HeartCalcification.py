import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image

from config import config
from .heart_calcification.heart_calcification_data_processor import HeartCalcificationDataProcessor


class HeartCalcificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        color_mode: str = 'RGB',  # 新增參數
        grid_size: int = 45,
        need_resize_height: bool = False,
        resize_height : int = 900,
        threshold: float = 1.0,
        contrast_factor : float = 1.0,
        enhance_method: str = 'none',
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.augmentation = augmentation
        self.grid_size = grid_size
        self.color_mode = color_mode

        self.data_dir = f"{self.root}/HeartCalcification/{'train' if self.train else 'test'}"
        self.data_processor = HeartCalcificationDataProcessor(
            grid_size=self.grid_size, data_dir=self.data_dir,
            need_resize_height = need_resize_height, resize_height= resize_height,
            threshold=threshold, contrast_factor = contrast_factor, enhance_method=enhance_method)

        self.model_ready_data = self.data_processor.get_model_ready_data()

        print(len(self.model_ready_data))
        self.data_processor.display_label_counts()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        key, img, label = self.model_ready_data[idx]

        # 確保 img 是 np.ndarray 而不是 Image.Image
        if isinstance(img, Image.Image):
            img = np.array(img)  # 將 Image.Image 轉換為 np.ndarray

        if self.color_mode == 'RGB':
            img = img.transpose((2, 0, 1))  # 轉換為 (C, H, W) 格式
        elif self.color_mode == 'L':
            img = img[None, ...]  # 增加一個維度以符合 (C, H, W) 格式
        else:
            raise ValueError("color_mode 必須是 'RGB' 或 'L'")

        img = torch.from_numpy(img).float() / 255.0  # 將 np.ndarray 轉換為張量

        # 将标签转换为长整型张量
        # label = torch.tensor(label, dtype=torch.long)

        y_onehot = np.eye(2)[label]
        y_onehot = torch.from_numpy(y_onehot)
        label = y_onehot

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.model_ready_data)


class HeartCalcificationColor(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, color_mode='RGB')

class HeartCalcificationGray(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        grid_size = config["heart_calcification"]["grid_size"]
        resize_height = config["heart_calcification"]["resize_height"]
        need_resize_height = config["heart_calcification"]["need_resize_height"]
        threshold = config["heart_calcification"]["threshold"]
        contrast_factor = config["heart_calcification"]["contrast_factor"]
        enhance_method = config["heart_calcification"]["enhance_method"]
        super().__init__(*args, **kwargs, color_mode='L',
                         grid_size = grid_size, resize_height = resize_height,
                         need_resize_height = need_resize_height, threshold=threshold,
                         contrast_factor = contrast_factor, enhance_method=enhance_method)

class HeartCalcificationGray60(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, color_mode='L', grid_size = 60)
