import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image
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
        grid_size: int = 45
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.augmentation = augmentation
        self.grid_size = grid_size
        self.color_mode = color_mode

        self.data_dir = f"{self.root}/HeartCalcification/{'train' if self.train else 'test'}"
        self.data_processor = HeartCalcificationDataProcessor(grid_size=self.grid_size, data_dir=self.data_dir)
        self.data_processor.generate_dataset()
        self.model_ready_data = self.data_processor.get_model_ready_data()
        print(len(self.model_ready_data))

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_name, img, label = self.model_ready_data[idx]

        if self.color_mode == 'RGB':
            img = img.convert('RGB')
        elif self.color_mode == 'L':
            img = img.convert('L')
        else:
            raise ValueError("color_mode 必須是 'RGB' 或 'L'")

        if self.transform is not None:
            img = self.transform(img)
        else:
            if self.color_mode == 'RGB':
                img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0
            else:
                img = torch.from_numpy(np.array(img)[None, ...]).float() / 255.0

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
        super().__init__(*args, **kwargs, color_mode='L')

class HeartCalcificationGray60(HeartCalcificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, color_mode='L', grid_size = 60)