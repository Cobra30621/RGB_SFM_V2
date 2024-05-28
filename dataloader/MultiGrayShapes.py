import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt

class MultiGrayShapesDataset(Dataset):
    def __init__(self,
        root: str,
        train: bool = True,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.train = train
        self.augmentation = augmentation
        labels = ['circle', 'rectangle', 'triangle']
        self.label_to_num = {k:i for i,k in enumerate(labels)}
        self.num_to_label = {i:k for i,k in enumerate(labels)}
        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{self.root}/MultiColor_Shapes_Database/{'train' if self.train else 'test'}/"
        image_dataset = []
        label_dataset = []
        for root, dirs, files in os.walk(image_file):
            for name in files:
                name_split = '_'.join(name.split('_')[:1])
                label = self.label_to_num[name_split]
                y_onehot = np.eye(3)[label]
                y_onehot = torch.from_numpy(y_onehot)
                label_dataset.append(y_onehot)

                image = read_image(os.path.join(root, name), mode=ImageReadMode.GRAY)
                image = image.permute(1,2,0).detach().numpy()
                image_dataset.append(image)

        return image_dataset, label_dataset
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
            threshold = torch.mean(img, dtype=float)
            img =(img<=threshold).to(img.dtype)

        if self.target_transform is not None:
            target = self.target_transform(target)

        

        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def reverse_bgcolor(self, image):
        image = image.permute(1,2,0)
        black_color = torch.tensor([0, 0, 0], dtype=torch.uint8)  # 黑色的 RGB 值
        # 定义背景颜色和蓝色球颜色的阈值
        background_threshold = 0.8  # 背景的阈值（每个通道大于这个值的像素被认为是背景）
        # 找到背景和蓝色球的像素位置
        background_indices = torch.all(image >= background_threshold, dim=-1)
        image[background_indices] = black_color
        image = image.permute(2,0,1)
        return image
