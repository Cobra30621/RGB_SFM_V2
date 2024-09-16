import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image


class Colorful_MNIST(Dataset):
    def __init__(
        self,
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
        self._load_data()
        
        
    def _load_data(self):
        self.data = np.load(f"{self.root}/Colorful_MNIST/{'Train' if self.train else 'Test'}_images.npy")

        self.targets = np.load(f"{self.root}/Colorful_MNIST/{'Train' if self.train else 'Test'}_labels.npy")
        self.targets = list(map(lambda x: np.eye(31)[x], self.targets))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)