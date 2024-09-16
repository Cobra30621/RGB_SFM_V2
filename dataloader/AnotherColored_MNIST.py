import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple


class AnotherColored_MNIST(Dataset):
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
        colors = {
            'brown': [151, 74, 0],
            'light_blue': [121, 196, 208],
            'light_pink': [221, 180, 212]
        }

        self.label_to_idx = {}
        i = 0
        for c in colors.keys():
            for n in range(10):
                self.label_to_idx[c+'_'+str(n)] = i
                i+=1

        self.data = np.load(f"{self.root}/AnotherColored_MNIST/{'Train' if self.train else 'Test'}_imgs.npy")
        print(f"{self.root}/AnotherColored_MNIST/{'Train' if self.train else 'Test'}_imgs.npy")
        print(len(self.data))
        self.data = self.data.astype(float)

        self.targets = np.load(f"{self.root}/AnotherColored_MNIST/{'Train' if self.train else 'Test'}_labels.npy")
        self.targets = list(map(lambda x: np.eye(30)[self.label_to_idx[x]], self.targets))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)
        if torch.max(img) > 1:
            img /= 255

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)