import os
import numpy as np

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple


class Colored_MNIST(Dataset):
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
        self.data, self.targets = self._load_data()
        self.targets = np.eye(30)[self.targets]
        
    def _load_data(self):
        self.data = np.load(f"{self.root}/Color_MNIST/{'Train' if self.train else 'Test'}_imgs.npy")
        self.targets = np.load(f"{self.root}/Color_MNIST/{'Train' if self.train else 'Test'}_labels.npy")
        label_to_idx = {}
        for i in range(30):
            for c in ['red', 'green', 'blue']:
                for n in range(10):
                    label_to_idx[c+'_'+n] = i


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)