import torch
import os
import numpy as np

from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms, datasets

class MalariaCellDataset(DatasetFolder):
    def __init__(self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root,
            default_loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=None,
            allow_empty=False,
        )
        self.imgs = self.samples
        train_imgs, train_labels, test_imgs, test_labels = self.split_data(self.imgs, self.targets)
        if train:
            self.imgs, self.targets = train_imgs, train_labels
        else:
            self.imgs, self.targets = test_imgs, test_labels

    def split_data(self, images, labels, test_size = 0.2):
        num_train = len(images)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        test_split = int(np.floor((test_size) * num_train))
        test_idx, train_idx = indices[:test_split], indices[test_split:]
        return  images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]