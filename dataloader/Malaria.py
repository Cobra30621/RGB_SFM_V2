import torch
import os
import numpy as np

from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms, datasets
from torchvision.io import read_image

class MalariaCellDataset(DatasetFolder):
    def __init__(self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root + '/cell_images/',
            default_loader,
            IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=None,
        )
        self.data = self.samples
        train_imgs, train_labels, test_imgs, test_labels = self.split_data(self.data, self.targets)

        if train:
            self.data, self.targets = train_imgs, train_labels
        else:
            self.data, self.targets = test_imgs, test_labels

    def split_data(self, images, labels, test_size = 0.2):
        num_train = len(images)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        test_split = int(np.floor((test_size) * num_train))
        test_idx, train_idx = indices[:test_split], indices[test_split:]
        images, labels = np.array(images), np.array(labels)
        return  images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.data[index][0], self.targets[index]

        img = read_image(path).permute(1,2,0).detach().numpy()
        target = torch.Tensor(np.eye(2)[target])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)