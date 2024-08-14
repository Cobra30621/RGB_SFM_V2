import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import Any, Callable, Optional, Tuple


class RGBCircle(Dataset):
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
        labels = ['red_circle', 'green_circle', 'blue_circle']
        self.label_to_num = {k:i for i,k in enumerate(labels)}
        self.num_to_label = {i:k for i,k in enumerate(labels)}
        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f"{self.root}/RGB_circle/{'Train' if self.train else 'Test'}/"
        image_dataset = []
        label_dataset = []
        for root, dirs, files in os.walk(image_file):
            for name in files:
                name_split = '_'.join(name.split('_')[:2])
                label = self.label_to_num[name_split]
                y_onehot = np.eye(3)[label]
                y_onehot = torch.from_numpy(y_onehot)
                label_dataset.append(y_onehot)

                image = read_image(os.path.join(root, name))
                image = image.detach().numpy().transpose(1,2,0)
                image_dataset.append(image)

        return image_dataset, label_dataset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.data)