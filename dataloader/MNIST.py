import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from mlxtend.data import loadlocal_mnist
from typing import Any, Callable, Optional, Tuple

class MNISTDataset(Dataset):
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
        self.targets = np.eye(10)[self.targets]
        
    def _load_data(self):
        image_file = f"{self.root}/MNIST/{'train' if self.train else 't10k'}-images.idx3-ubyte"
        label_file = f"{self.root}/MNIST/{'train' if self.train else 't10k'}-labels.idx1-ubyte"
        data, targets = loadlocal_mnist(
            images_path=os.path.join(self.root, image_file),
            labels_path=os.path.join(self.root, label_file)
        )
        data = data.reshape(len(data), 28, 28)
        all_masked_data = data.copy()
        all_targets = targets.copy()
        if self.augmentation:
            all_masked_data = all_masked_data[0:1, :, :]
            all_targets = all_targets[0:1]
            mask_value = 0
            positions = [
                [(0, 0), (14, 14)], # 左上
                [(0, 14), (14, 28)], # 右上
                [(14, 0), (28, 14)], # 左下
                [(14, 14), (28, 28)], # 右下
                [(0, 0), (14, 28)], # 上半
                [(14, 0), (28, 28)], # 下半
            ]
            
            for position in positions:
                left, right = position
                mask = np.zeros_like(data)
                mask[:, left[0]:right[0], left[1]:right[1]] = 1.0
                masked_data = np.where(mask == 1.0, mask_value, data)
                all_masked_data = np.concatenate((all_masked_data, masked_data), axis=0)
                all_targets = np.concatenate((all_targets, targets), axis=0)
            np.delete(all_masked_data, 0, axis=0)
            np.delete(all_targets, 0, axis=0)
        
        return all_masked_data, all_targets
        
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