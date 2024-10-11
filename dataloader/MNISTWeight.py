import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from mlxtend.data import loadlocal_mnist
from typing import Any, Callable, Optional, Tuple

class MNISTWeightDataset(Dataset):
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
        self.targets = np.eye(2)[self.targets]

    def _load_data(self):
        image_file = f"{self.root}/MNIST/{'train' if self.train else 't10k'}-images.idx3-ubyte"
        label_file = f"{self.root}/MNIST/{'train' if self.train else 't10k'}-labels.idx1-ubyte"
        data, targets = loadlocal_mnist(
            images_path=os.path.join(self.root, image_file),
            labels_path=os.path.join(self.root, label_file)
        )
        data = data.reshape(len(data), 28, 28)
        
        # 新增的過濾邏輯
        filtered_indices = np.where((targets == 0) | (targets == 1))[0]
        filtered_data = data[filtered_indices]
        filtered_targets = targets[filtered_indices]

        print(targets.shape)
        print(filtered_targets.shape)

        # 將 1 的資料數量乘以 0.1
        one_indices = np.where(filtered_targets == 1)[0]
        reduced_one_count = int(len(one_indices) * 0.1)
        reduced_one_indices = np.random.choice(one_indices, reduced_one_count, replace=False)

        # 合併 0 的資料和減少後的 1 的資料
        all_masked_data = np.concatenate((filtered_data[filtered_targets == 0], filtered_data[reduced_one_indices]), axis=0)
        all_targets = np.concatenate((filtered_targets[filtered_targets == 0], filtered_targets[reduced_one_indices]), axis=0)

        print(all_targets.shape)

        if self.augmentation:
            all_masked_data = all_masked_data[0:1, :, :]
            all_targets = all_targets[0:1]
            mask_value = 0
            positions = [
                [(0, 0), (14, 14)],  # 左上
                [(0, 14), (14, 28)],  # 右上
                [(14, 0), (28, 14)],  # 左下
                [(14, 14), (28, 28)],  # 右下
                [(0, 0), (14, 28)],  # 上半
                [(14, 0), (28, 28)],  # 下半
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