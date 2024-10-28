import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

import medmnist
from medmnist import INFO


class CustomerMedMNIST(Dataset):
    """
    醫學分裂影像 : https://github.com/MedMNIST/MedMNIST/tree/main
    圖像皆為 28 * 28
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            augmentation: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            data_flag : str = "pathmnist",
    ) -> None:
        if train:
            split = "train"
        else:
            split = "test"

        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        self.dataset = DataClass(transform = transform, download=True, split=split)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.dataset.__getitem__(index)
        #
        # 将 one-hot 编码转换为类标签
        y_onehot = np.eye(9)[label]  # 这行需要修改
        y_onehot = torch.tensor(y_onehot, dtype=torch.float)  # 转换为张量
        y_onehot = y_onehot.squeeze(0)  # 去掉多余的维度
        return img, y_onehot  # 返回类标签

        # return self.dataset.__getitem__(index)



    def __len__(self) -> int:
        return self.dataset.__len__()


def get_data_class(data_flag):
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])


class CustomerPathMNIST(CustomerMedMNIST):
    flag = "pathmnist"

class CustomerDermaMNIST(CustomerMedMNIST):
    flag = "dermamnist"

class CustomerRetinaMNIST(CustomerMedMNIST):
    flag = "retinamnist"

class CustomerBloodMNIST(CustomerMedMNIST):
    flag = "bloodmnist"
