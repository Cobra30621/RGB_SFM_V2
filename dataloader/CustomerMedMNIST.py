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
            flag : str = "pathmnist",
            size: int = 28
    ) -> None:
        if train:
            split = "train"
        else:
            split = "test"

        self.info = INFO[flag]
        self.n_classes = len(self.info['label'])
        DataClass = getattr(medmnist, self.info['python_class'])
        self.dataset = DataClass(transform = transform, download=True,
                                 split=split, size=size)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.dataset.__getitem__(index)
        #
        # 将 one-hot 编码转换为类标签
        y_onehot = np.eye(self.n_classes)[label]  # 这行需要修改
        y_onehot = torch.tensor(y_onehot, dtype=torch.float)  # 转换为张量
        y_onehot = y_onehot.squeeze(0)  # 去掉多余的维度
        return img, y_onehot  # 返回类标签




    def __len__(self) -> int:
        return self.dataset.__len__()

class CustomerPathMNIST(CustomerMedMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, flag="pathmnist")

class CustomerDermaMNIST(CustomerMedMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, flag="dermamnist")

class CustomerRetinaMNIST(CustomerMedMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, flag="retinamnist")


class CustomerRetinaMNIST_224(CustomerMedMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, flag="retinamnist", size=224)




class CustomerBloodMNIST(CustomerMedMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, flag="bloodmnist")
