import torch
import numpy as np

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

import medmnist
from medmnist import INFO

from dataloader.CustomerMedMNIST import CustomerMedMNIST
from diabetic_retinopathy_handler import preprocess_retinal_tensor_image


import torch.nn.functional as F

class PreprocessedRetinaMNIST224(CustomerMedMNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 augmentation: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 target_radius: int = 225,
                 final_size: int = 225):
        """
        RetinaMNIST (224x224) 資料集，圖像經 retina 預處理，並統一 resize 成固定大小。

        Args:
            final_size (int): 最終輸出的 H, W 尺寸，預設為 225
        """
        super().__init__(
            root=root,
            train=train,
            augmentation=augmentation,
            transform=transform,
            target_transform=target_transform,
            flag="retinamnist",
            size=224
        )
        self.target_radius = target_radius
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = self.dataset.__getitem__(index)  # img: Tensor (C, H, W)

        # Retina 預處理
        _, _, _, final_tensor = preprocess_retinal_tensor_image(img, self.target_radius)
        print(f"final 1: {final_tensor.shape}")

        # # Resize to fixed size (e.g. 224x224)
        # final_tensor = F.interpolate(final_tensor.unsqueeze(0), size=(self.final_size, self.final_size), mode='bilinear', align_corners=False)
        # final_tensor = final_tensor.squeeze(0)  # (C, H, W)


        if self.transform is not None:
            final_tensor  = self.transform(final_tensor )
        print(f"final 2: {final_tensor.shape}")

        # One-hot 編碼
        y_onehot = np.eye(self.n_classes)[label]
        y_onehot = torch.tensor(y_onehot, dtype=torch.float).squeeze(0)

        return final_tensor, y_onehot
