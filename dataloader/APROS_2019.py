import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional
import torch

from diabetic_retinopathy_handler import preprocess_retinal_tensor_image


# https://www.kaggle.com/datasets/mariaherrerot/aptos2019/data
class APROS_2019Dataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 use_retina_preprocess: bool = True,
                 target_radius: int = 224,
                 target_transform: Optional[Callable] = None):
        """
        Retina 圖像預處理版 APROS_2019 Dataset

        Args:
            root (str): 資料根目錄
            train (bool): 使用訓練集或測試集
            use_retina_preprocess (bool): 是否使用 retina 預處理
            target_radius (int): retina 預處理的目標半徑
            target_transform (Callable, optional): 標籤轉換器
        """
        csv_file = f"{root}/APROS_2019/{'train_1.csv' if train else 'test.csv'}"
        img_dir = f"{root}/APROS_2019/{'train_images/train_images' if train else 'test_images/test_images'}"
        print(f"[Dataset] Using CSV: {csv_file}")
        print(f"[Dataset] Image folder: {img_dir}")

        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.use_retina_preprocess = use_retina_preprocess
        self.target_radius = target_radius
        self.target_transform = target_transform

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0] + '.png')
        image_pil = Image.open(img_path).convert("RGB")
        label = int(self.data_frame.iloc[idx, 1])

        # One-hot encoding
        one_hot_label = np.zeros(5, dtype=np.float32)
        one_hot_label[label] = 1.0

        if self.target_transform:
            one_hot_label = self.target_transform(one_hot_label)

        # PIL → Tensor
        image_tensor = self.to_tensor(image_pil)

        if self.use_retina_preprocess:
            _, _, _, final_tensor = preprocess_retinal_tensor_image(image_tensor, self.target_radius)
            image_tensor = final_tensor  # 只保留最終處理結果

        return image_tensor, torch.tensor(one_hot_label)