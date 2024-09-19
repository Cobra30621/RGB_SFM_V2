import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image

class HeartCalcificationDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        augmentation: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
        # grid_size: int = 15
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.augmentation = augmentation
        # self.grid_size = grid_size
        self.grid_size = 15
        self._load_data()

    def _load_data(self):
        self.data_dir = f"{self.root}/HeartCalcification/{'train' if self.train else 'test'}"

        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        self.label_files = [f.replace('.png', '.txt') for f in self.image_files]


        self.splited_images = []
        self.splited_label = []

        for img_path, label_path in zip(self.image_files, self.label_files):
            img = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
            width, height = img.size

            num_blocks_h = height // self.grid_size
            num_blocks_w = width // self.grid_size
            label = np.zeros((num_blocks_h, num_blocks_w), dtype=np.int8)

            with open(os.path.join(self.data_dir, label_path), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        _, x_center, y_center, w, h = map(float, line.split())
                        x = int(x_center * width)
                        y = int(y_center * height)

                        block_i = y // self.grid_size
                        block_j = x // self.grid_size

                        if 0 <= block_i < num_blocks_h and 0 <= block_j < num_blocks_w:
                            label[block_i, block_j] = 1

            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block_img = img.crop((j*self.grid_size, i*self.grid_size, (j+1)*self.grid_size, (i+1)*self.grid_size))
                    self.splited_images.append(block_img)
                    y_onehot = np.eye(2)[label[i, j]]
                    y_onehot = torch.from_numpy(y_onehot)
                    self.splited_label.append(y_onehot)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = self.splited_images[idx]
        label = self.splited_label[idx]

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.splited_images)