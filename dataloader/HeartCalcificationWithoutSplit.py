import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class HeartCalcificationWithoutSplit(Dataset):
    def __init__(self, data_dir, grid_size=15):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.label_files = [f.replace('.png', '.txt') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        label_path = os.path.join(self.data_dir, self.label_files[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                _, x_center, y_center, w, h = map(float, line.split())
                x = int(x_center * width)
                y = int(y_center * height)
            else:
                x, y = -1, -1

        num_blocks_h = height // self.grid_size
        num_blocks_w = width // self.grid_size
        label = np.zeros((num_blocks_h, num_blocks_w), dtype=np.int8)

        with open(label_path , 'r') as f:
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

        return img, label


