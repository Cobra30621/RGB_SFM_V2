import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple

class FaceDataset(Dataset):
    def __init__(self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = self._load_data()
        train_imgs, train_labels, test_imgs, test_labels = self.split_data(self.data, self.targets)

        if train:
            self.data, self.targets  = train_imgs, train_labels
        else:
            self.data, self.targets = test_imgs, test_labels
        
    def _load_data(self):
        face_images = np.load(f"{self.root}/face_dataset/face_correspond_64_unit8.npy")
        face_images = face_images[:6000]

        baseball_images = np.load(f"{self.root}/face_dataset/baseball_64_unit8.npy")
        baseball_images = baseball_images[:2000]

        apple_images = np.load(f"{self.root}/face_dataset/apple_64_unit8.npy")
        apple_images = apple_images[:2000]

        circle_images = np.load(f"{self.root}/face_dataset/circle_64_unit8.npy")
        circle_images = circle_images[:2000]

        images = np.concatenate((face_images, baseball_images, apple_images, circle_images), axis=0, dtype=np.uint8)
        labels = torch.Tensor([0] * face_images.shape[0] + [1] * baseball_images.shape[0] + [1] * apple_images.shape[0] + [1] * circle_images.shape[0])
        return images, labels
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def target_to_oh(self, target):
        NUM_CLASS = 2  # hard code here, can do partial
        one_hot = torch.eye(NUM_CLASS)[target.long()]
        return one_hot
    
    def split_data(self, images, labels, test_size = 0.2):
        num_train = len(images)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        test_split = int(np.floor((test_size) * num_train))
        test_idx, train_idx = indices[:test_split], indices[test_split:]
        return  images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

    def __len__(self):
        return self.data.shape[0]