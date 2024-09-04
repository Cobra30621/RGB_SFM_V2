import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms, datasets
from torchvision.io import read_image
from PIL import Image

class MalariaCellDataset(Dataset):
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
        
        self.data, self.targets = self._load_data()

        train_imgs, train_labels, test_imgs, test_labels = self.split_data(self.data, self.targets)

        if train:
            self.data, self.targets = train_imgs, train_labels
        else:
            self.data, self.targets = test_imgs, test_labels
        

    def split_data(self, images, labels, test_size = 0.2):
        num_train = len(images)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        test_split = int(np.floor((test_size) * num_train))
        test_idx, train_idx = indices[:test_split], indices[test_split:]
        images, labels = np.array(images), np.array(labels)
        return  images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]

    def _load_data(self):
        image_file = f"{self.root}/Malaria_Dataset/"
        image_dataset = []
        label_dataset = []
        label_to_num = {'Parasitized':0, 'Uninfected':1}
        for root, dirs, files in os.walk(image_file):
            for name in files:
                name_split = root.split('/')[-1]
                label = label_to_num[name_split]
                y_onehot = np.eye(2)[label]
                y_onehot = torch.from_numpy(y_onehot)
                label_dataset.append(y_onehot)

                image = read_image(os.path.join(root, name))
                image = transforms.Resize([224,224])(image)
                image = image.detach().numpy().transpose(1,2,0)
                image_dataset.append(image)

        return image_dataset, label_dataset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(img))
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)