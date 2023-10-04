import os
import numpy as np
from mlxtend.data import loadlocal_mnist
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.io import read_image

root = os.path.dirname(__file__)

class CustomMNISTDataset(Dataset):

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
        
    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images.idx3-ubyte"
        label_file = f"{'train' if self.train else 't10k'}-labels.idx1-ubyte"
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

        img, target = self.data[index], int(self.targets[index])
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)

class RGBSimpleShapeDataset(Dataset):
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
        
        self.train = train
        self.augmentation = augmentation
        self.data, self.targets = self._load_data()

    def _load_data(self):
        image_file = f'{self.root}/{'train' if self.train else 'test'}'
        image_dataset = []
        label_dataset = []
        for root, dirs, files in os.walk(image_file):
            for name in files:
                name_split = name.split('_')
                if name_split[0] == 'red':
                    color = 0
                elif name_split[0] == 'green':
                    color = 1
                elif name_split[0] == 'blue':
                    color = 2

                if name_split[1] == 'circle':
                    shape = 0
                elif name_split[1] == 'ellipse':
                    shape = 1
                elif name_split[1] == 'rectangle':
                    shape = 2
                elif name_split[1] == 'square':
                    shape = 3
                elif name_split[1] == 'triangle':
                    shape = 4
                label_dataset.append([name_split[0], name_split[1]])
                image = read_image(os.path.join(root, name))
                image_dataset.append(image)

        return image_dataset, label_dataset
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, color, shape = self.data[index], self.targets[index][0], self.targets[index][1]
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            color = self.target_transform(color)
            shape = self.target_transform(shape)

        return img, color, shape
    
    def __len__(self):
        return len(self.data)

    

def get_MalariaCellImagesDataset(root: str = f"{root}/data/cell_images/", resize=[224, 224], valid_size=0.2, test_size = 0.1, batch_size=27558, shuffle=True):
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),  # 更改明亮度
        transforms.RandomRotation(
            degrees=45, expand = True
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])                                           
    train_data = datasets.ImageFolder(root, transform=train_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_train))
    test_split = int(np.floor((valid_size+test_size) * num_train))
    valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    test_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_loader, valid_loader, test_loader

def get_MalariaCellImagesDataset_split(root: str = f"{root}/data/cell_images_split/", resize=[224, 224], valid_size=0.2, test_size = 0.1, batch_size=27558, shuffle=True):
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),  # 更改明亮度
        transforms.RandomRotation(
            degrees=45, expand = True
        ),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root+'train/', transform=train_transforms)
    test_data = datasets.ImageFolder(root+'test/', transform=train_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    if shuffle:
        np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_train))
    # test_split = int(np.floor((valid_size+test_size) * num_train))
    # valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]
    valid_idx, train_idx = indices[:valid_split], indices[valid_split:]
    test_idx = list(range(len(test_data)))

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx))
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_loader, valid_loader, test_loader

def load_data(dataset: str = 'mnist', root: str = '.', batch_size: int = 256, input_size: tuple = (28, 28)):
    if dataset == 'mnist_aug':
        test_aug_data = CustomMNISTDataset(
            root=f'{root}/data/MNIST',
            train=False,
            transform=ToTensor(),
            augmentation=True
        )
        return DataLoader(test_aug_data, batch_size=batch_size, shuffle=True)
    
    
    if dataset == 'mnist':
        training_data = CustomMNISTDataset(
            root=f'{root}/data/MNIST',
            train=True,
            transform=ToTensor()
        )

        test_data = CustomMNISTDataset(
            root=f'{root}/data/MNIST',
            train=False,
            transform=ToTensor(),
            augmentation=False
        )
        
    elif dataset == 'fashion':
        training_data = datasets.FashionMNIST(
            root=f"{root}/data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root=f"{root}/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        
    elif dataset == 'cifar10':
        training_data = datasets.CIFAR10(
            root=f"{root}/data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        test_data = datasets.CIFAR10(
            root=f"{root}/data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    elif dataset == 'rgb_simple_shape':
        training_data = RGBSimpleShapeDataset(
            root=f"{root}/data/rgb-simple-shape",
            train=True,
            transform=ToTensor()
        )

        test_data = RGBSimpleShapeDataset(
            root=f"{root}/data",
            train=False,
            transform=ToTensor()
        )
    
    if dataset == 'malaria':
        train_dataloader, valid_dataloader, test_dataloader = get_MalariaCellImagesDataset(root=f"{root}/data/cell_images/", resize=[*input_size], valid_size=0.0, test_size = 0.2, batch_size=batch_size, shuffle=True)
    elif dataset == 'malaria_split':
        train_dataloader, valid_dataloader, test_dataloader = get_MalariaCellImagesDataset_split(root=f"{root}/data/cell_images_split/", resize=[*input_size], valid_size=0.0, test_size = 0.2, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader