import torch

from torch.utils.data import DataLoader

from .HeartCalcification import *
from .MNIST import MNISTDataset
from .MultiColorShapes import MultiColorShapesDataset
from .FaceDataset import FaceDataset
from .Malaria import MalariaCellDataset
from .RGB_circle import RGBCircle
from .MultiGrayShapes import MultiGrayShapesDataset
from .MultiEdgeShapes import MultiEdgeShapesDataset
from .Colored_MNIST import Colored_MNIST
from .Colored_FashionMNIST import Colored_FashionMNIST
from .AnotherColored_MNIST import AnotherColored_MNIST
from .AnotherColored_FashionMNIST import AnotherColored_FashionMNIST
from .CIFAR10 import CIFAR10
from .Colorful_MNIST import Colorful_MNIST
from torchvision import transforms

dataset_classes = {
    'mnist': MNISTDataset,
    'MultiColor_Shapes_Database': MultiColorShapesDataset,
    'face_dataset': FaceDataset,
    'malaria':MalariaCellDataset,
    'RGB_Circle':RGBCircle,
    'MultiGrayShapesDataset': MultiGrayShapesDataset,
    'MultiEdgeShapes': MultiEdgeShapesDataset,
    'Colored_MNIST':Colored_MNIST,
    'Colored_FashionMNIST':Colored_FashionMNIST,
    'AnotherColored_MNIST':AnotherColored_MNIST,
    'AnotherColored_FashionMNIST':AnotherColored_FashionMNIST,
    'CIFAR10': CIFAR10,
    'Colorful_MNIST':Colorful_MNIST,
    'HeartCalcification_Color': HeartCalcificationColor,
    'HeartCalcification_Gray': HeartCalcificationGray,
    'HeartCalcificationGray60' : HeartCalcificationGray60
}

def get_dataloader(dataset, root: str = '.', batch_size=32, input_size: tuple = (28, 28)):
    if dataset in dataset_classes:
        train_transform = transforms.Compose([
            transforms.Resize([*input_size]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

        test_transform = transforms.Compose([
            transforms.Resize([*input_size]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])
        
        train_dataset = dataset_classes[dataset](root, train = True, transform = train_transform)
        test_dataset = dataset_classes[dataset](root, train = False, transform = test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")