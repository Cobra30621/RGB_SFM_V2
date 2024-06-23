import torch

from torch.utils.data import DataLoader
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
}

def get_dataloader(dataset, root: str = '.', batch_size=32, input_size: tuple = (28, 28)):
    if dataset in dataset_classes:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([*input_size]),
            transforms.ConvertImageDtype(torch.float),
        ])
        
        train_dataset = dataset_classes[dataset](root, train = True, transform = transform)
        test_dataset = dataset_classes[dataset](root, train = False, transform = transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")