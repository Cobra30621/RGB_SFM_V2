import torch

from torch.utils.data import DataLoader
from .MNIST import MNISTDataset
from .MultiColorShapes import MultiColorShapesDataset
from .FaceDataset import FaceDataset
from .Malaria import MalariaCellDataset
from torchvision import transforms

dataset_classes = {
    'mnist': MNISTDataset,
    'MultiColor_Shapes_Database': MultiColorShapesDataset,
    'face_dataset': FaceDataset,
    'malaria':MalariaCellDataset,
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