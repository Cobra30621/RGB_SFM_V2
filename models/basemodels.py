import os
import math
import torch
import threading
import torchvision
import torchvision.models as models
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.transforms import Grayscale
from operator import truediv
from utils import _pair, get_rbf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = os.path.dirname(__file__)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()        
        self.fc1 = nn.Linear(28*28, 10) 
        self.softmax = nn.Softmax(-1) 
    def forward(self, x):
        x = x.view(x.size(0), -1)       
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels = 8):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, padding=3),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1), 
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1152, 512),
            nn.Linear(512, out_channels)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, layers=18):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        if layers == 34:
            self.model = models.resnet34(pretrained=True)
        elif layers == 50:
            self.model = models.resnet50(pretrained=True)
        elif layers == 101:
            self.model = models.resnet101(pretrained=True)
        elif layers == 152:
            self.model = models.resnet152(pretrained=True)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=4, padding=2, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        return self.model(x)
    
    
class AlexNet(nn.Module):   
    def __init__(self, num=10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*5*5,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1,32*5*5)
        x = self.classifier(x)
        return x
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.model = models.GoogLeNet(init_weights=True)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        if type(self.model(x)) == torchvision.models.GoogLeNetOutputs:
            return self.model(x).logits
        return self.model(x)