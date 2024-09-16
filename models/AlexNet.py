import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):   
    def __init__(self, in_channels=3, num=10,**kwargs):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=1),
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