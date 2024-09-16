import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, layers=18, in_channels=3, out_channels=30):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        if layers == 34:
            self.model = models.resnet34(pretrained=False)
        elif layers == 50:
            self.model = models.resnet50(pretrained=False)
        elif layers == 101:
            self.model = models.resnet101(pretrained=False)
        elif layers == 152:
            self.model = models.resnet152(pretrained=False)
        self.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=4, padding=2, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
    def forward(self, x):
        return self.model(x)