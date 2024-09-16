import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=4 , bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, out_channels)

    def forward(self, x):
        return self.model(x)