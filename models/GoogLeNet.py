import torch.nn as nn
import torchvision.models as models

class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=30):
        super(GoogLeNet, self).__init__()
        self.model = models.GoogLeNet(init_weights=False)
        self.model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_channels)
    def forward(self, x):
        if type(self.model(x)) == models.GoogLeNetOutputs:
            return self.model(x).logits
        return self.model(x)