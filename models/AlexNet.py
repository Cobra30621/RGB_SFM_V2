import torch
import torch.nn as nn
import torchvision.models as models


class AlexNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=10, input_size=(32, 32)):
        super(AlexNet, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=1),  # [B, 32, H1, W1]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        # 模擬一下 feature map 的展平維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)
            dummy_output = self.feature(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.flatten_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
