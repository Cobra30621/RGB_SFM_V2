import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 使用预训练的AlexNet模型
        alexnet = models.alexnet(pretrained=True)
        
        # 移除最后一层全连接层
        self.features = nn.Sequential(*list(alexnet.features.children())[:-1])
        
        # 修改第一层卷积层以适应新的输入通道和尺寸
        self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        # 添加全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 新的全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x