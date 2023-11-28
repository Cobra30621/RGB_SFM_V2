from torch import nn

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