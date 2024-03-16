from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels = 8):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels, out_channels=225, kernel_size=15, stride=15),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(in_channels=225, out_channels=625, kernel_size=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1)),
            nn.Conv2d(in_channels=625, out_channels=1225, kernel_size=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Conv2d(in_channels=1225, out_channels=2025, kernel_size=1), 
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(2*1*2025, out_channels)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x