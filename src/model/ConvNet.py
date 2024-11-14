import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))  # 6x28x28
        x = self.pool(x)  # 6x14x14
        x = self.sigmoid(self.conv2(x))  # 16x10x10
        x = self.pool(x)  # 16x5x5
        x = x.view(-1, 16 * 5 * 5)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x