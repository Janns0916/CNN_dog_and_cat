import torch.nn as nn
import torch.nn.functional as F


class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.linear1 = nn.Linear(3 * 3 * 64, 64)
        self.linear2 = nn.Linear(64, 10)
        self.output = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):  # 64*3*256*256
        x = self.conv1(x) # 64*16*63*63
        x = self.relu(x)
        x = self.conv2(x)  # 64*32*15*15
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)  # 64*64*3*3
        x = x.view(x.shape[0], -1)  # 64*576
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x