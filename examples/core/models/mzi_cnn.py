import torch.nn as nn
import torch
import torchonn as onn

class MZI_CLASS_CNN(nn.Module):
    def __init__(self, in_channels=1, hidden_sizes=[32,64], num_classes=10, device=None):
        super().__init__()
        self.layer1 = onn.layers.MZIBlockConv2d(in_channels, hidden_sizes[0], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.layer2 = onn.layers.MZIBlockConv2d(hidden_sizes[0], hidden_sizes[1], kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = onn.layers.MZIBlockLinear(hidden_sizes[1]*7*7, num_classes)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.pool(x)
        x = torch.relu(self.layer2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x