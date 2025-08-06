import torch
import torch.nn as nn
from torch import Tensor

class CNN(nn.Module):
    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.in_channels = 3
        self.layer1 = self.make_layer(64, 0.1, downsample=False)
        self.layer2 = self.make_layer(128, 0.25, downsample=True)
        self.layer3 = self.make_layer(256, 0.25, downsample=True)
        self.layer4 = self.make_layer(512, 0.3, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim)
        )

    def make_layer(self, out_channels: int, dropout: float, downsample: bool) -> nn.Sequential:
        layer = [
            nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if downsample:
            layer.append(nn.MaxPool2d(2, 2))
        layer.append(nn.Dropout2d(dropout))
        
        self.in_channels = out_channels

        return nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.layer1(x) # (B, 64, H, W)
        out = self.layer2(out) # (B, 128, H, W)
        out = self.layer3(out) # (B, 256, H, W)
        out = self.layer4(out) # (B, 512, H, W)

        out = self.avgpool(out) # (B, 512, 1, 1)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out) # (B, out_dim)
        return out