import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor: # (B, in_channels, H, W)
        identity = x

        out = self.conv1(x) # (B, out_channels, H, W)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # (B, out_channels, H, W)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) # (B, out_channels, H, W)

        out += identity # (B, out_channels, H, W)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim)
        )

    def _make_layer(self, out_channels: int, stride: int = 1) -> nn.Sequential:
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels), 
            )
        else:
            downsample = None

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        layers.append(BasicBlock(out_channels, out_channels))
        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.conv1(x) # (B, 64, H, W)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out) # (B, 64, H, W)
        out = self.layer2(out) # (B, 128, H, W)
        out = self.layer3(out) # (B, 256, H, W)
        out = self.layer4(out) # (B, 512, H, W)

        out = self.avgpool(out) # (B, 512, 1, 1)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out) # (B, out_dim)
        return out