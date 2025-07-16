import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Optional
import math

# CNN
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 3
        self.layer1 = self.make_layer(64, 0.3, downsample=False)
        self.layer2 = self.make_layer(128, 0.3, downsample=True)
        self.layer3 = self.make_layer(256, 0.4, downsample=True)
        self.layer4 = self.make_layer(512, 0.5, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def make_layer(self, planes: int, dropout: float, downsample: bool) -> nn.Sequential:
        layer = [
            nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        ]
        if downsample:
            layer.append(nn.MaxPool2d(2, 2))
        layer.append(nn.Dropout(dropout))
        
        self.in_planes = planes

        return nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.layer1(x) # (B, 64, H, W)
        out = self.layer2(out) # (B, 128, H, W)
        out = self.layer3(out) # (B, 256, H, W)
        out = self.layer4(out) # (B, 512, H, W)

        out = self.avgpool(out) # (B, 512, 1, 1)
        out = torch.flatten(out, 1) # (B, 512)
        out = self.fc(out) # (B, num_classes)

        return out

# ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor: # (B, in_planes, H, W)
        identity = x

        out = self.conv1(x) # (B, planes, H, W)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # (B, planes, H, W)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) # (B, planes, H, W)

        out += identity # (B, planes, H, W)
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes), 
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        layers.append(BasicBlock(self.in_planes, planes))

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
        out = torch.flatten(out, 1) # (B, 512)
        out = self.fc(out) # (B, num_classes)

        return out

# ViT
class PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, emb_dim: int = 256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.projection(x) # (B, emb_dim, H//patch_size, W//patch_size)
        out = out.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)
        return out


class Embeddings(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, emb_dim: int = 256):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, emb_dim))

    def forward(self, x): # (B, C, H, W)
        out = self.patch_embeddings(x) # (B, num_patches, emb_dim)
        batch_size, _, _ = out.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, emb_dim)
        out = torch.cat([cls_tokens, out], dim=1) # (B, num_patches + 1, emb_dim)
        out = out + self.position_embeddings # (B, num_patches + 1, emb_dim)
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.head_size = emb_dim // num_heads
        assert self.head_size * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        self.qkv_projection = nn.Linear(emb_dim, emb_dim * 3)
        self.output_projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        qkv = self.qkv_projection(x) # (B, num_patches + 1, emb_dim * 3)
        query, key, value = torch.chunk(qkv, 3, dim=-1) # (B, num_patches + 1, emb_dim)
        batch_size, emb_len, _ = query.size()
        query = query.view(batch_size, emb_len, -1, self.head_size).transpose(1, 2) # (B, num_heads, num_patches + 1, head_size)
        key = key.view(batch_size, emb_len, -1, self.head_size).transpose(1, 2)
        value = value.view(batch_size, emb_len, -1, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) # (B, num_heads, num_patches + 1, num_patches + 1)
        scores = scores / math.sqrt(self.head_size)
        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, value) # (B, num_heads, num_patches + 1, head_size)
        output = output.transpose(1, 2).contiguous().view(batch_size, emb_len, -1) # (B, num_patches + 1, emb_dim)
        output = self.output_projection(output) # (B, num_patches + 1, emb_dim)
        return output


class MLP(nn.Module):
    def __init__(self, emb_dim: int = 256, intermediate_dim: int = 4*256):
        super().__init__()
        self.dense1 = nn.Linear(emb_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(intermediate_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
        out = self.dense1(x) # (B, num_patches + 1, intermediate_dim)
        out = self.gelu(out)
        out = self.dense2(out) # (B, num_patches + 1, emb_dim)
        return out


class Block(nn.Module):
    def __init__(self, emb_dim: int = 256, num_heads: int = 8, intermediate_dim: int = 4*256):
        super().__init__()
        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, intermediate_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
        mha_output = self.mha(self.layernorm1(x)) # (B, num_patches + 1, emb_dim)
        out = x + mha_output
        mlp_output = self.mlp(self.layernorm2(out)) # (B, num_patches + 1, emb_dim)
        out = out + mlp_output
        return out


class Encoder(nn.Module):
    def __init__(self, num_blocks: int = 6, emb_dim: int = 256, num_heads: int = 8, intermediate_dim: int = 4*256):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads, intermediate_dim) for _ in range(num_blocks)])

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        out = self.blocks(x) # (B, num_patches + 1, emb_dim)
        return out


class ViT(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, emb_dim: int = 256, num_classes: int = 10, num_blocks: int = 6, num_heads: int = 8, intermediate_dim: int = 4*256, initializer_range: float = 0.02):
        super().__init__()
        self.initializer_range = initializer_range
        self.embedding = Embeddings(image_size, patch_size, emb_dim)
        self.encoder = Encoder(num_blocks, emb_dim, num_heads, intermediate_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        embedding_output = self.embedding(x) # (B, num_patches + 1, emb_dim)
        encoder_output = self.encoder(embedding_output) # (B, num_patches + 1, emb_dim)
        out = self.fc(encoder_output[:, 0]) # (B, num_classes)
        return out
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.cls_token.dtype)