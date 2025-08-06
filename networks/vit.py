import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, emb_dim: int = 512):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(3, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.projection(x) # (B, emb_dim, H//patch_size, W//patch_size)
        out = out.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)
        return out

class Embeddings(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, emb_dim: int = 512):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, emb_dim))
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        out = self.patch_embeddings(x) # (B, num_patches, emb_dim)
        batch_size, _, _ = out.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, emb_dim)
        out = torch.cat([cls_tokens, out], dim=1) # (B, num_patches + 1, emb_dim)
        out = out + self.position_embeddings # (B, num_patches + 1, emb_dim)
        return out 
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.1)
        nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.1)
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.head_size = emb_dim // num_heads
        assert self.head_size * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        self.qkv_projection = nn.Linear(emb_dim, emb_dim * 3)
        self.output_projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
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
    def __init__(self, emb_dim: int = 512, intermediate_dim: int = 4*512, dropout: int = 0.5):
        super().__init__()
        self.dense1 = nn.Linear(emb_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(intermediate_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
        out = self.dense1(x) # (B, num_patches + 1, intermediate_dim)
        out = self.gelu(out)
        out = self.dense2(out) # (B, num_patches + 1, emb_dim)
        return out


class Block(nn.Module):
    def __init__(self, emb_dim: int = 512, num_heads: int = 8, intermediate_dim: int = 4*512, dropout: int = 0.5):
        super().__init__()
        self.mha = MultiHeadAttention(emb_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, intermediate_dim, dropout)
        self.layernorm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
        mha_output = self.mha(self.layernorm1(x)) # (B, num_patches + 1, emb_dim)
        out = x + mha_output
        mlp_output = self.mlp(self.layernorm2(out)) # (B, num_patches + 1, emb_dim)
        out = out + mlp_output
        return out


class Encoder(nn.Module):
    def __init__(self, num_blocks: int = 6, emb_dim: int = 512, num_heads: int = 8, intermediate_dim: int = 4*512, dropout: int = 0.5):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads, intermediate_dim, dropout) for _ in range(num_blocks)])

    def forward(self, x: Tensor) -> Tensor: # (B, num_patches + 1, emb_dim)
        out = self.blocks(x) # (B, num_patches + 1, emb_dim)
        return out


class ViT(nn.Module):
    def __init__(self, out_dim: int = 10, image_size: int = 32, patch_size: int = 4, emb_dim: int = 512, num_blocks: int = 6, num_heads: int = 8, intermediate_dim: int = 4*512):
        super().__init__()
        self.embedding = Embeddings(image_size, patch_size, emb_dim)
        self.encoder = Encoder(num_blocks, emb_dim, num_heads, intermediate_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor: # (B, C, H, W)
        embedding_output = self.embedding(x) # (B, num_patches + 1, emb_dim)
        encoder_output = self.encoder(embedding_output) # (B, num_patches + 1, emb_dim)
        out = self.fc(encoder_output[:, 0]) # (B, out_dim)
        return out