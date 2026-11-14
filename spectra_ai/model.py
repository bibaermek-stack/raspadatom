"""1D-CNN + Transformer (Attention) моделі: көп белгілі түр классификациясы + температура регрессиясы."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    """Позиция кодтауы: онсыз Attention толқын ұзындығындағы орынды білмейді."""
    pos = torch.arange(length).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def conv_block(cin, cout, k):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU(), nn.MaxPool1d(4)
    )


class SpectralAttentionNet(nn.Module):
    def __init__(self, num_species: int, n_points: int = 2048, d_model: int = 64, num_layers: int = 2):
        super().__init__()
        # n_points -> n/4 -> n/16 токен (2048 -> 128, 1024 -> 64)
        self.conv = nn.Sequential(conv_block(2, 32, 7), conv_block(32, d_model, 5))
        tokens = n_points // 16
        self.register_buffer("pos", sinusoidal_positions(tokens, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=2 * d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.cls_head = nn.Linear(2 * d_model, num_species)  # логиттер (sigmoid -> бар/жоқ)
        self.temp_head = nn.Linear(2 * d_model, 1)           # T_norm ∈ [0, 1]

    def forward(self, x):                       # x: (B, 1, 2048), 0..1
        # 2 арна: сызықтық және √ (әлсіз сызықтарды шудан бөліп алуға көмектеседі)
        x = torch.cat([x, x.clamp_min(0).sqrt()], dim=1)
        f = self.conv(x).permute(0, 2, 1)       # (B, 128, d)
        f = self.attention(f + self.pos)
        z = torch.cat([f.mean(1), f.amax(1)], dim=1)
        return self.cls_head(z), self.temp_head(z).squeeze(-1)
