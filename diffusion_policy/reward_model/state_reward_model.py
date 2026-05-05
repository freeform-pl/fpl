"""
State-space reward model for lowdim trajectories.

Architecture:
    Linear projection -> positional encoding -> TransformerEncoder -> CLS tokens -> MLP heads -> K rewards
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class StateRewardModel(nn.Module):
    """
    Takes a state trajectory (B, T, obs_dim) and outputs K reward scalars (B, K).
    """

    def __init__(
        self,
        obs_dim: int = 23,
        num_rewards: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim=embed_dim, dropout=dropout)

        # CLS tokens (one per reward dimension)
        self.cls_tokens = nn.Parameter(torch.randn(1, num_rewards, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reward_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_rewards)
        ])

    def forward(self, states: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            states:       (B, T, obs_dim)
            padding_mask: (B, T) bool — True = padded (ignored)
        Returns:
            rewards: (B, K)
        """
        B, T, _ = states.shape
        K = len(self.reward_heads)

        x = self.proj(states)
        x = self.pos_enc(x)

        cls = self.cls_tokens.expand(B, -1, -1)
        seq = torch.cat([cls, x], dim=1)

        key_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros(B, K, dtype=torch.bool, device=padding_mask.device)
            key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        out = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        cls_out = out[:, :K]

        rewards = torch.stack(
            [head(cls_out[:, i]).squeeze(-1) for i, head in enumerate(self.reward_heads)],
            dim=-1,
        )
        return rewards


def bradley_terry_loss(rewards_a, rewards_b, labels):
    """
    Bradley-Terry preference loss.

    Args:
        rewards_a: (B, K)
        rewards_b: (B, K)
        labels:    (B, K) — 1.0=A preferred, 0.0=B preferred
    Returns:
        loss: scalar
        per_dim_acc: (K,)
    """
    prob_a = torch.sigmoid(rewards_a - rewards_b)
    loss = -labels * torch.log(prob_a + 1e-8) - (1 - labels) * torch.log(1 - prob_a + 1e-8)
    loss = loss.mean()

    with torch.no_grad():
        pred = (prob_a > 0.5).float()
        correct = (pred == labels).float()
        per_dim_acc = correct.mean(dim=0)

    return loss, per_dim_acc
