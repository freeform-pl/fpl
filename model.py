"""
Transformer-based reward model for preference learning.

Architecture:
    1. Per-frame encoder: encodes (third_person, wrist) image pair → embedding
    2. Transformer: processes the sequence of frame embeddings
    3. Reward head: outputs K scalars (one per preference dimension)

Training uses the Bradley-Terry model for pairwise preference loss.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models



# ---------------------------------------------------------------------------
# Frame encoder
# ---------------------------------------------------------------------------

class FrameEncoder(nn.Module):
    """
    Encodes a (third_person, wrist) image pair into a single embedding.

    Both images go through a shared backbone; their features are concatenated
    and projected to `embed_dim`.
    """

    def __init__(self, embed_dim: int = 256, backbone: str = "resnet18", frozen_backbone: bool = False):
        super().__init__()

        if backbone == "resnet18":
            net = tv_models.resnet18(weights=None)
            feature_dim = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if frozen_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Two cameras → 2 * feature_dim → embed_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, third_person: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            third_person: (B, 3, H, W)
            wrist:        (B, 3, H, W)
        Returns:
            embedding:    (B, embed_dim)
        """
        f_tp = self.backbone(third_person)   # (B, feature_dim)
        f_wr = self.backbone(wrist)          # (B, feature_dim)
        return self.proj(torch.cat([f_tp, f_wr], dim=-1))


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, embed_dim)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Takes a trajectory (sequence of strided frames) and outputs K reward scalars.

    Forward input:
        third_person: (B, T, 3, H, W)
        wrist:        (B, T, 3, H, W)

    Forward output:
        rewards: (B, K)  — one scalar per preference dimension
    """

    def __init__(
        self,
        num_preferences: int = 5,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        backbone: str = "resnet18",
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.frame_encoder = FrameEncoder(embed_dim=embed_dim, backbone=backbone)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim=embed_dim, dropout=dropout)

        # CLS token (one per preference dimension)
        self.cls_tokens = nn.Parameter(torch.randn(1, num_preferences, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate reward head per preference
        self.reward_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_preferences)
        ])

    def encode_trajectory(self, third_person: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """
        Encode a full trajectory into K reward scalars.

        Args:
            third_person: (B, T, 3, H, W)
            wrist:        (B, T, 3, H, W)
        Returns:
            rewards: (B, K)
        """
        B, T = third_person.shape[:2]

        # Encode each frame: (B*T, embed_dim)
        tp_flat = third_person.flatten(0, 1)  # (B*T, 3, H, W)
        wr_flat = wrist.flatten(0, 1)
        frame_embs = self.frame_encoder(tp_flat, wr_flat)  # (B*T, embed_dim)
        frame_embs = frame_embs.view(B, T, -1)              # (B, T, embed_dim)

        # Add positional encoding
        frame_embs = self.pos_enc(frame_embs)  # (B, T, embed_dim)

        # Prepend K CLS tokens
        K = len(self.reward_heads)
        cls = self.cls_tokens.expand(B, -1, -1)  # (B, K, embed_dim)
        seq = torch.cat([cls, frame_embs], dim=1)  # (B, K+T, embed_dim)

        # Transformer
        out = self.transformer(seq)  # (B, K+T, embed_dim)

        # Extract CLS token outputs → rewards in [0, 1]
        cls_out = out[:, :K]  # (B, K, embed_dim)
        rewards = torch.sigmoid(torch.stack(
            [head(cls_out[:, i]).squeeze(-1) for i, head in enumerate(self.reward_heads)],
            dim=-1,
        ))  # (B, K)
        return rewards

    def forward(
        self,
        third_person_a: torch.Tensor,
        wrist_a: torch.Tensor,
        third_person_b: torch.Tensor,
        wrist_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            rewards_a: (B, K)
            rewards_b: (B, K)
        """
        r_a = self.encode_trajectory(third_person_a, wrist_a)
        r_b = self.encode_trajectory(third_person_b, wrist_b)
        return r_a, r_b


# ---------------------------------------------------------------------------
# Bradley-Terry loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(
    rewards_a: torch.Tensor,
    rewards_b: torch.Tensor,
    labels: torch.Tensor,
    equal_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pairwise Bradley-Terry loss per preference dimension.

    Args:
        rewards_a:    (B, K) — predicted reward for trajectory A
        rewards_b:    (B, K) — predicted reward for trajectory B
        labels:       (B, K) — 1.0=A preferred, 0.0=B preferred, 0.5=Equal
        equal_weight: weight for Equal samples (0 = skip them)

    Returns:
        loss:         scalar
        per_dim_acc:  (K,) accuracy per preference dimension
    """
    # P(A > B) = r_A / (r_A + r_B)  — rewards are in [0, 1] via sigmoid
    prob_a = rewards_a / (rewards_a + rewards_b + 1e-8)  # (B, K)

    # Build per-sample mask
    is_a = (labels == 1.0)
    is_b = (labels == 0.0)
    is_eq = (labels == 0.5)

    loss = torch.zeros(1, device=rewards_a.device)
    n = 0

    if is_a.any():
        loss = loss + F.binary_cross_entropy(
            prob_a[is_a], torch.ones_like(prob_a[is_a]), reduction="sum"
        )
        n += is_a.sum()

    if is_b.any():
        loss = loss + F.binary_cross_entropy(
            prob_a[is_b], torch.zeros_like(prob_a[is_b]), reduction="sum"
        )
        n += is_b.sum()

    if is_eq.any() and equal_weight > 0:
        loss = loss + equal_weight * F.binary_cross_entropy(
            prob_a[is_eq], 0.5 * torch.ones_like(prob_a[is_eq]), reduction="sum"
        )
        n += is_eq.sum()

    loss = loss / n.clamp(min=1)

    # Accuracy: correct if P(A > B) > 0.5 matches preference
    with torch.no_grad():
        pred_a_wins = (prob_a > 0.5)  # (B, K)
        correct_a = (pred_a_wins & is_a)
        correct_b = (~pred_a_wins & is_b)
        labeled = (is_a | is_b).float()  # ignore Equal for accuracy
        per_dim_acc = (correct_a | correct_b).float().sum(0) / labeled.sum(0).clamp(min=1)

    return loss, per_dim_acc
