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

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)



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
            net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
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

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert uint8 or float [0,1] images to ImageNet-normalized float32."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        mean = _IMAGENET_MEAN.to(x.device)
        std  = _IMAGENET_STD.to(x.device)
        return (x - mean) / std

    def forward(self, third_person: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            third_person: (B, 3, H, W) uint8 or float32
            wrist:        (B, 3, H, W) uint8 or float32
        Returns:
            embedding:    (B, embed_dim)
        """
        f_tp = self.backbone(self._normalize(third_person))   # (B, feature_dim)
        f_wr = self.backbone(self._normalize(wrist))          # (B, feature_dim)
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
        reward_sigmoid: bool = False,
        frozen_backbone: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.reward_sigmoid = reward_sigmoid

        self.frame_encoder = FrameEncoder(embed_dim=embed_dim, backbone=backbone, frozen_backbone=frozen_backbone)
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

    def forward(self, third_person: torch.Tensor, wrist: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode a full trajectory into K reward scalars.

        Args:
            third_person:  (B, T, 3, H, W)
            wrist:         (B, T, 3, H, W)
            padding_mask:  (B, T) bool — True = padded frame (ignored by attention)
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

        # Build src_key_padding_mask: CLS tokens are never masked
        key_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros(B, K, dtype=torch.bool, device=padding_mask.device)
            key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, K+T)

        # Transformer
        out = self.transformer(seq, src_key_padding_mask=key_padding_mask)  # (B, K+T, embed_dim)

        # Extract CLS token outputs → reward scalars
        cls_out = out[:, :K]  # (B, K, embed_dim)

        rewards = torch.stack(
            [head(cls_out[:, i]).squeeze(-1) for i, head in enumerate(self.reward_heads)],
            dim=-1,
        )  # (B, K)

        if self.reward_sigmoid:
            rewards = torch.sigmoid(rewards)
        return rewards




# ---------------------------------------------------------------------------
# Discounted reward model
# ---------------------------------------------------------------------------

class DiscountedRewardModel(nn.Module):
    """
    Scores each frame independently and sums over time.

    Architecture:
        1. FrameEncoder: (third_person, wrist) → embedding  (same as RewardModel)
        2. Per-frame reward heads: embedding → K scalars
        3. Trajectory reward: sum_t( r_t )
    """

    def __init__(
        self,
        num_preferences: int = 5,
        embed_dim: int = 256,
        gamma: float = 0.99,
        dropout: float = 0.1,
        backbone: str = "resnet18",
        reward_sigmoid: bool = False,
        frozen_backbone: bool = False,
    ):
        super().__init__()
        self.reward_sigmoid = reward_sigmoid

        self.frame_encoder = FrameEncoder(embed_dim=embed_dim, backbone=backbone, frozen_backbone=frozen_backbone)
        self.dropout = nn.Dropout(p=dropout)

        # Separate reward head per preference (frame-level)
        self.reward_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(num_preferences)
        ])

    def forward(self, third_person: torch.Tensor, wrist: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            third_person:  (B, T, 3, H, W)
            wrist:         (B, T, 3, H, W)
            padding_mask:  (B, T) bool — True = padded frame (excluded from sum)
        Returns:
            rewards: (B, K)  — sum of per-frame rewards
        """
        B, T = third_person.shape[:2]

        # Encode all frames independently
        tp_flat = third_person.flatten(0, 1)   # (B*T, 3, H, W)
        wr_flat = wrist.flatten(0, 1)
        emb = self.frame_encoder(tp_flat, wr_flat)  # (B*T, embed_dim)
        emb = emb.view(B, T, -1)                    # (B, T, embed_dim)

        # Per-frame rewards: (B, T, K)
        frame_rewards = torch.stack(
            [head(emb).squeeze(-1) for head in self.reward_heads],
            dim=-1,
        )  # (B, T, K)

        # Zero out padded frames before sum
        if padding_mask is not None:
            frame_rewards = frame_rewards.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Sum over time: (B, K)
        rewards = frame_rewards.sum(dim=1)  # (B, K)

        if self.reward_sigmoid:
            rewards = torch.sigmoid(rewards)
        return rewards

    def forward_per_frame(self, third_person: torch.Tensor, wrist: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Same as forward but returns per-frame rewards instead of the discounted sum.

        Returns:
            frame_rewards: (B, T, K) — raw per-frame reward for each preference axis
        """
        B, T = third_person.shape[:2]

        tp_flat = third_person.flatten(0, 1)
        wr_flat = wrist.flatten(0, 1)
        emb = self.frame_encoder(tp_flat, wr_flat)
        emb = emb.view(B, T, -1)

        frame_rewards = torch.stack(
            [head(emb).squeeze(-1) for head in self.reward_heads],
            dim=-1,
        )  # (B, T, K)

        if padding_mask is not None:
            frame_rewards = frame_rewards.masked_fill(padding_mask.unsqueeze(-1), float("nan"))

        return frame_rewards


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
    # P(A > B) = sigmoid(r_A - r_B) — standard Bradley-Terry, works for any reward scale
    prob_a = torch.sigmoid(rewards_a - rewards_b)  # (B, K)

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

    if is_eq.any():
        loss = loss + F.binary_cross_entropy(
            prob_a[is_eq], 0.5 * torch.ones_like(prob_a[is_eq]), reduction="sum"
        )
        n += is_eq.sum()

    loss = loss / n.clamp(min=1)

    # Accuracy: correct if P(A > B) > 0.5 matches preference
    with torch.no_grad():
        pred_a_wins = (prob_a > 0.5)  # (B, K)
        correct_a = (pred_a_wins & is_a)
        correct_b = (~pred_a_wins & is_b)
        per_dim_correct = (correct_a | correct_b).float().sum(0)  # (K,)
        per_dim_labeled = (is_a | is_b).float().sum(0)            # (K,)

    return loss, per_dim_correct, per_dim_labeled


def anchor_loss(rewards: torch.Tensor, dim: int, target: float) -> torch.Tensor:
    """
    MSE loss pushing rewards[:, dim] toward a fixed target (0.0=bad, 1.0=good).

    Args:
        rewards: (1, K) reward vector for one anchor trajectory
        dim:     which preference dimension this anchor applies to
        target:  0.0 or 1.0
    Returns:
        scalar loss
    """
    return F.mse_loss(rewards[0, dim], torch.tensor(target, device=rewards.device))



def bradley_terry_loss_regression(
    rewards_a: torch.Tensor,
    rewards_b: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Regression-based preference loss: MSE to hard targets 0/1.

    For equal pairs, the target for each trajectory is the other's current
    prediction clipped to [0, 1] (detached), so equal pairs converge toward
    the same value without a gradient signal pushing either direction.

    Args:
        rewards_a:  (B, K)
        rewards_b:  (B, K)
        labels:     (B, K) — 1.0=A preferred, 0.0=B preferred, 0.5=Equal

    Returns:
        loss:             scalar
        per_dim_correct:  (K,) number of correct predictions
        per_dim_labeled:  (K,) number of labeled (non-equal) pairs
    """
    is_a = labels == 1.0
    is_b = labels == 0.0
    is_eq = labels == 0.5

    targets_a = torch.zeros_like(rewards_a)
    targets_a[is_a] = 1.0
    targets_a[is_b] = -1.0
    targets_a[is_eq] = rewards_b[is_eq].detach().clamp(-1.0, 1.0)

    targets_b = torch.zeros_like(rewards_b)
    targets_b[is_b] = 1.0
    targets_b[is_a] = -1.0
    targets_b[is_eq] = rewards_a[is_eq].detach().clamp(-1.0, 1.0)

    loss = F.mse_loss(rewards_a, targets_a) + F.mse_loss(rewards_b, targets_b)

    with torch.no_grad():
        pred_a_wins = rewards_a > rewards_b
        per_dim_correct = ((pred_a_wins & is_a) | (~pred_a_wins & is_b)).float().sum(0)
        per_dim_labeled = (is_a | is_b).float().sum(0)

    return loss, per_dim_correct, per_dim_labeled
