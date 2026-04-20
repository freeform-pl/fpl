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

from model import FrameEncoder, SinusoidalPositionalEncoding


# ---------------------------------------------------------------------------
# Velocity network (flow matching)
# ---------------------------------------------------------------------------

class VelocityNetwork(nn.Module):
    """
    Predicts the flow velocity v(x_t, t | c) for K reward dimensions in parallel.

    Each reward dimension has its own conditioning vector c_k (from the CLS tokens).
    All K dims are processed together by reshaping to (B*K, ...) and using a shared MLP.

    Args:
        embed_dim:  size of the per-dim conditioning vector c_k
        num_preferences: K
        hidden_dim: MLP hidden size
        t_dim:      time-embedding dimension
    """

    def __init__(self, embed_dim: int, num_preferences: int, hidden_dim: int = 256, t_dim: int = 64):
        super().__init__()
        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1 + t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, K)            current reward positions
            t:   (B, 1)            time in [0, 1]
            c:   (B, K, embed_dim) per-dim trajectory conditioning
        Returns:
            velocity: (B, K)
        """
        B, K = x_t.shape
        t_emb = self.t_encoder(t).unsqueeze(1).expand(-1, K, -1)  # (B, K, t_dim)
        inp = torch.cat([c, x_t.unsqueeze(-1), t_emb], dim=-1)    # (B, K, embed_dim+1+t_dim)
        return self.net(inp.reshape(B * K, -1)).reshape(B, K)


# ---------------------------------------------------------------------------
# Action network (flow matching for action chunk prediction)
# ---------------------------------------------------------------------------

class ActionNet(nn.Module):
    """
    Predicts flow velocity for action chunks conditioned on frame tokens.

    Processes all B*T frame tokens in one batched forward pass.

    Args:
        embed_dim:         size of the per-frame conditioning vector
        action_chunk_size: number of action steps per chunk (K_a)
        action_dim:        dimensionality of each action step (default 8: 7 joint + 1 gripper)
        hidden_dim:        MLP hidden size
        t_dim:             time-embedding dimension
    """

    def __init__(self, embed_dim: int, action_chunk_size: int, action_dim: int = 8,
                 hidden_dim: int = 256, t_dim: int = 64):
        super().__init__()
        self.action_out_dim = action_chunk_size * action_dim
        self.t_encoder = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim + self.action_out_dim + t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.action_out_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (N, action_out_dim)  noisy action chunk  (N = B*T)
            t:   (N, 1)               time in [0, 1]
            c:   (N, embed_dim)       frame token conditioning
        Returns:
            velocity: (N, action_out_dim)
        """
        t_emb = self.t_encoder(t)           # (N, t_dim)
        inp = torch.cat([c, x_t, t_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Reward model (flow matching)
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Flow-matching reward model.

    The transformer encoder produces K per-preference conditioning vectors c (one
    per CLS token). A VelocityNetwork then learns to transport samples from
    x_0 ~ N(0,1) to x_1 = target reward via a straight-line conditional flow.

    Training:
        loss = flow_matching_loss(cls_out_a, cls_out_b, labels)

    Inference:
        rewards = model(third_person, wrist, padding_mask)
                = encode → sample_reward (average of n_samples ODE integrations)
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
        frozen_backbone: bool = False,
        vel_hidden_dim: int = 256,
        vel_t_dim: int = 64,
        n_sample_steps: int = 10,
        n_samples: int = 10,
        ptp: bool = False,
        action_chunk_size: int = 16,
        action_dim: int = 8,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_preferences = num_preferences
        self.n_sample_steps = n_sample_steps
        self.n_samples = n_samples
        self.ptp = ptp

        self.frame_encoder = FrameEncoder(embed_dim=embed_dim, backbone=backbone, frozen_backbone=frozen_backbone)
        self.proprio_proj = nn.Linear(7, embed_dim)
        self.frame_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim=embed_dim, dropout=dropout)

        self.cls_tokens = nn.Parameter(torch.randn(1, num_preferences, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.velocity_net = VelocityNetwork(embed_dim, num_preferences, vel_hidden_dim, vel_t_dim)

        if ptp:
            self.action_net = ActionNet(embed_dim, action_chunk_size, action_dim, vel_hidden_dim, vel_t_dim)

    @staticmethod
    def _make_ptp_mask(T: int, K: int, device: torch.device) -> torch.Tensor:
        """
        Causal attention mask for PTP mode.

        Sequence layout: [frame_0, ..., frame_{T-1}, cls_0, ..., cls_{K-1}]

        - Frame token i attends only to frames 0..i (causal); masked from future
          frames and from all CLS tokens.
        - CLS tokens attend to everything (bidirectional).

        Returns (T+K, T+K) bool tensor where True = masked out.
        """
        S = T + K
        mask = torch.zeros(S, S, dtype=torch.bool, device=device)
        # Frame queries: mask future frame tokens (upper triangle of T×T block)
        mask[:T, :T] = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        # Frame queries: mask all CLS tokens (they come after the frames)
        mask[:T, T:] = True
        # CLS queries: no masking (already zero)
        return mask

    def encode(self, obs: dict) -> tuple:
        """
        Encode a trajectory into K conditioning vectors and per-frame tokens.

        Args:
            obs: dict with keys:
                "third_person": (B, T, 3, H, W) float32 normalized
                "wrist":        (B, T, 3, H, W) float32 normalized
                "padding_mask": (B, T) bool, True = padded  [optional]
                "proprio":      (B, T, 7) joint positions   [optional]

        Returns:
            cls_out:      (B, K, embed_dim)
            frame_tokens: (B, T, embed_dim)
        """
        third_person = obs["third_person"]
        wrist        = obs["wrist"]
        padding_mask = obs.get("padding_mask")
        proprio      = obs.get("proprio")

        B, T = third_person.shape[:2]
        K = self.num_preferences

        tp_flat = third_person.flatten(0, 1)
        wr_flat = wrist.flatten(0, 1)
        img_embs = self.frame_encoder(tp_flat, wr_flat).view(B, T, -1)  # (B, T, embed_dim)
        if proprio is None:
            import warnings
            warnings.warn("proprio not provided; using zeros.", stacklevel=2)
            proprio_embs = torch.zeros_like(img_embs)
        else:
            proprio_embs = self.proprio_proj(proprio)                    # (B, T, embed_dim)
        frame_embs = self.frame_proj(torch.cat([img_embs, proprio_embs], dim=-1))  # (B, T, embed_dim)
        frame_embs = self.pos_enc(frame_embs)

        cls = self.cls_tokens.expand(B, -1, -1)
        seq = torch.cat([frame_embs, cls], dim=1)

        key_padding_mask = None
        if padding_mask is not None:
            cls_mask = torch.zeros(B, K, dtype=torch.bool, device=padding_mask.device)
            key_padding_mask = torch.cat([padding_mask, cls_mask], dim=1)

        src_mask = self._make_ptp_mask(T, K, seq.device) if self.ptp else None

        out = self.transformer(seq, mask=src_mask, src_key_padding_mask=key_padding_mask)
        cls_out = out[:, -K:]      # (B, K, embed_dim)
        frame_tokens = out[:, :T]  # (B, T, embed_dim)
        return cls_out, frame_tokens

    def _fm_loss_single(self, cls_out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Flow matching loss for one batch of trajectories.

        Args:
            cls_out: (B, K, embed_dim)  trajectory conditioning
            targets: (B, K)             x_1 targets
        Returns:
            scalar loss
        """
        B, K = targets.shape
        x_0 = torch.randn(B, K, device=targets.device)
        t   = torch.rand(B, 1,  device=targets.device)

        x_t = (1 - t) * x_0 + t * targets
        u_t = targets - x_0

        v = self.velocity_net(x_t, t, cls_out)
        return F.mse_loss(v, u_t)

    def bradley_terry_flow_matching_loss(
        self,
        cls_out_a: torch.Tensor,
        cls_out_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Compute flow matching loss for a preference pair.

        Targets:
            A preferred  → target_A = 1,   target_B = 0
            B preferred  → target_A = 0,   target_B = 1
            Equal        → pick a random target either A or B

        Returns:
            loss, per_dim_correct (K,), per_dim_labeled (K,)
        """
        is_a  = (labels == 1.0)
        is_b  = (labels == 0.0)
        is_eq = (labels == 0.5)

        targets_a = torch.where(is_a, torch.ones_like(labels), torch.zeros_like(labels))
        targets_b = torch.where(is_b, torch.ones_like(labels), torch.zeros_like(labels))

        if is_eq.any():
            # For each equal (b, k): randomly pick A or B, sample one point from
            # that trajectory, and use it as the shared target for both.
            with torch.no_grad():
                pick_a = torch.rand_like(labels) < 0.5
                r_a_sample = self.sample_point(cls_out_a, n_steps=4)  # (B, K)
                r_b_sample = self.sample_point(cls_out_b, n_steps=4)  # (B, K)
                eq_target = torch.where(pick_a, r_a_sample, r_b_sample)

            targets_a = torch.where(is_eq, eq_target, targets_a)
            targets_b = torch.where(is_eq, eq_target, targets_b)

        loss = (self._fm_loss_single(cls_out_a, targets_a) +
                self._fm_loss_single(cls_out_b, targets_b)) / 2

        with torch.no_grad():
            r_a = self.sample_reward(cls_out_a)
            r_b = self.sample_reward(cls_out_b)
            pred_a_wins = r_a > r_b
            per_dim_correct = ((pred_a_wins & is_a) | (~pred_a_wins & is_b)).float().sum(0)
            per_dim_labeled = (is_a | is_b).float().sum(0)

        return loss, per_dim_correct, per_dim_labeled

    def sample_point(self, cls_out: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """
        Integrate the learned ODE from t=0 to t=1 (Euler) starting from x_0 ~ N(0,1).

        Args:
            cls_out: (B, K, embed_dim)
        Returns:
            x_1: (B, K)  one reward sample
        """
        if n_steps is None:
            n_steps = self.n_sample_steps
        B, K, _ = cls_out.shape
        x = torch.randn(B, K, device=cls_out.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=cls_out.device)
            x = x + dt * self.velocity_net(x, t, cls_out)
        return x

    def sample_reward(self, cls_out: torch.Tensor, n_samples: int = None, n_steps: int = None) -> torch.Tensor:
        """
        Average reward over multiple independent ODE samples to reduce variance.

        Args:
            cls_out: (B, K, embed_dim)
        Returns:
            reward: (B, K)
        """
        if n_samples is None:
            n_samples = self.n_samples
        samples = torch.stack([self.sample_point(cls_out, n_steps) for _ in range(n_samples)])
        return samples.mean(0)

    def _action_fm_loss_single(
        self,
        frame_tokens: torch.Tensor,
        action_chunks: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow matching loss for one batch of trajectories.

        Args:
            frame_tokens:  (B, T, embed_dim)
            action_chunks: (B, T, chunk_K, action_dim)
            padding_mask:  (B, T) bool — True = padded (excluded from loss)
        Returns:
            scalar loss
        """
        B, T, chunk_K, action_dim = action_chunks.shape
        action_out_dim = chunk_K * action_dim
        targets = action_chunks.reshape(B, T, action_out_dim)  # (B, T, action_out_dim)

        t = torch.rand(B, 1, 1, device=targets.device)  # broadcast over T and action_out_dim
        x_0 = torch.randn(B, T, action_out_dim, device=targets.device)
        x_t = (1 - t) * x_0 + t * targets
        u_t = targets - x_0

        N = B * T
        x_t_flat = x_t.reshape(N, action_out_dim)
        t_flat = t.expand(B, T, 1).reshape(N, 1)
        c_flat = frame_tokens.reshape(N, -1)
        u_t_flat = u_t.reshape(N, action_out_dim)

        v = self.action_net(x_t_flat, t_flat, c_flat)  # (N, action_out_dim)

        if padding_mask is not None:
            valid = ~padding_mask.reshape(N)  # (N,) True = valid
            if valid.any():
                return F.mse_loss(v[valid], u_t_flat[valid])
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
        return F.mse_loss(v, u_t_flat)

    def action_flow_loss(
        self,
        frame_tokens_a: torch.Tensor,
        action_chunks_a: torch.Tensor,
        padding_mask_a: torch.Tensor,
        frame_tokens_b: torch.Tensor,
        action_chunks_b: torch.Tensor,
        padding_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow matching loss for action chunk prediction (PTP), averaged over A and B.

        Args:
            frame_tokens_a:  (B, T, embed_dim)
            action_chunks_a: (B, T, chunk_K, action_dim)
            padding_mask_a:  (B, T) bool
        Returns:
            scalar loss
        """
        loss_a = self._action_fm_loss_single(frame_tokens_a, action_chunks_a, padding_mask_a)
        loss_b = self._action_fm_loss_single(frame_tokens_b, action_chunks_b, padding_mask_b)
        return (loss_a + loss_b) / 2

    def forward(self, obs: dict) -> torch.Tensor:
        """Encode trajectory and return sampled reward (B, K)."""
        cls_out, _ = self.encode(obs)
        return self.sample_reward(cls_out)


