"""
Visualization utilities for the preference reward model.

Produces animated GIFs showing validation samples with:
  - Side-by-side video of trajectory A and B (third-person + wrist)
  - Raw reward scalars r_A and r_B per preference dimension
  - Derived P(A>B) vs ground-truth labels
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import torch
from flow_model import RewardModel as FlowRewardModel


def _to_img(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) uint8 tensor to a (H, W, 3) uint8 numpy array."""
    return tensor.permute(1, 2, 0).cpu().numpy()


def visualize_validation_batch(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    out_dir: str,
    preference_keys: list,
    max_samples: int = 8,
    step: int = 0,
    fps: int = 10,
):
    """
    For up to `max_samples` validation items, save an animated GIF showing:
      - Top row: trajectory A (third-person | wrist) animated over time
      - Middle row: trajectory B (third-person | wrist) animated over time
      - Bottom: grouped bar chart of raw r_A, r_B, and P(A>B) vs ground truth

    Saves one GIF per sample to `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    K = len(preference_keys)
    n = min(max_samples, len(val_dataset))

    with torch.no_grad():
        for idx in range(n):
            item = val_dataset[idx]
            session = item["session"]
            labels = item["labels"]  # (K,)
            gt_vals = labels.numpy()

            # (T, 3, H, W) uint8 — kept as-is for display; normalized separately for model
            tp_a_frames = item["traj_a"]["third_person"]
            wr_a_frames = item["traj_a"]["wrist"]
            tp_b_frames = item["traj_b"]["third_person"]
            wr_b_frames = item["traj_b"]["wrist"]
            T = tp_a_frames.shape[0]

            def _to_batch(t):
                return t.unsqueeze(0).to(device)

            if isinstance(model, FlowRewardModel):
                obs_a = {k: _to_batch(v) for k, v in item["traj_a"].items()
                         if k in ("third_person", "wrist", "padding_mask", "proprio")}
                obs_b = {k: _to_batch(v) for k, v in item["traj_b"].items()
                         if k in ("third_person", "wrist", "padding_mask", "proprio")}
                r_a = model(obs_a)
                r_b = model(obs_b)
            else:
                mask_a = _to_batch(item["traj_a"]["padding_mask"])
                mask_b = _to_batch(item["traj_b"]["padding_mask"])
                r_a = model(_to_batch(tp_a_frames), _to_batch(wr_a_frames), mask_a)
                r_b = model(_to_batch(tp_b_frames), _to_batch(wr_b_frames), mask_b)
            r_a_np = r_a.squeeze(0).cpu().numpy()   # (K,)
            r_b_np = r_b.squeeze(0).cpu().numpy()   # (K,)
            prob_a = 1.0 / (1.0 + np.exp(-(r_a_np - r_b_np)))  # sigmoid(r_A - r_B)

            # ----------------------------------------------------------------
            # Build figure layout:
            #   row 0: 4 image axes (tp_A | wr_A | tp_B | wr_B)
            #   row 1: bar chart (r_A, r_B, P(A>B), GT)
            # ----------------------------------------------------------------
            fig = plt.figure(figsize=(16, 9))
            fig.suptitle(f"Session: {session} — step {step}", fontsize=10)

            gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

            ax_tp_a = fig.add_subplot(gs[0, 0])
            ax_wr_a = fig.add_subplot(gs[0, 1])
            ax_tp_b = fig.add_subplot(gs[0, 2])
            ax_wr_b = fig.add_subplot(gs[0, 3])
            ax_bar  = fig.add_subplot(gs[1, :])

            for ax, title in zip(
                [ax_tp_a, ax_wr_a, ax_tp_b, ax_wr_b],
                ["A: 3rd person", "A: wrist", "B: 3rd person", "B: wrist"],
            ):
                ax.set_title(title, fontsize=8)
                ax.axis("off")

            # Initialise image objects with frame 0
            im_tp_a = ax_tp_a.imshow(_to_img(tp_a_frames[0]))
            im_wr_a = ax_wr_a.imshow(_to_img(wr_a_frames[0]))
            im_tp_b = ax_tp_b.imshow(_to_img(tp_b_frames[0]))
            im_wr_b = ax_wr_b.imshow(_to_img(wr_b_frames[0]))

            # ---- bar chart (static, drawn once) ----
            x = np.arange(K)
            width = 0.25

            bars_ra = ax_bar.bar(x - width, r_a_np, width, label="r_A", color="steelblue",   alpha=0.85)
            bars_rb = ax_bar.bar(x,          r_b_np, width, label="r_B", color="darkorange",  alpha=0.85)
            bars_pa = ax_bar.bar(x + width,  prob_a, width, label="P(A>B)", color="gray",     alpha=0.6)

            # Color P(A>B) bar green=correct, red=wrong, gray=Equal GT
            pred_label = ["A" if p > 0.5 else ("B" if p < 0.5 else "=") for p in prob_a]
            gt_label   = ["A" if v == 1.0 else ("B" if v == 0.0 else "=") for v in gt_vals]
            for bar, pred, gt in zip(bars_pa, pred_label, gt_label):
                if gt == "=":
                    bar.set_facecolor("gray")
                elif pred == gt:
                    bar.set_facecolor("mediumseagreen")
                else:
                    bar.set_facecolor("firebrick")

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(preference_keys, fontsize=9)
            all_vals = np.concatenate([r_a_np, r_b_np, prob_a])
            ax_bar.set_ylim(min(0, all_vals.min() - 0.1), all_vals.max() + 0.2)
            ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
            ax_bar.set_ylabel("Score / Probability")
            ax_bar.set_title("Rewards and P(A>B)  |  P(A>B) bar: green=correct, red=wrong", fontsize=8)

            legend_patches = [
                mpatches.Patch(color="steelblue",      label="r_A"),
                mpatches.Patch(color="darkorange",     label="r_B"),
                mpatches.Patch(color="mediumseagreen", label="P(A>B) — correct"),
                mpatches.Patch(color="firebrick",      label="P(A>B) — wrong"),
                mpatches.Patch(color="gray",           label="P(A>B) — GT Equal"),
            ]
            ax_bar.legend(handles=legend_patches, fontsize=7, loc="upper right")

            # Value labels + GT/Pred text per dimension
            for bars in [bars_ra, bars_rb, bars_pa]:
                for bar in bars:
                    h = bar.get_height()
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6,
                    )
            for i, (pred, gt) in enumerate(zip(pred_label, gt_label)):
                correct = (pred == gt) if gt != "=" else None
                color = "darkgreen" if correct else ("firebrick" if correct is False else "gray")
                ax_bar.text(
                    x[i], 1.22,
                    f"GT: {gt}  Pred: {pred}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=color,
                )

            # ---- animation ----
            frame_label = ax_tp_a.text(
                0.5, -0.08, f"t=0/{T-1}", transform=ax_tp_a.transAxes,
                ha="center", fontsize=7,
            )

            def update(t):
                im_tp_a.set_data(_to_img(tp_a_frames[t]))
                im_wr_a.set_data(_to_img(wr_a_frames[t]))
                im_tp_b.set_data(_to_img(tp_b_frames[t]))
                im_wr_b.set_data(_to_img(wr_b_frames[t]))
                frame_label.set_text(f"t={t}/{T-1}")
                return im_tp_a, im_wr_a, im_tp_b, im_wr_b, frame_label

            anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=True)

            fname = os.path.join(out_dir, f"step{step:06d}_val{idx:02d}_{session}.mp4")
            anim.save(fname, writer=FFMpegWriter(fps=fps))
            plt.close(fig)

    model.train()
    return n


def visualize_top_bottom_trajectories(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    out_dir: str,
    preference_keys: list,
    n: int = 5,
    n_uniform: int = 10,
    step: int = 0,
    fps: int = 10,
):
    """
    For each preference dimension, collect rewards for every individual trajectory
    in the dataset, then save:
      - one video showing top-n vs bottom-n trajectories
      - one video showing n_uniform trajectories uniformly sampled across the reward spectrum
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    K = len(preference_keys)

    # Collect (tp_frames, reward_vector) for every individual trajectory
    entries = []  # list of {"frames": (T,H,W,3) uint8 numpy, "reward": (K,) float}
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            for traj_key in ("traj_a", "traj_b"):
                tp_frames = item[traj_key]["third_person"]   # (T, 3, H, W) uint8
                if isinstance(model, FlowRewardModel):
                    obs = {k: v.unsqueeze(0).to(device) for k, v in item[traj_key].items()
                           if k in ("third_person", "wrist", "padding_mask", "proprio")}
                    r = model(obs).squeeze(0).cpu().numpy()  # (K,)
                else:
                    wr_frames = item[traj_key]["wrist"]
                    mask = item[traj_key]["padding_mask"].unsqueeze(0).to(device)
                    r = model(tp_frames.unsqueeze(0).to(device),
                              wr_frames.unsqueeze(0).to(device), mask).squeeze(0).cpu().numpy()  # (K,)
                entries.append({
                    "frames": tp_frames.permute(0, 2, 3, 1).numpy(),  # (T, H, W, 3)
                    "reward": r,
                })

    rewards_all = np.stack([e["reward"] for e in entries])  # (N, K)
    N = len(entries)

    def _save_video(row_groups, row_labels, row_colors, title, fname):
        """Save a multi-row animation where each row_group is a list of entries."""
        n_rows = len(row_groups)
        n_cols = max(len(g) for g in row_groups)
        T_max = max(e["frames"].shape[0] for g in row_groups for e in g)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows + 0.8),
                                 squeeze=False)
        fig.suptitle(title, fontsize=10)

        for r, (group, label, color) in enumerate(zip(row_groups, row_labels, row_colors)):
            axes[r, 0].set_ylabel(label, fontsize=8)
            for c in range(n_cols):
                axes[r, c].axis("off")
                if c < len(group):
                    axes[r, c].set_title(f"r={group[c]['reward'][k]:.2f}", fontsize=7, color=color)

        ims = [[axes[r, c].imshow(row_groups[r][c]["frames"][0])
                if c < len(row_groups[r]) else None
                for c in range(n_cols)]
               for r in range(n_rows)]

        def update(t):
            artists = []
            for r, group in enumerate(row_groups):
                for c, entry in enumerate(group):
                    ft = min(t, entry["frames"].shape[0] - 1)
                    ims[r][c].set_data(entry["frames"][ft])
                    artists.append(ims[r][c])
            return artists

        anim = FuncAnimation(fig, update, frames=T_max, interval=1000 // fps, blit=True)
        anim.save(fname, writer=FFMpegWriter(fps=fps))
        plt.close(fig)

    top_bottom_paths = []
    uniform_paths = []

    for k, key in enumerate(preference_keys):
        sorted_idx = np.argsort(rewards_all[:, k])
        safe_key = key.replace("/", "_").replace(" ", "_")

        # --- top / bottom ---
        bottom_entries = [entries[i] for i in sorted_idx[:n]]
        top_entries    = [entries[i] for i in sorted_idx[-n:][::-1]]
        fname_tb = os.path.join(out_dir, f"step{step:06d}_top_bottom_{safe_key}.mp4")
        _save_video(
            row_groups=[top_entries, bottom_entries],
            row_labels=["High reward", "Low reward"],
            row_colors=["darkgreen", "firebrick"],
            title=f"[step {step}]  {key}  —  top {n} vs bottom {n}",
            fname=fname_tb,
        )
        top_bottom_paths.append(fname_tb)

        # --- uniform spectrum ---
        uniform_pick = np.round(np.linspace(0, N - 1, min(n_uniform, N))).astype(int)
        uniform_entries = [entries[sorted_idx[i]] for i in uniform_pick]
        fname_uni = os.path.join(out_dir, f"step{step:06d}_uniform_{safe_key}.mp4")
        _save_video(
            row_groups=[uniform_entries],
            row_labels=["Uniform"],
            row_colors=["steelblue"],
            title=f"[step {step}]  {key}  —  {n_uniform} uniform samples across reward spectrum",
            fname=fname_uni,
        )
        uniform_paths.append(fname_uni)

    model.train()
    return top_bottom_paths, uniform_paths


def plot_reward_correlation(rewards: np.ndarray, preference_keys: list) -> "plt.Figure":
    """
    Plot a correlation matrix of reward dimensions from a (N, K) array.

    Values are standardized (z-scored) before computing Pearson correlation.
    Returns a matplotlib Figure (caller is responsible for closing it).
    """
    K = len(preference_keys)
    # Standardize per dimension across the N trajectories in the buffer.
    std = rewards.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z = (rewards - rewards.mean(axis=0, keepdims=True)) / std  # (N, K)
    corr = np.corrcoef(z.T)  # (K, K)
    if corr.ndim == 0:
        corr = corr.reshape(1, 1)

    fig, ax = plt.subplots(figsize=(max(4, K * 1.1), max(3.5, K * 1.0)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(preference_keys, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(preference_keys, fontsize=8)
    ax.set_title(f"Reward correlation  (N={len(rewards)})", fontsize=9)

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr[i, j]) < 0.6 else "white")

    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: list,
    train_accs: list,
    val_losses: list,
    val_accs: list,
    out_path: str,
    preference_keys: list = None,
):
    """
    Plot loss curves and per-dimension train/val accuracy curves.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    steps = np.arange(len(train_losses))
    keys = preference_keys or []

    # ---- Loss ----
    axes[0].plot(steps, train_losses, label="Train", alpha=0.8)
    if val_losses:
        val_x = np.linspace(0, len(train_losses) - 1, len(val_losses))
        axes[0].plot(val_x, val_losses, label="Val", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("BT Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # ---- Train accuracy per dimension ----
    if train_accs:
        train_acc_arr = np.array(train_accs)  # (steps, K)
        K = train_acc_arr.shape[1]
        dim_keys = keys or [f"dim_{k}" for k in range(K)]
        # Smooth with a running mean for readability
        window = max(1, len(train_accs) // 20)
        for k, key in enumerate(dim_keys):
            raw = train_acc_arr[:, k]
            smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
            smooth_x = np.linspace(0, len(train_losses) - 1, len(smoothed))
            axes[1].plot(smooth_x, smoothed, label=key, alpha=0.85)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Train Accuracy per Dimension (smoothed)")
        axes[1].set_ylim(0, 1)
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        axes[1].legend(fontsize=7)

    # ---- Val accuracy per dimension ----
    if val_accs:
        val_acc_arr = np.array(val_accs)  # (num_evals, K)
        K = val_acc_arr.shape[1]
        dim_keys = keys or [f"dim_{k}" for k in range(K)]
        val_x = np.linspace(0, len(train_losses) - 1, len(val_accs))
        for k, key in enumerate(dim_keys):
            axes[2].plot(val_x, val_acc_arr[:, k], label=key, alpha=0.85)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Val Accuracy per Dimension")
        axes[2].set_ylim(0, 1)
        axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
