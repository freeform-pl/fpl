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

            # (T, 3, H, W) uint8
            tp_a_frames = item["traj_a"]["third_person"]
            wr_a_frames = item["traj_a"]["wrist"]
            tp_b_frames = item["traj_b"]["third_person"]
            wr_b_frames = item["traj_b"]["wrist"]
            T = tp_a_frames.shape[0]

            # Model forward
            tp_a = tp_a_frames.unsqueeze(0).to(device).float() / 255.0
            wr_a = wr_a_frames.unsqueeze(0).to(device).float() / 255.0
            tp_b = tp_b_frames.unsqueeze(0).to(device).float() / 255.0
            wr_b = wr_b_frames.unsqueeze(0).to(device).float() / 255.0

            r_a, r_b = model(tp_a, wr_a, tp_b, wr_b)
            r_a_np = r_a.squeeze(0).cpu().numpy()   # (K,)
            r_b_np = r_b.squeeze(0).cpu().numpy()   # (K,)
            prob_a = r_a_np / (r_a_np + r_b_np + 1e-8)  # (K,)

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
            width = 0.2

            bars_ra  = ax_bar.bar(x - 1.5 * width, r_a_np,  width, label="r_A",      color="steelblue",  alpha=0.85)
            bars_rb  = ax_bar.bar(x - 0.5 * width, r_b_np,  width, label="r_B",      color="darkorange",  alpha=0.85)
            bars_pa  = ax_bar.bar(x + 0.5 * width, prob_a,  width, label="P(A>B)",   color="mediumseagreen", alpha=0.85)
            bars_gt  = ax_bar.bar(x + 1.5 * width, gt_vals, width, label="GT",       color="gray",        alpha=0.6)

            # Color GT bars by label meaning
            for bar, val in zip(bars_gt, gt_vals):
                bar.set_facecolor("steelblue" if val == 1.0 else "firebrick" if val == 0.0 else "gray")

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(preference_keys, fontsize=9)
            ax_bar.set_ylim(0, 1.15)
            ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
            ax_bar.set_ylabel("Score / Probability")
            ax_bar.set_title(
                "Raw rewards and preference probabilities  "
                "(GT: blue=A preferred, red=B preferred, gray=Equal)",
                fontsize=8,
            )

            legend_patches = [
                mpatches.Patch(color="steelblue",     label="r_A  (raw reward A)"),
                mpatches.Patch(color="darkorange",    label="r_B  (raw reward B)"),
                mpatches.Patch(color="mediumseagreen",label="P(A>B) = r_A/(r_A+r_B)"),
                mpatches.Patch(color="gray",          label="Ground truth"),
            ]
            ax_bar.legend(handles=legend_patches, fontsize=7, loc="upper right")

            # Add value labels on bars
            for bars in [bars_ra, bars_rb, bars_pa]:
                for bar in bars:
                    h = bar.get_height()
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6,
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
