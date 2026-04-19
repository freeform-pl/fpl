"""
Training script for the preference reward model.

Usage:
    python main.py [--stride 4] [--seq_len 28] [--img_size 128] \
                   [--embed_dim 256] [--num_heads 8] [--num_layers 4] \
                   [--lr 1e-4] [--batch_size 8] [--epochs 50] \
                   [--val_fraction 0.2] [--seed 0] \
                   [--preferences_dir preferences] [--out_dir exp/]
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms.v2 as T
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_datasets, print_dataset_stats
from model import RewardModel, DiscountedRewardModel, bradley_terry_loss
from tasks import TASKS
from visualization import visualize_validation_batch, visualize_top_bottom_trajectories, plot_training_curves


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--task", default="cube_in_three_bowls", choices=list(TASKS.keys()),
                   help="Task name — determines which preference dimensions to use")
    p.add_argument("--preferences_dir",type=str, default="preferences")
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--stride", type=int, default=4,
                   help="Frame stride when sampling sequences from each trajectory")
    p.add_argument("--seq_len", type=int, default=28,
                   help="Number of frames per trajectory after striding")
    p.add_argument("--img_size", type=int, default=128,
                   help="Resize each frame to (img_size x img_size) when loading from video fallback")
    p.add_argument("--preload", action="store_true",
                   help="Preload all data into RAM at startup")

    # Model
    p.add_argument("--model", default="transformer", choices=["transformer", "discounted"],
                   help="Model architecture: transformer (default) or discounted (per-frame + gamma sum)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze the ResNet backbone; only train the projection and reward heads")
    p.add_argument("--backbone_lr_scale", type=float, default=1.0,
                   help="LR multiplier for backbone params (e.g. 0.1 for 10x smaller backbone lr)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor for --model discounted")
    p.add_argument("--reward_sigmoid", action="store_true",
                   help="Apply sigmoid to reward outputs (maps to [0,1]); default is unbounded rewards")

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--equal_weight", type=float, default=0.0,
                   help="Loss weight for Equal-labeled samples (0 = skip)")

    # Logging / saving
    p.add_argument("--out_dir", default="exp/")
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N steps")
    p.add_argument("--vis_interval", type=int, default=100, help="Visualize every N steps")
    p.add_argument("--save_interval", type=int, default=200, help="Save checkpoint every N steps")
    p.add_argument("--max_vis_samples", type=int, default=8)

    # Wandb
    p.add_argument("--wandb_project", default="reward_learning")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 images (B, T, 3, H, W) to ImageNet-normalized float32."""
    x = x.float() / 255.0
    mean = _imagenet_mean.to(x.device)
    std  = _imagenet_std.to(x.device)
    return (x - mean) / std


_augment = T.Compose([
    T.RandomResizedCrop(size=128, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
])


def augment_traj(x: torch.Tensor) -> torch.Tensor:
    """Apply consistent augmentation across all frames of each trajectory.

    x: (B, T, 3, H, W) float32 in [0, 1]
    Returns: (B, T, 3, H, W)
    """
    B, T, C, H, W = x.shape
    # Apply per-trajectory: same random params for all T frames of a given trajectory
    out = torch.stack([_augment(x[b]) for b in range(B)])  # each call samples new params
    return out


@torch.no_grad()
def evaluate(model, val_loader, device, equal_weight, num_preferences):
    model.eval()
    total_loss = 0.0
    total_correct = torch.zeros(num_preferences)
    total_labeled = torch.zeros(num_preferences)
    n_batches = 0

    for batch in val_loader:
        tp_a = normalize(batch["traj_a"]["third_person"].to(device))
        wr_a = normalize(batch["traj_a"]["wrist"].to(device))
        tp_b = normalize(batch["traj_b"]["third_person"].to(device))
        wr_b = normalize(batch["traj_b"]["wrist"].to(device))
        labels = batch["labels"].to(device)
        mask_a = batch["traj_a"]["padding_mask"].to(device)
        mask_b = batch["traj_b"]["padding_mask"].to(device)

        r_a = model(tp_a, wr_a, mask_a)
        r_b = model(tp_b, wr_b, mask_b)
        loss, per_dim_correct, per_dim_labeled = bradley_terry_loss(r_a, r_b, labels, equal_weight)

        total_loss += loss.item()
        total_correct += per_dim_correct.cpu()
        total_labeled += per_dim_labeled.cpu()
        n_batches += 1

    model.train()
    acc = (total_correct / total_labeled.clamp(min=1)).numpy()
    return total_loss / max(n_batches, 1), acc


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "visualizations")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Wandb
    # ------------------------------------------------------------------ #
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    preference_keys = TASKS[args.task]
    preference_dirs = args.preferences_dir.split(",")
    
    train_ds, val_ds = make_datasets(
        task=args.task,
        preferences_dir=preference_dirs,
        val_fraction=args.val_fraction,
        stride=args.stride,
        seq_len=args.seq_len,
        img_size=(args.img_size, args.img_size),
        seed=args.seed,
        preload=args.preload,
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    print_dataset_stats(train_ds, val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    if args.model == "discounted":
        model = DiscountedRewardModel(
            num_preferences=len(preference_keys),
            embed_dim=args.embed_dim,
            gamma=args.gamma,
            dropout=args.dropout,
            backbone=args.backbone,
            reward_sigmoid=args.reward_sigmoid,
            frozen_backbone=args.freeze_backbone,
        ).to(device)
    else:
        model = RewardModel(
            num_preferences=len(preference_keys),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            backbone=args.backbone,
            reward_sigmoid=args.reward_sigmoid,
            frozen_backbone=args.freeze_backbone,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Separate backbone params for optional lower lr
    backbone_params = list(model.frame_encoder.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    param_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    global_step = 0

    # ---- pre-training baseline ----
    val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys))
    val_losses.append(val_loss)
    val_accs.append(val_acc.tolist())
    acc_str = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc))
    print(f"[Pre-train] Loss {val_loss:.4f} | {acc_str}")

    epoch_pbar = tqdm(range(args.epochs), desc="Epochs")
    for epoch in epoch_pbar:
        t_data_end = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            t_data = time.perf_counter() - t_data_end

            t0 = time.perf_counter()
            tp_a = augment_traj(normalize(batch["traj_a"]["third_person"].to(device)))
            wr_a = augment_traj(normalize(batch["traj_a"]["wrist"].to(device)))
            tp_b = augment_traj(normalize(batch["traj_b"]["third_person"].to(device)))
            wr_b = augment_traj(normalize(batch["traj_b"]["wrist"].to(device)))
            labels = batch["labels"].to(device)
            mask_a = batch["traj_a"]["padding_mask"].to(device)
            mask_b = batch["traj_b"]["padding_mask"].to(device)
            t_transfer = time.perf_counter() - t0

            t0 = time.perf_counter()
            optimizer.zero_grad()
            r_a = model(tp_a, wr_a, mask_a)
            r_b = model(tp_b, wr_b, mask_b)
            t_forward = time.perf_counter() - t0

            t0 = time.perf_counter()
            loss, per_dim_correct, per_dim_labeled = bradley_terry_loss(r_a, r_b, labels, args.equal_weight)
            per_dim_acc = per_dim_correct / per_dim_labeled.clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_backward = time.perf_counter() - t0

            train_losses.append(loss.item())
            train_accs.append(per_dim_acc.cpu().numpy().tolist())
            global_step += 1
            t_data_end = time.perf_counter()

            # ---- logging ----
            if global_step % args.log_interval == 0:
                mean_acc = per_dim_acc.mean().item()
                epoch_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{mean_acc:.3f}")
                print(
                    f"Epoch {epoch+1:3d} | Step {global_step:5d} | "
                    f"Loss {loss.item():.4f} | Acc {mean_acc:.3f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"data={t_data:.2f}s  transfer={t_transfer:.2f}s  fwd={t_forward:.2f}s  bwd={t_backward:.2f}s"
                )
                if use_wandb:
                    wandb.log({
                        "timing/data_loading": t_data,
                        "timing/gpu_transfer": t_transfer,
                        "timing/forward": t_forward,
                        "timing/backward": t_backward,
                    }, step=global_step)
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/acc_mean": mean_acc,
                        "train/lr": scheduler.get_last_lr()[0],
                        **{f"train/acc_{k}": v for k, v in zip(preference_keys, per_dim_acc.cpu().numpy())},
                    }, step=global_step)

            # ---- validation ----
            if global_step % args.eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys))
                val_losses.append(val_loss)
                val_accs.append(val_acc.tolist())
                mean_val_acc = val_acc.mean()
                epoch_pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{per_dim_acc.mean().item():.3f}",
                    val_loss=f"{val_loss:.4f}", val_acc=f"{mean_val_acc:.3f}",
                )
                acc_str = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc))
                print(f"  [Val] Step {global_step:5d} | Loss {val_loss:.4f} | Mean acc {mean_val_acc:.3f} | {acc_str}")
                plot_training_curves(
                    train_losses, train_accs, val_losses, val_accs,
                    out_path=os.path.join(args.out_dir, "training_curves.png"),
                    preference_keys=preference_keys,
                )

            # ---- visualization + wandb val logging ----
            if global_step % args.vis_interval == 0:
                val_loss_vis, val_acc_vis = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys))
                mean_val_acc_vis = val_acc_vis.mean()
                acc_str_vis = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc_vis))
                print(f"  [Vis/Val] Step {global_step:5d} | Loss {val_loss_vis:.4f} | Mean acc {mean_val_acc_vis:.3f} | {acc_str_vis}")

                vis_step_dir = os.path.join(vis_dir, f"step{global_step:06d}")
                n_vis_val = visualize_validation_batch(
                    model, val_ds, device,
                    out_dir=os.path.join(vis_step_dir, "val"),
                    preference_keys=preference_keys,
                    max_samples=args.max_vis_samples,
                    step=global_step,
                )
                n_vis_train = visualize_validation_batch(
                    model, train_ds, device,
                    out_dir=os.path.join(vis_step_dir, "train"),
                    preference_keys=preference_keys,
                    max_samples=args.max_vis_samples,
                    step=global_step,
                )
                top_bottom_dir = os.path.join(vis_step_dir, "top_bottom")
                top_bottom_mp4s = visualize_top_bottom_trajectories(
                    model, train_ds, device,
                    out_dir=top_bottom_dir,
                    preference_keys=preference_keys,
                    n=5,
                    step=global_step,
                )
                print(f"  [Vis] Saved {n_vis_val} val + {n_vis_train} train + {len(top_bottom_mp4s)} top/bottom figures → {vis_step_dir}")

                if use_wandb:
                    val_mp4s = sorted(f for f in os.listdir(os.path.join(vis_step_dir, "val")) if f.endswith(".mp4"))
                    train_mp4s = sorted(f for f in os.listdir(os.path.join(vis_step_dir, "train")) if f.endswith(".mp4"))
                    wandb.log({
                        "val/loss": val_loss_vis,
                        "val/acc_mean": float(mean_val_acc_vis),
                        **{f"val/acc_{k}": float(v) for k, v in zip(preference_keys, val_acc_vis)},
                        **{f"val/video_{i}": wandb.Video(os.path.join(vis_step_dir, "val", f), format="mp4") for i, f in enumerate(val_mp4s)},
                        **{f"train/video_{i}": wandb.Video(os.path.join(vis_step_dir, "train", f), format="mp4") for i, f in enumerate(train_mp4s)},
                        **{f"top_bottom/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, top_bottom_mp4s)},
                    }, step=global_step)

            # ---- checkpoint ----
            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step{global_step:06d}.pt")
                torch.save({
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  [Ckpt] Saved → {ckpt_path}")

    # ---- final eval + curves ----
    val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys))
    print("\nFinal validation:")
    for k, v in zip(preference_keys, val_acc):
        print(f"  {k}: {v:.3f}")

    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        out_path=os.path.join(args.out_dir, "training_curves.png"),
        preference_keys=preference_keys,
    )

    # Final visualizations
    visualize_validation_batch(
        model, val_ds, device,
        out_dir=os.path.join(vis_dir, "final", "val"),
        preference_keys=preference_keys,
        max_samples=len(val_ds),
        step=global_step,
    )
    visualize_validation_batch(
        model, train_ds, device,
        out_dir=os.path.join(vis_dir, "final", "train"),
        preference_keys=preference_keys,
        max_samples=args.max_vis_samples,
        step=global_step,
    )

    # Final checkpoint
    torch.save({
        "step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, os.path.join(ckpt_dir, "final.pt"))
    if use_wandb:
        wandb.finish()
    print(f"\nDone. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
