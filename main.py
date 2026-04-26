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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as T
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_datasets, print_dataset_stats, load_anchors, load_cross_preferences
from model import RewardModel, DiscountedRewardModel, bradley_terry_loss, bradley_terry_loss_regression, anchor_loss
from flow_model import RewardModel as FlowRewardModel
from tasks import TASKS
from visualization import visualize_validation_batch, visualize_top_bottom_trajectories, plot_training_curves, plot_reward_correlation


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
    p.add_argument("--preload_offsets", type=int, default=5,
                   help="Number of temporal offsets to preload per trajectory pair (only used with --preload)")

    # Model
    p.add_argument("--model", default="transformer", choices=["transformer", "discounted", "flow"],
                   help="Model architecture: transformer, discounted, or flow (flow matching)")
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
    p.add_argument("--n_sample_steps", type=int, default=10,
                   help="ODE integration steps for --model flow")
    p.add_argument("--n_samples", type=int, default=10,
                   help="Number of samples to average for --model flow reward estimate")
    p.add_argument("--reward_sigmoid", action="store_true",
                   help="Apply sigmoid to reward outputs (maps to [0,1]); default is unbounded rewards")
    p.add_argument("--ptp", action="store_true",
                   help="Past token prediction: auxiliary flow matching loss for action chunk prediction (flow model only)")
    p.add_argument("--action_chunk_size", type=int, default=16,
                   help="Number of action steps per chunk for PTP")
    p.add_argument("--action_weight", type=float, default=1.0,
                   help="Weight for PTP action loss relative to BT loss")

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--equal_weight", type=float, default=0.0,
                   help="Loss weight for Equal-labeled samples (0 = skip)")
    p.add_argument("--regression", action="store_true",
                   help="Use MSE regression to 0/1 targets instead of Bradley-Terry cross-entropy (non-flow models)")
    p.add_argument("--anchor", action="store_true",
                   help="Add anchor loss to bound rewards: force specific trajectories toward 0 (bad) or 1 (good)")
    p.add_argument("--anchors_file", type=str, default=None,
                   help="Path to anchors JSON file (see preferences_*/anchors.json for format)")
    p.add_argument("--anchor_weight", type=float, default=1.0,
                   help="Weight for anchor loss relative to preference loss")
    p.add_argument("--cross_preferences_dir", type=str, default=None,
                   help="Directory containing cross-preference JSON files (preference_X.json). "
                        "Rollouts are looked up by timestamp from --preferences_dir sessions.")
    p.add_argument("--success_connection_rate", type=float, default=0.0,
                   help="Fraction of each batch to replace with success-vs-failure cross-pairs "
                        "(0=disabled; 0.1 targets ~10%% of batch size as cross-pairs, removing "
                        "the same number of original pairs so total batch size stays constant)")

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



def make_obs(traj: dict, device: torch.device, augment: bool = False) -> dict:
    """Move trajectory tensors to device and optionally augment images."""
    tp = traj["third_person"].to(device)
    wr = traj["wrist"].to(device)
    if augment:
        # augment_traj expects float [0,1]
        tp = augment_traj(tp.float() / 255.0)
        wr = augment_traj(wr.float() / 255.0)
    obs = {
        "third_person": tp,
        "wrist": wr,
        "padding_mask": traj["padding_mask"].to(device),
        "proprio": traj["proprio"].to(device),
    }
    return obs


def make_success_collate_fn(rate: float, num_preferences: int):
    """
    Returns a collate_fn that replaces up to round(rate * B) batch items with
    success-vs-failure cross-pairs drawn from within the same batch.

    Cross-pairs get labels=all-ones (success preferred over failure in all dims).
    If fewer cross-pairs are available than the target, the original batch is kept
    full and only the available cross-pairs are added instead.
    """
    from torch.utils.data.dataloader import default_collate

    def collate_fn(samples):
        if rate <= 0:
            return default_collate(samples)

        B = len(samples)
        n_target = max(1, round(rate * B))

        all_trajs, all_succ = [], []
        for s in samples:
            all_trajs.append(s["traj_a"])
            all_succ.append(s["succeeded_a"].item())
            all_trajs.append(s["traj_b"])
            all_succ.append(s["succeeded_b"].item())

        succ_idx = [i for i, v in enumerate(all_succ) if v == 1]
        fail_idx = [i for i, v in enumerate(all_succ) if v == 0]
        all_cross = [(si, fi) for si in succ_idx for fi in fail_idx]
        n_use = min(len(all_cross), n_target)

        if n_use == 0:
            batch = default_collate(samples)
            batch["n_cross_pairs"] = 0
            return batch

        chosen = random.sample(all_cross, n_use)
        new_samples = list(samples[: B - n_use])
        for si, fi in chosen:
            new_samples.append({
                "traj_a": all_trajs[si],
                "traj_b": all_trajs[fi],
                "labels": torch.ones(num_preferences, dtype=torch.float32),
                "succeeded_a": torch.tensor(1, dtype=torch.int8),
                "succeeded_b": torch.tensor(0, dtype=torch.int8),
                "session": "cross_pair",
            })

        batch = default_collate(new_samples)
        batch["n_cross_pairs"] = n_use
        return batch

    return collate_fn


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
def evaluate(model, val_loader, device, equal_weight, num_preferences, action_weight=0.0, regression=False):
    model.eval()
    total_loss = 0.0
    total_correct = torch.zeros(num_preferences)
    total_labeled = torch.zeros(num_preferences)
    n_batches = 0
    is_flow = isinstance(model, FlowRewardModel)

    for batch in val_loader:
        labels = batch["labels"].to(device)

        if is_flow:
            obs_a = make_obs(batch["traj_a"], device)
            obs_b = make_obs(batch["traj_b"], device)
            cls_a, frame_tokens_a = model.encode(obs_a)
            cls_b, frame_tokens_b = model.encode(obs_b)
            loss, per_dim_correct, per_dim_labeled = model.bradley_terry_flow_matching_loss(cls_a, cls_b, labels)
            if action_weight > 0 and model.ptp and "action_chunks" in batch["traj_a"]:
                action_loss = model.action_flow_loss(
                    frame_tokens_a, batch["traj_a"]["action_chunks"].to(device), obs_a["padding_mask"],
                    frame_tokens_b, batch["traj_b"]["action_chunks"].to(device), obs_b["padding_mask"],
                )
                loss = loss + action_weight * action_loss
        else:
            tp_a = batch["traj_a"]["third_person"].to(device)
            wr_a = batch["traj_a"]["wrist"].to(device)
            tp_b = batch["traj_b"]["third_person"].to(device)
            wr_b = batch["traj_b"]["wrist"].to(device)
            mask_a = batch["traj_a"]["padding_mask"].to(device)
            mask_b = batch["traj_b"]["padding_mask"].to(device)
            r_a = model(tp_a, wr_a, mask_a)
            r_b = model(tp_b, wr_b, mask_b)
            if regression:
                loss, per_dim_correct, per_dim_labeled = bradley_terry_loss_regression(r_a, r_b, labels)
            else:
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

    run_tags = [args.model]
    if args.model == "flow" and args.ptp:
        run_tags.append("ptp")
    if getattr(args, "regression", False):
        run_tags.append("regression")
    if getattr(args, "anchor", False):
        run_tags.append("anchor")
    if getattr(args, "success_connection_rate", 0.0) > 0:
        run_tags.append(f"scr{args.success_connection_rate}")
    if args.freeze_backbone:
        run_tags.append("frozen")
    slurm_id = os.environ.get("SLURM_JOB_ID")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + "_".join(run_tags)
    if slurm_id:
        run_name = run_name + f"_j{slurm_id}"
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
        action_chunk_size=args.action_chunk_size if args.ptp else 0,
        preload_offsets=args.preload_offsets,
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    print_dataset_stats(train_ds, val_ds)

    if args.cross_preferences_dir:
        if not os.path.isdir(args.cross_preferences_dir):
            print(f"[cross_preferences] Directory not found: {args.cross_preferences_dir}, skipping")
        else:
            cross_samples = load_cross_preferences(
                cross_dir=args.cross_preferences_dir,
                preference_dirs=preference_dirs,
                preference_keys=preference_keys,
                stride=args.stride,
                seq_len=args.seq_len,
                img_size=(args.img_size, args.img_size),
                action_chunk_size=args.action_chunk_size if args.ptp else 0,
            )
            train_ds.samples.extend(cross_samples)
            print(f"[cross_preferences] Train samples after cross-preferences: {len(train_ds.samples)}")

    anchor_entries = []
    if args.anchor:
        if not args.anchors_file:
            raise ValueError("--anchor requires --anchors_file")
        anchor_entries = load_anchors(
            args.anchors_file,
            preference_keys,
            stride=args.stride,
            seq_len=args.seq_len,
            img_size=(args.img_size, args.img_size),
            action_chunk_size=args.action_chunk_size if args.ptp else 0,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_success_collate_fn(args.success_connection_rate, len(preference_keys)),
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
    if args.model == "flow":
        model = FlowRewardModel(
            num_preferences=len(preference_keys),
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
            backbone=args.backbone,
            frozen_backbone=args.freeze_backbone,
            n_sample_steps=args.n_sample_steps,
            n_samples=args.n_samples,
            ptp=args.ptp,
            action_chunk_size=args.action_chunk_size,
        ).to(device)
    elif args.model == "discounted":
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
    import collections
    # Rolling buffer of per-trajectory reward vectors for correlation logging.
    # Keeps ~16 batches worth of individual trajectory rewards (r_a and r_b).
    _corr_buf_maxlen = 16 * args.batch_size * 2
    reward_corr_buffer = collections.deque(maxlen=_corr_buf_maxlen)
    global_step = 0

    # ---- pre-training baseline ----
    val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
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
            labels = batch["labels"].to(device)
            t_transfer = time.perf_counter() - t0

            t0 = time.perf_counter()
            optimizer.zero_grad()
            if isinstance(model, FlowRewardModel):
                obs_a = make_obs(batch["traj_a"], device, augment=True)
                obs_b = make_obs(batch["traj_b"], device, augment=True)
                t_transfer = time.perf_counter() - t0
                t0 = time.perf_counter()
                cls_a, frame_tokens_a = model.encode(obs_a)
                cls_b, frame_tokens_b = model.encode(obs_b)
                t_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                loss, per_dim_correct, per_dim_labeled = model.bradley_terry_flow_matching_loss(cls_a, cls_b, labels)
                if args.ptp:
                    action_chunks_a = batch["traj_a"]["action_chunks"].to(device)
                    action_chunks_b = batch["traj_b"]["action_chunks"].to(device)
                    mask_a = obs_a["padding_mask"]
                    mask_b = obs_b["padding_mask"]
                    action_loss = model.action_flow_loss(
                        frame_tokens_a, action_chunks_a, mask_a,
                        frame_tokens_b, action_chunks_b, mask_b,
                    )
                    loss = loss + args.action_weight * action_loss
            else:
                tp_a = augment_traj(batch["traj_a"]["third_person"].to(device).float() / 255.0)
                wr_a = augment_traj(batch["traj_a"]["wrist"].to(device).float() / 255.0)
                tp_b = augment_traj(batch["traj_b"]["third_person"].to(device).float() / 255.0)
                wr_b = augment_traj(batch["traj_b"]["wrist"].to(device).float() / 255.0)
                mask_a = batch["traj_a"]["padding_mask"].to(device)
                mask_b = batch["traj_b"]["padding_mask"].to(device)
                t_transfer = time.perf_counter() - t0
                t0 = time.perf_counter()
                r_a = model(tp_a, wr_a, mask_a)
                r_b = model(tp_b, wr_b, mask_b)
                t_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                if args.regression:
                    loss, per_dim_correct, per_dim_labeled = bradley_terry_loss_regression(r_a, r_b, labels)
                else:
                    loss, per_dim_correct, per_dim_labeled = bradley_terry_loss(r_a, r_b, labels, args.equal_weight)
            per_dim_acc = per_dim_correct / per_dim_labeled.clamp(min=1)

            # --- accumulate rewards into rolling buffer for correlation logging ---
            with torch.no_grad():
                if isinstance(model, FlowRewardModel):
                    # Use a single cheap ODE sample (n_steps=4) to avoid slowing training.
                    ra_buf = model.sample_reward(cls_a, n_samples=1, n_steps=4).cpu().numpy()
                    rb_buf = model.sample_reward(cls_b, n_samples=1, n_steps=4).cpu().numpy()
                else:
                    ra_buf = r_a.detach().cpu().numpy()
                    rb_buf = r_b.detach().cpu().numpy()
            for row in ra_buf:
                reward_corr_buffer.append(row)
            for row in rb_buf:
                reward_corr_buffer.append(row)

            if anchor_entries:
                anc_loss = torch.zeros(1, device=device)
                for entry in anchor_entries:
                    traj = entry["traj"]
                    if isinstance(model, FlowRewardModel):
                        obs_anc = {
                            "third_person": traj["third_person"].unsqueeze(0).to(device),
                            "wrist": traj["wrist"].unsqueeze(0).to(device),
                            "padding_mask": traj["padding_mask"].unsqueeze(0).to(device),
                            "proprio": traj["proprio"].unsqueeze(0).to(device),
                        }
                        r_anc = model(obs_anc)
                    else:
                        r_anc = model(
                            traj["third_person"].unsqueeze(0).to(device),
                            traj["wrist"].unsqueeze(0).to(device),
                            traj["padding_mask"].unsqueeze(0).to(device),
                        )
                    anc_loss = anc_loss + anchor_loss(r_anc, entry["dim"], entry["target"])
                loss = loss + args.anchor_weight * anc_loss / len(anchor_entries)

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
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/acc_mean": mean_acc,
                        "train/lr": scheduler.get_last_lr()[0],
                        **{f"train/acc_{k}": v for k, v in zip(preference_keys, per_dim_acc.cpu().numpy())},
                    }
                    if args.ptp and isinstance(model, FlowRewardModel):
                        log_dict["train/action_loss"] = action_loss.item()
                    if anchor_entries:
                        log_dict["train/anchor_loss"] = anc_loss.item() / len(anchor_entries)
                    if args.success_connection_rate > 0:
                        n_cross = batch.get("n_cross_pairs", 0)
                        log_dict["train/cross_pair_frac"] = n_cross / args.batch_size
                    if len(reward_corr_buffer) >= len(preference_keys) + 1:
                        corr_arr = np.array(reward_corr_buffer)
                        corr_fig = plot_reward_correlation(corr_arr, preference_keys)
                        log_dict["train/reward_correlation"] = wandb.Image(corr_fig)
                        plt.close(corr_fig)
                    wandb.log(log_dict, step=global_step)

            # ---- validation ----
            if global_step % args.eval_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
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
                val_loss_vis, val_acc_vis = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
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
                top_bottom_train_mp4s, uniform_train_mp4s = visualize_top_bottom_trajectories(
                    model, train_ds, device,
                    out_dir=os.path.join(vis_step_dir, "top_bottom_train"),
                    preference_keys=preference_keys,
                    n=5, n_uniform=10,
                    step=global_step,
                )
                top_bottom_val_mp4s, uniform_val_mp4s = visualize_top_bottom_trajectories(
                    model, val_ds, device,
                    out_dir=os.path.join(vis_step_dir, "top_bottom_val"),
                    preference_keys=preference_keys,
                    n=5, n_uniform=10,
                    step=global_step,
                )
                print(f"  [Vis] Saved {n_vis_val} val + {n_vis_train} train + top/bottom + uniform spectrum → {vis_step_dir}")

                if use_wandb:
                    val_mp4s = sorted(f for f in os.listdir(os.path.join(vis_step_dir, "val")) if f.endswith(".mp4"))
                    train_mp4s = sorted(f for f in os.listdir(os.path.join(vis_step_dir, "train")) if f.endswith(".mp4"))
                    wandb.log({
                        "val/loss": val_loss_vis,
                        "val/acc_mean": float(mean_val_acc_vis),
                        **{f"val/acc_{k}": float(v) for k, v in zip(preference_keys, val_acc_vis)},
                        **{f"val/video_{i}": wandb.Video(os.path.join(vis_step_dir, "val", f), format="mp4") for i, f in enumerate(val_mp4s)},
                        **{f"train/video_{i}": wandb.Video(os.path.join(vis_step_dir, "train", f), format="mp4") for i, f in enumerate(train_mp4s)},
                        **{f"top_bottom_train/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, top_bottom_train_mp4s)},
                        **{f"top_bottom_val/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, top_bottom_val_mp4s)},
                        **{f"uniform_train/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, uniform_train_mp4s)},
                        **{f"uniform_val/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, uniform_val_mp4s)},
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
    val_loss, val_acc = evaluate(model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
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
