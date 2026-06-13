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

from dataset import make_datasets, print_dataset_stats, load_anchors, load_cross_preferences, OpenPreferenceDataset, DummyPreferenceDataset, PreferenceDataset, auto_detect_preference_keys
from model import RewardModel, DiscountedRewardModel, bradley_terry_loss, bradley_terry_loss_regression, anchor_loss
from flow_model import RewardModel as FlowRewardModel
from qwen_model import QwenRewardModel
from tasks import TASKS
from visualization import visualize_validation_batch, visualize_top_bottom_trajectories, plot_training_curves, plot_reward_correlation


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--task", default="cube_in_three_bowls",
                   choices=list(TASKS.keys()) + ["auto"],
                   help="Task name — determines which preference dimensions to use. "
                        "'auto' infers the preference keys from the JSONs themselves "
                        "(union of keys observed across all preference files).")
    p.add_argument("--preferences_dir",type=str, default="preferences",
                   help="Comma-separated preference dirs. Pass an empty string to skip "
                        "regular preferences and train only on --cross_preferences_dir.")
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
    p.add_argument("--only_large", action="store_true",
                   help="Only use *_large.hdf5 files (skip pairs where the large variant doesn't exist)")

    # Model
    p.add_argument("--model", default="transformer", choices=[
                       "transformer", "discounted", "flow",
                       "qwen", "qwen_lora", "qwen_open",
                       "qwen_discounted", "qwen_open_discounted",
                       "qwen_open_cum",
                   ],
                   help="Model architecture")
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

    # Qwen-specific (following QwenLM/Qwen3-VL finetuning approach)
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank (--model qwen_lora)")
    p.add_argument("--lora_alpha", type=int, default=16,
                   help="LoRA alpha scaling factor (--model qwen_lora)")
    p.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                   help="HuggingFace model identifier (--model qwen/qwen_lora)")
    p.add_argument("--tune_vision", action="store_true",
                   help="Unfreeze vision encoder (default: frozen, following QwenLM approach)")
    p.add_argument("--tune_mlp", action="store_true", default=True,
                   help="Train MLP projector (default: True)")
    p.add_argument("--no_tune_mlp", action="store_false", dest="tune_mlp",
                   help="Freeze MLP projector")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Disable gradient checkpointing for Qwen model")

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--equal_weight", type=float, default=1.0,
                   help="Loss weight for Equal-labeled samples")
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
    p.add_argument("--small_vis_interval", type=int, default=999999,
                   help="Run only the cheap pair-visualizations every N steps "
                        "(skips top/bottom ranking; useful for slow models like Qwen)")
    p.add_argument("--small_vis_max_samples", type=int, default=4,
                   help="Number of pairs to visualize per call in small_vis mode")
    p.add_argument("--save_interval", type=int, default=200, help="Save checkpoint every N steps")
    p.add_argument("--max_vis_samples", type=int, default=8)

    # Wandb
    p.add_argument("--wandb_project", default="reward_learning")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    # Debug: synthetic dataset (skip HDF5 / cross-prefs / anchors entirely).
    p.add_argument("--debug_dummy", action="store_true",
                   help="Use a tiny in-memory random dataset for fast DDP / OOM debugging")
    p.add_argument("--debug_dummy_train", type=int, default=64,
                   help="Number of dummy train pairs (only with --debug_dummy)")
    p.add_argument("--debug_dummy_val", type=int, default=16,
                   help="Number of dummy val pairs (only with --debug_dummy)")
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
def axis_sensitivity_probe(model, val_loader, device, all_axes, n_samples: int = 4):
    """Feed the same trajectories to the model with every axis label and return
    the resulting per-axis rewards.

    Used to diagnose whether the model actually conditions its output on the
    axis prompt. If the returned matrix has ~0 spread across axes, the model
    is ignoring the label.

    Returns: (mat (n_samples, n_axes), axes (list[str])) or (None, None).
    """
    if not isinstance(model, QwenRewardModel):
        return None, None
    was_training = model.training
    model.eval()

    batch = next(iter(val_loader))
    tp = batch["traj_a"]["third_person"][:n_samples].to(device)
    wr = batch["traj_a"]["wrist"][:n_samples].to(device)
    mask = batch["traj_a"]["padding_mask"][:n_samples].to(device)
    n = tp.shape[0]

    cols = []
    for ax in all_axes:
        r = model(tp, wr, mask, axis_labels=[ax] * n)
        cols.append(r.squeeze(-1).float().cpu().numpy())
    mat = np.stack(cols, axis=1)  # (n_samples, n_axes)

    if was_training:
        model.train()
    return mat, list(all_axes)


def _per_sample_correct_labeled(r_a: torch.Tensor, r_b: torch.Tensor, labels: torch.Tensor):
    """Per-sample correctness for open-axis (B,1) reward pairs.

    Returns (correct: BoolTensor(B,), labeled: BoolTensor(B,)).
    """
    prob_a = torch.sigmoid(r_a - r_b).squeeze(-1)
    lbl = labels.squeeze(-1)
    is_a = lbl == 1.0
    is_b = lbl == 0.0
    pred_a_wins = prob_a > 0.5
    correct = (pred_a_wins & is_a) | (~pred_a_wins & is_b)
    labeled = is_a | is_b
    return correct, labeled


@torch.no_grad()
def evaluate(model, val_loader, device, equal_weight, num_preferences, action_weight=0.0, regression=False):
    model.eval()
    total_loss = 0.0
    total_correct = torch.zeros(num_preferences)
    total_labeled = torch.zeros(num_preferences)
    n_batches = 0
    is_flow = isinstance(model, FlowRewardModel)
    per_axis_correct: dict[str, int] = {}
    per_axis_labeled: dict[str, int] = {}

    for batch in val_loader:
        labels = batch["labels"].to(device)
        axis_labels = batch.get("axis_label")  # list[str] or None

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
            r_a = model(tp_a, wr_a, mask_a, axis_labels=axis_labels) if isinstance(model, QwenRewardModel) else model(tp_a, wr_a, mask_a)
            r_b = model(tp_b, wr_b, mask_b, axis_labels=axis_labels) if isinstance(model, QwenRewardModel) else model(tp_b, wr_b, mask_b)
            if regression:
                loss, per_dim_correct, per_dim_labeled = bradley_terry_loss_regression(r_a, r_b, labels)
            else:
                loss, per_dim_correct, per_dim_labeled = bradley_terry_loss(r_a, r_b, labels, equal_weight)

            if axis_labels is not None:
                correct_b, labeled_b = _per_sample_correct_labeled(r_a, r_b, labels)
                correct_b = correct_b.cpu().tolist()
                labeled_b = labeled_b.cpu().tolist()
                for ax, c, l in zip(axis_labels, correct_b, labeled_b):
                    per_axis_correct[ax] = per_axis_correct.get(ax, 0) + int(c)
                    per_axis_labeled[ax] = per_axis_labeled.get(ax, 0) + int(l)

        total_loss += loss.item()
        total_correct += per_dim_correct.cpu()
        total_labeled += per_dim_labeled.cpu()
        n_batches += 1

    model.train()
    acc = (total_correct / total_labeled.clamp(min=1)).numpy()
    per_axis_acc = {
        ax: (per_axis_correct[ax] / max(per_axis_labeled[ax], 1))
        for ax in per_axis_correct
    }
    return total_loss / max(n_batches, 1), acc, per_axis_acc


def init_distributed():
    """Initialize torch.distributed if launched via torchrun. Returns (is_ddp, rank, world_size, local_rank)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def main():
    args = parse_args()

    is_ddp, ddp_rank, ddp_world_size, ddp_local_rank = init_distributed()
    is_main = (ddp_rank == 0)
    # Different seed per rank so augmentation/sampling diverge,
    # but DistributedSampler is itself seeded deterministically.
    set_seed(args.seed + ddp_rank)

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
    # Build the run name on rank 0 then broadcast so all ranks share the same out_dir.
    if is_main:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + "_".join(run_tags)
        if slurm_id:
            run_name = run_name + f"_j{slurm_id}"
    else:
        run_name = None
    if is_ddp:
        import torch.distributed as dist
        bcast = [run_name]
        dist.broadcast_object_list(bcast, src=0)
        run_name = bcast[0]

    args.out_dir = os.path.join(args.out_dir, run_name)
    vis_dir = os.path.join(args.out_dir, "visualizations")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    if is_ddp:
        device = torch.device("cuda", ddp_local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main:
        print(f"Device: {device} | world_size={ddp_world_size} | rank={ddp_rank}", flush=True)

    # ------------------------------------------------------------------ #
    # Wandb
    # ------------------------------------------------------------------ #
    use_wandb = (not args.no_wandb) and is_main
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config=vars(args),
        )

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    preference_dirs = [d.strip() for d in args.preferences_dir.split(",") if d.strip()]
    missing_pref_dirs = [d for d in preference_dirs if not os.path.isdir(d)]
    if missing_pref_dirs and is_main:
        print(f"[preferences] Skipping non-existent preference dirs: {missing_pref_dirs}",
              flush=True)
    preference_dirs = [d for d in preference_dirs if os.path.isdir(d)]
    skip_preferences = len(preference_dirs) == 0

    if args.task == "auto":
        cross_dirs_for_scan = [d.strip() for d in (args.cross_preferences_dir or "").split(",") if d.strip()]
        preference_keys = auto_detect_preference_keys(preference_dirs, cross_dirs_for_scan)
        if not preference_keys:
            raise ValueError("--task auto: no preference keys found in any JSON. "
                             "Check --preferences_dir / --cross_preferences_dir.")
        if is_main:
            print(f"[task=auto] Detected {len(preference_keys)} preference keys: {preference_keys}",
                  flush=True)
    else:
        preference_keys = [k.lower() for k in TASKS[args.task]]

    if args.debug_dummy:
        if is_main:
            print(f"[debug_dummy] Synthetic dataset: "
                  f"{args.debug_dummy_train} train / {args.debug_dummy_val} val pairs", flush=True)
        train_ds = DummyPreferenceDataset(
            n_samples=args.debug_dummy_train,
            seq_len=args.seq_len,
            img_size=(args.img_size, args.img_size),
            num_preferences=len(preference_keys),
            seed=args.seed,
        )
        val_ds = DummyPreferenceDataset(
            n_samples=args.debug_dummy_val,
            seq_len=args.seq_len,
            img_size=(args.img_size, args.img_size),
            num_preferences=len(preference_keys),
            seed=args.seed + 1,
        )
    elif skip_preferences:
        if is_main:
            print("[preferences] --preferences_dir is empty, skipping regular preferences. "
                  "Train/val will come from --cross_preferences_dir only.", flush=True)
        if not args.cross_preferences_dir:
            raise ValueError("--preferences_dir is empty but --cross_preferences_dir is not set; "
                             "no training data available.")
        ac = args.action_chunk_size if args.ptp else 0
        train_ds = PreferenceDataset([], preference_keys=preference_keys, stride=args.stride,
                                     seq_len=args.seq_len, img_size=(args.img_size, args.img_size),
                                     training=True,  preload=args.preload, action_chunk_size=ac,
                                     preload_offsets=args.preload_offsets, only_large=args.only_large)
        val_ds   = PreferenceDataset([], preference_keys=preference_keys, stride=args.stride,
                                     seq_len=args.seq_len, img_size=(args.img_size, args.img_size),
                                     training=False, preload=args.preload, action_chunk_size=ac,
                                     preload_offsets=args.preload_offsets, only_large=args.only_large)
    else:
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
            only_large=args.only_large,
            preference_keys=preference_keys if args.task == "auto" else None,
        )

        if is_main:
            print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
            print_dataset_stats(train_ds, val_ds)

    if (not args.debug_dummy) and args.cross_preferences_dir:
        cross_dirs = [d.strip() for d in args.cross_preferences_dir.split(",") if d.strip()]
        valid_cross_dirs = []
        for cd in cross_dirs:
            if os.path.isdir(cd):
                valid_cross_dirs.append(cd)
            else:
                print(f"[cross_preferences] Directory not found: {cd}, skipping")
        cross_samples = []
        if valid_cross_dirs:
            for cd in valid_cross_dirs:
                cross_samples.extend(load_cross_preferences(
                    cross_dir=cd,
                    preference_dirs=preference_dirs,
                    preference_keys=preference_keys,
                    stride=args.stride,
                    seq_len=args.seq_len,
                    img_size=(args.img_size, args.img_size),
                    action_chunk_size=args.action_chunk_size if args.ptp else 0,
                ))
            if args.preload and cross_samples:
                from dataset import _load_raw_hdf5, _trajectories_from_raw
                n_off = args.preload_offsets
                offsets = [int(i * args.stride / n_off) for i in range(n_off)]
                ac = args.action_chunk_size if args.ptp else 0
                img = (args.img_size, args.img_size)
                print(f"[cross_preferences] Preloading {len(cross_samples)} cross-preference pairs "
                      f"× {n_off} offset(s)...", flush=True)
                failed = []
                t_cross_start = time.time()
                # Load each unique HDF5 file once
                raw_cache = {}
                all_paths = set()
                for s in cross_samples:
                    all_paths.add(s["hdf5_a"])
                    all_paths.add(s["hdf5_b"])
                for pi, path in enumerate(all_paths):
                    try:
                        raw_cache[path] = _load_raw_hdf5(path, ac)
                    except (OSError, KeyError) as e:
                        print(f"[cross_preferences] Skipping corrupted HDF5: {path}: {e}", flush=True)
                    if (pi + 1) % 20 == 0 or pi + 1 == len(all_paths):
                        print(f"  [cross] Loaded {pi + 1}/{len(all_paths)} unique files "
                              f"[{time.time() - t_cross_start:.1f}s]", flush=True)
                # Extract offset trajectories from cached raw data
                for si, s in enumerate(cross_samples):
                    if s["hdf5_a"] not in raw_cache or s["hdf5_b"] not in raw_cache:
                        failed.append(si)
                        continue
                    try:
                        s["trajs_a"] = _trajectories_from_raw(raw_cache[s["hdf5_a"]], args.stride, args.seq_len, img, offsets, ac)
                        s["trajs_b"] = _trajectories_from_raw(raw_cache[s["hdf5_b"]], args.stride, args.seq_len, img, offsets, ac)
                    except (OSError, KeyError) as e:
                        print(f"[cross_preferences] Error extracting trajectories: {e}", flush=True)
                        failed.append(si)
                    if (si + 1) % 10 == 0 or si + 1 == len(cross_samples):
                        print(f"  [cross] Extracted {si + 1}/{len(cross_samples)} pairs "
                              f"[{time.time() - t_cross_start:.1f}s]", flush=True)
                del raw_cache
                cross_preload_time = time.time() - t_cross_start
                print(f"[cross_preferences] Total preload time: {cross_preload_time:.1f}s "
                      f"({cross_preload_time / max(len(cross_samples), 1):.2f}s/pair)", flush=True)
                for si in reversed(failed):
                    cross_samples.pop(si)
        if cross_samples:
            # Print episode length stats for cross-preference trajectories.
            cross_lengths = []
            for s in cross_samples:
                for path in (s["hdf5_a"], s["hdf5_b"]):
                    try:
                        with __import__('h5py').File(path, "r") as f:
                            dk = next(iter(f["data"].keys()))
                            cross_lengths.append(f[f"data/{dk}/obs/agent_view"].shape[0])
                    except (OSError, KeyError):
                        pass
            if cross_lengths:
                arr = np.array(cross_lengths)
                print(f"[cross_preferences] Episode lengths ({len(arr)} rollouts): "
                      f"min={arr.min()}  max={arr.max()}  "
                      f"mean={arr.mean():.1f}  median={np.median(arr):.1f}  std={arr.std():.1f}")
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(len(cross_samples))
            n_val = max(1, int(len(cross_samples) * args.val_fraction)) if cross_samples else 0
            val_idx   = set(perm[:n_val].tolist())
            train_cross = [cross_samples[i] for i in range(len(cross_samples)) if i not in val_idx]
            val_cross   = [cross_samples[i] for i in range(len(cross_samples)) if i in val_idx]
            train_ds.samples.extend(train_cross)
            val_ds.samples.extend(val_cross)
            if is_main:
                print(f"[cross_preferences] Split into {len(train_cross)} train / "
                      f"{len(val_cross)} val pairs")

    if not args.debug_dummy:
        all_paths = set()
        for s in train_ds.samples:
            all_paths.add(s["hdf5_a"])
            all_paths.add(s["hdf5_b"])
        if is_main:
            print(f"Total train pairs: {len(train_ds.samples)}  "
                  f"({len(train_ds.samples) * 2} trajectory slots, {len(all_paths)} unique trajectories)")

    # Log preload times to wandb
    if use_wandb and args.preload:
        preload_log = {
            "preload/train_time_s": train_ds.preload_time_s,
            "preload/val_time_s": val_ds.preload_time_s,
            "preload/train_pairs": len(train_ds),
            "preload/val_pairs": len(val_ds),
        }
        cross_preload_time = locals().get("cross_preload_time", 0.0)
        if cross_preload_time > 0:
            preload_log["preload/cross_time_s"] = cross_preload_time
            preload_log["preload/cross_pairs"] = len(cross_samples)
        preload_log["preload/total_time_s"] = (
            train_ds.preload_time_s + val_ds.preload_time_s
            + preload_log.get("preload/cross_time_s", 0.0)
        )
        wandb.log(preload_log)

    anchor_entries = []
    if args.anchor and not args.debug_dummy:
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

    # Wrap datasets for qwen_open (per-axis samples).
    # When equal_weight == 0, equal-labeled (0.5) axes contribute no gradient,
    # so we drop them to keep DDP backward well-defined on every per-rank batch.
    axis_preference_keys = list(preference_keys)
    if args.model in ("qwen_open", "qwen_open_discounted", "qwen_open_cum"):
        skip_equal = (args.equal_weight == 0.0)
        train_ds = OpenPreferenceDataset(train_ds, preference_keys, skip_equal=skip_equal)
        val_ds = OpenPreferenceDataset(val_ds, preference_keys, skip_equal=skip_equal)
        print(f"[{args.model}] Expanded to {len(train_ds)} train / {len(val_ds)} val per-axis samples (skip_equal={skip_equal})")
        preference_keys = ["overall"]

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=ddp_world_size, rank=ddp_rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_success_collate_fn(args.success_connection_rate, len(preference_keys)),
    )
    # Validation runs on rank 0 only, so keep it non-distributed.
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
    elif args.model in ("qwen", "qwen_lora", "qwen_open", "qwen_discounted", "qwen_open_discounted", "qwen_open_cum"):
        model = QwenRewardModel(
            num_preferences=len(preference_keys),  # 1 for open variants
            model_name=args.qwen_model_name,
            use_lora=(args.model == "qwen_lora"),
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            reward_sigmoid=args.reward_sigmoid,
            tune_vision=args.tune_vision,
            tune_mlp=args.tune_mlp,
            tune_llm=True,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            discounted=args.model in ("qwen_discounted", "qwen_open_discounted"),
            open_cum=(args.model == "qwen_open_cum"),
        )
        print("Moving model to device...", flush=True)
        model = model.to(device)
        print("Model on device.", flush=True)
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
    if is_main:
        print(f"Model parameters: {n_params:,}")

    # Separate backbone params for optional lower lr (use the unwrapped model to
    # iterate parameters before wrapping in DDP).
    if args.model in ("qwen", "qwen_lora", "qwen_open", "qwen_discounted", "qwen_open_discounted", "qwen_open_cum"):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": trainable_params, "lr": args.lr}]
    else:
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

    # Wrap in DDP after optimizer captures the underlying parameters.
    #
    # Critical flags:
    # - static_graph=True: required for gradient checkpointing (use_reentrant=False)
    #   to actually save memory under DDP. Without it, DDP's autograd hooks
    #   prevent the checkpointing path from being taken, and the model falls
    #   back to saving every layer's activations — peak memory jumps by
    #   ~36 GiB for the 4B Qwen model on batch=32. static_graph requires the
    #   autograd graph to be identical every iteration (true for our model).
    # - find_unused_parameters=False: with static_graph=True this is the right
    #   default. All trainable params (MLP, LLM, reward heads) are touched
    #   every forward, so DDP knows what to wait for.
    # - gradient_as_bucket_view=True: reuses gradient storage as bucket views.
    # - broadcast_buffers=False: no running-stat buffers to sync.
    unwrapped_model = model
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank],
            output_device=ddp_local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        unwrapped_model = model.module

    # CRITICAL: HF's `from_pretrained` leaves the loaded model in eval mode,
    # which propagates `training=False` to every submodule including the
    # decoder layers. GradientCheckpointingLayer.__call__ tests
    # `self.gradient_checkpointing and self.training` on the LAYER; if the
    # layer is in eval mode it silently falls through to the non-checkpointing
    # path, blowing up activation memory by ~36 GiB on the 4B Qwen model.
    #
    # On single-GPU this is masked by the pre-train evaluate() call (which
    # runs on is_main=True and ends with `model.train()`, fixing the state).
    # Under DDP ranks 1-3 that branch is skipped, so the bug only shows up
    # in multi-GPU runs. Calling .train() here ensures all ranks enter the
    # training loop with layer.training=True so checkpointing actually fires.
    model.train()

    # Diagnostic: confirm layer-level gradient_checkpointing flag + training
    # state after the forced .train(). Print on every rank.
    if isinstance(unwrapped_model, QwenRewardModel):
        qmodel = getattr(unwrapped_model.qwen, "model", None)
        qlm = getattr(qmodel, "language_model", None) if qmodel is not None else None
        qlayers = getattr(qlm, "layers", None) if qlm is not None else None
        l0_gc = getattr(qlayers[0], "gradient_checkpointing", "?") if qlayers is not None else "no_layers"
        l0_tr = getattr(qlayers[0], "training", "?") if qlayers is not None else "?"
        rank_str = str(ddp_rank) if is_ddp else "single"
        print(f"[post_ddp_diag rank={rank_str}] layer0.gc={l0_gc} layer0.training={l0_tr} unwrapped.training={unwrapped_model.training}", flush=True)
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

    # ---- pre-training baseline (rank 0 only) ----
    if is_main:
        val_loss, val_acc, val_axis_acc = evaluate(unwrapped_model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
        val_losses.append(val_loss)
        val_accs.append(val_acc.tolist())
        acc_str = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc))
        print(f"[Pre-train] Loss {val_loss:.4f} | {acc_str}")
        if val_axis_acc:
            ax_str = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(val_axis_acc.items()))
            print(f"[Pre-train per-axis] {ax_str}")

    train_axis_correct: dict[str, int] = collections.defaultdict(int)
    train_axis_labeled: dict[str, int] = collections.defaultdict(int)

    epoch_pbar = tqdm(range(args.epochs), desc="Epochs", disable=not is_main)
    for epoch in epoch_pbar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t_data_end = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, disable=not is_main):
            t_data = time.perf_counter() - t_data_end

            t0 = time.perf_counter()
            labels = batch["labels"].to(device)
            t_transfer = time.perf_counter() - t0

            t0 = time.perf_counter()
            optimizer.zero_grad()
            if isinstance(unwrapped_model, FlowRewardModel):
                obs_a = make_obs(batch["traj_a"], device, augment=True)
                obs_b = make_obs(batch["traj_b"], device, augment=True)
                t_transfer = time.perf_counter() - t0
                t0 = time.perf_counter()
                # FlowRewardModel uses non-forward entry points; DDP wraps .forward only.
                # Use the unwrapped model and rely on autograd hooks across all ranks.
                cls_a, frame_tokens_a = unwrapped_model.encode(obs_a)
                cls_b, frame_tokens_b = unwrapped_model.encode(obs_b)
                t_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                loss, per_dim_correct, per_dim_labeled = unwrapped_model.bradley_terry_flow_matching_loss(cls_a, cls_b, labels)
                if args.ptp:
                    action_chunks_a = batch["traj_a"]["action_chunks"].to(device)
                    action_chunks_b = batch["traj_b"]["action_chunks"].to(device)
                    mask_a = obs_a["padding_mask"]
                    mask_b = obs_b["padding_mask"]
                    action_loss = unwrapped_model.action_flow_loss(
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
                axis_labels = batch.get("axis_label")  # list[str] or None
                if isinstance(unwrapped_model, QwenRewardModel):
                    r_a = model(tp_a, wr_a, mask_a, axis_labels=axis_labels)
                    r_b = model(tp_b, wr_b, mask_b, axis_labels=axis_labels)
                else:
                    r_a = model(tp_a, wr_a, mask_a)
                    r_b = model(tp_b, wr_b, mask_b)
                t_forward = time.perf_counter() - t0
                t0 = time.perf_counter()
                if args.regression:
                    loss, per_dim_correct, per_dim_labeled = bradley_terry_loss_regression(r_a, r_b, labels)
                else:
                    loss, per_dim_correct, per_dim_labeled = bradley_terry_loss(r_a, r_b, labels, args.equal_weight)
                if axis_labels is not None:
                    with torch.no_grad():
                        _c, _l = _per_sample_correct_labeled(r_a, r_b, labels)
                    _c = _c.cpu().tolist()
                    _l = _l.cpu().tolist()
                    for ax, c, l in zip(axis_labels, _c, _l):
                        train_axis_correct[ax] += int(c)
                        train_axis_labeled[ax] += int(l)
            per_dim_acc = per_dim_correct / per_dim_labeled.clamp(min=1)

            # --- accumulate rewards into rolling buffer for correlation logging ---
            with torch.no_grad():
                if isinstance(unwrapped_model, FlowRewardModel):
                    # Use a single cheap ODE sample (n_steps=4) to avoid slowing training.
                    ra_buf = unwrapped_model.sample_reward(cls_a, n_samples=1, n_steps=4).cpu().numpy()
                    rb_buf = unwrapped_model.sample_reward(cls_b, n_samples=1, n_steps=4).cpu().numpy()
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
                    if isinstance(unwrapped_model, FlowRewardModel):
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

            # ---- logging (rank 0 only) ----
            if is_main and global_step % args.log_interval == 0:
                mean_acc = per_dim_acc.mean().item()
                epoch_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{mean_acc:.3f}")
                # Peak GPU memory for this log window (then reset the counter).
                peak_mem_gib = 0.0
                if device.type == "cuda":
                    peak_mem_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    torch.cuda.reset_peak_memory_stats(device)
                print(
                    f"Epoch {epoch+1:3d} | Step {global_step:5d} | "
                    f"Loss {loss.item():.4f} | Acc {mean_acc:.3f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e} | "
                    f"PeakMem {peak_mem_gib:.2f} GiB | "
                    f"data={t_data:.2f}s  transfer={t_transfer:.2f}s  fwd={t_forward:.2f}s  bwd={t_backward:.2f}s"
                    + (f"  pil={unwrapped_model._last_prep_time:.2f}s" if hasattr(unwrapped_model, "_last_prep_time") else ""),
                    flush=True,
                )
                if use_wandb:
                    timing_dict = {
                        "timing/data_loading": t_data,
                        "timing/gpu_transfer": t_transfer,
                        "timing/forward": t_forward,
                        "timing/backward": t_backward,
                        "mem/peak_gib": peak_mem_gib,
                    }
                    if hasattr(unwrapped_model, "_last_prep_time"):
                        timing_dict["timing/pil_processor"] = unwrapped_model._last_prep_time
                    wandb.log(timing_dict, step=global_step)
                if use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/acc_mean": mean_acc,
                        "train/lr": scheduler.get_last_lr()[0],
                        **{f"train/acc_{k}": v for k, v in zip(preference_keys, per_dim_acc.cpu().numpy())},
                    }
                    if train_axis_labeled:
                        for ax in sorted(train_axis_correct):
                            if train_axis_labeled[ax] > 0:
                                log_dict[f"train/acc_axis/{ax}"] = (
                                    train_axis_correct[ax] / train_axis_labeled[ax]
                                )
                        train_axis_correct.clear()
                        train_axis_labeled.clear()
                    if args.ptp and isinstance(unwrapped_model, FlowRewardModel):
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

            # ---- validation (rank 0 only) ----
            if is_main and global_step % args.eval_interval == 0:
                val_loss, val_acc, val_axis_acc = evaluate(unwrapped_model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
                val_losses.append(val_loss)
                val_accs.append(val_acc.tolist())
                mean_val_acc = val_acc.mean()
                epoch_pbar.set_postfix(
                    loss=f"{loss.item():.4f}", acc=f"{per_dim_acc.mean().item():.3f}",
                    val_loss=f"{val_loss:.4f}", val_acc=f"{mean_val_acc:.3f}",
                )
                acc_str = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc))
                print(f"  [Val] Step {global_step:5d} | Loss {val_loss:.4f} | Mean acc {mean_val_acc:.3f} | {acc_str}")
                if val_axis_acc:
                    ax_str = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(val_axis_acc.items()))
                    print(f"  [Val per-axis] {ax_str}")
                if use_wandb:
                    val_log = {
                        "val/loss": val_loss,
                        "val/acc_mean": float(mean_val_acc),
                        **{f"val/acc_{k}": float(v) for k, v in zip(preference_keys, val_acc)},
                        **{f"val/acc_axis/{ax}": v for ax, v in val_axis_acc.items()},
                    }
                    wandb.log(val_log, step=global_step)

                # ---- axis-sensitivity probe (open models only) ----
                if args.model in ("qwen_open", "qwen_open_discounted", "qwen_open_cum"):
                    probe_mat, probe_axes = axis_sensitivity_probe(
                        unwrapped_model, val_loader, device, axis_preference_keys
                    )
                    if probe_mat is not None:
                        spread_per_sample = probe_mat.std(axis=1)
                        spread_mean = float(spread_per_sample.mean())
                        spread_max = float(spread_per_sample.max())
                        reward_range = float(probe_mat.max() - probe_mat.min())
                        print(f"  [Axis probe] mean spread {spread_mean:.4f} | "
                              f"max spread {spread_max:.4f} | range {reward_range:.4f}")
                        for i in range(probe_mat.shape[0]):
                            ax_str = " | ".join(
                                f"{a}: {probe_mat[i, j]:+.3f}"
                                for j, a in enumerate(probe_axes)
                            )
                            print(f"    sample {i}: {ax_str}")
                        if use_wandb:
                            wandb.log({
                                "diag/axis_spread_mean": spread_mean,
                                "diag/axis_spread_max": spread_max,
                                "diag/axis_reward_range": reward_range,
                                **{f"diag/axis_reward_mean/{a}": float(probe_mat[:, j].mean())
                                   for j, a in enumerate(probe_axes)},
                            }, step=global_step)
                plot_training_curves(
                    train_losses, train_accs, val_losses, val_accs,
                    out_path=os.path.join(args.out_dir, "training_curves.png"),
                    preference_keys=preference_keys,
                )

            # ---- visualization + wandb val logging (rank 0 only) ----
            if is_main and global_step % args.vis_interval == 0:
                val_loss_vis, val_acc_vis, val_axis_acc_vis = evaluate(unwrapped_model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
                mean_val_acc_vis = val_acc_vis.mean()
                acc_str_vis = " | ".join(f"{k}: {v:.2f}" for k, v in zip(preference_keys, val_acc_vis))
                print(f"  [Vis/Val] Step {global_step:5d} | Loss {val_loss_vis:.4f} | Mean acc {mean_val_acc_vis:.3f} | {acc_str_vis}")

                vis_step_dir = os.path.join(vis_dir, f"step{global_step:06d}")
                n_vis_val = visualize_validation_batch(
                    unwrapped_model, val_ds, device,
                    out_dir=os.path.join(vis_step_dir, "val"),
                    preference_keys=preference_keys,
                    max_samples=args.max_vis_samples,
                    step=global_step,
                )
                n_vis_train = visualize_validation_batch(
                    unwrapped_model, train_ds, device,
                    out_dir=os.path.join(vis_step_dir, "train"),
                    preference_keys=preference_keys,
                    max_samples=args.max_vis_samples,
                    step=global_step,
                )
                top_bottom_train_mp4s, uniform_train_mp4s = visualize_top_bottom_trajectories(
                    unwrapped_model, train_ds, device,
                    out_dir=os.path.join(vis_step_dir, "top_bottom_train"),
                    preference_keys=preference_keys,
                    n=5, n_uniform=10,
                    step=global_step,
                )
                top_bottom_val_mp4s, uniform_val_mp4s = visualize_top_bottom_trajectories(
                    unwrapped_model, val_ds, device,
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
                        **{f"val/acc_axis/{ax}": float(v) for ax, v in val_axis_acc_vis.items()},
                        **{f"val/video_{i}": wandb.Video(os.path.join(vis_step_dir, "val", f), format="mp4") for i, f in enumerate(val_mp4s)},
                        **{f"train/video_{i}": wandb.Video(os.path.join(vis_step_dir, "train", f), format="mp4") for i, f in enumerate(train_mp4s)},
                        **{f"top_bottom_train/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, top_bottom_train_mp4s)},
                        **{f"top_bottom_val/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, top_bottom_val_mp4s)},
                        **{f"uniform_train/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, uniform_train_mp4s)},
                        **{f"uniform_val/{k}": wandb.Video(p, format="mp4") for k, p in zip(preference_keys, uniform_val_mp4s)},
                    }, step=global_step)

            # ---- small visualization (pair plots only, no ranking; rank 0 only) ----
            if is_main and global_step % args.small_vis_interval == 0:
                small_vis_dir = os.path.join(vis_dir, f"step{global_step:06d}_small")
                n_vis_val = visualize_validation_batch(
                    unwrapped_model, val_ds, device,
                    out_dir=os.path.join(small_vis_dir, "val"),
                    preference_keys=preference_keys,
                    max_samples=args.small_vis_max_samples,
                    step=global_step,
                )
                n_vis_train = visualize_validation_batch(
                    unwrapped_model, train_ds, device,
                    out_dir=os.path.join(small_vis_dir, "train"),
                    preference_keys=preference_keys,
                    max_samples=args.small_vis_max_samples,
                    step=global_step,
                )
                print(f"  [SmallVis] Saved {n_vis_val} val + {n_vis_train} train → {small_vis_dir}")
                if use_wandb:
                    val_mp4s = sorted(f for f in os.listdir(os.path.join(small_vis_dir, "val")) if f.endswith(".mp4"))
                    train_mp4s = sorted(f for f in os.listdir(os.path.join(small_vis_dir, "train")) if f.endswith(".mp4"))
                    wandb.log({
                        **{f"small_val/video_{i}": wandb.Video(os.path.join(small_vis_dir, "val", f), format="mp4") for i, f in enumerate(val_mp4s)},
                        **{f"small_train/video_{i}": wandb.Video(os.path.join(small_vis_dir, "train", f), format="mp4") for i, f in enumerate(train_mp4s)},
                    }, step=global_step)

            # ---- checkpoint (rank 0 only) ----
            if is_main and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(ckpt_dir, f"step{global_step:06d}.pt")
                model_sd = unwrapped_model.get_checkpoint_state_dict() if isinstance(unwrapped_model, QwenRewardModel) else unwrapped_model.state_dict()
                torch.save({
                    "step": global_step,
                    "model": model_sd,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                print(f"  [Ckpt] Saved → {ckpt_path}")

    # ---- final eval + curves (rank 0 only) ----
    if is_main:
        val_loss, val_acc, val_axis_acc = evaluate(unwrapped_model, val_loader, device, args.equal_weight, len(preference_keys), getattr(args, "action_weight", 0.0), getattr(args, "regression", False))
        print("\nFinal validation:")
        for k, v in zip(preference_keys, val_acc):
            print(f"  {k}: {v:.3f}")
        if val_axis_acc:
            print("Final validation per-axis:")
            for ax in sorted(val_axis_acc):
                print(f"  {ax}: {val_axis_acc[ax]:.3f}")

        plot_training_curves(
            train_losses, train_accs, val_losses, val_accs,
            out_path=os.path.join(args.out_dir, "training_curves.png"),
            preference_keys=preference_keys,
        )

        # Final visualizations
        visualize_validation_batch(
            unwrapped_model, val_ds, device,
            out_dir=os.path.join(vis_dir, "final", "val"),
            preference_keys=preference_keys,
            max_samples=len(val_ds),
            step=global_step,
        )
        visualize_validation_batch(
            unwrapped_model, train_ds, device,
            out_dir=os.path.join(vis_dir, "final", "train"),
            preference_keys=preference_keys,
            max_samples=args.max_vis_samples,
            step=global_step,
        )

        # Final checkpoint
        model_sd = unwrapped_model.get_checkpoint_state_dict() if isinstance(unwrapped_model, QwenRewardModel) else unwrapped_model.state_dict()
        torch.save({
            "step": global_step,
            "model": model_sd,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }, os.path.join(ckpt_dir, "final.pt"))
        if use_wandb:
            wandb.finish()
        print(f"\nDone. Outputs in {args.out_dir}")

    if is_ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
