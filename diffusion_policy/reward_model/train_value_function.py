"""
Train a state-space value function V(s) on demos+rollouts and emit per-step
advantages as the conditioning signal (a simplification of the RECAP setup).

Per-step reward used to build V targets:
  r_t = 0                if t = T_end AND trajectory succeeded
  r_t = fail_penalty     if t = T_end AND trajectory failed
  r_t = -1               otherwise
V(s_t) = sum_{k>=t} r_k (no discount), so:
  - success traj ending at T_s: V[t] = -(T_s - t), V[T_s] = 0
  - failure traj ending at T_e: V[t] = -(T_e - t) + fail_penalty, V[T_e] = fail_penalty

After training the MLP V_hat, per-step advantage A_t = V_hat(s_{t+1}) - V_hat(s_t)
is computed for every (episode, step) and linearly mapped to [-1, 1] using the
global min/max so that test-time conditioning at +1 corresponds to the most
"success-progressing" step and -1 to the worst (failure-ending).
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.chdir(ROOT_DIR)

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import click
import torch
import torch.nn as nn
import numpy as np
import h5py
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from reward_functions import (_has_pickplace_obs, _object_in_bin,
                              peg_reward, order_reward, _drop_value,
                              PICKPLACE_CANONICAL_ORDER)


# ---------------------------------------------------------------------------
# Success-step detection
# ---------------------------------------------------------------------------
def find_success_step_pickplace_strict(obs, actions, n_active_objects):
    """First step where strict success holds — matches the strict_success
    eval metric:
      - every active object is in its bin at step t
      - order_reward(prefix) == +1 (placed in canonical right-first order)
      - per-object _drop_value(prefix) > 0 (soft release for each active obj)

    Returns None if the trajectory never reaches strict success.
    """
    if not _has_pickplace_obs(obs):
        return None
    active_ids = PICKPLACE_CANONICAL_ORDER[:max(1, min(int(n_active_objects), 4))]
    EPS = 1e-6
    for t in range(len(obs)):
        # Cheap per-step gate: skip until all objects sit in bins.
        if not all(_object_in_bin(obs[t], i) for i in active_ids):
            continue
        prefix_obs = obs[:t + 1]
        prefix_act = actions[:t + 1] if actions is not None else None
        if order_reward(prefix_obs) < 1.0 - EPS:
            continue
        drops_ok = True
        for i in active_ids:
            if _drop_value(prefix_obs, i, actions=prefix_act) <= 0:
                drops_ok = False
                break
        if drops_ok:
            return t
    return None


def find_success_step_slow_fast_right(obs):
    """First step where the nut lands on the RIGHT peg (peg_reward == +1).
    Left-peg landings do NOT count as success — they yield failure targets.
    """
    if obs is None or len(obs) == 0:
        return None
    for t in range(len(obs)):
        try:
            if peg_reward(obs[:t + 1]) >= 1.0 - 1e-6:
                return t
        except Exception:
            continue
    return None


def find_success_step(obs, actions, task, n_active_objects):
    if task == 'pickplace':
        return find_success_step_pickplace_strict(obs, actions, n_active_objects)
    elif task == 'slow_fast':
        return find_success_step_slow_fast_right(obs)
    raise ValueError(f"Unknown task '{task}', expected 'pickplace' or 'slow_fast'")


# ---------------------------------------------------------------------------
# Value targets per step
# ---------------------------------------------------------------------------
def value_targets_for_episode(obs_ep, act_ep, task, n_active_objects, fail_penalty):
    """Returns (target_V: (L,), success_t: int or -1, used_len: int).

    Trajectory is truncated at the success step (if any) — states after
    success contribute nothing useful to the value function and would push V
    toward 0 for all post-success steps regardless of the policy's behavior.
    """
    L = len(obs_ep)
    if L == 0:
        return np.zeros(0, dtype=np.float32), -1, 0
    success_t = find_success_step(obs_ep, act_ep, task, n_active_objects)
    if success_t is not None:
        used_len = success_t + 1
        targets = np.zeros(used_len, dtype=np.float32)
        targets[success_t] = 0.0
        for t in range(success_t - 1, -1, -1):
            targets[t] = targets[t + 1] - 1.0
        return targets, success_t, used_len
    # Failure: include the full trajectory with R_fail at the end.
    used_len = L
    targets = np.zeros(used_len, dtype=np.float32)
    targets[-1] = fail_penalty
    for t in range(used_len - 2, -1, -1):
        targets[t] = targets[t + 1] - 1.0
    return targets, -1, used_len


# ---------------------------------------------------------------------------
# Demo loader (mirrors train_reward_model.load_demo_obs)
# ---------------------------------------------------------------------------
def load_demo_obs(demo_hdf5, obs_keys, max_demos=None):
    episodes = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos = f['data']
        n_demos = len(demos)
        if max_demos is not None:
            n_demos = min(n_demos, max_demos)
        for i in range(n_demos):
            demo = demos[f'demo_{i}']
            obs_parts = [demo['obs'][key][:].astype(np.float32) for key in obs_keys]
            obs = np.concatenate(obs_parts, axis=-1)
            episodes.append(obs)
    return episodes


# ---------------------------------------------------------------------------
# Value function model (small MLP regressor)
# ---------------------------------------------------------------------------
class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ValueDataset(Dataset):
    def __init__(self, obs_flat, target_flat):
        self.obs = obs_flat.astype(np.float32)
        self.target = target_flat.astype(np.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return self.obs[i], self.target[i]


@click.command()
@click.option('--rollout_data', required=True, help='Comma-separated .npz from collect_rollouts.py')
@click.option('--demo_hdf5', required=True)
@click.option('--output_dir', required=True)
@click.option('--task', type=click.Choice(['pickplace', 'slow_fast']), required=True)
@click.option('--n_active_objects', default=2, type=int, help='Pickplace: number of active objects whose joint placement counts as success.')
@click.option('--obs_keys', default='object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos')
@click.option('--epochs', default=100, type=int)
@click.option('--batch_size', default=512, type=int)
@click.option('--lr', default=1e-3, type=float)
@click.option('--hidden_dim', default=256, type=int)
@click.option('--n_layers', default=3, type=int)
@click.option('--fail_penalty', default=-100.0, type=float, help='r_T applied at the last step of failure trajectories.')
@click.option('--max_demos', default=None, type=int)
@click.option('--device', default='cuda:0')
@click.option('--wandb_project', default='reward_cond_pipeline')
# Ignored for parity with train_reward_model CLI surface (so pipeline scripts
# can call either trainer with the same flag set).
@click.option('--n_pairs', default=None, type=int, hidden=True)
@click.option('--reward_axes', default=None, hidden=True)
@click.option('--stride', default=1, type=int, hidden=True)
@click.option('--max_seq_len', default=512, type=int, hidden=True)
def main(rollout_data, demo_hdf5, output_dir, task, n_active_objects, obs_keys,
         epochs, batch_size, lr, hidden_dim, n_layers, fail_penalty, max_demos,
         device, wandb_project,
         n_pairs, reward_axes, stride, max_seq_len):
    os.makedirs(output_dir, exist_ok=True)
    obs_keys = obs_keys.split(',')
    device = torch.device(device)

    wandb.init(
        project=wandb_project,
        name='phase3_value_function',
        config={
            'task': task,
            'n_active_objects': n_active_objects,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'fail_penalty': fail_penalty,
        },
    )

    # ---------------- Load rollouts ----------------
    rollout_paths = [p.strip() for p in rollout_data.split(',')
                     if p.strip() and p.strip() != 'none']
    has_rollouts = len(rollout_paths) > 0
    rollout_obs_list = []   # list of (T_i, D) per episode
    rollout_act_list = []   # list of (T_i, A) per episode (or None)
    if has_rollouts:
        for rpath in rollout_paths:
            data = np.load(rpath)
            n = len(data['episode_lengths'])
            obs_arr = data['obs'][:n]
            act_arr = data['actions'][:n] if 'actions' in data.files else None
            lengths = data['episode_lengths'][:n]
            for i in range(n):
                L = int(lengths[i])
                rollout_obs_list.append(obs_arr[i, :L].astype(np.float32))
                if act_arr is not None:
                    L_act = min(L, act_arr.shape[1])
                    rollout_act_list.append(act_arr[i, :L_act].astype(np.float32))
                else:
                    rollout_act_list.append(None)
            print(f"  Loaded {n} rollouts from {rpath}")
    n_rollouts = len(rollout_obs_list)
    print(f"Total rollouts: {n_rollouts}")

    # ---------------- Load demos ----------------
    demo_obs_list = load_demo_obs(demo_hdf5, obs_keys, max_demos=max_demos)
    n_demos = len(demo_obs_list)
    # Actions for demos — needed for pickplace strict-success drop check.
    demo_act_list = []
    with h5py.File(demo_hdf5, 'r') as f:
        demos = f['data']
        for i in range(n_demos):
            demo_act_list.append(demos[f'demo_{i}']['actions'][:].astype(np.float32))
    print(f"Total demos: {n_demos}")

    all_obs_list = rollout_obs_list + demo_obs_list
    all_act_list = rollout_act_list + demo_act_list
    N = len(all_obs_list)
    if N == 0:
        raise ValueError("No data (rollouts + demos both empty).")
    obs_dim = all_obs_list[0].shape[-1]

    # ---------------- Build per-step targets ----------------
    success_flags = np.zeros(N, dtype=np.int8)
    success_steps = np.full(N, -1, dtype=np.int32)
    used_lengths = np.zeros(N, dtype=np.int32)
    targets_per_ep = [None] * N
    for i, obs_ep in enumerate(all_obs_list):
        act_ep = all_act_list[i] if i < len(all_act_list) else None
        targets, succ_t, used_L = value_targets_for_episode(
            obs_ep, act_ep, task, n_active_objects, fail_penalty)
        targets_per_ep[i] = targets
        success_steps[i] = succ_t
        used_lengths[i] = used_L
        success_flags[i] = 1 if succ_t >= 0 else 0

    n_success = int(success_flags.sum())
    print(f"Success trajectories: {n_success}/{N} "
          f"({100.0 * n_success / max(N, 1):.1f}%)")
    succ_lens = used_lengths[success_flags == 1]
    fail_lens = used_lengths[success_flags == 0]
    if len(succ_lens):
        print(f"  Success time-to-go (V(s_0)=-T_s): mean={succ_lens.mean():.1f} "
              f"min={succ_lens.min()} max={succ_lens.max()}")
    if len(fail_lens):
        print(f"  Failure traj length:                  mean={fail_lens.mean():.1f} "
              f"min={fail_lens.min()} max={fail_lens.max()}")
    wandb.log({
        'value/n_episodes': N,
        'value/n_success': n_success,
        'value/success_rate': n_success / max(N, 1),
    })

    # Flatten (obs_t, target_V_t) pairs for training
    flat_obs = np.concatenate(
        [all_obs_list[i][:used_lengths[i]] for i in range(N)], axis=0)
    flat_target = np.concatenate(targets_per_ep, axis=0)
    print(f"Training pool: {len(flat_obs)} (state, target) pairs")

    # Train/val split at the episode level so val episodes are unseen
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_val = max(1, int(0.1 * N))
    val_ids = set(perm[:n_val].tolist())
    train_ids = [i for i in range(N) if i not in val_ids]
    val_ids_sorted = sorted(val_ids)
    train_obs = np.concatenate(
        [all_obs_list[i][:used_lengths[i]] for i in train_ids], axis=0)
    train_target = np.concatenate([targets_per_ep[i] for i in train_ids], axis=0)
    val_obs = np.concatenate(
        [all_obs_list[i][:used_lengths[i]] for i in val_ids_sorted], axis=0)
    val_target = np.concatenate([targets_per_ep[i] for i in val_ids_sorted], axis=0)
    print(f"  Train pairs: {len(train_obs)}  Val pairs: {len(val_obs)}")

    # Z-score normalize targets — raw values can span ~[-600, 0] when
    # fail_penalty=-100 and trajectories are ~500 steps. Without this the
    # failure tail dominates the MSE and success trajectories are underfit.
    # The mean cancels out when we compute A_t = V(s_{t+1}) - V(s_t), so only
    # the scale matters for downstream advantages.
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    if target_std < 1e-6:
        target_std = 1.0
    print(f"Target stats — mean={target_mean:.2f}  std={target_std:.2f}  "
          f"min={float(train_target.min()):.2f}  max={float(train_target.max()):.2f}")
    train_target_n = (train_target - target_mean) / target_std
    val_target_n = (val_target - target_mean) / target_std
    wandb.log({
        'value/target_mean': target_mean,
        'value/target_std': target_std,
        'value/target_min': float(train_target.min()),
        'value/target_max': float(train_target.max()),
    })

    train_ds = ValueDataset(train_obs, train_target_n)
    val_ds = ValueDataset(val_obs, val_target_n)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---------------- Model ----------------
    model = ValueFunction(obs_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"Training V(s) MLP for {epochs} epochs...")
    for ep in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        model.eval()
        val_running, n_val_batches = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                val_running += loss_fn(model(x), y).item()
                n_val_batches += 1
        val_loss = val_running / max(n_val_batches, 1)
        if ep % max(1, epochs // 20) == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1}/{epochs}  train_mse={train_loss:.3f}  val_mse={val_loss:.3f}")
        wandb.log({
            'value/train_mse': train_loss,
            'value/val_mse': val_loss,
            'value/epoch': ep,
        })

    # ---------------- Score every (episode, step) ----------------
    # Denormalize V back into the original units (the mean cancels in
    # A_t = V(s_{t+1}) - V(s_t), so technically only the std multiplication
    # is necessary — but denormalizing keeps the saved V values interpretable
    # as "time-to-go" for the diagnostics plot).
    model.eval()
    v_per_ep = []   # list of (L_i_full,) np arrays — V_hat over the entire raw episode
    with torch.no_grad():
        for obs_ep in all_obs_list:
            x = torch.from_numpy(obs_ep.astype(np.float32)).to(device)
            v_norm = model(x).cpu().numpy()
            v = v_norm * target_std + target_mean
            v_per_ep.append(v.astype(np.float32))

    # Per-step advantages A_t = V(s_{t+1}) - V(s_t); pad the final step with the
    # previous advantage (or 0 if L==1).
    adv_per_ep = []
    for v in v_per_ep:
        if len(v) <= 1:
            adv_per_ep.append(np.zeros(len(v), dtype=np.float32))
            continue
        a = np.diff(v).astype(np.float32)              # (L-1,)
        adv_per_ep.append(np.concatenate([a, a[-1:]], axis=0))   # (L,)

    flat_adv = np.concatenate(adv_per_ep, axis=0)
    score_min = float(flat_adv.min())
    score_max = float(flat_adv.max())
    score_range = max(score_max - score_min, 1e-8)
    print(f"Advantage stats: min={score_min:.3f} max={score_max:.3f}")

    def normalize(arr):
        return (2.0 * (arr - score_min) / score_range - 1.0).astype(np.float32)

    rollout_adv_perstep = [normalize(adv_per_ep[i]).tolist() for i in range(n_rollouts)]
    demo_adv_perstep = [normalize(adv_per_ep[n_rollouts + i]).tolist()
                        for i in range(n_demos)]

    # Per-episode mean for backward-compat field (dataset falls back if it
    # can't find the per-step entries).
    rollout_mean = np.array([
        float(np.mean(normalize(adv_per_ep[i]))) for i in range(n_rollouts)
    ], dtype=np.float32).reshape(-1, 1)
    demo_mean = np.array([
        float(np.mean(normalize(adv_per_ep[n_rollouts + i]))) for i in range(n_demos)
    ], dtype=np.float32).reshape(-1, 1)

    scores = {
        'reward_names': ['advantage'],
        'score_min': [score_min],
        'score_max': [score_max],
        'fail_penalty': fail_penalty,
        'task': task,
        'value_function': True,
        'target_mean': target_mean,
        'target_std': target_std,
        'rollout_scores_zscore_perstep': rollout_adv_perstep,
        'demo_scores_zscore_perstep': demo_adv_perstep,
        'rollout_scores_zscore': rollout_mean.tolist(),
        'demo_scores_zscore': demo_mean.tolist(),
        'rollout_scores_raw': rollout_mean.tolist(),
        'demo_scores_raw': demo_mean.tolist(),
        'n_rollouts': n_rollouts,
        'n_demos': n_demos,
        'success_rate': float(n_success / max(N, 1)),
    }
    scores_path = os.path.join(output_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f)
    print(f"Saved scores to {scores_path}")

    # ---------------- Diagnostic plots ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(flat_adv, bins=60, edgecolor='black')
    axes[0].set_title('Per-step advantages (raw)')
    axes[0].set_xlabel('A_t')
    v_per_ep_start = np.array([v[0] for v in v_per_ep], dtype=np.float32)
    axes[1].hist(v_per_ep_start[success_flags == 1], bins=30, alpha=0.6,
                 label='success', edgecolor='black')
    axes[1].hist(v_per_ep_start[success_flags == 0], bins=30, alpha=0.6,
                 label='failure', edgecolor='black')
    axes[1].set_title('V(s_0) by outcome')
    axes[1].set_xlabel('V(s_0)')
    axes[1].legend()
    flat_norm = normalize(flat_adv)
    axes[2].hist(flat_norm, bins=60, edgecolor='black')
    axes[2].set_title('Per-step advantage (normalized to [-1, 1])')
    axes[2].set_xlabel('z')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, 'value_diagnostics.png')
    fig.savefig(plot_path, dpi=120)
    wandb.log({'value/diagnostics': wandb.Image(fig)})
    plt.close(fig)
    print(f"Saved diagnostics to {plot_path}")

    wandb.summary['value/score_min'] = score_min
    wandb.summary['value/score_max'] = score_max
    wandb.summary['value/success_rate'] = float(n_success / max(N, 1))
    wandb.finish()


if __name__ == '__main__':
    main()
