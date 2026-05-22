"""
Per-axis reward functions for the PickPlace + slow_fast benchmarks.

Each function takes a trajectory and returns one scalar reward for that
trajectory. They are the single source of truth for axis values — the
reward-model dataset builder calls them on every trajectory (demo or
rollout) instead of reading per-episode metadata from HDF5/npz attrs.

Function signature
------------------
    fn(obs, actions=None, reward_series=None) -> float

where:
- obs:           (T, D) numpy array of observations, trimmed to the real
                 episode length (caller is responsible for this — these
                 functions do NOT see padded data).
- actions:       (T, A) numpy array of actions, similarly trimmed (optional;
                 needed for smoothness).
- reward_series: (T,) per-step env reward (optional; only used by `success`
                 as a peg-task fallback).

Obs schema
----------
PickPlace obs (65 dims):
    [0..55]: object info — 14 dims × 4 objects in order [Milk, Bread, Cereal, Can]
              each object: obj_pos(3) obj_quat(4) obj_to_eef_pos(3) obj_to_eef_quat(4)
    [56..58]: robot0_eef_pos
    [59..62]: robot0_eef_quat
    [63..64]: robot0_gripper_qpos

Peg/square obs (23 dims):
    [0..2]:  nut_pos (world)
    [3..13]: rest of `object`
    [14..]:  robot state

Bin regions match robosuite's PickPlace `not_in_bin` check exactly:
bin2_pos=(0.1, 0.28, 0.8), bin_size=(0.39, 0.49, 0.82). z window = bin2_pos[2] .. bin2_pos[2]+0.1.
"""

import numpy as np


# ---------- shared constants ----------
PICKPLACE_OBJ_NAMES = ['Milk', 'Bread', 'Cereal', 'Can']
PICKPLACE_CANONICAL_ORDER = [1, 3, 0, 2]   # right-first

# bin_id → (x_lo, x_hi, y_lo, y_hi) for the target bin in bin2 (world coords).
PICKPLACE_BIN_BOUNDS = {
    0: (-0.095, 0.1,    0.035, 0.28),    # Milk   (x-low,  y-low)
    1: ( 0.1,   0.295,  0.035, 0.28),    # Bread  (x-high, y-low)
    2: (-0.095, 0.1,    0.28,  0.525),   # Cereal (x-low,  y-high)
    3: ( 0.1,   0.295,  0.28,  0.525),   # Can    (x-high, y-high)
}
PICKPLACE_BIN_Z_LO = 0.8
PICKPLACE_BIN_Z_HI = 0.9

# Drop-vs-careful threshold for the eef z at the moment the gripper opens
# above an object's target bin. Tracking lag in the OSC controller means
# "careful intent" (commanded z ~ 0.84) shows up as observed eef z ~ 0.90,
# while "drop intent" (commanded ~ 0.95+) shows up as observed ~ 0.96+.
# A threshold of 0.93 separates the two cleanly.
DROP_HEIGHT_THRESHOLD = 0.93


def _obj_pos(obs_t, obj_id):
    """3-vector of object obj_id's world position at one timestep."""
    base = obj_id * 14
    return obs_t[base:base + 3]


def _eef_pos(obs_t):
    return obs_t[56:59]


def _has_pickplace_obs(obs):
    """True if the obs schema looks like PickPlace (≥65 dims: 4 objects × 14)."""
    return obs is not None and obs.ndim >= 2 and obs.shape[-1] >= 65


def _object_in_bin(obs_t, obj_id):
    p = _obj_pos(obs_t, obj_id)
    if p.shape[0] < 3:
        return False
    x_lo, x_hi, y_lo, y_hi = PICKPLACE_BIN_BOUNDS[obj_id]
    return (x_lo < float(p[0]) < x_hi
            and y_lo < float(p[1]) < y_hi
            and PICKPLACE_BIN_Z_LO < float(p[2]) < PICKPLACE_BIN_Z_HI)


def _first_lift_times(obs):
    """For each of the 4 PickPlace objects, the first step where its z rises
    >5 cm above its starting z. Returns a list of length 4; len(obs) if the
    object was never lifted.

    Objects parked far off-table by robosuite's `clear_objects` (e.g. xy ~= 10)
    are ignored — they're not in the scene for this task so their z drift is
    not a meaningful "lift" event.
    """
    T = len(obs)
    times = [T] * 4
    for obj_id in range(4):
        x0 = float(obs[0, obj_id * 14])
        y0 = float(obs[0, obj_id * 14 + 1])
        # Sanity check: bin1 + workspace is roughly x∈[-0.5, 0.5], y∈[-0.7, 0.7].
        # Anything outside that is a parked/cleared object — skip it.
        if abs(x0) > 1.0 or abs(y0) > 1.0:
            continue
        z = obs[:, obj_id * 14 + 2]
        z0 = float(z[0])
        ups = np.where(z > z0 + 0.05)[0]
        if len(ups) > 0:
            times[obj_id] = int(ups[0])
    return times


def _gripper_width(obs_t):
    return float(obs_t[63] - obs_t[64])


def _release_eef_z(obs, obj_id, actions=None):
    """Gripper z at the moment this object's body first enters its bin region.

    For careful releases the gripper has lowered into ~0.84-0.91 by then;
    for drop releases the gripper is still hovering above 0.93+. Combined
    with the DROP_HEIGHT_THRESHOLD this cleanly separates the two modes
    even when the demo terminates before the gripper-open command fires
    (last few settle steps may keep the gripper closed).
    """
    if obs is None or len(obs) == 0:
        return None
    for t in range(len(obs)):
        if _object_in_bin(obs[t], obj_id):
            return float(_eef_pos(obs[t])[2])
    return None


# ----------------------------------------------------------------------------
# Per-axis reward functions
# ----------------------------------------------------------------------------


def success(obs, actions=None, reward_series=None):
    """1.0 if any goal condition is met:
      - PickPlace (obs ≥ 65 dims): any object in its bin at episode end, OR
      - Peg / square_twopeg (obs ≥ 23 dims): nut ends on either peg, OR
      - reward_series ever ≥ 1 (env-reward fallback).
    """
    if obs is None or len(obs) == 0:
        return 0.0
    # PickPlace check
    if _has_pickplace_obs(obs):
        if any(_object_in_bin(obs[-1], i) for i in range(4)):
            return 1.0
    # Peg-task check (nut on either peg).
    if peg_reward(obs) != 0.0:
        return 1.0
    if reward_series is not None and len(reward_series) > 0:
        return 1.0 if float(np.max(reward_series)) >= 1.0 else 0.0
    return 0.0


def speed_reward(obs, actions=None, reward_series=None, max_steps=2000):
    """Linear in episode length: 1.0 fast, 0.1 at max_steps. 0 if the
    trajectory didn't succeed."""
    L = len(obs) if obs is not None else 0
    base = 1.0 - 0.9 * (L / max_steps)
    return float(base) if success(obs, reward_series=reward_series) >= 1.0 else 0.0


def smoothness(obs, actions=None, reward_series=None):
    """exp(-10 * mean_jerk) over actions, gated by success."""
    if actions is None or len(actions) < 4:
        s = 1.0
    else:
        jerk = np.diff(actions, n=3, axis=0)
        jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=-1)))
        s = float(np.exp(-10.0 * jerk_mag))
    if success(obs, reward_series=reward_series) < 1.0:
        s = 0.0
    return s


def _placed(obs, obj_id):
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    return 1.0 if _object_in_bin(obs[-1], obj_id) else 0.0


def milk_placed(obs, actions=None, reward_series=None):    return _placed(obs, 0)
def bread_placed(obs, actions=None, reward_series=None):   return _placed(obs, 1)
def cereal_placed(obs, actions=None, reward_series=None):  return _placed(obs, 2)
def can_placed(obs, actions=None, reward_series=None):     return _placed(obs, 3)


def order_reward(obs, actions=None, reward_series=None):
    """PickPlace-only axis. Returns 0 if the obs schema isn't PickPlace.

    +1 if objects were placed in the canonical right-first order
    (Bread → Can → Milk → Cereal, restricted to the active subset),
    -1 if the reversed-of-active order, 0 otherwise.

    Active objects are inferred from initial position: anything not parked
    off-table (|x|<1 and |y|<1 at t=0). For PickPlace_2 the active subset is
    [Bread, Can], so canonical = [1, 3] and reversed = [3, 1]; the 1-object
    case correctly resolves Bread→+1 and Can→-1.

    Detection event is object first-time-in-bin (placement), not first-lift —
    placement is the definitive "did this object actually get put in its bin".
    """
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    T = len(obs)
    # Determine active object subset from initial positions.
    active_ids = []
    for obj_id in range(4):
        x0 = float(obs[0, obj_id * 14])
        y0 = float(obs[0, obj_id * 14 + 1])
        if abs(x0) <= 1.0 and abs(y0) <= 1.0:
            active_ids.append(obj_id)
    if not active_ids:
        return 0.0
    # active_canonical: canonical right-first order restricted to active set.
    active_canonical = [i for i in PICKPLACE_CANONICAL_ORDER if i in active_ids]
    active_reversed = list(reversed(active_canonical))

    # First-placement timestep for each active object (T if never placed).
    first_place = {obj_id: T for obj_id in active_ids}
    for obj_id in active_ids:
        for t in range(T):
            if _object_in_bin(obs[t], obj_id):
                first_place[obj_id] = t
                break
    placed_in_order = [i for i in sorted(active_ids, key=lambda x: first_place[x])
                       if first_place[i] < T]
    if len(placed_in_order) == 0:
        return 0.0
    canonical_prefix = active_canonical[:len(placed_in_order)]
    reversed_prefix = active_reversed[:len(placed_in_order)]
    if placed_in_order == canonical_prefix:
        return 1.0
    if placed_in_order == reversed_prefix:
        return -1.0
    return 0.0


def _drop_value(obs, obj_id, actions=None):
    """+1 careful, -1 drop, 0 if the object never lands in its bin.
    PickPlace-only — returns 0 if the obs schema isn't PickPlace."""
    if obs is None or len(obs) == 0 or not _has_pickplace_obs(obs):
        return 0.0
    rz = _release_eef_z(obs, obj_id, actions=actions)
    if rz is None:
        return 0.0
    return 1.0 if rz < DROP_HEIGHT_THRESHOLD else -1.0


def milk_drop(obs, actions=None, reward_series=None):    return _drop_value(obs, 0, actions=actions)
def bread_drop(obs, actions=None, reward_series=None):   return _drop_value(obs, 1, actions=actions)
def cereal_drop(obs, actions=None, reward_series=None):  return _drop_value(obs, 2, actions=actions)
def can_drop(obs, actions=None, reward_series=None):     return _drop_value(obs, 3, actions=actions)


def drop_reward(obs, actions=None, reward_series=None):
    """Signed fraction-careful over placed objects, in [-1, +1].
    All careful → +1, all drop → -1, no placements → 0."""
    vals = [_drop_value(obs, i, actions=actions) for i in range(4)]
    n_careful = sum(1 for v in vals if v > 0)
    n_drop = sum(1 for v in vals if v < 0)
    n_placed = n_careful + n_drop
    if n_placed == 0:
        return 0.0
    return (n_careful - n_drop) / n_placed


def peg_reward(obs, actions=None, reward_series=None):
    """+1 if the nut ends on the right peg (y≈-0.1), -1 left (y≈+0.1),
    0 neither. Square/peg obs layout: nut_pos is at the first 3 dims of `obs`.
    The nut sits at z≈0.89–0.92 when on a peg (peg tops at ~0.95)."""
    if obs is None or len(obs) == 0:
        return 0.0
    nut_pos = obs[-1, :3]
    # Reasonable z window for "nut on a peg" — anywhere between the table
    # surface and a few cm above the peg top.
    on_table_or_peg = 0.85 < float(nut_pos[2]) < 0.95
    if abs(float(nut_pos[0]) - 0.23) < 0.04 and on_table_or_peg:
        if abs(float(nut_pos[1]) - 0.1) < 0.04:
            return -1.0   # left peg
        if abs(float(nut_pos[1]) - (-0.1)) < 0.04:
            return 1.0    # right peg
    return 0.0


# ----------------------------------------------------------------------------
# Registry: axis name -> function. Add new axes by adding a function and an
# entry here.
# ----------------------------------------------------------------------------
AXIS_FUNCTIONS = {
    'success':        success,
    'speed_reward':   speed_reward,
    'smoothness':     smoothness,
    'order_reward':   order_reward,
    'milk_placed':    milk_placed,
    'bread_placed':   bread_placed,
    'cereal_placed':  cereal_placed,
    'can_placed':     can_placed,
    'milk_drop':      milk_drop,
    'bread_drop':     bread_drop,
    'cereal_drop':    cereal_drop,
    'can_drop':       can_drop,
    'drop_reward':    drop_reward,
    'peg_reward':     peg_reward,
}


def compute_axes(axis_names, obs, actions=None, reward_series=None):
    """Compute a dict {axis_name: value} for one trajectory."""
    out = {}
    for name in axis_names:
        fn = AXIS_FUNCTIONS.get(name)
        if fn is None:
            raise KeyError(f"Unknown reward axis '{name}'. Known: {list(AXIS_FUNCTIONS.keys())}")
        out[name] = float(fn(obs, actions=actions, reward_series=reward_series))
    return out


# ----------------------------------------------------------------------------
# PickPlace per-axis eval-logging helpers
# ----------------------------------------------------------------------------
# Object id → name (matches the slot order in the 56-dim 'object' obs block).
PICKPLACE_OBJECT_NAMES = {0: 'milk', 1: 'bread', 2: 'cereal', 3: 'can'}


def get_pickplace_eval_axes(n_active_objects):
    """Canonical per-axis eval-logging set for a PickPlace variant:
    order_reward + per-active-object placed + per-active-object drop. Active
    subset is the right-first canonical prefix (Bread, Can, Milk, Cereal).
    """
    n = max(1, min(int(n_active_objects), 4))
    active_ids = PICKPLACE_CANONICAL_ORDER[:n]
    axes = ['order_reward']
    axes += [f'{PICKPLACE_OBJECT_NAMES[i]}_placed' for i in active_ids]
    axes += [f'{PICKPLACE_OBJECT_NAMES[i]}_drop' for i in active_ids]
    return axes


def compute_pickplace_eval_log(obs_seqs, action_seqs, prefixes,
                               n_active_objects, axis_names=None):
    """Aggregate per-axis reward values + strict-success across rollouts.

    Args:
      obs_seqs:    list of (T_i, D) np.ndarray per rollout
      action_seqs: list of (T_i, A) np.ndarray per rollout
      prefixes:    list of str per rollout (e.g. 'test/', 'train/')
      n_active_objects: used for strict_success bookkeeping
      axis_names: list of base axes to log. If None, derived from
                  get_pickplace_eval_axes(n_active_objects).

    Returns dict {prefix + key: mean_value} with one entry per (prefix, axis)
    plus '{prefix}mean_strict_success' (positive order AND every per-object
    *_drop axis > 0) when the relevant axes are present.
    """
    import collections
    if axis_names is None:
        axis_names = get_pickplace_eval_axes(n_active_objects)

    axis_accum = {ax: collections.defaultdict(list) for ax in axis_names}
    strict_drop_axes = [ax for ax in axis_names
                        if ax.endswith('_drop') and ax != 'drop_reward']
    strict_axes_available = ('order_reward' in axis_names
                             and len(strict_drop_axes) > 0)
    prefix_strict = collections.defaultdict(list)

    for obs_seq, act_seq, prefix in zip(obs_seqs, action_seqs, prefixes):
        if obs_seq is None or len(obs_seq) == 0:
            continue
        rollout_vals = {}
        for ax in axis_names:
            fn = AXIS_FUNCTIONS.get(ax)
            if fn is None:
                continue
            try:
                v = float(fn(obs_seq, actions=act_seq))
            except Exception:
                v = 0.0
            axis_accum[ax][prefix].append(v)
            rollout_vals[ax] = v
        if strict_axes_available:
            order_ok = rollout_vals.get('order_reward', 0.0) >= 1.0 - 1e-6
            drops_ok = all(rollout_vals.get(ax, 0.0) > 0
                           for ax in strict_drop_axes)
            prefix_strict[prefix].append(float(order_ok and drops_ok))

    log_data = {}
    for ax, by_prefix in axis_accum.items():
        for prefix, vals in by_prefix.items():
            if vals:
                log_data[prefix + ax] = float(np.mean(vals))
    for prefix, vals in prefix_strict.items():
        if vals:
            log_data[prefix + 'mean_strict_success'] = float(np.mean(vals))
    return log_data
