"""Scatter plot: demos + FPL + Single, by peg side.

x-axis: peg (left / right)
y-axis: throughput = 300 / first_success_step

Three data sources:
  - Demos:  shared_data_slow_fast_final_slower/scripted_data/demos.hdf5
  - FPL:    pipeline_output_slow_fast_final_rhp_slower/eval/per_rollout_steps.jsonl
  - Single: pipeline_output_slow_fast_final_single_pref_slower/eval/per_rollout_steps.jsonl

Only the z_pos (right-peg conditioning) eval rollouts are plotted; z_neg / z_zero
are skipped.
"""

import os
import json
import h5py
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── FONT ────────────────────────────────────────────────────────────────────
_FONT_DIR = "/scr/marcelto/.fonts/roboto-mono"
for _f in ("RobotoMono-Regular.ttf", "RobotoMono-Bold.ttf"):
    _p = os.path.join(_FONT_DIR, _f)
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
mpl.rcParams["font.family"] = "Roboto Mono"

# ── DATA: demos ─────────────────────────────────────────────────────────────
DEMO_HDF5 = (
    "/iris/u/marcelto/reward_learning/diffusion_policy/"
    "shared_data_slow_fast_final_slower/scripted_data/demos.hdf5"
)
demo_pts = {"left": [], "right": []}
with h5py.File(DEMO_HDF5, "r") as f:
    for k in sorted(f["data"].keys()):
        d = f["data"][k]
        side = d.attrs["target_peg"]
        L = int(d["actions"].shape[0])
        demo_pts[side].append(300.0 / L)


def load_rollouts(jsonl_path):
    """Read a per_rollout_steps.jsonl file and return per-peg step lists.

    Only keeps the z_pos (positive peg conditioning) entry. We split the
    rollouts by which peg they actually ended up at, so both right and left
    outcomes that came from z_pos conditioning are returned.
    """
    out = {"left": [], "right": []}
    if not os.path.exists(jsonl_path):
        return out
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            tr = rec.get("target_rewards") or []
            if not tr:
                continue
            peg_z = tr[-1]
            if peg_z <= 0:        # skip z_zero and z_neg
                continue
            test = rec["prefixes"].get("test/", {})
            out["right"].extend(test.get("first_success_step_right", []))
            out["left"].extend(test.get("first_success_step_left", []))
    return out


FPL_JSONL = (
    "/iris/u/marcelto/reward_learning/diffusion_policy/"
    "pipeline_output_slow_fast_final_rhp_slower/eval/per_rollout_steps.jsonl"
)
SINGLE_JSONL = (
    "/iris/u/marcelto/reward_learning/diffusion_policy/"
    "pipeline_output_slow_fast_final_single_pref_slower/eval/per_rollout_steps.jsonl"
)
fpl_rollouts    = load_rollouts(FPL_JSONL)
single_rollouts = load_rollouts(SINGLE_JSONL)

# ── PLOT ────────────────────────────────────────────────────────────────────
DEMO_COLOR   = "#9ABC6C"
FPL_COLOR    = "#C46B53"
SINGLE_COLOR = "#5B8FBF"

fig, ax = plt.subplots(figsize=(4, 3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_for = {"left": -1.0, "right": 1.0}
rng = np.random.default_rng(0)
SUBSAMPLE = {"left": 2, "right": 4}

# Demos
for side in ("left", "right"):
    vals = demo_pts[side][::SUBSAMPLE[side]]
    if not vals:
        continue
    jitter = rng.uniform(-0.18, 0.18, size=len(vals))
    ax.scatter(
        x_for[side] + jitter, vals,
        s=40, c=DEMO_COLOR, marker="o",
        edgecolor="black", linewidth=0.5, alpha=0.7, zorder=3,
    )

# FPL (only the z_pos right-peg rollouts)
for side in ("left", "right"):
    steps = fpl_rollouts[side][::SUBSAMPLE[side]]
    if not steps:
        continue
    thrs = [300.0 / s for s in steps]
    jitter = rng.uniform(-0.18, 0.18, size=len(thrs))
    ax.scatter(
        x_for[side] + jitter, thrs,
        s=40, c=FPL_COLOR, marker="D",
        edgecolor="black", linewidth=0.5, alpha=0.7, zorder=5,
    )

# Single (only the z_pos right-peg rollouts)
for side in ("left", "right"):
    steps = single_rollouts[side][::SUBSAMPLE[side]]
    if not steps:
        continue
    thrs = [300.0 / s for s in steps]
    jitter = rng.uniform(-0.18, 0.18, size=len(thrs))
    ax.scatter(
        x_for[side] + jitter, thrs,
        s=40, c=SINGLE_COLOR, marker="s",
        edgecolor="black", linewidth=0.5, alpha=0.7, zorder=4,
    )

ax.set_xlim(-2.0, 2.0)
ax.set_xticks([-1, 1])
ax.set_xticklabels(["Left peg", "Right peg"], fontsize=13)
ax.set_ylabel("Throughput", fontsize=13)
ax.tick_params(axis="y", labelsize=11)
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5,
        color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b0b0b0")
    ax.spines[s].set_linewidth(0.8)

handles = [
    mpatches.Patch(facecolor=DEMO_COLOR, edgecolor="black", linewidth=0.5,
                   label="Demos"),
    plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=SINGLE_COLOR,
               markeredgecolor="black", markersize=8, label="Single Pref."),
    plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=FPL_COLOR,
               markeredgecolor="black", markersize=8, label="FPL"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.01),
          ncol=len(handles), fontsize=11, frameon=True,
          edgecolor="#888888")

fig.savefig("slow_fast_scatter_with_single.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("slow_fast_scatter_with_single.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved slow_fast_scatter_with_single.{pdf,png}")
