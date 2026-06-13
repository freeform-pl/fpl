"""Scatter plot: first-success step per demo, by peg side, with FPL overlay.

x-axis: peg (left = -1, right = +1)
y-axis: first success step

Demos: 200 individual points from shared_data_slow_fast_final_slower.
FPL: 1 mean diamond per peg from the cached comparison_results.json of
run 8gldvuea (100 rollouts each, only the mean is stored locally — per-rollout
values would require modifying eval_conditioned.py to dump them).
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

# ── DATA: load 200 demos ────────────────────────────────────────────────────
DEMO_HDF5 = (
    "/iris/u/marcelto/reward_learning/diffusion_policy/"
    "shared_data_slow_fast_final_slower/scripted_data/demos.hdf5"
)

demo_pts = {"left": [], "right": []}  # peg -> list of throughputs (300 / step)
with h5py.File(DEMO_HDF5, "r") as f:
    for k in sorted(f["data"].keys()):
        d = f["data"][k]
        side = d.attrs["target_peg"]
        L = int(d["actions"].shape[0])
        demo_pts[side].append(300.0 / L)

# ── EVAL: load per-rollout first-success steps from RHP run 8gldvuea ────────
EVAL_JSONL = (
    "/iris/u/marcelto/reward_learning/diffusion_policy/"
    "pipeline_output_slow_fast_final_rhp_slower/eval/per_rollout_steps.jsonl"
)
# Each line = one eval condition's dump. We want:
#   - z_pos  (target_rewards[1] > 0) → right-peg rollouts
#   - z_neg  (target_rewards[1] < 0) → left-peg rollouts
rhp_rollouts = {"left": [], "right": []}
with open(EVAL_JSONL) as f:
    for line in f:
        rec = json.loads(line)
        tr = rec.get("target_rewards") or []
        if len(tr) < 2:
            continue
        peg_z = tr[1]  # 2nd reward dim is the peg axis
        prefix = rec["prefixes"].get("test/", {})
        # Only show z_pos (right-peg conditioning); skip z_neg / z_zero.
        if peg_z > 0:
            rhp_rollouts["right"].extend(prefix.get("first_success_step_right", []))

# ── PLOT ────────────────────────────────────────────────────────────────────
DEMO_COLOR = "#9ABC6C"
EVAL_COLOR = "#C46B53"

fig, ax = plt.subplots(figsize=(4, 3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_for = {"left": -1.0, "right": 1.0}
rng = np.random.default_rng(0)
# Per-side subsample factor (right peg clusters tighter, so we thin it more)
SUBSAMPLE = {"left": 2, "right": 4}

# One scatter point per demo, jittered horizontally so they don't overlap
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

# FPL: one scatter point per rollout (throughput = 300 / step)
for side in ("left", "right"):
    steps = rhp_rollouts[side][::SUBSAMPLE[side]]
    if not steps:
        continue
    thrs = [300.0 / s for s in steps]
    jitter = rng.uniform(-0.18, 0.18, size=len(thrs))
    ax.scatter(
        x_for[side] + jitter, thrs,
        s=40, c=EVAL_COLOR, marker="D",
        edgecolor="black", linewidth=0.5, alpha=0.7, zorder=5,
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
    plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=EVAL_COLOR,
               markeredgecolor="black", markersize=8,
               label="FPL"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=len(handles), fontsize=11, frameon=True,
          edgecolor="#888888")

fig.savefig("slow_fast_scatter.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("slow_fast_scatter.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved slow_fast_scatter.{pdf,png}")
