import os
import json
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── FONT ────────────────────────────────────────────────────────────────────

_FONT_DIR = "/scr/marcelto/.fonts/roboto-mono"
for _fname in ("RobotoMono-Regular.ttf", "RobotoMono-Bold.ttf"):
    _fpath = os.path.join(_FONT_DIR, _fname)
    if os.path.exists(_fpath):
        font_manager.fontManager.addfont(_fpath)
mpl.rcParams["font.family"] = "Roboto Mono"

# ── DATA ────────────────────────────────────────────────────────────────────

JSON_PATH = (
    "/iris/u/marcelto/reward_learning/infer_output/setup_table_iter3_open_cum_qwen/"
    "reward_model_2026-05-25_23-29-41_qwen_open_cum_j79567_step003000_score_demos_20.json"
)

KEYS = [
    ("Quality of placement of big plate",   "Big plate"),
    ("Quality of placement of small plate", "Small plate"),
    ("Quality of placement of cup",         "Cup"),
    ("Quality of placement of cutlery",     "Cutlery"),
    ("Formality of setup",                  "Formality"),
]

COLORS = ["#3B6FB6", "#E07A3E", "#4FA152", "#C84B45", "#8C5BC8"]

with open(JSON_PATH) as f:
    data = json.load(f)

per_frame = data["per_frame"]
series = {short: np.asarray(per_frame[full], dtype=float) for full, short in KEYS}
cumulative = {k: np.cumsum(v) for k, v in series.items()}
n = len(next(iter(series.values())))
t = np.arange(n)

# ── PLOT ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 3.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8, zorder=1)

for (full, short), color in zip(KEYS, COLORS):
    # No smoothing — direct cumsum of per_frame so every frame's contribution
    # is visible as a slope change.
    ax.plot(t, cumulative[short], color=color, linewidth=1.2,
            zorder=3, label=short, antialiased=True)

ax.set_xlim(0, n - 1)
ax.set_xlabel("Timestep (frame)", fontsize=10)
ax.set_ylabel("Cumulative reward", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b0b0b0")
    ax.spines[s].set_linewidth(0.8)

# ── Legend ──────────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.14)
fig.canvas.draw()
ax_bbox = ax.get_position()

swatch_w   = 0.018
swatch_gap = 0.005
item_gap   = 0.028
char_w     = 0.0075
side_pad   = 0.014

short_labels = [s for _, s in KEYS]
item_widths = [swatch_w + swatch_gap + char_w * len(lbl) for lbl in short_labels]
legend_width = 2 * side_pad + sum(item_widths) + item_gap * (len(short_labels) - 1)

fig_center_x = (ax_bbox.x0 + ax_bbox.x1) / 2
x0_legend    = fig_center_x - legend_width / 2
legend_h     = 0.07
legend_b     = ax_bbox.y1 + 0.04
legend_mid   = legend_b + legend_h / 2

fig.patches.append(
    mpatches.FancyBboxPatch(
        (x0_legend, legend_b), legend_width, legend_h,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#888888", linewidth=0.8,
        transform=fig.transFigure, clip_on=False, zorder=10,
    )
)
swatch_h = legend_h * 0.55
cursor = x0_legend + side_pad
for color, label, w in zip(COLORS, short_labels, item_widths):
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (cursor, legend_mid - swatch_h / 2), swatch_w, swatch_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="#000000", linewidth=0.8,
            transform=fig.transFigure, clip_on=False, zorder=11,
        )
    )
    fig.text(
        cursor + swatch_w + swatch_gap, legend_mid, label,
        ha="left", va="center", fontsize=9,
        transform=fig.transFigure, zorder=12,
    )
    cursor += w + item_gap

fig.savefig("reward_cumulative.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("reward_cumulative.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved reward_cumulative.pdf and reward_cumulative.png")
