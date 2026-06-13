import os
import json
import numpy as np
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

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

# (full key in JSON, short label for legend)
KEYS = [
    ("Quality of placement of big plate",   "Big plate"),
    ("Quality of placement of small plate", "Small plate"),
    ("Quality of placement of cup",         "Cup"),
    ("Quality of placement of cutlery",     "Cutlery"),
]

# Match the bar-chart palette in plot_graph.py
COLORS = ["#C46B53", "#AFC3C5", "#9ABC6C", "#F9CF48"]

SMOOTH_WIN = 15  # moving-average window for the smoothed line

# Per-curve "highlight" windows (inclusive). The curve is drawn bolder and a
# soft colored band is shaded behind it inside this range; outside, the curve
# is faded so the highlighted segment stands out.
HIGHLIGHTS = {
    "Big plate":   [(70, 100)],
    "Small plate": [(150, 200), (200, 240)],
    "Cup":         [(270, 300)],
    "Cutlery":     [(340, 400)],
}

# Event annotations: (start, end, label, color, kind)
# kind = "success" → just a label; "failure" → red hatched overlay + label
SUCCESS_COLOR = "#2D8A3F"
FAILURE_COLOR = "#A82A2A"
EVENTS = [
    (70,  100, "Success:\nbig plate placed",   SUCCESS_COLOR, "success"),
    (150, 200, "Failure:\nsmall plate falls",  FAILURE_COLOR, "failure"),
    (200, 240, "Success:\nsmall plate placed", SUCCESS_COLOR, "success"),
    (270, 300, "Success:\ncup placed",         SUCCESS_COLOR, "success"),
    (340, 400, "Success:\ncutlery placed",     SUCCESS_COLOR, "success"),
]
# Vertical dashed separators between regions, at the boundaries
REGION_BOUNDARIES = sorted({a for a, _, _, _, _ in EVENTS} | {b for _, b, _, _, _ in EVENTS})
HIGHLIGHT_BAND_ALPHA = 0.08
LINE_ALPHA_DIM       = 0.20
LINE_ALPHA_FOCUS     = 1.0
LW_DIM               = 1.1
LW_FOCUS             = 2.4

# ── LOAD ────────────────────────────────────────────────────────────────────

with open(JSON_PATH) as f:
    data = json.load(f)

per_frame = data["per_frame"]
series = {short: np.asarray(per_frame[full], dtype=float) for full, short in KEYS}
n = len(next(iter(series.values())))
t = np.arange(n)


def moving_avg(x, w):
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[: len(x)]


smoothed = {k: moving_avg(v, SMOOTH_WIN) for k, v in series.items()}

# ── PLOT ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 3.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Zero baseline
ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8, zorder=1)

color_for = dict(zip([s for _, s in KEYS], COLORS))

# Plot each curve once at full opacity, uniform width. These line artists are
# what the "no-shades" variant displays. For the shaded variant we'll dim them
# afterwards and add bold focus segments + bands.
main_lines = {}
for short, raw in series.items():
    color = color_for[short]
    sm = smoothed[short]
    ln, = ax.plot(t, sm, color=color, linewidth=1.6, alpha=1.0, zorder=3,
                  solid_capstyle="round")
    main_lines[short] = ln

ax.set_xlim(0, n - 1)

ax.set_xlabel("Time (s)", fontsize=14)
# Data is in frame indices recorded at 10 Hz → divide by 10 to display seconds
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
ax.set_ylabel("Reward", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b0b0b0")
    ax.spines[s].set_linewidth(0.8)

# ── Compact bordered legend, centered below the axes ────────────────────────
plt.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.28)
fig.canvas.draw()
ax_bbox = ax.get_position()

swatch_w   = 0.022
swatch_gap = 0.006
item_gap   = 0.032
char_w     = 0.0098
side_pad   = 0.016

title_text = "Reward dimension for Quality of placement of:"
short_labels = [s for _, s in KEYS]
title_w = char_w * len(title_text)
item_widths = [swatch_w + swatch_gap + char_w * len(lbl) for lbl in short_labels]
legend_width = (
    2 * side_pad
    + title_w + item_gap
    + sum(item_widths) + item_gap * (len(short_labels) - 1)
)

fig_center_x = (ax_bbox.x0 + ax_bbox.x1) / 2
x0_legend    = fig_center_x - legend_width / 2
legend_h     = 0.085
legend_b     = ax_bbox.y0 - 0.18
legend_mid   = legend_b + legend_h / 2

fig.patches.append(
    mpatches.FancyBboxPatch(
        (x0_legend, legend_b), legend_width, legend_h,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#888888", linewidth=0.8,
        transform=fig.transFigure, clip_on=False, zorder=10,
    )
)

# Title text on the left
cursor = x0_legend + side_pad
fig.text(
    cursor, legend_mid, title_text,
    ha="left", va="center", fontsize=12, fontweight="bold",
    transform=fig.transFigure, zorder=12,
)
cursor += title_w + item_gap

swatch_h = legend_h * 0.55
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
        ha="left", va="center", fontsize=12,
        transform=fig.transFigure, zorder=12,
    )
    cursor += w + item_gap

# ── Save the "no shades" variant first ──────────────────────────────────────
# All curves at full opacity, no bolded focus segments, no separators, no bands.
fig.set_size_inches(fig.get_size_inches()[0], 6.2)
fig.savefig("reward_curves_no_shades.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("reward_curves_no_shades.png", dpi=200, bbox_inches="tight", facecolor="white")

# ── Build the shaded variant on top of the same figure ──────────────────────
# Dim the main curves (so the bold focus segments stand out)
for ln in main_lines.values():
    ln.set_alpha(LINE_ALPHA_DIM)
    ln.set_linewidth(LW_DIM)

# Bold focus segments inside each highlight range
for short, sm in smoothed.items():
    color = color_for[short]
    for a, b in HIGHLIGHTS.get(short, []):
        mask = (t >= a) & (t <= b)
        ax.plot(t[mask], sm[mask], color=color, linewidth=LW_FOCUS,
                alpha=LINE_ALPHA_FOCUS, zorder=4, solid_capstyle="round")

# Gray dashed vertical separators between regions
for x in REGION_BOUNDARIES:
    ax.axvline(x, color="#9a9a9a", linestyle="--", linewidth=0.7,
               alpha=0.7, zorder=1.7)

# Colored region bands: green for success, red for failure
for (a, b, label, ecolor, kind) in EVENTS:
    band_alpha = 0.18 if kind == "failure" else 0.15
    ax.axvspan(a, b, facecolor=ecolor, alpha=band_alpha, zorder=1)

# Event text labels just inside the top of the axes
x_trans = ax.get_xaxis_transform()
for (a, b, label, ecolor, kind) in EVENTS:
    ax.text(
        (a + b) / 2, 0.97, label,
        transform=x_trans, ha="center", va="top",
        fontsize=9, color=ecolor, fontweight="bold", zorder=20,
        clip_on=False,
    )

fig.savefig("reward_curves.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("reward_curves.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved reward_curves.{pdf,png} and reward_curves_no_shades.{pdf,png}")
