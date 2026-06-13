import os
import json
import numpy as np
import cv2
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter

# ── FONT ────────────────────────────────────────────────────────────────────

_FONT_DIR = "/scr/marcelto/.fonts/roboto-mono"
for _fname in ("RobotoMono-Regular.ttf", "RobotoMono-Bold.ttf"):
    _fpath = os.path.join(_FONT_DIR, _fname)
    if os.path.exists(_fpath):
        font_manager.fontManager.addfont(_fpath)
mpl.rcParams["font.family"] = "Roboto Mono"

# ── INPUTS ──────────────────────────────────────────────────────────────────

JSON_PATH = (
    "/iris/u/marcelto/reward_learning/infer_output/setup_table_iter3_open_cum_qwen/"
    "reward_model_2026-05-25_23-29-41_qwen_open_cum_j79567_step003000_score_demos_20.json"
)
VIDEO_IN  = "/iris/u/marcelto/reward_learning/demos_video_20.mp4"
VIDEO_OUT = "/iris/u/marcelto/reward_learning/reward_curves_video.mp4"
FPS_OUT   = 15

KEYS = [
    ("Quality of placement of big plate",   "Big plate"),
    ("Quality of placement of small plate", "Small plate"),
    ("Quality of placement of cup",         "Cup"),
    ("Quality of placement of cutlery",     "Cutlery"),
]
COLORS = ["#C46B53", "#AFC3C5", "#9ABC6C", "#F9CF48"]
SMOOTH_WIN = 15

HIGHLIGHTS = {
    "Big plate":   [(70, 100)],
    "Small plate": [(150, 200), (200, 240)],
    "Cup":         [(270, 300)],
    "Cutlery":     [(340, 400)],
}

SUCCESS_COLOR = "#2D8A3F"
FAILURE_COLOR = "#A82A2A"
EVENTS = [
    (70,  100, "Success:\nbig plate placed",   SUCCESS_COLOR, "success"),
    (150, 200, "Failure:\nsmall plate falls",  FAILURE_COLOR, "failure"),
    (200, 240, "Success:\nsmall plate placed", SUCCESS_COLOR, "success"),
    (270, 300, "Success:\ncup placed",         SUCCESS_COLOR, "success"),
    (340, 400, "Success:\ncutlery placed",     SUCCESS_COLOR, "success"),
]
REGION_BOUNDARIES = sorted({a for a, _, _, _, _ in EVENTS} | {b for _, b, _, _, _ in EVENTS})
HIGHLIGHT_BAND_ALPHA = 0.08
LINE_ALPHA_DIM       = 0.20
LW_DIM               = 1.1
LW_FOCUS             = 2.4

# ── LOAD DATA ───────────────────────────────────────────────────────────────

with open(JSON_PATH) as f:
    data = json.load(f)

per_frame = data["per_frame"]
short_for_full = {full: short for full, short in KEYS}
series   = {short: np.asarray(per_frame[full], dtype=float) for full, short in KEYS}
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
color_for = dict(zip([s for _, s in KEYS], COLORS))

# Pre-read all video frames (395 frames × 2560×720 ≈ 2 GB raw; downsample once)
cap = cv2.VideoCapture(VIDEO_IN)
n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
assert n_video == n, f"video frames ({n_video}) != reward frames ({n})"
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DOWNSAMPLE_W = 1280  # halve the resolution to save memory and disk
scale = DOWNSAMPLE_W / W
new_w, new_h = DOWNSAMPLE_W, int(round(H * scale))
frames = np.empty((n, new_h, new_w, 3), dtype=np.uint8)
for i in range(n):
    ok, fr = cap.read()
    if not ok:
        break
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    fr = cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    frames[i] = fr
cap.release()

# ── FIGURE LAYOUT ───────────────────────────────────────────────────────────

# Aspect ratio: frame is new_w/new_h. We'll fit it on top, then plot below.
# Make sure fig_w * dpi and fig_h * dpi are both even (libx264 requirement).
DPI = 120
fig_w = 12.0                        # 1440 px wide  (even)
fig_h = 8.0                         # 960  px tall  (even)
# Compute image panel height in inches to preserve the frame's aspect ratio.
frame_h = fig_w * (new_h / new_w)
top_pad = 0.15                      # small gap above the plot
bottom_pad = 1.1                    # room for x-axis label + legend
plot_h = fig_h - frame_h - top_pad - bottom_pad

fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")

# Image axes (top)
img_ax_h = frame_h / fig_h
img_ax = fig.add_axes([0.03, 1 - img_ax_h - 0.02, 0.94, img_ax_h - 0.02])
img_ax.set_xticks([]); img_ax.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    img_ax.spines[s].set_visible(False)
img_artist = img_ax.imshow(frames[0])

# Plot axes (bottom)
plot_top = 1 - img_ax_h - top_pad / fig_h
plot_bot = bottom_pad / fig_h
ax = fig.add_axes([0.06, plot_bot, 0.92, plot_top - plot_bot])
ax.set_facecolor("white")

ax.axhline(0, color="#999999", linestyle=":", linewidth=0.8, zorder=1)

# Animated event bands + labels: created hidden, revealed as the playhead crosses
# their start, and the band grows in width up to the current frame.
x_trans = ax.get_xaxis_transform()
event_bands  = []   # list of (Rectangle, a, b)
event_labels = []   # list of (Text,      a, b)
for (a, b, label, ecolor, kind) in EVENTS:
    band_alpha = 0.18 if kind == "failure" else 0.15
    rect = Rectangle((a, 0), 0, 1, transform=x_trans,
                     facecolor=ecolor, alpha=band_alpha, edgecolor="none",
                     zorder=1)
    rect.set_visible(False)
    ax.add_patch(rect)
    event_bands.append((rect, a, b))

# Gray dashed vertical separators between regions (hidden until crossed)
sep_lines = []  # list of (Line2D, x)
for x in REGION_BOUNDARIES:
    ln = ax.axvline(x, color="#9a9a9a", linestyle="--", linewidth=0.7,
                    alpha=0.7, zorder=1.7)
    ln.set_visible(False)
    sep_lines.append((ln, x))

# Y-limits from full data so the view doesn't jump as curves grow
all_vals = np.concatenate([v for v in smoothed.values()])
y_lo = float(np.min(all_vals)) - 0.3
y_hi = float(np.max(all_vals)) + 0.3
ax.set_xlim(0, n - 1)
ax.set_ylim(y_lo, y_hi)

# Event labels at the top of the data area (hidden until the playhead reaches
# the start of each range)
for (a, b, label, ecolor, kind) in EVENTS:
    txt = ax.text(
        (a + b) / 2, 0.97, label,
        transform=x_trans, ha="center", va="top",
        fontsize=7, color=ecolor, fontweight="bold", zorder=20,
        clip_on=False,
    )
    txt.set_visible(False)
    event_labels.append((txt, a, b))

ax.set_xlabel("Time (s)", fontsize=10)
# Data is in frame indices recorded at 10 Hz → divide by 10 to display seconds
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
ax.set_ylabel("Reward", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="#d0d0d0", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#b0b0b0")
    ax.spines[s].set_linewidth(0.8)

# Animated line artists (one per curve, plus per-(curve,range) focus lines)
dim_lines  = {}
focus_lines = {}   # short -> list of (Line2D, range)
for short in [s for _, s in KEYS]:
    color = color_for[short]
    dim_lines[short], = ax.plot([], [], color=color, linewidth=LW_DIM,
                                alpha=LINE_ALPHA_DIM, zorder=3,
                                solid_capstyle="round")
    focus_lines[short] = []
    for (a, b) in HIGHLIGHTS.get(short, []):
        ln, = ax.plot([], [], color=color, linewidth=LW_FOCUS, alpha=1.0,
                      zorder=4, solid_capstyle="round")
        focus_lines[short].append((ln, a, b))

playhead = ax.axvline(0, color="#222222", linewidth=1.2, alpha=0.7, zorder=5)

# ── Legend (static) ─────────────────────────────────────────────────────────
fig.canvas.draw()
ax_bbox = ax.get_position()
swatch_w   = 0.018
swatch_gap = 0.005
item_gap   = 0.028
char_w     = 0.0075
side_pad   = 0.014
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
legend_h     = 0.035
legend_b     = ax_bbox.y0 - 0.07
legend_mid   = legend_b + legend_h / 2
fig.patches.append(
    mpatches.FancyBboxPatch(
        (x0_legend, legend_b), legend_width, legend_h,
        boxstyle="square,pad=0",
        facecolor="white", edgecolor="#888888", linewidth=0.8,
        transform=fig.transFigure, clip_on=False, zorder=10,
    )
)
sw_h = legend_h * 0.55
cursor = x0_legend + side_pad
fig.text(cursor, legend_mid, title_text,
         ha="left", va="center", fontsize=9, fontweight="bold",
         transform=fig.transFigure, zorder=12)
cursor += title_w + item_gap
for color, label, w in zip(COLORS, short_labels, item_widths):
    fig.patches.append(
        mpatches.FancyBboxPatch(
            (cursor, legend_mid - sw_h / 2), swatch_w, sw_h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="#000000", linewidth=0.8,
            transform=fig.transFigure, clip_on=False, zorder=11,
        )
    )
    fig.text(cursor + swatch_w + swatch_gap, legend_mid, label,
             ha="left", va="center", fontsize=9,
             transform=fig.transFigure, zorder=12)
    cursor += w + item_gap

# ── Animate ─────────────────────────────────────────────────────────────────

def update(i):
    img_artist.set_data(frames[i])
    for short in [s for _, s in KEYS]:
        upto = i + 1
        dim_lines[short].set_data(t[:upto], smoothed[short][:upto])
        for (ln, a, b) in focus_lines[short]:
            mask = (t >= a) & (t <= min(b, i))
            ln.set_data(t[mask], smoothed[short][mask])
    playhead.set_xdata([i, i])

    # Reveal event bands once the playhead enters the range; band grows up to i
    for rect, a, b in event_bands:
        if i >= a:
            rect.set_visible(True)
            rect.set_width(min(i, b) - a)
        else:
            rect.set_visible(False)
    # Reveal labels at the same time
    for txt, a, b in event_labels:
        txt.set_visible(i >= a)
    # Reveal separators when crossed
    for ln, x in sep_lines:
        ln.set_visible(i >= x)

    artists = [img_artist, playhead]
    artists.extend(dim_lines.values())
    for short in focus_lines:
        artists.extend(ln for (ln, _, _) in focus_lines[short])
    artists.extend(rect for rect, _, _ in event_bands)
    artists.extend(txt for txt, _, _ in event_labels)
    artists.extend(ln for ln, _ in sep_lines)
    return artists

anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / FPS_OUT,
                               blit=False)

writer = animation.FFMpegWriter(fps=FPS_OUT, codec="libx264",
                                bitrate=4000,
                                extra_args=["-pix_fmt", "yuv420p"])
print(f"Writing {VIDEO_OUT} ({n} frames @ {FPS_OUT} fps)...")
anim.save(VIDEO_OUT, writer=writer, dpi=DPI)
print("Done.")
