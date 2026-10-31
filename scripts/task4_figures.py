"""Task 4 figures: track timeline, detections per frame, and key frames of the tracked video."""
import json, os
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/task4"
SURF, INK2, GRID, GREY = "#fcfcfb", "#52514e", "#e4e3df", "#a3a29c"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

tr = pd.read_csv(f"{OUT}/tracks.csv")
fps = 24.0
ndet = np.array([cv2.imread(f"{OUT}/det/f_{f:05d}.png", 0).max() for f in range(int(tr.frame.max()) + 200)
                 if os.path.exists(f"{OUT}/det/f_{f:05d}.png")])

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
ids = sorted(tr.id.unique())
for k, i in enumerate(ids):
    d = tr[tr.id == i]
    for vis, alpha in [(True, 1.0), (False, 0.3)]:
        f = d[d.visible == vis].frame.values / fps
        a1.scatter(f, np.full(len(f), k), marker="|", s=120, color=PAL[(i - 1) % 8], alpha=alpha, linewidths=1.2)
a1.set_yticks(range(len(ids))); a1.set_yticklabels([f"egg {i}" for i in ids]); a1.grid(axis="y", visible=False)
a1.set_title("Track timeline (solid = egg detected, faded = hidden / coasting on Kalman prediction)")
a2.step(np.arange(len(ndet)) / fps, ndet, where="post", color=PAL[0], lw=1.5)
a2.set_ylabel("eggs detected"); a2.set_xlabel("time (s)"); a2.set_yticks(range(0, int(ndet.max()) + 1))
a2.set_title("Per-frame detections from the classical segmenter")
fig.savefig(f"{OUT}/fig_track_timeline.png"); plt.close(fig)

# key frames of the rendered video
cap = cv2.VideoCapture(f"{OUT}/tracked.mp4")
keys = json.load(open(f"{OUT}/key_frames.json")) if os.path.exists(f"{OUT}/key_frames.json") else \
    [240, 384, 500, 580, 640, 700, 760, 816, 900]
tiles = []
for f in keys:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    if ok: tiles.append(cv2.resize(im, (640, 360)))
while len(tiles) % 3: tiles.append(np.full_like(tiles[0], 252))
cv2.imwrite(f"{OUT}/key_frames.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
print("written", f"{OUT}/fig_track_timeline.png", f"{OUT}/key_frames.jpg")
