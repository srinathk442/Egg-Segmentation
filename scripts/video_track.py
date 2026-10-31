"""Task 4 step 2 - track eggs through the video and render the result.

Reads per-frame masks from scripts/video_detect.py, links them with EggTracker and writes
  outputs/task4/tracked.mp4   video: each egg coloured by its track ID, ID label, motion trail;
                              eggs hidden by the hand are shown as a dashed circle at the predicted position
  outputs/task4/tracks.csv    one row per (frame, track): position, area, visible/hidden
  outputs/task4/track_stats.json
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, cv2
import imageio.v2 as imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg.tracking import EggTracker

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video/egg_video.mp4")
ap.add_argument("--det", default="outputs/task4/det")
ap.add_argument("--out", default="outputs/task4")
ap.add_argument("--out_width", type=int, default=1280)
ap.add_argument("--max_age", type=int, default=240, help="frames a lost track is kept (10 s)")
ap.add_argument("--min_hits", type=int, default=3)
ap.add_argument("--gate", type=float, default=1.0)
ap.add_argument("--reid_after", type=int, default=24, help="frames; -1 = no re-identification")
ap.add_argument("--show_hidden", type=int, default=24, help="draw a hidden egg's marker for this many frames")
a = ap.parse_args()

# categorical palette (BGR): blue, orange, aqua, yellow, magenta, green, violet, red
PAL = [(0xd6, 0x78, 0x2a), (0x34, 0x68, 0xeb), (0x7a, 0xaf, 0x1b), (0x00, 0xa1, 0xed),
       (0xa4, 0x7b, 0xe8), (0x00, 0x83, 0x00), (0xd6, 0x50, 0x62), (0x48, 0x49, 0xe3)]

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = a.out_width; H = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * W / cap.get(cv2.CAP_PROP_FRAME_WIDTH))
tracker = EggTracker(gate=a.gate, max_age=a.max_age, min_hits=a.min_hits,
                     reid_after=None if a.reid_after < 0 else a.reid_after)
writer = imageio.get_writer(os.path.join(a.out, "tracked.mp4"), fps=fps, codec="libx264", quality=7,
                            macro_block_size=8)
rows, n_det = [], []


def label(img, text, org, col):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = int(org[0] - w / 2), int(org[1])
    cv2.rectangle(img, (x - 5, y - h - 7), (x + w + 5, y + 6), col, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


for f in range(n):
    ok, frame = cap.read()
    if not ok:
        break
    inst = cv2.imread(os.path.join(a.det, f"f_{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
    dets = [inst == k for k in range(1, inst.max() + 1) if (inst == k).any()]
    n_det.append(len(dets))
    s = W / inst.shape[1]                                    # detection resolution -> output resolution
    out = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    over = out.copy()
    active = tracker.step(dets)
    active = [(t, seen) for t, seen in active if seen or t.miss <= a.show_hidden]
    for t, seen in active:
        col = PAL[(t.id - 1) % len(PAL)]
        c = t.centre * s
        if seen:
            m = cv2.resize(t.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            over[m] = col
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, col, 3)
        else:                                                # coasting: dashed circle at the prediction
            r = int(t.diam * s / 2)
            for k in range(0, 360, 30):
                cv2.ellipse(out, (int(c[0]), int(c[1])), (r, r), 0, k, k + 15, col, 3)
        pts = (np.array(t.trail[-36:]) * s).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, col, 2, cv2.LINE_AA)
        label(out, f"egg {t.id}" + ("" if seen else " (hidden)"), (c[0], c[1] - t.diam * s / 2 - 12), col)
        rows.append(dict(frame=f, t=f / fps, id=t.id, x=float(t.centre[0] * s), y=float(t.centre[1] * s),
                         area=int(t.mask.sum() * s * s), visible=seen))
    out = cv2.addWeighted(over, 0.35, out, 0.65, 0)
    n_vis = sum(seen for _, seen in active)
    cv2.rectangle(out, (0, 0), (W, 34), (20, 20, 20), -1)
    cv2.putText(out, f"frame {f}  t={f / fps:5.1f}s   eggs tracked: {n_vis}" +
                (f" (+{len(active) - n_vis} hidden)" if len(active) > n_vis else ""),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    writer.append_data(out[..., ::-1])
writer.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "tracks.csv"), index=False)
per_id = df.groupby("id").agg(first=("frame", "min"), last=("frame", "max"), visible=("visible", "sum"),
                              hidden=("visible", lambda v: int((~v).sum())))
stats = dict(frames=n, ids_created=int(df.id.nunique()) if len(df) else 0,
             frames_with_detections=int(np.sum(np.array(n_det) > 0)),
             detections_per_frame=dict(zip(*map(lambda x: x.tolist(), np.unique(n_det, return_counts=True)))),
             per_id=per_id.reset_index().to_dict("records"),
             params=dict(gate=a.gate, max_age=a.max_age, min_hits=a.min_hits, reid_after=a.reid_after,
                         show_hidden=a.show_hidden))
json.dump(stats, open(os.path.join(a.out, "track_stats.json"), "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in stats.items() if k != "per_id"}, default=int))
print(per_id.to_string())
