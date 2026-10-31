"""Diagnostic for the threshold-sweep mode (dev only).

1. Oracle: for each GT egg, best IoU of any candidate -> upper bound if selection were perfect.
   Low oracle = candidate generation problem; high oracle but low result = selection/scoring problem.
2. Score separation: scores of candidates that match a GT egg (IoU>=0.7) vs. those that don't.
"""
import json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical, data

params = {**classical.DEFAULTS, **json.loads(sys.argv[1])}
paths = data.image_paths()
best_ious, pos, neg, worst = [], [], [], []
for sid in data.split_ids("dev"):
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid, instance=True)
    f = classical.feature_map(bgr, params)
    if params["blur"]:
        f = cv2.GaussianBlur(f, (params["blur"],) * 2, 0)
    cands = classical.sweep_candidates(f.astype(np.float32), params)
    H, W = gt.shape
    eggs = [gt == k for k in range(1, gt.max() + 1)]
    ious = np.zeros((len(cands), len(eggs)))
    for i, c in enumerate(cands):
        full = np.zeros((H, W), bool); x0, y0, x1, y1 = c["box"]; full[y0:y1, x0:x1] = c["mask"] > 0
        for j, e in enumerate(eggs):
            inter = np.logical_and(full, e).sum()
            ious[i, j] = inter / (full.sum() + e.sum() - inter)
    b = ious.max(0) if len(cands) else np.zeros(len(eggs))
    best_ious += list(b)
    worst += [(sid.replace("not_damaged_", "nd"), round(v, 2)) for v in b if v < 0.7]
    for i, c in enumerate(cands):
        (pos if ious[i].max() >= 0.7 else neg).append(c["score"])
best_ious = np.array(best_ious)
print(f"oracle: GT eggs={len(best_ious)}  mean best IoU={best_ious.mean():.3f}  "
      f">=0.7: {(best_ious >= .7).mean():.2f}  >=0.85: {(best_ious >= .85).mean():.2f}")
print("eggs without a good candidate:", worst)
pos, neg = np.array(pos), np.array(neg)
print(f"candidates: egg-like={len(pos)} other={len(neg)}")
for q in [10, 25, 50, 75, 90]:
    print(f"  p{q}: egg {np.percentile(pos, q):.3f}   other {np.percentile(neg, q):.3f}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"  score>={t}: keeps {np.mean(pos >= t):.2f} of egg cands, {np.mean(neg >= t):.3f} of others")
