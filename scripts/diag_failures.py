"""Diagnose images a trial fails on (IoU < 0.3): was a good egg region among the refined candidates
(-> selection/scoring problem) or never produced (-> candidate generation problem)?

usage: python scripts/diag_failures.py T10b_multiview
"""
import json, os, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical as C, data

run = sys.argv[1]
summ = json.load(open(f"experiments/runs/{run}/summary.json"))
p = {**C.DEFAULTS, **summ["params"]}
paths = data.image_paths()
df = pd.read_csv(f"experiments/runs/{run}/per_image.csv")
out = []
for sid in df[df.iou < .3].id:
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid); H, W = gt.shape
    grad = C.gradient_map(bgr); gm = np.median(grad) + 1e-6
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    best, top = (0, 0, None), (0, -1, None)
    for ft in p["features"] or [p["feature"]]:
        f = C.feature_map(bgr, {**p, "feature": ft})
        f = cv2.GaussianBlur(f, (p["blur"],) * 2, 0) if p["blur"] else f
        for core in C.select_candidates(C.sweep_candidates(f.astype(np.float32), p), (H, W), {**p, "accept": 0})[:p["topk"]]:
            r = C.refine_watershed(bgr, [core], p, grad)[0]
            if r.sum() < 50: continue
            sc = C.region_score(r, grad, gm, L, p)   # same scoring as the segmenter (v1 or v2 per params)
            iou = (r.astype(bool) & gt).sum() / (r.astype(bool) | gt).sum()
            if iou > best[0]: best = (iou, sc, ft)
            if sc > top[1]: top = (iou, sc, ft)
    out.append(dict(id=sid.replace("not_damaged_", "nd"), best_iou=round(best[0], 2), best_score=round(best[1], 2),
                    best_view=best[2], chosen_iou=round(top[0], 2), chosen_score=round(top[1], 2), chosen_view=top[2]))
o = pd.DataFrame(out).sort_values("best_iou")
o.to_csv(f"experiments/runs/{run}/failure_diag.csv", index=False)
print(o.to_string(index=False))
print(f"\n{len(o)} failures | no usable egg region (best<0.5): {sum(o.best_iou < .5)} "
      f"| egg region existed but lost on score: {sum(o.best_iou >= .5)}")
