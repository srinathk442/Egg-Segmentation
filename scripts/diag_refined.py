"""Diagnostic for refine-then-score (dev only): grow the top-K cores with watershed and check
 (1) oracle: best refined IoU per image (can the right egg region be produced at all?)
 (2) whether the score components rank the true egg first, and which component separates best."""
import json, os, sys
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical as C, data
from sklearn.metrics import roc_auc_score

p = {**C.DEFAULTS, **json.loads(sys.argv[1])}
paths = data.image_paths()
rows, per_img = [], []
for sid in data.split_ids("dev"):
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid, instance=True); H, W = gt.shape
    f = C.feature_map(bgr, p)
    f = cv2.GaussianBlur(f, (p["blur"],) * 2, 0) if p["blur"] else f
    cands = C.sweep_candidates(f.astype(np.float32), p)
    cores = C.select_candidates(cands, (H, W), {**p, "accept": 0.0})[:p["topk"]]
    grad = C.gradient_map(bgr); g_med = np.median(grad) + 1e-6
    best = 0
    for core in cores:
        r = C.refine_watershed(bgr, [core], p, grad)[0]
        if r.sum() < 50: continue
        s = C.blob_shape(r)
        bd = r - cv2.erode(r, C._kernel(3))
        ious = [np.logical_and(r, gt == k).sum() / np.logical_or(r, gt == k).sum() for k in range(1, gt.max() + 1)]
        core_ious = [np.logical_and(core, gt == k).sum() / np.logical_or(core, gt == k).sum() for k in range(1, gt.max() + 1)]
        rows.append(dict(id=sid, iou=max(ious), core_iou=max(core_ious), ell=s["ellipse_iou"], sol=s["solidity"],
                         edge=grad[bd > 0].mean() / g_med, logarea=np.log10(r.sum() / (H * W)),
                         inside_grad=grad[cv2.erode(r, C._kernel(7)) > 0].mean() / g_med if cv2.erode(r, C._kernel(7)).any() else 9,
                         touches_border=float(r[0].any() or r[-1].any() or r[:, 0].any() or r[:, -1].any())))
        best = max(best, max(ious))
    per_img.append(best)
df = pd.DataFrame(rows)
per_img = np.array(per_img)
print(f"refined oracle (best of top-{p['topk']}): mean IoU {per_img.mean():.3f}, median {np.median(per_img):.3f}, "
      f">=0.7: {(per_img >= .7).mean():.2f}")
df["is_egg"] = df.iou >= 0.6
print(f"regions: {len(df)}, egg-like: {df.is_egg.sum()}")
for col in ["ell", "sol", "edge", "logarea", "inside_grad", "touches_border", "core_iou"]:
    if col == "core_iou": continue
    auc = roc_auc_score(df.is_egg, df[col])
    print(f"  {col:15s} AUC {max(auc, 1 - auc):.3f} ({'higher' if auc >= .5 else 'lower'} for eggs)  "
          f"egg median {df[df.is_egg][col].median():.2f}  other median {df[~df.is_egg][col].median():.2f}")
df.to_csv("outputs/task2/diag_refined.csv", index=False)
