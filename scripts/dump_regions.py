"""Dump every refined candidate region of every dev image with its features + IoU vs GT.
Used to design / compare region-scoring formulas offline (fast) instead of re-running segmentation.

usage: python scripts/dump_regions.py T10b_multiview   -> outputs/task2/regions_<run>.csv
"""
import json, os, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical as C, data

run = sys.argv[1]
p = {**C.DEFAULTS, **json.load(open(f"experiments/runs/{run}/summary.json"))["params"]}
paths = data.image_paths()
rows = []
for sid in data.split_ids("dev"):
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid, instance=True); H, W = gt.shape
    grad = C.gradient_map(bgr); gm = np.median(grad) + 1e-6
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = np.sqrt(np.maximum(cv2.boxFilter(gray ** 2, -1, (7, 7)) - cv2.boxFilter(gray, -1, (7, 7)) ** 2, 0))
    cores = []
    for ft in p["features"] or [p["feature"]]:
        f = C.feature_map(bgr, {**p, "feature": ft})
        f = cv2.GaussianBlur(f, (p["blur"],) * 2, 0) if p["blur"] else f
        for c in C.select_candidates(C.sweep_candidates(f.astype(np.float32), p), (H, W), {**p, "accept": 0})[:p["topk"]]:
            if all(np.logical_and(c, o).sum() / np.logical_or(c, o).sum() < 0.7 for o, _ in cores):
                cores.append((c, ft))
    for core, ft in cores:
        r = C.refine_watershed(bgr, [core], p, grad)[0]
        if r.sum() < 50: continue
        s = C.blob_shape(r)
        bd = (r - cv2.erode(r, C._kernel(3))) > 0
        inner = cv2.erode(r, C._kernel(9)) > 0
        ring = (cv2.dilate(r, C._kernel(15)) - r) > 0
        rb = r > 0
        ious = [np.logical_and(rb, gt == k).sum() / np.logical_or(rb, gt == k).sum() for k in range(1, gt.max() + 1)]
        rows.append(dict(
            id=sid, view=ft, iou=max(ious),
            ell=s["ellipse_iou"], sol=s["solidity"],
            edge_rel=grad[bd].mean() / gm,                                  # boundary grad / image median grad
            edge_abs=grad[bd].mean(),                                       # boundary grad (absolute)
            edge_vs_ring=grad[bd].mean() / (grad[ring].mean() + 1e-6),       # boundary vs. just-outside texture
            inside_tex=tex[inner].mean() if inner.any() else 99,            # eggs are smooth inside
            ring_tex=tex[ring].mean() if ring.any() else 0,
            L_in=lab[..., 0][rb].mean(), L_ring=lab[..., 0][ring].mean() if ring.any() else 0,
            chroma_in=chroma[rb].mean(), chroma_ring=chroma[ring].mean() if ring.any() else 0,
            logarea=np.log10(rb.mean()),
            border=float(r[0].any() or r[-1].any() or r[:, 0].any() or r[:, -1].any())))
df = pd.DataFrame(rows)
os.makedirs("outputs/task2", exist_ok=True)
df.to_csv(f"outputs/task2/regions_{run}.csv", index=False)
print(len(df), "regions from", df.id.nunique(), "images; oracle mean best IoU:", round(df.groupby("id").iou.max().mean(), 3))
