"""Extra report figures: Task 1 QC before/after, and a stage-by-stage walkthrough of the classical algorithm."""
import json, os, sys
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical as C, data

OUT = "outputs/report_figs"; os.makedirs(OUT, exist_ok=True)


def put_label(img, text, org=(8, 26), scale=0.7):
    (w, h), _ = cv2.getTextSize(text, 0, scale, 2)
    cv2.rectangle(img, (org[0] - 5, org[1] - h - 7), (org[0] + w + 5, org[1] + 7), (20, 20, 20), -1)
    cv2.putText(img, text, org, 0, scale, (255, 255, 255), 2, cv2.LINE_AA)


# ---- 1. Task 1 QC: first run vs corrected run ----
rows = []
for i in [22, 39, 117]:
    sid = f"not_damaged_{i}"
    a = cv2.resize(cv2.imread(f"outputs/task1_outputs_v1_before_qc/overlays/{sid}.jpg"), (360, 360))
    b = cv2.resize(cv2.imread(f"outputs/task1_outputs/overlays/{sid}.jpg"), (360, 360))
    put_label(a, f"nd{i}: first run"); put_label(b, f"nd{i}: corrected prompts")
    rows.append(np.hstack([a, np.full((360, 8, 3), 255, np.uint8), b]))
cv2.imwrite(f"{OUT}/task1_qc_fixes.jpg", np.vstack([np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---- 2. algorithm walkthrough on one dev image ----
P = {**C.DEFAULTS, **json.load(open("configs/final_classical.json"))}
sid = "not_damaged_105"
bgr = cv2.imread(data.image_paths()[sid]); H, W = bgr.shape[:2]
t = 320
tiles = []
im = bgr.copy(); put_label(im, "1. input"); tiles.append(im)
views = {}
for ft in P["features"]:
    f = C.feature_map(bgr, {**P, "feature": ft}); f = cv2.GaussianBlur(f, (P["blur"],) * 2, 0)
    views[ft] = f
    v = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR); put_label(v, f"2. view: {ft}"); tiles.append(v)
# candidate cores
grad = C.gradient_map(bgr); gm = np.median(grad) + 1e-6
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
L, chroma = lab[..., 0], np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
cores = []
for ft in P["features"]:
    for c in C.select_candidates(C.sweep_candidates(views[ft].astype(np.float32), P), (H, W), {**P, "accept": 0})[:P["topk"]]:
        if all(np.logical_and(c, o).sum() / np.logical_or(c, o).sum() < 0.7 for o in cores):
            cores.append(c)
v = (bgr * 0.5).astype(np.uint8)
for k, c in enumerate(cores):
    cnts, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(v, cnts, -1, (0, 255, 255), 2)
put_label(v, f"3. sweep: {len(cores)} candidate cores"); tiles.append(v)
# grown regions with score
v = (bgr * 0.5).astype(np.uint8)
for c in cores:
    r = C.refine_watershed(bgr, [c], P, grad)[0]
    s = C.region_score(r, grad, gm, L, P, chroma)
    col = (0x7a, 0xaf, 0x1b) if s >= P["accept"] else (0x48, 0x49, 0xe3)
    cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(v, cnts, -1, col, 2)
    ys, xs = np.nonzero(r)
    if len(xs):
        cv2.putText(v, f"{s:.2f}", (int(xs.mean()) - 20, int(ys.mean())), 0, 0.6, col, 2, cv2.LINE_AA)
put_label(v, "4. watershed growth + score"); tiles.append(v)
# final
inst = C.segment(bgr, P); v = bgr.copy()
for k in range(1, inst.max() + 1):
    cnts, _ = cv2.findContours((inst == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(v, cnts, -1, [(0xd6, 0x78, 0x2a), (0x34, 0x68, 0xeb), (0x7a, 0xaf, 0x1b)][(k - 1) % 3], 3)
put_label(v, f"5. final: {inst.max()} eggs"); tiles.append(v)
gt = data.load_gt(sid, instance=True); v = bgr.copy()
cnts, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(v, cnts, -1, (0x7a, 0xaf, 0x1b), 3); put_label(v, "SAM ground truth"); tiles.append(v)
tiles = [cv2.resize(x, (t, t)) for x in tiles]
gap = np.full((t, 6, 3), 255, np.uint8)
row1 = np.hstack(sum([[x, gap] for x in tiles[:4]], [])[:-1])
row2 = np.hstack(sum([[x, gap] for x in tiles[4:8]], [])[:-1])
cv2.imwrite(f"{OUT}/algorithm_walkthrough.jpg", np.vstack([row1, np.full((6, row1.shape[1], 3), 255, np.uint8), row2]),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print("written", os.listdir(OUT))
