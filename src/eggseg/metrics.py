"""Segmentation metrics. pred / gt are boolean HxW masks (or instance label maps where noted).

Pixel level (semantic: egg vs background)
  TP = egg in both, FP = predicted egg but background in GT, FN = missed egg pixels
  IoU (Jaccard) = TP / (TP+FP+FN)        Dice (F1) = 2TP / (2TP+FP+FN)
  precision = TP/(TP+FP)  "how much of what I called egg is egg"
  recall    = TP/(TP+FN)  "how much of the egg did I find"
Boundary level
  boundary F1 = F1 of contour pixels matched within `tol` px -> sensitive to outline quality,
  which IoU on big eggs hides (a 3-px offset barely changes IoU of a large egg).
Instance level (counting / separating eggs)
  each GT egg is matched to at most one predicted egg with IoU >= 0.5
"""
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def pixel_metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    eps = 1e-9
    return dict(iou=tp / (tp + fp + fn + eps), dice=2 * tp / (2 * tp + fp + fn + eps),
                precision=tp / (tp + fp + eps) if (tp + fp) else 0.0, recall=tp / (tp + fn + eps))


def _boundary(m):
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f1(pred, gt, tol=5):
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return float(bp.sum() == bg.sum())
    # distance from every pixel to the nearest boundary pixel of the other mask
    d_to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    d_to_pr = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    p = (d_to_gt[bp] <= tol).mean()
    r = (d_to_pr[bg] <= tol).mean()
    return 2 * p * r / (p + r + 1e-9)


def instance_metrics(pred_inst, gt_inst, thr=0.5):
    """pred_inst / gt_inst: label maps (0 = background, 1..N = eggs)."""
    P = [pred_inst == k for k in range(1, int(pred_inst.max()) + 1) if (pred_inst == k).any()]
    G = [gt_inst == k for k in range(1, int(gt_inst.max()) + 1) if (gt_inst == k).any()]
    if not P or not G:
        return dict(n_pred=len(P), n_gt=len(G), tp_inst=0, matched_iou=np.nan)
    M = np.array([[pixel_metrics(p, g)["iou"] for p in P] for g in G])
    r, c = linear_sum_assignment(-M)                 # best one-to-one matching
    ok = M[r, c] >= thr
    return dict(n_pred=len(P), n_gt=len(G), tp_inst=int(ok.sum()),
                matched_iou=float(M[r, c][ok].mean()) if ok.any() else np.nan)


def all_metrics(pred_inst, gt_inst):
    pred, gt = pred_inst > 0, gt_inst > 0
    out = pixel_metrics(pred, gt)
    out["bf1"] = boundary_f1(pred, gt)
    out.update(instance_metrics(pred_inst, gt_inst))
    return out


def summarize(df):
    """Aggregate a per-image metrics DataFrame into one row (mean over images + instance P/R/F1)."""
    s = {k: df[k].mean() for k in ["iou", "dice", "precision", "recall", "bf1"]}
    s["iou_median"] = df.iou.median()
    s["iou_p10"] = df.iou.quantile(0.10)
    tp, npred, ngt = df.tp_inst.sum(), df.n_pred.sum(), df.n_gt.sum()
    s["inst_precision"] = tp / max(npred, 1)
    s["inst_recall"] = tp / max(ngt, 1)
    s["inst_f1"] = 2 * tp / max(npred + ngt, 1)
    s["count_acc"] = (df.n_pred == df.n_gt).mean()
    return {k: round(float(v), 4) for k, v in s.items()}
