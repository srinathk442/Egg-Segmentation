"""Classical (no deep learning) egg segmentation.

segment(bgr, params) -> instance label map (uint8, 0 = background, 1..N = eggs)

The pipeline is a sequence of stages, each controlled by a parameter, so every trial of the
experiment log is reproducible from its parameter dict:
  1. feature map   - turn the colour image into one "egg-likeness" channel
  2. smoothing     - suppress noise / texture before thresholding
  3. threshold     - per-image Otsu (adaptive) or fixed
  4. morphology    - open (remove specks), close (bridge gaps), fill holes
  5. components    - each connected blob = one egg candidate, filtered by size
"""
import numpy as np
import cv2
from scipy import ndimage as ndi

DEFAULTS = dict(
    feature="gray",        # gray | L | V | whiteness
    blur=0,                # Gaussian kernel size (0 = off)
    thresh="otsu",         # otsu | fixed
    fixed_thr=200,
    open_k=0,              # morphological opening kernel (0 = off)
    close_k=0,             # morphological closing kernel (0 = off)
    fill_holes=False,
    min_area_frac=0.0,     # drop blobs smaller than this fraction of the image
    keep_largest=False,    # keep only the biggest blob
    chroma_w=2.0,          # whiteness = L - chroma_w * chroma
    shape_filter=False,    # keep only egg-shaped blobs (convex, ellipse-like)
    min_solidity=0.85,     # area / convex hull area
    min_ellipse_iou=0.75,  # IoU between blob and its fitted ellipse
    complete="none",       # none | hull | ellipse : replace blob by its convex hull / fitted ellipse
    # --- threshold sweep (mode="sweep") ---
    mode="single",         # single : one threshold | sweep : many thresholds + egg-likeness scoring
    n_levels=24,           # number of thresholds tried between the feature's 5th and 99.5th percentile
    max_area_frac=0.6,     # candidates larger than this are background
    ring=7,                # width (px) of the ring around a blob used for the contrast score
    contrast_scale=30.0,   # contrast (feature units) that counts as "fully separated"
    accept=0.45,           # min egg-likeness score for extra eggs (the best one is always kept)
    max_overlap=0.2,       # max intersection / smaller-area between accepted eggs
    # --- refinement of each selected egg core to the real boundary ---
    refine="none",         # none | watershed | grabcut
    grow=1.6,              # search region = core's fitted ellipse scaled by this factor
    core_erode=5,          # erode the core before using it as sure-foreground marker
    gc_iters=5,
    # --- size prior + refine-then-score selection ---
    size_prior=False,      # multiply scores by a Gaussian prior on log10(egg area / image area)
    size_mu=-1.03,         # fitted on dev GT eggs (median egg = 9.8% of the image)
    size_sd=0.37,          # 1.5 x the dev std (0.245) to stay tolerant
    rescore=False,         # refine top-K candidates first, then score the refined regions
    topk=8,
    edge_ratio=3.0,        # boundary gradient / image median gradient that counts as a full edge
    # --- multi-view candidates + calibrated refined score (T10+) ---
    features=None,         # list of feature maps to sweep, e.g. ["whiteness", "darkness", "smooth"]
    ell_pow=1.0,           # refined score = ellipse_iou**ell_pow * solidity * edge * size * border
    rs_size_sd=0.37,       # size prior width used on refined regions
    border_pen=0.0,        # score multiplied by (1 - border_pen) if the region touches the image border
    tex_win=15,            # window for local std (texture) in the "smooth" feature
    score_v=1,             # region score version (see region_score)
    edge_ring_scale=3.0,   # v2: boundary/ring gradient ratio that counts as a full edge
    lc_scale=8.0,          # v2: brightness contrast (L units) of the sigmoid
    post_smooth=0,         # final open+close kernel on each egg mask (0 = off)
    min_score=0.0,         # reject even the best region below this score (0 = always keep one egg)
    min_lcon=None,         # if set: region must be at least this much brighter (Lab L) than its surroundings
    max_chroma=None,       # if set: reject regions whose mean Lab chroma exceeds this (colourful objects)
)


def feature_map(bgr, p):
    if p["feature"] == "gray":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if p["feature"] == "L":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    if p["feature"] == "V":
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    if p["feature"] == "whiteness":
        # bright AND colourless: L minus a penalty for chroma (distance from neutral grey)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
        w = lab[..., 0] - p.get("chroma_w", 2.0) * chroma
        return cv2.normalize(w, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if p["feature"] == "darkness":
        # egg slightly greyer than a white background (paper, tray)
        return 255 - feature_map(bgr, {**p, "feature": "whiteness"})
    if p["feature"] == "smooth":
        # egg shell is untextured: high where the local standard deviation of brightness is low
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = (p["tex_win"], p["tex_win"])
        std = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, k) - cv2.boxFilter(g, -1, k) ** 2, 0))
        return 255 - cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    raise ValueError(p["feature"])


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(m):
    """Fill enclosed holes: redraw every outer contour filled (same result as scipy's binary_fill_holes
    for 8-connected blobs, ~10x faster)."""
    m = m.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def blob_shape(blob):
    """Shape descriptors of one binary blob: solidity, ellipse-fit IoU, the hull and ellipse masks."""
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = blob.sum()
    hull = cv2.convexHull(c)
    hull_m = np.zeros_like(blob); cv2.fillPoly(hull_m, [hull], 1)
    ell_m = np.zeros_like(blob)
    if len(c) >= 5:
        cv2.ellipse(ell_m, cv2.fitEllipse(c), 1, -1)
    inter = np.logical_and(blob, ell_m).sum()
    return dict(solidity=area / max(hull_m.sum(), 1),
                ellipse_iou=inter / max(np.logical_or(blob, ell_m).sum(), 1),
                hull=hull_m, ellipse=ell_m)


def size_term(area_frac, p):
    z = (np.log10(max(area_frac, 1e-6)) - p["size_mu"]) / p["size_sd"]
    return float(np.exp(-0.5 * z * z))


def sweep_candidates(f, p):
    """Threshold the feature map at many levels; every blob at every level is an egg candidate.

    Egg-likeness score of a blob = ellipse_iou * solidity * contrast_term, where
      ellipse_iou : how well a fitted ellipse explains the blob (eggs are ellipses)
      solidity    : area / convex-hull area (eggs are convex, no bites or tentacles)
      contrast    : mean feature inside the blob - mean in a thin ring just outside it,
                    squashed to [0,1]; a real object boundary has a jump, a random level-set cut does not
    """
    H, W = f.shape
    lo, hi = np.percentile(f, [5, 99.5])
    ring_k = _kernel(2 * p["ring"] + 1)
    open_k = _kernel(p["open_k"]) if p["open_k"] else None
    cands = []
    for t in np.linspace(lo, hi, p["n_levels"] + 2)[1:-1]:
        m = (f >= t).astype(np.uint8)
        if open_k is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if not (p["min_area_frac"] * H * W <= area <= p["max_area_frac"] * H * W):
                continue
            # work in a padded crop around the blob for speed
            pad = p["ring"] + 2
            x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
            blob = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
            blob = fill_holes(blob)
            s = blob_shape(blob)
            ring = cv2.dilate(blob, ring_k) - blob
            fc = f[y0:y1, x0:x1]
            contrast = fc[blob > 0].mean() - (fc[ring > 0].mean() if ring.any() else 0)
            c_term = float(np.clip(contrast / p["contrast_scale"], 0, 1))
            score = s["ellipse_iou"] * s["solidity"] * c_term
            if p["size_prior"]:
                score *= size_term(blob.sum() / (H * W), p)
            cands.append(dict(score=score, box=(x0, y0, x1, y1), mask=blob, area=int(blob.sum())))
    return cands


def select_candidates(cands, shape, p):
    """Greedy non-maximum suppression on masks: best score first, skip candidates overlapping an accepted one."""
    H, W = shape

    def overlap(c, a):  # intersection / smaller area, computed only where the two boxes intersect
        x0, y0 = max(c["box"][0], a["box"][0]), max(c["box"][1], a["box"][1])
        x1, y1 = min(c["box"][2], a["box"][2]), min(c["box"][3], a["box"][3])
        if x0 >= x1 or y0 >= y1:
            return 0.0
        mc = c["mask"][y0 - c["box"][1]:y1 - c["box"][1], x0 - c["box"][0]:x1 - c["box"][0]]
        ma = a["mask"][y0 - a["box"][1]:y1 - a["box"][1], x0 - a["box"][0]:x1 - a["box"][0]]
        return np.count_nonzero(mc & ma) / min(c["area"], a["area"])

    accepted = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        if accepted and c["score"] < p["accept"]:
            break
        if all(overlap(c, a) <= p["max_overlap"] for a in accepted):
            accepted.append(c)
    out = []
    for c in accepted:
        full = np.zeros((H, W), np.uint8)
        x0, y0, x1, y1 = c["box"]; full[y0:y1, x0:x1] = c["mask"]
        out.append(full)
    return out


def _grown_ellipse(core, scale, shape):
    cnts, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    m = np.zeros(shape, np.uint8)
    if len(c) < 5:
        return cv2.dilate(core.astype(np.uint8), _kernel(15))
    (cx, cy), (a, b), ang = cv2.fitEllipse(c)
    cv2.ellipse(m, ((cx, cy), (a * scale, b * scale), ang), 1, -1)
    return m | core.astype(np.uint8)


def gradient_map(bgr):
    """Colour gradient magnitude: Sobel on L, a, b (an edge can be a brightness OR a colour change)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    g = np.zeros(bgr.shape[:2], np.float32)
    for c in range(3):
        g += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    return np.sqrt(g)


def refine_watershed(bgr, cores, p, grad=None):
    """Marker-controlled watershed on the colour gradient.
    markers: 1 = background (outside every grown ellipse), 2..N+1 = eroded egg cores."""
    from skimage.segmentation import watershed
    H, W = bgr.shape[:2]
    grad = gradient_map(bgr) if grad is None else grad
    markers = np.zeros((H, W), np.int32)
    search = np.zeros((H, W), np.uint8)
    for c in cores:
        search |= _grown_ellipse(c, p["grow"], (H, W))
    markers[search == 0] = 1
    for i, c in enumerate(cores, 2):
        e = cv2.erode(c.astype(np.uint8), _kernel(p["core_erode"])) if p["core_erode"] else c
        markers[(e if e.any() else c) > 0] = i
    # everything outside the search region is already a fixed background marker, so flooding only
    # a crop around it gives the identical result, much faster
    ys, xs = np.where(search > 0)
    y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, H)
    x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, W)
    lab = np.ones((H, W), np.int32)
    lab[y0:y1, x0:x1] = watershed(grad[y0:y1, x0:x1], markers[y0:y1, x0:x1])
    return [fill_holes(lab == i) for i in range(2, len(cores) + 2)]


def region_score(region, grad, g_med, L, p, chroma=None):
    """Egg-likeness of a grown region (product of terms in [0,1]).
    v1 (T09-T10): ellipse^a * solidity * edge(boundary grad / image median grad) * size * border
    v2 (T11+)   : edge term replaced by boundary grad / grad in a ring just outside (robust on textured
                  backgrounds such as carpet or ash) and a brightness-contrast term added
                  (egg brighter than its surroundings; single best cue on dev, AUC 0.92)."""
    H, W = region.shape
    s = blob_shape(region)
    boundary = (region - cv2.erode(region, _kernel(3))) > 0
    if not boundary.any():
        return 0.0
    border = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
    common = (s["ellipse_iou"] ** p["ell_pow"] * s["solidity"]
              * size_term(region.sum() / (H * W), {**p, "size_sd": p["rs_size_sd"]})
              * (1 - p["border_pen"] * border))
    if p["score_v"] == 1:
        return common * float(np.clip(grad[boundary].mean() / (p["edge_ratio"] * g_med), 0, 1))
    ring = (cv2.dilate(region, _kernel(15)) - region) > 0
    if not ring.any():
        return 0.0
    edge = np.clip(grad[boundary].mean() / (grad[ring].mean() + 1e-6) / p["edge_ring_scale"], 0, 1)
    l_con = L[region > 0].mean() - L[ring].mean()
    # optional hard constraints (off by default; used for the video, see configs/video.json)
    if p["min_lcon"] is not None and l_con < p["min_lcon"]:
        return 0.0                      # must be brighter than its surroundings
    if p["max_chroma"] is not None and chroma is not None and chroma[region > 0].mean() > p["max_chroma"]:
        return 0.0                      # eggs are white/cream: dataset eggs have mean chroma <= 11.3
    return float(common * edge / (1 + np.exp(-l_con / p["lc_scale"])))


def select_refined(bgr, cands, p):
    """Refine-then-score: grow the top-K distinct candidates with watershed, score the grown regions
    (ellipse fit x solidity x edge support x size prior), then pick eggs greedily and do a joint
    watershed with all chosen cores so touching eggs are split between them."""
    H, W = bgr.shape[:2]
    grad = gradient_map(bgr)
    g_med = np.median(grad) + 1e-6
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, chroma = lab[..., 0], np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    # top-K distinct cores per feature view (mask NMS with accept=0 so we keep K alternatives)
    cand_lists = cands if cands and isinstance(cands[0], list) else [cands]
    cores = []
    for cl in cand_lists:
        for c in select_candidates(cl, (H, W), {**p, "accept": 0.0})[:p["topk"]]:
            # skip near-duplicates of cores already found by another view
            if all(np.logical_and(c, o).sum() / np.logical_or(c, o).sum() < 0.7 for o in cores):
                cores.append(c)
    scored = []
    for core in cores:
        region = refine_watershed(bgr, [core], p, grad)[0]
        if region.sum() < 50:
            continue
        scored.append((region_score(region, grad, g_med, L, p, chroma), core, region))
    chosen = []
    for score, core, region in sorted(scored, key=lambda t: -t[0]):
        if (chosen and score < p["accept"]) or score < p["min_score"]:
            break
        if all(np.logical_and(region, r).sum() / min(region.sum(), r.sum()) <= p["max_overlap"] for _, r in chosen):
            chosen.append((core, region))
    if len(chosen) <= 1:
        return [r for _, r in chosen]
    return refine_watershed(bgr, [c for c, _ in chosen], p, grad)


def refine_grabcut(bgr, cores, p):
    """GrabCut per egg: colour GMMs for egg/background + graph min-cut that prefers cutting at edges.
    Init: sure-FG = eroded core, probable-FG = core, probable-BG = grown ellipse,
    sure-BG = outside grown ellipse and the other eggs' cores."""
    H, W = bgr.shape[:2]
    out = []
    for i, c in enumerate(cores):
        grown = _grown_ellipse(c, p["grow"], (H, W))
        m = np.full((H, W), cv2.GC_BGD, np.uint8)
        m[grown > 0] = cv2.GC_PR_BGD
        m[c > 0] = cv2.GC_PR_FGD
        e = cv2.erode(c.astype(np.uint8), _kernel(p["core_erode"])) if p["core_erode"] else c
        m[(e if e.any() else c) > 0] = cv2.GC_FGD
        for j, o in enumerate(cores):
            if j != i: m[o > 0] = cv2.GC_BGD
        ys, xs = np.where(grown > 0)
        pad = 10
        y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, H), max(xs.min() - pad, 0), min(xs.max() + pad, W)
        crop = m[y0:y1, x0:x1].copy()
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(bgr[y0:y1, x0:x1], crop, None, bgd, fgd, p["gc_iters"], cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            out.append(c.astype(np.uint8)); continue
        fg = np.isin(crop, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
        full = np.zeros((H, W), np.uint8); full[y0:y1, x0:x1] = fg
        # keep the connected piece that contains the core
        n, lab = cv2.connectedComponents(full)
        ids = np.unique(lab[(c > 0) & (full > 0)])
        full = np.isin(lab, ids[ids > 0]).astype(np.uint8) if len(ids) else c.astype(np.uint8)
        out.append(fill_holes(full))
    return out


def segment(bgr, params=None):
    p = {**DEFAULTS, **(params or {})}
    H, W = bgr.shape[:2]

    def prep(feat):
        f = feature_map(bgr, {**p, "feature": feat})
        return cv2.GaussianBlur(f, (p["blur"], p["blur"]), 0) if p["blur"] else f

    f = prep(p["feature"])

    if p["mode"] == "sweep":
        if p["features"]:
            cands = [sweep_candidates(prep(ft).astype(np.float32), p) for ft in p["features"]]
        else:
            cands = sweep_candidates(f.astype(np.float32), p)
        if p["rescore"]:
            blobs = select_refined(bgr, cands, p)
        else:
            blobs = select_candidates(cands, (H, W), p)
        if p["rescore"]:
            pass
        elif p["refine"] == "watershed" and blobs:
            blobs = refine_watershed(bgr, blobs, p)
        elif p["refine"] == "grabcut" and blobs:
            blobs = refine_grabcut(bgr, blobs, p)
        if p["complete"] in ("hull", "ellipse"):
            blobs = [blob_shape(b)[p["complete"]] for b in blobs]
        if p["post_smooth"]:
            k = _kernel(p["post_smooth"])
            blobs = [fill_holes(cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k))
                     for b in blobs]
        inst = np.zeros((H, W), np.uint8)
        for i, b in enumerate(sorted(blobs, key=lambda b: -int(b.sum()))[:255], 1):
            inst[b > 0] = i
        return inst

    if p["thresh"] == "otsu":
        _, m = cv2.threshold(f, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        m = (f >= p["fixed_thr"]).astype(np.uint8)

    if p["open_k"]:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(p["open_k"]))
    if p["close_k"]:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(p["close_k"]))
    if p["fill_holes"]:
        m = ndi.binary_fill_holes(m).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= p["min_area_frac"] * H * W]
    if p["keep_largest"] and keep:
        keep = [max(keep, key=lambda k: stats[k, cv2.CC_STAT_AREA])]

    blobs = []
    for k in keep:
        b = (lab == k).astype(np.uint8)
        if p["shape_filter"] or p["complete"] != "none":
            s = blob_shape(b)
            if p["shape_filter"] and (s["solidity"] < p["min_solidity"] or s["ellipse_iou"] < p["min_ellipse_iou"]):
                continue
            if p["complete"] in ("hull", "ellipse"):
                b = s[p["complete"]]
        blobs.append(b)

    inst = np.zeros((H, W), np.uint8)
    for i, b in enumerate(sorted(blobs, key=lambda b: -int(b.sum()))[:255], 1):  # large first
        inst[b > 0] = i
    return inst
