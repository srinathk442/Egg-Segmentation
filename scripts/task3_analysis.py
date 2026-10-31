"""Task 3 - evaluation & analysis of the classical method (predictions) against SAM (ground truth).

Reads the saved predictions (scripts/predict_classical.py) and writes to outputs/task3/:
  per_image.csv          metrics + difficulty factors + failure category for all 110 images
  summary.csv            aggregate metrics per split (with bootstrap 95% CI on mean IoU)
  by_egg_count.csv       single- vs multi-egg images
  fig_*.png              quantitative plots
  qual_*.jpg             qualitative grids (image | SAM GT | ours | error map)
"""
import json, os, sys
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import data, metrics

OUT = "outputs/task3"; PRED = "outputs/task2/pred_instance"
os.makedirs(OUT, exist_ok=True)

# ---------- palette (validated reference palette, light mode) ----------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_DEV, C_EVAL, C_GREY = "#2a78d6", "#eb6834", "#a3a29c"
TP_C, FP_C, FN_C = (0x7a, 0xaf, 0x1b), (0x48, 0x49, 0xe3), (0xd6, 0x78, 0x2a)   # BGR: aqua, red, blue
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

SCENES = {1: "printed lid (pink tub)", 2: "green leaves", 3: "white egg tray", 4: "tapioca pearls",
          5: "steel bowl, boiling water", 6: "carpet + earphones", 7: "white sheet with slot", 8: "laptop keyboard",
          9: "patterned cloth", 10: "printed plastic bag", 11: "foil packets", 12: "blue / red tub lid",
          13: "orange bowl", 14: "pink tub", 15: "checked cloth / carpet", 16: "wall corner",
          17: "hand-held, grey floor", 18: "dark room, hand", 19: "steel plate on tiles (A)", 20: "marble table, 2-3 eggs",
          21: "steel bowl, 2 eggs", 22: "steel plate (B)", 23: "white sheet (B)", 24: "grey floor, shadow", 25: "ash"}

# ---------- per-image metrics + difficulty factors ----------
split = pd.read_csv("splits/split.csv").set_index("id")
paths = data.image_paths()
rows = []
for sid in split.index:
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid, instance=True)
    pred = cv2.imread(f"{PRED}/{sid}.png", cv2.IMREAD_GRAYSCALE)
    m = metrics.all_metrics(pred, gt)
    g = (gt > 0).astype(np.uint8)
    ring = (cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) - g) > 0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = np.sqrt(np.maximum(cv2.boxFilter(gray ** 2, -1, (7, 7)) - cv2.boxFilter(gray, -1, (7, 7)) ** 2, 0))
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    rows.append(dict(id=sid, split=split.loc[sid, "split"], group=split.loc[sid, "group"],
                     scene=SCENES[split.loc[sid, "group"]], **m,
                     egg_area_frac=g.mean(),
                     L_contrast=lab[..., 0][g > 0].mean() - lab[..., 0][ring].mean(),     # egg vs. 20px surround
                     chroma_contrast=chroma[ring].mean() - chroma[g > 0].mean(),
                     bg_texture=tex[ring].mean(), egg_texture=tex[g > 0].mean()))
df = pd.DataFrame(rows)


def category(r):
    if r.iou < 0.3:
        return "wrong object / missed"
    if r.precision >= 0.85 and r.recall >= 0.85:
        return "good"
    if r.precision >= 0.85:
        return "under-segmented (part of egg)"
    if r.recall >= 0.85:
        return "over-segmented (leaks into bg)"
    return "partly wrong (both)"


df["category"] = df.apply(category, axis=1)
df.to_csv(f"{OUT}/per_image.csv", index=False)

# ---------- aggregate tables ----------
rng = np.random.default_rng(0)


def boot_ci(x, n=5000):
    x = np.asarray(x); bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


summ = []
for sp in ["dev", "eval"]:
    d = df[df.split == sp]
    s = metrics.summarize(d)
    lo, hi = boot_ci(d.iou)
    pooled = None
    summ.append(dict(split=sp, n_images=len(d), n_eggs=int(d.n_gt.sum()), **s,
                     iou_ci95=f"[{lo:.3f}, {hi:.3f}]", frac_iou_ge_0_7=round((d.iou >= .7).mean(), 3),
                     frac_failed_lt_0_3=round((d.iou < .3).mean(), 3)))
summ = pd.DataFrame(summ)
summ.to_csv(f"{OUT}/summary.csv", index=False)
print(summ.T.to_string())

df["eggs"] = np.where(df.n_gt > 1, "multi-egg", "single egg")
bycount = df.groupby(["split", "eggs"]).agg(n=("iou", "size"), iou=("iou", "mean"), count_acc=("n_pred", lambda s: 0),
                                          ).reset_index()
bycount["count_acc"] = [((df.split == r.split) & (df.eggs == r.eggs) & (df.n_pred == df.n_gt)).sum() / r.n
                        for r in bycount.itertuples()]
bycount["inst_f1"] = [2 * d.tp_inst.sum() / (d.n_pred.sum() + d.n_gt.sum())
                      for _, d in df.groupby(["split", "eggs"])]
bycount.round(3).to_csv(f"{OUT}/by_egg_count.csv", index=False)
print(bycount.round(3).to_string(index=False))
print(df.groupby(["split", "category"]).size().unstack(0).fillna(0).astype(int).to_string())

# ---------- Fig 1: experiment progression ----------
log = pd.read_csv("experiments/log.csv")
KEPT = ["T01_otsu_gray", "T02_cleanup", "T04_white_w4", "T07b_accept0.85", "T08a_watershed",
        "T10a_calibrated_score", "T10b_multiview", "T11_score_v2", "T12a_grow1.3", "T13c_smooth11"]
LABEL = {"T01_otsu_gray": "Otsu (gray)", "T04_white_w4": "whiteness", "T07b_accept0.85": "threshold sweep",
         "T08a_watershed": "watershed growth", "T10b_multiview": "3 views", "T11_score_v2": "score v2",
         "T13c_smooth11": "final"}
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(log))
kept = log.trial.isin(KEPT).values
best = np.maximum.accumulate(np.where(kept, log.iou, 0))
ax.step(x, best, where="post", color=C_DEV, lw=2, label="best kept so far")
ax.scatter(x[kept], log.iou[kept], s=40, color=C_DEV, zorder=3, edgecolor=SURF, linewidth=2, label="kept")
ax.scatter(x[~kept], log.iou[~kept], s=40, facecolor=SURF, edgecolor=C_GREY, linewidth=1.5, zorder=3,
           label="discarded")
for i, t in enumerate(log.trial):
    if t in LABEL:
        ax.annotate(LABEL[t], (i, log.iou[i]), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([t.replace("_", " ", 1) for t in log.trial], rotation=90, fontsize=7.5)
ax.set_ylabel("dev mean IoU"); ax.set_ylim(0.2, 0.8)
ax.set_title("Experiment log: dev mean IoU per trial")
ax.legend(frameon=False, loc="upper left")
fig.savefig(f"{OUT}/fig_progression.png"); plt.close(fig)

# ---------- Fig 2: IoU distribution (ECDF) ----------
fig, ax = plt.subplots(figsize=(6.4, 4))
for sp, c in [("dev", C_DEV), ("eval", C_EVAL)]:
    v = np.sort(df[df.split == sp].iou.values)
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2, label=f"{sp} (n={len(v)})")
ax.axvline(0.3, color=C_GREY, lw=1, ls="--"); ax.text(0.31, 0.03, "failure (IoU<0.3)", color=INK2, fontsize=8.5)
ax.set_xlabel("per-image IoU"); ax.set_ylabel("fraction of images with IoU ≤ x")
ax.set_title("Per-image IoU is bimodal: either the egg is found or it is not")
ax.legend(frameon=False, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.savefig(f"{OUT}/fig_iou_ecdf.png"); plt.close(fig)

# ---------- Fig 3: per-scene IoU ----------
sc = df.groupby(["scene", "split"]).agg(iou=("iou", "mean"), n=("iou", "size")).reset_index().sort_values("iou")
fig, ax = plt.subplots(figsize=(7.5, 7))
y = np.arange(len(sc))
ax.barh(y, sc.iou, color=[C_DEV if s == "dev" else C_EVAL for s in sc.split], height=0.7, edgecolor=SURF, linewidth=2)
for i, r in enumerate(sc.itertuples()):
    ax.text(r.iou + 0.01, i, f"{r.iou:.2f}  (n={r.n}, {r.split})", va="center", fontsize=8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(sc.scene, fontsize=8.5); ax.set_xlim(0, 1.25); ax.grid(axis="y", visible=False)
ax.set_xlabel("mean IoU"); ax.set_title("Mean IoU per scene group")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_DEV, label="dev scene"), Patch(color=C_EVAL, label="eval scene")],
          frameon=False, loc="lower right")
fig.savefig(f"{OUT}/fig_per_scene.png"); plt.close(fig)

# ---------- Fig 4: precision vs recall per image ----------
fig, ax = plt.subplots(figsize=(5.6, 5.2))
for sp, c in [("dev", C_DEV), ("eval", C_EVAL)]:
    d = df[df.split == sp]
    ax.scatter(d.recall, d.precision, s=38, color=c, edgecolor=SURF, linewidth=1.5, label=sp, alpha=0.9)
ax.axhline(0.85, color=C_GREY, lw=1, ls="--"); ax.axvline(0.85, color=C_GREY, lw=1, ls="--")
for txt, xy in [("good", (0.88, 1.035)), ("under-segmented (part of egg)", (0.30, 1.035)),
                ("leaks into\nbackground", (0.88, 0.45)), ("wrong object / missed", (0.05, 0.12))]:
    ax.text(*xy, txt, fontsize=8.5, color=INK2)
n0 = int(((df.recall < 0.01) & (df.precision < 0.01)).sum())
ax.annotate(f"{n0} images stacked here\n(egg not found at all)", (0, 0), xytext=(0.12, 0.25), fontsize=8.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=C_GREY, lw=1))
ax.set_xlabel("recall (share of SAM egg pixels found)"); ax.set_ylabel("precision (share of predicted pixels that are egg)")
ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.08); ax.set_title("Error type per image")
ax.legend(frameon=False, loc="center left")
fig.savefig(f"{OUT}/fig_precision_recall.png"); plt.close(fig)

# ---------- Fig 5: IoU vs difficulty factors ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, col, xl in zip(axes, ["L_contrast", "bg_texture", "egg_area_frac"],
                       ["egg brightness − surrounding brightness (Lab L)", "background texture (local std)",
                        "egg area / image area"]):
    for sp, c in [("dev", C_DEV), ("eval", C_EVAL)]:
        d = df[df.split == sp]
        ax.scatter(d[col], d.iou, s=30, color=c, edgecolor=SURF, linewidth=1.2, label=sp, alpha=0.9)
    ax.set_xlabel(xl)
    rho = pd.Series(df[col]).corr(df.iou, method="spearman")
    ax.set_title(f"Spearman ρ = {rho:+.2f}", fontsize=10, fontweight="normal", color=INK2)
axes[0].set_ylabel("per-image IoU"); axes[0].legend(frameon=False, loc="lower right")
axes[2].set_xscale("log"); axes[2].set_xticks([0.03, 0.05, 0.1, 0.2, 0.4])
axes[2].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].axvline(50, color=C_GREY, lw=1, ls="--")
fig.suptitle("What makes an image hard? IoU vs. difficulty factors (measured on SAM masks)", fontweight="bold", y=1.02)
fig.savefig(f"{OUT}/fig_difficulty.png"); plt.close(fig)

# ---------- Fig 6: failure categories ----------
cats = ["good", "under-segmented (part of egg)", "over-segmented (leaks into bg)", "partly wrong (both)",
        "wrong object / missed"]
cat_c = ["#1baf7a", "#2a78d6", "#eda100", "#6250d6", "#e34948"]
fig, ax = plt.subplots(figsize=(9, 2.4))
for j, sp in enumerate(["eval", "dev"]):
    d = df[df.split == sp]; left = 0
    for c, col in zip(cats, cat_c):
        w = (d.category == c).mean()
        ax.barh(j, w, left=left, color=col, edgecolor=SURF, linewidth=2, height=0.6,
                label=c if j == 0 else None)
        if w >= 0.06:
            ax.text(left + w / 2, j, f"{w:.0%}", ha="center", va="center", fontsize=8.5, color=SURF)
        left += w
ax.set_yticks([0, 1]); ax.set_yticklabels(["eval", "dev"]); ax.set_xlim(0, 1); ax.grid(visible=False)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8.5)
ax.set_title("Share of images per error category")
fig.savefig(f"{OUT}/fig_categories.png"); plt.close(fig)


# ---------- qualitative grids ----------
def put_label(img, text, org=(6, 24)):
    (w, h), _ = cv2.getTextSize(text, 0, 0.65, 2)
    cv2.rectangle(img, (org[0] - 4, org[1] - h - 6), (org[0] + w + 4, org[1] + 6), (20, 20, 20), -1)
    cv2.putText(img, text, org, 0, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def panel_row(sid, t=300):
    bgr = cv2.imread(paths[sid]); gt = data.load_gt(sid, instance=True) > 0
    pr = cv2.imread(f"{PRED}/{sid}.png", cv2.IMREAD_GRAYSCALE) > 0

    def contour(img, m, col):
        v = img.copy()
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cnts, -1, col, 3); return v
    err = (bgr * 0.35).astype(np.uint8)
    err[gt & pr] = TP_C; err[~gt & pr] = FP_C; err[gt & ~pr] = FN_C
    tiles = [bgr, contour(bgr, gt, (0x7a, 0xaf, 0x1b)), contour(bgr, pr, FP_C), err]
    tiles = [cv2.resize(x, (t, t)) for x in tiles]
    r = df[df.id == sid].iloc[0]
    put_label(tiles[0], f"{sid.replace('not_damaged_', 'nd')} ({r.split}, {r.scene})")
    put_label(tiles[2], f"IoU {r.iou:.2f}")
    return np.hstack(tiles)


def grid(ids, name, title):
    t = 300
    head = np.full((34, 4 * t, 3), 252, np.uint8)
    for k, h in enumerate(["image", "SAM (ground truth)", "classical (ours)", "error: TP aqua / FP red / FN blue"]):
        cv2.putText(head, h, (k * t + 6, 23), 0, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
    img = np.vstack([head] + [panel_row(s, t) for s in ids])
    cv2.imwrite(f"{OUT}/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])


ev = df[df.split == "eval"].sort_values("iou")
dv = df[df.split == "dev"].sort_values("iou")
grid(ev.id[:6].tolist(), "qual_worst_eval", "worst eval")
# dev failures: one per failure scene type
pick = []
for g in dv[dv.iou < 0.3].group.unique():
    pick.append(dv[(dv.group == g)].iloc[0].id)
grid(pick[:8], "qual_worst_dev", "worst dev")
# typical partial errors (IoU 0.3-0.8)
grid(df[(df.iou >= 0.3) & (df.iou < 0.8)].sort_values("iou").id[:5].tolist(), "qual_partial", "partial")
# successes incl. touching eggs
succ = df[(df.iou > 0.9)]
grid([succ[succ.n_gt > 1].sort_values("iou").id.iloc[-1], succ[succ.n_gt > 1].sort_values("iou").id.iloc[0]]
     + succ[succ.n_gt == 1].sample(2, random_state=1).id.tolist(), "qual_success", "success")
print("figures written to", OUT)
