"""Task 2a - leakage-aware dev/eval split (4:1).

Idea: photos of the same scene (same bowl / table / hand / eggs) are near-duplicates. If one lands in
dev and its twin in eval, tuning on dev "sees" the eval scene -> optimistic eval score (leakage).
So we split by *scene group*, not by image:
  1. DINOv2 embedding per image (computed in the Task 1 notebook), cosine similarity = scene similarity
  2. agglomerative clustering (average linkage) -> groups whose members are >= SIM_THR similar on average
  3. assign whole groups to eval until ~20% of images, choosing among many random assignments the one
     that best matches the dev egg-count distribution (single vs multi-egg images)
  4. report leakage: highest dev<->eval similarity before (random split) vs after (group split)
"""
import argparse, json, os
import numpy as np, pandas as pd, cv2
from scipy.cluster.hierarchy import linkage, fcluster

ap = argparse.ArgumentParser()
ap.add_argument("--task1", default="outputs/task1_outputs")
ap.add_argument("--images", default="data/raw/egg")
ap.add_argument("--out", default="splits")
ap.add_argument("--sim_thr", type=float, default=0.70)
ap.add_argument("--eval_frac", type=float, default=0.20)
ap.add_argument("--trials", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_manual", action="store_true", help="skip the manual merges below")
a = ap.parse_args()

# Manual review of split_groups.jpg (sim_thr=0.70): same physical scene, but clustering left them apart.
# Each tuple = images that must end up in the same group.
MANUAL_MERGES = [
    ("not_damaged_104", "not_damaged_106", "not_damaged_111"),  # white sheet with slot, behind it a box
    ("not_damaged_43", "not_damaged_18"),                        # laptop keyboard
    ("not_damaged_135", "not_damaged_153"),                      # steel bowl with water
    ("not_damaged_21", "not_damaged_40"),                        # pink tub
    ("not_damaged_117", "not_damaged_116"),                      # white egg tray
]
os.makedirs(a.out, exist_ok=True)

z = np.load(f"{a.task1}/dinov2_embeddings.npz")
ids, E = list(z["ids"]), z["emb"]
n_eggs = {r["id"]: r["n_eggs"] for r in json.load(open(f"{a.task1}/detections.json"))["images"]}
multi = np.array([n_eggs[i] > 1 for i in ids])

# 1-2. cluster
Z = linkage(E, method="average", metric="cosine")
group = fcluster(Z, t=1 - a.sim_thr, criterion="distance")
if not a.no_manual:
    for members in MANUAL_MERGES:
        gs = {group[ids.index(m)] for m in members}
        group[np.isin(group, list(gs))] = min(gs)
group = np.unique(group, return_inverse=True)[1] + 1   # relabel 1..G
G = group.max()

# 3. choose which groups go to eval
rng = np.random.default_rng(a.seed)
target = round(a.eval_frac * len(ids))
best, best_cost = None, 1e9
for _ in range(a.trials):
    ev = set()
    for g in rng.permutation(np.arange(1, G + 1)):
        if np.isin(group, list(ev | {g})).sum() <= target:
            ev.add(g)
    is_ev = np.isin(group, list(ev))
    # cost: hit the target size, keep multi-egg ratio equal, prefer more (smaller) groups in eval = more scene variety
    cost = abs(is_ev.sum() - target) * 10 + abs(multi[is_ev].mean() - multi[~is_ev].mean()) * 20 - len(ev) * 0.1
    if cost < best_cost:
        best, best_cost = is_ev, cost
split = np.where(best, "eval", "dev")

# 4. leakage report: for each eval image, similarity to its closest dev image
S = E @ E.T
def max_cross(mask_eval):
    return S[np.ix_(mask_eval, ~mask_eval)].max(1)
rand_cross = []
for _ in range(200):  # random image-level split for comparison
    m = np.zeros(len(ids), bool); m[rng.choice(len(ids), target, replace=False)] = True
    rand_cross.append(max_cross(m))
rand_cross, grp_cross = np.concatenate(rand_cross), max_cross(best)

df = pd.DataFrame(dict(id=ids, group=group, split=split, n_eggs=[n_eggs[i] for i in ids]))
df.to_csv(f"{a.out}/split.csv", index=False)
report = dict(sim_thr=a.sim_thr, n_groups=int(G), n_dev=int((split == "dev").sum()), n_eval=int((split == "eval").sum()),
              eval_groups=int(len(set(group[best]))),
              multi_egg_frac_dev=float(multi[~best].mean()), multi_egg_frac_eval=float(multi[best].mean()),
              eval_to_dev_maxsim_random_split=dict(mean=float(rand_cross.mean()), p90=float(np.percentile(rand_cross, 90)),
                                                   frac_above_0_9=float((rand_cross > 0.9).mean())),
              eval_to_dev_maxsim_group_split=dict(mean=float(grp_cross.mean()), p90=float(np.percentile(grp_cross, 90)),
                                                  frac_above_0_9=float((grp_cross > 0.9).mean())))
json.dump(report, open(f"{a.out}/split_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))

# visual check: one row per group, eval groups marked
paths = {}
for root, _, fs in os.walk(a.images):
    for f in fs:
        if f.endswith(".jpg"): paths[f.split("_jpg")[0]] = os.path.join(root, f)
t, maxc = 110, 16
rows = []
for g in sorted(set(group), key=lambda g: -(group == g).sum()):
    members = [i for i, gg in zip(ids, group) if gg == g][:maxc]
    ev = split[ids.index(members[0])] == "eval"
    row = np.full((t, (maxc + 1) * t, 3), 255, np.uint8)
    cv2.putText(row, f"G{g}", (5, 40), 0, 0.8, (0, 0, 0), 2)
    cv2.putText(row, "EVAL" if ev else "dev", (5, 80), 0, 0.7, (0, 0, 255) if ev else (0, 140, 0), 2)
    for k, m in enumerate(members):
        im = cv2.resize(cv2.imread(paths[m]), (t - 4, t - 4))
        cv2.putText(im, m.replace("not_damaged_", "nd"), (2, 14), 0, 0.45, (0, 0, 255), 1)
        row[2:t - 2, (k + 1) * t + 2:(k + 2) * t - 2] = im
    rows.append(row)
cv2.imwrite(f"{a.out}/split_groups.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
