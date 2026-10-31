"""Task 2 - data analysis before designing the algorithm (dev split only).

Question: which colour channel separates egg pixels from background pixels best?
For each channel we compute ROC-AUC = P(random egg pixel scores higher than random background pixel).
AUC 1.0 = perfect, 0.5 = useless, <0.5 = separable but inverted (egg is *lower*).
 * global AUC   : one threshold for all images
 * per-image AUC: median over images -> what an adaptive (per-image) threshold could reach
"""
import glob, os, json
import numpy as np, pandas as pd, cv2
from sklearn.metrics import roc_auc_score

IMG, GT = "data/raw/egg", "outputs/task1_outputs/masks_binary"
split = pd.read_csv("splits/split.csv")
dev = split[split.split == "dev"].id.tolist()
paths = {os.path.basename(p).split("_jpg")[0]: p for p in glob.glob(f"{IMG}/**/*.jpg", recursive=True)}

def channels(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {"gray": gray, "R": bgr[..., 2], "G": bgr[..., 1], "B": bgr[..., 0],
            "H": hsv[..., 0], "S": hsv[..., 1], "V": hsv[..., 2],
            "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2],
            # chroma = distance from neutral grey in Lab; white/grey things have ~0
            "chroma": np.hypot(lab[..., 1].astype(float) - 128, lab[..., 2].astype(float) - 128)}

rng = np.random.default_rng(0)
pool = {k: ([], []) for k in channels(np.zeros((2, 2, 3), np.uint8))}
per_img = {k: [] for k in pool}
for i in dev:
    bgr = cv2.imread(paths[i]); gt = cv2.imread(f"{GT}/{i}.png", 0) > 0
    ch = channels(bgr)
    idx = rng.choice(gt.size, 4000, replace=False)          # subsample pixels for speed
    y = gt.ravel()[idx]
    for k, v in ch.items():
        x = v.ravel()[idx].astype(float)
        pool[k][0].append(x); pool[k][1].append(y)
        per_img[k].append(roc_auc_score(y, x))

rows = []
for k in pool:
    x, y = np.concatenate(pool[k][0]), np.concatenate(pool[k][1])
    g = roc_auc_score(y, x); pi = np.array(per_img[k])
    # orientation-free: max(auc, 1-auc) - a channel where egg is darker is equally useful
    rows.append(dict(channel=k, global_auc=round(max(g, 1 - g), 3), egg_is=("higher" if g >= .5 else "lower"),
                     per_image_auc_median=round(np.median(np.maximum(pi, 1 - pi)), 3),
                     per_image_auc_p10=round(np.percentile(np.maximum(pi, 1 - pi), 10), 3)))
df = pd.DataFrame(rows).sort_values("per_image_auc_median", ascending=False)
os.makedirs("outputs/task2", exist_ok=True)
df.to_csv("outputs/task2/channel_auc_dev.csv", index=False)
print(df.to_string(index=False))

# worst images for the best channels -> where a simple threshold will fail
best = df.channel.iloc[:3].tolist()
for k in best:
    pi = np.maximum(np.array(per_img[k]), 1 - np.array(per_img[k]))
    worst = np.argsort(pi)[:6]
    print(f"worst for {k}:", [(dev[j].replace('not_damaged_', 'nd'), round(pi[j], 2)) for j in worst])
