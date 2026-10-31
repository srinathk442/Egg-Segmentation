"""Run one trial of the classical segmenter on a split, score it against SAM GT and log it.

usage: python scripts/run_experiment.py --name T01_otsu_gray --params '{"feature": "gray"}' --note "baseline"
Writes  experiments/runs/<name>/per_image.csv  (+ overlays with --vis)
Appends experiments/log.csv  (one row per trial: params + summary metrics + note)
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical, data, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--params", default="{}", help="JSON dict or path to a .json file overriding classical.DEFAULTS")
ap.add_argument("--split", default="dev")
ap.add_argument("--note", default="")
ap.add_argument("--vis", action="store_true")
ap.add_argument("--no_log", action="store_true")
a = ap.parse_args()

_p = json.load(open(a.params)) if a.params.endswith(".json") and os.path.exists(a.params) else json.loads(a.params)
params = {**classical.DEFAULTS, **_p}
out_dir = os.path.join(data.ROOT, "experiments", "runs" if a.split == "dev" else f"runs_{a.split}", a.name)
os.makedirs(out_dir, exist_ok=True)
paths = data.image_paths()

rows, t0 = [], time.time()
for sid in data.split_ids(a.split):
    bgr = cv2.imread(paths[sid])
    pred = classical.segment(bgr, params)
    gt = data.load_gt(sid, instance=True)
    rows.append(dict(id=sid, **metrics.all_metrics(pred, gt)))
    if a.vis:
        vis = bgr.copy()
        for m, c in [(gt > 0, (0, 255, 0)), (pred > 0, (0, 0, 255))]:   # green = GT, red = prediction
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, c, 2)
        cv2.putText(vis, f"IoU {rows[-1]['iou']:.2f}", (8, 28), 0, 0.9, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{sid}.jpg"), vis)
ms = (time.time() - t0) / len(rows) * 1000

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
summ = metrics.summarize(df)
summ["ms_per_img"] = round(ms, 1)
json.dump(dict(name=a.name, split=a.split, params=params, summary=summ, note=a.note),
          open(os.path.join(out_dir, "summary.json"), "w"), indent=1)

print(f"[{a.name}] {a.split} n={len(df)}  " + "  ".join(f"{k}={v}" for k, v in summ.items()))
worst = df.nsmallest(6, "iou")
print("worst:", ", ".join(f"{r.id.replace('not_damaged_', 'nd')}={r.iou:.2f}" for r in worst.itertuples()))

if not a.no_log and a.split == "dev":
    # rebuild the log from every run's summary.json (safe when several trials run in parallel)
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(data.ROOT, "experiments", "runs", "*", "summary.json"))):
        s = json.load(open(f))
        if s["name"].startswith("_"):
            continue
        changed = {k: v for k, v in s["params"].items() if classical.DEFAULTS.get(k) != v}
        rows.append(dict(trial=s["name"], **s["summary"], params_changed=json.dumps(changed), note=s["note"]))
    pd.DataFrame(rows).to_csv(os.path.join(data.ROOT, "experiments", "log.csv"), index=False)
