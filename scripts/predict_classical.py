"""Run the frozen classical algorithm on every image and save instance masks (0 = bg, 1..N = eggs).

usage: python scripts/predict_classical.py [--config configs/final_classical.json] [--out outputs/task2/pred_instance]
"""
import argparse, json, os, sys
from multiprocessing import Pool
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical, data

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/final_classical.json")
ap.add_argument("--out", default="outputs/task2/pred_instance")
ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
a = ap.parse_args()
P = json.load(open(a.config))
paths = data.image_paths()


def run(sid):
    cv2.setNumThreads(1)
    inst = classical.segment(cv2.imread(paths[sid]), P)
    cv2.imwrite(os.path.join(a.out, f"{sid}.png"), inst)
    return sid


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    ids = data.split_ids("all")
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(run, ids), 1):
            if i % 20 == 0: print(f"{i}/{len(ids)}", flush=True)
    print("saved", len(ids), "masks to", a.out)
