"""Task 4 step 1 - per-frame egg segmentation of the video with the classical algorithm.

Each worker decodes its own contiguous block of frames (so frames are never all in memory), segments them at
640x360 and saves instance masks to <out>/f_<frame>.png (0 = background, 1..N = eggs of that frame).

usage: python scripts/video_detect.py --video data/video/egg_video.mp4 --config configs/video.json
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eggseg import classical

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/video/egg_video.mp4")
ap.add_argument("--config", default="configs/video.json")
ap.add_argument("--out", default="outputs/task4/det")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
a = ap.parse_args()
P = json.load(open(a.config))


def run(block):
    start, end = block
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        h = round(frame.shape[0] * a.width / frame.shape[1])
        small = cv2.resize(frame, (a.width, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(a.out, f"f_{f:05d}.png"), classical.segment(small, P))
    return end - start


if __name__ == "__main__":
    os.makedirs(a.out, exist_ok=True)
    n = int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_COUNT))
    blocks = [(s, min(s + a.chunk, n)) for s in range(0, n, a.chunk)]
    t, done = time.time(), 0
    with Pool(a.workers) as pool:
        for k in pool.imap_unordered(run, blocks):
            done += k
            print(f"{done}/{n} frames  {time.time() - t:.0f}s", flush=True)
    json.dump(dict(config=a.config, width=a.width, frames=n, seconds=time.time() - t),
              open(os.path.join(a.out, "_meta.json"), "w"))
