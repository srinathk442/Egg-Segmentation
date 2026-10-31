"""Dataset access: image paths, split membership and SAM ground-truth masks."""
import glob, os
import cv2, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "egg")
GT_DIR = os.path.join(ROOT, "outputs", "task1_outputs")
SPLIT_CSV = os.path.join(ROOT, "splits", "split.csv")


def image_paths():
    """short id ('not_damaged_101') -> jpg path."""
    return {os.path.basename(p).split("_jpg")[0]: p
            for p in glob.glob(os.path.join(IMG_DIR, "**", "*.jpg"), recursive=True)}


def split_ids(split):
    df = pd.read_csv(SPLIT_CSV)
    return df[df.split == split].id.tolist() if split != "all" else df.id.tolist()


def load_gt(sid, instance=False):
    sub = "masks_instance" if instance else "masks_binary"
    m = cv2.imread(os.path.join(GT_DIR, sub, f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
    return m if instance else (m > 0)
