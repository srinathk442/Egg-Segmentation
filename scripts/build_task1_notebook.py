"""Generates notebooks/01_task1_sam_groundtruth.ipynb (cells defined below)."""
import json, sys

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Task 1 — Egg segmentation with a pretrained model (Grounded-SAM)

**Pipeline:** `text "egg." → Grounding DINO (boxes) → box filtering → SAM ViT-H (one mask per box) → mask clean-up`

* **Grounding DINO** is an *open-vocabulary detector*: it matches image regions to words in a text prompt, so it can find "egg" without being trained on this dataset. It answers *where* the eggs are.
* **SAM (Segment Anything)** is a *promptable segmenter*: given a box/point prompt it returns a precise mask. It answers *which pixels* belong to the object, but it has no notion of the class "egg" — hence the detector in front of it.
* We also compute **DINOv2 embeddings** (one vector per image) used later for a leakage-aware dev/eval split.

Runtime → Change runtime type → **T4 GPU**. Then *Run all*. At the end a `task1_outputs.zip` downloads.
""")

code(r"""
!nvidia-smi -L
!pip -q install -U transformers accelerate gdown
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
""")

md("## 1. Get the dataset (same zip as the task's Google Drive link)")
code(r"""
import os, glob
if not os.path.exists("/content/egg"):
    !gdown 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv -O /content/egg.zip
    !unzip -q -o /content/egg.zip -d /content/egg
paths = sorted(glob.glob("/content/egg/**/*.jpg", recursive=True))
print(len(paths), "images")

def short_id(p):  # 'not_damaged_101_jpg.rf.<hash>.jpg' -> 'not_damaged_101' (unique in this dataset)
    return os.path.basename(p).split("_jpg")[0]
ids = [short_id(p) for p in paths]
assert len(set(ids)) == len(ids), "short ids are not unique"
""")

md("## 2. Load models")
code(r"""
import numpy as np, cv2, json, time
from PIL import Image
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor, AutoImageProcessor, AutoModel)
device = "cuda" if torch.cuda.is_available() else "cpu"

GD_ID, SAM_ID, DINO_ID = "IDEA-Research/grounding-dino-base", "facebook/sam-vit-huge", "facebook/dinov2-base"
gd_proc = AutoProcessor.from_pretrained(GD_ID)
gd = AutoModelForZeroShotObjectDetection.from_pretrained(GD_ID).to(device).eval()
sam_proc = SamProcessor.from_pretrained(SAM_ID)
sam = SamModel.from_pretrained(SAM_ID).to(device).eval()
print("models loaded")
""")

md(r"""
## 3. Detection: text prompt → boxes

Grounding DINO scores every candidate box against each text token. Two thresholds:
* `BOX_THR` – minimum box confidence;
* `TEXT_THR` – minimum similarity between the box and the word "egg".

Then we clean the boxes:
1. drop boxes covering almost the whole image (usually the bowl / table),
2. **NMS** (non-max suppression): if two boxes overlap with IoU > 0.5 keep only the higher-scoring one,
3. drop **group boxes**: a box that contains ≥2 other egg boxes is "the eggs" as a group, not one egg.
""")
code(r"""
PROMPT, BOX_THR, TEXT_THR = "egg.", 0.30, 0.25
MAX_AREA_FRAC, NMS_IOU, CONTAIN_FRAC = 0.90, 0.50, 0.85

@torch.no_grad()
def detect(img, prompt=PROMPT, box_thr=BOX_THR, text_thr=TEXT_THR):
    inputs = gd_proc(images=img, text=prompt, return_tensors="pt").to(device)
    out = gd(**inputs)
    kw = dict(input_ids=inputs.input_ids, text_threshold=text_thr, target_sizes=[img.size[::-1]])
    try:   # argument name changed across transformers versions
        r = gd_proc.post_process_grounded_object_detection(out, threshold=box_thr, **kw)[0]
    except TypeError:
        r = gd_proc.post_process_grounded_object_detection(out, box_threshold=box_thr, **kw)[0]
    return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()

def box_area(b): return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
def inter(a, b): return box_area([max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])])
def iou(a, b):
    i = inter(a, b); return i / (box_area(a) + box_area(b) - i + 1e-9)

def filter_boxes(boxes, scores, W, H):
    idx = [i for i in range(len(boxes)) if box_area(boxes[i]) < MAX_AREA_FRAC * W * H]
    idx.sort(key=lambda i: -scores[i])
    keep = []
    for i in idx:                                   # greedy NMS
        if all(iou(boxes[i], boxes[j]) <= NMS_IOU for j in keep):
            keep.append(i)
    final = []
    for i in keep:                                  # remove group boxes
        inside = sum(inter(boxes[i], boxes[j]) / (box_area(boxes[j]) + 1e-9) > CONTAIN_FRAC for j in keep if j != i)
        if inside < 2:
            final.append(i)
    return boxes[final], scores[final]
""")

md(r"""
## 4. Segmentation: boxes → SAM masks

SAM's heavy image encoder runs **once** per image; each box is encoded as a prompt and the light mask decoder produces one mask per box (`multimask_output=False` – a box is an unambiguous prompt).

Clean-up per mask: keep the largest connected component and fill holes (specular highlights sometimes punch holes). Overlaps between eggs are resolved by painting larger masks first so a smaller, partly-occluded egg keeps its pixels.

Saved per image:
* `masks_instance/<id>.png` – 0 = background, 1..N = egg instance id
* `masks_binary/<id>.png` – 0/255 union of all eggs (used as ground truth in Tasks 2–3)
* `overlays/<id>.jpg` – visual check
""")
code(r"""
from scipy import ndimage as ndi

def clean_mask(m, multi_cc=False):
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        # multi_cc: an egg split by an occluder (e.g. a rod) is several pieces -> keep all big pieces
        keep = np.where(areas >= 0.2 * areas.max())[0] + 1 if multi_cc else [1 + np.argmax(areas)]
        m = np.isin(lab, keep).astype(np.uint8)
    return ndi.binary_fill_holes(m).astype(np.uint8)

@torch.no_grad()
def segment(img, boxes, points=None, multi_cc=False):
    # points (optional): one list per box of [x, y, label]; label 1 = "this is egg", 0 = "this is NOT egg"
    if len(boxes) == 0:
        return np.zeros((0, img.size[1], img.size[0]), np.uint8), np.zeros(0)
    kw = dict(input_boxes=[boxes.tolist()])
    if points:
        kw["input_points"] = [[[p[:2] for p in pts] for pts in points]]
        kw["input_labels"] = [[[p[2] for p in pts] for pts in points]]
    inputs = sam_proc(img, return_tensors="pt", **kw).to(device)
    out = sam(**inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
    masks = masks[:, 0].numpy().astype(np.uint8)          # (n_boxes, H, W)
    return np.stack([clean_mask(m, multi_cc) for m in masks]), out.iou_scores.cpu().numpy()[0, :, 0]

def to_instance_map(masks, min_area):
    inst = np.zeros(masks.shape[1:], np.uint8)
    order = np.argsort([-m.sum() for m in masks])         # large first, small painted on top
    k = 0
    for i in order:
        if masks[i].sum() < min_area: continue
        k += 1; inst[masks[i] > 0] = k
    return inst

PALETTE = [(0,255,0),(255,0,255),(0,200,255),(255,128,0),(255,0,0),(0,0,255)]
def overlay(img_bgr, inst, boxes=None, scores=None):
    vis = img_bgr.copy()
    for k in range(1, inst.max() + 1):
        m = inst == k; c = PALETTE[(k-1) % len(PALETTE)]
        vis[m] = (0.55 * vis[m] + 0.45 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, c, 2)
    if boxes is not None:
        for b, s in zip(boxes, scores):
            x0, y0, x1, y1 = map(int, b)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.putText(vis, f"{s:.2f}", (x0, max(12, y0-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    return vis
""")

md(r"""
## 5. Prompt corrections found during visual QC (first run had 3/110 bad images)

Instead of hand-painting masks, failures are fixed by giving SAM **better prompts**, so the ground truth stays model-generated and reproducible. Every override is logged in `detections.json`.

| image | problem in first run | fix |
|---|---|---|
| nd22, nd39 | 2nd box (score 0.36 / 0.49) is the egg's shadow on the printed lid → false positive | keep only the top-scoring box |
| nd117 | (a) lower box (0.38) is an empty egg-shaped tray cup → false positive; (b) a rod crosses the real egg, SAM masked only the left half and largest-component clean-up dropped the rest | keep top box, add positive points on both halves + a negative point on the rod, keep all large components |

Keys: `keep` = indices (by score) of detector boxes to keep · `boxes` = manual boxes · `points` = per box `[x, y, label]` · `multi_cc` = keep several components.
""")
code(r"""
OVERRIDES = {
    "not_damaged_22":  {"keep": [0]},
    "not_damaged_39":  {"keep": [0]},
    "not_damaged_117": {"keep": [0], "points": [[[310, 190, 1], [485, 185, 1], [425, 190, 0]]], "multi_cc": True},
}
""")

md("## 6. Run on all images")
code(r"""
OUT = "/content/task1_outputs"
for d in ["masks_instance", "masks_binary", "overlays"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

records, t0 = [], time.time()
for p, sid in zip(paths, ids):
    img = Image.open(p).convert("RGB"); W, H = img.size
    ov = OVERRIDES.get(sid, {})
    if "boxes" in ov:
        boxes, scores = np.array(ov["boxes"], float), np.ones(len(ov["boxes"]))
        raw_n = len(boxes)
    else:
        raw_b, raw_s = detect(img, ov.get("prompt", PROMPT), ov.get("box_thr", BOX_THR), ov.get("text_thr", TEXT_THR))
        raw_n = len(raw_b)
        boxes, scores = filter_boxes(raw_b, raw_s, W, H)
        if "keep" in ov:
            boxes, scores = boxes[ov["keep"]], scores[ov["keep"]]
    masks, sam_iou = segment(img, boxes, ov.get("points"), ov.get("multi_cc", False))
    inst = to_instance_map(masks, min_area=0.001 * W * H)
    cv2.imwrite(f"{OUT}/masks_instance/{sid}.png", inst)
    cv2.imwrite(f"{OUT}/masks_binary/{sid}.png", (inst > 0).astype(np.uint8) * 255)
    cv2.imwrite(f"{OUT}/overlays/{sid}.jpg", overlay(cv2.imread(p), inst, boxes, scores))
    records.append(dict(id=sid, file=os.path.relpath(p, "/content/egg"), n_raw_boxes=int(raw_n),
                        n_eggs=int(inst.max()), boxes=np.round(boxes, 1).tolist(),
                        det_scores=np.round(scores, 3).tolist(), sam_pred_iou=np.round(sam_iou, 3).tolist(),
                        egg_area_frac=float((inst > 0).mean()), override=ov or None))
dt = time.time() - t0
print(f"{len(records)} images in {dt:.1f}s ({dt/len(records):.2f}s/img)")
json.dump(dict(prompt=PROMPT, box_thr=BOX_THR, text_thr=TEXT_THR, max_area_frac=MAX_AREA_FRAC,
               nms_iou=NMS_IOU, contain_frac=CONTAIN_FRAC, models=[GD_ID, SAM_ID],
               sec_per_img=dt/len(records), images=records), open(f"{OUT}/detections.json", "w"), indent=1)

import collections
print("eggs per image:", sorted(collections.Counter(r["n_eggs"] for r in records).items()))
print("images with 0 eggs:", [r["id"] for r in records if r["n_eggs"] == 0])
""")

md("## 7. Visual QC — contact sheet of all overlays")
code(r"""
import matplotlib.pyplot as plt
def contact_sheet(ids_, folder, cols=10, t=200):
    rows = (len(ids_) + cols - 1) // cols
    sheet = np.full((rows*t, cols*t, 3), 255, np.uint8)
    for i, sid in enumerate(ids_):
        im = cv2.resize(cv2.imread(f"{folder}/{sid}.jpg"), (t, t))
        cv2.putText(im, sid.replace("not_damaged_", "nd"), (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        r, c = divmod(i, cols); sheet[r*t:(r+1)*t, c*t:(c+1)*t] = im
    return sheet
sheet = contact_sheet(ids, f"{OUT}/overlays")
cv2.imwrite(f"{OUT}/qc_contact_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
plt.figure(figsize=(20, 22)); plt.imshow(sheet[..., ::-1]); plt.axis("off"); plt.show()
""")

md(r"""
## 8. DINOv2 embeddings for the leakage-aware split (Task 2)

DINOv2 is a self-supervised ViT; its CLS-token vector summarises the whole image (background, lighting, objects). Photos of the same scene end up close together (high cosine similarity), so we can cluster them and keep each cluster entirely in dev **or** eval.
""")
code(r"""
dproc = AutoImageProcessor.from_pretrained(DINO_ID)
dmodel = AutoModel.from_pretrained(DINO_ID).to(device).eval()
embs = []
with torch.no_grad():
    for p in paths:
        x = dproc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(device)
        e = dmodel(**x).pooler_output[0]
        embs.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
np.savez(f"{OUT}/dinov2_embeddings.npz", ids=np.array(ids), emb=np.stack(embs))
print("embeddings", np.stack(embs).shape)
""")

md("## 9. Download results")
code(r"""
!cd /content && zip -q -r task1_outputs.zip task1_outputs && ls -lh task1_outputs.zip
from google.colab import files
files.download("/content/task1_outputs.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
# notebook JSON stores source as list of lines
for c in cells:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
