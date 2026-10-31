# Egg Segmentation and Tracking: SAM vs. Classical Image Processing

**Video tracking output (Task 4):** [Google Drive – tracked.mp4](https://drive.google.com/file/d/1Hmf4o2abnqC_pELmrqlz3xM_3BRuzZE-/view?usp=sharing)

## Summary

| task | what was done | headline result |
|---|---|---|
| 1. Pretrained segmentation | Grounding DINO ("egg.") → boxes → SAM ViT-H → masks for all 110 images; visual QC, 3 images fixed with better prompts | 138 eggs in 110 images (1–3 per image) |
| 2. Classical segmentation | Leakage-aware 4:1 split by scene groups (DINOv2 embeddings + clustering); 27 logged trials leading to a threshold-sweep + watershed + egg-likeness-score algorithm | dev mIoU 0.704 (from 0.341 baseline) |
| 3. Evaluation | IoU, Dice, precision/recall, boundary F1, instance F1, count accuracy, bootstrap CIs; per-scene, difficulty and failure-mode analysis | eval mIoU 0.539 [0.36, 0.70]; IoU 0.91 when the egg is found; failures driven by low egg-background contrast |
| 4. Video tracking | Adapted segmenter + Kalman/Hungarian tracker with re-identification by elimination | 2 IDs for 2 eggs, 0 identity errors in 12 checked key frames, 0 false positives on title cards |

**Repository layout:** `notebooks/` (Task 1, Colab GPU), `src/eggseg/` (classical segmenter, metrics, tracker),
`scripts/` (split, experiments, analysis, video), `configs/` (frozen parameters), `experiments/log.csv` (all trials),
`outputs/` (masks, figures, video), `reports/` (this report). Setup and run instructions: `README.md`.
