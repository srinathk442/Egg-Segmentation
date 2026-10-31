# Egg segmentation & tracking: SAM vs. classical image processing

CV intern task. Full write-up with all figures: **[report.pdf](report.pdf)**.
Video tracking output: **[tracked.mp4 on Google Drive](https://drive.google.com/file/d/1Hmf4o2abnqC_pELmrqlz3xM_3BRuzZE-/view?usp=sharing)**.

| task | approach | result |
|---|---|---|
| 1. Pretrained model | Grounding DINO (`"egg."`) → boxes → SAM ViT-H; visual QC of all 110 images, 3 fixed with better prompts | 138 egg masks, used as ground truth |
| 2. Classical method | leakage-aware 4:1 split by scene groups (DINOv2 + clustering); threshold sweep over 3 feature views → marker-controlled watershed → egg-likeness score; 27 logged trials | dev mIoU **0.704** (baseline 0.341) |
| 3. Evaluation | IoU, Dice, precision/recall, boundary F1, instance F1, count accuracy, bootstrap CI; per-scene / difficulty / failure-mode analysis | eval mIoU **0.539** [0.36, 0.70]; IoU 0.91 when the egg is found |
| 4. Video tracking | adapted segmenter + Kalman filter + Hungarian matching + re-identification by elimination | 2 IDs for 2 eggs, 0 ID errors in checked key frames |

## Repository layout

```
notebooks/01_task1_sam_groundtruth.ipynb   Task 1 (Google Colab, GPU): SAM masks + DINOv2 embeddings
src/eggseg/
  classical.py     classical segmenter (all trials = parameter settings of this one pipeline)
  metrics.py       IoU, Dice, precision/recall, boundary F1, instance matching
  tracking.py      Kalman + Hungarian egg tracker
  data.py          paths, split, ground-truth loading
configs/
  final_classical.json   frozen Task 2 algorithm (trial T13c)
  video.json             Task 4 settings (adds size prior / rejection constraints for the video)
scripts/           every step below is one script (see "Run")
splits/            split.csv (dev/eval + scene group per image), split_report.json, split_groups.jpg
experiments/       log.csv (all trials) + runs/<trial>/{per_image.csv, summary.json}
outputs/           task1_outputs (SAM masks), task2 (predictions), task3 (metrics, figures), task4 (tracks, figures)
reports/           markdown sources of report.pdf
```

## Setup

Python 3.12 (tested on Windows 11, CPU only for Tasks 2–4).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Data** (not stored in the repo):

```bash
# dataset (Roboflow egg dataset, zip from the task's Google Drive link) -> data/raw/egg/
gdown 1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv -O data/raw/egg.zip
python -c "import zipfile; zipfile.ZipFile('data/raw/egg.zip').extractall('data/raw/egg')"

# video (1080p, video only) -> data/video/egg_video.mp4
yt-dlp -f "bv*[ext=mp4][vcodec^=avc1]" -o data/video/egg_video.mp4 "https://www.youtube.com/watch?v=Y4JyV94y2_A"
```

(yt-dlp needs a JavaScript runtime for YouTube; `deno` is installed by `requirements.txt` and must be on `PATH`,
e.g. the venv's `Scripts`/`bin` folder.)

## Run

All commands from the repository root. Outputs of every step are already committed, so any step can be run on its own.

**Task 1 – SAM ground truth (Colab GPU).** Open `notebooks/01_task1_sam_groundtruth.ipynb` in Google Colab,
select a T4 GPU, *Run all* (≈ 6 min). It downloads the dataset itself and produces `task1_outputs.zip`;
extract it to `outputs/task1_outputs/`.

**Task 2 – split and classical segmentation**

```bash
python scripts/make_split.py                 # leakage-aware 4:1 split -> splits/
python scripts/channel_analysis.py           # colour-channel AUC analysis on dev
# one trial = one parameter setting, scored on dev and appended to experiments/log.csv, e.g. the final one:
python scripts/run_experiment.py --name T13c_smooth11 --params configs/final_classical.json --vis
python scripts/make_experiment_log.py        # experiment-log table for the report
```

Diagnostics used during development: `scripts/diag_candidates.py`, `scripts/diag_refined.py`,
`scripts/dump_regions.py`, `scripts/diag_failures.py <trial>`.

**Task 3 – evaluation**

```bash
python scripts/predict_classical.py          # frozen algorithm on all 110 images -> outputs/task2/pred_instance
python scripts/run_experiment.py --name FINAL --split eval --params configs/final_classical.json
python scripts/task3_analysis.py             # metrics, plots, qualitative grids -> outputs/task3/
```

**Task 4 – video tracking**

```bash
python scripts/video_detect.py               # per-frame segmentation, parallel (~7.5 min on 12 cores)
python scripts/video_track.py                # tracking + rendering -> outputs/task4/tracked.mp4
python scripts/task4_figures.py
```

**Report**

```bash
python scripts/fig_report_extras.py
python scripts/build_report.py               # reports/*.md -> report.pdf
```


## Use of AI tools

Claude Code (Anthropic's AI coding assistant) was used as a pair programmer for writing and debugging code, running
experiments and drafting the report; details in the last section of `report.pdf`.
