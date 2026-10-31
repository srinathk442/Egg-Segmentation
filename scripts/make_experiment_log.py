"""Builds reports/task2_experiment_log.md: one row per trial with what changed, why, the dev metrics
(read from experiments/log.csv, so numbers always match the runs) and why it was kept or discarded."""
import pandas as pd

log = pd.read_csv("experiments/log.csv").set_index("trial")

# trial -> (what changed + parameters, why it was tried, decision + reason)
T = {
 "T01_otsu_gray": ("Baseline: grayscale, per-image Otsu threshold, connected components",
                   "Simplest classical method; eggs are mostly the bright object",
                   "Baseline. Precision 0.37: hands, carpets, metal are bright too"),
 "T02_cleanup": ("+ Gaussian blur 5, opening 5, closing 9, fill holes, drop blobs < 0.2% of image",
                 "Remove speckles, fill glare holes", "**Kept** (+0.02)"),
 "T03_largest": ("+ keep only the largest blob", "Most images contain one egg",
                 "Discarded: drops real eggs in multi-egg images (instance recall 0.37 to 0.21)"),
 "T04_white_w1": ("Feature 'whiteness' = Lab L - 1 x chroma", "Channel analysis: eggs are bright AND colourless",
                  "Discarded: w=4 better"),
 "T04_white_w2": ("Whiteness with chroma weight 2", "Weight sweep", "Discarded: w=4 better"),
 "T04_white_w4": ("Whiteness with chroma weight 4", "Weight sweep",
                  "**Kept**, but one global threshold still merges egg and background"),
 "T05_shape_filter": ("+ drop blobs with solidity < 0.85 or ellipse-fit IoU < 0.75", "Eggs are convex ellipses",
                      "Discarded: egg is merged with background into one non-elliptic blob and removed with it "
                      "(recall 0.92 to 0.54). Insight: the problem is candidate generation"),
 "T06a_hull": ("+ replace blob by convex hull", "Recover the shadowed side of the egg", "Discarded: same merged-blob problem"),
 "T06b_ellipse": ("+ replace blob by fitted ellipse", "Same", "Discarded: same merged-blob problem"),
 "T07_sweep": ("Threshold sweep: 24 levels; every blob scored by ellipse IoU x solidity x ring contrast; "
               "mask NMS; extra eggs if score >= 0.45", "At some threshold each egg separates as a clean ellipse",
               "Idea kept; too many extra eggs accepted (instance precision 0.13)"),
 "T07b_accept0.7": ("Sweep, accept extras >= 0.7", "Fewer false extras", "Discarded: 0.85 better"),
 "T07b_accept0.85": ("Sweep, accept extras >= 0.85", "Fewer false extras", "**Kept**: count accuracy 0.13 (T07) to 0.88"),
 "T08a_watershed": ("+ grow each core by marker-controlled watershed on the Lab gradient (search = 1.6 x core ellipse)",
                    "Recall 0.56: shadowed half of the egg missing; egg interior is smooth so flooding stops at its outline",
                    "**Kept** (+0.07, boundary F1 0.40 to 0.51)"),
 "T08b_grabcut": ("Instead: GrabCut per egg (GMM colour models + graph cut), same init", "Alternative refinement",
                  "Discarded: 0.54 < watershed and 1.4x slower"),
 "T09a_sizeprior": ("+ log-normal egg-size prior on the candidate score (fit on dev eggs)",
                    "Tiny elliptical blobs (earphone bud) beat the egg",
                    "Discarded: also penalises partial egg cores, which are smaller than a full egg"),
 "T09b_refine_then_score": ("Grow top-8 cores first, then score grown regions (ellipse x solidity x edge x size)",
                            "Score complete regions instead of partial cores",
                            "Discarded (0.36). Diagnostic: edge term saturated (egg 10.6, other 4.2, both capped at 3)"),
 "T10a_calibrated_score": ("Refine-then-score with calibrated score: ellipse^4 x solidity x edge/10 x size x border "
                           "penalty; extras >= 0.3", "Formula chosen offline on a table of 638 refined regions",
                           "**Kept** (0.59, count accuracy 0.92)"),
 "T10b_multiview": ("+ candidates from 3 views: whiteness, darkness (inverted), smoothness (low local std); top-5 each",
                    "Oracle analysis: in 19 of 32 hard images no good core existed (white paper, ash)",
                    "**Kept** (0.62, median 0.83)"),
 "T11_score_v2": ("Score v2: edge = boundary gradient / outer-ring gradient, + sigmoid brightness contrast; extras >= 0.25",
                  "Region table: brightness contrast is the best cue (AUC 0.92); image-median edge fails on textured bg",
                  "**Kept** (+0.065)"),
 "T12a_grow1.3": ("Watershed search region 1.6 to 1.3 x core ellipse", "Large search region lets growth leak",
                  "**Kept** (0.703)"),
 "T12b_grow2.0": ("Search region 2.0x", "Check other direction", "Discarded: leaks (0.635)"),
 "T12c_grow1.15": ("Search region 1.15x", "Check smaller", "Discarded: cuts eggs (0.683)"),
 "T12d_grow1.3_erode9": ("grow 1.3 + core erosion 9 px", "Pull back slightly leaking cores",
                         "Discarded: no change, simpler version kept"),
 "T12e_grow1.3_topk8": ("grow 1.3 + top-8 candidates per view", "Larger candidate pool",
                        "Discarded: no gain, 60% slower"),
 "T13a_hull": ("+ final mask = convex hull", "Smooth jagged watershed outline",
               "Discarded: over-fills eggs partly hidden by fingers/rods"),
 "T13b_ellipse": ("+ final mask = fitted ellipse", "Same", "Discarded: 0.688, boundary F1 drops"),
 "T13c_smooth11": ("+ final opening + closing (11 px) per egg", "Smooth jagged watershed outline",
                   "**Kept = FINAL** (boundary F1 0.584 to 0.590)"),
}

missing = set(log.index) - set(T)
assert not missing, f"no description for {missing}"
lines = ["| trial | what changed (parameters) | why | dev mIoU | Dice | inst. F1 | decision |",
         "|---|---|---|---|---|---|---|"]
for name in log.index:
    c, w, d = T[name]
    r = log.loc[name]
    lines.append(f"| {name.replace('_', ' ')} | {c} | {w} | {r.iou:.3f} | {r.dice:.3f} | {r.inst_f1:.3f} | {d} |")
open("reports/task2_experiment_log.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(len(log), "trials written")
