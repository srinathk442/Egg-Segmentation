| trial | what changed (parameters) | why | dev mIoU | Dice | inst. F1 | decision |
|---|---|---|---|---|---|---|
| T01 otsu gray | Baseline: grayscale, per-image Otsu threshold, connected components | Simplest classical method; eggs are mostly the bright object | 0.341 | 0.455 | 0.007 | Baseline. Precision 0.37: hands, carpets, metal are bright too |
| T02 cleanup | + Gaussian blur 5, opening 5, closing 9, fill holes, drop blobs < 0.2% of image | Remove speckles, fill glare holes | 0.361 | 0.478 | 0.203 | **Kept** (+0.02) |
| T03 largest | + keep only the largest blob | Most images contain one egg | 0.359 | 0.461 | 0.231 | Discarded: drops real eggs in multi-egg images (instance recall 0.37 to 0.21) |
| T04 white w1 | Feature 'whiteness' = Lab L - 1 x chroma | Channel analysis: eggs are bright AND colourless | 0.355 | 0.472 | 0.191 | Discarded: w=4 better |
| T04 white w2 | Whiteness with chroma weight 2 | Weight sweep | 0.370 | 0.484 | 0.206 | Discarded: w=4 better |
| T04 white w4 | Whiteness with chroma weight 4 | Weight sweep | 0.384 | 0.496 | 0.205 | **Kept**, but one global threshold still merges egg and background |
| T05 shape filter | + drop blobs with solidity < 0.85 or ellipse-fit IoU < 0.75 | Eggs are convex ellipses | 0.301 | 0.360 | 0.352 | Discarded: egg is merged with background into one non-elliptic blob and removed with it (recall 0.92 to 0.54). Insight: the problem is candidate generation |
| T06a hull | + replace blob by convex hull | Recover the shadowed side of the egg | 0.293 | 0.354 | 0.343 | Discarded: same merged-blob problem |
| T06b ellipse | + replace blob by fitted ellipse | Same | 0.290 | 0.353 | 0.343 | Discarded: same merged-blob problem |
| T07 sweep | Threshold sweep: 24 levels; every blob scored by ellipse IoU x solidity x ring contrast; mask NMS; extra eggs if score >= 0.45 | At some threshold each egg separates as a clean ellipse | 0.369 | 0.491 | 0.219 | Idea kept; too many extra eggs accepted (instance precision 0.13) |
| T07b accept0.7 | Sweep, accept extras >= 0.7 | Fewer false extras | 0.436 | 0.541 | 0.379 | Discarded: 0.85 better |
| T07b accept0.85 | Sweep, accept extras >= 0.85 | Fewer false extras | 0.501 | 0.581 | 0.588 | **Kept**: count accuracy 0.13 (T07) to 0.88 |
| T08a watershed | + grow each core by marker-controlled watershed on the Lab gradient (search = 1.6 x core ellipse) | Recall 0.56: shadowed half of the egg missing; egg interior is smooth so flooding stops at its outline | 0.572 | 0.630 | 0.658 | **Kept** (+0.07, boundary F1 0.40 to 0.51) |
| T08b grabcut | Instead: GrabCut per egg (GMM colour models + graph cut), same init | Alternative refinement | 0.539 | 0.607 | 0.614 | Discarded: 0.54 < watershed and 1.4x slower |
| T09a sizeprior | + log-normal egg-size prior on the candidate score (fit on dev eggs) | Tiny elliptical blobs (earphone bud) beat the egg | 0.502 | 0.576 | 0.545 | Discarded: also penalises partial egg cores, which are smaller than a full egg |
| T09b refine then score | Grow top-8 cores first, then score grown regions (ellipse x solidity x edge x size) | Score complete regions instead of partial cores | 0.363 | 0.470 | 0.426 | Discarded (0.36). Diagnostic: edge term saturated (egg 10.6, other 4.2, both capped at 3) |
| T10a calibrated score | Refine-then-score with calibrated score: ellipse^4 x solidity x edge/10 x size x border penalty; extras >= 0.3 | Formula chosen offline on a table of 638 refined regions | 0.589 | 0.645 | 0.682 | **Kept** (0.59, count accuracy 0.92) |
| T10b multiview | + candidates from 3 views: whiteness, darkness (inverted), smoothness (low local std); top-5 each | Oracle analysis: in 19 of 32 hard images no good core existed (white paper, ash) | 0.617 | 0.655 | 0.727 | **Kept** (0.62, median 0.83) |
| T11 score v2 | Score v2: edge = boundary gradient / outer-ring gradient, + sigmoid brightness contrast; extras >= 0.25 | Region table: brightness contrast is the best cue (AUC 0.92); image-median edge fails on textured bg | 0.682 | 0.730 | 0.784 | **Kept** (+0.065) |
| T12a grow1.3 | Watershed search region 1.6 to 1.3 x core ellipse | Large search region lets growth leak | 0.703 | 0.756 | 0.815 | **Kept** (0.703) |
| T12b grow2.0 | Search region 2.0x | Check other direction | 0.635 | 0.687 | 0.730 | Discarded: leaks (0.635) |
| T12c grow1.15 | Search region 1.15x | Check smaller | 0.683 | 0.741 | 0.805 | Discarded: cuts eggs (0.683) |
| T12d grow1.3 erode9 | grow 1.3 + core erosion 9 px | Pull back slightly leaking cores | 0.703 | 0.756 | 0.815 | Discarded: no change, simpler version kept |
| T12e grow1.3 topk8 | grow 1.3 + top-8 candidates per view | Larger candidate pool | 0.700 | 0.754 | 0.811 | Discarded: no gain, 60% slower |
| T13a hull | + final mask = convex hull | Smooth jagged watershed outline | 0.696 | 0.752 | 0.815 | Discarded: over-fills eggs partly hidden by fingers/rods |
| T13b ellipse | + final mask = fitted ellipse | Same | 0.688 | 0.748 | 0.805 | Discarded: 0.688, boundary F1 drops |
| T13c smooth11 | + final opening + closing (11 px) per egg | Smooth jagged watershed outline | 0.704 | 0.756 | 0.815 | **Kept = FINAL** (boundary F1 0.584 to 0.590) |
