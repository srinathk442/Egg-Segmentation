# Use of AI tools

**Claude Code** (Anthropic's AI coding assistant, model Claude Opus) was used throughout this project as a pair
programmer:

* **Code:** drafting and debugging the Colab notebook, the split script, the classical segmenter, metrics, the
  experiment runner, diagnostics, the tracker, plotting and report-building scripts.
* **Experiments:** running trials and diagnostics on the dev split and summarising their results.
* **Writing:** first drafts of this report and the README.

How it was used: the work was done step by step. At each step the approach and the underlying concepts (Grounded-SAM,
leakage-aware splitting, Otsu, morphology, threshold sweep / MSER idea, marker-controlled watershed, GrabCut, Kalman
filter, Hungarian assignment) were discussed before implementation, and design decisions were taken on the basis of
the measured results in the experiment log. All results in this report come from the code in the repository and can
be reproduced with the commands in the README.
