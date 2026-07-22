# Repository cleanup for public release

This repository was pruned to contain **only** the code and artifacts used to produce
the results reported in the paper. The following were removed from the internal
working version, with rationale:

## Removed scripts (not part of the paper's headline method or reported results)
- `advanced_train.py` — exploratory trainer with Supervised Contrastive + R-Drop +
  label smoothing. Not the headline method; its extra objectives are not reported.
- `ensemble.py` — per-task checkpoint ensembling. Not used in the paper.
- `train.py` — superseded MLflow-based trainer. Replaced by `src/rigorous_train.py`
  (the script behind the reported numbers). Removing it also drops the MLflow dependency.
- `ablation.py` — an older ablation runner whose ablations do **not** match the paper's
  Section 5.3 ablation (which was produced by config variations of `rigorous_train.py`).
- `losses.py` — unused by the retained code. The focal loss and Kendall uncertainty
  weighting used in the paper are defined inline in `src/rigorous_train.py`.
  (This file also contained a second, regression-form uncertainty-weighting variant
  that did not match the paper's Eq. 2; removing it avoids that inconsistency.)

## Edits to retained files
- `src/model.py` — removed unused `hierarchical` and `head_hidden_size` code paths
  (Phase-4 experiments not used in the paper). The headline model uses one linear head
  per task.
- `src/evaluate.py` — removed an unused `import mlflow` and the corresponding
  `head_hidden_size` / `hierarchical` constructor arguments.

## Config
- `configs/config.yaml` — reset to the paper's headline configuration (nine active tasks,
  focal gamma = 5, uncertainty weighting **on**, 20 epochs, patience 5,
  seeds 42/123/456/789/2024). The prior working snapshot had been left on a single-task
  run with several non-headline flags; those have been corrected/removed.
