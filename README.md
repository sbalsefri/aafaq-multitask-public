# Efficient Multi-Task Arabic Question Classification with a Shared MARBERT Encoder

> One **MARBERTv2** encoder with **nine** classification heads matches nine separate
> single-task models on the **AAFAQ** dataset — at about **9× less storage** and
> **10× less inference time** when all nine outputs are required.

Code, split metadata, and analysis scripts for the paper
*"Efficient Multi-Task Arabic Question Classification with a Shared MARBERT Encoder"*
(Alsafari & Yafoz, manuscript under review).

A single MARBERTv2 encoder with nine task-specific linear heads is trained jointly on
AAFAQ (5,009 Modern Standard Arabic questions annotated along nine dimensions), using
**focal loss (γ = 5)** and **Kendall homoscedastic uncertainty weighting**. The
multi-task model is statistically equivalent to nine single-task baselines on mean
macro-F1 (five seeds), while producing all nine outputs from one shared encoder.

## Highlights
- **One model, nine tasks** — a shared encoder replaces nine per-task models.
- **Statistically equivalent, not just non-significant** — a two one-sided test (TOST)
  finds the multi-task and single-task settings equivalent on mean macro-F1 within a
  ±0.02 margin across five seeds.
- **Measured efficiency** — ~9× fewer parameters/disk and ~10× faster inference when all
  nine outputs are required.
- **Reproducible** — deterministic 80/10/10 split (stratified on Intent), multi-seed
  protocol, bootstrap confidence intervals, and paired significance testing.

---

## Repository structure
```
.
├── configs/
│   └── config.yaml              # headline configuration (γ=5, UW on, 9 tasks, 20 epochs)
├── src/
│   ├── preprocess.py            # deterministic stratified 80/10/10 split + label encoders
│   ├── dataset.py               # PyTorch Dataset for AAFAQ
│   ├── model.py                 # shared encoder + 9 linear heads
│   ├── rigorous_train.py        # headline trainer (focal + uncertainty weighting, multi-seed)
│   ├── evaluate.py              # val/test evaluation, confusion matrices, error CSV
│   ├── statistical_analysis.py  # multi-seed mean±std, bootstrap CIs, paired bootstrap test
│   └── error_analysis.py        # per-class reports + confusion matrices from saved predictions
├── data/                        # dataset + split artifacts (see data/README.md)
├── requirements.txt
├── CITATION.cff
└── LICENSE                      # MIT
```

---

## Setup
```bash
pip install -r requirements.txt
```
Experiments were run on Google Colab with a single **NVIDIA T4 GPU (16 GB)**,
PyTorch 2.x and HuggingFace Transformers.

---

## Data
The AAFAQ dataset is **not** redistributed here. Download it from the original
publication and place it at `data/AAFAQ.csv` (see [`data/README.md`](data/README.md)):

> Abdelaziz, M.E.; Deif, M.A.; Algamdi, S.A.; Elgohary, R. *A Benchmark Arabic Dataset for
> Arabic Question Classification Using AAFAQ Framework.* Scientific Data **12**, 1444 (2025).
> https://doi.org/10.1038/s41597-025-05688-0

Then regenerate the exact split (deterministic, stratified on Intent):
```bash
python -m src.preprocess
```

---

## Reproducing the results

**1. Train the multi-task model across the five seeds** (headline config in `configs/config.yaml`):
```bash
for S in 42 123 456 789 2024; do
    python -m src.rigorous_train --seed $S --run_suffix _seed$S
done
```
Each run saves per-seed val/test metrics (`results/*_results.json`) and raw predictions
(`results/*_predictions.npz`) used for bootstrap confidence intervals.

**2. Aggregate across seeds with bootstrap CIs and paired significance tests:**
```bash
python -m src.statistical_analysis --prefix marbert_multitask --split test
# add --baseline_prefix <single_task_prefix> for the paired bootstrap comparison
```

**3. Per-class error analysis (Section 5.2):**
```bash
python -m src.error_analysis --run marbert_multitask_seed42 --split test
```

The prediction files in `results/` are sufficient to regenerate all tables and figures
without re-training.

---

## Configuration
All hyperparameters live in `configs/config.yaml`. The released values are the paper's
headline configuration: MARBERTv2 encoder, nine active tasks, focal loss γ = 5, class
weighting off, **uncertainty weighting on**, encoder LR 2e-5 (heads 10×), 20 epochs,
early-stopping patience 5, and seeds 42/123/456/789/2024.

---

## Citation
If you use this code, please cite the paper (see `CITATION.cff`):
```bibtex
@article{alsafari2026aafaq,
  title   = {Efficient Multi-Task Arabic Question Classification with a Shared MARBERT Encoder},
  author  = {Alsafari, Safa and Yafoz, Ayman},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License
Released under the [MIT License](LICENSE).
