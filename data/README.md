# Data

This folder holds the dataset and derived split artifacts.

## What is committed here
- `label_encoders.json` — class names and counts per task (no question text).
- `split_info.json` — seed and split sizes for the exact partition used in the paper.

## What you must add locally
The raw **AAFAQ** dataset is **not** redistributed here (it belongs to the original authors).
Download it from the original publication and place it at `data/AAFAQ.csv`:

> Abdelaziz, M.E.; Deif, M.A.; Algamdi, S.A.; Elgohary, R. *A Benchmark Arabic Dataset for
> Arabic Question Classification Using AAFAQ Framework.* Scientific Data 12, 1444 (2025).
> https://doi.org/10.1038/s41597-025-05688-0

## Regenerating the exact split
The split is deterministic (stratified on `Intent`, fixed seed). After placing `AAFAQ.csv`:

```bash
python -m src.preprocess
```

This writes `data/splits/{train,val,test}.csv` and `data/label_encoders/label_encoders.json`,
reproducing the 80/10/10 partition used for all experiments.
