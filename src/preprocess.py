"""
preprocess.py — RUN ONCE ONLY
Outputs saved to data/splits/ and data/label_encoders/
Never re-run this after your first experiment — it would change the split
and make all runs incomparable.
"""

import pandas as pd
import numpy as np
import json
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Load config ──────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg["data"]
SEED = cfg["experiment"]["seed"]

TASK_COLUMNS = [
    "Intent", "AnswerType", "CognitiveLevel", "QuestionParticleType",
    "QuestionType", "Subjectivity", "TemporalContext", "PurposeContext", "List"
]

os.makedirs(DATA_CFG["splits_dir"], exist_ok=True)
os.makedirs(DATA_CFG["encoders_dir"], exist_ok=True)

# ── Load raw data ─────────────────────────────────────────────
print("Loading AAFAQ.csv...")
df = pd.read_csv(DATA_CFG["raw_path"], encoding="utf-8-sig")
print(f"  Raw shape: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────
# Fix whitespace in label values (e.g. ' Synthesis' with leading space)
for col in TASK_COLUMNS:
    df[col] = df[col].astype(str).str.strip()

# Fix boolean List column
df["List"] = df["List"].astype(str).str.strip().str.upper()
df["List"] = df["List"].map({"TRUE": "List", "FALSE": "Single"})

# Drop rows with missing QuestionText or any label
df = df.dropna(subset=["QuestionText"] + TASK_COLUMNS)
print(f"  After cleaning: {df.shape}")

# ── Label encoding ────────────────────────────────────────────
print("\nEncoding labels...")
encoders = {}
for col in TASK_COLUMNS:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col])
    encoders[col] = {
        "classes": le.classes_.tolist(),
        "num_classes": len(le.classes_)
    }
    print(f"  {col}: {len(le.classes_)} classes → {le.classes_.tolist()}")

# Save encoders
encoder_path = os.path.join(DATA_CFG["encoders_dir"], "label_encoders.json")
with open(encoder_path, "w", encoding="utf-8") as f:
    json.dump(encoders, f, ensure_ascii=False, indent=2)
print(f"\nLabel encoders saved to {encoder_path}")

# ── Stratified split ─────────────────────────────────────────
# Stratify on Intent (most complex, 14 classes)
stratify_col = f"{DATA_CFG['stratify_on']}_encoded"

print(f"\nSplitting: {DATA_CFG['train_ratio']*100:.0f}/{DATA_CFG['val_ratio']*100:.0f}/{DATA_CFG['test_ratio']*100:.0f}")
print(f"Stratifying on: {DATA_CFG['stratify_on']}")

# First split: train vs (val+test)
val_test_ratio = DATA_CFG["val_ratio"] + DATA_CFG["test_ratio"]
train_df, valtest_df = train_test_split(
    df,
    test_size=val_test_ratio,
    random_state=SEED,
    stratify=df[stratify_col]
)

# Second split: val vs test
val_ratio_adjusted = DATA_CFG["val_ratio"] / val_test_ratio
val_df, test_df = train_test_split(
    valtest_df,
    test_size=(1 - val_ratio_adjusted),
    random_state=SEED,
    stratify=valtest_df[stratify_col]
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

# Save splits
train_df.to_csv(os.path.join(DATA_CFG["splits_dir"], "train.csv"), index=False, encoding="utf-8")
val_df.to_csv(os.path.join(DATA_CFG["splits_dir"], "val.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(DATA_CFG["splits_dir"], "test.csv"), index=False, encoding="utf-8")

# Save split info for reproducibility
split_info = {
    "seed": SEED,
    "total": len(df),
    "train": len(train_df),
    "val": len(val_df),
    "test": len(test_df),
    "stratify_on": DATA_CFG["stratify_on"],
    "columns": TASK_COLUMNS
}
with open(os.path.join(DATA_CFG["splits_dir"], "split_info.json"), "w") as f:
    json.dump(split_info, f, indent=2)

# ── Class distribution check ──────────────────────────────────
print("\n⚠️  Class imbalance summary (train set):")
for col in TASK_COLUMNS:
    counts = train_df[col].value_counts()
    min_class = counts.idxmin()
    min_count = counts.min()
    max_count = counts.max()
    ratio = max_count / min_count
    flag = " ← SEVERE" if ratio > 20 else (" ← MODERATE" if ratio > 5 else "")
    print(f"  {col}: min={min_count} ({min_class}), max={max_count}, ratio={ratio:.1f}x{flag}")

print("\n✅ Preprocessing complete. Splits saved to data/splits/")
print("   DO NOT re-run this script. All future experiments use these same splits.")
