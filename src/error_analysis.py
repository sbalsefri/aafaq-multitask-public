"""
error_analysis.py — Per-class breakdown + confusion matrices + qualitative examples.

For each task:
- Per-class precision, recall, F1
- Confusion matrix saved as PNG
- Top-K misclassified examples for qualitative analysis

Usage: python -m src.error_analysis --run run_022_seed42 --split test
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

TASKS = ['Intent','AnswerType','CognitiveLevel','QuestionParticleType',
         'QuestionType','Subjectivity','TemporalContext','PurposeContext','List']

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=str, required=True)
ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
args = ap.parse_args()

# Load predictions
npz_path = f"results/{args.run}_predictions.npz"
if not os.path.exists(npz_path):
    print(f"❌ Predictions not found: {npz_path}")
    sys.exit(1)
npz = np.load(npz_path)

# Load label encoders for class names
with open("data/label_encoders/label_encoders.json") as f:
    encoders = json.load(f)

# Load split CSV for question text
df = pd.read_csv(f"data/splits/{args.split}.csv")

out_dir = f"results/error_analysis/{args.run}_{args.split}"
os.makedirs(out_dir, exist_ok=True)

# ── Per-task analysis ─────────────────────────────────────────
print(f"\nError analysis: {args.run} on {args.split}\n")
all_errors = []

for t in TASKS:
    preds  = npz[f"{args.split}_preds_{t}"]
    labels = npz[f"{args.split}_labels_{t}"]
    classes = encoders[t]["classes"]
    n_cls = len(classes)

    # Classification report
    print(f"\n{'='*60}")
    print(f"Task: {t}")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=classes, zero_division=0))

    # Save report as CSV
    rep = classification_report(labels, preds, target_names=classes,
                                  zero_division=0, output_dict=True)
    pd.DataFrame(rep).T.to_csv(f"{out_dir}/{t}_classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    plt.figure(figsize=(max(6, n_cls), max(5, n_cls * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{t} Confusion Matrix ({args.split})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{t}_confusion_matrix.png", dpi=150)
    plt.close()

    # Errors: predictions that don't match
    err_mask = preds != labels
    err_indices = np.where(err_mask)[0]
    for idx in err_indices[:20]:  # top 20 errors per task
        all_errors.append({
            "task": t,
            "question": df.iloc[idx]["QuestionText"],
            "true": classes[labels[idx]],
            "predicted": classes[preds[idx]],
            "row_idx": int(idx),
        })

# Save all errors
pd.DataFrame(all_errors).to_csv(f"{out_dir}/error_examples.csv", index=False)
print(f"\n✅ Saved error analysis to {out_dir}/")
print(f"   - Per-task classification reports (.csv)")
print(f"   - Per-task confusion matrices (.png)")
print(f"   - error_examples.csv ({len(all_errors)} samples)")
