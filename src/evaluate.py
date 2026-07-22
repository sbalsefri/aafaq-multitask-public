"""
evaluate.py — Full evaluation on VALIDATION set (during research)
             or TEST set (only when writing the paper)

IMPORTANT: Only run on test set when you have finished all iterations.
           See ITERATION_GUIDE.md → STOP CONDITIONS
"""

import os, json, yaml, argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, accuracy_score, classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import AAFAQDataset, get_num_classes
from src.model import MultiTaskClassifier



# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pt checkpoint file")
parser.add_argument("--split", type=str, default="val",
                    choices=["val", "test"],
                    help="Which split to evaluate. Use 'test' ONLY for final paper results.")
parser.add_argument("--save-errors", action="store_true",
                    help="Save error analysis CSV for ErrorAnalysis notebook")
args = parser.parse_args()

if args.split == "test":
    print("\n" + "="*60)
    print("⚠️  WARNING: You are evaluating on the TEST set.")
    print("   Only do this when you have finished ALL iterations.")
    print("   See ITERATION_GUIDE.md → STOP CONDITIONS")
    confirm = input("   Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        exit()

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA = cfg["data"]
MODEL_CFG = cfg["model"]
TASKS_CFG = cfg["tasks"]
TRAIN_CFG = cfg["training"]
LOG_CFG = cfg["logging"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVE_TASKS = TASKS_CFG["active"]
ENCODER_PATH = os.path.join(DATA["encoders_dir"], "label_encoders.json")
NUM_CLASSES = get_num_classes(ENCODER_PATH, ACTIVE_TASKS)

with open(ENCODER_PATH) as f:
    ENCODERS = json.load(f)

# ── Load model ────────────────────────────────────────────────
#ckpt = torch.load(args.checkpoint, map_location=DEVICE)
#ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
# ── Load model ────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

model = MultiTaskClassifier(
    encoder_name=MODEL_CFG["encoder_name"],
    num_classes_per_task=NUM_CLASSES,
    dropout=MODEL_CFG["dropout"],
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Dataset ───────────────────────────────────────────────────
split_csv = os.path.join(DATA["splits_dir"], f"{args.split}.csv")
dataset = AAFAQDataset(split_csv, MODEL_CFG["encoder_name"],
                        MODEL_CFG["max_length"], ACTIVE_TASKS)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
raw_df = pd.read_csv(split_csv, encoding="utf-8")

# ── Inference ─────────────────────────────────────────────────
all_preds = {t: [] for t in ACTIVE_TASKS}
all_labels = {t: [] for t in ACTIVE_TASKS}
all_probs = {t: [] for t in ACTIVE_TASKS}

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        for task in ACTIVE_TASKS:
            probs = torch.softmax(logits[task], dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            labs = batch[task].numpy()
            all_preds[task].extend(preds)
            all_labels[task].extend(labs)
            all_probs[task].extend(probs.max(axis=-1))

# ── Metrics ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS — Split: {args.split.upper()}")
print(f"Checkpoint: {args.checkpoint}")
print(f"{'='*60}\n")

results = {}
for task in ACTIVE_TASKS:
    acc = accuracy_score(all_labels[task], all_preds[task])
    f1_macro = f1_score(all_labels[task], all_preds[task], average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels[task], all_preds[task], average="weighted", zero_division=0)
    class_names = ENCODERS[task]["classes"]
    report = classification_report(all_labels[task], all_preds[task],
                                    target_names=class_names, zero_division=0)
    results[task] = {
        "accuracy": acc, "macro_f1": f1_macro, "weighted_f1": f1_weighted,
        "report": report
    }
    print(f"{'─'*50}")
    print(f"Task: {task}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro-F1:    {f1_macro:.4f}")
    print(f"  Weighted-F1: {f1_weighted:.4f}")
    print(f"\n{report}")

mean_f1 = np.mean([results[t]["macro_f1"] for t in ACTIVE_TASKS])
print(f"\n{'='*60}")
print(f"MEAN MACRO-F1 across all tasks: {mean_f1:.4f}")
print(f"{'='*60}")

# ── Confusion matrices ────────────────────────────────────────
cm_dir = os.path.join(LOG_CFG["results_dir"], "confusion_matrices")
os.makedirs(cm_dir, exist_ok=True)
for task in ACTIVE_TASKS:
    class_names = ENCODERS[task]["classes"]
    cm = confusion_matrix(all_labels[task], all_preds[task])
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Confusion Matrix: {task}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(cm_dir, f"{task}_cm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
print(f"\nConfusion matrices saved to {cm_dir}/")

# ── Error analysis output ─────────────────────────────────────
if args.save_errors:
    error_rows = []
    for task in ACTIVE_TASKS:
        class_names = ENCODERS[task]["classes"]
        for i, (pred, true, conf) in enumerate(
                zip(all_preds[task], all_labels[task], all_probs[task])):
            if pred != true:
                error_rows.append({
                    "task": task,
                    "question": raw_df.iloc[i]["QuestionText"],
                    "true_label": class_names[true],
                    "pred_label": class_names[pred],
                    "confidence": conf
                })
    error_df = pd.DataFrame(error_rows).sort_values("confidence", ascending=False)
    error_path = os.path.join(LOG_CFG["results_dir"], "error_analysis.csv")
    error_df.to_csv(error_path, index=False, encoding="utf-8")
    print(f"Error analysis saved to {error_path}")
    print(f"Total errors: {len(error_df)} across all tasks")

# ── Save summary JSON ─────────────────────────────────────────
run_name = cfg["experiment"]["run_name"]
summary = {t: {"accuracy": results[t]["accuracy"], "macro_f1": results[t]["macro_f1"],
               "weighted_f1": results[t]["weighted_f1"]} for t in ACTIVE_TASKS}
summary["mean_macro_f1"] = mean_f1
summary["split"] = args.split
summary["checkpoint"] = args.checkpoint

out_path = os.path.join(LOG_CFG["results_dir"], f"{run_name}_{args.split}_results.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults summary saved to {out_path}")
print(f"\n👉 Open notebooks/01_Results.ipynb to compare with previous runs")
