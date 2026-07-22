"""
rigorous_train.py — Reviewer-proof training script.

Addresses all major reviewer concerns:
- Multi-seed support (--seed flag)
- Uncertainty-weighted multi-task loss (Kendall 2018) — principled task balancing
- Capped class weights with documented cap
- Saves both val AND test predictions per run
- Saves per-class metrics + raw predictions for bootstrap CI later
- Logs full hyperparameters and seed in checkpoint
"""
import os, sys, json, yaml, time, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, accuracy_score, classification_report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.dataset import AAFAQDataset, get_num_classes
from src.model import MultiTaskClassifier

# ── Args ──────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--run_suffix", type=str, default="",
                help="Appended to run_name for seed identification")
args = ap.parse_args()

# ── Seed everything ───────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}  |  Seed: {args.seed}")

# ── Load config ───────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

base_run_name = cfg["experiment"]["run_name"]
RUN_NAME    = f"{base_run_name}{args.run_suffix}" if args.run_suffix else base_run_name
ENCODER     = cfg["model"]["encoder_name"]
MAX_LENGTH  = cfg["model"]["max_length"]
TASKS       = cfg["tasks"]["active"]
EPOCHS      = cfg["training"]["epochs"]
PATIENCE    = cfg["training"]["patience"]
LR          = cfg["training"]["learning_rate"]
BATCH       = cfg["training"]["batch_size"]
FP16        = cfg["training"]["fp16"]
GAMMA       = cfg["loss"]["focal_gamma"]
CLASS_W     = cfg["loss"]["class_weighting"]
WEIGHT_CAP  = cfg.get("loss", {}).get("class_weight_cap", 10.0)

# Advanced
USE_UW              = cfg.get("advanced", {}).get("use_uncertainty_weighting", True)
LABEL_SMOOTHING     = cfg.get("advanced", {}).get("label_smoothing", 0.0)
USE_RDROP           = cfg.get("advanced", {}).get("use_rdrop", False)
RDROP_ALPHA         = cfg.get("advanced", {}).get("rdrop_alpha", 1.0)

print(f"\n{'='*60}")
print(f"Run: {RUN_NAME}")
print(f"Encoder: {ENCODER}  |  Tasks: {len(TASKS)}  |  Epochs: {EPOCHS}")
print(f"Loss: focal γ={GAMMA}  |  class_weights={CLASS_W} (cap={WEIGHT_CAP}x)")
print(f"Uncertainty weighting: {USE_UW}  |  Label smoothing: {LABEL_SMOOTHING}")
print(f"R-Drop: {USE_RDROP} (α={RDROP_ALPHA})")
print(f"{'='*60}\n")

# ── Datasets ──────────────────────────────────────────────────
train_ds = AAFAQDataset("data/splits/train.csv", tokenizer_name=ENCODER,
                        max_length=MAX_LENGTH, active_tasks=TASKS)
val_ds   = AAFAQDataset("data/splits/val.csv",   tokenizer_name=ENCODER,
                        max_length=MAX_LENGTH, active_tasks=TASKS)
test_ds  = AAFAQDataset("data/splits/test.csv",  tokenizer_name=ENCODER,
                        max_length=MAX_LENGTH, active_tasks=TASKS)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2)

num_classes = get_num_classes("data/label_encoders/label_encoders.json", TASKS)

# ── Class weights (capped at WEIGHT_CAP) ──────────────────────
class_weights = {}
if CLASS_W:
    df_train = pd.read_csv("data/splits/train.csv")
    for t in TASKS:
        col = f"{t}_encoded"
        counts = df_train[col].value_counts().sort_index().values
        weights = len(df_train) / (len(counts) * counts)
        weights = np.clip(weights, 1.0 / WEIGHT_CAP, WEIGHT_CAP)
        class_weights[t] = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ── Model ─────────────────────────────────────────────────────
model = MultiTaskClassifier(
    encoder_name=ENCODER,
    num_classes_per_task=num_classes,
    dropout=0.1,
).to(DEVICE)

# ── Losses ────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=5.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

focal_losses = {
    t: FocalLoss(gamma=GAMMA,
                 weight=class_weights.get(t),
                 label_smoothing=LABEL_SMOOTHING)
    for t in TASKS
}

# ── Uncertainty Weighting (Kendall 2018) ──────────────────────
# Each task gets a learned log-variance parameter; loss = (1/2σ²)·L_i + log σ
class UncertaintyWeighter(nn.Module):
    """Learnable log-variance per task for principled multi-task balancing."""
    def __init__(self, tasks):
        super().__init__()
        # Initialize log(σ²) = 0 for each task (σ = 1)
        self.log_vars = nn.Parameter(torch.zeros(len(tasks)))
        self.tasks = tasks

    def forward(self, task_losses):
        total = 0.0
        for i, t in enumerate(self.tasks):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * task_losses[t] + self.log_vars[i] / 2
        return total

uw = UncertaintyWeighter(TASKS).to(DEVICE) if USE_UW else None

def rdrop_kl_loss(logits1, logits2):
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    kl1 = F.kl_div(p1, F.softmax(logits2, dim=-1), reduction="batchmean")
    kl2 = F.kl_div(p2, F.softmax(logits1, dim=-1), reduction="batchmean")
    return (kl1 + kl2) / 2

# ── Optimizer + Scheduler ─────────────────────────────────────
encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
head_params    = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
param_groups = [
    {"params": encoder_params, "lr": LR},
    {"params": head_params,    "lr": LR * 10},
]
if USE_UW:
    param_groups.append({"params": uw.parameters(), "lr": LR * 10})

optimizer = AdamW(param_groups, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=FP16)

# ── Evaluation function ───────────────────────────────────────
def evaluate_loader(loader, return_preds=False):
    model.eval()
    preds = {t: [] for t in TASKS}
    labels = {t: [] for t in TASKS}
    probs = {t: [] for t in TASKS}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=FP16):
                logits = model(input_ids, attn_mask)
            for t in TASKS:
                p = torch.softmax(logits[t].float(), dim=-1)
                probs[t].append(p.cpu().numpy())
                preds[t].append(logits[t].argmax(dim=-1).cpu().numpy())
                labels[t].append(batch[t].numpy())

    metrics = {}
    f1s = []
    for t in TASKS:
        preds[t]  = np.concatenate(preds[t])
        labels[t] = np.concatenate(labels[t])
        probs[t]  = np.concatenate(probs[t])
        f1  = f1_score(labels[t], preds[t], average="macro", zero_division=0)
        acc = accuracy_score(labels[t], preds[t])
        # Per-class F1 for reviewer
        per_cls = f1_score(labels[t], preds[t], average=None, zero_division=0)
        metrics[t] = {"macro_f1": float(f1), "accuracy": float(acc),
                      "per_class_f1": per_cls.tolist()}
        f1s.append(f1)
    metrics["mean_macro_f1"] = float(np.mean(f1s))

    if return_preds:
        return metrics, preds, labels, probs
    return metrics

# ── Training Loop ─────────────────────────────────────────────
best_val_f1 = 0.0
patience_left = PATIENCE
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    if uw is not None:
        uw.train()
    t0 = time.time()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)
        labels = {t: batch[t].to(DEVICE) for t in TASKS}

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=FP16):
            logits1 = model(input_ids, attn_mask)
            task_losses = {t: focal_losses[t](logits1[t], labels[t]) for t in TASKS}

            if USE_RDROP:
                logits2 = model(input_ids, attn_mask)
                task_losses2 = {t: focal_losses[t](logits2[t], labels[t]) for t in TASKS}
                task_losses = {t: (task_losses[t] + task_losses2[t]) / 2 for t in TASKS}

            # Combine task losses
            if USE_UW:
                base_loss = uw(task_losses)
            else:
                base_loss = sum(task_losses.values())

            # R-Drop KL term
            rdrop_term = 0.0
            if USE_RDROP:
                for t in TASKS:
                    rdrop_term = rdrop_term + rdrop_kl_loss(logits1[t], logits2[t])
                base_loss = base_loss + RDROP_ALPHA * rdrop_term

        scaler.scale(base_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += base_loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)

    # Validation
    val_metrics = evaluate_loader(val_loader)
    mean_f1 = val_metrics["mean_macro_f1"]
    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Mean F1: {mean_f1:.4f} | Time: {elapsed:.0f}s")
    for t in TASKS:
        m = val_metrics[t]
        print(f"  {t:<25} F1={m['macro_f1']:.4f}  Acc={m['accuracy']:.4f}")

    history.append({"epoch": epoch, "train_loss": avg_loss, "val_mean_f1": mean_f1,
                    **{f"{t}_f1": val_metrics[t]["macro_f1"] for t in TASKS}})

    if mean_f1 > best_val_f1:
        best_val_f1 = mean_f1
        patience_left = PATIENCE
        ckpt_path = f"checkpoints/{RUN_NAME}_best.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "uw_state": uw.state_dict() if uw else None,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": cfg,
            "seed": args.seed,
        }, ckpt_path)
        print(f"  ✅ New best (mean F1={mean_f1:.4f})")
    else:
        patience_left -= 1
        print(f"  No improvement ({PATIENCE - patience_left}/{PATIENCE})")
        if patience_left <= 0:
            print(f"\n⏹️ Early stopping at epoch {epoch}")
            break

# ── Load best checkpoint and evaluate on BOTH val and test ────
print("\n" + "="*60)
print("FINAL EVALUATION on best checkpoint")
print("="*60)
ckpt = torch.load(f"checkpoints/{RUN_NAME}_best.pt", map_location=DEVICE,
                  weights_only=False)
model.load_state_dict(ckpt["model_state"])

# Validation
val_metrics, val_preds, val_labels, val_probs = evaluate_loader(val_loader, return_preds=True)
print(f"\n📊 VAL Mean F1: {val_metrics['mean_macro_f1']:.4f}")
for t in TASKS:
    print(f"  {t:<25} F1={val_metrics[t]['macro_f1']:.4f}")

# TEST (held out)
test_metrics, test_preds, test_labels, test_probs = evaluate_loader(test_loader, return_preds=True)
print(f"\n🎯 TEST Mean F1: {test_metrics['mean_macro_f1']:.4f}")
for t in TASKS:
    print(f"  {t:<25} F1={test_metrics[t]['macro_f1']:.4f}")

# ── Save val + test results in same JSON format ───────────────
os.makedirs("results", exist_ok=True)
with open(f"results/{RUN_NAME}_val_results.json", "w") as f:
    json.dump(val_metrics, f, indent=2)
with open(f"results/{RUN_NAME}_test_results.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

# Save predictions for bootstrap CI later
np.savez(f"results/{RUN_NAME}_predictions.npz",
         **{f"val_preds_{t}": val_preds[t] for t in TASKS},
         **{f"val_labels_{t}": val_labels[t] for t in TASKS},
         **{f"test_preds_{t}": test_preds[t] for t in TASKS},
         **{f"test_labels_{t}": test_labels[t] for t in TASKS})

# Save training history
pd.DataFrame(history).to_csv(f"results/{RUN_NAME}_history.csv", index=False)

print(f"\n✅ Saved: results/{RUN_NAME}_val_results.json")
print(f"✅ Saved: results/{RUN_NAME}_test_results.json")
print(f"✅ Saved: results/{RUN_NAME}_predictions.npz (for bootstrap CI)")
print(f"\nBest val F1: {best_val_f1:.4f}  |  Test F1: {test_metrics['mean_macro_f1']:.4f}")
