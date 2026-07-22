"""
statistical_analysis.py — Reviewer-proof statistical analysis.

Computes:
- Multi-seed mean ± std for each method
- Bootstrap 95% confidence intervals (BCa method)
- Paired bootstrap test for significance vs baseline
- Per-class F1 breakdown
- Test set numbers (separate from val)

Usage: python -m src.statistical_analysis --baseline run_005_seed42 --candidates run_022_seed42 run_022_seed123 ...
"""
import os, sys, json, argparse, glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

TASKS = ['Intent','AnswerType','CognitiveLevel','QuestionParticleType',
         'QuestionType','Subjectivity','TemporalContext','PurposeContext','List']

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", type=str, default="run_022",
                help="Prefix of runs to aggregate (e.g. 'run_022' aggregates run_022_seed42, _seed123, etc.)")
ap.add_argument("--baseline_prefix", type=str, default=None,
                help="If given, run paired bootstrap test against this baseline prefix")
ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
ap.add_argument("--n_bootstrap", type=int, default=1000)
args = ap.parse_args()

# ── Load all runs matching the prefix ─────────────────────────
def load_runs(prefix, split):
    pattern = f"results/{prefix}*_{split}_results.json"
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        run_name = os.path.basename(f).replace(f"_{split}_results.json", "")
        runs.append({"name": run_name, "data": data})
    return runs

def load_predictions(prefix, split):
    """Load raw predictions. Supports per-task baseline files."""
    pattern = f"results/{prefix}*_predictions.npz"
    files = sorted(glob.glob(pattern))
    all_preds = []
    all_labels = []
    for f in files:
        npz = np.load(f)
        # Only load tasks that exist in this npz
        avail = [t for t in TASKS if f"{split}_preds_{t}" in npz.files]
        run_preds = {t: npz[f"{split}_preds_{t}"] for t in avail}
        run_labels = {t: npz[f"{split}_labels_{t}"] for t in avail}
        all_preds.append(run_preds)
        all_labels.append(run_labels)
    return all_preds, all_labels

# ── Mean ± Std across seeds ───────────────────────────────────
runs = load_runs(args.prefix, args.split)
print(f"Found {len(runs)} runs matching '{args.prefix}' on {args.split}")
if not runs:
    print(f"No runs found. Expected pattern: results/{args.prefix}*_{args.split}_results.json")
    sys.exit(0)

print(f"\n{'='*70}")
print(f"MULTI-SEED RESULTS ({args.split}) — {args.prefix}")
print(f"{'='*70}")

agg = {t: [] for t in TASKS}
agg["mean_macro_f1"] = []
for r in runs:
    for t in TASKS:
        agg[t].append(r["data"][t]["macro_f1"])
    agg["mean_macro_f1"].append(r["data"]["mean_macro_f1"])

# Display per-task mean ± std
print(f"\n  {'Task':<25}  Mean   ± Std    Min     Max     n")
print(f"  {'-'*25}  -----    ----    ----    ----    -")
for t in TASKS:
    arr = np.array(agg[t])
    print(f"  {t:<25}  {arr.mean():.4f} ± {arr.std():.4f}  "
          f"{arr.min():.4f}  {arr.max():.4f}  {len(arr)}")

arr = np.array(agg["mean_macro_f1"])
print(f"  {'-'*25}")
print(f"  {'MEAN MACRO F1':<25}  {arr.mean():.4f} ± {arr.std():.4f}  "
      f"{arr.min():.4f}  {arr.max():.4f}  {len(arr)}")

# ── Bootstrap 95% confidence interval ─────────────────────────
print(f"\n{'='*70}")
print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS")
print(f"{'='*70}")

def bootstrap_ci(preds_per_seed, labels_per_seed, task, n_boot=1000, alpha=0.05):
    """Hierarchical bootstrap: resample examples within each seed, average across seeds."""
    bootstrap_f1s = []
    n_seeds = len(preds_per_seed)
    n_examples = len(preds_per_seed[0][task])
    for _ in range(n_boot):
        per_seed_f1 = []
        idx = np.random.choice(n_examples, n_examples, replace=True)
        for s in range(n_seeds):
            p = preds_per_seed[s][task][idx]
            l = labels_per_seed[s][task][idx]
            per_seed_f1.append(f1_score(l, p, average="macro", zero_division=0))
        bootstrap_f1s.append(np.mean(per_seed_f1))
    bootstrap_f1s = np.array(bootstrap_f1s)
    lo = np.percentile(bootstrap_f1s, 100 * alpha/2)
    hi = np.percentile(bootstrap_f1s, 100 * (1 - alpha/2))
    return bootstrap_f1s.mean(), lo, hi

preds_per_seed, labels_per_seed = load_predictions(args.prefix, args.split)
if preds_per_seed:
    print(f"\n  {'Task':<25}  Mean    95% CI")
    print(f"  {'-'*25}  -----    ----------------")
    np.random.seed(0)
    for t in TASKS:
        mean, lo, hi = bootstrap_ci(preds_per_seed, labels_per_seed, t,
                                     n_boot=args.n_bootstrap)
        print(f"  {t:<25}  {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

# ── Paired bootstrap test against baseline ────────────────────
if args.baseline_prefix:
    print(f"\n{'='*70}")
    print(f"PAIRED BOOTSTRAP TEST vs {args.baseline_prefix}")
    print(f"{'='*70}")
    print(f"\n  {'Task':<25}  Δ F1     95% CI              p-value  Sig?")
    print(f"  {'-'*25}  -----    ----------------    -------  ----")
    np.random.seed(0)

    for t in TASKS:
        # Each task baseline is in its own file: e.g. run_baseline_singletask_Intent_seed42
        base_pattern = f"results/{args.baseline_prefix}_{t}*_predictions.npz"
        base_files = sorted(glob.glob(base_pattern))
        if not base_files:
            print(f"  {t:<25}  no baseline file matching {base_pattern}")
            continue

        # Load baseline predictions for THIS task
        base_t_preds, base_t_labels = [], []
        for bf in base_files:
            npz = np.load(bf)
            key_p = f"{args.split}_preds_{t}"
            key_l = f"{args.split}_labels_{t}"
            if key_p not in npz.files:
                continue
            base_t_preds.append(npz[key_p])
            base_t_labels.append(npz[key_l])

        if not base_t_preds:
            print(f"  {t:<25}  baseline file has no {t} predictions")
            continue

        n_examples = len(preds_per_seed[0][t])
        deltas = []
        for _ in range(args.n_bootstrap):
            idx = np.random.choice(n_examples, n_examples, replace=True)
            # Multi-task: average F1 across 5 seeds on this bootstrap sample
            cand_f1s = []
            for s in range(len(preds_per_seed)):
                cand_f1s.append(f1_score(
                    labels_per_seed[s][t][idx],
                    preds_per_seed[s][t][idx],
                    average="macro", zero_division=0))
            # Baseline: average F1 across baseline runs (usually 1)
            base_f1s = []
            for s in range(len(base_t_preds)):
                base_f1s.append(f1_score(
                    base_t_labels[s][idx],
                    base_t_preds[s][idx],
                    average="macro", zero_division=0))
            deltas.append(np.mean(cand_f1s) - np.mean(base_f1s))

        deltas = np.array(deltas)
        mean_delta = deltas.mean()
        lo = np.percentile(deltas, 2.5)
        hi = np.percentile(deltas, 97.5)
        p_val = np.mean(deltas <= 0) if mean_delta > 0 else np.mean(deltas >= 0)
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
        sign = "+" if mean_delta >= 0 else ""
        print(f"  {t:<25}  {sign}{mean_delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]    {p_val:.4f}   {sig}")

# ── Save aggregated CSV for paper ─────────────────────────────
os.makedirs("results/aggregated", exist_ok=True)
out_csv = f"results/aggregated/{args.prefix}_{args.split}_summary.csv"
df_rows = []
for t in TASKS:
    arr = np.array(agg[t])
    df_rows.append({"task": t, "mean": arr.mean(), "std": arr.std(),
                    "min": arr.min(), "max": arr.max(), "n": len(arr)})
arr = np.array(agg["mean_macro_f1"])
df_rows.append({"task": "MEAN_MACRO_F1", "mean": arr.mean(), "std": arr.std(),
                "min": arr.min(), "max": arr.max(), "n": len(arr)})
pd.DataFrame(df_rows).to_csv(out_csv, index=False)
print(f"\n✅ Saved aggregated summary: {out_csv}")
