"""
Build every figure as a single merged chart containing BOTH datasets.

No titles on any chart. Axis labels only. No baseline lines.

Reads:
    <assist_dir>/results/<assist_tag>_history.csv      + _checkpoint.pt
    <oulad_dir>/results/<oulad_tag>_history.csv        + _checkpoint.pt

Writes 12 files into --out:
    accuracy_curve.png          recall_curve.png
    aggregated_loss_curve.png   roc_auc_curve.png
    f1_score_curve.png          specificity_curve.png
    kappa_curve.png             training_time_curve.png
    mcc_curve.png               bandwidth_curve.png
    precision_curve.png         dataset_comparison.png

Usage:
    python3 make_all_merged.py \
        --assist ~/Documents/fedkt/results/final \
        --oulad  results/L30r100 \
        --out    figures_merged
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_LABEL = "ASSISTments 2012--13"
O_LABEL = "OULAD"
A_STYLE = dict(color="#d95f02", linestyle="--", marker="s")
O_STYLE = dict(color="#1f77b4", linestyle="-", marker="o")

CURVES = [
    ("accuracy_curve",        "accuracy",     "Aggregated Accuracy"),
    ("aggregated_loss_curve", "loss",         "Aggregated Loss"),
    ("f1_score_curve",        "f1_score",     "Aggregated F1 Score"),
    ("kappa_curve",           "cohens_kappa", "Aggregated Kappa"),
    ("mcc_curve",             "mcc",          "Aggregated MCC"),
    ("precision_curve",       "precision",    "Aggregated Precision"),
    ("recall_curve",          "recall",       "Aggregated Recall"),
    ("roc_auc_curve",         "roc_auc",      "Aggregated ROC-AUC"),
    ("specificity_curve",     "specificity",  "Aggregated Specificity"),
]

# Final test metrics, used for the grouped bar chart.
FINAL = {
    "Accuracy":       (0.7473, 0.9215),
    "Precision":      (0.7676, 0.8874),
    "Recall":         (0.9031, 0.9521),
    "Specificity":    (0.4130, 0.8949),
    "F1-score":       (0.8298, 0.9186),
    "ROC-AUC":        (0.7722, 0.9720),
    "MCC":            (0.3697, 0.8449),
    "Cohen's $\\kappa$": (0.3517, 0.8430),
}


def read_history(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    rounds, cols = [], {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        names = [n for n in r.fieldnames if n != "round"]
        for n in names:
            cols[n] = []
        for row in r:
            rounds.append(int(row["round"]))
            for n in names:
                cols[n].append(float(row[n]))
    return rounds, cols


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def merged_curve(a_rounds, a_vals, o_rounds, o_vals, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if a_vals is not None:
        ax.plot(a_rounds, a_vals, label=A_LABEL, linewidth=1.9, markersize=6,
                markevery=max(1, len(a_rounds) // 30), **A_STYLE)
    if o_vals is not None:
        ax.plot(o_rounds, o_vals, label=O_LABEL, linewidth=1.9, markersize=6,
                markevery=max(1, len(o_rounds) // 30), **O_STYLE)
    ax.set_xlabel("Communication Round", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True, color="#cccccc", linewidth=0.8)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, frameon=True, framealpha=0.95)
    save(fig, out_path)


def client_means(ckpt_path, key, n_rounds):
    """Mean across clients, per round, from a run checkpoint."""
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.exists(ckpt_path):
        print(f"  [skip] {ckpt_path} not found")
        return None
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    per_client = ck.get(key, {})
    series = [v for v in per_client.values() if v]
    if not series:
        return None
    n = min(min(len(s) for s in series), n_rounds)
    return [sum(s[i] for s in series) / len(series) for i in range(n)]


def bar_comparison(out_path):
    labels = list(FINAL)
    a = [FINAL[k][0] for k in labels]
    o = [FINAL[k][1] for k in labels]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6.5))
    b1 = ax.bar([i - w / 2 for i in x], a, w, label=A_LABEL, color="#d95f02")
    b2 = ax.bar([i + w / 2 for i in x], o, w, label=O_LABEL, color="#1f77b4")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc")
    ax.set_axisbelow(True)
    ax.legend(fontsize=13, loc="lower left", ncol=2, framealpha=0.95)
    save(fig, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assist", required=True,
                    help="path prefix, e.g. ~/Documents/fedkt/results/final")
    ap.add_argument("--oulad", required=True,
                    help="path prefix, e.g. results/L30r100")
    ap.add_argument("--out", default="figures_merged")
    args = ap.parse_args()

    a_rounds, a_cols = read_history(args.assist + "_history.csv")
    o_rounds, o_cols = read_history(args.oulad + "_history.csv")
    print(f"ASSISTments : {len(a_rounds)} rounds")
    print(f"OULAD       : {len(o_rounds)} rounds")

    os.makedirs(args.out, exist_ok=True)
    print("building merged figures...")

    for stem, col, ylabel in CURVES:
        merged_curve(a_rounds, a_cols.get(col), o_rounds, o_cols.get(col),
                     ylabel, os.path.join(args.out, f"{stem}.png"))

    a_t = client_means(args.assist + "_checkpoint.pt", "client_times", len(a_rounds))
    o_t = client_means(args.oulad + "_checkpoint.pt", "client_times", len(o_rounds))
    if a_t or o_t:
        merged_curve(range(1, len(a_t) + 1) if a_t else None, a_t,
                     range(1, len(o_t) + 1) if o_t else None, o_t,
                     "Mean Local Training Time (s)",
                     os.path.join(args.out, "training_time_curve.png"))

    a_b = client_means(args.assist + "_checkpoint.pt", "client_upload", len(a_rounds))
    o_b = client_means(args.oulad + "_checkpoint.pt", "client_upload", len(o_rounds))
    if a_b or o_b:
        merged_curve(range(1, len(a_b) + 1) if a_b else None, a_b,
                     range(1, len(o_b) + 1) if o_b else None, o_b,
                     "Model Update Uploaded (MB)",
                     os.path.join(args.out, "bandwidth_curve.png"))

    bar_comparison(os.path.join(args.out, "dataset_comparison.png"))
    print(f"\nall figures written to {args.out}/")


if __name__ == "__main__":
    main()
