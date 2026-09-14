"""
Regenerate every figure in the results section from the saved run.

Produces, in figures/:
    fig02_accuracy.png       fig08_recall.png
    fig03_loss.png           fig09_roc_auc.png
    fig04_f1.png             fig10_specificity.png
    fig05_kappa.png          fig11_client_training_time.png
    fig06_mcc.png            fig12_bandwidth.png
    fig07_precision.png      fig13_all_metrics.png   (combined overview)

    python -m src.make_figures --config configs/paper.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import Config

LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

SINGLE = [
    ("fig02_accuracy", "accuracy", "Accuracy", "Aggregated Accuracy over Communication Rounds"),
    ("fig03_loss", "loss", "Loss", "Aggregated Loss over Communication Rounds"),
    ("fig04_f1", "f1_score", "F1-score", "Aggregated F1-score over Communication Rounds"),
    ("fig05_kappa", "cohens_kappa", "Cohen's kappa", "Cohen's Kappa over Communication Rounds"),
    ("fig06_mcc", "mcc", "MCC", "Matthews Correlation Coefficient over Communication Rounds"),
    ("fig07_precision", "precision", "Precision", "Aggregated Precision over Communication Rounds"),
    ("fig08_recall", "recall", "Recall", "Aggregated Recall over Communication Rounds"),
    ("fig09_roc_auc", "roc_auc", "ROC-AUC", "Aggregated ROC-AUC over Communication Rounds"),
    ("fig10_specificity", "specificity", "Specificity", "Aggregated Specificity over Communication Rounds"),
]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def curve(rounds, values, ylabel, title, path, baseline=None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rounds, values, marker="o", markersize=3, linewidth=1.6, color="#1f4e79")
    if baseline is not None:
        ax.axhline(baseline, color="#b03a2e", linestyle="--", linewidth=1.2,
                   label=f"majority baseline = {baseline:.3f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    _save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = Config.load(args.config).apply_overrides(args.set)

    res_fp = Path(cfg.out_dir) / f"{cfg.tag}_results.json"
    if not res_fp.exists():
        raise SystemExit(f"{res_fp} not found - run src.run_federated first")
    res = json.load(open(res_fp))

    fig_dir = Path(cfg.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    h = res["history"]
    rounds = h["round"]
    baseline = res["majority_baseline"]

    print("generating figures...")
    for stem, key, ylabel, title in SINGLE:
        bl = baseline if key == "accuracy" else None
        curve(rounds, h[key], ylabel, title, fig_dir / f"{stem}.png", baseline=bl)

    # Figure 11 - per-client training time
    lat = res["operational"]["client_latency_s"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cids = sorted(lat, key=lambda x: int(x))
    means = [lat[c]["mean"] for c in cids]
    mins = [lat[c]["min"] for c in cids]
    maxs = [lat[c]["max"] for c in cids]
    xs = np.arange(len(cids))
    ax.bar(xs, means, color="#2e6da4", width=0.55,
           yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
           capsize=4, error_kw={"linewidth": 1})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Client {c}" for c in cids])
    ax.set_ylabel("Local training time per round (s)")
    ax.set_title("Per-client training time across federated rounds "
                 "(bar = mean, whisker = min/max)", fontsize=10)
    ax.grid(alpha=0.3, axis="y", linestyle=":")
    _save(fig, fig_dir / "fig11_client_training_time.png")

    # Figure 12 - bandwidth per client per round
    up = res["operational"]["client_upload_mb"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, c in enumerate(sorted(up, key=lambda x: int(x))):
        ax.plot(rounds, [up[c]] * len(rounds), linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                linewidth=1.7, label=f"Client {c}")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Model update uploaded (MB)")
    ax.set_title("Bandwidth consumed per client per round", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_ylim(0, max(up.values()) * 1.6)
    _save(fig, fig_dir / "fig12_bandwidth.png")

    # Combined overview
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ax, (stem, key, ylabel, _t) in zip(axes.ravel(), SINGLE):
        ax.plot(rounds, h[key], linewidth=1.5, color="#1f4e79")
        ax.set_title(ylabel, fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
    for ax in axes[-1]:
        ax.set_xlabel("Round")
    fig.suptitle("Federated knowledge tracing - all metrics over communication rounds",
                 fontsize=12)
    _save(fig, fig_dir / "fig13_all_metrics.png")

    print(f"\nall figures written to {fig_dir}/")


if __name__ == "__main__":
    main()
