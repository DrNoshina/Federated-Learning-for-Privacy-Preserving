"""
Evaluation metrics, paper Section 5.1.

Positive class (y = 1) denotes a CORRECT next response; negative class (y = 0)
denotes an incorrect next response.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix,
)

METRIC_ORDER = [
    "accuracy", "precision", "recall", "specificity",
    "f1_score", "roc_auc", "mcc", "cohens_kappa",
]


def all_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_pos) >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auc = float(roc_auc_score(y_true, p_pos))
    except ValueError:          # single-class slice
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def majority_baseline(y_true: np.ndarray) -> float:
    y = np.asarray(y_true)
    return float(max(y.mean(), 1.0 - y.mean()))


def format_block(name: str, m: dict, baseline: float | None = None) -> str:
    lines = [name, "-" * 56]
    for k in METRIC_ORDER:
        lines.append(f"  {k:<16}{m[k]:>10.4f}")
    lines.append(f"  {'confusion':<16}  TP={m['tp']}  TN={m['tn']}  "
                 f"FP={m['fp']}  FN={m['fn']}")
    if baseline is not None:
        lines.append(f"  {'majority base':<16}{baseline:>10.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# decision-threshold selection
# --------------------------------------------------------------------------- #
# The dataset is roughly 70% positive. At a fixed 0.5 cut-off a well-calibrated
# model predicts "correct" almost everywhere: accuracy looks fine because it
# tracks the majority class, but specificity collapses and kappa/MCC sit near
# zero. Choosing the cut-off on a held-out DEV split fixes that. This is
# ordinary practice for imbalanced binary tasks and changes no part of the
# model, the training procedure or the aggregation - only the point at which a
# probability becomes a decision.
#
# Policies:
#   fixed         0.5, as in the original run
#   dev_youden    maximise recall + specificity - 1 (Youden's J)
#   dev_balanced  maximise (recall + specificity) / 2
#   dev_f1        maximise F1
#   dev_kappa     maximise Cohen's kappa
#   dev_accuracy  maximise plain accuracy (keeps the headline number highest)
THRESHOLD_POLICIES = ["fixed", "dev_youden", "dev_balanced",
                      "dev_f1", "dev_kappa", "dev_accuracy"]


def _policy_score(policy: str, m: dict) -> float:
    if policy == "dev_youden":
        return m["recall"] + m["specificity"] - 1.0
    if policy == "dev_balanced":
        return 0.5 * (m["recall"] + m["specificity"])
    if policy == "dev_f1":
        return m["f1_score"]
    if policy == "dev_kappa":
        return m["cohens_kappa"]
    if policy == "dev_accuracy":
        return m["accuracy"]
    raise ValueError(f"unknown threshold policy {policy!r}")


def choose_threshold(y_dev: np.ndarray, p_dev: np.ndarray,
                     policy: str = "dev_youden", n_grid: int = 49) -> float:
    """Pick a decision threshold on the DEV split only."""
    if policy == "fixed":
        return 0.5
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(f"threshold_policy must be one of {THRESHOLD_POLICIES}")

    qs = np.linspace(0.005, 0.995, n_grid)
    grid = np.unique(np.quantile(np.asarray(p_dev), qs))
    best_thr, best_score = 0.5, -np.inf
    for thr in grid:
        score = _policy_score(policy, all_metrics(y_dev, p_dev, thr))
        if score > best_score:
            best_thr, best_score = float(thr), score
    return best_thr


def threshold_sweep(y_dev, p_dev, y_test, p_test) -> dict:
    """Every policy's dev-chosen threshold and the test metrics it yields.

    Printed for transparency so the reported operating point is an explicit,
    visible choice rather than a silent default.
    """
    out = {}
    for pol in THRESHOLD_POLICIES:
        thr = choose_threshold(y_dev, p_dev, pol)
        out[pol] = {"threshold": thr, "test": all_metrics(y_test, p_test, thr)}
    return out


def format_sweep(sweep: dict, chosen: str) -> str:
    head = (f"  {'policy':<14}{'thr':>7}{'acc':>9}{'f1':>9}{'spec':>9}"
            f"{'kappa':>9}{'mcc':>9}{'auc':>9}")
    lines = ["THRESHOLD POLICIES - cut-off fitted on dev, metrics on test",
             "-" * 76, head, "-" * 76]
    for pol, d in sweep.items():
        m = d["test"]
        mark = " <-- reported" if pol == chosen else ""
        lines.append(f"  {pol:<14}{d['threshold']:>7.3f}{m['accuracy']:>9.4f}"
                     f"{m['f1_score']:>9.4f}{m['specificity']:>9.4f}"
                     f"{m['cohens_kappa']:>9.4f}{m['mcc']:>9.4f}"
                     f"{m['roc_auc']:>9.4f}{mark}")
    lines.append("-" * 76)
    lines.append("  ROC-AUC is threshold-free, so it is identical on every row.")
    return "\n".join(lines)
