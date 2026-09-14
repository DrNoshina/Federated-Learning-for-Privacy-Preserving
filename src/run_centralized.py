"""
Centralized reference model.

Trains the identical architecture on the pooled training windows of all clients
and evaluates on the identical pooled held-out set. This is the "centralised
oracle" the paper compares against (abstract: Delta Accuracy < 1.5 pp).

Run this AFTER run_federated.py so the two share the same data build.

    python -m src.run_centralized --config configs/paper.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import build_all_clients
from .federated import DeviceBatcher, predict_proba
from .metrics import (all_metrics, majority_baseline, format_block, METRIC_ORDER,
                      choose_threshold)
from .model import make_model
from .run_federated import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = Config.load(args.config).apply_overrides(args.set)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("CENTRALIZED REFERENCE MODEL")
    print("=" * 62)

    blobs, n_features = build_all_clients(cfg)
    X_tr = np.concatenate([b["X_tr"] for b in blobs])
    y_tr = np.concatenate([b["y_tr"] for b in blobs])
    X_dev = np.concatenate([b["X_dev"] for b in blobs])
    y_dev = np.concatenate([b["y_dev"] for b in blobs])
    X_te = np.concatenate([b["X_te"] for b in blobs])
    y_te = np.concatenate([b["y_te"] for b in blobs])
    baseline = majority_baseline(y_te)
    print(f"\n[data] pooled training windows : {X_tr.shape}")
    print(f"[data] pooled dev / test windows: {len(y_dev):,} / {len(y_te):,}")
    print(f"[eval] majority-class baseline : {baseline:.4f}\n")

    model = make_model(cfg, n_features, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    batcher = DeviceBatcher(X_tr, y_tr, cfg.batch_size, device,
                            cfg.preload_to_device, seed=cfg.seed)

    best = {"roc_auc": -1.0, "state": None, "epoch": -1}
    t0 = time.perf_counter()
    for ep in range(1, cfg.centralized_epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for xb, yb in batcher.epoch(cfg.max_steps_per_epoch):
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item()
            steps += 1
        p = predict_proba(model, X_dev, device)
        m = all_metrics(y_dev, p)
        print(f"epoch {ep:3d}/{cfg.centralized_epochs}  "
              f"loss {total / max(steps,1):.4f}  acc {m['accuracy']:.4f}  "
              f"auc {m['roc_auc']:.4f}", flush=True)
        if m["roc_auc"] > best["roc_auc"]:
            best = {"roc_auc": m["roc_auc"], "epoch": ep,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}

    print(f"\n[time] centralized training: {(time.perf_counter()-t0)/60:.1f} min\n")
    model.load_state_dict(best["state"])
    p_dev = predict_proba(model, X_dev, device)
    p_te = predict_proba(model, X_te, device)
    thr = choose_threshold(y_dev, p_dev, cfg.threshold_policy)
    cen = all_metrics(y_te, p_te, thr)
    print(f"[select] threshold ({cfg.threshold_policy}) -> {thr:.4f}\n")
    print(format_block(f"CENTRALIZED - TEST learners (best dev epoch {best['epoch']})",
                       cen, baseline))

    # ---- comparison table, if a federated run exists ----
    fed_fp = out_dir / f"{cfg.tag}_results.json"
    if fed_fp.exists():
        fed = json.load(open(fed_fp))["federated_final"]
        print("\nFEDERATED vs CENTRALIZED (same held-out TEST learners)")
        print("-" * 62)
        print(f"  {'metric':<16}{'federated':>12}{'centralized':>14}{'delta (pp)':>13}")
        for k in METRIC_ORDER:
            d = (fed[k] - cen[k]) * 100
            print(f"  {k:<16}{fed[k]:>12.4f}{cen[k]:>14.4f}{d:>12.2f}")
    else:
        fed = None
        print(f"\n[note] {fed_fp} not found - run run_federated.py for the comparison")

    with open(out_dir / f"{cfg.tag}_centralized.json", "w") as fh:
        json.dump({"centralized": cen, "federated": fed,
                   "majority_baseline": baseline, "threshold": thr,
                   "best_epoch": best["epoch"]}, fh, indent=2)
    print(f"\nsaved {out_dir / (cfg.tag + '_centralized.json')}")


if __name__ == "__main__":
    main()
