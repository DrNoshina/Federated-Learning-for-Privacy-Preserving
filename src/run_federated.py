"""
Federated training driver.

Usage
-----
    python -m src.run_federated --config configs/paper.yaml
    python -m src.run_federated --config configs/paper.yaml --set rounds=10 device=cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import build_all_clients
from .federated import (SchoolClient, fedavg, predict_proba, bandwidth_report,
                        average_states)
from .metrics import (all_metrics, majority_baseline, format_block, METRIC_ORDER,
                      choose_threshold, threshold_sweep, format_sweep)
from .model import make_model, model_size_report


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, falling back to CPU")
        name = "cpu"
    return torch.device(name)


def environment_report(device) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last saved checkpoint if one exists")
    ap.add_argument("--ckpt-every", type=int, default=1,
                    help="write a checkpoint every N rounds (default 1)")
    args = ap.parse_args()

    cfg = Config.load(args.config).apply_overrides(args.set)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("FEDERATED KNOWLEDGE TRACING - FedAvg over school-level domains")
    print("=" * 62)
    env = environment_report(device)
    for k, v in env.items():
        print(f"  {k:<18}{v}")
    print(f"\nconfig:\n{cfg.to_json()}\n")

    if cfg.feature_mode == "diagnostic_leaky":
        print("!" * 62)
        print("feature_mode = diagnostic_leaky")
        print("Target-attempt process variables are exposed to the model.")
        print("These metrics are a LEAKAGE DIAGNOSTIC, not predictive performance.")
        print("Do not report them as results. See README.")
        print("!" * 62 + "\n")

    # ---------------- data ----------------
    t_data = time.time()
    blobs, n_features = build_all_clients(cfg)
    print(f"[data] built in {(time.time() - t_data) / 60:.1f} min\n")

    clients = [SchoolClient(b, cfg, n_features, device) for b in blobs]
    dev_X = np.concatenate([b["X_dev"] for b in blobs])
    dev_y = np.concatenate([b["y_dev"] for b in blobs])
    test_X = np.concatenate([b["X_te"] for b in blobs])
    test_y = np.concatenate([b["y_te"] for b in blobs])
    baseline = majority_baseline(test_y)
    print(f"[eval] pooled dev windows      : {len(dev_y):,}  "
          f"(round + threshold selection)")
    print(f"[eval] pooled test windows     : {len(test_y):,}  (reported)")
    print(f"[eval] majority-class baseline : {baseline:.4f}  "
          f"(the model must beat this)\n")

    # ---------------- model ----------------
    global_model = make_model(cfg, n_features, device)
    size_info = model_size_report(global_model)
    print(f"[model] {size_info['parameters']:,} parameters, "
          f"{size_info['update_mb_fp32']:.3f} MB per transfer (fp32), "
          f"{size_info['update_mb_fp16']:.3f} MB (fp16)\n")

    state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    history = {"round": [], "loss": []}
    for m in METRIC_ORDER:
        history[m] = []
    agg_ms = []
    recent: list = []
    best = {"roc_auc": -1.0, "state": None, "round": -1}

    # ---------------- federated rounds ----------------
    # ---------------- checkpoint restore ----------------
    ckpt_fp = out_dir / f"{cfg.tag}_checkpoint.pt"
    start_round = 1
    if args.resume and ckpt_fp.exists():
        ck = torch.load(ckpt_fp, map_location=device, weights_only=False)
        state = {k: v.to(device) for k, v in ck["state"].items()}
        history = ck["history"]
        best = ck["best"]
        if best.get("state") is not None:
            best["state"] = {k: v.to(device) for k, v in best["state"].items()}
        recent = [{k: v.to(device) for k, v in st.items()} for st in ck["recent"]]
        agg_ms = ck["agg_ms"]
        for c in clients:
            c.train_times = ck["client_times"].get(c.cid, [])
            c.upload_mb = ck["client_upload"].get(c.cid, [])
        start_round = ck["round"] + 1
        print(f"[resume] restored from {ckpt_fp} at round {ck['round']}, "
              f"continuing at round {start_round}\n")
        if start_round > cfg.rounds:
            print("[resume] checkpoint already covers all rounds; "
                  "skipping straight to evaluation\n")
    elif args.resume:
        print(f"[resume] no checkpoint at {ckpt_fp}, starting fresh\n")

    t_start = time.perf_counter()
    for rnd in range(start_round, cfg.rounds + 1):
        states, weights, losses = [], [], []
        for c in clients:
            st, ls = c.local_train(state)
            states.append(st)
            weights.append(c.n)
            losses.append(ls)

        t_agg = time.perf_counter()
        state = fedavg(states, weights)
        agg_ms.append((time.perf_counter() - t_agg) * 1000)
        global_model.load_state_dict(state)

        if rnd == 1 or rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            p = predict_proba(global_model, dev_X, device)
            m = all_metrics(dev_y, p)
            history["round"].append(rnd)
            history["loss"].append(float(np.mean(losses)))
            for k in METRIC_ORDER:
                history[k].append(m[k])
            print(f"round {rnd:3d}/{cfg.rounds}  loss {np.mean(losses):.4f}  "
                  f"acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  "
                  f"f1 {m['f1_score']:.4f}  kappa {m['cohens_kappa']:.4f}  "
                  f"mcc {m['mcc']:.4f}", flush=True)
            recent.append(copy.deepcopy(state))
            if cfg.swa_last_k > 0:
                del recent[:-cfg.swa_last_k]
            else:
                del recent[:-1]
            if m["roc_auc"] > best["roc_auc"]:
                best = {"roc_auc": m["roc_auc"], "round": rnd,
                        "state": copy.deepcopy(state)}

        # Checkpoint AFTER the round is fully done, so a crash mid-round simply
        # replays that round rather than resuming from a half-updated state.
        if rnd % max(args.ckpt_every, 1) == 0 or rnd == cfg.rounds:
            tmp = ckpt_fp.with_suffix(".tmp")
            torch.save({
                "round": rnd,
                "state": {k: v.cpu() for k, v in state.items()},
                "history": history,
                "best": {**best, "state": None if best["state"] is None
                         else {k: v.cpu() for k, v in best["state"].items()}},
                "recent": [{k: v.cpu() for k, v in st.items()} for st in recent],
                "agg_ms": agg_ms,
                "client_times": {c.cid: c.train_times for c in clients},
                "client_upload": {c.cid: c.upload_mb for c in clients},
                "config": json.loads(cfg.to_json()),
            }, tmp)
            tmp.replace(ckpt_fp)   # atomic: never leaves a truncated file

    train_min = (time.perf_counter() - t_start) / 60
    print(f"\n[time] federated training: {train_min:.1f} min "
          f"({cfg.rounds} rounds x {len(clients)} clients)\n")

    # ---------------- final evaluation ----------------
    # Model selection: best dev round, optionally averaged over the last K.
    if cfg.swa_last_k > 0 and len(recent) > 1:
        final_state = average_states(recent)
        selection = f"weight average of last {len(recent)} evaluated rounds"
        global_model.load_state_dict(final_state)
        m_swa = all_metrics(dev_y, predict_proba(global_model, dev_X, device))
        if m_swa["roc_auc"] < best["roc_auc"]:
            final_state, selection = best["state"], f"best dev round {best['round']}"
    else:
        final_state, selection = best["state"], f"best dev round {best['round']}"
    global_model.load_state_dict(final_state)
    print(f"[select] {selection}")

    # Decision threshold: fitted on dev, applied to test.
    p_dev = predict_proba(global_model, dev_X, device)
    p_test = predict_proba(global_model, test_X, device)
    thr = choose_threshold(dev_y, p_dev, cfg.threshold_policy)
    print(f"[select] threshold policy '{cfg.threshold_policy}' -> {thr:.4f}\n")

    sweep = threshold_sweep(dev_y, p_dev, test_y, p_test)
    print(format_sweep(sweep, cfg.threshold_policy))

    fed = all_metrics(test_y, p_test, thr)
    print()
    print(format_block(
        f"FEDERATED GLOBAL MODEL - held-out TEST learners ({selection})",
        fed, baseline))

    print("\nPER-CLIENT TEST PERFORMANCE")
    print("-" * 56)
    per_client = {}
    for c in clients:
        mc = all_metrics(c.y_te, predict_proba(global_model, c.X_te, device), thr)
        per_client[c.cid] = mc
        print(f"  client {c.cid}: acc {mc['accuracy']:.4f}  auc {mc['roc_auc']:.4f}  "
              f"f1 {mc['f1_score']:.4f}  kappa {mc['cohens_kappa']:.4f}")

    # ---------------- operational profile ----------------
    bw = bandwidth_report(size_info, len(clients), cfg.rounds)
    print("\nOPERATIONAL PROFILE")
    print("-" * 56)
    print(f"  model parameters                 {bw['parameters']:,}")
    print(f"  model update size (fp32)         {bw['update_mb_fp32']:.3f} MB")
    print(f"  bandwidth / client / round       {bw['per_client_per_round_mb']:.3f} MB")
    print(f"  bandwidth / round, all clients   {bw['all_clients_per_round_mb']:.3f} MB")
    print(f"  cumulative over {cfg.rounds} rounds        {bw['cumulative_gb']:.3f} GB")
    print(f"  server aggregation               {np.mean(agg_ms):.1f} ms / round")
    print("\n  per-client local training latency (s / round):")
    latency = {}
    for c in clients:
        latency[c.cid] = {
            "mean": float(np.mean(c.train_times)),
            "min": float(np.min(c.train_times)),
            "max": float(np.max(c.train_times)),
        }
        print(f"    client {c.cid}: mean {latency[c.cid]['mean']:.2f}  "
              f"min {latency[c.cid]['min']:.2f}  max {latency[c.cid]['max']:.2f}")
    print("\n  raw learner records transmitted: 0")

    # ---------------- persist ----------------
    payload = {
        "config": json.loads(cfg.to_json()),
        "environment": env,
        "feature_mode": cfg.feature_mode,
        "n_clients_used": len(clients),
        "feature_set": cfg.feature_set,
        "feature_names": blobs[0]["feature_names"],
        "pooled_dev_windows": int(len(dev_y)),
        "pooled_test_windows": int(len(test_y)),
        "threshold": thr,
        "threshold_policy": cfg.threshold_policy,
        "threshold_sweep": sweep,
        "selection": selection,
        "majority_baseline": baseline,
        "federated_final": fed,
        "best_round": best["round"],
        "per_client": per_client,
        "history": history,
        "operational": {
            **bw,
            "aggregation_ms_mean": float(np.mean(agg_ms)),
            "client_latency_s": latency,
            "client_upload_mb": {c.cid: float(np.mean(c.upload_mb)) for c in clients},
            "training_minutes": train_min,
        },
    }
    fp = out_dir / f"{cfg.tag}_results.json"
    with open(fp, "w") as fh:
        json.dump(payload, fh, indent=2)

    # tidy per-round CSV for plotting / supplementary tables
    import csv
    csv_fp = out_dir / f"{cfg.tag}_history.csv"
    with open(csv_fp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "loss"] + METRIC_ORDER)
        for i, r in enumerate(history["round"]):
            w.writerow([r, history["loss"][i]] + [history[k][i] for k in METRIC_ORDER])

    torch.save(final_state, out_dir / f"{cfg.tag}_global_model.pt")
    print(f"\nsaved {fp}")
    print(f"saved {csv_fp}")
    if ckpt_fp.exists():
        print(f"checkpoint kept at {ckpt_fp} "
              f"(delete it to force a fresh run)")
    print(f"saved {out_dir / (cfg.tag + '_global_model.pt')}")


if __name__ == "__main__":
    main()
