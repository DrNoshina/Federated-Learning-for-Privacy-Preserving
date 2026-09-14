"""
Flower-native federated runner (Table 1: "Federated Learning Framework — Flower v1.7.0").

Why this file exists
--------------------
`src/run_federated.py` implements FedAvg directly, which keeps the training loop
readable and removes a heavy dependency. This file does the same experiment but
routes aggregation through Flower's own `flwr.server.strategy.FedAvg`, so the
weighted average is computed by library code rather than by ours. Running both
and getting the same numbers is the check that our Eq. 3 implementation is
correct — see `tools/verify_fedavg_equivalence.py`.

On the simulation engine
------------------------
Flower 1.7.0's `start_simulation` requires Ray 2.6.3, which has no wheel for
Python 3.12. Rather than ship a script that only runs on some machines, this
runner drives Flower's real `FedAvg` strategy object directly: clients are
trained in-process, results are wrapped in genuine `flwr.common.FitRes`
messages, and `strategy.aggregate_fit` produces the new global parameters.
The aggregation path is Flower's, unchanged.

If you are on Python 3.10/3.11 and want the Ray-backed `start_simulation` path
as well, `pip install "flwr[simulation]==1.7.0" "ray==2.6.3"` will give it to
you; the client logic here transfers unchanged.

Usage
-----
    python -m src.run_federated_flower --config configs/paper.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .config import Config
from .data import build_all_clients
from .federated import SchoolClient, predict_proba, bandwidth_report
from .metrics import all_metrics, majority_baseline, format_block, METRIC_ORDER
from .model import make_model, model_size_report
from .run_federated import pick_device, environment_report


# --------------------------------------------------------------------------- #
# torch <-> flower parameter plumbing
# --------------------------------------------------------------------------- #
def state_to_ndarrays(state: dict) -> list[np.ndarray]:
    return [v.detach().cpu().numpy() for v in state.values()]


def ndarrays_to_state(ndarrays: list[np.ndarray], reference: dict) -> dict:
    out = OrderedDict()
    for (k, ref), arr in zip(reference.items(), ndarrays):
        out[k] = torch.tensor(arr, dtype=ref.dtype, device=ref.device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = Config.load(args.config).apply_overrides(args.set)
    if not cfg.tag.endswith("_flower"):
        cfg.tag = f"{cfg.tag}_flower"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import flwr
    print("=" * 62)
    print(f"FEDERATED KNOWLEDGE TRACING - Flower {flwr.__version__} FedAvg strategy")
    print("=" * 62)
    for k, v in environment_report(device).items():
        print(f"  {k:<18}{v}")
    print(f"  {'flwr':<18}{flwr.__version__}\n")

    blobs, n_features = build_all_clients(cfg)
    clients = [SchoolClient(b, cfg, n_features, device) for b in blobs]
    val_X = np.concatenate([b["X_dev"] for b in blobs])
    val_y = np.concatenate([b["y_dev"] for b in blobs])
    baseline = majority_baseline(val_y)
    print(f"\n[eval] pooled held-out windows : {len(val_y):,}")
    print(f"[eval] majority-class baseline : {baseline:.4f}\n")

    global_model = make_model(cfg, n_features, device)
    size_info = model_size_report(global_model)
    print(f"[model] {size_info['parameters']:,} parameters, "
          f"{size_info['update_mb_fp32']:.3f} MB per transfer (fp32)\n")

    reference_state = global_model.state_dict()
    parameters = ndarrays_to_parameters(state_to_ndarrays(reference_state))

    # Flower's own FedAvg. min_* are set so every client participates in every
    # round, matching Section 4.8 ("synchronous participation of all clients").
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=len(clients),
        min_available_clients=len(clients),
        inplace=False,
    )

    history = {"round": [], "loss": []}
    for m in METRIC_ORDER:
        history[m] = []
    agg_ms = []
    best = {"roc_auc": -1.0, "state": None, "round": -1}

    t_start = time.perf_counter()
    for rnd in range(1, cfg.rounds + 1):
        global_state = ndarrays_to_state(parameters_to_ndarrays(parameters),
                                         reference_state)

        results, losses = [], []
        for c in clients:
            st, ls = c.local_train(global_state)
            losses.append(ls)
            fit_res = FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=ndarrays_to_parameters(state_to_ndarrays(st)),
                num_examples=c.n,          # n_k, the FedAvg weight
                metrics={"loss": float(ls)},
            )
            results.append((None, fit_res))   # ClientProxy unused by aggregate_fit

        t_agg = time.perf_counter()
        parameters, _ = strategy.aggregate_fit(rnd, results, [])
        agg_ms.append((time.perf_counter() - t_agg) * 1000)

        if rnd == 1 or rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            new_state = ndarrays_to_state(parameters_to_ndarrays(parameters),
                                          reference_state)
            global_model.load_state_dict(new_state)
            p = predict_proba(global_model, val_X, device)
            m = all_metrics(val_y, p)
            history["round"].append(rnd)
            history["loss"].append(float(np.mean(losses)))
            for k in METRIC_ORDER:
                history[k].append(m[k])
            print(f"round {rnd:3d}/{cfg.rounds}  loss {np.mean(losses):.4f}  "
                  f"acc {m['accuracy']:.4f}  auc {m['roc_auc']:.4f}  "
                  f"f1 {m['f1_score']:.4f}  kappa {m['cohens_kappa']:.4f}  "
                  f"mcc {m['mcc']:.4f}", flush=True)
            if m["roc_auc"] > best["roc_auc"]:
                best = {"roc_auc": m["roc_auc"], "round": rnd,
                        "state": copy.deepcopy(new_state)}

    train_min = (time.perf_counter() - t_start) / 60
    print(f"\n[time] federated training: {train_min:.1f} min\n")

    global_model.load_state_dict(best["state"])
    fed = all_metrics(val_y, predict_proba(global_model, val_X, device))
    print(format_block(
        f"FEDERATED GLOBAL MODEL via Flower FedAvg (best round {best['round']})",
        fed, baseline))

    per_client = {}
    print("\nPER-CLIENT HELD-OUT PERFORMANCE")
    print("-" * 56)
    for c in clients:
        mc = all_metrics(c.y_dev, predict_proba(global_model, c.X_dev, device))
        per_client[c.cid] = mc
        print(f"  client {c.cid}: acc {mc['accuracy']:.4f}  auc {mc['roc_auc']:.4f}  "
              f"f1 {mc['f1_score']:.4f}  kappa {mc['cohens_kappa']:.4f}")

    bw = bandwidth_report(size_info, len(clients), cfg.rounds)
    latency = {c.cid: {"mean": float(np.mean(c.train_times)),
                       "min": float(np.min(c.train_times)),
                       "max": float(np.max(c.train_times))} for c in clients}
    print("\nOPERATIONAL PROFILE")
    print("-" * 56)
    print(f"  bandwidth / client / round       {bw['per_client_per_round_mb']:.3f} MB")
    print(f"  server aggregation (Flower)      {np.mean(agg_ms):.1f} ms / round")
    for c in clients:
        print(f"    client {c.cid} latency: mean {latency[c.cid]['mean']:.2f} s/round")

    payload = {
        "config": json.loads(cfg.to_json()),
        "framework": f"flwr {flwr.__version__} (FedAvg strategy)",
        "majority_baseline": baseline,
        "federated_final": fed,
        "best_round": best["round"],
        "per_client": per_client,
        "history": history,
        "operational": {**bw, "aggregation_ms_mean": float(np.mean(agg_ms)),
                        "client_latency_s": latency,
                        "client_upload_mb": {c.cid: float(np.mean(c.upload_mb))
                                             for c in clients},
                        "training_minutes": train_min},
    }
    fp = out_dir / f"{cfg.tag}_results.json"
    with open(fp, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nsaved {fp}")


if __name__ == "__main__":
    main()
