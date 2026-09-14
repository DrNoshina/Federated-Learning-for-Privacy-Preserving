"""
Prove that our Eq. 3 implementation equals Flower's FedAvg.

Reviewers reasonably ask why the main training loop does not call Flower, given
that Table 1 names Flower v1.7.0. This script answers that: it feeds identical
client updates through both aggregators and reports the maximum absolute
difference across every parameter tensor.

    python tools/verify_fedavg_equivalence.py

Expected output: a max difference at float32 round-off, ~1e-8 or smaller.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated import fedavg
from src.model import KTLSTM


def main():
    torch.manual_seed(0)
    n_clients = 5
    # Deliberately uneven client sizes: FedAvg weights by n_k, so an unweighted
    # mean would disagree here. This is the case that actually discriminates.
    weights = [11_284, 3_907, 22_650, 8_133, 1_402]

    states = []
    for _ in range(n_clients):
        m = KTLSTM(n_features=10, hidden=128, layers=2, dropout=0.5)
        for p in m.parameters():
            with torch.no_grad():
                p.add_(torch.randn_like(p) * 0.05)
        states.append({k: v.detach().clone() for k, v in m.state_dict().items()})

    # --- ours (src/federated.py, Eq. 3) ---
    ours = fedavg(states, weights)

    # --- Flower's own strategy ---
    from flwr.common import FitRes, Status, Code
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    strategy = FedAvg(fraction_fit=1.0, min_fit_clients=n_clients,
                      min_available_clients=n_clients, inplace=False)
    results = []
    for st, w in zip(states, weights):
        results.append((None, FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(
                [v.detach().cpu().numpy() for v in st.values()]),
            num_examples=w,
            metrics={},
        )))
    params, _ = strategy.aggregate_fit(1, results, [])
    theirs = parameters_to_ndarrays(params)

    # --- compare ---
    print(f"{'tensor':<28}{'shape':<20}{'max |diff|':>14}")
    print("-" * 62)
    worst = 0.0
    for (name, our_t), their_a in zip(ours.items(), theirs):
        d = float(np.max(np.abs(our_t.detach().cpu().numpy() - their_a)))
        worst = max(worst, d)
        print(f"{name:<28}{str(tuple(our_t.shape)):<20}{d:>14.3e}")

    print("-" * 62)
    print(f"{'worst across all tensors':<48}{worst:>14.3e}")
    tol = 1e-6
    if worst < tol:
        print(f"\nPASS - the two aggregators agree to within {tol:g}.")
        print("src/federated.py::fedavg is equivalent to flwr FedAvg.")
        return 0
    print(f"\nFAIL - difference exceeds {tol:g}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
