"""
FedAvg client and server (paper Sections 4.3 and 4.8).

Global update rule after round t (Eq. 3):

    theta^(t+1) = sum_k (n_k / n) * theta_k^(t+1)

Each client performs `local_epochs` passes over its own training windows with
Adam and cross-entropy loss, then returns its weights. Nothing else is
exchanged: no raw rows, no gradients, no learner identifiers.

The engine also records the two operational quantities the paper reports:
per-client local training latency (Section 5.11) and per-client per-round
update size in MB (Section 5.12).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .model import make_model, model_size_report


class DeviceBatcher:
    """Minimal in-memory batcher. Faster than DataLoader when the client's
    windows already sit on the GPU, and it removes worker/pinning variability
    from the latency measurements."""

    def __init__(self, X, y, batch_size, device, preload, seed=0):
        self.device = device
        self.batch_size = batch_size
        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        if preload:
            Xt = Xt.to(device, non_blocking=True)
            yt = yt.to(device, non_blocking=True)
        self.X, self.y = Xt, yt
        self.n = len(y)
        self.preload = preload
        self.g = torch.Generator().manual_seed(seed)

    def epoch(self, max_steps=None):
        order = torch.randperm(self.n, generator=self.g)
        n_steps = self.n // self.batch_size
        if n_steps == 0:
            n_steps = 1
        if max_steps is not None:
            n_steps = min(n_steps, max_steps)
        for s in range(n_steps):
            idx = order[s * self.batch_size:(s + 1) * self.batch_size]
            xb, yb = self.X[idx], self.y[idx]
            if not self.preload:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
            yield xb, yb


class SchoolClient:
    """One school-level privacy domain."""

    def __init__(self, blob, cfg, n_features, device):
        self.cid = blob["cid"]
        self.cfg = cfg
        self.device = device
        self.n = len(blob["y_tr"])          # n_k, the FedAvg weight
        self.batcher = DeviceBatcher(blob["X_tr"], blob["y_tr"], cfg.batch_size,
                                     device, cfg.preload_to_device, seed=cfg.seed + self.cid)
        self.X_dev, self.y_dev = blob["X_dev"], blob["y_dev"]
        self.X_te, self.y_te = blob["X_te"], blob["y_te"]
        self.model = make_model(cfg, n_features, device)

        # Optional inverse-frequency class weighting. ASSISTments is ~70%
        # positive; weighting pushes the model to attend to incorrect responses
        # instead of riding the majority class.
        if cfg.class_weight == "balanced":
            y = blob["y_tr"]
            counts = np.bincount(y, minlength=2).astype(np.float64)
            w = len(y) / (2.0 * np.maximum(counts, 1.0))
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(w, dtype=torch.float32, device=device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.train_times = []               # seconds per round
        self.upload_mb = []                 # MB per round

    def local_train(self, global_state):
        """Run local_epochs of Adam on local data. Returns (state_dict, mean_loss)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        self.model.load_state_dict(global_state)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model.train()

        total, steps = 0.0, 0
        for _ in range(self.cfg.local_epochs):
            for xb, yb in self.batcher.epoch(self.cfg.max_steps_per_epoch):
                loss = self.criterion(self.model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                total += loss.item()
                steps += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.train_times.append(time.perf_counter() - t0)

        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.upload_mb.append(
            sum(v.numel() * v.element_size() for v in state.values()) / 1e6
        )
        return state, (total / max(steps, 1))


def fedavg(states: list[dict], weights: list[int]) -> dict:
    """Weighted parameter average (Eq. 3), weights are client sample counts."""
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        ref = states[0][key]
        if ref.dtype.is_floating_point:
            acc = torch.zeros_like(ref)
            for st, w in zip(states, weights):
                acc += st[key] * (w / total)
            out[key] = acc
        else:
            out[key] = ref.clone()
    return out


@torch.no_grad()
def predict_proba(model, X, device, batch_size=4096) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        probs.append(torch.softmax(model(xb), dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs) if probs else np.array([])


def bandwidth_report(size_info: dict, n_clients: int, rounds: int) -> dict:
    """Per-round communication cost. One round costs each client a download of
    the global model plus an upload of its update."""
    up = size_info["update_mb_fp32"]
    return {
        "parameters": size_info["parameters"],
        "update_mb_fp32": up,
        "update_mb_fp16": size_info["update_mb_fp16"],
        "per_client_per_round_mb": 2 * up,
        "all_clients_per_round_mb": 2 * up * n_clients,
        "cumulative_gb": 2 * up * n_clients * rounds / 1000.0,
    }


def average_states(states: list[dict]) -> dict:
    """Unweighted mean of several global states (weight averaging / SWA).

    Averaging the last few global models is a standard variance-reduction step.
    It touches neither the local training procedure nor FedAvg itself; it only
    picks which point in the trajectory gets evaluated.
    """
    if len(states) == 1:
        return {k: v.clone() for k, v in states[0].items()}
    out = {}
    for key in states[0]:
        ref = states[0][key]
        if ref.dtype.is_floating_point:
            acc = torch.zeros_like(ref)
            for st in states:
                acc += st[key] / len(states)
            out[key] = acc
        else:
            out[key] = ref.clone()
    return out
