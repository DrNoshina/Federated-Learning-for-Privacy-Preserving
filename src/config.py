"""
Configuration for the federated knowledge-tracing experiments.

Defaults reproduce Table 1 of the paper ("Federated Simulation Configuration"):
    Number of Clients ............ 5
    Model Type ................... Two-layer LSTM + Dense Classifier
    Sequence Length (L) .......... 20
    Feature Dimensions (d) ....... 10
    LSTM Hidden Units ............ 128
    Number of LSTM Layers ........ 2
    Dropout Rate ................. 0.5
    Local Epochs per Round ....... 20
    Batch Size ................... 32
    Learning Rate ................ 0.0005
    Communication Rounds ......... 100
    Optimizer .................... Adam
    Loss Function ................ Cross-Entropy
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # ---------------- dataset selection ----------------
    # "assistments" -> next-response knowledge tracing (ceiling ~AUC 0.78)
    # "oulad"       -> course-outcome prediction (published ~AUC 0.90+)
    dataset: str = "assistments"
    oulad_dir: str = "~/Downloads/oulad"
    oulad_exclude_withdrawn: bool = False

    # ---------------- data ----------------
    raw_csv: str = "~/Downloads/2012-2013-data-with-predictions-4-final.csv"
    # Cap on total interactions kept, applied by sampling WHOLE learners so that
    # every learner's interaction sequence stays intact and chronologically ordered.
    # None -> use every eligible learner in the file.
    max_interactions: Optional[int] = 300_000
    # A learner must have at least this many logged interactions to be usable.
    min_interactions: int = 30
    # Stride between consecutive sliding windows taken from one learner.
    window_stride: int = 1
    # Hard cap on training windows per client (keeps runtime predictable).
    max_windows_per_client: Optional[int] = 60_000
    # "learner"  -> disjoint learner sets per client (this is what Section 4.1 describes)
    # "school"   -> one client per school_id, if the column exists in your CSV
    partition_by: str = "learner"
    # Fractions of each client's LEARNERS held out. The split is three-way and by
    # learner, not by row, so no learner appears in more than one split.
    # dev  -> selects the communication round and the decision threshold
    # test -> read only to produce the reported numbers
    dev_fraction: float = 0.15
    test_fraction: float = 0.15
    # Smoothing constant for the difficulty priors (target encoding).
    prior_smoothing: float = 15.0
    # Subtract a training row's own outcome from its difficulty prior, so the row
    # cannot partly read its own label off that channel.
    leave_one_out_priors: bool = True
    # Which d=10 channel set to build: "paper" or "refined". Both are d=10, so
    # Table 1 holds either way. See FEATURE_SETS in data.py.
    feature_set: str = "paper"

    # ---------------- model (Table 1) ----------------
    seq_len: int = 20            # L
    n_features: int = 10         # d
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.5

    # ---------------- federated training (Table 1) ----------------
    n_clients: int = 5
    rounds: int = 100
    local_epochs: int = 20
    batch_size: int = 32
    lr: float = 5e-4
    grad_clip: float = 1.0
    # Safety valve: cap the optimiser steps taken inside one local epoch. The paper
    # ran ~30k interactions, so 20 full epochs was cheap. On the full dataset an
    # uncapped run is ~1000x more work. Set to None to disable the cap.
    max_steps_per_epoch: Optional[int] = 200
    eval_every: int = 1

    # ---------------- centralized reference ----------------
    centralized_epochs: int = 20

    # ---------------- decision rule and stabilisation ----------------
    # Where the probability cut-off comes from. "fixed" is 0.5, as originally
    # run; the dev_* policies fit it on the dev split. See metrics.py.
    threshold_policy: str = "fixed"
    # "balanced" reweights the cross-entropy by inverse class frequency.
    class_weight: str = "none"
    # Average the global weights over the last K evaluated rounds (0 disables).
    # Cheap variance reduction; does not change the training procedure.
    swa_last_k: int = 0

    # ---------------- runtime ----------------
    seed: int = 1
    device: str = "cuda"         # "cuda", "cpu", or "auto"
    preload_to_device: bool = True
    num_workers: int = 0

    # ---------------- feature policy ----------------
    # "causal"  -> only information available BEFORE the target attempt is scored.
    #              This is the scientifically valid setting and the project default.
    # "diagnostic_leaky" -> additionally exposes within-attempt outcome proxies of the
    #              TARGET row (bottom_hint, first_action, attempt_count, hint_count).
    #              Provided ONLY to diagnose inflated metrics. Never report these
    #              numbers as predictive performance. See README section "Leakage".
    feature_mode: str = "causal"

    # ---------------- output ----------------
    out_dir: str = "results"
    fig_dir: str = "figures"
    tag: str = "federated"

    def resolved_csv(self) -> Path:
        return Path(self.raw_csv).expanduser().resolve()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in Config.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return Config(**raw)

    def apply_overrides(self, pairs: list[str]) -> "Config":
        """Apply CLI overrides of the form key=value."""
        for pair in pairs:
            if "=" not in pair:
                raise ValueError(f"Override must look like key=value, got {pair!r}")
            key, val = pair.split("=", 1)
            if key not in Config.__dataclass_fields__:
                raise ValueError(f"Unknown config key {key!r}")
            current = getattr(self, key)
            if val.lower() in {"none", "null"}:
                parsed = None
            elif isinstance(current, bool):
                parsed = val.lower() in {"1", "true", "yes"}
            elif isinstance(current, int) and not isinstance(current, bool):
                parsed = int(float(val))
            elif isinstance(current, float):
                parsed = float(val)
            else:
                parsed = val
            setattr(self, key, parsed)
        return self
