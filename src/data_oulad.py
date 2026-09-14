"""
OULAD loader - Open University Learning Analytics Dataset.

Task
----
Course-outcome prediction: from a learner's first `seq_len` WEEKS of activity,
predict the final course result.

    label 1 = Pass or Distinction
    label 0 = Fail (and Withdrawn, unless oulad_exclude_withdrawn is true)

This is a different task from ASSISTments next-response tracing, and it has a
much higher ceiling: published OULAD results sit around accuracy 0.85-0.93 and
ROC-AUC 0.90+.

Federated split
---------------
One client per course module (AAA, BBB, CCC, ...). Modules are naturally
disjoint institutions-within-an-institution, which matches the paper's
"mutually disjoint learner sets" requirement without any artificial slicing.

Causality
---------
Only the first `seq_len` weeks of the course feed the model, and the label is
the outcome recorded at course end. Nothing from after the observation window
enters the input, so a real deployment could make this prediction at week
`seq_len` and act on it. Set `oulad_exclude_withdrawn: true` if you want the
harder, cleaner Pass-vs-Fail problem without withdrawal-driven inactivity.

Expected files in `oulad_dir`
-----------------------------
    studentInfo.csv  studentVle.csv  studentAssessment.csv  assessments.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log_clicks",              # 1  weekly click volume
    "log_active_days",         # 2  days active that week
    "log_distinct_sites",      # 3  distinct VLE materials touched
    "clicks_per_active_day",   # 4  intensity when present
    "log_cum_clicks",          # 5  cumulative engagement to date
    "weeks_since_active",      # 6  recency / disengagement signal
    "mean_score_so_far",       # 7  assessment performance to date
    "n_assessments_so_far",    # 8  assessments submitted to date
    "click_delta",             # 9  change vs previous week
    "studied_credits",         # 10 course load (static, tiled)
]

POSITIVE = {"Pass", "Distinction"}


def _read(oulad_dir: Path, name: str) -> pd.DataFrame:
    fp = oulad_dir / name
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} not found. Download OULAD from "
            "https://analyse.kmi.open.ac.uk/open-dataset and unzip it so that "
            f"{oulad_dir}/ contains studentInfo.csv, studentVle.csv, "
            "studentAssessment.csv and assessments.csv"
        )
    return pd.read_csv(fp)


def build_all_clients_oulad(cfg, verbose: bool = True):
    oulad_dir = Path(cfg.oulad_dir).expanduser().resolve()
    L = cfg.seq_len

    info = _read(oulad_dir, "studentInfo.csv")
    vle = _read(oulad_dir, "studentVle.csv")
    sa = _read(oulad_dir, "studentAssessment.csv")
    asm = _read(oulad_dir, "assessments.csv")

    if verbose:
        print(f"[oulad] studentInfo {len(info):,} rows | studentVle {len(vle):,} rows")

    if cfg.oulad_exclude_withdrawn:
        info = info[info["final_result"] != "Withdrawn"].copy()
        if verbose:
            print(f"[oulad] excluded Withdrawn -> {len(info):,} enrolments")

    info["label"] = info["final_result"].isin(POSITIVE).astype(np.int64)
    info["key"] = (info["code_module"] + "|" + info["code_presentation"]
                   + "|" + info["id_student"].astype(str))

    # ---- weekly VLE aggregation, first L weeks only ----
    vle = vle[(vle["date"] >= 0) & (vle["date"] < L * 7)].copy()
    vle["week"] = (vle["date"] // 7).astype(int)
    vle["key"] = (vle["code_module"] + "|" + vle["code_presentation"]
                  + "|" + vle["id_student"].astype(str))

    wk = vle.groupby(["key", "week"]).agg(
        clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        sites=("id_site", "nunique"),
    ).reset_index()

    # ---- assessment scores, restricted to the same window ----
    asm_win = asm[["id_assessment", "date"]].copy()
    asm_win["date"] = pd.to_numeric(asm_win["date"], errors="coerce")
    sa = sa.merge(asm_win, on="id_assessment", how="left")
    sa = sa[sa["date_submitted"] < L * 7].copy()
    sa["week"] = (sa["date_submitted"] // 7).clip(lower=0).astype(int)
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce").fillna(0.0)

    # studentAssessment has no module columns, so join through studentInfo
    id2keys = info.groupby("id_student")["key"].apply(list).to_dict()
    rows = []
    for sid, week, score in sa[["id_student", "week", "score"]].itertuples(index=False):
        for k in id2keys.get(sid, ()):
            rows.append((k, week, score))
    sa_wk = (pd.DataFrame(rows, columns=["key", "week", "score"])
             .groupby(["key", "week"])["score"].agg(["mean", "count"]).reset_index()
             if rows else
             pd.DataFrame(columns=["key", "week", "mean", "count"]))

    # ---- pivot to dense (student x week) matrices ----
    keys = info["key"].to_numpy()
    kidx = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    clicks = np.zeros((n, L), np.float32)
    adays = np.zeros((n, L), np.float32)
    sites = np.zeros((n, L), np.float32)
    for k, w, c, a, s in wk[["key", "week", "clicks", "active_days",
                             "sites"]].itertuples(index=False):
        i = kidx.get(k)
        if i is not None and 0 <= w < L:
            clicks[i, w], adays[i, w], sites[i, w] = c, a, s

    sc_mean = np.zeros((n, L), np.float32)
    sc_cnt = np.zeros((n, L), np.float32)
    for k, w, m, c in sa_wk.itertuples(index=False):
        i = kidx.get(k)
        if i is not None and 0 <= int(w) < L:
            sc_mean[i, int(w)], sc_cnt[i, int(w)] = m, c

    # ---- build the 10 channels ----
    cum = np.cumsum(clicks, axis=1)
    cum_sc_cnt = np.cumsum(sc_cnt, axis=1)
    cum_sc_sum = np.cumsum(sc_mean * sc_cnt, axis=1)
    mean_so_far = np.where(cum_sc_cnt > 0, cum_sc_sum / np.maximum(cum_sc_cnt, 1), 0.0)

    weeks_since = np.zeros((n, L), np.float32)
    last = np.full(n, -1.0, np.float32)
    for w in range(L):
        active = clicks[:, w] > 0
        last = np.where(active, float(w), last)
        weeks_since[:, w] = np.where(last >= 0, w - last, float(L))

    delta = np.diff(clicks, axis=1, prepend=clicks[:, :1])
    credits = info["studied_credits"].to_numpy(np.float32)[:, None] / 120.0

    X = np.stack([
        np.log1p(clicks) / 6.0,
        np.log1p(adays) / 2.0,
        np.log1p(sites) / 3.0,
        np.where(adays > 0, clicks / np.maximum(adays, 1), 0.0) / 20.0,
        np.log1p(cum) / 8.0,
        weeks_since / float(L),
        mean_so_far / 100.0,
        np.log1p(cum_sc_cnt) / 3.0,
        np.tanh(delta / 50.0),
        np.repeat(credits, L, axis=1),
    ], axis=2).astype(np.float32)          # (n, L, 10)

    y = info["label"].to_numpy(np.int64)
    module = info["code_module"].to_numpy()

    if verbose:
        print(f"[oulad] built {X.shape[0]:,} learner sequences, "
              f"{L} weeks x {X.shape[2]} features")
        print(f"[oulad] positive (Pass/Distinction) rate: {y.mean():.4f}")

    # ---- one client per module ----
    top = pd.Series(module).value_counts().head(cfg.n_clients).index.tolist()
    if verbose:
        print(f"[oulad] clients = modules {top}")

    rng = np.random.RandomState(cfg.seed)
    clients = []
    for cid, mod in enumerate(top, start=1):
        sel = np.flatnonzero(module == mod)
        if len(sel) < 60:
            continue
        rng.shuffle(sel)
        n_te = max(1, int(cfg.test_fraction * len(sel)))
        n_dv = max(1, int(cfg.dev_fraction * len(sel)))
        te, dv, tr = sel[:n_te], sel[n_te:n_te + n_dv], sel[n_te + n_dv:]

        Xc = X[sel].copy()
        # standardise on this client's TRAINING learners only
        tr_pos = np.arange(len(sel)) >= (n_te + n_dv)
        flat = Xc[tr_pos].reshape(-1, X.shape[2])
        mu, sg = flat.mean(0), flat.std(0)
        sg[sg < 1e-6] = 1.0

        def norm(idx):
            return ((X[idx] - mu) / sg).astype(np.float32)

        clients.append({
            "cid": cid,
            "X_tr": norm(tr), "y_tr": y[tr],
            "X_dev": norm(dv), "y_dev": y[dv],
            "X_te": norm(te), "y_te": y[te],
            "feature_names": FEATURE_NAMES,
            "n_learners": {"train": len(tr), "dev": len(dv), "test": len(te)},
        })
        if verbose:
            print(f"[oulad] client {cid} ({mod}): train {len(tr):,} | "
                  f"dev {len(dv):,} | test {len(te):,} learners "
                  f"(positive rate {y[tr].mean():.3f})")

    if len(clients) < 2:
        raise RuntimeError("fewer than 2 usable OULAD modules were built")
    return clients, X.shape[2]
