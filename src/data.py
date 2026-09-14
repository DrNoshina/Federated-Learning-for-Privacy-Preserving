"""
Data pipeline for federated knowledge tracing on ASSISTments 2012-2013.

Implements Section 4.5 ("Preprocessing Pipeline") of the paper:
  * preprocessing is executed independently for each client,
  * categorical variables are encoded numerically,
  * invalid / incomplete records are removed,
  * sliding windows of length L are generated per learner,
  * each input tensor gets the binary label of the NEXT response,
  * train/validation splits are created inside each local domain.

The feature block is exactly d = 10 channels per interaction, matching Table 1.
Two channel sets are available (`feature_set` in the config); both are d = 10, so
Table 1 is unchanged either way. See FEATURE_SETS below.

Splitting
---------
Each client's LEARNERS are split three ways: train / dev / test. `dev` is used
for choosing the communication round and the decision threshold; `test` is only
ever read to produce the reported numbers. No learner appears in more than one
split, and difficulty priors plus feature standardisation are fitted on training
learners alone.

IMPORTANT - the causality rule
------------------------------
Every channel describes an interaction that has already been *completed and
scored* before the target attempt begins, or describes a property of the target
question that is known when it is served (its identity and its historical
difficulty). Nothing that is only observable *during* the target attempt is
allowed into the input. See README, section "Why the numbers look the way they do".
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

WANTED_COLS = [
    "user_id", "school_id", "problem_id", "skill_id", "correct",
    "attempt_count", "hint_count", "ms_first_response", "overlap_time",
    "start_time", "end_time", "template_id", "position", "original",
    "tutor_mode", "actions", "first_action", "bottom_hint",
]

REQUIRED = ["user_id", "correct"]

# ----------------------------------------------------------------------------
# The two d=10 channel sets.
#
# "paper"   - the original set. Kept so earlier numbers stay reproducible.
# "refined" - same dimensionality, better-chosen channels. Three weak channels
#             (raw attempt count, dwell time, correct x skill-difficulty) are
#             swapped for three the knowledge-tracing literature consistently
#             finds informative: skill-matched past outcomes, within-window
#             accuracy on the target's own skill, and the target item's
#             historical difficulty. All are known before the target attempt
#             starts, so the causality rule still holds.
# ----------------------------------------------------------------------------
FEATURE_SETS = {
    "paper": [
        "correct",
        "attempt_count",
        "hint_count",
        "log_response_time",
        "log_time_on_task",
        "skill_prior",
        "problem_prior",
        "correct_x_skill",
        "running_accuracy",
        "same_skill_as_target",
    ],
    "refined": [
        "correct",
        "same_skill_as_target",
        "correct_on_same_skill",
        "hint_used",
        "log_response_time",
        "skill_prior",
        "running_accuracy",
        "running_accuracy_same_skill",
        "target_skill_prior",
        "target_problem_prior",
    ],
}

# Only used when feature_mode == "diagnostic_leaky". These four describe the
# TARGET attempt itself and therefore cannot exist at prediction time.
LEAKY_NAMES = ["tgt_attempt_count", "tgt_hint_count", "tgt_bottom_hint", "tgt_first_action"]


# --------------------------------------------------------------------------- #
# loading and cleaning
# --------------------------------------------------------------------------- #
def load_raw(csv_path, verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    header = pd.read_csv(csv_path, nrows=0)
    present = [c for c in WANTED_COLS if c in header.columns]
    missing_required = [c for c in REQUIRED if c not in present]
    if missing_required:
        raise ValueError(
            f"CSV is missing required column(s) {missing_required}. "
            f"Found columns: {list(header.columns)[:40]}"
        )
    df = pd.read_csv(csv_path, usecols=present, low_memory=False)
    if verbose:
        skipped = [c for c in WANTED_COLS if c not in present]
        print(f"[data] loaded {len(df):,} rows x {len(present)} cols "
              f"in {time.time() - t0:.0f}s")
        if skipped:
            print(f"[data] columns not present in this CSV (defaults used): {skipped}")
    return df


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["user_id", "correct"]).copy()
    df = df[df["correct"].isin([0, 1, 0.0, 1.0])]
    df["correct"] = df["correct"].astype(np.int8)

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df = df.sort_values(["user_id", "start_time"], kind="mergesort")
    else:
        df = df.sort_values(["user_id"], kind="mergesort")
    df = df.reset_index(drop=True)

    numeric_spec = [
        ("attempt_count", 1.0, 0, 20),
        ("hint_count", 0.0, 0, 10),
        ("ms_first_response", 0.0, 0, 600_000),
        ("overlap_time", 0.0, 0, 600_000),
        ("position", 0.0, 0, 200),
        ("original", 1.0, 0, 1),
        ("tutor_mode", 0.0, 0, 1),
        ("actions", 1.0, 0, 200),
        ("first_action", 0.0, 0, 2),
        ("bottom_hint", 0.0, 0, 1),
    ]
    for c, fill, lo, hi in numeric_spec:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill).clip(lo, hi)
        else:
            df[c] = fill

    for c in ("problem_id", "skill_id", "template_id"):
        if c in df.columns:
            df[c] = df[c].fillna(-1)
        else:
            df[c] = -1

    if verbose:
        print(f"[data] retained {len(df):,}/{n0:,} rows "
              f"({len(df)/max(n0,1):.1%}) after cleaning")
        print(f"[data] learners: {df['user_id'].nunique():,}   "
              f"positive rate: {df['correct'].mean():.4f}")
    return df


def subsample_learners(df: pd.DataFrame, cfg, verbose: bool = True) -> pd.DataFrame:
    """Keep eligible learners, then optionally cap total interactions.

    The cap drops WHOLE learners, never individual rows, so every retained
    sequence stays complete and in chronological order.
    """
    sizes = df.groupby("user_id").size()
    eligible = np.array(sizes[sizes >= cfg.min_interactions].index.to_numpy(), copy=True)
    if len(eligible) == 0:
        raise ValueError(
            f"No learner has >= min_interactions={cfg.min_interactions} rows. "
            f"Longest sequence in this file is {int(sizes.max())} rows."
        )
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(eligible)

    if cfg.max_interactions is not None:
        sizes_e = sizes.loc[eligible].to_numpy()
        keep_n = int(np.searchsorted(np.cumsum(sizes_e), cfg.max_interactions) + 1)
        keep_n = max(min(keep_n, len(eligible)), cfg.n_clients * 30)
        keep_n = min(keep_n, len(eligible))
        eligible = eligible[:keep_n]

    out = df[df["user_id"].isin(set(eligible))].reset_index(drop=True)
    if verbose:
        print(f"[data] eligible learners kept: {out['user_id'].nunique():,}  "
              f"interactions: {len(out):,}")
    return out


def partition(df: pd.DataFrame, cfg, verbose: bool = True) -> list[pd.DataFrame]:
    """Split into n_clients local data domains (Section 4.1: disjoint learners)."""
    if cfg.partition_by == "school" and "school_id" in df.columns \
            and df["school_id"].notna().any():
        top = df["school_id"].value_counts().head(cfg.n_clients).index.tolist()
        parts = [df[df["school_id"] == s].reset_index(drop=True) for s in top]
        if verbose:
            print(f"[data] partitioned by school_id: {[int(s) for s in top]}")
        return parts

    users = np.array(df["user_id"].unique(), copy=True)
    rng = np.random.RandomState(cfg.seed + 7)
    rng.shuffle(users)
    chunks = np.array_split(users, cfg.n_clients)
    parts = [df[df["user_id"].isin(set(ch))].reset_index(drop=True) for ch in chunks]
    if verbose:
        print(f"[data] partitioned by learner into {cfg.n_clients} disjoint domains: "
              f"{[p['user_id'].nunique() for p in parts]} learners each")
    return parts


# --------------------------------------------------------------------------- #
# difficulty priors
# --------------------------------------------------------------------------- #
def _priors(train_df: pd.DataFrame, col: str, k: float, gmean: float):
    """Smoothed target encoding fitted on this client's TRAINING learners only."""
    g = train_df.groupby(col)["correct"].agg(["sum", "count"])
    return (g["sum"] + k * gmean) / (g["count"] + k), g


def _apply_prior(sd, col, prior, counts, gmean, k, is_train_row, leave_one_out):
    """Map the prior onto every row.

    For TRAINING rows the row's own outcome is subtracted out (leave-one-out
    encoding). Without this a training row can partly read its own label off the
    difficulty channel, so the model over-trusts that channel and generalises
    worse to held-out learners.
    """
    vals = sd[col].map(prior).fillna(gmean).astype(np.float64).to_numpy()
    if not leave_one_out:
        return vals.astype(np.float32)

    cnt = sd[col].map(counts["count"]).fillna(0.0).to_numpy()
    own = sd["correct"].to_numpy(np.float64)
    num = vals * (cnt + k) - own          # prior = (sum + k*g) / (count + k)
    den = np.maximum(cnt - 1 + k, 1e-6)
    loo = np.where(is_train_row & (cnt > 1), num / den, vals)
    return np.clip(loo, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# window construction (runs independently inside each client)
# --------------------------------------------------------------------------- #
def build_client(local_df: pd.DataFrame, cfg, client_seed: int, verbose: bool = True):
    """Return train/dev/test window tensors for one federated client."""
    L = cfg.seq_len
    users = np.array(local_df["user_id"].unique(), copy=True)
    if len(users) < 12:
        return None

    rng = np.random.RandomState(client_seed)
    perm = rng.permutation(len(users))
    n_test = max(1, int(cfg.test_fraction * len(users)))
    n_dev = max(1, int(cfg.dev_fraction * len(users)))
    n_train = len(users) - n_dev - n_test
    if n_train < 4:
        return None

    train_users = set(users[perm[:n_train]])
    dev_users = set(users[perm[n_train:n_train + n_dev]])
    test_users = set(users[perm[n_train + n_dev:]])

    sd = local_df.copy()
    is_train_row = sd["user_id"].isin(train_users).to_numpy()
    tr = sd[is_train_row]
    gmean = float(tr["correct"].mean()) if len(tr) else float(sd["correct"].mean())
    k = cfg.prior_smoothing

    for tgt, col in [("skill_prior", "skill_id"), ("problem_prior", "problem_id")]:
        prior, counts = _priors(tr, col, k, gmean)
        sd[tgt] = _apply_prior(sd, col, prior, counts, gmean, k,
                               is_train_row, cfg.leave_one_out_priors)

    names = FEATURE_SETS[cfg.feature_set]
    refined = cfg.feature_set == "refined"
    leaky = cfg.feature_mode == "diagnostic_leaky"

    Xs, ys, us = [], [], []
    for uid, ud in sd.groupby("user_id", sort=False):
        n = len(ud)
        if n < L + 1:
            continue

        def g(c):
            return ud[c].to_numpy(np.float32)

        corr = g("correct")
        att_n = np.minimum(g("attempt_count"), 10.0) / 10.0
        hint_raw = g("hint_count")
        hint_n = np.minimum(hint_raw, 5.0) / 5.0
        hint_used = (hint_raw > 0).astype(np.float32)
        lrt = np.log1p(g("ms_first_response") / 1000.0) / 6.0
        ltot = np.log1p(g("overlap_time") / 1000.0) / 6.0
        sp = g("skill_prior")
        pp = g("problem_prior")
        skl = ud["skill_id"].to_numpy()

        cum = np.concatenate([[0.0], np.cumsum(corr)])
        idx = np.arange(n, dtype=np.float32)
        run = np.where(idx > 0, cum[:-1] / np.maximum(idx, 1.0), gmean).astype(np.float32)

        if leaky:
            tgt_att, tgt_hint = g("attempt_count"), g("hint_count")
            tgt_bh, tgt_fa = g("bottom_hint"), g("first_action")

        for t in range(L, n, cfg.window_stride):
            past = slice(t - L, t)
            same = (skl[past] == skl[t]).astype(np.float32)

            if refined:
                c_same = corr[past] * same
                cnt_before = np.cumsum(same) - same
                sum_before = np.cumsum(c_same) - c_same
                run_same = np.where(cnt_before > 0,
                                    sum_before / np.maximum(cnt_before, 1.0),
                                    gmean).astype(np.float32)
                block = np.column_stack([
                    corr[past],
                    same,
                    c_same,
                    hint_used[past],
                    lrt[past],
                    sp[past],
                    run[past],
                    run_same,
                    np.full(L, sp[t], np.float32),   # target skill difficulty
                    np.full(L, pp[t], np.float32),   # target item difficulty
                ]).astype(np.float32)
            else:
                block = np.column_stack([
                    corr[past], att_n[past], hint_n[past], lrt[past], ltot[past],
                    sp[past], pp[past], corr[past] * sp[past], run[past], same,
                ]).astype(np.float32)

            if leaky:
                lk = np.tile(np.array([tgt_att[t] / 10.0, tgt_hint[t] / 5.0,
                                       tgt_bh[t], float(tgt_fa[t] != 0)], np.float32),
                             (L, 1))
                block = np.concatenate([block, lk], axis=1)

            Xs.append(block)
            ys.append(int(corr[t]))
            us.append(uid)

    if not Xs:
        return None

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    u = np.asarray(us)

    m_tr = np.isin(u, list(train_users))
    m_dev = np.isin(u, list(dev_users))
    m_te = np.isin(u, list(test_users))

    if cfg.max_windows_per_client is not None:
        tr_idx = np.flatnonzero(m_tr)
        if len(tr_idx) > cfg.max_windows_per_client:
            drop = rng.choice(tr_idx, len(tr_idx) - cfg.max_windows_per_client,
                              replace=False)
            keep = np.ones(len(X), bool)
            keep[drop] = False
            X, y = X[keep], y[keep]
            m_tr, m_dev, m_te = m_tr[keep], m_dev[keep], m_te[keep]

    flat = X[m_tr].reshape(-1, X.shape[-1])
    mu, sg = flat.mean(0), flat.std(0)
    sg[sg < 1e-6] = 1.0
    X = (X - mu) / sg

    if not np.isfinite(X).all():
        raise RuntimeError("non-finite values in feature tensor")
    if m_dev.sum() == 0 or m_te.sum() == 0:
        return None

    return {
        "X_tr": X[m_tr], "y_tr": y[m_tr],
        "X_dev": X[m_dev], "y_dev": y[m_dev],
        "X_te": X[m_te], "y_te": y[m_te],
        "feature_names": names + (LEAKY_NAMES if leaky else []),
        "n_learners": {"train": len(train_users), "dev": len(dev_users),
                       "test": len(test_users)},
    }


def build_all_clients(cfg, verbose: bool = True):
    """Dispatch to the right dataset loader."""
    if getattr(cfg, "dataset", "assistments") == "oulad":
        from .data_oulad import build_all_clients_oulad
        return build_all_clients_oulad(cfg, verbose)

    if cfg.feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")

    df = load_raw(cfg.resolved_csv(), verbose)
    df = clean(df, verbose)
    df = subsample_learners(df, cfg, verbose)
    parts = partition(df, cfg, verbose)

    clients = []
    for cid, part in enumerate(parts, start=1):
        out = build_client(part, cfg, client_seed=1234 + cid, verbose=verbose)
        if out is None:
            print(f"[data] client {cid}: not enough usable data, skipped")
            continue
        out["cid"] = cid
        clients.append(out)
        if verbose:
            print(f"[data] client {cid}: train {len(out['y_tr']):,} | "
                  f"dev {len(out['y_dev']):,} | test {len(out['y_te']):,} windows "
                  f"(positive rate {out['y_tr'].mean():.3f})")

    if len(clients) < 2:
        raise RuntimeError("fewer than 2 usable clients were built; "
                           "lower min_interactions or raise max_interactions")

    n_feat = clients[0]["X_tr"].shape[2]
    if cfg.feature_mode == "causal" and n_feat != cfg.n_features:
        raise RuntimeError(f"expected d={cfg.n_features} features, built {n_feat}")
    if verbose:
        print(f"[data] feature set '{cfg.feature_set}', "
              f"sequence length L={cfg.seq_len}, feature dimension d={n_feat}")
    return clients, n_feat
