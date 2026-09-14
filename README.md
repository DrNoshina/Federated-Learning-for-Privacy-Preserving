# Federated Learning for Privacy-Preserving Student Performance Prediction

Reference implementation and reproduction package for the manuscript. Everything
reported in the results section is produced by the commands below.

**Task.** From a learner's first 30 weeks of virtual learning environment
activity, predict the final course outcome (Pass/Distinction vs Fail/Withdrawn),
training across five course modules that never exchange raw learner records.

**Dataset.** OULAD (Open University Learning Analytics Dataset) — 32,593
enrolments, 10.6 million VLE interaction records. Freely available at
<https://analyse.kmi.open.ac.uk/open-dataset>.

---

## 1. Results

Evaluated on 4,395 held-out test learners who appear in no other split.

| Metric | Federated | Centralized | Δ (pp) |
|---|---|---|---|
| Accuracy | **0.9215** | 0.9258 | −0.43 |
| ROC-AUC | **0.9720** | 0.9734 | −0.14 |
| F1-score | **0.9186** | 0.9231 | −0.45 |
| Precision | **0.8874** | 0.8912 | −0.38 |
| Recall | **0.9521** | 0.9574 | −0.54 |
| Specificity | **0.8949** | 0.8983 | −0.34 |
| Cohen's κ | **0.8430** | 0.8516 | −0.87 |
| MCC | **0.8449** | 0.8537 | −0.88 |

Majority-class baseline: 0.5349. Every federated-vs-centralized gap is below
1.5 percentage points, so distributing training across five institutional
domains costs almost nothing in predictive quality.

**Operational profile.** 204,034 parameters, 0.816 MB per model transfer,
1.632 MB per client per round, ~0.8 ms server aggregation. Zero raw learner
records leave any client.

---

## 2. Configuration

| Setting | Value |
|---|---|
| Model | Two-layer LSTM (128 hidden) + dense classifier |
| Sequence length L | 30 weeks |
| Feature dimensions d | 10 |
| Dropout | 0.6 |
| Clients | 5 (course modules BBB, FFF, DDD, CCC, EEE) |
| Communication rounds | 100 |
| Local epochs per round | 5 |
| Batch size | 32 |
| Learning rate | 0.0005 |
| Optimizer / loss | Adam / cross-entropy |
| Aggregation | FedAvg, weighted by client sample count |

---

## 3. Reproduce

```bash
# environment
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# data: unzip OULAD so the folder holds studentInfo.csv, studentVle.csv,
# studentAssessment.csv and assessments.csv, then set oulad_dir in
# configs/oulad.yaml to point at it.

# run
python -m src.run_federated   --config configs/oulad.yaml --set rounds=100 local_epochs=5 max_steps_per_epoch=60 dropout=0.6 seq_len=30 tag=L30r100
python -m src.run_centralized --config configs/oulad.yaml --set local_epochs=5 dropout=0.6 seq_len=30 tag=L30r100
python -m src.make_figures    --config configs/oulad.yaml --set tag=L30r100
```

Runtime is roughly five minutes on a single RTX 3080. `--resume` continues from
the last checkpoint if a run is interrupted.

### Outputs

```
results/L30r100_results.json      per-round metrics, final metrics, operational profile
results/L30r100_centralized.json  federated vs centralized comparison
results/L30r100_history.csv       tidy per-round table
figures/fig02..fig13 *.png        every figure in the results section
```

---

## 4. Evaluation protocol

Learners are split **70 / 15 / 15** into train / dev / test **by learner**, not
by row, so no learner appears in more than one split. Feature standardisation
uses training statistics only.

| Split | Sequences | Role |
|---|---|---|
| Train | 20,521 | Model fitting |
| Dev | 4,395 | Communication round and decision threshold |
| Test | 4,395 | Reported metrics only |

The communication round and the probability threshold are both chosen on dev.
Test is read once, to produce the table in Section 1. The reported round is 29,
selected by dev ROC-AUC.

### Causality

Only the first 30 weeks (210 days) of a roughly 36-week course enter the model.
The label is the outcome recorded at course end. Nothing observable after the
observation window closes is used as input, so the prediction could be made in
deployment at week 30 and acted on while the course is still running.

---

## 5. Repository layout

```
src/
  config.py           typed configuration with CLI overrides
  data_oulad.py       OULAD loader: weekly sequences, one client per module
  data.py             dataset dispatcher
  model.py            two-layer LSTM + dense classifier
  federated.py        client local training, FedAvg, latency/bandwidth telemetry
  metrics.py          the eight evaluation metrics and threshold selection
  run_federated.py    federated training entry point
  run_centralized.py  centralized reference and comparison table
  make_figures.py     figure generation
configs/oulad.yaml    the configuration used for the reported run
results/              metrics and per-round history for the reported run
figures/              figures at 300 dpi
```

## 6. Feature construction

Ten channels per weekly step, all derived from information available before the
observation window closes:

| # | Channel | Source |
|---|---|---|
| 1 | Weekly click volume (log) | studentVle |
| 2 | Active days that week (log) | studentVle |
| 3 | Distinct materials accessed (log) | studentVle |
| 4 | Clicks per active day | studentVle |
| 5 | Cumulative clicks to date (log) | studentVle |
| 6 | Weeks since last activity | studentVle |
| 7 | Mean assessment score to date | studentAssessment |
| 8 | Assessments submitted to date (log) | studentAssessment |
| 9 | Week-on-week change in activity | studentVle |
| 10 | Studied credits | studentInfo |

## 7. Reproducibility

The `seed` field fixes learner shuffling, client partitioning, the three-way
split and weight initialisation. Same seed plus the same OULAD release gives the
same numbers. cuDNN LSTM kernels are not bit-wise deterministic across different
GPU models, so expect drift in the third decimal on other hardware; run with
`--set device=cpu` for bit-exact reruns.

## Two-dataset results

| | ASSISTments | OULAD |
|---|---|---|
| Task | Next-response correctness | Final course outcome |
| Accuracy | 0.7473 | 0.9215 |
| ROC-AUC | 0.7722 | 0.9720 |
| F1-score | 0.8298 | 0.9186 |
| Cohen's kappa | 0.3517 | 0.8430 |
| Baseline | 0.6822 | 0.5349 |

Federated vs centralized: every metric within 1.5 pp on both datasets.

Reproduce:

    python -m src.run_federated --config configs/oulad.yaml \
        --set rounds=100 local_epochs=5 max_steps_per_epoch=60 dropout=0.6 seq_len=30 tag=L30r100

    python -m src.run_federated --config configs/refined.yaml \
        --set rounds=60 threshold_policy=dev_accuracy tag=final
