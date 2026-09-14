# Results Tables — Federated Student Performance Prediction (OULAD)

Run tag: `L30r100`. All numbers taken from `results/L30r100_results.json` and
`results/L30r100_centralized.json`.

---

## Table 1. Federated Simulation Configuration

| Parameter | Value |
|---|---|
| Federated learning framework | Flower-compatible FedAvg |
| Number of clients | 5 |
| Client partitioning | Course module (BBB, FFF, DDD, CCC, EEE) |
| Dataset | OULAD (Open University Learning Analytics Dataset) |
| Model type | Two-layer LSTM + dense classifier |
| Sequence length (L) | 30 weeks |
| Feature dimensions (d) | 10 |
| LSTM hidden units | 128 |
| Number of LSTM layers | 2 |
| Dropout rate | 0.6 |
| Local epochs per round | 5 |
| Batch size | 32 |
| Learning rate | 0.0005 |
| Communication rounds | 100 |
| Optimizer | Adam |
| Loss function | Cross-entropy |
| Aggregation | FedAvg, weighted by client sample count |

---

## Table 2. Dataset Description

| Property | Value |
|---|---|
| Dataset | OULAD |
| Enrolments | 32,593 |
| VLE interaction records | 10,655,280 |
| Course modules used | 5 |
| Prediction target | Final course result (Pass/Distinction vs Fail/Withdrawn) |
| Positive class rate | 0.4720 |
| Observation window | Weeks 0–29 (first 210 days) |
| Majority-class baseline | 0.5349 |

---

## Table 3. Data Partitioning Across Clients

| Client | Module | Train | Dev | Test | Positive rate (train) |
|---|---|---|---|---|---|
| 1 | BBB | 5,537 | 1,186 | 1,186 | 0.482 |
| 2 | FFF | 5,434 | 1,164 | 1,164 | 0.473 |
| 3 | DDD | 4,392 | 940 | 940 | 0.413 |
| 4 | CCC | 3,104 | 665 | 665 | 0.372 |
| 5 | EEE | 2,054 | 440 | 440 | 0.554 |
| **Total** | | **20,521** | **4,395** | **4,395** | **0.472** |

Learners are split 70/15/15 **by learner**, so no learner appears in more than
one split. Dev selects the communication round and decision threshold; test is
read only to produce reported metrics.

---

## Table 4. Global Model Performance (held-out test learners)

| Metric | Value |
|---|---|
| Accuracy | 0.9215 |
| Precision | 0.8874 |
| Recall | 0.9521 |
| Specificity | 0.8949 |
| F1-score | 0.9186 |
| ROC-AUC | 0.9720 |
| Matthews Correlation Coefficient | 0.8449 |
| Cohen's Kappa | 0.8430 |
| Majority-class baseline | 0.5349 |

Selected at communication round 29 by dev ROC-AUC. Confusion matrix:
TP = 1,946, TN = 2,104, FP = 247, FN = 98.

---

## Table 5. Per-Client Performance

| Client | Module | Accuracy | ROC-AUC | F1-score | Cohen's Kappa |
|---|---|---|---|---|---|
| 1 | BBB | 0.9115 | 0.9579 | 0.9115 | 0.8238 |
| 2 | FFF | 0.9356 | 0.9824 | 0.9350 | 0.8712 |
| 3 | DDD | 0.9181 | 0.9743 | 0.9110 | 0.8353 |
| 4 | CCC | 0.9248 | 0.9811 | 0.9088 | 0.8451 |
| 5 | EEE | 0.9136 | 0.9796 | 0.9215 | 0.8259 |

Performance is consistent across all five institutional domains, with accuracy
spanning 0.9115–0.9356 and ROC-AUC 0.9579–0.9824.

---

## Table 6. Federated vs Centralized Comparison

| Metric | Federated | Centralized | Δ (pp) |
|---|---|---|---|
| Accuracy | 0.9215 | 0.9258 | −0.43 |
| Precision | 0.8874 | 0.8912 | −0.38 |
| Recall | 0.9521 | 0.9574 | −0.54 |
| Specificity | 0.8949 | 0.8983 | −0.34 |
| F1-score | 0.9186 | 0.9231 | −0.45 |
| ROC-AUC | 0.9720 | 0.9734 | −0.14 |
| MCC | 0.8449 | 0.8537 | −0.88 |
| Cohen's Kappa | 0.8430 | 0.8516 | −0.87 |

Both models share the same architecture, hyperparameters and held-out test
learners. Every gap falls below 1.5 percentage points, confirming that
distributing training across five institutional domains costs almost nothing in
predictive quality.

---

## Table 7. Communication and Computation Cost

| Quantity | Value |
|---|---|
| Model parameters | 204,034 |
| Model update size (fp32) | 0.816 MB |
| Bandwidth per client per round | 1.632 MB |
| Bandwidth per round, all clients | 8.161 MB |
| Cumulative over 100 rounds | 0.816 GB |
| Server aggregation time | 0.8 ms / round |
| Raw learner records transmitted | 0 |

---

## Table 8. Per-Client Local Training Latency

| Client | Module | Mean (s) | Min (s) | Max (s) |
|---|---|---|---|---|
| 1 | BBB | 0.58 | 0.45 | 1.65 |
| 2 | FFF | 0.49 | 0.45 | 0.54 |
| 3 | DDD | 0.49 | 0.45 | 0.54 |
| 4 | CCC | 0.49 | 0.45 | 0.53 |
| 5 | EEE | 0.49 | 0.45 | 0.55 |

Measured on a single NVIDIA RTX 3080.

---

## Table 9. Input Feature Set (d = 10)

| # | Channel | Source table |
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

All channels are derived from information available before the observation
window closes at week 30, while the label is recorded at course end.

---

## Figure list

| Figure | File |
|---|---|
| 1 | architecture.png |
| 2 | accuracy_curve.png |
| 3 | aggregated_loss_curve.png |
| 4 | f1_score_curve.png |
| 5 | kappa_curve.png |
| 6 | mcc_curve.png |
| 7 | precision_curve.png |
| 8 | recall_curve.png |
| 9 | roc_auc_curve.png |
| 10 | specificity_curve.png |
| 11 | final_training_time_100_rounds.png |
| 12 | final_bandwidth_usage_100_rounds.png |
