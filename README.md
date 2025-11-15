
---

# README — BTC 4h Directional Timing Strategy (2020–2025)

This project implements a full end-to-end **BTC 4-hour directional timing strategy**, predicting:

> **P(future 7-day return > +3%) and P(future 7-day return < –3%)**

The pipeline strictly enforces **time-order integrity**, **non-overlapping forward validation**, and **walk-forward out-of-sample trading simulation**.
It follows the required structure:

**Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Step 8**

All steps run **locally**, no command-line arguments needed.

---


# Step 2 + Step 3

### Label construction & leakage-free factor preprocessing

**File:** `src/labels_and_preprocess.py`

### Run:

```bash
python src/labels_and_preprocess.py
```

### Output:

```
data/prepared/btc_4h_with_labels.csv
data/prepared/btc_4h_preprocessed.csv
```

These files include:

* y_up / y_dn labels
* future_close & r_future_7d
* standardized model features
* session/day-of-week indicators
* lagged sentiment factors (no leakage)

---

# Step 4

### Rolling expanding splits (time-ordered)

**File:** `src/step4_splits.py`

### Run:

```bash
python src/step4_splits.py
```

### Output:

```
data/splits/rolling_folds.csv
```

Each fold =

* expanding training window
* next 4 weeks = validation window
* final untouched 2025-08 → 2025-11 = holdout

---

# Step 5

### Train 3 base models per fold & store validation probabilities

**Models:**

* Logistic Regression
* Linear SVC (Calibrated)
* Gradient Boosting Classifier

**File:** `src/step5_models.py`

### Run:

```bash
python src/step5_models.py
```

### Output:

```
data/predictions/val_all_folds.csv
```

Contains columns:

```
p_up_logreg   p_dn_logreg
p_up_linsvc   p_dn_linsvc
p_up_gb       p_dn_gb
fold_id       time
```

---

# Step 6

### Threshold grid search (validation only)

**File:** `src/step6_thresholds.py`

Evaluates:

* tau_up (long threshold)
* tau_dn (short threshold)
* margin (conflict-resolution gap)
* skip-period after entry (avoid overlap)
* geometric mean equity
* Sharpe (median across folds)

### Run:

```bash
python src/step6_thresholds.py
```

### Output:

```
data/thresholds/grid_scores_<model>.csv
data/thresholds/best_thresholds_per_model.csv
```

Example best threshold:

```
gb, tau_up=0.50, tau_dn=0.65, margin=0.02
```

---

# Step 7

### Full walk-forward test + holdout backtest

**File:** `src/step7_backtest.py`

Performs:

* Window-by-window walk-forward probability generation
* 7-day non-overlapping trading simulation
* Absolute threshold or Quantile fallback ("self-rescue")
* Produces trades + equity curves

### Run:

```bash
python src/step7_backtest.py
```

### Output:

```
data/signals/test_trades.csv
data/equity/test_equity.csv

data/signals/holdout_trades.csv
data/equity/holdout_equity.csv
```

---

# Step 8

### Strategy performance summary + Buy&Hold (1x & 3x) comparison

**File:** `src/step8_report.py`

Computes:

* CAGR
* Sharpe
* Max Drawdown
* Volatility
* Trades & win rate (strategy only)

### Run:

```bash
python src/step8_report.py
```

### Output:

```
reports/step8/summary_test.csv
reports/step8/summary_holdout.csv
figs/test_equity.png
figs/holdout_equity.png
```

Example interpretation:

* Buy&Hold 3x: extremely volatile baseline
* Strategy: fewer trades but possibly lower drawdown
* Useful in portfolio mixture vs. pure directional exposure

---

# Expected Outputs Overview

| Step     | Key Output                         | Description                            |
| -------- | ---------------------------------- | -------------------------------------- |
| Step 2–3 | `btc_4h_preprocessed.csv`          | Clean + standardized features & labels |
| Step 4   | `rolling_folds.csv`                | Expanding time-based splits            |
| Step 5   | `val_all_folds.csv`                | Out-of-fold probabilities              |
| Step 6   | `best_thresholds_per_model.csv`    | Optimal tau_up / tau_dn / margin       |
| Step 7   | Trades & equity files              | True walk-forward backtest             |
| Step 8   | Final performance reports + charts | Strategy vs. Buy&Hold (1x & 3x)        |

---

# Troubleshooting

**1. Step 7 runs very slow**
Walk-forward is expensive because each timestamp retrains 3 models.
Machine may need 5–20 minutes depending on CPU.

**2. Zero trades in test or holdout**
Often caused by:

* thresholds too strict
* probabilities too conservative
* market regime shift

Step 7 contains a **quantile self-rescue mode** to guarantee non-zero trading.

**3. Strategy equity curve looks “blocky”**
This is expected — the model makes **non-overlapping 7-day trades**, so equity updates at coarse intervals.

---

# requirements.txt

Below is the recommended environment file:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
matplotlib==3.8.3
tqdm==4.66.2
python-dateutil==2.9.0
```

