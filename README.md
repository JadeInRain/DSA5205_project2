# # BTC 4-Hour Directional Trading Strategy
## RF & XGB Probability-Calibrated Walk-Forward Strategy

A leakage-safe, walk-forward machine learning pipeline for 3-day directional forecasting on Bitcoin using 4-hour bars.

---

## 📋 Project Overview

This strategy implements a **dual-model probability framework** for predicting 3-day directional movements in BTC/USDT:
- **UP**: r_{t→t+3d} > 1% (18 bars @ 4h)
- **DN**: r_{t→t+3d} < -1%

### Key Features
- ✅ **Strict time-series discipline** - no data leakage
- ✅ **Rolling walk-forward validation** - 28 folds (365-day train + 90-day val)
- ✅ **Platt-calibrated probabilities** - stable across market regimes
- ✅ **Absolute threshold strategy** - τ_up=0.30, τ_dn=0.25, margin=0.08
- ✅ **Full execution simulation** - 3× leverage, 0.1% round-trip costs, mark-to-market

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or download this repository
cd btc-trading-strategy

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### 2. Data Preparation

Place your data file in the root directory:
```
BTCUSDT_4h_klines_with_factors.csv
```

**Required columns:**
- Timestamp column (e.g., `open_time`, `timestamp`)
- OHLCV: `open`, `high`, `low`, `close`, `volume`
- 83 engineered features (momentum, volatility, volume, technical indicators, etc.)

### 3. Run the Complete Pipeline
```bash
python main_pipeline.py
```

**Execution time:** ~0.9 minutes (fast mode with pre-trained models)

---

## 📂 Output Structure

After running, you'll find:
```
.
├── data/
│   ├── labels/
│   │   └── btc_4h_with_labels.csv              # Labels with UP/DN targets
│   ├── prepared/
│   │   └── btc_4h_preprocessed.csv             # Standardized features
│   ├── splits/
│   │   ├── rolling_folds.csv                   # Train/val fold definitions
│   │   └── holdout.csv                         # Holdout period definition
│   ├── predictions/
│   │   └── val_all_folds.csv                   # Calibrated probabilities
│   ├── thresholds/
│   │   └── best_thresholds_per_model.csv       # Optimal thresholds
│   ├── signals/
│   │   ├── test_trades.csv                     # Test period trades
│   │   └── holdout_trades.csv                  # Holdout period trades
│   └── equity/
│       ├── test_equity.csv                     # Test equity curve
│       └── holdout_equity.csv                  # Holdout equity curve
│
├── reports/step8/
│   ├── summary_test.csv                        # Test metrics
│   └── summary_holdout.csv                     # Holdout metrics
│
└── figs/
    ├── test_equity.png                         # Test equity plot
    └── holdout_equity.png                      # Holdout equity plot
```

---

## 📊 Results Summary

### Test Period (2025-07-18 to 2025-07-31)

| Metric         | Strategy | B&H 1× | B&H 3× |
|----------------|----------|--------|--------|
| Final Equity   | 0.68     | 0.91   | 0.72   |
| Sharpe         | -5.04    | -2.82  | -4.28  |
| Max Drawdown   | -43.2%   | -13.9% | -35.1% |
| Trades / Win%  | 8 / 37.5%| -      | -      |

❌ **Test period failure** - Strategy underperformed during July correction with whipsaw losses

### Holdout Period (2025-09-01 to 2025-11-05)

| Metric         | Strategy    | B&H 1× | B&H 3× |
|----------------|-------------|--------|--------|
| Final Equity   | **1.04**    | 0.95   | 0.79   |
| CAGR           | **+24.5%**  | -8.0%  | -32.4% |
| Sharpe         | **1.08**    | -0.18  | -0.86  |
| Max Drawdown   | -32.1%      | -8.8%  | -24.3% |
| Trades / Win%  | 11 / 45.5%  | -      | -      |

✅ **Holdout validation success**:
- +32% outperformance vs 3× B&H
- Sharpe 1.08 with only 11 trades (2.8% activity)
- Neutral positioning preserved capital during uncertainty

---

## 🔧 Configuration

Key parameters in `Config` class (lines 44-99):
```python
# Time horizon
HORIZON_DAYS = 3                    # 3-day forecast
HORIZON_BARS = 18                   # 18 × 4h bars

# Label thresholds
UP_THRESH = 0.01                    # 1% for UP
DOWN_THRESH = -0.01                 # -1% for DN

# Walk-forward settings
TRAIN_DAYS = 365                    # 365-day training window
VAL_DAYS = 90                       # 90-day validation window
STEP_DAYS = 90                      # 90-day step forward

# Trading execution
LEVERAGE = 3.0                      # 3× leverage
FEE_PER_SIDE = 0.0005              # 0.05% per side (0.1% round-trip)

# Model thresholds (grid search result)
TAU_UP = 0.30                       # UP probability threshold
TAU_DN = 0.25                       # DN probability threshold
MARGIN = 0.08                       # Conflict resolution margin

# Fast mode (recommended)
FAST_MODE = True                    # Use last fold model (no per-bar retraining)
```

---

## 🧪 Methodology

### 1. Label Construction (Step 2)
- Forward 3-day return: `r_future_3d = (close[t+18] / close[t]) - 1`
- Binary labels: UP if > 1%, DN if < -1%

### 2. Feature Preprocessing (Step 3)
- **83 features** selected (momentum, trend, volatility, volume, technical indicators)
- **Rolling standardization** with 504-bar window + 1-bar shift (prevents leakage)
- All transformations use past data only

### 3. Time-Series Split (Step 4)
- **28 rolling folds**: 365-day train + 90-day validation, step 90 days
- **Holdout**: 2025-09-01 to 2025-11-05 (never used for training/tuning)

### 4. Model Training (Step 5)
- **Dual models**: Random Forest & XGBoost
- **Separate UP/DN classifiers** - capture asymmetric patterns
- **Platt calibration** on validation set - stable probabilities

### 5. Threshold Search (Step 6)
- Grid search on validation folds only
- Selection by **median Sharpe** across folds
- Final: XGBoost with τ_up=0.30, τ_dn=0.25, m=0.08

### 6. Signal Generation (Step 7)
```
Long  if: p_up ≥ 0.30 AND (p_dn < 0.25 OR p_up - p_dn > 0.08)
Short if: p_dn ≥ 0.25 AND (p_up < 0.30 OR p_dn - p_up > 0.08)
Otherwise: Neutral
```

### 7. Execution (Step 7)
- Entry at next open, hold 3 days (18 bars), exit at next open
- 3× leverage, 0.1% round-trip costs
- Mark-to-market every 4h during holding period
- Non-overlapping trades

---

## 🎓 Academic Integrity

This project follows strict time-series best practices:
1. ✅ No look-ahead bias - all preprocessing uses past data only
2. ✅ Proper walk-forward validation - models retrained with expanding window
3. ✅ Separate validation/test/holdout - thresholds tuned on validation only
4. ✅ Realistic execution - costs, slippage assumptions stated clearly
5. ✅ Reproducible - fixed seeds, documented parameters

**References:**
- Chinco, Clark-Joseph, and Ye (2019) - Sparse signals in the cross-section of returns
- Kelly, Malamud, and Zhou (2024) - The virtue of complexity in return prediction

---

## 📝 Requirements

See `requirements.txt` for complete dependencies.

**Core libraries:**
- Python 3.10+
- pandas, numpy, matplotlib
- scikit-learn (LogisticRegression for Platt scaling)
- XGBoost

---

## ⚠️ Disclaimer

This is a **research/backtesting exercise** using historical data only. 

**DO NOT:**
- Place real orders or trade live accounts
- Use this for actual investment decisions without proper risk management
- Redistribute proprietary data

**Results shown:**
- Are historical simulations only
- Do not guarantee future performance
- May not reflect real-world slippage, market impact, or execution issues

---

## 🤝 Contributing

This project is for educational purposes (NUS DSA5205 Project 2).

**Team members:**
- [List your team members here]

**Individual contributions:**
- Data preprocessing & feature engineering: [Name]
- Model training & calibration: [Name]
- Walk-forward evaluation: [Name]
- Reporting & visualization: [Name]

---

## 📧 Contact

For questions about this project, please contact:
- [Your Name] - [Your Email]

---

## 📄 License

This project is for academic use only. Data sources should be properly attributed.

---

**Last Updated:** 2025-11-13
