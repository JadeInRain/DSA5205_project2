# -*- coding: utf-8 -*-
"""
step4_time_splits.py
====================

Purpose
-------
Implements **Step 4: Time-respecting Train/Validation/Holdout splits** for the BTC 4h dataset.
- Rolling Train/Validation folds before the final Holdout period
- Final Holdout (completely untouched for final report)

Run
---
Place this file into your project (e.g., `src/step4_time_splits.py`) and run it directly in PyCharm.
It reads `data/prepared/btc_4h_preprocessed.csv` and writes split definitions as CSVs.

Outputs
-------
- `data/splits/rolling_folds.csv`    # per-fold train/val ranges (one row per segment per fold)
- `data/splits/holdout.csv`          # final holdout start/end rows
- Console summary (counts, dates)

Edit constants below to adjust windows and holdout dates.
"""
import os
import pandas as pd
import numpy as np

# ========================
# User-editable constants
# ========================
INPUT_PREP_CSV   = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
TIMESTAMP_CANDIDATES = ["timestamp", "open_time", "time", "datetime"]

BAR_HOURS        = 4
TRAIN_DAYS       = 365
VAL_DAYS         = 90
STEP_DAYS        = 30    # window advance per fold
HOLDOUT_START    = "2025-08-01"   # inclusive
HOLDOUT_END      = "2025-11-30"   # inclusive

# Derived bars
TRAIN_BARS = (TRAIN_DAYS * 24) // BAR_HOURS
VAL_BARS   = (VAL_DAYS * 24) // BAR_HOURS
STEP_BARS  = (STEP_DAYS * 24) // BAR_HOURS

OUT_FOLDS_CSV   = os.path.join("data", "splits", "rolling_folds.csv")
OUT_HOLDOUT_CSV = os.path.join("data", "splits", "holdout.csv")

def _find_ts_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in cols:
            return c
    # fallback: first datetime-like or the first column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return df.columns[0]

def _load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Make sure Step 2&3 output exists.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = _find_ts_col(df)
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception:
        pass
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df, ts_col

def build_rolling_folds(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """
    Build rolling (train, val) folds prior to HOLDOUT_START.
    Each fold advances by STEP_BARS.
    """
    # Work only on pre-holdout region
    pre_mask = df[ts_col] < pd.to_datetime(HOLDOUT_START)
    pre_df = df.loc[pre_mask].reset_index(drop=True)
    n = len(pre_df)
    if n < TRAIN_BARS + VAL_BARS + 1:
        raise ValueError("Not enough pre-holdout data to build at least one fold. "
                         f"Need >= {TRAIN_BARS + VAL_BARS + 1} rows; got {n}.")

    rows = []
    fold_id = 0
    start = 0
    while True:
        train_start = start
        train_end   = train_start + TRAIN_BARS               # exclusive
        val_start   = train_end
        val_end     = val_start + VAL_BARS                   # exclusive

        if val_end > n:
            break

        train_s_ts = pre_df.loc[train_start, ts_col]
        train_e_ts = pre_df.loc[train_end-1, ts_col]
        val_s_ts   = pre_df.loc[val_start, ts_col]
        val_e_ts   = pre_df.loc[val_end-1, ts_col]

        rows.append({
            "fold_id": fold_id, "segment": "train",
            "start_idx": int(train_start), "end_idx": int(train_end-1),
            "start_time": train_s_ts, "end_time": train_e_ts,
            "length": int(TRAIN_BARS)
        })
        rows.append({
            "fold_id": fold_id, "segment": "val",
            "start_idx": int(val_start), "end_idx": int(val_end-1),
            "start_time": val_s_ts, "end_time": val_e_ts,
            "length": int(VAL_BARS)
        })

        fold_id += 1
        start += STEP_BARS
        if start + TRAIN_BARS + VAL_BARS > n:
            break

    return pd.DataFrame(rows)

def build_holdout(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    h_start = pd.to_datetime(HOLDOUT_START)
    h_end   = pd.to_datetime(HOLDOUT_END)
    mask = (df[ts_col] >= h_start) & (df[ts_col] <= h_end)
    sub = df.loc[mask, [ts_col]].copy()
    if sub.empty:
        raise ValueError("Holdout range has no rows. Adjust HOLDOUT_START/HOLDOUT_END.")
    return pd.DataFrame({
        "start_time": [sub[ts_col].iloc[0]],
        "end_time":   [sub[ts_col].iloc[-1]],
        "start_idx":  [int(sub.index[0])],
        "end_idx":    [int(sub.index[-1])],
        "n_rows":     [int(len(sub))]
    })

def main():
    df, ts_col = _load_data(INPUT_PREP_CSV)

    folds = build_rolling_folds(df, ts_col)
    os.makedirs(os.path.dirname(OUT_FOLDS_CSV), exist_ok=True)
    folds.to_csv(OUT_FOLDS_CSV, index=False)

    holdout = build_holdout(df, ts_col)
    holdout.to_csv(OUT_HOLDOUT_CSV, index=False)

    # Summary
    print("="*72)
    print("[Step 4] Time-respecting splits constructed.")
    print(f"Timestamp column: {ts_col}")
    print(f"Bars — train: {TRAIN_BARS}, val: {VAL_BARS}, step: {STEP_BARS}")
    print(f"Rolling folds: {folds['fold_id'].nunique()} (rows written to {OUT_FOLDS_CSV})")

    # Print first and last fold
    if not folds.empty:
        first_train = folds[folds["segment"]=="train"].iloc[0]
        first_val   = folds[folds["segment"]=="val"].iloc[0]
        last_train  = folds[folds["segment"]=="train"].iloc[-1]
        last_val    = folds[folds["segment"]=="val"].iloc[-1]

        print("\nFirst fold:")
        print(f"  Train: {first_train['start_time']} → {first_train['end_time']}  "
              f"({int(first_train['length'])} bars)")
        print(f"  Val  : {first_val['start_time']} → {first_val['end_time']}  "
              f"({int(first_val['length'])} bars)")

        print("\nLast fold:")
        print(f"  Train: {last_train['start_time']} → {last_train['end_time']}")
        print(f"  Val  : {last_val['start_time']} → {last_val['end_time']}")

    print("\nHoldout window written to:", OUT_HOLDOUT_CSV)
    print(holdout.to_string(index=False))
    print("="*72)

if __name__ == "__main__":
    main()
