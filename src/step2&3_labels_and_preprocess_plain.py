# -*- coding: utf-8 -*-
"""
step2&3_labels_and_preprocess_plain.py  (no CLI args; run directly in PyCharm)

Step 2: Label construction (7-day ±3% events) on 4h bars
Step 3: Anti-leakage preprocessing (rolling z-score with shift(1))

Inputs:
  - btc_4h_factors.csv  (must contain price 'close' and a time column such as 'open_time')

Outputs (CSV):
  - data/labels/btc_4h_with_labels.csv
  - data/prepared/btc_4h_preprocessed.csv
"""

import os
import pandas as pd
import numpy as np
from typing import List

# ---------------------------
# User-editable constants
# ---------------------------
INPUT_CSV = "btc_4h_factors.csv"
BAR_HOURS = 4
HORIZON_DAYS = 7
HORIZON_BARS = (HORIZON_DAYS * 24) // BAR_HOURS  # 7天=42根4h
UP_THRESH = 0.02
DOWN_THRESH = -0.02

# 更稳妥的默认滚动窗：504 bars ≈ 12周（避免前期超长NaN）
ROLLING_WINDOW_BARS = 504

OUT_LABELS_CSV = os.path.join("data", "labels", "btc_4h_with_labels.csv")
OUT_PREP_CSV   = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")

# ---------------------------
# Helpers
# ---------------------------
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _autodetect_time_col(cols: List[str]) -> str:
    candidates = ["timestamp", "open_time", "datetime", "date"]
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"无法识别时间列，请确保存在其中之一：{candidates}；当前列名：{list(cols)[:20]} ...")

def _ensure_time_sorted(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    # 尝试转为 datetime（若已是字符串）
    try:
        out[ts_col] = pd.to_datetime(out[ts_col])
    except Exception:
        pass
    out = out.sort_values(ts_col).reset_index(drop=True)
    return out

def build_labels(df: pd.DataFrame, ts_col: str, close_col: str,
                 horizon_bars: int, up_thresh: float, down_thresh: float) -> pd.DataFrame:
    out = df.copy()
    out["future_close"] = out[close_col].shift(-horizon_bars)
    out["r_future_7d"] = (out["future_close"] / out[close_col]) - 1.0
    out["y_up"] = (out["r_future_7d"] > up_thresh).astype("Int64")
    out["y_dn"] = (out["r_future_7d"] < down_thresh).astype("Int64")
    return out

def add_next_open(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "open" in out.columns:
        out["next_open"] = out["open"].shift(-1)
    return out

def select_feature_columns(df: pd.DataFrame,
                           ts_col: str,
                           price_cols: List[str],
                           label_cols: List[str]) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set([ts_col] + price_cols + label_cols + ["future_close", "r_future_7d", "next_open"])
    # 排除任何以 'future_' / 'label_' 开头的列
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    features = [c for c in num_cols if c not in exclude]
    return features

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def rolling_standardize(df: pd.DataFrame, features: List[str], window: int) -> pd.DataFrame:
    """
    Anti-leakage rolling z-score:
      z_t = (x_t - mean_{t-1}) / std_{t-1}
    with mean/std computed by rolling(window) and then shift(1).

    - Uses a safe epsilon for std to avoid all-NaN columns when std==0.
    - If dataset is small, fall back to expanding stats (also shift(1)).
    """
    out = df.copy()
    n = len(out)
    if n < window + 50:
        # fallback: expanding stats
        for c in features:
            mean_exp = out[c].expanding(min_periods=20).mean().shift(1)
            std_exp  = out[c].expanding(min_periods=20).std(ddof=0).shift(1)
            std_safe = std_exp.where(std_exp > 1e-12, 1e-12)
            out[c] = (out[c] - mean_exp) / std_safe
        return out

    minp = max(20, min(window // 4, 200))  # 更保守的 min_periods，减少整列NaN的风险
    for c in features:
        roll_mean = out[c].rolling(window=window, min_periods=minp).mean().shift(1)
        roll_std  = out[c].rolling(window=window, min_periods=minp).std(ddof=0).shift(1)
        std_safe  = roll_std.where(roll_std > 1e-12, 1e-12)
        out[c] = (out[c] - roll_mean) / std_safe

    return out

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到输入文件：{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = _normalize_colnames(df)

    # ---- autodetect time & basic columns ----
    ts_col = _autodetect_time_col(df.columns)
    close_col = "close"
    if close_col not in df.columns:
        raise ValueError(f"缺少收盘价列 'close'。现有列：{list(df.columns)[:20]} ...")

    df = _ensure_time_sorted(df, ts_col)

    # ---- Step 2: Labels ----
    df_lab = build_labels(df, ts_col, close_col, HORIZON_BARS, UP_THRESH, DOWN_THRESH)
    df_lab = add_next_open(df_lab)

    # ---- Step 3: Feature preprocessing (anti-leakage) ----
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_lab.columns]
    label_cols = ["y_up", "y_dn"]
    features = select_feature_columns(df_lab, ts_col, price_cols, label_cols)

    # 强制特征列转为数值（避免 “字符串” 导致的整列空值）
    df_num = coerce_numeric(df_lab, features)

    # 滚动标准化（带 shift(1) + std 安全下限）
    df_prep = rolling_standardize(df_num, features, window=ROLLING_WINDOW_BARS)

    # ---- save ----
    os.makedirs(os.path.dirname(OUT_LABELS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PREP_CSV), exist_ok=True)
    df_lab.to_csv(OUT_LABELS_CSV, index=False)
    df_prep.to_csv(OUT_PREP_CSV, index=False)

    # ---- diagnostics ----
    n_total = len(df_prep)
    n_valid = int(df_prep["r_future_7d"].notna().sum()) if "r_future_7d" in df_prep.columns else 0
    up_rate = float(df_prep["y_up"].dropna().mean()) if "y_up" in df_prep.columns else float("nan")
    dn_rate = float(df_prep["y_dn"].dropna().mean()) if "y_dn" in df_prep.columns else float("nan")

    # 哪些特征是“整列NaN”（帮助排错）
    all_nan_features = [c for c in features if df_prep[c].isna().all()]

    print("="*72)
    print("[Step 2] Labels created on 4h grid with 7d horizon (42 bars).")
    print(f"Total rows: {n_total:,}; rows with full horizon: {n_valid:,}")
    print(f"Positive rate — up: {up_rate:.3f} | down: {dn_rate:.3f}")
    print(f"Saved labels to: {OUT_LABELS_CSV}")

    print("\n[Step 3] Rolling standardization applied (anti-leakage).")
    print(f"Rolling window (bars): {ROLLING_WINDOW_BARS}")
    print(f"#Features standardized: {len(features)}")
    if len(features) <= 25:
        print("Features:", features)
    else:
        print("First 25 features:", features[:25], "...")
    print(f"Saved preprocessed data to: {OUT_PREP_CSV}")

    if all_nan_features:
        print("\n[Diagnosis] Features that are ALL NaN after standardization:")
        for c in all_nan_features:
            print("  -", c)
        print("可能原因：该列为常数/极低方差；或原始数据列是非数值字符串。已对特征列统一做了 to_numeric，并对std做了epsilon保护。")
    print("="*72)

if __name__ == "__main__":
    main()
