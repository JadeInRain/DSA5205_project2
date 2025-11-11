# -*- coding: utf-8 -*-
"""
step7_v1.py
=================
Walk-forward 概率 → 信号 → 非重叠 7d 执行（Test 段 + Holdout 段）

输入
- data/prepared/btc_4h_preprocessed.csv
- data/splits/rolling_folds.csv
- data/splits/holdout.csv
- data/thresholds/best_thresholds_per_model.csv   # 来自 Step 6

输出
- data/signals/test_trades.csv,    data/equity/test_equity.csv
- data/signals/holdout_trades.csv, data/equity/holdout_equity.csv
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

# ------------------ 路径 ------------------
PREP_CSV     = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FOLDS_CSV    = os.path.join("data", "splits", "rolling_folds.csv")
HOLDOUT_CSV  = os.path.join("data", "splits", "holdout.csv")
BEST_THR_CSV = os.path.join("data", "thresholds", "best_thresholds_per_model.csv")

os.makedirs(os.path.join("data", "signals"), exist_ok=True)
os.makedirs(os.path.join("data", "equity"), exist_ok=True)

# ------------------ 窗口 & 执行参数（与前面保持一致） ------------------
BAR_HOURS    = 4
TRAIN_DAYS   = 365
VAL_DAYS     = 90
TRAIN_BARS   = (TRAIN_DAYS*24)//BAR_HOURS       # 2190
VAL_BARS     = (VAL_DAYS*24)//BAR_HOURS         # 540
HORIZON_DAYS = 7
H_BARS       = (HORIZON_DAYS*24)//BAR_HOURS     # 42

LEVERAGE     = 3.0
FEE_PER_SIDE = 0.0005
ROUND_TRIP   = 2*FEE_PER_SIDE

# ------------------ 模型配置（与 Step 5 对齐，避免重复 random_state） ------------------
LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear",
                  class_weight="balanced", max_iter=2000, random_state=42)
LSVC_CFG   = dict(C=1.0, class_weight="balanced", random_state=42)
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2,
                  subsample=0.8, random_state=42)

TIMESTAMP_CANDIDATES = ["timestamp","open_time","time","datetime"]
PRICE_COLS  = ["open","high","low","close","volume"]
LABEL_COLS  = ["y_up","y_dn","r_future_7d"]
EXCLUDE_ALSO= ["future_close","next_open"]  # 训练用特征需排除

# 若想强制指定模型，可赋值为 "logreg" / "linsvc" / "gb"；否则自动用 Step6 第一名
MODEL_CHOICE = None

# ------------------ 工具函数 ------------------
def _find_ts_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in cols: return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception:
            pass
    return df.columns[0]

def _load_base():
    if not os.path.exists(PREP_CSV):
        raise FileNotFoundError(f"Not found: {PREP_CSV}")
    df = pd.read_csv(PREP_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = _find_ts_col(df)
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception:
        pass
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df, ts_col

def _load_splits():
    folds = pd.read_csv(FOLDS_CSV); folds.columns=[c.strip().lower() for c in folds.columns]
    hold  = pd.read_csv(HOLDOUT_CSV); hold.columns=[c.strip().lower() for c in hold.columns]
    return folds, hold

def _load_thresholds():
    if not os.path.exists(BEST_THR_CSV):
        raise FileNotFoundError(f"Not found: {BEST_THR_CSV}")
    best = pd.read_csv(BEST_THR_CSV)
    best.columns = [c.strip().lower() for c in best.columns]
    # 用 median_sharpe → geo_equity 选第一名；若列不存在则降级
    if "median_sharpe" in best.columns:
        best = best.sort_values(["median_sharpe","geo_equity"], ascending=[False, False]).reset_index(drop=True)
    elif "avg_sharpe" in best.columns:
        best = best.sort_values(["avg_sharpe","total_equity"], ascending=[False, False]).reset_index(drop=True)
    else:
        best = best.sort_values(["geo_equity"], ascending=False).reset_index(drop=True)
    return best

def _choose_model_and_thresholds(best_df):
    global MODEL_CHOICE
    if MODEL_CHOICE is None:
        MODEL_CHOICE = best_df.iloc[0]["model"]
    row = best_df[best_df["model"]==MODEL_CHOICE].iloc[0]
    return MODEL_CHOICE, float(row["tau_up"]), float(row["tau_dn"]), float(row["margin"])

def _select_features(df: pd.DataFrame, ts_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude  = set([ts_col] + PRICE_COLS + LABEL_COLS + EXCLUDE_ALSO)
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    return [c for c in num_cols if c not in exclude]

def _build_model(model_key: str):
    if model_key == "logreg":
        return LogisticRegression(**LOGREG_CFG)
    elif model_key == "linsvc":
        return LinearSVC(**LSVC_CFG)
    elif model_key == "gb":
        return GradientBoostingClassifier(**GBC_CFG)
    else:
        raise ValueError(f"Unknown model key: {model_key}")

def _score_raw(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

def _platt_fit(raw_scores, y_val):
    """在验证窗对原始分数做逻辑回归校准；输入既支持 Series 也支持 ndarray。"""
    s = np.asarray(raw_scores).reshape(-1, 1)
    y = np.asarray(y_val).astype(int).ravel()
    scaler = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
    scaler.fit(s, y)
    return scaler

def _segment_indices(df: pd.DataFrame, ts_col: str, folds: pd.DataFrame, hold: pd.DataFrame) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    # Test 段：最后一个验证结束之后 到 Holdout 起点之前
    last_val_end = pd.to_datetime(folds[folds["segment"]=="val"]["end_time"].max())
    mask_after_last_val = df[ts_col] > last_val_end
    test_start_idx = int(df.index[mask_after_last_val][0]) if mask_after_last_val.any() else None

    hstart = pd.to_datetime(hold["start_time"].iloc[0])
    hend   = pd.to_datetime(hold["end_time"].iloc[0])
    hold_start_idx = int(df.index[df[ts_col] >= hstart][0])
    hold_end_idx   = int(df.index[df[ts_col] <= hend][-1])

    test_end_idx = hold_start_idx - 1 if test_start_idx is not None else None
    return (test_start_idx, test_end_idx), (hold_start_idx, hold_end_idx)

# ------------------ 概率生成（走进式） ------------------
def _walk_forward_probs(df: pd.DataFrame, ts_col: str, start_idx: int, end_idx: int, model_key: str) -> pd.DataFrame:
    feats = _select_features(df, ts_col)
    out = []

    for t in range(start_idx, end_idx + 1):
        tr_start = t - (TRAIN_BARS + VAL_BARS)
        val_start= t - VAL_BARS
        if tr_start < 0 or val_start <= tr_start:
            continue

        # 取窗口
        X_tr = df.loc[tr_start:val_start-1, feats].values
        X_va = df.loc[val_start:t-1,        feats].values

        y_up_tr = df.loc[tr_start:val_start-1, "y_up"].to_numpy()
        y_up_va = df.loc[val_start:t-1,        "y_up"].to_numpy()
        y_dn_tr = df.loc[tr_start:val_start-1, "y_dn"].to_numpy()
        y_dn_va = df.loc[val_start:t-1,        "y_dn"].to_numpy()

        # 掩码（剔除 NaN）
        tr_mask = np.isfinite(X_tr).all(axis=1) & ~np.isnan(y_up_tr) & ~np.isnan(y_dn_tr)
        va_mask = np.isfinite(X_va).all(axis=1) & ~np.isnan(y_up_va) & ~np.isnan(y_dn_va)

        if tr_mask.sum() < 50 or va_mask.sum() < 20:
            continue

        Xtr = X_tr[tr_mask]; Xva = X_va[va_mask]
        y_up_tr2 = y_up_tr[tr_mask].astype(int); y_up_va2 = y_up_va[va_mask].astype(int)
        y_dn_tr2 = y_dn_tr[tr_mask].astype(int); y_dn_va2 = y_dn_va[va_mask].astype(int)

        # 仅训练 Step6 胜出模型（效率更高）
        # up 任务
        m_up = _build_model(model_key); m_up.fit(Xtr, y_up_tr2)
        s_up = _score_raw(m_up, Xva)
        cal_up = _platt_fit(s_up, y_up_va2)

        # down 任务
        m_dn = _build_model(model_key); m_dn.fit(Xtr, y_dn_tr2)
        s_dn = _score_raw(m_dn, Xva)
        cal_dn = _platt_fit(s_dn, y_dn_va2)

        # 当前时点的特征
        x_t = df.loc[[t], feats].values
        # 原始分数 → 概率
        p_up = cal_up.predict_proba(np.asarray(_score_raw(m_up, x_t)).reshape(-1,1))[:,1][0]
        p_dn = cal_dn.predict_proba(np.asarray(_score_raw(m_dn, x_t)).reshape(-1,1))[:,1][0]

        out.append({"idx": t, "time": df.loc[t, ts_col], "p_up": float(p_up), "p_dn": float(p_dn)})

    return pd.DataFrame(out)

# ------------------ 执行（非重叠 7d，下一根开盘入、+7d 下一根开盘出） ------------------
def _find_exec_price_columns(df: pd.DataFrame) -> str:
    if "open" in df.columns:
        return "open"
    elif "close" in df.columns:
        print("[Step 7] WARNING: 'open' not found. Falling back to 'close' as execution price.")
        return "close"
    else:
        raise KeyError("Neither 'open' nor 'close' price column found for execution.")

def _simulate_execution(df: pd.DataFrame, probs: pd.DataFrame, ts_col: str, tau_up: float, tau_dn: float, margin: float):
    price_col = _find_exec_price_columns(df)
    d = df[[ts_col, price_col]].copy()
    d.rename(columns={price_col: "px"}, inplace=True)
    d["next_px"] = d["px"].shift(-1)
    d["exit_px"] = d["px"].shift(-(H_BARS+1))  # +7d 的下一根开盘（或价格代理）
    d = d.merge(probs, left_on=ts_col, right_on="time", how="inner").sort_values(ts_col).reset_index(drop=True)

    eq = 1.0
    equity = [{"time": d.loc[0, ts_col], "equity": eq}]
    trades = []
    i = 0; n = len(d)

    while i < n:
        p_up = d.loc[i, "p_up"]; p_dn = d.loc[i, "p_dn"]
        long_sig  = (p_up >= tau_up) and ((p_dn < tau_dn) or ((p_dn >= tau_dn) and (p_up - p_dn >  margin)))
        short_sig = (p_dn >= tau_dn) and ((p_up < tau_up) or ((p_up >= tau_up) and (p_dn - p_up >  margin)))

        if long_sig and not short_sig and pd.notna(d.loc[i, "next_px"]) and pd.notna(d.loc[i, "exit_px"]):
            entry = float(d.loc[i, "next_px"]); exitp = float(d.loc[i, "exit_px"])
            gross = (exitp / entry - 1.0) * LEVERAGE
            net   = gross - ROUND_TRIP
            eq   *= (1.0 + net)
            trades.append({"time": d.loc[i, ts_col], "side": "long",  "entry": entry, "exit": exitp, "gross": gross, "net": net})
            i += H_BARS
        elif short_sig and not long_sig and pd.notna(d.loc[i, "next_px"]) and pd.notna(d.loc[i, "exit_px"]):
            entry = float(d.loc[i, "next_px"]); exitp = float(d.loc[i, "exit_px"])
            gross = (entry / exitp - 1.0) * LEVERAGE  # short
            net   = gross - ROUND_TRIP
            eq   *= (1.0 + net)
            trades.append({"time": d.loc[i, ts_col], "side": "short", "entry": entry, "exit": exitp, "gross": gross, "net": net})
            i += H_BARS
        else:
            i += 1

        equity.append({"time": d.loc[min(i, n-1), ts_col] if i < n else d.loc[n-1, ts_col], "equity": eq})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).drop_duplicates(subset=["time"], keep="last")
    return trades_df, equity_df

# ------------------ 主流程 ------------------
def run():
    df, ts_col     = _load_base()
    folds, hold    = _load_splits()
    best           = _load_thresholds()
    model_key, tau_up, tau_dn, margin = _choose_model_and_thresholds(best)

    print(f"[Step 7] Using model={model_key}, thresholds: tau_up={tau_up}, tau_dn={tau_dn}, margin={margin}")

    (ts_start, ts_end), (ho_start, ho_end) = _segment_indices(df, ts_col, folds, hold)

    # --- Test 段 ---
    if ts_start is not None and ts_end is not None and ts_start < ts_end:
        probs_test = _walk_forward_probs(df, ts_col, ts_start, ts_end, model_key)
        if probs_test.empty:
            print("[Step 7] Test: insufficient data to produce probabilities.")
        else:
            trades_t, equity_t = _simulate_execution(df, probs_test, ts_col, tau_up, tau_dn, margin)
            trades_t.to_csv(os.path.join("data", "signals", "test_trades.csv"), index=False)
            equity_t.to_csv(os.path.join("data", "equity", "test_equity.csv"), index=False)
            print(f"[Step 7] Test: wrote {len(trades_t)} trades; equity rows={len(equity_t)}")
    else:
        print("[Step 7] Test segment not available.")

    # --- Holdout 段 ---
    probs_hold = _walk_forward_probs(df, ts_col, ho_start, ho_end, model_key)
    if probs_hold.empty:
        print("[Step 7] Holdout: insufficient data to produce probabilities.")
    else:
        trades_h, equity_h = _simulate_execution(df, probs_hold, ts_col, tau_up, tau_dn, margin)
        trades_h.to_csv(os.path.join("data", "signals", "holdout_trades.csv"), index=False)
        equity_h.to_csv(os.path.join("data", "equity", "holdout_equity.csv"), index=False)
        print(f"[Step 7] Holdout: wrote {len(trades_h)} trades; equity rows={len(equity_h)}")

    print("="*72)
    print("[Step 7] Done. Files written to data/signals/ and data/equity/.")
    print("Next: Step 8 — metrics & plots, and benchmark comparison.")
    print("="*72)

if __name__ == "__main__":
    run()
