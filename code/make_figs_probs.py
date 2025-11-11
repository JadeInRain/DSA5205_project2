#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate figures for:
(1) Calibrated probabilities around trade entries (Step7 trades, walk-forward recompute)
(2) Distribution of calibrated probabilities on validation folds (Step5 out-of-fold)

Outputs:
- figs/probs_around_entries.png
- figs/val_prob_distributions.png

Assumptions follow your Step5/6/7 pipeline conventions:
- data/prepared/btc_4h_preprocessed.csv
- data/predictions/val_all_folds.csv
- data/signals/test_trades.csv, data/signals/holdout_trades.csv
- data/thresholds/best_thresholds_per_model.csv

Author: you
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as PlattScaler

# -------------------- Paths (align with Step 5/6/7) --------------------
PREP_CSV   = os.path.join("data","prepared","btc_4h_preprocessed.csv")
VAL_ALL    = os.path.join("data","predictions","val_all_folds.csv")
TEST_TRDS  = os.path.join("data","signals","test_trades.csv")
HOLD_TRDS  = os.path.join("data","signals","holdout_trades.csv")
BEST_THR   = os.path.join("data","thresholds","best_thresholds_per_model.csv")

FIG_DIR    = os.path.join("figs")
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------- Globals (mirror Step7) --------------------
BAR_HOURS    = 4
TRAIN_DAYS   = 365
VAL_DAYS     = 90
HORIZON_DAYS = 7

TRAIN_BARS = (TRAIN_DAYS*24)//BAR_HOURS
VAL_BARS   = (VAL_DAYS*24)//BAR_HOURS
H_BARS     = (HORIZON_DAYS*24)//BAR_HOURS  # = 42 on 4h

TIMESTAMP_CANDIDATES = ["timestamp","open_time","time","datetime"]
PRICE_COLS  = ["open","high","low","close","volume"]
LABEL_COLS  = ["y_up","y_dn","r_future_7d"]
EXCLUDE_ALSO= ["future_close","next_open"]
RANDOM_STATE= 42

LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)
LSVC_CFG   = dict(C=1.0, class_weight="balanced")
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42)

# -------------------- Utils --------------------
def _find_ts_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in cols:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception:
            pass
    return df.columns[0]

def _load_prep() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(PREP_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = _find_ts_col(df)
    df[ts] = pd.to_datetime(df[ts], errors="coerce")
    df = df.sort_values(ts).reset_index(drop=True)
    return df, ts

def _select_features(df: pd.DataFrame, ts_col: str) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set([ts_col] + PRICE_COLS + LABEL_COLS + EXCLUDE_ALSO)
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    feats = [c for c in num_cols if c not in exclude]
    if not feats:
        raise RuntimeError("No numeric feature columns left after exclusion.")
    return feats

def _fit_base_models(X_tr, y_tr):
    m1 = LogisticRegression(**LOGREG_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    m2 = LinearSVC(**LSVC_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    m3 = GradientBoostingClassifier(**GBC_CFG).fit(X_tr, y_tr)
    return m1, m2, m3

def _score_raw(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

def _platt(scores, y):
    s = np.asarray(scores).reshape(-1,1)
    yy= np.asarray(y).astype(int)
    scaler = PlattScaler(C=1.0, solver="liblinear", max_iter=1000)
    scaler.fit(s, yy)
    return scaler

def _choose_model_key():
    if not os.path.exists(BEST_THR):
        # default fallback
        return "gb"
    best = pd.read_csv(BEST_THR)
    best.columns = [c.strip().lower() for c in best.columns]
    # take first row after sorting by median_sharpe (if present)
    if "median_sharpe" in best.columns:
        best = best.sort_values(["median_sharpe","geo_equity"], ascending=[False,False]).reset_index(drop=True)
    elif "avg_sharpe" in best.columns:
        best = best.sort_values(["avg_sharpe","total_equity"], ascending=[False,False]).reset_index(drop=True)
    else:
        best = best.sort_values(["geo_equity"], ascending=False).reset_index(drop=True)
    return str(best.iloc[0]["model"])

def _get_model_by_key(m1,m2,m3,key):
    return {"logreg":m1, "linsvc":m2, "gb":m3}[key]

# -------------------- Figure 1: probs around trade entries --------------------
def _load_trades():
    frames = []
    for p in [TEST_TRDS, HOLD_TRDS]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["segment"] = "test" if "test_" in os.path.basename(p) else "holdout"
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["time","side","entry","exit","segment"])
    trades = pd.concat(frames, ignore_index=True)
    trades = trades.sort_values("time").reset_index(drop=True)
    return trades

def compute_probs_around_entries(window_bars=12, sample_n=3, seed=7):
    """
    Recompute calibrated probabilities p_up,p_dn on bars [t-window_bars, t] for a few trades.
    This strictly uses trailing Train/Val windows (same sizes as Step7) to avoid leakage.
    """
    rng = np.random.default_rng(seed)
    df, ts = _load_prep()
    feats = _select_features(df, ts)
    trades = _load_trades()

    if trades.empty:
        print("[WARN] No trades found under data/signals/. Skipping probs-around-entries figure.")
        return None

    # sample a few entries (prefer holdout)
    hold = trades[trades["segment"]=="holdout"]
    base = hold if len(hold) >= sample_n else trades
    pick = base.sample(n=min(sample_n, len(base)), random_state=seed).sort_values("time")

    panels = []  # list of DataFrames with columns: time, p_up, p_dn, entry_time
    model_key = _choose_model_key()
    print(f"[INFO] Using model '{model_key}' for recompute around entries.")

    for _, row in pick.iterrows():
        # find index of entry decision bar:
        # trades CSV time should be the decision time t (entry at t+1 in your Step7),
        # so we recompute [t-window, t]
        entry_t = pd.Timestamp(row["time"])
        # find index in prep
        idx = df.index[df[ts] == entry_t]
        if len(idx)==0:
            # find nearest earlier bar
            idx = np.searchsorted(df[ts].values, np.datetime64(entry_t)) - 1
        else:
            idx = int(idx[0])

        start_k = max(0, idx - window_bars)
        series_out = []
        for t in range(start_k, idx+1):
            tr_start = t - (TRAIN_BARS + VAL_BARS)
            val_start= t - VAL_BARS
            if tr_start < 0:
                continue

            X_tr = df.loc[tr_start:val_start-1, feats].values
            X_va = df.loc[val_start:t-1, feats].values

            y_up_tr = df.loc[tr_start:val_start-1, "y_up"].astype("Int64")
            y_up_va = df.loc[val_start:t-1, "y_up"].astype("Int64")
            y_dn_tr = df.loc[tr_start:val_start-1, "y_dn"].astype("Int64")
            y_dn_va = df.loc[val_start:t-1, "y_dn"].astype("Int64")

            tr_mask = np.isfinite(X_tr).all(axis=1) & y_up_tr.notna().values & y_dn_tr.notna().values
            va_mask = np.isfinite(X_va).all(axis=1) & y_up_va.notna().values & y_dn_va.notna().values
            if tr_mask.sum() < 50 or va_mask.sum() < 20:
                continue

            Xtr, Xva = X_tr[tr_mask], X_va[va_mask]
            yup_tr = y_up_tr[tr_mask].astype(int).values
            yup_va = y_up_va[va_mask].astype(int).values
            ydn_tr = y_dn_tr[tr_mask].astype(int).values
            ydn_va = y_dn_va[va_mask].astype(int).values

            # fit UP/DN models + Platt on val
            m1, m2, m3 = _fit_base_models(Xtr, yup_tr)
            s1, s2, s3 = _score_raw(m1, Xva), _score_raw(m2, Xva), _score_raw(m3, Xva)
            up_platts = {"logreg": _platt(s1, yup_va), "linsvc": _platt(s2, yup_va), "gb": _platt(s3, yup_va)}

            m1_, m2_, m3_ = _fit_base_models(Xtr, ydn_tr)
            d1, d2, d3 = _score_raw(m1_, Xva), _score_raw(m2_, Xva), _score_raw(m3_, Xva)
            dn_platts = {"logreg": _platt(d1, ydn_va), "linsvc": _platt(d2, ydn_va), "gb": _platt(d3, ydn_va)}

            x_t = df.loc[[t], feats].values
            up_model = _get_model_by_key(m1, m2, m3, model_key)
            dn_model = _get_model_by_key(m1_, m2_, m3_, model_key)

            up_raw = _score_raw(up_model, x_t)
            dn_raw = _score_raw(dn_model, x_t)
            p_up = float(up_platts[model_key].predict_proba(np.asarray(up_raw).reshape(-1,1))[:,1][0])
            p_dn = float(dn_platts[model_key].predict_proba(np.asarray(dn_raw).reshape(-1,1))[:,1][0])

            series_out.append({"time": df.loc[t, ts], "p_up": p_up, "p_dn": p_dn})

        if series_out:
            out_df = pd.DataFrame(series_out).sort_values("time")
            out_df["entry_time"] = entry_t
            panels.append(out_df)

    if not panels:
        print("[WARN] Could not recompute probabilities around entries. Check data coverage.")
        return None
    return panels

def plot_probs_around_entries():
    panels = compute_probs_around_entries(window_bars=12, sample_n=3, seed=7)
    if panels is None:
        return None
    # one figure with 3 stacked subplots
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 6), sharex=False)
    if len(panels)==1:
        axes = [axes]
    for ax, dfp in zip(axes, panels):
        ax.plot(dfp["time"], dfp["p_up"], label="p_up")
        ax.plot(dfp["time"], dfp["p_dn"], label="p_dn", linestyle="--")
        ax.axvline(dfp["entry_time"].iloc[0], linestyle=":", linewidth=1.2)
        ax.set_ylabel("Calibrated p")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Calibrated probabilities around trade entries")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "probs_around_entries.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[OK] Saved: {out}")
    return out

# -------------------- Figure 2: distribution on validation folds --------------------
def plot_val_prob_distributions():
    if not os.path.exists(VAL_ALL):
        print("[WARN] Missing val_all_folds.csv, skip distributions.")
        return None
    val = pd.read_csv(VAL_ALL)
    val.columns = [c.strip().lower() for c in val.columns]

    # Gather p_up_* and p_dn_* columns across models
    up_cols = [c for c in val.columns if c.startswith("p_up_")]
    dn_cols = [c for c in val.columns if c.startswith("p_dn_")]
    if not up_cols or not dn_cols:
        print("[WARN] No p_up_* or p_dn_* columns found in val predictions.")
        return None

    # Two separate figures: UP and DN
    # UP
    plt.figure(figsize=(8,5))
    for c in up_cols:
        plt.hist(val[c].dropna(), bins=40, alpha=0.4, density=True, label=c)
    plt.xlabel("Calibrated probability (UP)")
    plt.ylabel("Density")
    plt.title("Distribution of calibrated probabilities on validation folds — UP")
    plt.legend(loc="best")
    out_up = os.path.join(FIG_DIR, "val_prob_distributions_up.png")
    plt.tight_layout(); plt.savefig(out_up, dpi=140); plt.close()
    print(f"[OK] Saved: {out_up}")

    # DN
    plt.figure(figsize=(8,5))
    for c in dn_cols:
        plt.hist(val[c].dropna(), bins=40, alpha=0.4, density=True, label=c)
    plt.xlabel("Calibrated probability (DN)")
    plt.ylabel("Density")
    plt.title("Distribution of calibrated probabilities on validation folds — DN")
    plt.legend(loc="best")
    out_dn = os.path.join(FIG_DIR, "val_prob_distributions_dn.png")
    plt.tight_layout(); plt.savefig(out_dn, dpi=140); plt.close()
    print(f"[OK] Saved: {out_dn}")
    return (out_up, out_dn)

# -------------------- Main --------------------
if __name__ == "__main__":
    _ = plot_probs_around_entries()
    _ = plot_val_prob_distributions()
    print("[DONE]")
