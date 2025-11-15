# -*- coding: utf-8 -*-
"""
step6_thresholds.py  (Refined)
==============================
Validation-only threshold search with:
- Non-overlapping trades (skip 42 bars after each entry)
- Minimum trade-count filter per fold
- Median Sharpe across folds (robust)

Inputs
------
- data/predictions/val_all_folds.csv     (Step 5)
- data/prepared/btc_4h_preprocessed.csv  (for r_future_7d + timestamps)

Outputs
-------
- data/thresholds/grid_scores_<model>.csv
- data/thresholds/best_thresholds_per_model.csv

Run: direct (no CLI args) in PyCharm.
"""

import os
import numpy as np
import pandas as pd

# -------------------- User constants --------------------
PRED_CSV   = os.path.join("data", "predictions", "val_all_folds.csv")
PREP_CSV   = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")

# Trading params
BAR_HOURS      = 4
HORIZON_DAYS   = 7
H_BARS         = (HORIZON_DAYS * 24) // BAR_HOURS   # = 42 for 4h
LEVERAGE       = 3.0
FEE_PER_SIDE   = 0.0005          # 0.05% each side
ROUND_TRIP     = 2*FEE_PER_SIDE  # 0.1%

# Grid
# TAU_UP_GRID    = np.round(np.linspace(0.50, 0.80, 7), 3)
# TAU_DN_GRID    = np.round(np.linspace(0.50, 0.80, 7), 3)
# MARGIN_GRID    = np.round(np.linspace(0.02, 0.20, 10), 3)
TAU_UP_GRID  = np.round(np.linspace(0.35, 0.75, 9), 3)   # 0.35, 0.40, …, 0.75
TAU_DN_GRID  = np.round(np.linspace(0.35, 0.75, 9), 3)
MARGIN_GRID  = np.round(np.linspace(0.00, 0.20, 11), 3)  # 0.00, 0.02, …, 0.20
# Models to evaluate (suffix in PRED_CSV columns)
MODEL_KEYS     = ["logreg","linsvc","gb"]

# Robustness
MIN_TRADES_PER_FOLD = 5         # fold producing < N trades is ignored for Sharpe; tweak as needed
GLOBAL_MIN_TRADES   = 30        # total trades across folds must be >= this to be considered
ANN_FACTOR          = 365*24/BAR_HOURS   # per-bar annualization

def _load_data():
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Missing predictions: {PRED_CSV}")
    if not os.path.exists(PREP_CSV):
        raise FileNotFoundError(f"Missing base data: {PREP_CSV}")

    pred = pd.read_csv(PRED_CSV)
    pred.columns = [c.strip().lower() for c in pred.columns]
    pred["time"] = pd.to_datetime(pred["time"], errors="coerce")

    base = pd.read_csv(PREP_CSV)
    base.columns = [c.strip().lower() for c in base.columns]
    # detect timestamp column
    ts_col = "timestamp" if "timestamp" in base.columns else ("open_time" if "open_time" in base.columns else base.columns[0])
    base[ts_col] = pd.to_datetime(base[ts_col], errors="coerce")
    base = base[[ts_col, "r_future_7d"]].rename(columns={ts_col:"time"})

    df = pred.merge(base, on="time", how="left")
    df = df.dropna(subset=["r_future_7d","y_up","y_dn"]).sort_values(["fold_id","time"]).reset_index(drop=True)
    return df

def _simulate_fold(df_fold, tau_up, tau_dn, margin, model_key):
    """
    Non-overlapping trades: after taking a trade at i, skip next H_BARS rows.
    Use r_future_7d[i] as the realized 7d return proxy.
    """
    p_up = df_fold[f"p_up_{model_key}"].values
    p_dn = df_fold[f"p_dn_{model_key}"].values
    r7   = df_fold["r_future_7d"].values

    equity = 1.0
    path = [equity]
    i = 0
    n = len(df_fold)
    trades = 0
    wins = 0
    rets = []  # per-trade net return sequence (for diagnostics)

    while i < n:
        long_sig  = (p_up[i] >= tau_up) and ( (p_dn[i] < tau_dn) or ((p_dn[i] >= tau_dn) and (p_up[i]-p_dn[i] > margin)) )
        short_sig = (p_dn[i] >= tau_dn) and ( (p_up[i] < tau_up) or ((p_up[i] >= tau_up) and (p_dn[i]-p_up[i] > margin)) )

        if long_sig and not short_sig:
            net = LEVERAGE * r7[i] - ROUND_TRIP
            equity *= (1.0 + net)
            trades += 1
            wins += (net > 0)
            rets.append(net)
            # append flat path for the H_BARS holding window (no compounding between decisions)
            for _ in range(min(H_BARS, n - i)):
                path.append(equity)
            i += H_BARS  # skip 42 bars
        elif short_sig and not long_sig:
            net = LEVERAGE * (-r7[i]) - ROUND_TRIP
            equity *= (1.0 + net)
            trades += 1
            wins += (net > 0)
            rets.append(net)
            for _ in range(min(H_BARS, n - i)):
                path.append(equity)
            i += H_BARS
        else:
            i += 1
            path.append(equity)

    path = np.array(path, dtype=float)
    # Per-bar return series for Sharpe
    ret_series = pd.Series(path).pct_change().dropna()
    mu = ret_series.mean() * ANN_FACTOR
    sd = ret_series.std(ddof=0) * np.sqrt(ANN_FACTOR)
    sharpe = (mu / sd) if (sd and sd > 0) else np.nan

    win_rate = (wins / trades) if trades > 0 else np.nan
    return {
        "final_equity": float(equity),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "trades": int(trades),
        "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan
    }

def _grid_search(df):
    rows = []
    for mk in MODEL_KEYS:
        for tu in TAU_UP_GRID:
            for td in TAU_DN_GRID:
                for m in MARGIN_GRID:
                    fold_stats = []
                    for fid, sub in df.groupby("fold_id"):
                        sub = sub.sort_values("time")
                        stats = _simulate_fold(sub, tu, td, m, mk)
                        # apply per-fold trade-count filter
                        if stats["trades"] >= MIN_TRADES_PER_FOLD:
                            fold_stats.append(stats)
                    # aggregate across folds
                    if fold_stats:
                        sharpe_vals = [s["sharpe"] for s in fold_stats if pd.notna(s["sharpe"])]
                        median_sharpe = float(np.median(sharpe_vals)) if sharpe_vals else np.nan
                        geo_equity = float(np.prod([s["final_equity"] for s in fold_stats]) ** (1.0/len(fold_stats))) if fold_stats else np.nan
                        avg_trades = float(np.mean([s["trades"] for s in fold_stats]))
                        avg_win    = float(np.mean([s["win_rate"] for s in fold_stats if pd.notna(s["win_rate"])])) if fold_stats else np.nan
                        total_trades = int(np.sum([s["trades"] for s in fold_stats]))
                    else:
                        median_sharpe = np.nan; geo_equity = np.nan; avg_trades = 0.0; avg_win = np.nan; total_trades = 0

                    rows.append({
                        "model": mk, "tau_up": tu, "tau_dn": td, "margin": m,
                        "median_sharpe": median_sharpe,
                        "geo_equity": geo_equity,
                        "avg_trades": avg_trades,
                        "total_trades": total_trades,
                        "avg_win_rate": avg_win
                    })
    return pd.DataFrame(rows)

def _select_best(df_grid):
    # enforce global minimum trades
    cand = df_grid[df_grid["total_trades"] >= GLOBAL_MIN_TRADES].copy()
    if cand.empty:
        cand = df_grid.copy()  # fallback

    winners = []
    for mk in MODEL_KEYS:
        sub = cand[cand["model"]==mk].copy()
        sub = sub.sort_values(
            by=["median_sharpe", "geo_equity", "avg_trades"],
            ascending=[False, False, False]
        )
        if len(sub):
            winners.append(sub.iloc[0].to_dict())
    return pd.DataFrame(winners)

def run():
    df = _load_data()
    grid = _grid_search(df)

    outdir = os.path.join("data", "thresholds")
    os.makedirs(outdir, exist_ok=True)
    for mk in MODEL_KEYS:
        grid[grid["model"]==mk].to_csv(os.path.join(outdir, f"grid_scores_{mk}.csv"), index=False)

    best = _select_best(grid)
    best.to_csv(os.path.join(outdir, "best_thresholds_per_model.csv"), index=False)

    print("="*72)
    print("[Step 6] Grid search complete (refined).")
    print("Best per model:")
    if not best.empty:
        print(best.to_string(index=False))
    print("Files written to:", outdir)
    print("="*72)

if __name__ == "__main__":
    run()
