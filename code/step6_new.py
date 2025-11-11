# -*- coding: utf-8 -*-
"""
step6_thresholds.py  — Quantile Threshold Search
------------------------------------------------
在验证集上，按分位数阈值 (q_up, q_dn) + margin 搜索最佳组合：
- 每个 fold：用该 fold 验证段的概率分布计算阈值：
    tau_up = quantile(p_up, q_up), tau_dn = quantile(p_dn, q_dn)
- 非重叠 7 天持有（跳过 42 根），3x 杠杆，0.05%/边手 续费
- 过滤掉交易数过少的 fold（MIN_TRADES_PER_FOLD），跨折以 **Sharpe 中位数** 作为主指标
- 按模型分别选出最优 (q_up, q_dn, margin)，落盘到 best_thresholds_per_model.csv
  （并标注 mode='quantile'，供 Step 7 读取）

输入:
- data/predictions/val_all_folds.csv  (Step 5)
- data/prepared/btc_4h_preprocessed.csv  (取 r_future_7d)
输出:
- data/thresholds/grid_scores_<model>.csv
- data/thresholds/best_thresholds_per_model.csv
"""

import os
import numpy as np
import pandas as pd

# ---------- 路径 ----------
PRED_CSV = os.path.join("data", "predictions", "val_all_folds.csv")
PREP_CSV = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
OUT_DIR  = os.path.join("data", "thresholds")

# ---------- 交易 & 时间参数 ----------
BAR_HOURS      = 4
HORIZON_DAYS   = 7
H_BARS         = (HORIZON_DAYS * 24) // BAR_HOURS  # 42
LEVERAGE       = 3.0
FEE_PER_SIDE   = 0.0005
ROUND_TRIP     = 2 * FEE_PER_SIDE
ANN_FACTOR     = 365 * 24 / BAR_HOURS

# ---------- 分位数搜索网格（可按需加宽或变细） ----------
Q_UP_GRID  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]      # 上涨概率的分位数
Q_DN_GRID  = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]      # 下跌概率的分位数
MARGIN_GRID= [0.00, 0.02, 0.05, 0.10]                  # 冲突安全带 (绝对差)

MODEL_KEYS = ["logreg", "linsvc", "gb"]

# ---------- 稳健性 ----------
MIN_TRADES_PER_FOLD = 5     # 每折最少交易数
GLOBAL_MIN_TRADES   = 30    # 总交易数下限（用于择优过滤）

def _load_data():
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(f"Missing {PRED_CSV}. Run Step 5 first.")
    if not os.path.exists(PREP_CSV):
        raise FileNotFoundError(f"Missing {PREP_CSV}.")
    pred = pd.read_csv(PRED_CSV)
    pred.columns = [c.strip().lower() for c in pred.columns]
    pred["time"] = pd.to_datetime(pred["time"], errors="coerce")

    base = pd.read_csv(PREP_CSV)
    base.columns = [c.strip().lower() for c in base.columns]
    ts_col = "timestamp" if "timestamp" in base.columns else ("open_time" if "open_time" in base.columns else base.columns[0])
    base[ts_col] = pd.to_datetime(base[ts_col], errors="coerce")
    base = base[[ts_col, "r_future_7d"]].rename(columns={ts_col:"time"})

    df = pred.merge(base, on="time", how="left")
    df = df.dropna(subset=["r_future_7d"]).sort_values(["fold_id", "time"]).reset_index(drop=True)
    return df

def _simulate_fold(df_fold, tau_up, tau_dn, margin, model_key):
    """
    非重叠持仓：触发后跳过 H_BARS 根；收益用 r_future_7d(i) 近似（3x 杠杆 & 手续费）
    """
    p_up = df_fold[f"p_up_{model_key}"].values
    p_dn = df_fold[f"p_dn_{model_key}"].values
    r7   = df_fold["r_future_7d"].values

    eq = 1.0
    path = [eq]
    trades = wins = 0
    i = 0
    n = len(df_fold)

    while i < n:
        long_sig  = (p_up[i] >= tau_up) and ( (p_dn[i] < tau_dn) or ((p_dn[i] >= tau_dn) and (p_up[i]-p_dn[i] > margin)) )
        short_sig = (p_dn[i] >= tau_dn) and ( (p_up[i] < tau_up) or ((p_up[i] >= tau_up) and (p_dn[i]-p_up[i] > margin)) )

        if long_sig and not short_sig:
            net = LEVERAGE * r7[i] - ROUND_TRIP
            eq *= (1.0 + net)
            trades += 1; wins += (net > 0)
            for _ in range(min(H_BARS, n - i)):
                path.append(eq)
            i += H_BARS
        elif short_sig and not long_sig:
            net = LEVERAGE * (-r7[i]) - ROUND_TRIP
            eq *= (1.0 + net)
            trades += 1; wins += (net > 0)
            for _ in range(min(H_BARS, n - i)):
                path.append(eq)
            i += H_BARS
        else:
            i += 1
            path.append(eq)

    ret = pd.Series(path).pct_change().dropna()
    mu = ret.mean() * ANN_FACTOR
    sd = ret.std(ddof=0) * np.sqrt(ANN_FACTOR)
    sharpe = (mu / sd) if sd and sd > 0 else np.nan
    win_rate = (wins / trades) if trades > 0 else np.nan

    return {"final_equity": float(eq), "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "trades": int(trades), "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan}

def _grid_search_quantile(df):
    rows = []
    for mk in MODEL_KEYS:
        for q_up in Q_UP_GRID:
            for q_dn in Q_DN_GRID:
                for m in MARGIN_GRID:
                    fold_stats = []
                    for fid, sub in df.groupby("fold_id"):
                        sub = sub.sort_values("time")
                        # 每个 fold 内，用该 fold 验证概率的分位数做阈值
                        tau_up = sub[f"p_up_{mk}"].quantile(q_up)
                        tau_dn = sub[f"p_dn_{mk}"].quantile(q_dn)
                        stats = _simulate_fold(sub, tau_up, tau_dn, m, mk)
                        if stats["trades"] >= MIN_TRADES_PER_FOLD:
                            fold_stats.append(stats)

                    if fold_stats:
                        median_sharpe = float(np.median([s["sharpe"] for s in fold_stats if pd.notna(s["sharpe"])])) if fold_stats else np.nan
                        geo_equity    = float(np.prod([s["final_equity"] for s in fold_stats]) ** (1.0 / len(fold_stats)))
                        avg_trades    = float(np.mean([s["trades"] for s in fold_stats]))
                        total_trades  = int(np.sum([s["trades"] for s in fold_stats]))
                        avg_win_rate  = float(np.mean([s["win_rate"] for s in fold_stats if pd.notna(s["win_rate"])])) if fold_stats else np.nan
                    else:
                        median_sharpe = np.nan; geo_equity = np.nan; avg_trades = 0.0; total_trades = 0; avg_win_rate = np.nan

                    rows.append({
                        "model": mk, "mode": "quantile",
                        "q_up": q_up, "q_dn": q_dn, "margin": m,
                        "median_sharpe": median_sharpe,
                        "geo_equity": geo_equity,
                        "avg_trades": avg_trades,
                        "total_trades": total_trades,
                        "avg_win_rate": avg_win_rate
                    })
    return pd.DataFrame(rows)

def _select_best(df_grid):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 写每模型的完整评分
    for mk in MODEL_KEYS:
        df_grid[df_grid["model"] == mk].to_csv(os.path.join(OUT_DIR, f"grid_scores_{mk}.csv"), index=False)

    # 全局最少交易过滤
    cand = df_grid[df_grid["total_trades"] >= GLOBAL_MIN_TRADES].copy()
    if cand.empty:
        cand = df_grid.copy()

    winners = []
    for mk in MODEL_KEYS:
        sub = cand[(cand["model"] == mk) & (cand["mode"] == "quantile")].copy()
        sub = sub.sort_values(by=["median_sharpe", "geo_equity", "avg_trades"],
                              ascending=[False, False, False])
        if len(sub):
            winners.append(sub.iloc[0].to_dict())

    best = pd.DataFrame(winners)
    if not best.empty:
        best.to_csv(os.path.join(OUT_DIR, "best_thresholds_per_model.csv"), index=False)
    return best

def run():
    df = _load_data()
    grid = _grid_search_quantile(df)
    best = _select_best(grid)

    print("="*72)
    print("[Step 6|Quantile] Grid search complete.")
    if not best.empty:
        print("Best per model (quantile mode):")
        print(best.to_string(index=False))
    print("Files written to:", OUT_DIR)
    print("="*72)

if __name__ == "__main__":
    run()
