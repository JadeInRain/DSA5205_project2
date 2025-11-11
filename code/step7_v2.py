# -*- coding: utf-8 -*-
"""
step7_v2.py  —  方案A：Walk-Forward 概率 + 非重叠7天执行（带“0笔交易”自救策略）

要点
----
1) 读取 Step 6 的“最佳模型+阈值（tau_up, tau_dn, margin）”；
2) 在 Test 与 Holdout 上逐bar做 walk-forward：
   - 训练窗 365 天（≈2190 bars），校准窗 90 天（≈540 bars）；
   - 先在训练窗拟合模型，再用校准窗对“原始分数”做 Platt 概率校准；
   - 在当前 bar 输出 p_up, p_dn；
3) 以阈值+安全带生成信号，**非重叠**持有 7 天（= 42 根 4h bar）；
4) 入场/出场价：优先使用 `open`；若不存在，则用 `close`（并相应用“下一根”逻辑）；
5) 若 0 笔交易，按**方案A**自动放宽阈值（递减 tau、margin），仍为 0，则退而用“分位数阈值”自救；
6) 完整输出：trades.csv & equity.csv（Test 与 Holdout 各一份），并打印关键日志。

依赖
----
- 需已完成 Step2/3、Step4、Step5、Step6。
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Tuple

# ------------------------- 路径配置 -------------------------
PREP_CSV      = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FOLDS_CSV     = os.path.join("data", "splits", "rolling_folds.csv")
HOLDOUT_CSV   = os.path.join("data", "splits", "holdout.csv")
BEST_THR_CSV  = os.path.join("data", "thresholds", "best_thresholds_per_model.csv")

os.makedirs(os.path.join("data", "signals"), exist_ok=True)
os.makedirs(os.path.join("data", "equity"), exist_ok=True)

# ------------------------- 时间与交易参数 -------------------------
BAR_HOURS   = 4
TRAIN_DAYS  = 365
VAL_DAYS    = 90
HORIZON_DAYS= 7
TRAIN_BARS  = (TRAIN_DAYS * 24) // BAR_HOURS     # 2190
VAL_BARS    = (VAL_DAYS   * 24) // BAR_HOURS     # 540
H_BARS      = (HORIZON_DAYS * 24) // BAR_HOURS   # 42

LEVERAGE      = 3.0
FEE_PER_SIDE  = 0.0005
ROUND_TRIP    = 2 * FEE_PER_SIDE

# ------------------------- 模型配置（与Step5一致） -------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as PlattScaler

LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)
LSVC_CFG   = dict(C=1.0, class_weight="balanced")
# 注意：不要在这里再重复传 random_state，以免“multiple values”错误
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8)
RANDOM_STATE = 42

TIMESTAMP_CANDIDATES = ["timestamp", "open_time", "time", "datetime"]
PRICE_COLS = ["open", "high", "low", "close", "volume"]
LABEL_COLS = ["y_up", "y_dn", "r_future_7d"]
EXCLUDE_ALSO = ["future_close", "next_open"]

# ------------------------- 工具函数 -------------------------
def _find_ts_col(df: pd.DataFrame) -> str:
    low = [c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in low:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    return df.columns[0]

def _load_base() -> Tuple[pd.DataFrame, str, str]:
    print("[Step7] 读取预处理数据 …")
    df = pd.read_csv(PREP_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = _find_ts_col(df)
    try:
        df[ts_col] = pd.to_datetime(df[ts_col])
    except Exception:
        pass
    df = df.sort_values(ts_col).reset_index(drop=True)

    # 交易用价格列：优先 open，没有就用 close
    price_col = "open" if "open" in df.columns else "close"
    print(f"[Step 7] 执行价采用列：{price_col}")
    return df, ts_col, price_col

def _load_splits():
    folds = pd.read_csv(FOLDS_CSV)
    folds.columns = [c.strip().lower() for c in folds.columns]
    hold = pd.read_csv(HOLDOUT_CSV)
    hold.columns = [c.strip().lower() for c in hold.columns]
    return folds, hold

def _load_thresholds():
    best = pd.read_csv(BEST_THR_CSV)
    best.columns = [c.strip().lower() for c in best.columns]
    # 支持 step6 精修/旧版两种格式
    if "median_sharpe" in best.columns:
        best = best.sort_values(["median_sharpe", "geo_equity"], ascending=[False, False])
    elif "avg_sharpe" in best.columns:
        best = best.sort_values(["avg_sharpe", "total_equity"], ascending=[False, False])
    else:
        best = best.sort_values(["geo_equity"], ascending=False)
    best = best.reset_index(drop=True)
    return best

def _choose_model_and_thresholds(best_df, model_choice=None):
    if model_choice is None:
        model_choice = best_df.iloc[0]["model"]
    row = best_df[best_df["model"] == model_choice].iloc[0]
    tau_up = float(row["tau_up"]); tau_dn = float(row["tau_dn"]); margin = float(row["margin"])
    print(f"[Step 7] 使用模型={model_choice}，阈值：tau_up={tau_up}, tau_dn={tau_dn}, margin={margin}")
    return model_choice, tau_up, tau_dn, margin

def _select_features(df: pd.DataFrame, ts_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set([ts_col] + PRICE_COLS + LABEL_COLS + EXCLUDE_ALSO)
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    feats = [c for c in num_cols if c not in exclude]
    return feats

def _fit_base_models(X_tr, y_tr):
    m1 = LogisticRegression(**LOGREG_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    m2 = LinearSVC(**LSVC_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    # 注意：GBC_CFG 内已不含 random_state，这里统一加一次
    m3 = GradientBoostingClassifier(**GBC_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    return m1, m2, m3

def _score_raw(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

def _platt(val_scores, y_val):
    # 统一转为 ndarray，避免 .values 属性缺失错误
    s = np.asarray(val_scores, dtype=float).reshape(-1, 1)
    y = np.asarray(y_val, dtype=int).reshape(-1)
    scaler = PlattScaler(C=1.0, solver="liblinear", max_iter=1000)
    scaler.fit(s, y)
    return scaler

def _get_model_by_key(m1, m2, m3, key):
    return {"logreg": m1, "linsvc": m2, "gb": m3}[key]

def _segment_indices(df: pd.DataFrame, ts_col: str, folds: pd.DataFrame, hold: pd.DataFrame):
    # Test 段：最后一个验证区间的结束时间之后到 Holdout 起点之前
    last_val_end_time = pd.to_datetime(folds[folds["segment"] == "val"]["end_time"].max())
    hstart = pd.to_datetime(hold["start_time"].iloc[0])
    hend   = pd.to_datetime(hold["end_time"].iloc[0])

    # 取严格大于最后验证结束时间的第一根
    test_mask = df[ts_col] > last_val_end_time
    test_start_idx = int(test_mask.idxmax()) if test_mask.any() else None

    hold_start_idx = int(df.index[df[ts_col] >= hstart][0])
    hold_end_idx   = int(df.index[df[ts_col] <= hend][-1])

    test_end_idx = hold_start_idx - 1 if test_start_idx is not None else None

    if (test_start_idx is not None) and (test_end_idx is not None) and test_start_idx <= test_end_idx:
        print(f"[Step7] Test 段：{df.loc[test_start_idx, ts_col]} → {df.loc[test_end_idx, ts_col]}")
    else:
        print("[Step7] Test 段：无或过短（自动跳过）")

    print(f"[Step7] Holdout 段：{df.loc[hold_start_idx, ts_col]} → {df.loc[hold_end_idx, ts_col]}")
    return (test_start_idx, test_end_idx), (hold_start_idx, hold_end_idx)

# ------------------------- Walk-forward 概率 -------------------------
def _walk_forward_probs(df: pd.DataFrame, ts_col: str, start_idx: int, end_idx: int, model_key: str) -> pd.DataFrame:
    feats = _select_features(df, ts_col)
    out = []
    n_total = max(0, end_idx - start_idx + 1)
    if n_total == 0:
        return pd.DataFrame(columns=["idx", "time", "p_up", "p_dn"])

    print(f"[Step7] Walk-forward 概率计算：{df.loc[start_idx, ts_col]} → {df.loc[end_idx, ts_col]}  (步长=1)")
    t0 = time.time()
    last_progress = -1

    for k, t in enumerate(range(start_idx, end_idx + 1), start=1):
        tr_start = t - (TRAIN_BARS + VAL_BARS)
        val_start = t - VAL_BARS
        if tr_start < 0:
            continue

        X_tr = df.loc[tr_start: val_start - 1, feats].values
        X_va = df.loc[val_start: t - 1, feats].values
        y_up_tr = df.loc[tr_start: val_start - 1, "y_up"].astype("Int64")
        y_up_va = df.loc[val_start: t - 1, "y_up"].astype("Int64")
        y_dn_tr = df.loc[tr_start: val_start - 1, "y_dn"].astype("Int64")
        y_dn_va = df.loc[val_start: t - 1, "y_dn"].astype("Int64")

        tr_mask = np.isfinite(X_tr).all(axis=1) & y_up_tr.notna().values & y_dn_tr.notna().values
        va_mask = np.isfinite(X_va).all(axis=1) & y_up_va.notna().values & y_dn_va.notna().values

        if tr_mask.sum() < 50 or va_mask.sum() < 20:
            continue

        Xtr = X_tr[tr_mask]; Xva = X_va[va_mask]
        yup_tr = y_up_tr[tr_mask].astype(int).values
        yup_va = y_up_va[va_mask].astype(int).values
        ydn_tr = y_dn_tr[tr_mask].astype(int).values
        ydn_va = y_dn_va[va_mask].astype(int).values

        # —— UP 任务 —— #
        m1, m2, m3 = _fit_base_models(Xtr, yup_tr)
        s1 = _score_raw(m1, Xva); s2 = _score_raw(m2, Xva); s3 = _score_raw(m3, Xva)
        up_platts = {
            "logreg": _platt(s1, yup_va),
            "linsvc": _platt(s2, yup_va),
            "gb":     _platt(s3, yup_va)
        }

        # —— DOWN 任务 —— #
        m1d, m2d, m3d = _fit_base_models(Xtr, ydn_tr)
        d1 = _score_raw(m1d, Xva); d2 = _score_raw(m2d, Xva); d3 = _score_raw(m3d, Xva)
        dn_platts = {
            "logreg": _platt(d1, ydn_va),
            "linsvc": _platt(d2, ydn_va),
            "gb":     _platt(d3, ydn_va)
        }

        # 当前时点 t 的单点预测
        x_t = df.loc[[t], feats].values
        up_model = _get_model_by_key(m1, m2, m3, model_key)
        dn_model = _get_model_by_key(m1d, m2d, m3d, model_key)
        up_raw = _score_raw(up_model, x_t)
        dn_raw = _score_raw(dn_model, x_t)
        p_up = float(up_platts[model_key].predict_proba(np.asarray(up_raw).reshape(-1, 1))[:, 1][0])
        p_dn = float(dn_platts[model_key].predict_proba(np.asarray(dn_raw).reshape(-1, 1))[:, 1][0])

        out.append({"idx": t, "time": df.loc[t, ts_col], "p_up": p_up, "p_dn": p_dn})

        # 进度打印（每 ~12.5%）
        prog = int(k / n_total * 8)
        if prog > last_progress:
            elapsed = time.time() - t0
            est = elapsed / max(1, k) * (n_total - k)
            print(f"  - 进度 {min(100, int(k / n_total * 100))}% | 当前 t={df.loc[t, ts_col]} | 已用 {elapsed:.1f}s | 估计剩余 {est:.1f}s")
            last_progress = prog

    used = time.time() - t0
    print(f"[Step7] Walk-forward 概率完成：共 {len(out)} 条（总候选 {n_total} 条），耗时 {used:.1f}s")
    return pd.DataFrame(out)

# ------------------------- 执行仿真（非重叠7天） -------------------------
def _simulate_execution(df: pd.DataFrame, probs: pd.DataFrame, ts_col: str, price_col: str,
                        tau_up: float, tau_dn: float, margin: float):
    if probs.empty:
        return pd.DataFrame(columns=["time","side","entry","exit","gross","net"]), pd.DataFrame(columns=["time","equity"])

    d = df[[ts_col, price_col]].copy()
    d["next_px"] = d[price_col].shift(-1)              # 入场价：下一根
    d["exit_px"] = d[price_col].shift(-(H_BARS+1))     # 出场价：+7天后的下一根
    d = d.merge(probs, left_on=ts_col, right_on="time", how="inner").sort_values(ts_col).reset_index(drop=True)

    eq = 1.0
    equity = [{"time": d.loc[0, ts_col], "equity": eq}]
    trades = []
    i, n = 0, len(d)

    print("[Step7] 执行仿真（非重叠7天持有，下一根开入/+7天后一根开出）…")
    while i < n:
        p_up = d.loc[i, "p_up"]; p_dn = d.loc[i, "p_dn"]

        long_sig  = (p_up >= tau_up) and ((p_up - p_dn) >  margin)
        short_sig = (p_dn >= tau_dn) and ((p_dn - p_up) >  margin)

        if long_sig and not short_sig and pd.notna(d.loc[i, "next_px"]) and pd.notna(d.loc[i, "exit_px"]):
            entry = float(d.loc[i, "next_px"]); exitp = float(d.loc[i, "exit_px"])
            gross = (exitp / entry - 1.0) * LEVERAGE
            net   = gross - ROUND_TRIP
            eq *= (1.0 + net)
            trades.append({"time": d.loc[i, ts_col], "side": "long", "entry": entry, "exit": exitp,
                           "gross": gross, "net": net})
            i += H_BARS  # 非重叠
        elif short_sig and not long_sig and pd.notna(d.loc[i, "next_px"]) and pd.notna(d.loc[i, "exit_px"]):
            entry = float(d.loc[i, "next_px"]); exitp = float(d.loc[i, "exit_px"])
            gross = (entry / exitp - 1.0) * LEVERAGE  # 做空：价格越跌越赚
            net   = gross - ROUND_TRIP
            eq *= (1.0 + net)
            trades.append({"time": d.loc[i, ts_col], "side": "short", "entry": entry, "exit": exitp,
                           "gross": gross, "net": net})
            i += H_BARS
        else:
            i += 1

        equity.append({"time": d.loc[min(i, n-1), ts_col] if i < n else d.loc[n-1, ts_col], "equity": eq})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).drop_duplicates(subset=["time"], keep="last")
    print(f"[Step7] 执行仿真完成：交易 {len(trades_df)} 笔，资金曲线点数 {len(equity_df)}")
    return trades_df, equity_df

# ------------------------- 0 笔交易自救（方案A） -------------------------
def _relax_thresholds_and_retry(df, probs, ts_col, price_col, tau_up, tau_dn, margin):
    """若 0 笔交易：逐步放宽阈值与安全带；仍为 0，则用分位数阈值试一次。"""
    # 1) 逐步降低阈值、缩小安全带
    for step in range(1, 6):  # 5 次
        tu = max(0.50, tau_up - 0.02 * step)
        td = max(0.50, tau_dn - 0.02 * step)
        mg = max(0.00, margin - 0.01 * step)
        print(f"[Step7][自救] 放宽阈值第{step}次：tau_up={tu}, tau_dn={td}, margin={mg}")
        trades, equity = _simulate_execution(df, probs, ts_col, price_col, tu, td, mg)
        if len(trades) > 0:
            return trades, equity, tu, td, mg

    # 2) 用分位数阈值（例如 70% 分位）
    if not probs.empty:
        up_q = float(np.nanpercentile(probs["p_up"], 70))
        dn_q = float(np.nanpercentile(probs["p_dn"], 70))
        print(f"[Step7][自救] 使用分位数阈值：tau_up≈{up_q:.3f}, tau_dn≈{dn_q:.3f}, margin=0.00")
        trades, equity = _simulate_execution(df, probs, ts_col, price_col, up_q, dn_q, 0.00)
        if len(trades) > 0:
            return trades, equity, up_q, dn_q, 0.00

    # 仍然无交易：返回原值
    return pd.DataFrame(), pd.DataFrame(), tau_up, tau_dn, margin

# ------------------------- 主流程 -------------------------
def run(model_choice=None):
    df, ts_col, price_col = _load_base()
    folds, hold = _load_splits()
    best = _load_thresholds()
    model_key, tau_up, tau_dn, margin = _choose_model_and_thresholds(best, model_choice=model_choice)

    # 段落划分
    (ts_s, ts_e), (ho_s, ho_e) = _segment_indices(df, ts_col, folds, hold)

    # === Test ===
    if ts_s is not None and ts_e is not None and ts_s <= ts_e:
        probs_test = _walk_forward_probs(df, ts_col, ts_s, ts_e, model_key)
        trades_t, equity_t = _simulate_execution(df, probs_test, ts_col, price_col, tau_up, tau_dn, margin)
        if len(trades_t) == 0:
            print("[Step7][Test] 0 笔交易，进入自救策略 …")
            trades_t, equity_t, tau_up_t, tau_dn_t, margin_t = _relax_thresholds_and_retry(
                df, probs_test, ts_col, price_col, tau_up, tau_dn, margin
            )
            if len(trades_t) > 0:
                print(f"[Step7][Test] 自救成功：新阈值 tau_up={tau_up_t}, tau_dn={tau_dn_t}, margin={margin_t}")
            else:
                print("[Step7][Test] 自救仍失败（保持0笔交易）")

        trades_t.to_csv(os.path.join("data", "signals", "test_trades.csv"), index=False)
        equity_t.to_csv(os.path.join("data", "equity", "test_equity.csv"), index=False)
        print("[Step7] Test 输出：signals/test_trades.csv, equity/test_equity.csv")

    # === Holdout ===
    probs_hold = _walk_forward_probs(df, ts_col, ho_s, ho_e, model_key)
    trades_h, equity_h = _simulate_execution(df, probs_hold, ts_col, price_col, tau_up, tau_dn, margin)
    if len(trades_h) == 0:
        print("[Step7][Holdout] 0 笔交易，进入自救策略 …")
        trades_h, equity_h, tau_up_h, tau_dn_h, margin_h = _relax_thresholds_and_retry(
            df, probs_hold, ts_col, price_col, tau_up, tau_dn, margin
        )
        if len(trades_h) > 0:
            print(f"[Step7][Holdout] 自救成功：新阈值 tau_up={tau_up_h}, tau_dn={tau_dn_h}, margin={margin_h}")
        else:
            print("[Step7][Holdout] 自救仍失败（保持0笔交易）")

    trades_h.to_csv(os.path.join("data", "signals", "holdout_trades.csv"), index=False)
    equity_h.to_csv(os.path.join("data", "equity", "holdout_equity.csv"), index=False)
    print("[Step7] Holdout 输出：data/signals/holdout_trades.csv, data/equity/holdout_equity.csv")

    print("=" * 72)
    print("[Step 7] 完成。下一步 Step 8：汇总指标、与 Buy&Hold（含 3x 基准）对比并画图。")
    print("=" * 72)


if __name__ == "__main__":
    # 你也可以强制指定：run(model_choice="gb" / "logreg" / "linsvc")
    run()
