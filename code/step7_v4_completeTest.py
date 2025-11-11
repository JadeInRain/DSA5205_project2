# -*- coding: utf-8 -*-
"""
Step 7 — Walk-Forward 回测（含补丁：逐bar盯市MTM + 自救阈值 + 详细进度日志）

- 使用 Step 6 的最佳模型与阈值；
- Test 段 + Holdout 段 逐bar走进式训练/校准/预测概率；
- 概率 -> 信号（tau_up, tau_dn, margin），非重叠持有 7 天（H_BARS）；
- 资金曲线改为“逐bar盯市”（按 close），不再是阶梯状；
- 交易=0 时自动尝试“自救阈值”（放宽 + 分位数）。

输出：
- data/signals/test_trades.csv / data/equity/test_equity.csv
- data/signals/holdout_trades.csv / data/equity/holdout_equity.csv
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as PlattScaler

# ------------------ 路径与窗口 ------------------
PREP_CSV    = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FOLDS_CSV   = os.path.join("data", "splits", "rolling_folds.csv")
HOLDOUT_CSV = os.path.join("data", "splits", "holdout.csv")
BEST_THR_CSV= os.path.join("data", "thresholds", "best_thresholds_per_model.csv")

BAR_HOURS   = 4
TRAIN_DAYS  = 365
VAL_DAYS    = 90
TRAIN_BARS  = (TRAIN_DAYS*24)//BAR_HOURS
VAL_BARS    = (VAL_DAYS*24)//BAR_HOURS
HORIZON_DAYS= 7
H_BARS      = (HORIZON_DAYS*24)//BAR_HOURS   # 42

# ------------------ 交易参数 ------------------
LEVERAGE     = 3.0
FEE_PER_SIDE = 0.0005              # 0.05%/边
ROUND_TRIP   = 2 * FEE_PER_SIDE     # 0.1% 往返
EXEC_PRICE_COL = "close"            # 统一按 close 执行与估值

# ------------------ 模型配置（与 Step5 一致） ------------------
LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000)
LSVC_CFG   = dict(C=1.0, class_weight="balanced")
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42)
RANDOM_STATE=42

TIMESTAMP_CANDIDATES = ["timestamp","open_time","time","datetime"]
PRICE_COLS = ["open","high","low","close","volume"]
LABEL_COLS = ["y_up","y_dn","r_future_7d"]
EXCLUDE_ALSO = ["future_close","next_open"]

MODEL_CHOICE = None  # 默认取 Step6 第一名；也可手动设 "gb"/"logreg"/"linsvc"

# ------------------ 基础工具 ------------------
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

def _load_base():
    df = pd.read_csv(PREP_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = _find_ts_col(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.sort_values(ts_col).reset_index(drop=True)
    if EXEC_PRICE_COL not in df.columns:
        raise ValueError(f"执行价列 {EXEC_PRICE_COL} 不存在，请在预处理阶段写入 close。")
    return df, ts_col

def _load_splits():
    folds = pd.read_csv(FOLDS_CSV)
    folds.columns = [c.strip().lower() for c in folds.columns]
    hold = pd.read_csv(HOLDOUT_CSV)
    hold.columns = [c.strip().lower() for c in hold.columns]
    return folds, hold

def _load_thresholds():
    best = pd.read_csv(BEST_THR_CSV)
    best.columns = [c.strip().lower() for c in best.columns]
    if "median_sharpe" in best.columns:
        best = best.sort_values(["median_sharpe","geo_equity"], ascending=[False,False]).reset_index(drop=True)
    elif "avg_sharpe" in best.columns:
        best = best.sort_values(["avg_sharpe","total_equity"], ascending=[False,False]).reset_index(drop=True)
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
    exclude = set([ts_col] + PRICE_COLS + LABEL_COLS + EXCLUDE_ALSO)
    exclude |= {c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    return [c for c in num_cols if c not in exclude]

def _fit_base_models(X_tr, y_tr):
    m1 = LogisticRegression(**LOGREG_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    m2 = LinearSVC(**LSVC_CFG, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    m3 = GradientBoostingClassifier(**GBC_CFG).fit(X_tr, y_tr)  # GBC_CFG 已含 random_state
    return m1, m2, m3

def _score_raw(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

def _platt(val_scores, y_val):
    s = np.asarray(val_scores).reshape(-1,1)
    y = np.asarray(y_val).astype(int)
    scaler = PlattScaler(C=1.0, solver="liblinear", max_iter=1000)
    scaler.fit(s, y)
    return scaler

def _get_model_by_key(m1,m2,m3,key):
    return {"logreg":m1, "linsvc":m2, "gb":m3}[key]

# ------------------ 概率（walk-forward） ------------------
def _walk_forward_probs(df: pd.DataFrame, ts_col: str, start_idx: int, end_idx: int, model_key: str,
                        progress_tag: str):
    feats = _select_features(df, ts_col)
    out = []
    n_total = max(0, end_idx - start_idx + 1)
    t0 = time.time()
    for k, t in enumerate(range(start_idx, end_idx+1), start=1):
        tr_start = t - (TRAIN_BARS + VAL_BARS)
        val_start = t - VAL_BARS
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
            # 数据不足，跳过该时点
            continue

        Xtr, Xva = X_tr[tr_mask], X_va[va_mask]
        yup_tr = y_up_tr[tr_mask].astype(int).values
        yup_va = y_up_va[va_mask].astype(int).values
        ydn_tr = y_dn_tr[tr_mask].astype(int).values
        ydn_va = y_dn_va[va_mask].astype(int).values

        # up 任务
        m1, m2, m3 = _fit_base_models(Xtr, yup_tr)
        s1, s2, s3 = _score_raw(m1, Xva), _score_raw(m2, Xva), _score_raw(m3, Xva)
        up_platts = {"logreg": _platt(s1, yup_va), "linsvc": _platt(s2, yup_va), "gb": _platt(s3, yup_va)}

        # dn 任务
        m1_, m2_, m3_ = _fit_base_models(Xtr, ydn_tr)
        d1, d2, d3 = _score_raw(m1_, Xva), _score_raw(m2_, Xva), _score_raw(m3_, Xva)
        dn_platts = {"logreg": _platt(d1, ydn_va), "linsvc": _platt(d2, ydn_va), "gb": _platt(d3, ydn_va)}

        # 当前 t 的特征
        x_t = df.loc[[t], feats].values
        up_model = _get_model_by_key(m1, m2, m3, model_key)
        dn_model = _get_model_by_key(m1_, m2_, m3_, model_key)
        up_raw = _score_raw(up_model, x_t)
        dn_raw = _score_raw(dn_model, x_t)
        p_up = float(up_platts[model_key].predict_proba(np.asarray(up_raw).reshape(-1,1))[:,1][0])
        p_dn = float(dn_platts[model_key].predict_proba(np.asarray(dn_raw).reshape(-1,1))[:,1][0])

        out.append({"idx": t, "time": df.loc[t, ts_col], "p_up": p_up, "p_dn": p_dn})

        # 进度
        if k % max(1, n_total//6 or 1) == 0:
            used = time.time() - t0
            pct = 100.0 * k / max(1, n_total)
            print(f"  - 进度 {pct:.1f}% | 当前 t={df.loc[t, ts_col]} | 已用 {used:.1f}s", flush=True)

    return pd.DataFrame(out)

# ------------------ 执行（逐bar盯市 MTM） ------------------
def _simulate_execution_mtm(df: pd.DataFrame, probs: pd.DataFrame, ts_col: str,
                            tau_up: float, tau_dn: float, margin: float):
    """
    逐bar盯市：
    - 在 t 时刻生成信号（基于 p_up/p_dn）；
    - 在 t+1 的 close 视为开仓价，并在接下来的 H_BARS 根按 close-to-close 实时盯市；
    - 入场当根扣 FEE_PER_SIDE，离场当根再扣 FEE_PER_SIDE；中间 bar 只按方向 * 收益 * LEVERAGE 计入；
    - 非重叠：持有期间忽略新信号。
    """
    if probs.empty:
        return pd.DataFrame(columns=["time","side","entry","exit","gross","net"]), \
               pd.DataFrame(columns=["time","equity"])

    # 将概率对齐到 base df
    d = df[[ts_col, EXEC_PRICE_COL]].rename(columns={EXEC_PRICE_COL: "close"}).copy()
    d = d.merge(probs, left_on=ts_col, right_on="time", how="right").sort_values(ts_col).reset_index(drop=True)
    # 为了取 t+1 / 出场 close，需要对齐一个移位数组
    d["next_close"] = d["close"].shift(-1)

    equity = 1.0
    eq_path = []
    trades = []
    i = 0
    n = len(d)

    while i < n:
        # 记录当前 bar 的 equity（plot 用）
        eq_path.append({"time": d.loc[i, ts_col], "equity": equity})

        # 生成多空信号
        p_up = d.loc[i, "p_up"]; p_dn = d.loc[i, "p_dn"]
        long_sig  = (p_up >= tau_up) and ( (p_dn < tau_dn) or ((p_dn >= tau_dn) and (p_up - p_dn >  margin)) )
        short_sig = (p_dn >= tau_dn) and ( (p_up < tau_up) or ((p_up >= tau_up) and (p_dn - p_up >  margin)) )

        if long_sig and not short_sig and pd.notna(d.loc[i, "next_close"]):
            entry = float(d.loc[i, "next_close"])   # 在下一根 close 入
            # 入场费
            equity *= (1.0 - FEE_PER_SIDE)

            # 逐bar盯市，方向 +1
            held = 0
            j = i + 1
            while j < n and held < H_BARS:
                # 当根收益（close-to-close）
                if pd.notna(d.loc[j-1, "close"]) and pd.notna(d.loc[j, "close"]):
                    r = (d.loc[j, "close"] / d.loc[j-1, "close"]) - 1.0
                    equity *= (1.0 + LEVERAGE * (+1.0) * r)
                eq_path.append({"time": d.loc[j, ts_col], "equity": equity})
                held += 1
                j += 1

            # 结束时再扣出场费
            equity *= (1.0 - FEE_PER_SIDE)

            exitp = float(d.loc[j-1, "close"]) if j-1 < n and pd.notna(d.loc[j-1, "close"]) else entry
            gross = (exitp/entry - 1.0) * LEVERAGE
            net   = (equity/1.0) - 1.0  # 此处仅用于记录；实际净值已在 eq_path 中逐bar体现

            trades.append({"time": d.loc[i, ts_col], "side": "long",
                           "entry": entry, "exit": exitp, "gross": gross, "net": net})
            i = j  # 非重叠：跳到持仓结束后一根
            continue

        if short_sig and not long_sig and pd.notna(d.loc[i, "next_close"]):
            entry = float(d.loc[i, "next_close"])
            equity *= (1.0 - FEE_PER_SIDE)

            held = 0
            j = i + 1
            while j < n and held < H_BARS:
                if pd.notna(d.loc[j-1, "close"]) and pd.notna(d.loc[j, "close"]):
                    r = (d.loc[j, "close"] / d.loc[j-1, "close"]) - 1.0
                    equity *= (1.0 + LEVERAGE * (-1.0) * r)
                eq_path.append({"time": d.loc[j, ts_col], "equity": equity})
                held += 1
                j += 1

            equity *= (1.0 - FEE_PER_SIDE)
            exitp = float(d.loc[j-1, "close"]) if j-1 < n and pd.notna(d.loc[j-1, "close"]) else entry
            gross = (entry/exitp - 1.0) * LEVERAGE
            net   = (equity/1.0) - 1.0

            trades.append({"time": d.loc[i, ts_col], "side": "short",
                           "entry": entry, "exit": exitp, "gross": gross, "net": net})
            i = j
            continue

        # 无信号：前进一步
        i += 1

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(eq_path).drop_duplicates(subset=["time"], keep="last")
    return trades_df, equity_df

# ------------------ 辅助：自救阈值 ------------------
def _self_rescue_thresholds(probs: pd.DataFrame, tau_up: float, tau_dn: float, margin: float):
    """
    当交易数=0时：
    1) 连续 5 次放宽（每次 up-0.02, dn-0.02, margin->0）
    2) 仍无交易：回退到分位数阈值（up: 65% 分位；dn: 65% 分位），margin=0
    """
    tries = []
    for k in range(1, 6):
        tries.append((max(0.0, tau_up - 0.02*k), max(0.0, tau_dn - 0.02*k), 0.0))

    # 最后一次：分位数
    if not probs.empty:
        q_up = float(probs["p_up"].quantile(0.65))
        q_dn = float(probs["p_dn"].quantile(0.65))
        tries.append((q_up, q_dn, 0.0))
    return tries

# ------------------ 段落索引 ------------------
def _segment_indices(df: pd.DataFrame, ts_col: str, folds: pd.DataFrame, hold: pd.DataFrame):
    # Test: 最后一个验证折的 end_time 之后，直到 holdout.start 前一根
    last_val_end = pd.to_datetime(folds[folds["segment"]=="val"]["end_time"].max())
    hstart = pd.to_datetime(hold["start_time"].iloc[0]); hend = pd.to_datetime(hold["end_time"].iloc[0])

    test_mask = (df[ts_col] > last_val_end) & (df[ts_col] < hstart)
    hold_mask = (df[ts_col] >= hstart) & (df[ts_col] <= hend)

    ts_idx = df.index[test_mask]
    ho_idx = df.index[hold_mask]

    test_span = (int(ts_idx[0]), int(ts_idx[-1])) if len(ts_idx)>0 else (None, None)
    hold_span = (int(ho_idx[0]), int(ho_idx[-1]))
    return test_span, hold_span

# ------------------ 主流程 ------------------
def _run_one_segment(name: str, df: pd.DataFrame, ts_col: str,
                     start_idx: int, end_idx: int,
                     model_key: str, tau_up: float, tau_dn: float, margin: float):
    if start_idx is None or end_idx is None or start_idx >= end_idx:
        print(f"[Step7][{name}] 段过短或不存在，跳过。")
        return pd.DataFrame(), pd.DataFrame()

    print(f"[Step7] Walk-forward 概率计算：{df.loc[start_idx, ts_col]} → {df.loc[end_idx, ts_col]}  (步长=1)")
    t0 = time.time()
    probs = _walk_forward_probs(df, ts_col, start_idx, end_idx, model_key, progress_tag=name)
    print(f"[Step7] Walk-forward 概率完成：共 {len(probs)} 条（总候选 {end_idx-start_idx+1} 条），耗时 {time.time()-t0:.1f}s")

    print("[Step7] 执行仿真（非重叠7天持有，逐bar盯市 MTM）…")
    trades, equity = _simulate_execution_mtm(df, probs, ts_col, tau_up, tau_dn, margin)
    print(f"[Step7] 执行仿真完成：交易 {len(trades)} 笔，资金曲线点数 {len(equity)}")

    # 自救：若 0 笔交易，尝试放宽阈值/分位数阈值
    if len(trades) == 0 and not probs.empty:
        print(f"[Step7][{name}] 0 笔交易，进入自救策略 …")
        for i, (tu, td, m) in enumerate(_self_rescue_thresholds(probs, tau_up, tau_dn, margin), start=1):
            print(f"[Step7][自救] 第{i}次：tau_up={tu:.3f}, tau_dn={td:.3f}, margin={m:.2f}")
            trades2, equity2 = _simulate_execution_mtm(df, probs, ts_col, tu, td, m)
            print(f"[Step7] 执行仿真完成：交易 {len(trades2)} 笔，资金曲线点数 {len(equity2)}")
            if len(trades2) > 0:
                print(f"[Step7][{name}] 自救成功：采用新阈值 tu={tu:.6f}, td={td:.6f}, margin={m:.2f}")
                trades, equity = trades2, equity2
                break

    # 落盘
    os.makedirs(os.path.join("data", "signals"), exist_ok=True)
    os.makedirs(os.path.join("data", "equity"), exist_ok=True)
    trades.to_csv(os.path.join("data", "signals", f"{name}_trades.csv"), index=False)
    equity.to_csv(os.path.join("data", "equity", f"{name}_equity.csv"), index=False)
    print(f"[Step7] {name} 输出：data/signals/{name}_trades.csv, data/equity/{name}_equity.csv")
    return trades, equity

def run():
    # 读取数据
    print("[Step7] 读取预处理数据 …")
    df, ts_col = _load_base()
    folds, hold = _load_splits()
    best = _load_thresholds()
    model_key, tau_up, tau_dn, margin = _choose_model_and_thresholds(best)

    print(f"[Step 7] 使用模型={model_key}，阈值：tau_up={tau_up}, tau_dn={tau_dn}, margin={margin}")
    print(f"[Step 7] 执行价/估值采用列：{EXEC_PRICE_COL}")

    # 切分段
    (ts_start, ts_end), (ho_start, ho_end) = _segment_indices(df, ts_col, folds, hold)
    if ts_start is not None:
        print(f"[Step7] Test 段：{df.loc[ts_start, ts_col]} → {df.loc[ts_end, ts_col]}")
    else:
        print("[Step7] Test 段：无（最后一个验证折到 Holdout 之间没有足够数据）")
    print(f"[Step7] Holdout 段：{df.loc[ho_start, ts_col]} → {df.loc[ho_end, ts_col]}")

    # Test
    _run_one_segment("test", df, ts_col, ts_start, ts_end, model_key, tau_up, tau_dn, margin)

    # Holdout
    _run_one_segment("holdout", df, ts_col, ho_start, ho_end, model_key, tau_up, tau_dn, margin)

    print("="*72)
    print("[Step 7] 完成。下一步 Step 8：汇总指标、对比 Buy&Hold 并画图。")
    print("="*72)

if __name__ == "__main__":
    run()
