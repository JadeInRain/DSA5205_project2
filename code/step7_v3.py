# -*- coding: utf-8 -*-
"""
step7_v1.py — 双阈值模式：absolute / quantile
---------------------------------------------------
- 读取 Step 6 的最佳“阈值方案”
  - 若包含列 q_up/q_dn → 采用 **分位数阈值模式** (quantile)
  - 否则读取 tau_up/tau_dn → **绝对阈值模式** (absolute)
- 概率获取：walk-forward（训练窗 365 天 + 校准窗 90 天）
- 生成信号：按模式转换（分位数模式采用滚动分位数阈值，严格 shift(1) 防泄漏）
- 执行：非重叠 7 天持有，下一根入、+7天后一根出，3x 杠杆，0.05%/边
- 输出：Test & Holdout 的 trades / equity

输入:
- data/prepared/btc_4h_preprocessed.csv
- data/splits/rolling_folds.csv
- data/splits/holdout.csv
- data/thresholds/best_thresholds_per_model.csv
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

# ---------- 路径 ----------
PREP_CSV     = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FOLDS_CSV    = os.path.join("data", "splits", "rolling_folds.csv")
HOLDOUT_CSV  = os.path.join("data", "splits", "holdout.csv")
BEST_THR_CSV = os.path.join("data", "thresholds", "best_thresholds_per_model.csv")

# ---------- 时间窗 ----------
BAR_HOURS   = 4
TRAIN_DAYS  = 365
VAL_DAYS    = 90
TRAIN_BARS  = (TRAIN_DAYS*24)//BAR_HOURS
VAL_BARS    = (VAL_DAYS*24)//BAR_HOURS
HORIZON_DAYS= 7
H_BARS      = (HORIZON_DAYS*24)//BAR_HOURS  # 42

# ---------- 交易参数 ----------
LEVERAGE     = 3.0
FEE_PER_SIDE = 0.0005
ROUND_TRIP   = 2*FEE_PER_SIDE

# ---------- 模型配置（与 Step 5 保持一致；避免 duplicate random_state） ----------
LOGREG_CFG = dict(C=1.0, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000, random_state=42)
LSVC_CFG   = dict(C=1.0, class_weight="balanced", random_state=42)
GBC_CFG    = dict(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=42)

TIMESTAMP_CANDIDATES = ["timestamp","open_time","time","datetime"]
PRICE_COLS = ["open","high","low","close","volume"]
LABEL_COLS = ["y_up","y_dn","r_future_7d"]
EXCLUDE_ALSO = ["future_close","next_open"]

# 你可手动指定模型；默认读取 Step 6 第一名
MODEL_CHOICE = None  # "gb" / "logreg" / "linsvc"

def _find_ts_col(df: pd.DataFrame) -> str:
    cols=[c.lower() for c in df.columns]
    for c in TIMESTAMP_CANDIDATES:
        if c in cols: return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception: pass
    return df.columns[0]

def _load_base():
    df = pd.read_csv(PREP_CSV)
    df.columns=[c.strip().lower() for c in df.columns]
    ts_col=_find_ts_col(df)
    df[ts_col]=pd.to_datetime(df[ts_col], errors="coerce")
    df=df.sort_values(ts_col).reset_index(drop=True)
    # 选择执行价列：优先 open；没有则用 close
    price_col = "open" if "open" in df.columns else "close"
    return df, ts_col, price_col

def _load_splits():
    folds = pd.read_csv(FOLDS_CSV); folds.columns=[c.strip().lower() for c in folds.columns]
    hold  = pd.read_csv(HOLDOUT_CSV); hold.columns=[c.strip().lower() for c in hold.columns]
    return folds, hold

def _load_thresholds():
    best = pd.read_csv(BEST_THR_CSV)
    best.columns=[c.strip().lower() for c in best.columns]
    # 优先 quantile 模式
    is_quantile = ("q_up" in best.columns) and ("q_dn" in best.columns)
    # 排序优先级
    if "median_sharpe" in best.columns:
        best = best.sort_values(["median_sharpe","geo_equity","avg_trades"], ascending=[False,False,False])
    elif "avg_sharpe" in best.columns:
        best = best.sort_values(["avg_sharpe","total_equity"], ascending=[False,False])
    else:
        best = best.sort_values(["geo_equity" if "geo_equity" in best.columns else "total_equity"], ascending=False)
    return best.reset_index(drop=True), is_quantile

def _choose_model(best_df):
    global MODEL_CHOICE
    if MODEL_CHOICE is None:
        MODEL_CHOICE = best_df.iloc[0]["model"]
    row = best_df[best_df["model"]==MODEL_CHOICE].iloc[0]
    return row

def _select_features(df: pd.DataFrame, ts_col: str) -> List[str]:
    num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    exclude=set([ts_col]+PRICE_COLS+LABEL_COLS+EXCLUDE_ALSO)
    exclude|={c for c in num_cols if c.startswith("future_") or c.startswith("label_")}
    return [c for c in num_cols if c not in exclude]

def _fit_three(X, y):
    m1 = LogisticRegression(**LOGREG_CFG).fit(X, y)
    m2 = LinearSVC(**LSVC_CFG).fit(X, y)
    m3 = GradientBoostingClassifier(**GBC_CFG).fit(X, y)
    return m1,m2,m3

def _score_raw(model, X):
    if hasattr(model,"predict_proba"):
        return model.predict_proba(X)[:,1]
    elif hasattr(model,"decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

def _platt(scores, y):
    s = np.asarray(scores).reshape(-1,1)
    y = np.asarray(y).astype(int).ravel()
    scaler = PlattScaler(C=1.0, solver="liblinear", max_iter=1000)
    scaler.fit(s, y)
    return scaler

def _get(m1,m2,m3,key):
    return {"logreg":m1,"linsvc":m2,"gb":m3}[key]

def _walk_forward_probs(df: pd.DataFrame, ts_col: str, start_idx: int, end_idx: int, model_key: str, progress_name: str):
    feats=_select_features(df, ts_col)
    out=[]
    t0=time.time()
    total=end_idx-start_idx+1
    for i,t in enumerate(range(start_idx, end_idx+1), 1):
        tr_start=t-(TRAIN_BARS+VAL_BARS); val_start=t-VAL_BARS
        if tr_start<0: continue
        X_tr=df.loc[tr_start:val_start-1, feats].values
        X_va=df.loc[val_start:t-1, feats].values
        y_up_tr=df.loc[tr_start:val_start-1,"y_up"].astype("Int64")
        y_up_va=df.loc[val_start:t-1,"y_up"].astype("Int64")
        y_dn_tr=df.loc[tr_start:val_start-1,"y_dn"].astype("Int64")
        y_dn_va=df.loc[val_start:t-1,"y_dn"].astype("Int64")

        tr_mask=np.isfinite(X_tr).all(axis=1)&y_up_tr.notna().values&y_dn_tr.notna().values
        va_mask=np.isfinite(X_va).all(axis=1)&y_up_va.notna().values&y_dn_va.notna().values
        if tr_mask.sum()<50 or va_mask.sum()<20:
            continue

        Xtr=X_tr[tr_mask]; Xva=X_va[va_mask]
        yup_tr=y_up_tr[tr_mask].astype(int).values
        yup_va=y_up_va[va_mask].astype(int).values
        ydn_tr=y_dn_tr[tr_mask].astype(int).values
        ydn_va=y_dn_va[va_mask].astype(int).values

        # up 任务
        up_m1,up_m2,up_m3=_fit_three(Xtr, yup_tr)
        up_s1=_score_raw(up_m1, Xva); up_s2=_score_raw(up_m2, Xva); up_s3=_score_raw(up_m3, Xva)
        up_platts={"logreg":_platt(up_s1,yup_va),"linsvc":_platt(up_s2,yup_va),"gb":_platt(up_s3,yup_va)}

        # dn 任务
        dn_m1,dn_m2,dn_m3=_fit_three(Xtr, ydn_tr)
        dn_s1=_score_raw(dn_m1, Xva); dn_s2=_score_raw(dn_m2, Xva); dn_s3=_score_raw(dn_m3, Xva)
        dn_platts={"logreg":_platt(dn_s1,ydn_va),"linsvc":_platt(dn_s2,ydn_va),"gb":_platt(dn_s3,ydn_va)}

        # bar t 概率
        x_t=df.loc[[t],feats].values
        up_raw=_score_raw(_get(up_m1,up_m2,up_m3,model_key), x_t)
        dn_raw=_score_raw(_get(dn_m1,dn_m2,dn_m3,model_key), x_t)
        p_up=float(up_platts[model_key].predict_proba(np.asarray(up_raw).reshape(-1,1))[:,1][0])
        p_dn=float(dn_platts[model_key].predict_proba(np.asarray(dn_raw).reshape(-1,1))[:,1][0])

        out.append({"idx":t, "time":df.loc[t,ts_col], "p_up":p_up, "p_dn":p_dn})

        # 进度条
        if i==1 or i==total or i%int(max(1,total/5))==0:
            elapsed=time.time()-t0
            pct=100.0*i/total
            eta=elapsed*(total/i-1)
            print(f"  - {progress_name} 进度 {pct:.1f}% | 当前 t={df.loc[t,ts_col]} | 已用 {elapsed:.1f}s | 估计剩余 {eta:.1f}s")

    return pd.DataFrame(out)

def _segment_indices(df: pd.DataFrame, ts_col: str, folds: pd.DataFrame, hold: pd.DataFrame):
    last_val_end = pd.to_datetime(folds[folds["segment"]=="val"]["end_time"].max())
    hstart = pd.to_datetime(hold["start_time"].iloc[0]); hend = pd.to_datetime(hold["end_time"].iloc[0])

    test_start_idx = df.index[df[ts_col] > last_val_end][0] if (df[ts_col] > last_val_end).any() else None
    hold_start_idx = df.index[df[ts_col] >= hstart][0]
    hold_end_idx   = df.index[df[ts_col] <= hend][-1]
    test_end_idx   = hold_start_idx - 1 if test_start_idx is not None else None
    return (test_start_idx, test_end_idx), (hold_start_idx, hold_end_idx)

def _build_threshold_series(probs: pd.DataFrame, mode: str, conf: dict):
    """
    mode == 'absolute' : 返回常数阈值列
    mode == 'quantile' : 用滚动窗口（默认 = VAL_BARS）对 p_up/p_dn 计算分位数阈值，并 shift(1) 防泄漏
    """
    df = probs.sort_values("time").reset_index(drop=True)
    if mode == "absolute":
        df["tau_up"] = conf["tau_up"]
        df["tau_dn"] = conf["tau_dn"]
        df["margin"] = conf["margin"]
        return df

    # quantile rolling
    q_up = conf["q_up"]; q_dn = conf["q_dn"]; margin = conf.get("margin", 0.0)
    win = conf.get("roll_bars", VAL_BARS)

    df["tau_up"] = df["p_up"].rolling(win, min_periods=max(20,int(win/10))).quantile(q_up).shift(1)
    df["tau_dn"] = df["p_dn"].rolling(win, min_periods=max(20,int(win/10))).quantile(q_dn).shift(1)
    df["margin"] = margin
    # 开始几根阈值为 NaN，后续执行时会跳过
    return df

def _simulate_execution(data: pd.DataFrame, ts_col: str, price_col: str):
    """
    输入 data 必须包含：time, p_up, p_dn, tau_up, tau_dn, margin
    执行：非重叠 7 天；入场价=下一根 price；出场价=+H_BARS 后的下一根 price
    """
    d = data.copy()
    d = d.sort_values("time").reset_index(drop=True)

    # 准备价格序列（对齐到同时间轴）
    # 这里 d["time"] 直接来自 base df；我们用 merge_asof 对齐价格
    price = base_df[[ts_col, price_col]].rename(columns={ts_col:"time", price_col:"px"})
    d = pd.merge_asof(d, price.sort_values("time"), on="time")
    d["next_px"] = d["px"].shift(-1)
    d["exit_px"] = d["px"].shift(-(H_BARS+1))

    eq = 1.0
    equity=[{"time": d.loc[0,"time"], "equity": eq}]
    trades=[]
    i=0; n=len(d)

    while i<n:
        row = d.loc[i]
        if pd.isna(row["tau_up"]) or pd.isna(row["tau_dn"]) or pd.isna(row["p_up"]) or pd.isna(row["p_dn"]):
            i += 1; equity.append({"time":row["time"], "equity":eq}); continue

        long_sig  = (row["p_up"] >= row["tau_up"]) and ( (row["p_dn"] < row["tau_dn"]) or ((row["p_dn"] >= row["tau_dn"]) and (row["p_up"]-row["p_dn"] > row["margin"])) )
        short_sig = (row["p_dn"] >= row["tau_dn"]) and ( (row["p_up"] < row["tau_up"]) or ((row["p_up"] >= row["tau_up"]) and (row["p_dn"]-row["p_up"] > row["margin"])) )

        if long_sig and not short_sig and pd.notna(row["next_px"]) and pd.notna(row["exit_px"]):
            entry=row["next_px"]; exitp=row["exit_px"]
            gross=(exitp/entry - 1.0)*LEVERAGE
            net=gross - ROUND_TRIP
            eq *= (1.0+net)
            trades.append({"time":row["time"], "side":"long", "entry":float(entry), "exit":float(exitp), "gross":float(gross), "net":float(net)})
            i += H_BARS
        elif short_sig and not long_sig and pd.notna(row["next_px"]) and pd.notna(row["exit_px"]):
            entry=row["next_px"]; exitp=row["exit_px"]
            gross=(entry/exitp - 1.0)*LEVERAGE
            net=gross - ROUND_TRIP
            eq *= (1.0+net)
            trades.append({"time":row["time"], "side":"short", "entry":float(entry), "exit":float(exitp), "gross":float(gross), "net":float(net)})
            i += H_BARS
        else:
            i += 1
        equity.append({"time": d.loc[min(i, n-1), "time"], "equity": eq})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity).drop_duplicates(subset=["time"], keep="last")
    return trades_df, equity_df

def run():
    global base_df
    base_df, ts_col, price_col = _load_base()
    folds, hold = _load_splits()
    best_df, is_quantile = _load_thresholds()
    row = _choose_model(best_df)

    model_key = row["model"]
    print(f"[Step 7] 使用模型={model_key} | 模式={'quantile' if is_quantile else 'absolute'}")
    if is_quantile:
        print(f"  - q_up={row['q_up']}, q_dn={row['q_dn']}, margin={row.get('margin',0.0)}")
    else:
        print(f"  - tau_up={row['tau_up']}, tau_dn={row['tau_dn']}, margin={row.get('margin',0.0)}")
    print(f"[Step 7] 执行价采用列：{price_col}")

    # 确定 Test & Holdout index
    (ts_start, ts_end), (ho_start, ho_end) = _segment_indices(base_df, ts_col, folds, hold)
    feats = _select_features(base_df, ts_col)

    # ---- Test 段 ----
    if ts_start is not None and ts_end is not None and ts_start < ts_end:
        print(f"[Step7] Test 段：{base_df.loc[ts_start,ts_col]} → {base_df.loc[ts_end,ts_col]}")
        probs_test = _walk_forward_probs(base_df, ts_col, ts_start, ts_end, model_key, progress_name="Test")
        # 构建阈值序列
        if is_quantile:
            conf = {"q_up": float(row["q_up"]), "q_dn": float(row["q_dn"]), "margin": float(row.get("margin",0.0)), "roll_bars": VAL_BARS}
            data = _build_threshold_series(probs_test, mode="quantile", conf=conf)
        else:
            conf = {"tau_up": float(row["tau_up"]), "tau_dn": float(row["tau_dn"]), "margin": float(row.get("margin",0.0))}
            data = _build_threshold_series(probs_test, mode="absolute", conf=conf)

        trades_t, equity_t = _simulate_execution(data, ts_col, price_col)
        os.makedirs(os.path.join("data", "signals"), exist_ok=True)
        os.makedirs(os.path.join("data", "equity"), exist_ok=True)
        trades_t.to_csv(os.path.join("data", "signals", "test_trades.csv"), index=False)
        equity_t.to_csv(os.path.join("data", "equity", "test_equity.csv"), index=False)
        print(f"[Step7] Test 完成：交易 {len(trades_t)} 笔，资金曲线点数 {len(equity_t)}")
    else:
        print("[Step7] Test 段不可用（太短或不存在）。")

    # ---- Holdout 段 ----
    print(f"[Step7] Holdout 段：{base_df.loc[ho_start,ts_col]} → {base_df.loc[ho_end,ts_col]}")
    probs_hold = _walk_forward_probs(base_df, ts_col, ho_start, ho_end, model_key, progress_name="Holdout")
    if is_quantile:
        conf = {"q_up": float(row["q_up"]), "q_dn": float(row["q_dn"]), "margin": float(row.get("margin",0.0)), "roll_bars": VAL_BARS}
        data = _build_threshold_series(probs_hold, mode="quantile", conf=conf)
    else:
        conf = {"tau_up": float(row["tau_up"]), "tau_dn": float(row["tau_dn"]), "margin": float(row.get("margin",0.0))}
        data = _build_threshold_series(probs_hold, mode="absolute", conf=conf)

    trades_h, equity_h = _simulate_execution(data, ts_col, price_col)
    os.makedirs(os.path.join("data", "signals"), exist_ok=True)
    os.makedirs(os.path.join("data", "equity"), exist_ok=True)
    trades_h.to_csv(os.path.join("data", "signals", "holdout_trades.csv"), index=False)
    equity_h.to_csv(os.path.join("data", "equity", "holdout_equity.csv"), index=False)
    print(f"[Step7] Holdout 完成：交易 {len(trades_h)} 笔，资金曲线点数 {len(equity_h)}")

    print("="*72)
    print("[Step 7] 完成。下一步 Step 8：汇总指标、对比 Buy&Hold（含 3x 基准）并画图。")
    print("="*72)

if __name__ == "__main__":
    run()
