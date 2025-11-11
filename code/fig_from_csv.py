# -*- coding: utf-8 -*-
"""
step8_plot_from_equity.py
---------------------------------
直接根据 Step 7 产出的 equity CSV 作图，并叠加 Buy&Hold（1x/3x）。
不修改 Step 7，不做外联；Buy&Hold 统一对齐到“策略曲线”的时间戳。

输入（任选其一或两者）：
- data/equity/test_equity.csv
- data/equity/holdout_equity.csv
- data/prepared/btc_4h_preprocessed.csv  (提供 close 用于 Buy&Hold)

输出：
- figs/test_equity.png
- figs/holdout_equity.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 可按需修改路径 ======
TEST_EQ_PATH   = os.path.join("data", "equity", "test_equity.csv")
HOLD_EQ_PATH   = os.path.join("data", "equity", "holdout_equity.csv")
PRICES_PATH    = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
FIG_DIR        = os.path.join("figs")

os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 工具函数 ----------
def _read_equity(csv_path: str) -> pd.DataFrame:
    """读取 time, equity 两列并按时间排序；不做外联，不丢时间。"""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["time", "equity"])
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time"); ecol = cols.get("equity")
    if tcol is None or ecol is None:
        raise ValueError(f"{csv_path} 必须包含列 'time' 和 'equity'")
    df = df[[tcol, ecol]].rename(columns={tcol: "time", ecol: "equity"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # 去重：若 time 有重复，只保留最后一条
    return df.drop_duplicates(subset=["time"], keep="last")

def _load_prices(path: str) -> pd.DataFrame:
    """加载价格，返回 [time, close]；自动识别时间列，若无 close 则回退到 open。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"价格文件缺失：{path}")
    base = pd.read_csv(path)
    base.columns = [c.strip().lower() for c in base.columns]
    # 找时间列
    ts = None
    for cand in ["timestamp", "open_time", "time", "datetime"]:
        if cand in base.columns:
            ts = cand; break
    if ts is None:
        ts = base.columns[0]
    base[ts] = pd.to_datetime(base[ts], errors="coerce")
    if "close" not in base.columns:
        if "open" in base.columns:
            base["close"] = base["open"]
        else:
            raise ValueError("未找到价格列（需要 close 或 open）")
    return base[[ts, "close"]].rename(columns={ts: "time"}).dropna().sort_values("time")

def _bh_equity(prices: pd.DataFrame, start_t, end_t, leverage: float) -> pd.DataFrame:
    """在 [start_t, end_t] 窗口上构建 Buy&Hold 等权净值（从1起点）。"""
    px = prices[(prices["time"] >= start_t) & (prices["time"] <= end_t)].copy()
    px = px.sort_values("time").reset_index(drop=True)
    if px.empty:
        return pd.DataFrame(columns=["time", "equity"])
    ret = px["close"].pct_change().fillna(0.0) * leverage
    eq  = (1.0 + ret).cumprod()
    return pd.DataFrame({"time": px["time"], "equity": eq})

def _align_to_strategy(eq_bh: pd.DataFrame, eq_strat: pd.DataFrame) -> pd.DataFrame:
    """把 Buy&Hold 净值对齐到“策略曲线”的时间戳（后向对齐 + 前后填充）。"""
    out = pd.merge_asof(
        eq_strat[["time"]].sort_values("time"),
        eq_bh.sort_values("time"),
        on="time",
        direction="backward",
    )
    out["equity"] = out["equity"].ffill().bfill()
    return out

def _plot_three(eq_strat: pd.DataFrame, eq_bh1: pd.DataFrame, eq_bh3: pd.DataFrame,
                title: str, out_path: str):
    """按统一样式作图；自动稀疏日期刻度，避免重叠。"""
    if eq_strat.empty:
        print(f"[Skip] {title}: 策略曲线为空。"); return
    plt.figure(figsize=(10, 6))
    # 先画策略曲线（不做任何合并，不截断）
    plt.plot(eq_strat["time"], eq_strat["equity"], label="Strategy", linewidth=1.6)
    plt.plot(eq_bh1["time"],  eq_bh1["equity"],  label="Buy&Hold 1x")
    plt.plot(eq_bh3["time"],  eq_bh3["equity"],  label="Buy&Hold 3x")

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] {title}: {out_path}")

# ---------- 主流程 ----------
def _one_segment(eq_path: str, fig_name: str, title_prefix: str):
    eq = _read_equity(eq_path)
    if eq.empty:
        print(f"[Skip] {title_prefix}: 未找到或为空 -> {eq_path}")
        return
    prices = _load_prices(PRICES_PATH)
    start_t, end_t = eq["time"].iloc[0], eq["time"].iloc[-1]
    bh1  = _bh_equity(prices, start_t, end_t, leverage=1.0)
    bh3  = _bh_equity(prices, start_t, end_t, leverage=3.0)
    # 严格对齐到策略时间戳
    bh1a = _align_to_strategy(bh1, eq)
    bh3a = _align_to_strategy(bh3, eq)
    _plot_three(eq, bh1a, bh3a, f"{title_prefix} (Strategy vs Buy&Hold)",
                os.path.join(FIG_DIR, fig_name))

def main():
    _one_segment(TEST_EQ_PATH,   "test_equity.png",    "Test Equity")
    _one_segment(HOLD_EQ_PATH,   "holdout_equity.png", "Holdout Equity")
    print("Done.")

if __name__ == "__main__":
    main()
