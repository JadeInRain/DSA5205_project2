# -*- coding: utf-8 -*-
"""
step8_report_fixed.py
---------------------
Fixes:
1) X 轴日期刻度重叠：使用 AutoDateLocator + ConciseDateFormatter 自动稀疏刻度；
2) Strategy 线条只显示一段：绘图基于“策略资金曲线”的时间戳，使用 merge_asof
   将 Buy&Hold 1x / 3x 对齐到策略时间戳，三条曲线共享同一时间轴。
Outputs (与原版一致):
- reports/step8/summary_test.csv
- reports/step8/summary_holdout.csv
- figs/test_equity.png
- figs/holdout_equity.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ANN_FACTOR = 365 * 24 / 4  # 4h bars per year

PREP_CSV   = os.path.join("data", "prepared", "btc_4h_preprocessed.csv")
TEST_TRDS  = os.path.join("data", "signals", "test_trades.csv")
HOLD_TRDS  = os.path.join("data", "signals", "holdout_trades.csv")
TEST_EQ    = os.path.join("data", "equity", "test_equity.csv")
HOLD_EQ    = os.path.join("data", "equity", "holdout_equity.csv")

OUT_DIR    = os.path.join("reports", "step8")
FIG_DIR    = os.path.join("figs")


# -------------------- Helpers --------------------
def _find_ts_col(df: pd.DataFrame) -> str:
    for c in ["timestamp", "open_time", "time", "datetime"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _load_prep() -> pd.DataFrame:
    base = pd.read_csv(PREP_CSV)
    base.columns = [c.strip().lower() for c in base.columns]
    ts = _find_ts_col(base)
    base[ts] = pd.to_datetime(base[ts], errors="coerce")
    # 兜底 close
    if "close" not in base.columns:
        for c in ["open", "price"]:
            if c in base.columns:
                base["close"] = base[c]
                break
        if "close" not in base.columns:
            raise ValueError("No price column found for Buy&Hold baseline.")
    base = base[[ts, "close"]].rename(columns={ts: "time"}).dropna().sort_values("time")
    return base


def _metrics_from_equity(eq: pd.DataFrame) -> dict:
    eq = eq.dropna().sort_values("time").reset_index(drop=True)
    if eq.empty:
        return dict(final_equity=np.nan, cagr=np.nan, sharpe=np.nan, maxdd=np.nan, bars=0)
    r = eq["equity"].pct_change().dropna()

    # CAGR
    if len(eq) > 1:
        years = (eq["time"].iloc[-1] - eq["time"].iloc[0]).total_seconds() / (365*24*3600)
        cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0])**(1/max(years, 1e-9)) - 1 if years > 0 else np.nan
    else:
        cagr = np.nan

    mu = r.mean() * ANN_FACTOR
    sd = r.std(ddof=0) * np.sqrt(ANN_FACTOR)
    sharpe = (mu / sd) if (sd and sd > 0) else np.nan

    cum = eq["equity"].values
    peak = np.maximum.accumulate(cum)
    maxdd = (cum / peak - 1.0).min()

    return dict(
        final_equity=float(eq["equity"].iloc[-1]),
        cagr=float(cagr) if pd.notna(cagr) else np.nan,
        sharpe=float(sharpe) if pd.notna(sharpe) else np.nan,
        maxdd=float(maxdd) if pd.notna(maxdd) else np.nan,
        bars=int(len(eq))
    )


def _trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return dict(n_trades=0, win_rate=np.nan, avg_net=np.nan, med_net=np.nan)
    wins = (trades["net"] > 0).mean() if "net" in trades.columns else np.nan
    return dict(
        n_trades=int(len(trades)),
        win_rate=float(wins) if pd.notna(wins) else np.nan,
        avg_net=float(trades["net"].mean()) if "net" in trades.columns else np.nan,
        med_net=float(trades["net"].median()) if "net" in trades.columns else np.nan,
    )


def _bh_equity(base: pd.DataFrame, start_t, end_t, leverage=1.0) -> pd.DataFrame:
    """Buy&Hold 基准，按 close 计算，限定在策略时间窗。"""
    px = base[(base["time"] >= start_t) & (base["time"] <= end_t)].copy()
    px = px.sort_values("time").reset_index(drop=True)
    if px.empty:
        return pd.DataFrame(columns=["time", "equity"])
    ret = px["close"].pct_change().fillna(0.0) * leverage
    eq = (1 + ret).cumprod()
    return pd.DataFrame({"time": px["time"], "equity": eq})


def _align_to_strategy_times(eq_bh: pd.DataFrame, eq_strat: pd.DataFrame) -> pd.DataFrame:
    """把 BH 净值对齐到策略时间戳（避免外连接造成的时间轴不一致）。"""
    a = pd.merge_asof(
        eq_strat[["time"]].sort_values("time"),
        eq_bh.sort_values("time"),
        on="time",
        direction="backward"
    )
    a["equity"] = a["equity"].ffill().bfill()
    return a


def _plot_aligned(eq_strat: pd.DataFrame, eq_bh1: pd.DataFrame, eq_bh3: pd.DataFrame,
                  title: str, out_path: str):
    plt.figure(figsize=(8.8, 5.6))
    plt.plot(eq_strat["time"], eq_strat["equity"], label="Strategy")
    plt.plot(eq_bh1["time"],   eq_bh1["equity"],   label="Buy&Hold 1x")
    plt.plot(eq_bh3["time"],   eq_bh3["equity"],   label="Buy&Hold 3x")

    # 稀疏日期刻度
    ax = plt.gca()
    locator   = mdates.AutoDateLocator(minticks=4, maxticks=8)
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


def _segment(eq_path: str, trades_path: str, base: pd.DataFrame, name: str):
    if not os.path.exists(eq_path):
        print(f"[Step 8] {name}: equity file not found, skip.")
        return

    eq = pd.read_csv(eq_path)
    if eq.empty:
        print(f"[Step 8] {name}: equity file is empty, skip.")
        return
    eq["time"] = pd.to_datetime(eq["time"], errors="coerce")
    eq = eq.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 策略指标（直接用策略的时间戳）
    kpis = _metrics_from_equity(eq)

    trades = None
    if os.path.exists(trades_path):
        trades = pd.read_csv(trades_path)
        if "time" in trades.columns:
            trades["time"] = pd.to_datetime(trades["time"], errors="coerce")
    tstats = _trade_stats(trades)

    # 构建基准并对齐到策略时间戳
    start_t, end_t = eq["time"].iloc[0], eq["time"].iloc[-1]
    bh1  = _bh_equity(base, start_t, end_t, leverage=1.0)
    bh3  = _bh_equity(base, start_t, end_t, leverage=3.0)
    bh1a = _align_to_strategy_times(bh1, eq)
    bh3a = _align_to_strategy_times(bh3, eq)

    # 汇总输出
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [
        dict(model="strategy", **kpis, **tstats),
        dict(model="bh_1x",    **_metrics_from_equity(bh1a),
             n_trades=np.nan, win_rate=np.nan, avg_net=np.nan, med_net=np.nan),
        dict(model="bh_3x",    **_metrics_from_equity(bh3a),
             n_trades=np.nan, win_rate=np.nan, avg_net=np.nan, med_net=np.nan),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"summary_{name}.csv"), index=False)

    # 画图（基于策略时间戳对齐；自动稀疏刻度）
    os.makedirs(FIG_DIR, exist_ok=True)
    _plot_aligned(eq, bh1a, bh3a,
                  f"{name.capitalize()} Equity (Strategy vs Buy&Hold)",
                  os.path.join(FIG_DIR, f"{name}_equity.png"))
    print(f"[Step 8] {name}: summary -> reports/step8/summary_{name}.csv; "
          f"figure -> figs/{name}_equity.png")


def main():
    base = _load_prep()
    _segment(TEST_EQ, TEST_TRDS, base, "test")
    _segment(HOLD_EQ, HOLD_TRDS, base, "holdout")
    print("="*72)
    print("[Step 8] Done.")
    print("="*72)


if __name__ == "__main__":
    main()
