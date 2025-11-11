# -*- coding: utf-8 -*-
"""
step8_plot_from_equity.py
---------------------------------
直接根据 Step 7 产出的 equity CSV 作图（不修改 Step 7，不做合并对齐）。

输入（任选其一或两者）：
- data/equity/test_equity.csv
- data/equity/holdout_equity.csv

输出：
- figs/test_equity_strategy.png
- figs/holdout_equity_strategy.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 可按需修改路径 ======
TEST_EQ_PATH = os.path.join("data", "equity", "test_equity.csv")
HOLD_EQ_PATH = os.path.join("data", "equity", "holdout_equity.csv")
FIG_DIR = os.path.join("figs")

os.makedirs(FIG_DIR, exist_ok=True)

def _read_equity(csv_path: str) -> pd.DataFrame:
    """读取 time, equity 两列并按时间排序；不做外联，不丢时间。"""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["time", "equity"])
    df = pd.read_csv(csv_path)
    # 只保留需要的列，允许不同大小写
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time", None)
    ecol = cols.get("equity", None)
    if tcol is None or ecol is None:
        raise ValueError(f"{csv_path} 必须包含列 'time' 和 'equity'")
    df = df[[tcol, ecol]].rename(columns={tcol: "time", ecol: "equity"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    # 保底去重：若 time 有重复，只保留最后一条（通常是更“新”的一条）
    df = df.drop_duplicates(subset=["time"], keep="last")
    return df

def _plot_strategy(eq: pd.DataFrame, title: str, out_path: str):
    """仅绘制策略自身的资金曲线；改进 X 轴刻度，避免重叠。"""
    if eq.empty:
        print(f"[Skip] {title}: 输入为空，未绘图。")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(eq["time"], eq["equity"], label="Strategy", linewidth=1.6)

    ax = plt.gca()
    # 使用自动日期定位 + 简洁格式，减少重叠
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

def main():
    # Test
    test_eq = _read_equity(TEST_EQ_PATH)
    _plot_strategy(test_eq, "Test Equity (Strategy)", os.path.join(FIG_DIR, "test_equity_strategy.png"))

    # Holdout（如有）
    hold_eq = _read_equity(HOLD_EQ_PATH)
    _plot_strategy(hold_eq, "Holdout Equity (Strategy)", os.path.join(FIG_DIR, "holdout_equity_strategy.png"))

    print("Done.")

if __name__ == "__main__":
    main()
