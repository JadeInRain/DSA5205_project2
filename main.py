# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ========================= User Parameters =========================
DATA_PATH      = "btc_4h_dataset.parquet"
INITIAL_CAP    = 10_000
FEE            = 0.001
TRAIN_RATIO    = 0.70
PROB_TH        = 0.51          # Original V1 threshold
MIN_GAP_BARS   = 1             
POS_FRAC       = 0.60          
RANDOM_STATE   = 42

HARD_STOP      = 0.10          # Loosened stop-loss
TP_TRIGGER     = 0.20          # Loosened take-profit
TRAIL_AFTER_TP = 0.08          # Loosened trailing stop
MAX_HOLD       = 120           # Loosened max holding period
# ======================================================================


def annualize_factor_from_timestamps(ts: pd.Series) -> float:
    ts = pd.to_datetime(ts.values)
    if len(ts) < 2:
        return 252.0
    sec = np.median(np.diff(ts).astype("timedelta64[s]").astype(float))
    return 365 * 24 * 3600 / sec if sec > 0 else 252.0


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    return float(dd.max()) if len(dd) else 0.0


class CapitalFixedBitcoinTrader:
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.model = None
        self.split_idx = None

    # ---------- Features ----------
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        need_cols = {"timestamp", "open", "high", "low", "close"}
        missing = need_cols - set(df.columns)
        if missing:
            raise ValueError(f"Input data is missing required columns: {missing}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["ret_1"]  = df["close"].pct_change(1)
        df["ret_4"]  = df["close"].pct_change(4)
        df["ret_12"] = df["close"].pct_change(12)

        df["ma_8"]   = df["close"].rolling(8).mean()
        df["ma_20"]  = df["close"].rolling(20).mean()
        df["ma_50"]  = df["close"].rolling(50).mean()
        
        df["px_vs_ma8"]  = (df["close"] - df["ma_8"]) / df["ma_8"]
        df["px_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

        df["vol_8"]  = df["ret_1"].rolling(8).std()
        df["vol_20"] = df["ret_1"].rolling(20).std()

        df["hi_20"] = df["high"].rolling(20).max()
        df["lo_20"] = df["low"].rolling(20).min()
        df["from_hi"] = (df["hi_20"] - df["close"]) / df["close"]
        
        # [BUG FIX]: Denominator changed from df["from_lo"] to df["close"]
        df["from_lo"] = (df["close"] - df["lo_20"]) / df["close"]
        
        df["ma_200"] = df["close"].rolling(200).mean()
        df['golden_cross'] = (df['ma_50'] > df['ma_200']).astype(int)

        fut_ret = df["close"].shift(-1) / df["close"] - 1.0
        df["target"] = (fut_ret > 0).astype(int)
        return df

    @property
    def feature_cols(self):
        # Original V1 features
        return [
            "ret_1", "ret_4", "ret_12",
            "ma_8", "ma_20", "ma_50",
            "px_vs_ma8", "px_vs_ma20",
            "vol_8", "vol_20",
            "from_hi", "from_lo",
        ]

    # ---------- Model Training ----------
    def train_model(self, df: pd.DataFrame, train_ratio=0.7, random_state=42):
        df_feat = self.create_features(df)
        df_clean = df_feat.dropna(subset=self.feature_cols + ["ma_200", "target"]).reset_index(drop=True)

        X = df_clean[self.feature_cols]
        y = df_clean["target"]
        self.split_idx = int(len(X) * train_ratio)
        if self.split_idx <= 0 or self.split_idx >= len(X):
             raise ValueError(f"Invalid TRAIN_RATIO={train_ratio}")

        X_total_tr, X_test = X.iloc[:self.split_idx], X.iloc[self.split_idx:]
        y_total_tr, y_test = y.iloc[:self.split_idx], y.iloc[self.split_idx:]

        tr_main_end = int(len(X_total_tr) * 0.8)
        X_tr, X_val = X_total_tr.iloc[:tr_main_end], X_total_tr.iloc[tr_main_end:]
        y_tr, y_val = y_total_tr.iloc[:tr_main_end], y_total_tr.iloc[tr_main_end:]

        grid = [
            {"n_estimators": 100, "max_depth": 5,  "min_samples_split": 20},
            {"n_estimators": 100, "max_depth": 10, "min_samples_split": 20},
            {"n_estimators": 200, "max_depth": 8,  "min_samples_split": 30},
        ]
        best_score, best_params = -np.inf, None
        
        if X_val.empty:
             best_params = grid[0]
             best_score = 0
        else:
            for params in grid:
                clf = RandomForestClassifier(random_state=random_state, class_weight="balanced", **params)
                clf.fit(X_tr, y_tr)
                score = clf.score(X_val, y_val) # Original V1 logic
                if score > best_score:
                    best_score, best_params = score, params

        self.model = RandomForestClassifier(random_state=random_state, class_weight="balanced", **best_params)
        self.model.fit(X_total_tr, y_total_tr)

        print("Training model...")
        print(f"Target positive class ratio: {y.mean():.3f}")
        print(f"Training set accuracy: {self.model.score(X_total_tr, y_total_tr):.3f}")
        print(f"Test set accuracy: {self.model.score(X_test, y_test) if len(X_test)>0 else 0.0:.3f}")
        print(f"Selected model params: {best_params} (Validation accuracy: {best_score:.3f})")
        
        return df_clean

    # ---------- OOS Backtest ----------
    def backtest_oos(self, df_slice: pd.DataFrame,
                     prob_th=0.51, min_gap_bars=1,
                     pos_frac=0.6, hard_stop=0.02,
                     tp_trigger=0.05, trail_after_tp=0.02,
                     max_hold=20, index_offset=0):
        
        if 'golden_cross' not in df_slice.columns:
            raise ValueError("Backtest data (df_slice) is missing 'golden_cross' feature")
        
        capital = float(self.initial_capital)
        position = 0.0
        entry_price = 0.0
        entry_capital = 0.0
        peak_price = 0.0
        holding = 0
        last_exit_i = -10

        trades, equity_curve = [], []
        
        probs = self.model.predict_proba(df_slice[self.feature_cols])[:, 1]

        print(f"\nStarting backtest ( ML Entry + Long-Term Exit, {len(df_slice)} samples, ML threshold = {prob_th:.2f})")
        print(f"ML signals (unfiltered): {int((probs > prob_th).sum())}")
        print(f"Exits: Stop {hard_stop:.0%}, Profit {tp_trigger:.0%}({trail_after_tp:.0%}), Hold {max_hold} bars")

        for i in range(len(df_slice)):
            px = float(df_slice["close"].iloc[i])
            p_ml = float(probs[i]) # ML model probability
            is_bull_regime = df_slice["golden_cross"].iloc[i] == 1 # Rule 1: Bull market filter
            
            equity_curve.append(capital + position * px)

            # --- Exit Logic (V14 logic + filter)
            if position > 0:
                holding += 1
                pnl_pct_mark = (px - entry_price) / entry_price
                peak_price = max(peak_price, px)
                dd_from_peak = (peak_price - px) / peak_price if peak_price > 0 else 0.0
                
                exit_cond = (
                    (not is_bull_regime) 
                    or (pnl_pct_mark <= -hard_stop)
                    or (pnl_pct_mark >= tp_trigger and dd_from_peak >= trail_after_tp)
                    or (holding >= max_hold)
                )
                
                if exit_cond:
                    exit_value = position * px
                    pnl = exit_value - entry_capital
                    capital += exit_value * (1 - self.transaction_cost)
                    
           
                    trades[-1].update({
                        "exit_idx": index_offset + i,
                        "exit_price": px,
                        "exit_value": exit_value,
                        "pnl": pnl,
                        "pnl_pct": pnl / entry_capital,
                        "holding_period": holding,
                        "type": ("DEATH_CROSS" if not is_bull_regime else
                                 "STOP_LOSS" if pnl_pct_mark <= -hard_stop else
                                 "TRAIL_TP" if (pnl_pct_mark >= tp_trigger and dd_from_peak >= trail_after_tp)
                                 else "MAX_HOLD")
                    })
                    position = 0.0
                    entry_price = entry_capital = 0.0
                    peak_price = 0.0
                    holding = 0
                    last_exit_i = i 

            # --- Entry Logic )
            if position == 0 and is_bull_regime and (p_ml > prob_th) and (i > last_exit_i + min_gap_bars):
                trade_amt = capital * pos_frac
                if trade_amt > 0:
                    fee = trade_amt * self.transaction_cost
                    position   = trade_amt / px
                    entry_price = px
                    entry_capital = trade_amt
                    capital -= (trade_amt + fee)
                    peak_price = px
                    holding = 0 
                    
                    trades.append({
                        "entry_idx": index_offset + i,
                        "entry_price": entry_price,
                        "entry_capital": entry_capital,
                        "probability": p_ml
                    })

        # Finalize
        if position > 0:
            px = float(df_slice["close"].iloc[-1])
            exit_value = position * px
            pnl = exit_value - entry_capital
            capital += exit_value * (1 - self.transaction_cost)
            
            trades[-1].update({
                "exit_idx": index_offset + len(df_slice) - 1,
                "exit_price": px,
                "exit_value": exit_value,
                "pnl": pnl,
                "pnl_pct": pnl / entry_capital,
                "holding_period": holding,
                "type": "FINAL_EXIT"
            })

        if not equity_curve:
            equity_curve = [self.initial_capital]
        if equity_curve and len(equity_curve) < len(df_slice):
            equity_curve.append(capital)
        equity_curve[-1] = capital

        completed = len([t for t in trades if "pnl" in t])
        print(f"Backtest complete | Total Trades: {completed} | Final Capital: ${capital:,.2f} | Total Return: {capital / self.initial_capital - 1.0:.2%}")
        return trades, equity_curve

    # (summarize function unchanged)
    def summarize(self, trades, equity_curve, df_slice):
        done = [t for t in trades if "pnl" in t] 
        if not done:
            print("⚠️ No completed trades, cannot calculate metrics")
            return None
        pnl_pct = np.array([t["pnl_pct"] for t in done], float)
        wins, losses = pnl_pct[pnl_pct > 0], pnl_pct[pnl_pct < 0]

        total_return = equity_curve[-1] / self.initial_capital - 1.0
        ppyear = annualize_factor_from_timestamps(df_slice["timestamp"])
        nper = max(len(equity_curve) - 1, 1)
        ann_ret = (equity_curve[-1] / self.initial_capital) ** (ppyear / nper) - 1.0

        ec_series = pd.Series(equity_curve, dtype=float)
        if ec_series.empty:
            print("⚠️ Equity curve is empty, cannot calculate metrics")
            return None
        ec = ec_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        sharpe = (ec.mean() / ec.std() * np.sqrt(ppyear)) if ec.std() > 0 and not ec.empty else 0.0
        mdd = max_drawdown(np.array(equity_curve, float))

        return {
            "total_trades": int(len(done)),
            "win_rate": float((pnl_pct > 0).mean()) if len(pnl_pct > 0) else 0.0,
            "avg_profit": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(abs(wins.sum() / losses.sum())) if len(losses) and losses.sum() != 0 else np.inf,
            "total_return": float(total_return),
            "annualized_return": float(ann_ret),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(mdd),
            "final_capital": float(equity_curve[-1]),
        }

    # (Plot optimization V3 - Remove P&L labels)
    def plot_results(self, df_slice, equity_curve, trades, title_suffix="(Test Set)"):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        if df_slice.empty:
            print("⚠️ No test data, cannot plot")
            return
        
        if not isinstance(df_slice.index, pd.RangeIndex):
            print("Plotting Warning: df_slice index not reset, X-axis may be misaligned.")
            df_slice = df_slice.reset_index(drop=True)

        plot_base_idx = self.split_idx 
        
        ax1.plot(df_slice["close"].values, label="BTC Price", alpha=0.8, lw=1.1)
        if "ma_50" in df_slice.columns and "ma_200" in df_slice.columns:
            ax1.plot(df_slice["ma_50"].values, label="MA 50", alpha=0.6, lw=0.8, ls="--")
            ax1.plot(df_slice["ma_200"].values, label="MA 200", alpha=0.6, lw=0.8, ls="--", color="red")
        
        buys = [t for t in trades if "entry_price" in t]
        sells = [t for t in trades if "exit_price" in t]
        
        buy_indices = [t["entry_idx"] - plot_base_idx for t in buys]
        sell_indices = [t["exit_idx"] - plot_base_idx for t in sells]

        ax1.scatter(buy_indices, [t["entry_price"] for t in buys],
                    c="green", marker="^", s=36, label="Buy", zorder=5)
        ax1.scatter(sell_indices, [t["exit_price"] for t in sells],
                    c="red", marker="v", s=36, label="Sell", zorder=5)
        
        ax1.set_title(f"BTC Price with Trading Signals {title_suffix}")
        ax1.set_ylabel("Price ($)")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        if equity_curve:
            if len(equity_curve) != len(df_slice):
                print(f"Plotting Warning: EC length ({len(equity_curve)}) does not match OOS length ({len(df_slice)}).")
                if len(equity_curve) > len(df_slice):
                    equity_curve = equity_curve[:len(df_slice)]
                else:
                    fill_values = [equity_curve[-1]] * (len(df_slice) - len(equity_curve))
                    equity_curve = equity_curve + fill_values
            
            ax2.plot(equity_curve, lw=2, label="Strategy Equity")
            ax2.axhline(self.initial_capital, color="gray", ls="--", label="Initial Capital")
            bh = (df_slice["close"] / df_slice["close"].iloc[0]) * self.initial_capital
            ax2.plot(bh.values, ls="--", label="Buy & Hold")
            ax2.set_title(f"Equity Curve {title_suffix}")
            ax2.set_ylabel("Capital ($)")
            ax2.legend(); ax2.grid(True, alpha=0.3)
        else:
             ax2.set_title("Equity Curve (No Data)")

        done = [t for t in trades if "pnl_pct" in t]
        if done:
            pnls = [t["pnl_pct"] for t in done]
            types = [t["type"] for t in done]
            colors = ["green" if p > 0 else "red" for p in pnls]
            bars = ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.75)
            ax3.axhline(0, color="black", lw=1)
            
            # =========== Plot Optimization V3: Remove Labels ===========
            # (Labels are commented out for cleanliness)
            # ===========================================================
            
            ax3.set_title("Individual Trade P&L"); ax3.set_xlabel("Trade #"); ax3.set_ylabel("P&L (%)")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.set_title("Individual Trade P&L (No Trades)")

        plt.tight_layout()
        plt.savefig("trading_results_oos_improved.png", dpi=150, bbox_inches="tight")
        plt.show()


# ========================= Main Execution  =========================
if __name__ == "__main__":
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    trader = CapitalFixedBitcoinTrader(initial_capital=INITIAL_CAP, transaction_cost=FEE)
    
    df_clean = trader.train_model(
        df, 
        train_ratio=TRAIN_RATIO, 
        random_state=RANDOM_STATE
    )
    
    oos = df_clean.iloc[trader.split_idx:].copy().reset_index(drop=True) 
    
    if oos.empty:
        print("❌ Error: Test set (OOS) is empty, check data or TRAIN_RATIO")
    else:
        print(f"\nExecuting strict OOS backtest (Test set, {len(oos)} data points)...")

        # Passing  "long-term exit" backtest parameters
        trades, equity_curve = trader.backtest_oos(
            oos,
            prob_th=PROB_TH,
            min_gap_bars=MIN_GAP_BARS,
            pos_frac=POS_FRAC,
            hard_stop=HARD_STOP,
            tp_trigger=TP_TRIGGER,
            trail_after_tp=TRAIL_AFTER_TP,
            max_hold=MAX_HOLD,
            index_offset=trader.split_idx
        )

        print("\n" + "=" * 50)
        print("Trading Strategy Performance Evaluation (Test Set)")
        print("=" * 50)
        mets = trader.summarize(trades, equity_curve, oos)
        
        if mets:
            print(f"Total Trades: {mets['total_trades']}")
            print(f"Win Rate: {mets['win_rate']:.1%}")
            print(f"Avg Profit: {mets['avg_profit']:.2%}")
            print(f"Avg Loss: {mets['avg_loss']:.2%}")
            print(f"Profit Factor: {mets['profit_factor']:.2f}")
            print(f"Total Return: {mets['total_return']:.2%}")
            print(f"Annualized Return: {mets['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {mets['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {mets['max_drawdown']:.2%}")
            print(f"Final Capital: ${mets['final_capital']:,.2f}")

            bh_total = 0.0 
            if len(oos) >= 2:
                bh_start, bh_end = oos["close"].iloc[0], oos["close"].iloc[-1] 
                if pd.notna(bh_start) and pd.notna(bh_end) and bh_start != 0:
                    ppyear = annualize_factor_from_timestamps(oos["timestamp"])
                    nper = max(len(oos) - 1, 1)
                    bh_total = bh_end / bh_start - 1.0
                    bh_ann = (bh_end / bh_start) ** (ppyear / nper) - 1.0
                    bh_rets = oos["close"].pct_change().dropna()
                    bh_sharpe = (bh_rets.mean() / bh_rets.std() * np.sqrt(ppyear)) if bh_rets.std() > 0 and not bh_rets.empty else 0.0
                    print(f"Benchmark (Buy & Hold) - Total Return: {bh_total:.2%}, Annualized: {bh_ann:.2%}, Sharpe: {bh_sharpe:.2f}")
                else:
                    print("Benchmark (Buy & Hold) - Cannot calculate (insufficient data or start price is 0)")
            
            print("\nStrategy Profitable" if mets["total_return"] > 0 else "\nStrategy Unprofitable")
            
            if mets["total_return"] > bh_total:
                print("Strategy outperformed benchmark!")
            else:
                print("Strategy did not outperform benchmark, needs tuning.")

        trader.plot_results(oos, equity_curve, trades, title_suffix=f"(ML V1 + MA Filter)")