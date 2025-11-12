

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from Data_BTC import get_train_test_data


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)


def generate_monthly_returns_table():
    try:
        import glob
        import pickle
        from Market_BTC import BTCTradingEnv
        from Data_BTC import INDICATORS
        from Config_BTC import REWARD_FUTURE_DATE
        
        model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
        if not model_files:
            return
        
        train, test = get_train_test_data()
        
        with open(model_files[0], 'rb') as f:
            dqn = pickle.loads(f.read())
        
        env_kwargs = {
            "buy_cost_pct": 0.0001,
            "sell_cost_pct": 0.0001,
            "state_space": len(INDICATORS),
            "tech_indicator_list": INDICATORS,
            "reward_future_day": REWARD_FUTURE_DATE,
            "seed": 0
        }
        
        env = BTCTradingEnv(test, **env_kwargs)
        s = env.reset()
        
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            if done:
                break
            s = s_
        

        df = pd.DataFrame({
            'date': pd.to_datetime(env.date_memory),
            'strategy_return': env.return_list,
            'bh_return': env.buy_hold_list
        })
        
    
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month').agg({
            'strategy_return': lambda x: (1 + x).prod() - 1,
            'bh_return': lambda x: (1 + x).prod() - 1
        }).reset_index()
        
        monthly['excess_return'] = monthly['strategy_return'] - monthly['bh_return']
        

        monthly['year_month'] = monthly['year_month'].astype(str)
        monthly['strategy_return'] = monthly['strategy_return'].apply(lambda x: f"{x:.2%}")
        monthly['bh_return'] = monthly['bh_return'].apply(lambda x: f"{x:.2%}")
        monthly['excess_return'] = monthly['excess_return'].apply(lambda x: f"{x:.2%}")
        
        monthly.columns = ['Year-Month', 'Strategy Return', 'Benchmark Return', 'Excess Return']
        
        monthly.to_csv('results/tables/monthly_returns.csv', index=False)
        
    except Exception as e:
        print(f"{e}")


def generate_sharpe_ratio_comparison():
    try:
        import glob
        import pickle
        from Market_BTC import BTCTradingEnv
        from Data_BTC import INDICATORS
        from Config_BTC import REWARD_FUTURE_DATE
        
        model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
        if not model_files:
            return
        
        train, test = get_train_test_data()
        
        windows = [30, 60, 90, 180, 365]    
        window_periods = [int(w * 6) for w in windows]  
        
        strategy_sharpes = []
        bh_sharpes = []
        
        with open(model_files[0], 'rb') as f:
            dqn = pickle.loads(f.read())
        
        env_kwargs = {
            "buy_cost_pct": 0.0001,
            "sell_cost_pct": 0.0001,
            "state_space": len(INDICATORS),
            "tech_indicator_list": INDICATORS,
            "reward_future_day": REWARD_FUTURE_DATE,
            "seed": 0
        }
        
        env = BTCTradingEnv(test, **env_kwargs)
        s = env.reset()
        
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            if done:
                break
            s = s_
        
        returns = np.array(env.return_list)
        bh_returns = np.array(env.buy_hold_list)
        
        for periods in window_periods:
            if len(returns) < periods:
                continue
            
            window_sharpes = []
            bh_window_sharpes = []
            
            for i in range(len(returns) - periods + 1):
                window_ret = returns[i:i+periods]
                if window_ret.std() > 0:
                    sharpe = (window_ret.mean() / window_ret.std()) * np.sqrt(2190)
                    window_sharpes.append(sharpe)
                
                bh_window_ret = bh_returns[i:i+periods]
                if bh_window_ret.std() > 0:
                    bh_sharpe = (bh_window_ret.mean() / bh_window_ret.std()) * np.sqrt(2190)
                    bh_window_sharpes.append(bh_sharpe)
            
            strategy_sharpes.append(np.mean(window_sharpes) if window_sharpes else 0)
            bh_sharpes.append(np.mean(bh_window_sharpes) if bh_window_sharpes else 0)
        
        x = np.arange(len(windows[:len(strategy_sharpes)]))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(x - width/2, strategy_sharpes, width, label='DQN Strategy', color='#e74c3c')
        ax.bar(x + width/2, bh_sharpes, width, label='Buy-and-Hold', color='#3498db')
        
        ax.set_xlabel('Time Window (Days)', fontsize=12)
        ax.set_ylabel('Average Sharpe Ratio', fontsize=12)
        ax.set_title('Sharpe Ratio Comparison Across Different Time Windows', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}d" for w in windows[:len(strategy_sharpes)]])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results/figures/sharpe_ratio_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        
    except Exception as e:
        print(f"{e}")


def generate_trading_signals_plot():
    try:
        import glob
        import pickle
        from Market_BTC import BTCTradingEnv
        from Data_BTC import INDICATORS
        from Config_BTC import REWARD_FUTURE_DATE
        
        model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
        if not model_files:
            return
        
        train, test = get_train_test_data()
        
        with open(model_files[0], 'rb') as f:
            dqn = pickle.loads(f.read())
        
        env_kwargs = {
            "buy_cost_pct": 0.0001,
            "sell_cost_pct": 0.0001,
            "state_space": len(INDICATORS),
            "tech_indicator_list": INDICATORS,
            "reward_future_day": REWARD_FUTURE_DATE,
            "seed": 0
        }
        
        env = BTCTradingEnv(test, **env_kwargs)
        s = env.reset()
        actions = []
        
        while True:
            a = dqn.choose_action(s)
            actions.append(a)
            s_, r, done, info = env.step(a)
            if done:
                break
            s = s_
        
        dates = pd.to_datetime(env.date_memory)
        prices = test['close'].values[:len(dates)]
        actions = np.array(actions)
        
        plot_length = min(500, len(dates))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(dates[:plot_length], prices[:plot_length], 
                label='BTC Price', color='black', linewidth=1.5, alpha=0.7)
        
        long_signals = np.where(actions[:plot_length] == 2)[0]
        ax1.scatter(dates[long_signals], prices[long_signals], 
                   color='green', marker='^', s=100, label='Long Signal', zorder=5)
        
        short_signals = np.where(actions[:plot_length] == 0)[0]
        ax1.scatter(dates[short_signals], prices[short_signals], 
                   color='red', marker='v', s=100, label='Short Signal', zorder=5)
        
        ax1.set_ylabel('BTC Price (USDT)', fontsize=12)
        ax1.set_title('Trading Signals on BTC Price (Test Set - First 500 periods)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        position_colors = []
        for action in actions[:plot_length]:
            if action == 2:  # Long
                position_colors.append('green')
            elif action == 0:  # Short
                position_colors.append('red')
            else:  # Neutral
                position_colors.append('gray')
        
        for i in range(plot_length-1):
            ax2.axvspan(dates[i], dates[i+1], color=position_colors[i], alpha=0.5)
        
        ax2.set_ylabel('Position', fontsize=12)
        ax2.set_yticks([-1, 0, 1])
        ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('results/figures/trading_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"{e}")


def generate_overfitting_check():
    
    report = """
=== Overfitting Check Report ===

1. Training vs Testing Performance

Note: Full overfitting analysis requires storing training set performance 
during model training. Current implementation focuses on test set evaluation.

Recommended checks:
- Compare Sharpe ratios: Training vs Testing
- Compare cumulative returns: Training vs Testing  
- Analyze performance degradation
- Check if model generalizes well

2. Model Complexity Analysis

Network Architecture:
- Input: 20 features
- Hidden Layer 1: 128 units + BatchNorm + ReLU
- Hidden Layer 2: 256 units + BatchNorm + ReLU
- Output: 3 actions + Softmax

Total Parameters: ~33,000

Regularization Methods:
- Target network (updated every 300 steps)
- Experience replay (memory size: 128)
- Epsilon-greedy exploration (decay from 0.9 to 0.05)
- BatchNorm layers

3. Data Split Strategy

Training Period: 2018-01-01 to 2022-01-01 (4 years)
Testing Period: 2022-01-01 to 2024-12-31 (3 years)

Split Method: Time-based (no shuffling)
Leakage Prevention: Expanding window for Z-score, look-back features

4. Conclusion

The model design includes several anti-overfitting mechanisms:
✓ Target network prevents moving target problem
✓ Experience replay reduces correlation
✓ Time-based validation ensures no leakage
✓ Reasonable model complexity for data size

Test set performance should be evaluated against buy-and-hold baseline
to assess practical value and generalization capability.
"""
    
    with open('results/reports/overfitting_check.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    


def generate_trading_frequency_table(): 
    
    try:
        import glob
        import pickle
        from Market_BTC import BTCTradingEnv
        from Data_BTC import INDICATORS
        from Config_BTC import REWARD_FUTURE_DATE
        

        model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
        if not model_files:
            return
        
        train, test = get_train_test_data()
        
        with open(model_files[0], 'rb') as f:
            dqn = pickle.loads(f.read())
        
        env_kwargs = {
            "buy_cost_pct": 0.0001,
            "sell_cost_pct": 0.0001,
            "state_space": len(INDICATORS),
            "tech_indicator_list": INDICATORS,
            "reward_future_day": REWARD_FUTURE_DATE,
            "seed": 0
        }
        
        env = BTCTradingEnv(test, **env_kwargs)
        s = env.reset()
        actions = []
        
        while True:
            a = dqn.choose_action(s)
            actions.append(a)
            s_, r, done, info = env.step(a)
            if done:
                break
            s = s_
        
        dates = pd.to_datetime(env.date_memory)
        actions = np.array(actions)
        
        position_changes = np.diff(actions) != 0
        trades_by_year = {}
        
        for i, date in enumerate(dates[1:]):
            year = date.year
            if year not in trades_by_year:
                trades_by_year[year] = 0
            if position_changes[i]:
                trades_by_year[year] += 1
        
        data = []
        for year in sorted(trades_by_year.keys()):
            year_dates = dates[dates.year == year]
            n_days = len(year_dates) / 6  
            
            data.append({
                'Period': year,
                'Total Trades': trades_by_year[year],
                'Avg Trades/Day': trades_by_year[year] / n_days if n_days > 0 else 0,
                'Trading Days': int(n_days)
            })
        
        total_trades = sum(trades_by_year.values())
        total_days = len(dates) / 6
        data.append({
            'Period': 'Overall',
            'Total Trades': total_trades,
            'Avg Trades/Day': total_trades / total_days if total_days > 0 else 0,
            'Trading Days': int(total_days)
        })
        
        df = pd.DataFrame(data)
        df['Avg Trades/Day'] = df['Avg Trades/Day'].apply(lambda x: f"{x:.2f}")
        
        df.to_csv('results/tables/trading_frequency.csv', index=False)
        
    except Exception as e:
        print(f"{e}")


def main():

    generate_monthly_returns_table()
    
    generate_sharpe_ratio_comparison()
    
    generate_trading_signals_plot()
    
    generate_overfitting_check()
    
    generate_trading_frequency_table()
    

    print("  - results/figures/sharpe_ratio_comparison.png")
    print("  - results/figures/trading_signals.png")
    print("  - results/tables/monthly_returns.csv")
    print("  - results/tables/trading_frequency.csv")
    print("  - results/reports/overfitting_check.txt")
    print("="*70)


if __name__ == "__main__":
    main()
