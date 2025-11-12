"""
Generate comprehensive performance analysis reports, charts, and data tables
For Project 2 report materials
"""

# Fix OpenMP library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import glob
from datetime import datetime

# Set matplotlib style (no Chinese fonts needed for English output)
plt.style.use('seaborn-v0_8-whitegrid')

# Create output directories
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)
os.makedirs('results/reports', exist_ok=True)


def load_model_results(model_path):
    """
    Load model and run backtest, collect performance metrics
    """
    from Market_BTC import BTCTradingEnv
    from Data_BTC import get_train_test_data, INDICATORS
    from Config_BTC import REWARD_FUTURE_DATE
    import torch
    
    # Load data
    train, test = get_train_test_data()
    
    # Extract seed from filename
    import re
    match = re.search(r'seed_(\d+)', model_path)
    seed = int(match.group(1)) if match else 0
    
    # Load model
    with open(model_path, 'rb') as f:
        dqn = pickle.loads(f.read())
    
    # Setup environment
    env_kwargs = {
        "buy_cost_pct": 0.0001,
        "sell_cost_pct": -0.0001,
        "state_space": len(INDICATORS),
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
        "seed": seed
    }
    
    # Backtest on test set
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
    
    # Collect results
    results = {
        'model_path': model_path,
        'seed': seed,
        'returns': np.array(env.return_list),
        'dates': pd.to_datetime(env.date_memory),
        'actions': np.array(actions),
        'buy_hold_returns': np.array(env.buy_hold_list),
        'trades': env.trades,
        'position_stats': env.position_stats.copy()
    }
    
    return results


def calculate_metrics(returns, dates):
    """
    Calculate complete performance metrics
    """
    # Cumulative return
    cum_return = (1 + returns).prod() - 1
    
    # Annualized return
    n_periods = len(returns)
    years = n_periods / 2190  # 4h data
    annual_return = (1 + cum_return) ** (1/years) - 1 if years > 0 else 0
    
    # Sharpe ratio
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(2190)
    else:
        sharpe = 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    # Volatility
    volatility = returns.std() * np.sqrt(2190)
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = (returns.mean() / downside_returns.std()) * np.sqrt(2190)
    else:
        sortino = 0
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # VaR and CVaR (95%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    metrics = {
        'cumulative_return': cum_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate
    }
    
    return metrics


def generate_comparison_table():
    """
    Generate model comparison table
    """
    print("\nGenerating model comparison table...")
    
    # Find all models
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        print("No model files found!")
        return
    
    results_list = []
    
    for model_path in model_files:
        print(f"  Processing: {model_path}")
        try:
            result = load_model_results(model_path)
            metrics = calculate_metrics(result['returns'], result['dates'])
            
            results_list.append({
                'Model': f"seed_{result['seed']}",
                'Cumulative Return': f"{metrics['cumulative_return']:.2%}",
                'Annual Return': f"{metrics['annual_return']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Volatility': f"{metrics['volatility']:.2%}",
                'Total Trades': result['trades'],
                'Long%': f"{result['position_stats']['long']/(result['position_stats']['long']+result['position_stats']['short']+result['position_stats']['neutral']):.2%}",
                'Short%': f"{result['position_stats']['short']/(result['position_stats']['long']+result['position_stats']['short']+result['position_stats']['neutral']):.2%}"
            })
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    # Add Buy-and-Hold benchmark
    if results_list:
        # Use buy_hold data from first model
        result = load_model_results(model_files[0])
        bh_metrics = calculate_metrics(result['buy_hold_returns'], result['dates'])
        
        results_list.append({
            'Model': 'Buy-Hold',
            'Cumulative Return': f"{bh_metrics['cumulative_return']:.2%}",
            'Annual Return': f"{bh_metrics['annual_return']:.2%}",
            'Sharpe Ratio': f"{bh_metrics['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{bh_metrics['max_drawdown']:.2%}",
            'Volatility': f"{bh_metrics['volatility']:.2%}",
            'Total Trades': 0,
            'Long%': '100.00%',
            'Short%': '0.00%'
        })
    
    # Save table
    df = pd.DataFrame(results_list)
    df.to_csv('results/tables/model_comparison_table.csv', index=False)
    print(f"✓ Saved: results/tables/model_comparison_table.csv")
    
    return df


def generate_cumulative_return_plot():
    """
    Generate cumulative return comparison plot
    """
    print("\nGenerating cumulative return comparison plot...")
    
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot all models
    for i, model_path in enumerate(model_files[:5]):  # Maximum 5 models
        try:
            result = load_model_results(model_path)
            cum_returns = (1 + result['returns']).cumprod() - 1
            plt.plot(result['dates'], cum_returns * 100, 
                    label=f"DQN seed_{result['seed']}", alpha=0.7, linewidth=1.5)
        except:
            continue
    
    # Plot Buy-and-Hold
    try:
        result = load_model_results(model_files[0])
        bh_cum_returns = (1 + result['buy_hold_returns']).cumprod() - 1
        plt.plot(result['dates'], bh_cum_returns * 100, 
                label='Buy-and-Hold', color='blue', linewidth=2, linestyle='--')
    except:
        pass
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.title('Cumulative Return Comparison (Test Set: 2022-2024)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/figures/cumulative_return_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/figures/cumulative_return_comparison.png")


def generate_drawdown_plot():
    """
    Generate drawdown comparison plot
    """
    print("\nGenerating drawdown comparison plot...")
    
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Plot strategy drawdown
    try:
        result = load_model_results(model_files[0])
        cum_returns = (1 + result['returns']).cumprod()
        cum_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - cum_max) / cum_max
        plt.plot(result['dates'], drawdown * 100, label='DQN Strategy', color='red', linewidth=2)
        
        # Mark maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        plt.scatter(result['dates'][max_dd_idx], drawdown[max_dd_idx] * 100, 
                   color='darkred', s=100, zorder=5, label=f'Max DD: {drawdown[max_dd_idx]:.2%}')
    except:
        pass
    
    # Plot Buy-and-Hold drawdown
    try:
        bh_cum_returns = (1 + result['buy_hold_returns']).cumprod()
        bh_cum_max = np.maximum.accumulate(bh_cum_returns)
        bh_drawdown = (bh_cum_returns - bh_cum_max) / bh_cum_max
        plt.plot(result['dates'], bh_drawdown * 100, label='Buy-and-Hold', 
                color='blue', linewidth=2, linestyle='--')
        
        # Mark maximum drawdown
        bh_max_dd_idx = np.argmin(bh_drawdown)
        plt.scatter(result['dates'][bh_max_dd_idx], bh_drawdown[bh_max_dd_idx] * 100, 
                   color='darkblue', s=100, zorder=5, label=f'BH Max DD: {bh_drawdown[bh_max_dd_idx]:.2%}')
    except:
        pass
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title('Drawdown Comparison (Test Set: 2022-2024)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('results/figures/drawdown_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: results/figures/drawdown_comparison.png")


def generate_position_distribution():
    """
    Generate position distribution plot
    """
    print("\nGenerating position distribution plot...")
    
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        return
    
    try:
        result = load_model_results(model_files[0])
        stats = result['position_stats']
        
        total = stats['long'] + stats['short'] + stats['neutral']
        
        labels = ['Long', 'Short', 'Neutral']
        sizes = [stats['long']/total*100, stats['short']/total*100, stats['neutral']/total*100]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.05, 0.05, 0.05)
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
        plt.title('Position Distribution (Test Set)', fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        plt.savefig('results/figures/position_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: results/figures/position_distribution.png")
    except Exception as e:
        print(f"    Error: {e}")


def generate_performance_report():
    """
    Generate complete performance analysis report
    """
    print("\nGenerating performance analysis report...")
    
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        return
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("BTC DQN STRATEGY PERFORMANCE REPORT")
    report_lines.append("="*70)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*70)
    report_lines.append("1. DATA INFORMATION")
    report_lines.append("="*70)
    report_lines.append("Asset: BTC/USDT")
    report_lines.append("Frequency: 4-hour")
    report_lines.append("Training Period: 2018-01-01 to 2022-01-01")
    report_lines.append("Testing Period: 2022-01-01 to 2024-12-31")
    
    # Process each model
    for i, model_path in enumerate(model_files):
        try:
            result = load_model_results(model_path)
            metrics = calculate_metrics(result['returns'], result['dates'])
            
            report_lines.append(f"\n{'='*70}")
            report_lines.append(f"MODEL {i+1}: seed_{result['seed']}")
            report_lines.append(f"{'='*70}")
            
            report_lines.append("\nOut-of-Sample Performance:")
            report_lines.append(f"  Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
            report_lines.append(f"  Annual Return:        {metrics['annual_return']:>10.2%}")
            report_lines.append(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>10.3f}")
            report_lines.append(f"  Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
            report_lines.append(f"  Volatility (Annual):  {metrics['volatility']:>10.2%}")
            report_lines.append(f"  Sortino Ratio:        {metrics['sortino_ratio']:>10.3f}")
            report_lines.append(f"  Calmar Ratio:         {metrics['calmar_ratio']:>10.3f}")
            report_lines.append(f"  VaR (95%):            {metrics['var_95']:>10.2%}")
            report_lines.append(f"  CVaR (95%):           {metrics['cvar_95']:>10.2%}")
            report_lines.append(f"  Win Rate:             {metrics['win_rate']:>10.2%}")
            
            report_lines.append(f"\nTrading Statistics:")
            report_lines.append(f"  Total Trades:         {result['trades']:>10d}")
            
            total = result['position_stats']['long'] + result['position_stats']['short'] + result['position_stats']['neutral']
            report_lines.append(f"  Long Position:        {result['position_stats']['long']/total:>10.2%}")
            report_lines.append(f"  Short Position:       {result['position_stats']['short']/total:>10.2%}")
            report_lines.append(f"  Neutral Position:     {result['position_stats']['neutral']/total:>10.2%}")
        
        except Exception as e:
            print(f"Error processing model {model_path}: {e}")
            continue
    
    # Add Buy-and-Hold benchmark
    try:
        result = load_model_results(model_files[0])
        bh_metrics = calculate_metrics(result['buy_hold_returns'], result['dates'])
        
        report_lines.append(f"\n{'='*70}")
        report_lines.append("BENCHMARK: BUY-AND-HOLD")
        report_lines.append(f"{'='*70}")
        report_lines.append(f"\n  Cumulative Return:    {bh_metrics['cumulative_return']:>10.2%}")
        report_lines.append(f"  Annual Return:        {bh_metrics['annual_return']:>10.2%}")
        report_lines.append(f"  Sharpe Ratio:         {bh_metrics['sharpe_ratio']:>10.3f}")
        report_lines.append(f"  Max Drawdown:         {bh_metrics['max_drawdown']:>10.2%}")
        report_lines.append(f"  Volatility (Annual):  {bh_metrics['volatility']:>10.2%}")
    except:
        pass
    
    report_lines.append("\n" + "="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    # Save report
    report_text = "\n".join(report_lines)
    with open('results/reports/performance_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("✓ Saved: results/reports/performance_analysis_report.txt")
    
    # Also print to console
    print("\n" + report_text)


def generate_risk_metrics_table():
    """
    Generate risk metrics comparison table
    """
    print("\nGenerating risk metrics table...")
    
    model_files = sorted(glob.glob("models/btc_dqn_seed_*.pkl"))
    
    if not model_files:
        return
    
    # Calculate strategy metrics (using first model as representative)
    result = load_model_results(model_files[0])
    strategy_metrics = calculate_metrics(result['returns'], result['dates'])
    
    # Calculate benchmark metrics
    bh_metrics = calculate_metrics(result['buy_hold_returns'], result['dates'])
    
    # Create table
    risk_data = {
        'Metric': [
            'Volatility (Annual)',
            'Max Drawdown',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'VaR (95%)',
            'CVaR (95%)',
            'Win Rate'
        ],
        'Strategy': [
            f"{strategy_metrics['volatility']:.2%}",
            f"{strategy_metrics['max_drawdown']:.2%}",
            f"{strategy_metrics['sharpe_ratio']:.3f}",
            f"{strategy_metrics['sortino_ratio']:.3f}",
            f"{strategy_metrics['calmar_ratio']:.3f}",
            f"{strategy_metrics['var_95']:.2%}",
            f"{strategy_metrics['cvar_95']:.2%}",
            f"{strategy_metrics['win_rate']:.2%}"
        ],
        'Benchmark': [
            f"{bh_metrics['volatility']:.2%}",
            f"{bh_metrics['max_drawdown']:.2%}",
            f"{bh_metrics['sharpe_ratio']:.3f}",
            f"{bh_metrics['sortino_ratio']:.3f}",
            f"{bh_metrics['calmar_ratio']:.3f}",
            f"{bh_metrics['var_95']:.2%}",
            f"{bh_metrics['cvar_95']:.2%}",
            f"{bh_metrics['win_rate']:.2%}"
        ]
    }
    
    df = pd.DataFrame(risk_data)
    df.to_csv('results/tables/risk_metrics.csv', index=False)
    print("✓ Saved: results/tables/risk_metrics.csv")


def main():
    """
    Main function: generate all analysis materials
    """
    print("\n" + "="*70)
    print("Starting to generate Project 2 report materials")
    print("="*70)
    
    # 1. Generate comparison table
    generate_comparison_table()
    
    # 2. Generate charts
    generate_cumulative_return_plot()
    generate_drawdown_plot()
    generate_position_distribution()
    
    # 3. Generate report
    generate_performance_report()
    
    # 4. Generate risk metrics table
    generate_risk_metrics_table()
    
    print("\n" + "="*70)
    print("All materials generated!")
    print("="*70)
    print("\nOutput file locations:")
    print("  - Charts: results/figures/")
    print("  - Tables: results/tables/")
    print("  - Reports: results/reports/")
    print("\nSee 'README.md' for complete materials list")
    print("="*70)


if __name__ == "__main__":
    main()
