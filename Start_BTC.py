"""
比特币DQN择时策略 - 主程序
用于训练和测试DQN模型
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
import pandas as pd
import pickle
import torch
import numpy as np
from Market_BTC import BTCTradingEnv
from Agent_BTC import train_DQN, test_DQN, plot
from Data_BTC import get_train_test_data, INDICATORS
from Config_BTC import REWARD_FUTURE_DATE

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

print("正在加载比特币数据...")
train, test = get_train_test_data()

if train is None or test is None:
    exit(1)

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)


def create_model(save_place: str, seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    

    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * len(INDICATORS)
    
    
    env_kwargs = {
        "buy_cost_pct": 0.001,     
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
        "seed": seed
    }
    
    
    e_train_gym = BTCTradingEnv(df=train, **env_kwargs)
    
    dqn = train_DQN(e_train_gym, seed=seed)
    
    
    if save_place.endswith('.pkl'):
        with open(save_place, 'wb') as out_put:
            dqn_pkl = pickle.dumps(dqn)
            out_put.write(dqn_pkl)
    
    
    test_env = test_DQN(env_kwargs, test, dqn)
    
    calculate_performance(test_env)
    
    return dqn


def load_model(save_place: str, seed, generate_plot=True, plot_suffix=""):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with open(save_place, 'rb') as file:
        dqn = pickle.loads(file.read())
    
    env_kwargs = {
        "buy_cost_pct": 0.0001,
        "sell_cost_pct": 0.0001,
        "state_space": len(INDICATORS),
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
        "seed": seed
    }
    
    env = BTCTradingEnv(test, **env_kwargs)
    s = env.reset()
    action_list = []
    
    while True:
        a = dqn.choose_action(s)
        action_list.append(a)
        s_, r, done, info = env.step(a)
        
        if done:
            break
        
        s = s_
    
    if generate_plot:
        plot_name = f"results/BTC_DQN_TEST{plot_suffix}.png"
        plot(env, plot_name)
        
        calculate_performance(env)
    
    return action_list, env


def calculate_performance(env):
    df = pd.DataFrame({
        'datetime': env.date_memory,
        'return': env.return_list
    })
    
    close_prices = env.df['close'].values[:len(df)]
    buy_hold_returns = np.diff(close_prices) / close_prices[:-1]
    buy_hold_returns = np.insert(buy_hold_returns, 0, 0)
    df['buy_hold_return'] = buy_hold_returns[:len(df)]
    
    strategy_total_return = (1 + df['return']).prod() - 1
    buy_hold_total_return = (1 + df['buy_hold_return']).prod() - 1
    
  
    n_periods = len(df)
    years = n_periods / 2190
    strategy_annual_return = (1 + strategy_total_return) ** (1/years) - 1
    buy_hold_annual_return = (1 + buy_hold_total_return) ** (1/years) - 1
    

    if df['return'].std() > 0:
        sharpe_ratio = (df['return'].mean() / df['return'].std()) * np.sqrt(2190)
    else:
        sharpe_ratio = 0
    
    if df['buy_hold_return'].std() > 0:
        buy_hold_sharpe = (df['buy_hold_return'].mean() / df['buy_hold_return'].std()) * np.sqrt(2190)
    else:
        buy_hold_sharpe = 0
 
    cum_returns = (1 + df['return']).cumprod()
    cum_max = cum_returns.expanding().max()
    drawdown = (cum_returns - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    cum_bh_returns = (1 + df['buy_hold_return']).cumprod()
    cum_bh_max = cum_bh_returns.expanding().max()
    bh_drawdown = (cum_bh_returns - cum_bh_max) / cum_bh_max
    bh_max_drawdown = bh_drawdown.min()
    
    
    total_periods = env.position_stats['long'] + env.position_stats['short'] + env.position_stats['neutral']

    

def test_all_models():
    import glob
    
    model_files = glob.glob("models/btc_dqn_seed_*.pkl")
    
    if not model_files:

        return
    

    
    for model_path in sorted(model_files):
 
        import re
        match = re.search(r'seed_(\d+)', model_path)
        if match:
            seed = int(match.group(1))
        else:
            seed = 0
        
        try:
            actions, env = load_model(model_path, seed=seed, generate_plot=True,
                                     plot_suffix=f"_seed{seed}")
        except Exception as e:
            print(f" {e}")
            continue


def ensemble_models(n_models=5):
    
    model2signal = pd.DataFrame()
    
    
    for i in range(n_models):
        seed = 10
        model_path = f"models/btc_dqn_seed_{seed}.pkl"
        
        if not os.path.exists(model_path):
            create_model(model_path, seed=seed)
        
        actions, _ = load_model(model_path, seed=seed, generate_plot=True, 
                                plot_suffix=f"_seed{seed}")
        model2signal[f'seed{seed}'] = actions
    
    
    final_signal = model2signal.mode(axis=1)[0].astype(int).tolist()
    
    
    env_kwargs = {
        "buy_cost_pct": 0.0001,
        "sell_cost_pct": 0.0001,
        "state_space": len(INDICATORS),
        "tech_indicator_list": INDICATORS,
        "reward_future_day": REWARD_FUTURE_DATE,
    }
    
    env = BTCTradingEnv(test, **env_kwargs)
    s = env.reset()
    step = 0
    
    while True:
        a = final_signal[step]
        s_, r, done, info = env.step(a)
        if done:
            break
        step += 1
    
    plot(env, "results/BTC_DQN_ensemble.png")
    calculate_performance(env)


if __name__ == "__main__":

    mode = 2
    
    if mode == 1:
        create_model(f"models/btc_dqn_seed_10.pkl", seed=10)
    
    elif mode == 2:
        ensemble_models(n_models=1)
    
    elif mode == 3:
        
        models_to_test = [
            ("models/btc_dqn_seed_0.pkl", 0),

        ]
        
        for model_path, seed in models_to_test:
            if os.path.exists(model_path):
                actions, env = load_model(model_path, seed=seed, generate_plot=True,
                                         plot_suffix=f"_seed{seed}")
            else:
                print(f"{model_path}")
        

    
    elif mode == 4:
        test_all_models()
    

