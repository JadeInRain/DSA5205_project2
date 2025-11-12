# Bitcoin DQN Trading Strategy

Deep Q-Network (DQN) reinforcement learning strategy for Bitcoin 4-hour trading with short-selling capability.

## Project Overview

This project implements a DQN-based algorithmic trading system for Bitcoin (BTC/USDT) using 4-hour candlestick data. The agent can execute long, short, and neutral positions, enabling profit opportunities in both bull and bear markets.

**Key Features:**
- Three-action space: Long (+1), Neutral (0), Short (-1) positions
- Expanding-window Z-score normalization to prevent look-ahead bias
- Experience replay and target network for stable training
- Comprehensive out-of-sample evaluation on 2022-2024 test data
- Transaction cost modeling (0.01% per trade)

## Project Structure

### Core Files

#### 1. **Config_BTC.py**
Configuration file containing all hyperparameters:
- Training parameters: iterations, batch size, learning rate
- DQN parameters: discount factor (gamma), memory capacity
- Exploration parameters: epsilon-greedy decay schedule
- Environment parameters: reward horizon, transaction costs
- Network architecture specifications

#### 2. **Data_BTC.py**
Data loading and preprocessing module:
- Loads Bitcoin 4-hour OHLCV data from CSV
- Splits data into training (2018-2021) and testing (2022-2024) sets
- Applies expanding-window Z-score normalization
- Generates technical features (20 features from OHLC history)
- Prevents look-ahead bias through strict temporal ordering

#### 3. **Market_BTC.py**
Trading environment (OpenAI Gym compatible):
- Simulates Bitcoin spot and futures trading
- Supports long/short/neutral position management
- Calculates rewards based on future price movements
- Tracks performance metrics: returns, positions, trades
- Implements transaction costs (0.01% maker fee)
- Maintains buy-and-hold benchmark for comparison

#### 4. **Agent_BTC.py**
DQN agent implementation:
- Neural network architecture: FC(20→128→256→3)
- Experience replay buffer for breaking temporal correlations
- Target network updated every 300 steps
- Epsilon-greedy exploration with exponential decay
- Adam optimizer with learning rate 0.0005
- Training and testing functions

#### 5. **Start_BTC.py**
Main execution script:
- **Mode 1**: Train single model with specified seed
- **Mode 2**: Train ensemble of models (different seeds) with majority voting
- **Mode 3**: Load and test pre-trained models
- **Mode 4**: Test all trained models in batch
- Generates performance visualizations
- Calculates comprehensive metrics (Sharpe, drawdown, etc.)

#### 6. **generate_analysis.py**
Performance analysis and reporting:
- Loads trained models and runs backtests on test set
- Generates comparison tables (DQN vs. Buy-and-Hold)
- Creates visualizations:
  - Cumulative return comparison
  - Drawdown analysis
  - Position distribution charts
- Calculates risk metrics: Sharpe, Sortino, Calmar, VaR, CVaR
- Outputs results to `results/` directory

#### 7. **generate_advanced_analysis.py**
Advanced analytical tools:
- Monthly returns breakdown
- Rolling Sharpe ratio analysis
- Trading signal visualization
- Overfitting checks
- Trading frequency statistics
- Generates supplementary materials for research report

## Input Files

### Required Data File

**File:** `binance_btc_usdt_4h_full_history.csv`  
**Location:** Parent directory or specified path in Data_BTC.py

**Format:** CSV file with the following columns:
- `open_time` or `timestamp`: Unix timestamp or datetime
- `open`: Opening price (USDT)
- `high`: Highest price in period (USDT)
- `low`: Lowest price in period (USDT)
- `close`: Closing price (USDT)
- `volume`: Trading volume (BTC)

**Data Specifications:**
- Frequency: 4-hour candlesticks
- Coverage: August 2017 to November 2024 (17,999 data points)
- Training period: 2018-01-01 to 2021-12-31
- Testing period: 2022-01-01 to 2024-12-31

## Output Files

### 1. Trained Models

**Directory:** `models/`

Files:
- `btc_dqn_seed_0.pkl` - Model trained with seed 0
- `btc_dqn_seed_10.pkl` - Model trained with seed 10
- `btc_dqn_seed_20.pkl` - Model trained with seed 20
- ... (additional seeds for ensemble)

Format: Pickled PyTorch DQN objects

### 2. Visualization Outputs

**Directory:** `results/figures/`

Files:
- `cumulative_return_comparison.png` - Cumulative returns (DQN vs. Buy-Hold)
- `drawdown_comparison.png` - Drawdown analysis over time
- `position_distribution.png` - Bar chart of position allocation
- `BTC_DQN_TEST_seed*.png` - Individual model test results
- `BTC_DQN_train_ep*.png` - Training progress snapshots
- `sharpe_ratio_comparison.png` - Sharpe across different windows
- `trading_signals.png` - Price chart with trading signals

### 3. Performance Tables

**Directory:** `results/tables/`

Files:
- `model_comparison_table.csv` - Performance metrics for all models
- `risk_metrics.csv` - Comprehensive risk analysis (Sharpe, VaR, CVaR, etc.)
- `monthly_returns.csv` - Month-by-month return breakdown
- `trading_frequency.csv` - Trading statistics by period

### 4. Text Reports

**Directory:** `results/reports/`

Files:
- `performance_analysis_report.txt` - Detailed performance summary
- `overfitting_check.txt` - Train vs. test comparison analysis

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=1.9.0` - PyTorch for neural networks
- `numpy>=1.20.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Visualization
- `gym>=0.21.0` - Reinforcement learning environment

### 2. Prepare Data

Place the Bitcoin data file in the parent directory:
```
Project2/binance_btc_usdt_4h_full_history.csv
```

Or modify the path in `Data_BTC.py` line 12-15.

### 3. Configure Parameters (Optional)

Edit `Config_BTC.py` to adjust:
- Training iterations: `TRAIN_ITER = 20`
- Discount factor: `GAMMA = 0.9`
- Learning rate: `LR = 0.0005`
- Reward horizon: `REWARD_FUTURE_DATE = 4`

## Usage

### Training a Single Model

```bash
python Start_BTC.py
```

Set `mode = 1` in `Start_BTC.py` to train one model with default parameters.

### Training Ensemble Models

```bash
python Start_BTC.py
```

Set `mode = 2` to train multiple models with different random seeds. The ensemble uses majority voting for final predictions.

### Generating Analysis Reports

```bash
python generate_analysis.py
```

This will:
1. Load all trained models from `models/`
2. Run backtests on 2022-2024 test data
3. Generate all figures and tables
4. Output comprehensive performance report

### Advanced Analysis

```bash
python generate_advanced_analysis.py
```

Generates supplementary materials including monthly returns, rolling metrics, and trading signal visualizations.

## Performance Metrics

The system calculates the following metrics:

**Return Metrics:**
- Cumulative Return
- Annualized Return
- Excess Return over Buy-and-Hold

**Risk-Adjusted Metrics:**
- Sharpe Ratio (annualized)
- Sortino Ratio (downside deviation)
- Calmar Ratio (return/max drawdown)

**Risk Metrics:**
- Maximum Drawdown
- Volatility (annualized)
- Value-at-Risk (VaR 95%)
- Conditional VaR (CVaR 95%)

**Trading Statistics:**
- Total number of trades
- Win rate
- Position distribution (long/short/neutral percentages)
- Average holding period

## Algorithm Details

### DQN Architecture

```
Input (20 features)
    ↓
Fully Connected (128 units) + BatchNorm + ReLU
    ↓
Fully Connected (256 units) + BatchNorm + ReLU
    ↓
Output (3 Q-values for actions: short, neutral, long)
```

### Training Process

1. **Experience Collection**: Agent interacts with environment
2. **Memory Storage**: Transitions stored in replay buffer (capacity: 128)
3. **Batch Sampling**: Random mini-batches (size: 32) sampled
4. **Q-Learning Update**: Minimize TD error using target network
5. **Target Network Update**: Copy weights every 300 steps
6. **Exploration Decay**: Epsilon decays from 0.9 to 0.05

### Reward Function

```
reward = position_return - transaction_cost

where:
  position_return = (P_t+4 / P_t - 1)  for long
                  = (1 - P_t+4 / P_t)  for short
                  = 0                   for neutral
  
  transaction_cost = 0.01% per trade
```

## Reproducibility

For exact reproduction of results:
1. Use the same data file with identical preprocessing
2. Set specific random seeds (0, 10, 20, ..., 70 for ensemble)
3. Use hyperparameters specified in `Config_BTC.py`
4. Run on CPU (no GPU required)
5. Python 3.8+, PyTorch 1.9+

All random seeds are explicitly set for:
- PyTorch (`torch.manual_seed()`)
- NumPy (`np.random.seed()`)
- Environment initialization

## Notes and Limitations

**Assumptions:**
- Market orders executed at close price (no slippage modeled)
- Sufficient liquidity (no market impact)
- Funding rates ignored for perpetual futures
- No position sizing (only -1/0/+1 positions)

**Best Practices:**
- Monitor overfitting: compare train vs. test performance
- Use ensemble methods to reduce variance
- Validate on multiple time periods if possible
- Consider transaction costs in live trading scenarios

## License

This project is for educational and research purposes only. Not financial advice.

## Citation

If you use this code in your research, please cite:
```
Bitcoin DQN Trading Strategy
National University of Singapore
Project 2: Machine Learning for Trading
2024
```
