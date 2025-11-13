# Project 2: ML Trading Strategy for Bitcoin

This project implements a complete trading strategy for Bitcoin (BTC) on a 4-hour time frame. The process consists of two main stages:

1.  **Data Processing (`data_process.py`):** This script takes raw data from multiple sources (derivatives data, Fear & Greed index, on-chain data, and OHLCV market data), merges them, cleans, normalizes, and saves them as a ready-to-use dataset.
2.  **Modeling and Backtesting (`main.py`):** This script loads the processed data, creates additional technical indicators, trains a Random Forest model to predict price direction, and then runs an out-of-sample backtest to evaluate the strategy's performance.

## 