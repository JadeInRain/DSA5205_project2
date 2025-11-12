# 比特币数据加载和预处理模块
import pandas as pd
import numpy as np
import os

# 数据参数
train_start_date = '2018-01-01'
train_stop_date = '2022-01-01'
val_start_date = '2022-01-01'
val_stop_date = '2024-12-31'
look_back = 5  # 回看周期数

# 特征列表：回看的开高低收价格
INDICATORS = [f'look_back_{i}_open' for i in range(look_back)] + \
             [f'look_back_{i}_close' for i in range(look_back)] + \
             [f'look_back_{i}_high' for i in range(look_back)] + \
             [f'look_back_{i}_low' for i in range(look_back)]


def data_split(df, start, end, target_date_col="datetime"):
    """
    分割数据集
    :param df: DataFrame
    :param start: 开始日期
    :param end: 结束日期
    :param target_date_col: 日期列名
    :return: 分割后的DataFrame
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def load_btc_data(csv_file=None):
    """
    加载比特币4小时K线数据并进行特征工程
    使用Z-score标准化方法（参考研报）
    
    :param csv_file: CSV文件路径，如果为None则自动搜索
    :return: 预处理后的DataFrame
    """
    # 如果没有指定文件，尝试多个可能的路径
    if csv_file is None:
        possible_paths = [
            'binance_btc_usdt_4h_full_history.csv',
            '../binance_btc_usdt_4h_full_history.csv',
            '../../binance_btc_usdt_4h_full_history.csv',
        ]
        
        csv_file = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_file = path
                break
        
        if csv_file is None:
            raise FileNotFoundError(
                "找不到比特币数据文件。请确保 binance_btc_usdt_4h_full_history.csv "
                "存在于当前目录或父目录中"
            )
    
    print(f"加载比特币数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 处理时间列
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['open_time'])
    else:
        raise ValueError("数据中没有找到时间列")
    
    # 选择需要的列
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # 按时间排序
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"原始数据形状: {df.shape}")
    print(f"日期范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    
    # 特征工程：Z-score标准化（完全按照研报方法）
    # 使用expanding窗口（累计扩展），从第252个周期开始计算
    # 研报使用的是252个交易日，这里用252个4h周期（约6周）
    window = 252
    
    print("\n进行Z-score标准化（expanding窗口）...")
    for i in range(look_back):
        # 对开高低收价格分别进行Z-score标准化
        # (当前值 - 累计均值) / 累计标准差
        # 使用expanding(window)表示从第window个数据点开始计算累计统计量
        df[f'look_back_{i}_open'] = (df['open'].shift(i) - df['close'].expanding(window).mean()) / \
                                     (df['close'].expanding(window).std() + 1e-10)
        
        df[f'look_back_{i}_close'] = (df['close'].shift(i) - df['close'].expanding(window).mean()) / \
                                      (df['close'].expanding(window).std() + 1e-10)
        
        df[f'look_back_{i}_high'] = (df['high'].shift(i) - df['close'].expanding(window).mean()) / \
                                     (df['close'].expanding(window).std() + 1e-10)
        
        df[f'look_back_{i}_low'] = (df['low'].shift(i) - df['close'].expanding(window).mean()) / \
                                    (df['close'].expanding(window).std() + 1e-10)
    
    # 删除NaN值
    original_len = len(df)
    df.dropna(inplace=True)
    print(f"删除NaN后: {len(df)} 行 (删除了 {original_len - len(df)} 行)")
    
    # 添加标识列（模仿原代码）
    df['tic'] = 'BTC-USDT'
    
    return df


def get_train_test_data(csv_file=None):
    """
    获取比特币数据并分割为训练集和测试集
    
    :param csv_file: CSV文件路径，如果为None则自动搜索
    :return: (训练集, 测试集) DataFrame元组
    """
    try:
        df = load_btc_data(csv_file)
        
        # 分割数据
        train = data_split(df, train_start_date, train_stop_date)
        test = data_split(df, val_start_date, val_stop_date)
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train)} 行, {train['datetime'].min()} 至 {train['datetime'].max()}")
        print(f"  测试集: {len(test)} 行, {test['datetime'].min()} 至 {test['datetime'].max()}")
        
        return train, test
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None


# 如果直接运行此文件，则加载数据并显示
if __name__ == "__main__":
    train, test = get_train_test_data()
    
    if train is not None and test is not None:
        print("\n训练集特征示例:")
        print(train[INDICATORS].head())
        print("\n测试集特征示例:")
        print(test[INDICATORS].head())
        
        # 可选：保存为CSV
        # train.to_csv('btc_train_data.csv', index=False)
        # test.to_csv('btc_test_data.csv', index=False)
        print("\n数据加载完成！")
    else:
        print("\n数据加载失败！请检查数据文件路径。")

# 供其他模块导入使用
# 注意：导入时不会自动加载数据，避免导入错误
# 其他模块应该调用 get_train_test_data() 来获取数据
train = None
test = None
