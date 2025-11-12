try:
    from Market_BTC import BTCTradingEnv
except:
    BTCTradingEnv = None

try:
    from Data_BTC import INDICATORS
except:
    INDICATORS = [f'look_back_{i}_open' for i in range(5)] + \
                 [f'look_back_{i}_close' for i in range(5)] + \
                 [f'look_back_{i}_high' for i in range(5)] + \
                 [f'look_back_{i}_low' for i in range(5)]
    
import numpy as np
import torch

# 项目参数（超参数）
TRAIN_ITER = 30               # 学习迭代次数（比特币数据更多，可以训练更多轮）
BATCH_SIZE = 32               # 批次大小
LR = 0.0005                   # 学习率（比特币波动大，稍微降低学习率）
EPSILON_START = 0.9           # 探索率初始值
EPSILON_END = 0.05            # 探索率最终值
EPSILON_DECAY = 1000          # 探索率衰减速度
GAMMA = 0.9                   # 奖励折扣因子（稍微提高以考虑长期收益）
TARGET_REPLACE_ITER = 300     # Q网络更新频率（约50小时）
MEMORY_CAPACITY = 128          # 记忆库大小
REWARD_FUTURE_DATE = 4        # 根据未来第N个周期（24小时）的价格定奖励

# 环境配置
state_space = len(INDICATORS) if INDICATORS else 20
N_ACTIONS = 3                 # 3个动作：做空(0), 空仓(1), 做多(2)
N_STATES = len(INDICATORS) if INDICATORS else 20    # 状态空间维度
ENV_A_SHAPE = 0               # 动作形状

# 根据研报修正EPSILON（适配3个动作）
_EPSILON_START = EPSILON_START
_EPSILON_END = EPSILON_END
EPSILON_START = _EPSILON_START / N_ACTIONS + 1 - _EPSILON_START
EPSILON_END = _EPSILON_END / N_ACTIONS + 1 - _EPSILON_END

# 可视化配置（可选）
# 注意：如果没有安装visdom，可以注释掉这部分
try:
    import visdom
    vis = visdom.Visdom(env=u'BTC_DQN', use_incoming_socket=False)
    USE_VISDOM = True
except:
    print("Visdom未安装或无法连接，跳过可视化")
    vis = None
    USE_VISDOM = False

# 打印配置信息
if __name__ == "__main__":
    print("="*60)
    print("比特币DQN择时策略配置")
    print("="*60)
    print(f"\n训练参数:")
    print(f"  训练轮数: {TRAIN_ITER}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  学习率: {LR}")
    print(f"  折扣因子: {GAMMA}")
    print(f"  记忆库大小: {MEMORY_CAPACITY}")
    print(f"\n探索策略:")
    print(f"  初始探索率: {EPSILON_START:.4f}")
    print(f"  最终探索率: {EPSILON_END:.4f}")
    print(f"  衰减速度: {EPSILON_DECAY}")
    print(f"\n环境配置:")
    print(f"  状态维度: {N_STATES}")
    print(f"  动作数量: {N_ACTIONS} (做空/空仓/做多)")
    print(f"  奖励时间窗口: {REWARD_FUTURE_DATE} 个4h周期 ({REWARD_FUTURE_DATE*4}小时)")
    print(f"  目标网络更新频率: {TARGET_REPLACE_ITER} 步")
    print(f"\n特征列表 ({len(INDICATORS)}个):")
    print(f"  {INDICATORS[:4]}...等")
    print("="*60)
