
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config_BTC import *
import matplotlib.pyplot as plt



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
     
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        self.softmax_out = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.out(x)
        actions_value = self.softmax_out(x)
        return actions_value


class DQN(object):
    def __init__(self, seed=0):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.train()
        self.target_net.eval()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=LR, betas=(0.9, 0.999)
        )
        self.loss_func = nn.SmoothL1Loss()
        self.loss = 0
        self.EPSILON = 0
        
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        
        self.EPSILON = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                       np.exp(-self.learn_step_counter / EPSILON_DECAY)
        
        if np.random.uniform() < self.EPSILON:
            self.eval_net.eval()
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        self.eval_net.train()
        
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        
        q_eval = self.eval_net(b_s).gather(1, b_a)
        
        q_next = self.target_net(b_s_).detach()
        
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        
        self.loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
        self._visualize()
    
    def _visualize(self, step=100):
        if USE_VISDOM and self.learn_step_counter % step == 0:
            vis.line(
                Y=torch.tensor([self.loss]),
                X=torch.tensor([self.learn_step_counter]),
                win="loss",
                env="BTC_DQN",
                update="append",
                name="train loss",
                opts={"title": "Training Loss"}
            )


def plot(env, save_place='BTC_DQN.png'):
    plot_df = pd.DataFrame(env.date_memory, columns=['datetime'])
    plot_df['return'] = env.return_list
    plot_df['close'] = env.df['close']
    
    plot_df['close_return'] = plot_df['close'].shift(-1) / plot_df['close'] - 1
    plot_df['return'] = plot_df['return'].shift(-1)
    
    plot_df.set_index('datetime', inplace=True)
    
    plot_df['cum_strategy'] = (1 + plot_df['return']).cumprod() - 1
    plot_df['cum_close'] = (1 + plot_df['close_return']).cumprod() - 1
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    ax.plot(plot_df.index, plot_df['cum_close'], label='Buy & Hold', linewidth=2)
    ax.plot(plot_df.index, plot_df['cum_strategy'], label='DQN Strategy', linewidth=2)
    ax.plot(plot_df.index, plot_df['cum_strategy'] - plot_df['cum_close'], 
            label='Excess Return', linewidth=1.5, linestyle='--')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.title('Bitcoin DQN Trading Strategy Performance', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_place, dpi=300)
    print(f"图表已保存: {save_place}")
    plt.close()


def train_DQN(env, seed=0):

    dqn = DQN(seed=seed)

    
    for i_episode in range(TRAIN_ITER):
        s = env.reset()
        ep_r = 0
        
        while True:
            a = dqn.choose_action(s)
            s_, r, done, info = env.step(a)
            

            dqn.store_transition(s, a, r, info)
            ep_r += r
            

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print(f'Episode: {i_episode+1} | Reward: {round(ep_r, 2)} | ' + 
                          f'Epsilon: {dqn.EPSILON:.4f}')
                    print('='*60)
            
            if done:
                break
            
            s = s_
        

        if (i_episode + 1) % 5 == 0:
            plot(env, save_place=f"results/BTC_DQN_train_ep{i_episode+1}.png")
    
    return dqn


def test_DQN(env_config, test_data, dqn):


    from Market_BTC import BTCTradingEnv
    
    env = BTCTradingEnv(test_data, **env_config)
    s = env.reset()
    ep_r = 0
    
    while True:
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)
        ep_r += r
        
        if done:
            break
        
        s = s_

    plot(env=env, save_place="results/BTC_DQN_TEST.png")
    
    return env
