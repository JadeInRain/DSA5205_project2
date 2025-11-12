import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
matplotlib.use("Agg")


# BTC Trading Environment
# state: position + current close + technical indicators (features)
# action: buy(long), hold, sell(short)  -> corresponding to action 2, 1, 0
# reward: return after reward_future_day periods
# Key difference: Bitcoin allows shorting, position can be -1(short), 0(neutral), 1(long)

class BTCTradingEnv(gym.Env):
    """
    Bitcoin Trading Environment (with shorting support)
    
    Main differences from original stock environment:
    1. action space: 3 actions (short, neutral, long)
    2. Support short position (position = -1)
    3. Calculate short position returns
    4. 4h frequency continuous data
    """
    
    def __init__(
        self, df, buy_cost_pct, sell_cost_pct, state_space, tech_indicator_list,
        day=0, initial=True, seed=0, reward_future_day=5
    ):
        self.day = day                              # Current trading period
        self.df = df                                # Trading data
        self.buy_cost_pct = buy_cost_pct            # Buying transaction cost
        self.sell_cost_pct = sell_cost_pct          # Selling transaction cost
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list  # List of technical indicator names
        
        # Key change: 3 actions - 0:short, 1:neutral, 2:long
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        
        self.data = self.df.loc[self.day, :]
        self.terminal = False                      # Whether reached terminal state
        self.initial = initial
        self.reward_future_day = reward_future_day
        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0
        self.trades = 0
        self.episode = 0
        
        # memorize all the total balance change
        self.date_memory = [self._get_date()]
        self._seed(seed=seed)
        self.return_list = [0]                       # Strategy returns list
        self.buy_hold_list = [0]                     # Buy-and-hold returns list
        
        # Position statistics
        self.position_stats = {'long': 0, 'short': 0, 'neutral': 0}
    
    def _buy_and_sell(self, actions):
        """
        Execute trade
        :param actions: int 0:short, 1:neutral, 2:long
        
        Position definition:
        self.state[0] = -1: short
        self.state[0] = 0:  neutral
        self.state[0] = 1:  long
        """
        current_position = self.state[0]
        
        # Map action to target position (0->-1, 1->0, 2->1)
        target_position = actions - 1
        
        # Record trade if position changes
        if current_position != target_position:
            self.state[0] = target_position
            self.trades += 1
    
    def step(self, actions):
        """
        Execute one trading step
        :param actions: 0:short, 1:neutral, 2:long
        :return: state, reward, done, info
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        
        if self.terminal:
            # Episode ended
            print(f"Episode: {self.episode} end; total trades: {self.trades}")
            df = pd.DataFrame(self.date_memory, columns=['datetime'])
            df['daily_return'] = self.return_list
            
            if df["daily_return"].std() != 0:
                # Bitcoin trades 24/7, annualization factor is approx 365*24/4 = 2190 4h periods
                sharpe = ((2190**0.5) * df["daily_return"].mean() / df["daily_return"].std())
                print(f"end_return: {df['daily_return'].sum():0.2f}; Sharpe: {sharpe:0.3f}")
            
            # Print position statistics
            print(f"Position stats - Long: {self.position_stats['long']}, " +
                  f"Short: {self.position_stats['short']}, Neutral: {self.position_stats['neutral']}")
        else:
            # Execute trade
            current_position = self.state[0]  # Record current position
            self._buy_and_sell(actions)
            new_position = self.state[0]  # New position
            
            self.day += 1
            self.data = self.df.loc[self.day, :]
            close_record = self.state[1]
            
            # Get future reference state for reward calculation
            try:
                self.refer_state = self._update_state(future_date=self.reward_future_day)
            except:
                self.refer_state = self._update_state(future_date=1)
            
            # Calculate reward (Key: support short positions)
            future_price = self.refer_state[1]
            current_price = self.state[1]
            price_change_ratio = future_price / current_price
            
            # Calculate reward based on current position and action
            if current_position == 1:  # Previously long
                if new_position == 1:  # Continue long
                    self.reward = price_change_ratio - 1
                elif new_position == 0:  # Close position (sell)
                    self.reward = (1 - self.sell_cost_pct) * (2 - price_change_ratio) - 1
                else:  # From long to short (close long, open short)
                    self.reward = (1 - self.sell_cost_pct - self.buy_cost_pct) * (2 - price_change_ratio) - 1
                    
            elif current_position == -1:  # Previously short
                if new_position == -1:  # Continue short
                    self.reward = 1 - price_change_ratio
                elif new_position == 0:  # Close position (buy to cover)
                    self.reward = (1 - self.buy_cost_pct) * (2 - price_change_ratio) - 1
                else:  # From short to long (close short, open long)
                    self.reward = (1 - self.buy_cost_pct - self.sell_cost_pct) * price_change_ratio - 1
                    
            else:  # Previously neutral
                if new_position == 1:  # Open long
                    self.reward = (1 - self.buy_cost_pct) * price_change_ratio - 1
                elif new_position == -1:  # Open short
                    self.reward = (1 - self.sell_cost_pct) * (2 - price_change_ratio) - 1
                else:  # Continue neutral
                    self.reward = 0
            
            # Update state to next time point
            self.state = self._update_state(future_date=1)
            self.date_memory.append(self._get_date())
            
            # Record strategy returns
            current_pos = self.state[0]
            if current_pos == 1:  # Long position
                self.return_list.append(self.state[1] / close_record - 1)
                self.position_stats['long'] += 1
            elif current_pos == -1:  # Short position
                self.return_list.append(1 - self.state[1] / close_record)
                self.position_stats['short'] += 1
            else:  # Neutral position
                self.return_list.append(0)
                self.position_stats['neutral'] += 1
            
            # Record buy-and-hold returns (always long)
            self.buy_hold_list.append(self.state[1] / close_record - 1)
        
        return self.state[2:], self.reward, self.terminal, self.refer_state[2:]
    
    def reset(self):
        """Initialize environment"""
        self.state = self._initiate_state()
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.return_list = [0]
        self.buy_hold_list = [0]
        self.trades = 0
        self.reward = 0
        self.terminal = False
        self.date_memory = [self._get_date()]
        self.episode += 1
        self.position_stats = {'long': 0, 'short': 0, 'neutral': 0}
        return self.state[2:]
    
    def render(self):
        """Print current state"""
        position_name = {-1: 'Short', 0: 'Neutral', 1: 'Long'}
        print(f"Trading time: {self.df.iloc[self.day]['datetime']}, " +
              f"Position: {position_name.get(self.state[0], 'Unknown')}")
        return self.state
    
    def _initiate_state(self):
        """
        Initialize state variable
        state = position(-1/0/1) + current close + technical indicators
        """
        state = [0] + self.df.loc[0, :][['close'] + self.tech_indicator_list].tolist()
        return state
    
    def _update_state(self, future_date=1):
        """
        Update account state after future_date days
        :param future_date: int
        :return: state
        """
        state = [self.state[0]] + self.df.loc[self.day + future_date - 1, :][['close'] + self.tech_indicator_list].tolist()
        return state
    
    def _get_date(self):
        """Get current date"""
        return self.data.datetime
    
    def _seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
