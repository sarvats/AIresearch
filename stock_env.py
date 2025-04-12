import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index()
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values
        return np.append(obs, [self.shares_held]).astype(np.float32)

    def step(self, action):
        price = self.df.iloc[self.current_step]['Close']
        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= price
        elif action == 2 and self.shares_held > 0:  # Sell
            self.shares_held -= 1
            self.balance += price

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        total_value = self.balance + self.shares_held * price
        reward = total_value - 10000  # Net gain/loss

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares: {self.shares_held}")
