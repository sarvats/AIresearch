import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for OpenAI gym
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, max_shares=100, transaction_fee_percent=0.001):
        super(StockTradingEnv, self).__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.max_shares = max_shares
        self.transaction_fee_percent = transaction_fee_percent
        
        # Action space: percentage of max shares to buy/sell (-1.0 to 1.0)
        # -1.0 means sell all shares, 1.0 means buy max shares
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Dynamic feature count based on dataframe
        price_features = len(self.df.columns) - 1  # Excluding the Date column
        account_features = 2  # Balance and shares held
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(price_features + account_features,), 
            dtype=np.float32
        )
        
        # Trading history for visualization
        self.history = []
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.history = []
        
        return self._get_observation(), {}  # Return observation and empty info dict
    
    def _get_observation(self):
    # Get all features except Date
    # Fix the multi-index warning by explicitly converting to a pandas Series first
        row = self.df.iloc[self.current_step]
        features = row.drop('Date' if 'Date' in row.index else None).values
    
    # Add account information
        current_price = float(self.df.iloc[self.current_step]['Close'])
        account_info = np.array([
           self.balance / self.initial_balance,  # Normalize balance
           self.shares_held / self.max_shares    # Normalize shares
        ])
    
        return np.concatenate([features, account_info]).astype(np.float32)
        
    def step(self, action):
        # Get current price - ensure it's a float, not a Series
        current_price = float(self.df.iloc[self.current_step]['Close'])
        previous_net_worth = self.balance + self.shares_held * current_price
        
        # Determine the number of shares to buy or sell
        action_value = action[0]  # Extract scalar from action array
        
        if action_value > 0:  # Buy
            # Calculate max shares we can buy
            max_affordable = int(self.balance // current_price)
            shares_to_buy = int(action_value * self.max_shares)
            shares_to_buy = min(shares_to_buy, max_affordable)
            
            if shares_to_buy > 0:
                # Transaction cost
                transaction_cost = shares_to_buy * current_price * self.transaction_fee_percent
                self.balance -= (shares_to_buy * current_price + transaction_cost)
                self.shares_held += shares_to_buy
                
        elif action_value < 0:  # Sell
            shares_to_sell = int(-action_value * self.shares_held)
            
            if shares_to_sell > 0:
                # Transaction cost
                transaction_cost = shares_to_sell * current_price * self.transaction_fee_percent
                self.balance += (shares_to_sell * current_price - transaction_cost)
                self.shares_held -= shares_to_sell
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward - Enhance with multiple components
        current_net_worth = self.balance + self.shares_held * current_price
        self.net_worth = current_net_worth
    
        # Primary reward: change in portfolio value
        portfolio_change = (current_net_worth - previous_net_worth) / self.initial_balance
    
        # Add reward for holding profitable positions
        profit_holding_reward = 0
        price_change = 0
        if self.shares_held > 0 and action_value >= 0:  # We're holding or buying more
          price_change = current_price / float(self.df.iloc[self.current_step-1]['Close']) - 1
        if price_change > 0:
            profit_holding_reward = price_change * 0.1 * (self.shares_held / self.max_shares)
    
        # Penalize excessive trading (transaction costs)
        trading_penalty = 0
        if action_value != 0:  # If we traded
            trading_penalty = -0.001
    
        # Combine rewards
        reward = portfolio_change + profit_holding_reward + trading_penalty
        
        # Record history for visualization
        self.history.append({
            'step': self.current_step,
            'date': self.df.iloc[self.current_step-1]['Date'],
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': current_net_worth,
            'action': action_value
        })
        
        # Check if we're done
        done = self.current_step >= len(self.df) - 1
        
        # For gymnasium compatibility
        truncated = False
        info = {
            'net_worth': current_net_worth,
            'trade_count': len([h for h in self.history if h['action'] != 0])
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Print information about the current state
        """
        if self.current_step > 0:
            entry = self.history[-1]
            print(f"Date: {entry['date']}, Price: ${entry['price']:.2f}, Balance: ${entry['balance']:.2f}, "
                  f"Shares: {entry['shares_held']}, Net Worth: ${entry['net_worth']:.2f}")
    
    def plot_performance(self):
        """
        Plot performance after episode ends
        """
        if not self.history:
            print("No trading history to plot")
            return
            
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history)
        
        # Set style
        sns.set_style('darkgrid')
        plt.figure(figsize=(15, 10))
        
        # Plot net worth
        plt.subplot(3, 1, 1)
        plt.plot(history_df['date'], history_df['net_worth'], label='Net Worth', color='green')
        plt.title('Portfolio Performance')
        plt.ylabel('Value ($)')
        plt.legend()
        
        # Plot price and buy/sell actions
        plt.subplot(3, 1, 2)
        plt.plot(history_df['date'], history_df['price'], label='Stock Price', color='blue')
        
        # Plot buy signals
        buys = history_df[history_df['action'] > 0]
        sells = history_df[history_df['action'] < 0]
        
        if not buys.empty:
            plt.scatter(buys['date'], buys['price'], marker='^', color='green', label='Buy')
        if not sells.empty:
            plt.scatter(sells['date'], sells['price'], marker='v', color='red', label='Sell')
            
        plt.title('Stock Price and Trading Actions')
        plt.ylabel('Price ($)')
        plt.legend()
        
        # Plot shares held
        plt.subplot(3, 1, 3)
        plt.plot(history_df['date'], history_df['shares_held'], label='Shares Held', color='orange')
        plt.title('Position Size')
        plt.ylabel('Shares')
        plt.legend()
        
        plt.tight_layout()
        plt.show()