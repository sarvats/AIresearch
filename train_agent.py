import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



from stable_baselines3 import PPO
from stock_env import StockTradingEnv
from data_fetcher import fetch_data

data = fetch_data('AAPL')
env = StockTradingEnv(data)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ppo_stock_model")
