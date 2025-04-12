from stable_baselines3 import PPO
from stock_env import StockTradingEnv
from data_fetcher import fetch_data

data = fetch_data('AAPL')
env = StockTradingEnv(data)

model = PPO.load("ppo_stock_model")

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"Total Reward: {total_reward}")
