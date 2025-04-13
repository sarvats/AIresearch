import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from data_fetcher import fetch_data
from stock_env import StockTradingEnv
import pandas as pd
import numpy as np

def train_agent(ticker='AAPL', start_date='2020-01-01', end_date='2022-12-31', 
                total_timesteps=50000, save_path="models"):
    """
    Train a trading agent
    """
    print(f"Fetching data for {ticker}...")
    data = fetch_data(ticker, start=start_date, end=end_date)
    
    # Split data: 80% train, 20% validation
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size].copy()
    val_data = data.iloc[train_size:].copy()
    
    print(f"Training data size: {len(train_data)}, Validation data size: {len(val_data)}")
    
    # Create environments
    env = StockTradingEnv(train_data)
    # Create a separate monitor-wrapped environment for evaluation
    eval_env = Monitor(StockTradingEnv(val_data), filename=os.path.join(save_path, f"{ticker}_eval_log"))

    # Add debug logging to verify rewards
    print("Testing environments:")
    train_obs, _ = env.reset()
    eval_obs, _ = eval_env.reset()
    print(f"Training observation shape: {train_obs.shape}")
    print(f"Eval observation shape: {eval_obs.shape}")

    # Test a step in each environment
    train_action = [0.1]  # Small buy action
    eval_action = [0.1]   # Same action
    _, train_reward, _, _, _ = env.step(train_action)
    _, eval_reward, _, _, _ = eval_env.step(eval_action)
    print(f"Test step - Training reward: {train_reward}, Eval reward: {eval_reward}")

    # Reset environments
    env.reset()
    eval_env.reset()
    
    # Create directory for saving models
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f"ppo_{ticker}")
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_path}_best",
        log_path=f"{model_path}_logs",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # ... [rest of the function]
    
    # Create and train the model
    print("Training agent...")
    model = PPO(
       "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4,  # Reduced learning rate for more stability
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=f"{save_path}/tensorboard_{ticker}"  # Add tensorboard logging
)
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # Save the final model
    final_model_path = f"{model_path}_final"
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return model, final_model_path

if __name__ == "__main__":
    # Example usage
    train_agent(ticker='AAPL', total_timesteps=50000)