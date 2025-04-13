import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_fetcher import fetch_data
from stock_env import StockTradingEnv
import argparse

def evaluate_agent(model_path, ticker='AAPL', start_date='2023-01-01', end_date='2023-12-31'):
    """
    Evaluate a trained agent on test data
    """
    # Load test data
    print(f"Fetching test data for {ticker} from {start_date} to {end_date}...")
    test_data = fetch_data(ticker, start=start_date, end=end_date)
    
    # Create environment
    env = StockTradingEnv(test_data)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Run evaluation with debugging
    print("Evaluating agent...")
    obs, _ = env.reset()
    done = False
    
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Debug info
        step_count += 1
        total_reward += reward
        if step_count % 20 == 0:  # Print every 20 steps
            print(f"Step {step_count}: Action={action[0]:.3f}, Reward={reward:.5f}, "
                  f"Total Reward={total_reward:.5f}")
        
        env.render()
        
        if done or truncated:
            break
    
    # ... [rest of the function] ...
    
    # Baseline comparison: Buy and Hold strategy
    initial_price = float(test_data.iloc[0]['Close'])
    final_price = float(test_data.iloc[-1]['Close'])
    initial_capital = env.initial_balance
    shares_bought = initial_capital / initial_price
    buy_hold_value = shares_bought * final_price
    
    # Calculate metrics
    roi_agent = (env.net_worth - initial_capital) / initial_capital * 100
    roi_buy_hold = (buy_hold_value - initial_capital) / initial_capital * 100
    
    print("\n----- PERFORMANCE SUMMARY -----")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Agent Final Net Worth: ${env.net_worth:.2f}")
    print(f"Agent Return on Investment: {roi_agent:.2f}%")
    print(f"Buy & Hold Strategy Value: ${buy_hold_value:.2f}")
    print(f"Buy & Hold Return on Investment: {roi_buy_hold:.2f}%")
    print(f"Outperformance vs Buy & Hold: {roi_agent - roi_buy_hold:.2f}%")
    
    # Plot performance
    env.plot_performance()
    
    return env.history

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained trading agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    evaluate_agent(args.model, args.ticker, args.start, args.end)

if __name__ == "__main__":
    main()