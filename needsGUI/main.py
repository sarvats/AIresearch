import argparse
from train_agent import train_agent
from evaluate_agent import evaluate_agent

def main():
    parser = argparse.ArgumentParser(description='Stock Trading RL Agent')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'], 
                        help='Mode: train or evaluate')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--train_start', type=str, default='2020-01-01', help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2022-12-31', help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--eval_start', type=str, default='2023-01-01', help='Evaluation start date (YYYY-MM-DD)')
    parser.add_argument('--eval_end', type=str, default='2023-12-31', help='Evaluation end date (YYYY-MM-DD)')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--model', type=str, help='Path to trained model (for evaluation)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, model_path = train_agent(
            ticker=args.ticker,
            start_date=args.train_start,
            end_date=args.train_end,
            total_timesteps=args.timesteps
        )
        # Optionally evaluate after training
        print(f"\nTraining complete. Evaluating trained model...")
        evaluate_agent(
            model_path=model_path,
            ticker=args.ticker,
            start_date=args.eval_start,
            end_date=args.eval_end
        )
    
    elif args.mode == 'evaluate':
        if not args.model:
            print("Error: Model path must be provided for evaluation mode")
            return
        
        evaluate_agent(
            model_path=args.model,
            ticker=args.ticker,
            start_date=args.eval_start,
            end_date=args.eval_end
        )

if __name__ == "__main__":
    main()