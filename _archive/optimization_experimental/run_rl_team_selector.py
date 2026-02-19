#!/usr/bin/env python3
"""
Main runner script for MLB RL Team Selector

This script provides a command-line interface to run different modes of the
reinforcement learning team selection system.

Usage:
    python run_rl_team_selector.py --mode train
    python run_rl_team_selector.py --mode evaluate
    python run_rl_team_selector.py --mode predict --date 2025-07-04
    python run_rl_team_selector.py --mode walkforward
"""

import argparse
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our modules
from rl_team_selector import MLBRLTeamSelector
from walkforward_rl_validator import WalkForwardRLValidator
import rl_config as config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def setup_random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(config.RANDOM_SEED)
    import torch
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)

def train_model(episodes=None, save_path=None):
    """Train the RL model"""
    logger.info("Starting model training...")
    
    episodes = episodes or config.TRAINING_EPISODES
    save_path = save_path or config.MODEL_SAVE_PATH
    
    # Initialize selector
    selector = MLBRLTeamSelector(config.DATA_PATH, save_path)
    
    # Load data and setup environment
    selector.load_data()
    selector.setup_environment()
    
    # Train the model
    selector.train(episodes=episodes, save_freq=config.SAVE_FREQUENCY)
    
    # Plot training history
    selector.plot_training_history()
    
    logger.info(f"Training completed. Model saved to {save_path}")
    return selector

def evaluate_model(model_path=None, num_episodes=None):
    """Evaluate the trained model"""
    logger.info("Starting model evaluation...")
    
    model_path = model_path or config.MODEL_SAVE_PATH
    num_episodes = num_episodes or config.EVALUATION_EPISODES
    
    # Initialize selector
    selector = MLBRLTeamSelector(config.DATA_PATH, model_path)
    
    # Load data and setup environment
    selector.load_data()
    selector.setup_environment()
    
    # Evaluate the model
    evaluation_results = selector.evaluate(num_episodes=num_episodes)
    
    # Save evaluation results
    eval_df = pd.DataFrame(evaluation_results)
    eval_path = config.RESULTS_PATH.replace('.csv', '_evaluation.csv')
    eval_df.to_csv(eval_path, index=False)
    
    logger.info(f"Evaluation completed. Results saved to {eval_path}")
    return evaluation_results

def predict_lineup(date=None, model_path=None):
    """Predict optimal lineup for a specific date"""
    logger.info("Predicting optimal lineup...")
    
    date = date or datetime.now().strftime('%Y-%m-%d')
    model_path = model_path or config.MODEL_SAVE_PATH
    
    # Initialize selector
    selector = MLBRLTeamSelector(config.DATA_PATH, model_path)
    
    # Load data and setup environment
    selector.load_data()
    selector.setup_environment()
    
    # Predict optimal lineup
    optimal_lineup = selector.predict_optimal_lineup(date)
    
    if optimal_lineup is not None:
        # Save lineup
        lineup_path = config.RESULTS_PATH.replace('.csv', f'_lineup_{date}.csv')
        optimal_lineup.to_csv(lineup_path, index=False)
        
        logger.info(f"Optimal lineup for {date}:")
        logger.info(optimal_lineup.to_string(index=False))
        logger.info(f"Lineup saved to {lineup_path}")
    else:
        logger.warning(f"Could not generate lineup for {date}")
    
    return optimal_lineup

def run_walkforward_validation(start_date=None, end_date=None, max_validations=None):
    """Run walk-forward validation"""
    logger.info("Starting walk-forward validation...")
    
    start_date = start_date or config.VALIDATION_START_DATE
    end_date = end_date or config.VALIDATION_END_DATE
    max_validations = max_validations or config.MAX_VALIDATIONS
    
    # Initialize validator
    validator = WalkForwardRLValidator(
        data_path=config.DATA_PATH,
        initial_train_days=config.INITIAL_TRAIN_DAYS,
        validation_window=config.VALIDATION_WINDOW,
        retrain_frequency=config.RETRAIN_FREQUENCY
    )
    
    # Run validation
    results = validator.run_walk_forward_validation(
        start_date=start_date,
        end_date=end_date,
        max_validations=max_validations
    )
    
    # Analyze results
    analysis = validator.analyze_results()
    
    # Plot results
    plot_path = config.PLOTS_PATH.replace('.png', '_walkforward.png')
    validator.plot_results(save_path=plot_path)
    
    # Save results
    results_path = config.RESULTS_PATH.replace('.csv', '_walkforward.csv')
    validator.save_results(results_path)
    
    logger.info("Walk-forward validation completed!")
    return results, analysis

def compare_with_existing_model():
    """Compare RL model with existing prediction model"""
    logger.info("Comparing RL model with existing prediction model...")
    
    # This would integrate with your existing predction01.py model
    # For now, we'll just note that this comparison should be implemented
    logger.info("Comparison with existing model not yet implemented")
    logger.info("To implement: compare RL predictions with predction01.py predictions")
    
    return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MLB RL Team Selector')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict', 'walkforward', 'compare'],
                       required=True, help='Operation mode')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--start-date', type=str, help='Start date for validation')
    parser.add_argument('--end-date', type=str, help='End date for validation')
    parser.add_argument('--max-validations', type=int, help='Maximum number of validations')
    
    args = parser.parse_args()
    
    # Setup random seed
    setup_random_seed()
    
    try:
        if args.mode == 'train':
            train_model(episodes=args.episodes, save_path=args.model_path)
            
        elif args.mode == 'evaluate':
            evaluate_model(model_path=args.model_path)
            
        elif args.mode == 'predict':
            predict_lineup(date=args.date, model_path=args.model_path)
            
        elif args.mode == 'walkforward':
            run_walkforward_validation(
                start_date=args.start_date,
                end_date=args.end_date,
                max_validations=args.max_validations
            )
            
        elif args.mode == 'compare':
            compare_with_existing_model()
            
    except Exception as e:
        logger.error(f"Error running {args.mode}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
