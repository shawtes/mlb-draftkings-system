#!/usr/bin/env python3
"""
Simple example script to demonstrate the RL team selector

This script shows how to use the RL system with minimal setup.
Run this after ensuring your data file is in the correct location.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_demo():
    """Quick demonstration of the RL system"""
    
    # Configuration
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    MODEL_PATH = 'demo_rl_model.pth'
    
    try:
        # Import the RL system
        from rl_team_selector import MLBRLTeamSelector
        
        logger.info("=== MLB RL Team Selector Demo ===")
        
        # Initialize the selector
        logger.info("1. Initializing RL Team Selector...")
        selector = MLBRLTeamSelector(DATA_PATH, MODEL_PATH)
        
        # Load data
        logger.info("2. Loading data...")
        selector.load_data()
        
        # Setup environment
        logger.info("3. Setting up RL environment...")
        selector.setup_environment()
        
        logger.info(f"   - State space size: {selector.env.state_size}")
        logger.info(f"   - Action space size: {selector.env.action_space.n}")
        logger.info(f"   - Number of unique players: {len(selector.data_df['Name'].unique())}")
        logger.info(f"   - Date range: {selector.data_df['date'].min()} to {selector.data_df['date'].max()}")
        
        # Quick training (reduced episodes for demo)
        logger.info("4. Training RL agent (quick demo - 100 episodes)...")
        selector.train(episodes=100, save_freq=25)
        
        # Evaluate
        logger.info("5. Evaluating trained model...")
        evaluation_results = selector.evaluate(num_episodes=10)
        
        # Show evaluation summary
        avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
        avg_performance = np.mean([r['lineup_performance'] for r in evaluation_results])
        avg_lineup_size = np.mean([r['lineup_size'] for r in evaluation_results])
        
        logger.info(f"   - Average Reward: {avg_reward:.2f}")
        logger.info(f"   - Average Lineup Performance: {avg_performance:.2f} fantasy points")
        logger.info(f"   - Average Lineup Size: {avg_lineup_size:.1f} players")
        
        # Predict optimal lineup for the latest available date
        logger.info("6. Predicting optimal lineup for latest available date...")
        
        # Get the latest date from the data
        latest_date = selector.data_df['date'].max()
        logger.info(f"   Using latest available date: {latest_date.strftime('%Y-%m-%d')}")
        
        optimal_lineup = selector.predict_optimal_lineup(latest_date.strftime('%Y-%m-%d'))
        
        if optimal_lineup is not None and len(optimal_lineup) > 0:
            logger.info("   Optimal Lineup:")
            logger.info(f"   Total Salary: ${optimal_lineup['Salary'].sum():,.0f}")
            logger.info(f"   Total Predicted Points: {optimal_lineup['Predicted_Points'].sum():.1f}")
            logger.info("   Players:")
            for _, player in optimal_lineup.iterrows():
                logger.info(f"     {player['Name']} ({player['Position']}) - "
                           f"${player['Salary']:,.0f} - {player['Predicted_Points']:.1f} pts")
        else:
            logger.info("   Could not generate lineup for the latest date (insufficient data)")
            
            # Let's try to predict for a date with known data
            logger.info("   Trying prediction for a known date with data...")
            sample_date = selector.data_df['date'].value_counts().index[0]  # Date with most players
            logger.info(f"   Using sample date: {sample_date.strftime('%Y-%m-%d')}")
            
            optimal_lineup = selector.predict_optimal_lineup(sample_date.strftime('%Y-%m-%d'))
            
            if optimal_lineup is not None and len(optimal_lineup) > 0:
                logger.info("   Sample Lineup:")
                logger.info(f"   Total Salary: ${optimal_lineup['Salary'].sum():,.0f}")
                logger.info(f"   Total Predicted Points: {optimal_lineup['Predicted_Points'].sum():.1f}")
                logger.info("   Players:")
                for _, player in optimal_lineup.iterrows():
                    logger.info(f"     {player['Name']} ({player['Position']}) - "
                               f"${player['Salary']:,.0f} - {player['Predicted_Points']:.1f} pts")
            else:
                logger.info("   Still could not generate lineup - this may indicate data issues")
        
        logger.info("7. Demo completed successfully!")
        logger.info("\nNext steps:")
        logger.info("- Test synthetic salary generation: python enhanced_rl_demo.py --salary-test")
        logger.info("- Run enhanced demo with synthetic data: python enhanced_rl_demo.py --full-demo")
        logger.info("- Run full training with more episodes: python run_rl_team_selector.py --mode train --episodes 2000")
        logger.info("- Run walk-forward validation: python run_rl_team_selector.py --mode walkforward")
        logger.info("- Predict for specific date: python run_rl_team_selector.py --mode predict --date 2025-07-04")
        
        logger.info("\n💡 IMPORTANT: Since you only have a few days of salary data,")
        logger.info("   consider using the enhanced demo with synthetic salaries:")
        logger.info("   python enhanced_rl_demo.py --full-demo")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed:")
        logger.error("pip install pandas numpy torch gymnasium matplotlib scikit-learn joblib")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(f"Make sure your data file exists at: {DATA_PATH}")
        logger.error("Update the DATA_PATH variable if your file is in a different location")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error("Check the logs above for more details")

def test_data_format():
    """Test if the data file has the expected format"""

    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_dat1a.csv'

    try:
        logger.info("Testing data format...")
        
        # Load a small sample
        df = pd.read_csv(DATA_PATH, nrows=100)
        
        # Check required columns
        required_cols = ['Name', 'date', 'calculated_dk_fpts']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            logger.info("Available columns:")
            for col in df.columns:
                logger.info(f"  - {col}")
        else:
            logger.info("✓ All required columns present")
            
        # Check date format
        try:
            df['date'] = pd.to_datetime(df['date'])
            logger.info("✓ Date column can be parsed")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        except:
            logger.warning("✗ Date column cannot be parsed")
            
        # Check for calculated_dk_fpts or ability to calculate
        if 'calculated_dk_fpts' in df.columns:
            logger.info("✓ Fantasy points column exists")
            logger.info(f"  Points range: {df['calculated_dk_fpts'].min():.1f} to {df['calculated_dk_fpts'].max():.1f}")
        else:
            # Check if we can calculate fantasy points
            dk_cols = ['1B', '2B', '3B', 'HR', 'RBI', 'R', 'BB', 'HBP', 'SB']
            available_dk_cols = [col for col in dk_cols if col in df.columns]
            
            if len(available_dk_cols) >= 5:
                logger.info("✓ Can calculate fantasy points from available columns")
            else:
                logger.warning("✗ Cannot calculate fantasy points - missing stat columns")
                
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Unique players: {df['Name'].nunique()}")
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.error("Please update the DATA_PATH variable with the correct file location")
        
    except Exception as e:
        logger.error(f"Error reading data: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Team Selector Demo')
    parser.add_argument('--test-data', action='store_true', help='Test data format only')
    parser.add_argument('--demo', action='store_true', help='Run full demo')
    
    args = parser.parse_args()
    
    if args.test_data:
        test_data_format()
    elif args.demo:
        quick_demo()
    else:
        # Run both by default
        test_data_format()
        print("\n" + "="*50 + "\n")
        quick_demo()
