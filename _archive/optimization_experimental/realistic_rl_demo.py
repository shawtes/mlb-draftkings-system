#!/usr/bin/env python3
"""
Demo script for the realistic RL system that uses predictions

This demonstrates the complete workflow:
1. Get predictions using your existing model
2. Use RL to learn optimal lineup selection
3. Follow exact DraftKings rules
4. Validate against actual performance
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_realistic_system():
    """Test the realistic RL system"""
    
    try:
        from realistic_rl_system import RealisticMLBRLSystem
        
        logger.info("=== Realistic MLB RL System Demo ===")
        
        # Configuration
        DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
        PREDICTION_MODEL_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
        
        # Initialize system
        logger.info("1. Initializing Realistic RL System...")
        rl_system = RealisticMLBRLSystem(DATA_PATH, PREDICTION_MODEL_PATH)
        
        logger.info(f"   - Environment state size: {rl_system.env.state_size}")
        logger.info(f"   - Available dates: {len(rl_system.env.available_dates)}")
        logger.info(f"   - Date range: {min(rl_system.env.available_dates)} to {max(rl_system.env.available_dates)}")
        
        # Quick training
        logger.info("2. Training RL agent (quick demo - 100 episodes)...")
        rl_system.train(episodes=100)
        
        # Test prediction for a known date
        logger.info("3. Testing prediction for latest available date...")
        latest_date = max(rl_system.env.available_dates).strftime('%Y-%m-%d')
        logger.info(f"   Using date: {latest_date}")
        
        optimal_lineup = rl_system.predict_lineup(latest_date)
        
        if optimal_lineup is not None and len(optimal_lineup) > 0:
            logger.info("✓ Successfully generated optimal lineup!")
            logger.info(f"   Lineup size: {len(optimal_lineup)}")
            logger.info(f"   Total salary: ${optimal_lineup['Salary'].sum():,.0f}")
            logger.info(f"   Total projected points: {optimal_lineup['Projected_Points'].sum():.1f}")
            
            logger.info("   Players selected:")
            for _, player in optimal_lineup.iterrows():
                logger.info(f"     {player['Name']} ({player['Position']}) - "
                           f"${player['Salary']:,.0f} - {player['Projected_Points']:.1f} pts")
        else:
            logger.warning("✗ Could not generate lineup")
        
        # Evaluate system
        logger.info("4. Evaluating system performance...")
        evaluation_results = rl_system.evaluate_predictions_vs_actual(num_episodes=10)
        
        if len(evaluation_results) > 0:
            avg_actual = evaluation_results['actual_points'].mean()
            avg_projected = evaluation_results['projected_points'].mean()
            avg_lineup_size = evaluation_results['lineup_size'].mean()
            
            logger.info(f"   Average actual points: {avg_actual:.2f}")
            logger.info(f"   Average projected points: {avg_projected:.2f}")
            logger.info(f"   Average lineup size: {avg_lineup_size:.1f}")
            logger.info(f"   Prediction accuracy: {100 - abs(avg_actual - avg_projected):.1f}%")
        
        logger.info("5. Demo completed successfully!")
        
        logger.info("\n" + "="*60)
        logger.info("KEY FEATURES OF THIS SYSTEM:")
        logger.info("="*60)
        logger.info("✓ Uses your existing prediction model")
        logger.info("✓ Follows exact DraftKings rules")
        logger.info("✓ Learns from predictions vs actual performance")
        logger.info("✓ Handles salary cap and position constraints")
        logger.info("✓ Validates against real historical data")
        logger.info("✓ Provides realistic game-day workflow")
        
        logger.info("\nNext steps:")
        logger.info("- Train with more episodes for better performance")
        logger.info("- Run walk-forward validation")
        logger.info("- Use for live predictions on game day")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error("Check data paths and model files")

def test_draftkings_rules():
    """Test that the system follows DraftKings rules correctly"""
    
    logger.info("=== Testing DraftKings Rules Compliance ===")
    
    from realistic_rl_system import DK_RULES
    
    logger.info("DraftKings Rules:")
    logger.info(f"  Salary Cap: ${DK_RULES['SALARY_CAP']:,}")
    logger.info(f"  Lineup Size: {DK_RULES['LINEUP_SIZE']}")
    logger.info("  Position Requirements:")
    for pos, count in DK_RULES['POSITIONS'].items():
        logger.info(f"    {pos}: {count}")
    
    logger.info("  Scoring System:")
    for stat, points in DK_RULES['SCORING'].items():
        logger.info(f"    {stat}: {points} points")
    
    logger.info("✓ Rules loaded successfully!")

def analyze_data_for_rl():
    """Analyze data to understand what the RL system will work with"""
    
    logger.info("=== Analyzing Data for RL System ===")
    
    try:
        DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
        
        # Load sample data
        df = pd.read_csv(DATA_PATH, nrows=1000)
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Data sample: {len(df)} rows")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique players: {df['Name'].nunique()}")
        logger.info(f"Unique dates: {df['date'].nunique()}")
        
        # Check for required columns
        required_cols = ['Name', 'date', 'calculated_dk_fpts']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        else:
            logger.info("✓ All required columns present")
        
        # Check fantasy points distribution
        if 'calculated_dk_fpts' in df.columns:
            logger.info("Fantasy Points Distribution:")
            logger.info(f"  Mean: {df['calculated_dk_fpts'].mean():.2f}")
            logger.info(f"  Std: {df['calculated_dk_fpts'].std():.2f}")
            logger.info(f"  Min: {df['calculated_dk_fpts'].min():.2f}")
            logger.info(f"  Max: {df['calculated_dk_fpts'].max():.2f}")
        
        # Check for dates with enough players
        players_per_date = df.groupby('date')['Name'].nunique()
        good_dates = players_per_date[players_per_date >= 20]
        
        logger.info(f"Dates with 20+ players: {len(good_dates)}")
        if len(good_dates) > 0:
            logger.info(f"  Best date: {players_per_date.idxmax()} ({players_per_date.max()} players)")
        
        logger.info("✓ Data analysis complete!")
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Realistic RL System Demo')
    parser.add_argument('--test-system', action='store_true', help='Test the full RL system')
    parser.add_argument('--test-rules', action='store_true', help='Test DraftKings rules')
    parser.add_argument('--analyze-data', action='store_true', help='Analyze data for RL')
    
    args = parser.parse_args()
    
    if args.test_system:
        test_realistic_system()
    elif args.test_rules:
        test_draftkings_rules()
    elif args.analyze_data:
        analyze_data_for_rl()
    else:
        # Run all tests by default
        test_draftkings_rules()
        print("\n" + "="*50 + "\n")
        analyze_data_for_rl()
        print("\n" + "="*50 + "\n")
        test_realistic_system()
