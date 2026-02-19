#!/usr/bin/env python3
"""
Data Analysis Script for MLB RL System

This script analyzes your MLB data to understand its structure and help optimize
the RL system performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_data_structure():
    """Analyze the structure of the MLB data"""
    
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    
    try:
        logger.info("Loading data for analysis...")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        
        # Basic info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {len(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Date analysis
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Date span: {(df['date'].max() - df['date'].min()).days} days")
        
        # Check for calculated_dk_fpts
        if 'calculated_dk_fpts' in df.columns:
            logger.info("✓ Fantasy points column exists")
            
            # Analyze fantasy points distribution
            fpts_stats = df['calculated_dk_fpts'].describe()
            logger.info("Fantasy Points Statistics:")
            logger.info(f"  Mean: {fpts_stats['mean']:.2f}")
            logger.info(f"  Std: {fpts_stats['std']:.2f}")
            logger.info(f"  Min: {fpts_stats['min']:.2f}")
            logger.info(f"  Max: {fpts_stats['max']:.2f}")
            logger.info(f"  Zero points: {(df['calculated_dk_fpts'] == 0).sum()}")
            
            # Check for negative points
            negative_points = df[df['calculated_dk_fpts'] < 0]
            if len(negative_points) > 0:
                logger.warning(f"Found {len(negative_points)} rows with negative fantasy points")
                logger.warning("This may cause issues with the RL reward system")
        
        # Player analysis
        player_counts = df['Name'].value_counts()
        logger.info(f"Unique players: {len(player_counts)}")
        logger.info(f"Most games by player: {player_counts.max()}")
        logger.info(f"Least games by player: {player_counts.min()}")
        logger.info(f"Average games per player: {player_counts.mean():.1f}")
        
        # Date analysis
        date_counts = df['date'].value_counts().sort_index()
        logger.info(f"Unique dates: {len(date_counts)}")
        logger.info(f"Most players on single date: {date_counts.max()}")
        logger.info(f"Least players on single date: {date_counts.min()}")
        logger.info(f"Average players per date: {date_counts.mean():.1f}")
        
        # Check for required columns for salary and position
        required_cols = ['salary', 'position', 'team', 'Team']
        missing_cols = []
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"Missing columns for RL system: {missing_cols}")
            logger.info("The RL system will create synthetic data for missing columns")
        
        # Recent data analysis
        logger.info("\nRecent Data Analysis:")
        recent_data = df[df['date'] >= '2024-01-01']
        if len(recent_data) > 0:
            logger.info(f"2024+ data: {len(recent_data)} rows")
            logger.info(f"2024+ date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
            logger.info(f"2024+ unique players: {recent_data['Name'].nunique()}")
            
            # Check latest dates with good data
            latest_dates = recent_data['date'].value_counts().head(10)
            logger.info("Top 10 dates with most players (2024+):")
            for date, count in latest_dates.items():
                logger.info(f"  {date.strftime('%Y-%m-%d')}: {count} players")
        else:
            logger.warning("No recent data (2024+) found")
            
        # Check for key statistical columns
        stat_cols = ['HR', 'RBI', 'R', 'SB', 'BB', 'H', '1B', '2B', '3B', 'SO', 'AVG', 'OBP', 'SLG']
        available_stats = [col for col in stat_cols if col in df.columns]
        missing_stats = [col for col in stat_cols if col not in df.columns]
        
        logger.info(f"\nAvailable stat columns: {len(available_stats)}")
        logger.info(f"Missing stat columns: {len(missing_stats)}")
        
        if missing_stats:
            logger.warning(f"Missing stats: {missing_stats}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return None

def suggest_improvements(df):
    """Suggest improvements based on data analysis"""
    
    logger.info("\n" + "="*50)
    logger.info("SUGGESTIONS FOR OPTIMAL RL PERFORMANCE")
    logger.info("="*50)
    
    if df is None:
        logger.error("Cannot provide suggestions - data analysis failed")
        return
    
    # Check fantasy points distribution
    if 'calculated_dk_fpts' in df.columns:
        fpts_mean = df['calculated_dk_fpts'].mean()
        fpts_std = df['calculated_dk_fpts'].std()
        
        if fpts_mean < 5:
            logger.warning("LOW FANTASY POINTS DETECTED:")
            logger.warning(f"  Average fantasy points: {fpts_mean:.2f}")
            logger.warning("  Consider adjusting reward scaling in RL system")
            logger.warning("  Increase REWARD_SCALING['PLAYER_POINTS'] in rl_config.py")
        
        if fpts_std > 20:
            logger.warning("HIGH FANTASY POINTS VOLATILITY:")
            logger.warning(f"  Standard deviation: {fpts_std:.2f}")
            logger.warning("  Consider using reward clipping or normalization")
    
    # Check data recency
    latest_date = df['date'].max()
    days_old = (datetime.now() - latest_date).days
    
    if days_old > 7:
        logger.warning(f"DATA FRESHNESS ISSUE:")
        logger.warning(f"  Latest data is {days_old} days old ({latest_date.strftime('%Y-%m-%d')})")
        logger.warning("  Update data for better predictions")
    
    # Check for training data sufficiency
    recent_data = df[df['date'] >= '2023-01-01']
    if len(recent_data) < 1000:
        logger.warning("INSUFFICIENT RECENT DATA:")
        logger.warning(f"  Only {len(recent_data)} rows since 2023")
        logger.warning("  Consider reducing INITIAL_TRAIN_DAYS in config")
    
    # Check player coverage
    avg_games_per_player = df['Name'].value_counts().mean()
    if avg_games_per_player < 10:
        logger.warning("LOW PLAYER COVERAGE:")
        logger.warning(f"  Average {avg_games_per_player:.1f} games per player")
        logger.warning("  This may reduce RL learning effectiveness")
    
    # Suggest optimal training parameters
    logger.info("\nRECOMMENDED TRAINING PARAMETERS:")
    
    total_rows = len(df)
    if total_rows < 5000:
        logger.info("  - Use 500-1000 training episodes (small dataset)")
        logger.info("  - Set INITIAL_TRAIN_DAYS = 180")
    elif total_rows < 20000:
        logger.info("  - Use 1000-2000 training episodes (medium dataset)")
        logger.info("  - Set INITIAL_TRAIN_DAYS = 365")
    else:
        logger.info("  - Use 2000+ training episodes (large dataset)")
        logger.info("  - Set INITIAL_TRAIN_DAYS = 500")
    
    # Suggest validation parameters
    unique_dates = df['date'].nunique()
    if unique_dates < 50:
        logger.info("  - Use max_validations = 10-20 for walk-forward")
    elif unique_dates < 200:
        logger.info("  - Use max_validations = 20-50 for walk-forward")
    else:
        logger.info("  - Use max_validations = 50+ for walk-forward")
    
    logger.info("\nOPTIMIZATION TIPS:")
    logger.info("  - Start with demo training (100 episodes)")
    logger.info("  - Monitor reward trends during training")
    logger.info("  - Use walk-forward validation for realistic performance")
    logger.info("  - Compare against baseline strategies")

def create_training_schedule(df):
    """Create a recommended training schedule"""
    
    if df is None:
        return
    
    logger.info("\n" + "="*50)
    logger.info("RECOMMENDED TRAINING SCHEDULE")
    logger.info("="*50)
    
    total_rows = len(df)
    unique_dates = df['date'].nunique()
    
    logger.info("Phase 1: Quick Demo (5 minutes)")
    logger.info("  python rl_demo.py --demo")
    logger.info("  Purpose: Verify system works with your data")
    
    logger.info("\nPhase 2: Short Training (30 minutes)")
    logger.info("  python run_rl_team_selector.py --mode train --episodes 500")
    logger.info("  Purpose: Initial learning and parameter validation")
    
    logger.info("\nPhase 3: Full Training (2-4 hours)")
    if total_rows < 10000:
        logger.info("  python run_rl_team_selector.py --mode train --episodes 1500")
    else:
        logger.info("  python run_rl_team_selector.py --mode train --episodes 3000")
    logger.info("  Purpose: Complete model training")
    
    logger.info("\nPhase 4: Validation (1-2 hours)")
    if unique_dates < 100:
        logger.info("  python run_rl_team_selector.py --mode walkforward --max-validations 20")
    else:
        logger.info("  python run_rl_team_selector.py --mode walkforward --max-validations 50")
    logger.info("  Purpose: Realistic performance evaluation")
    
    logger.info("\nPhase 5: Production Use")
    logger.info("  python run_rl_team_selector.py --mode predict --date 2025-07-04")
    logger.info("  Purpose: Generate optimal lineups for DraftKings")
    
    logger.info("\nExpected Timeline: 4-8 hours total")

def main():
    """Main analysis function"""
    
    logger.info("=== MLB Data Analysis for RL System ===\n")
    
    # Analyze data structure
    df = analyze_data_structure()
    
    # Provide suggestions
    suggest_improvements(df)
    
    # Create training schedule
    create_training_schedule(df)
    
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*50)
    logger.info("Next step: Run the improved demo:")
    logger.info("python rl_demo.py --demo")

if __name__ == "__main__":
    main()
