#!/usr/bin/env python3
"""
Enhanced RL Demo with Realistic Salary Data

This script demonstrates the complete workflow using the salary-enriched dataset:
1. Load salary-enriched data
2. Test lineup optimization
3. Validate DraftKings rules
4. Show performance analysis
"""

import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhanced_rl_demo():
    """Enhanced demo with realistic salary data"""
    
    try:
        logger.info("=== Enhanced MLB RL System with Realistic Salary Data ===")
        
        # Configuration
        DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
        PREDICTION_MODEL_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
        ENHANCED_DATA_PATH = '5_ENTRIES/data_with_dk_entries_salaries.csv'
        
        # Step 1: Load the salary-enriched data
        logger.info("1. Loading salary-enriched data...")
        
        if os.path.exists(ENHANCED_DATA_PATH):
            logger.info("   Using DKEntries-based salary data with 30-day rolling PPG averages...")
            df = pd.read_csv(ENHANCED_DATA_PATH, low_memory=False)
        else:
            logger.error(f"   Salary-enriched data not found at: {ENHANCED_DATA_PATH}")
            logger.error("   Please run dk_entries_salary_generator.py first to create the DKEntries-based dataset")
            return
        
        # Validate the data has required columns
        required_columns = ['Name', 'salary', 'calculated_dk_fpts', 'position', 'date', 'rolling_30_ppg']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"   Missing required columns: {missing_columns}")
            return
        
        logger.info(f"   Data loaded: {len(df)} rows, {len(df['Name'].unique())} unique players")
        logger.info(f"   Salary range: ${df['salary'].min():.0f} - ${df['salary'].max():.0f}")
        logger.info(f"   Rolling PPG range: {df['rolling_30_ppg'].min():.1f} - {df['rolling_30_ppg'].max():.1f}")
        logger.info(f"   Average rolling PPG: {df['rolling_30_ppg'].mean():.1f}")
        
        # Show position distribution
        position_counts = df['position'].value_counts()
        logger.info(f"   Position distribution: {position_counts.to_dict()}")
        
        # Show date range
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Test basic lineup optimization
        logger.info("2. Testing lineup optimization...")
        
        # Get a sample date with sufficient players
        daily_counts = df.groupby('date').size()
        viable_dates = daily_counts[daily_counts >= 50].index
        
        if len(viable_dates) == 0:
            logger.error("   No dates with sufficient players for lineup optimization")
            return
        
        test_date = viable_dates[-1]  # Use latest viable date
        test_data = df[df['date'] == test_date].copy()
        
        logger.info(f"   Testing with date: {test_date.strftime('%Y-%m-%d')} ({len(test_data)} players)")
        
        # Simple greedy optimization
        lineup = optimize_lineup_greedy(test_data)
        
        if lineup is not None:
            logger.info("   Optimal lineup found:")
            logger.info(f"   Total salary: ${lineup['salary'].sum():,}")
            logger.info(f"   Total points: {lineup['calculated_dk_fpts'].sum():.1f}")
            logger.info("   Players:")
            for _, player in lineup.iterrows():
                logger.info(f"     {player['Name']} ({player['position']}) - ${player['salary']:,} - {player['calculated_dk_fpts']:.1f} pts")
        else:
            logger.error("   Could not optimize lineup for test date")
            return
        
        # Show salary efficiency
        efficiency = lineup['calculated_dk_fpts'].sum() / (lineup['salary'].sum() / 1000)
        logger.info(f"   Salary efficiency: {efficiency:.2f} points per $1000")
        
        # Step 3: Validate DraftKings rules
        logger.info("3. Validating DraftKings rules...")
        validate_lineup_rules(lineup)
        
        # Step 4: Run multiple simulations
        logger.info("4. Running multiple lineup simulations...")
        run_multiple_simulations(df, viable_dates, num_simulations=10)
        
        # Step 5: Analyze salary vs performance
        logger.info("5. Analyzing salary vs performance correlation...")
        analyze_salary_performance(df)
        
        logger.info("6. Demo complete!")
        logger.info("   The enhanced system now uses DKEntries-based salary generation with 30-day rolling PPG averages")
        logger.info("   Next steps:")
        logger.info("   - Run full RL training using the realistic_rl_system.py")
        logger.info("   - Compare RL performance vs greedy/random baselines")
        logger.info("   - Implement walk-forward validation for temporal robustness")
        logger.info("   - Use DKEntries.csv for real-time salary updates")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

def optimize_lineup_greedy(data):
    """
    Greedy lineup optimization based on points per dollar
    """
    SALARY_CAP = 50000
    LINEUP_SIZE = 8
    POSITION_REQUIREMENTS = {
        'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3, 'UTIL': 1
    }
    
    # Calculate value (points per dollar)
    data = data.copy()
    data['value'] = data['calculated_dk_fpts'] / data['salary']
    
    # Sort by value
    sorted_data = data.sort_values('value', ascending=False)
    
    lineup = []
    remaining_salary = SALARY_CAP
    position_filled = {pos: 0 for pos in POSITION_REQUIREMENTS.keys()}
    
    for _, player in sorted_data.iterrows():
        if len(lineup) >= LINEUP_SIZE:
            break
            
        if player['salary'] <= remaining_salary:
            player_pos = player['position']
            
            # Check if we can add this player
            can_add = False
            
            # Check primary position requirement
            if player_pos in position_filled and position_filled[player_pos] < POSITION_REQUIREMENTS.get(player_pos, 0):
                can_add = True
                position_filled[player_pos] += 1
            # Check utility spot
            elif position_filled['UTIL'] < POSITION_REQUIREMENTS['UTIL']:
                can_add = True
                position_filled['UTIL'] += 1
            
            if can_add:
                lineup.append(player)
                remaining_salary -= player['salary']
    
    if len(lineup) == LINEUP_SIZE:
        return pd.DataFrame(lineup)
    else:
        return None

def validate_lineup_rules(lineup):
    """Validate that lineup follows DraftKings rules"""
    SALARY_CAP = 50000
    LINEUP_SIZE = 8
    POSITION_REQUIREMENTS = {
        'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3, 'UTIL': 1
    }
    
    total_salary = lineup['salary'].sum()
    lineup_size = len(lineup)
    
    logger.info(f"   Salary cap check: ${total_salary:,} / ${SALARY_CAP:,} {'✓' if total_salary <= SALARY_CAP else '✗'}")
    logger.info(f"   Lineup size check: {lineup_size} / {LINEUP_SIZE} {'✓' if lineup_size == LINEUP_SIZE else '✗'}")
    
    # Position requirements check
    position_counts = lineup['position'].value_counts()
    all_positions_valid = True
    
    for pos, required in POSITION_REQUIREMENTS.items():
        if pos == 'UTIL':
            continue
        actual = position_counts.get(pos, 0)
        logger.info(f"   {pos} position: {actual} / {required} {'✓' if actual == required else '✗'}")
        if actual != required:
            all_positions_valid = False
    
    # Check OF requirement
    of_count = position_counts.get('OF', 0)
    logger.info(f"   OF position: {of_count} / 3 {'✓' if of_count == 3 else '✗'}")
    if of_count != 3:
        all_positions_valid = False
    
    logger.info(f"   Overall validation: {'✓ VALID' if all_positions_valid and total_salary <= SALARY_CAP and lineup_size == LINEUP_SIZE else '✗ INVALID'}")

def run_multiple_simulations(df, viable_dates, num_simulations=10):
    """Run multiple lineup optimization simulations"""
    results = []
    
    for i in range(num_simulations):
        if i >= len(viable_dates):
            break
            
        test_date = viable_dates[-(i+1)]
        test_data = df[df['date'] == test_date].copy()
        
        if len(test_data) < 50:
            continue
            
        lineup = optimize_lineup_greedy(test_data)
        
        if lineup is not None:
            total_salary = lineup['salary'].sum()
            total_points = lineup['calculated_dk_fpts'].sum()
            efficiency = total_points / (total_salary / 1000)
            
            results.append({
                'date': test_date.strftime('%Y-%m-%d'),
                'total_salary': total_salary,
                'total_points': total_points,
                'efficiency': efficiency,
                'lineup_size': len(lineup)
            })
    
    if results:
        results_df = pd.DataFrame(results)
        
        logger.info(f"   Completed {len(results)} simulations")
        logger.info(f"   Average points: {results_df['total_points'].mean():.1f} ± {results_df['total_points'].std():.1f}")
        logger.info(f"   Average salary: ${results_df['total_salary'].mean():,.0f} ± ${results_df['total_salary'].std():,.0f}")
        logger.info(f"   Average efficiency: {results_df['efficiency'].mean():.2f} pts/$1000")
        logger.info(f"   Points range: {results_df['total_points'].min():.1f} - {results_df['total_points'].max():.1f}")
        
        # Save results
        results_df.to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/5_ENTRIES/lineup_simulation_results.csv', index=False)
        logger.info("   Results saved to lineup_simulation_results.csv")

def analyze_salary_performance(df):
    """Analyze the relationship between salary and performance"""
    correlation = df['salary'].corr(df['calculated_dk_fpts'])
    logger.info(f"   Salary-Performance correlation: {correlation:.3f}")
    
    # Show by position
    position_stats = df.groupby('position').agg({
        'salary': ['mean', 'std'],
        'calculated_dk_fpts': ['mean', 'std']
    }).round(0)
    
    logger.info("   Position statistics:")
    for position in position_stats.index:
        avg_salary = position_stats.loc[position, ('salary', 'mean')]
        avg_points = position_stats.loc[position, ('calculated_dk_fpts', 'mean')]
        efficiency = avg_points / (avg_salary / 1000)
        logger.info(f"     {position}: ${avg_salary:,.0f} avg salary, {avg_points:.1f} avg points, {efficiency:.2f} efficiency")

def quick_salary_test():
    """Quick test of DKEntries-based salary generation"""
    
    try:
        logger.info("=== Quick DKEntries Salary Test ===")
        
        # Check if we have the DKEntries salary data
        dk_entries_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
        
        if not os.path.exists(dk_entries_file):
            logger.error("DKEntries salary data not found!")
            logger.info("Please run: python dk_entries_salary_generator.py")
            return
        
        # Load and validate the data
        df = pd.read_csv(dk_entries_file, nrows=1000, low_memory=False)
        
        logger.info(f"Testing with {len(df)} rows of DKEntries salary data")
        
        # Show sample with 30-day rolling PPG
        logger.info("\nSample players with DKEntries-based salaries:")
        sample = df[['Name', 'position', 'salary', 'rolling_30_ppg', 'calculated_dk_fpts']].head(15)
        sample['value'] = sample['rolling_30_ppg'] / (sample['salary'] / 1000)
        
        for _, row in sample.iterrows():
            logger.info(f"  {row['Name']} ({row['position']}): ${row['salary']:,.0f}, "
                       f"30-day PPG: {row['rolling_30_ppg']:.1f}, "
                       f"Game: {row['calculated_dk_fpts']:.1f}, "
                       f"Value: {row['value']:.2f}")
        
        # Validation stats
        logger.info(f"\nValidation:")
        logger.info(f"  Salary range: ${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")
        logger.info(f"  PPG range: {df['rolling_30_ppg'].min():.1f} - {df['rolling_30_ppg'].max():.1f}")
        logger.info(f"  Salary-PPG correlation: {df['salary'].corr(df['rolling_30_ppg']):.3f}")
        
        logger.info("✅ DKEntries salary generation test completed!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    enhanced_rl_demo()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced RL Demo with Synthetic Salaries')
    parser.add_argument('--full-demo', action='store_true', help='Run full enhanced demo')
    parser.add_argument('--salary-test', action='store_true', help='Test salary generation only')
    
    args = parser.parse_args()
    
    if args.full_demo:
        enhanced_rl_demo()
    elif args.salary_test:
        quick_salary_test()
    else:
        # Run salary test by default (safer)
        quick_salary_test()
        
        print("\n" + "="*50)
        print("To run full demo with DKEntries salary system:")
        print("python enhanced_rl_demo.py --full-demo")
        print("="*50)
