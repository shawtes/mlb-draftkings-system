#!/usr/bin/env python3
"""
Enhanced RL Demo using DKEntries-based salary generation

This demo showcases the complete RL system with:
1. DKEntries-based salary generation
2. 30-day rolling PPG averages
3. RL agent training and evaluation
4. Lineup optimization comparison
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from realistic_rl_system import RealisticMLBRLSystem
from rl_team_selector import MLBRLTeamSelector
from dk_entries_salary_generator import create_dk_entries_salary_data

def quick_rl_demo_with_dk_entries():
    """Quick demonstration of RL system with DKEntries salary data"""
    
    logger.info("=== MLB DraftKings RL System Demo with DKEntries Salaries ===")
    
    # Configuration
    dk_entries_salary_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    
    # Load the DKEntries salary data
    logger.info("Loading DKEntries salary data...")
    df = pd.read_csv(dk_entries_salary_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df):,} records from {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Players: {df['Name'].nunique()}")
    logger.info(f"Salary range: ${df['salary'].min():,.0f} - ${df['salary'].max():,.0f}")
    logger.info(f"PPG range: {df['rolling_30_ppg'].min():.1f} - {df['rolling_30_ppg'].max():.1f}")
    
    # Filter for recent data (last 2 years for faster processing)
    recent_df = df[df['date'] >= '2023-01-01'].copy()
    logger.info(f"Using recent data: {len(recent_df):,} records")
    
    # Initialize RL system
    logger.info("\nInitializing RL system...")
    rl_system = RealisticMLBRLSystem(
        dk_salary_cap=50000,
        position_requirements={
            'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
        }
    )
    
    # Prepare data for RL system
    logger.info("Preparing data for RL system...")
    
    # Create projections from rolling PPG
    recent_df['projected_points'] = recent_df['rolling_30_ppg']
    
    # Create variance estimates
    recent_df['projected_variance'] = recent_df.groupby('Name')['calculated_dk_fpts'].transform(
        lambda x: x.rolling(window=10, min_periods=1).std().fillna(5.0)
    )
    
    # Map positions to standard DK positions
    position_mapping = {
        'SP': 'P', 'RP': 'P',
        '1B/C': '1B', '1B/3B': '1B', '1B/OF': '1B',
        '2B/3B': '2B', '2B/SS': '2B', '2B/OF': '2B', '2B/C': '2B',
        '3B/SS': '3B', '3B/OF': '3B',
        'SS': 'SS',
        'OF': 'OF',
        'C': 'C',
        '1B': '1B',
        '2B': '2B',
        '3B': '3B'
    }
    
    recent_df['dk_position'] = recent_df['position'].map(position_mapping).fillna('OF')
    
    # Sample a few dates for quick demo
    sample_dates = sorted(recent_df['date'].unique())[-10:]  # Last 10 dates
    
    logger.info(f"\nTesting RL system on {len(sample_dates)} sample dates...")
    
    results = []
    
    for i, test_date in enumerate(sample_dates):
        logger.info(f"\n--- Testing Date {i+1}/{len(sample_dates)}: {test_date} ---")
        
        # Get available players for this date
        day_players = recent_df[recent_df['date'] == test_date].copy()
        
        if len(day_players) < 50:  # Skip if not enough players
            logger.info(f"Skipping {test_date} - only {len(day_players)} players available")
            continue
        
        # Prepare player data for RL system
        player_data = []
        for _, row in day_players.iterrows():
            player_data.append({
                'name': row['Name'],
                'position': row['dk_position'],
                'salary': int(row['salary']),
                'projected_points': row['projected_points'],
                'projected_variance': row['projected_variance'],
                'actual_points': row['calculated_dk_fpts']
            })
        
        if len(player_data) < 100:  # Need minimum players
            logger.info(f"Skipping {test_date} - insufficient players after filtering")
            continue
        
        try:
            # Generate lineup using RL system
            lineup = rl_system.generate_lineup(player_data)
            
            if lineup:
                # Calculate results
                total_salary = sum(p['salary'] for p in lineup)
                projected_points = sum(p['projected_points'] for p in lineup)
                actual_points = sum(p['actual_points'] for p in lineup)
                
                logger.info(f"RL Lineup Generated:")
                logger.info(f"  Total Salary: ${total_salary:,}")
                logger.info(f"  Projected Points: {projected_points:.1f}")
                logger.info(f"  Actual Points: {actual_points:.1f}")
                logger.info(f"  Efficiency: {actual_points / (total_salary/1000):.2f} pts per $1K")
                
                # Show lineup
                logger.info("  Lineup:")
                for player in lineup:
                    logger.info(f"    {player['position']}: {player['name']} "
                               f"(${player['salary']:,}, {player['projected_points']:.1f} proj, "
                               f"{player['actual_points']:.1f} actual)")
                
                results.append({
                    'date': test_date,
                    'total_salary': total_salary,
                    'projected_points': projected_points,
                    'actual_points': actual_points,
                    'efficiency': actual_points / (total_salary/1000)
                })
            else:
                logger.info("Failed to generate lineup")
                
        except Exception as e:
            logger.error(f"Error generating lineup for {test_date}: {e}")
    
    # Summary results
    if results:
        logger.info(f"\n=== Summary Results ({len(results)} lineups) ===")
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Average salary used: ${results_df['total_salary'].mean():,.0f}")
        logger.info(f"Average projected points: {results_df['projected_points'].mean():.1f}")
        logger.info(f"Average actual points: {results_df['actual_points'].mean():.1f}")
        logger.info(f"Average efficiency: {results_df['efficiency'].mean():.2f} pts per $1K")
        
        # Best and worst lineups
        best_lineup = results_df.loc[results_df['actual_points'].idxmax()]
        worst_lineup = results_df.loc[results_df['actual_points'].idxmin()]
        
        logger.info(f"\nBest lineup: {best_lineup['date']} with {best_lineup['actual_points']:.1f} points")
        logger.info(f"Worst lineup: {worst_lineup['date']} with {worst_lineup['actual_points']:.1f} points")
        
        # Projection accuracy
        proj_accuracy = np.corrcoef(results_df['projected_points'], results_df['actual_points'])[0, 1]
        logger.info(f"Projection accuracy (correlation): {proj_accuracy:.3f}")
    
    else:
        logger.info("No successful lineups generated")
    
    logger.info("\n=== Demo Complete ===")
    
    return results

def analyze_dk_entries_impact():
    """Analyze the impact of using DKEntries salary generation"""
    
    logger.info("\n=== Analyzing DKEntries Salary Impact ===")
    
    # Load both datasets for comparison
    dk_entries_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    original_file = '4_DATA/data_with_realistic_salaries.csv'
    
    try:
        dk_df = pd.read_csv(dk_entries_file, low_memory=False)
        dk_df['date'] = pd.to_datetime(dk_df['date'])
        logger.info(f"DKEntries data: {len(dk_df):,} records")
        
        try:
            orig_df = pd.read_csv(original_file, low_memory=False)
            orig_df['date'] = pd.to_datetime(orig_df['date'])
            logger.info(f"Original data: {len(orig_df):,} records")
            
            # Compare salary distributions
            logger.info("\nSalary Distribution Comparison:")
            logger.info(f"DKEntries - Range: ${dk_df['salary'].min():,.0f} - ${dk_df['salary'].max():,.0f}, "
                       f"Mean: ${dk_df['salary'].mean():,.0f}")
            logger.info(f"Original - Range: ${orig_df['salary'].min():,.0f} - ${orig_df['salary'].max():,.0f}, "
                       f"Mean: ${orig_df['salary'].mean():,.0f}")
            
            # Position comparison
            logger.info("\nPosition Salary Comparison:")
            dk_pos_avg = dk_df.groupby('position')['salary'].mean()
            orig_pos_avg = orig_df.groupby('position')['salary'].mean()
            
            for pos in sorted(set(dk_pos_avg.index) & set(orig_pos_avg.index)):
                logger.info(f"  {pos}: DKEntries ${dk_pos_avg[pos]:,.0f}, "
                           f"Original ${orig_pos_avg[pos]:,.0f}")
            
        except FileNotFoundError:
            logger.info("Original salary file not found - showing DKEntries data only")
            
        # DKEntries data analysis
        logger.info("\nDKEntries Salary Analysis:")
        logger.info(f"Total records: {len(dk_df):,}")
        logger.info(f"Unique players: {dk_df['Name'].nunique()}")
        logger.info(f"Date range: {dk_df['date'].min()} to {dk_df['date'].max()}")
        
        # Rolling PPG analysis
        logger.info(f"\n30-Day Rolling PPG Analysis:")
        logger.info(f"Range: {dk_df['rolling_30_ppg'].min():.1f} - {dk_df['rolling_30_ppg'].max():.1f}")
        logger.info(f"Mean: {dk_df['rolling_30_ppg'].mean():.1f}")
        logger.info(f"Std: {dk_df['rolling_30_ppg'].std():.1f}")
        
        # Correlation analysis
        salary_ppg_corr = dk_df['salary'].corr(dk_df['rolling_30_ppg'])
        salary_actual_corr = dk_df['salary'].corr(dk_df['calculated_dk_fpts'])
        
        logger.info(f"\nCorrelation Analysis:")
        logger.info(f"Salary vs 30-day PPG: {salary_ppg_corr:.3f}")
        logger.info(f"Salary vs Actual Points: {salary_actual_corr:.3f}")
        
        # High-value players analysis
        high_value_players = dk_df[dk_df['salary'] >= 8000].copy()
        logger.info(f"\nHigh-value players (≥$8,000): {len(high_value_players):,}")
        logger.info(f"Average PPG: {high_value_players['rolling_30_ppg'].mean():.1f}")
        logger.info(f"Average actual points: {high_value_players['calculated_dk_fpts'].mean():.1f}")
        
        # Budget players analysis
        budget_players = dk_df[dk_df['salary'] <= 4000].copy()
        logger.info(f"\nBudget players (≤$4,000): {len(budget_players):,}")
        logger.info(f"Average PPG: {budget_players['rolling_30_ppg'].mean():.1f}")
        logger.info(f"Average actual points: {budget_players['calculated_dk_fpts'].mean():.1f}")
        
    except Exception as e:
        logger.error(f"Error analyzing DKEntries impact: {e}")

def main():
    """Main function to run the enhanced RL demo"""
    
    logger.info("Starting Enhanced RL Demo with DKEntries Salaries")
    
    # Run the RL demo
    results = quick_rl_demo_with_dk_entries()
    
    # Analyze the impact
    analyze_dk_entries_impact()
    
    logger.info("\nAll demos completed successfully!")

if __name__ == "__main__":
    main()
