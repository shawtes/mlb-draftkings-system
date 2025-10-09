#!/usr/bin/env python3
"""
Simple demo using DKEntries salary data with basic lineup selection
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

class SimpleMLBLineupSelector:
    """Simple lineup selector using DKEntries salary data"""
    
    def __init__(self, salary_cap=50000):
        self.salary_cap = salary_cap
        self.position_requirements = {
            'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
        }
        
    def select_lineup(self, players_df):
        """Select optimal lineup using greedy approach based on value (points/salary)"""
        
        # Calculate value per dollar
        players_df['value'] = players_df['rolling_30_ppg'] / (players_df['salary'] / 1000)
        
        # Sort by value
        players_df = players_df.sort_values('value', ascending=False)
        
        selected = []
        remaining_budget = self.salary_cap
        position_counts = {pos: 0 for pos in self.position_requirements.keys()}
        
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
        
        # Add mapped position
        players_df['dk_position'] = players_df['position'].map(position_mapping).fillna('OF')
        
        # First pass: select highest value players for each position
        for _, player in players_df.iterrows():
            pos = player['dk_position']
            
            if pos in position_counts and position_counts[pos] < self.position_requirements[pos]:
                if player['salary'] <= remaining_budget:
                    selected.append(player)
                    remaining_budget -= player['salary']
                    position_counts[pos] += 1
                    
                    # Check if we've filled all positions
                    if sum(position_counts.values()) == sum(self.position_requirements.values()):
                        break
        
        # Check if we have a complete lineup
        if sum(position_counts.values()) == sum(self.position_requirements.values()):
            return selected
        else:
            return None
    
    def validate_lineup(self, lineup):
        """Validate that lineup meets DK requirements"""
        if not lineup:
            return False
            
        position_counts = {}
        total_salary = 0
        
        for player in lineup:
            pos = player['dk_position']
            position_counts[pos] = position_counts.get(pos, 0) + 1
            total_salary += player['salary']
        
        # Check position requirements
        for pos, required in self.position_requirements.items():
            if position_counts.get(pos, 0) != required:
                return False
        
        # Check salary cap
        if total_salary > self.salary_cap:
            return False
            
        return True

def demo_dk_entries_lineup_selection():
    """Demo lineup selection using DKEntries salary data"""
    
    logger.info("=== DKEntries Lineup Selection Demo ===")
    
    # Load the DKEntries salary data
    dk_entries_salary_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    
    logger.info("Loading DKEntries salary data...")
    df = pd.read_csv(dk_entries_salary_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df):,} records from {df['date'].min()} to {df['date'].max()}")
    
    # Initialize lineup selector
    selector = SimpleMLBLineupSelector(salary_cap=50000)
    
    # Get recent dates for testing
    recent_dates = sorted(df['date'].unique())[-20:]  # Last 20 dates
    
    logger.info(f"Testing lineup selection on {len(recent_dates)} recent dates...")
    
    results = []
    
    for i, test_date in enumerate(recent_dates):
        logger.info(f"\n--- Date {i+1}/{len(recent_dates)}: {test_date} ---")
        
        # Get players for this date
        day_players = df[df['date'] == test_date].copy()
        
        if len(day_players) < 50:
            logger.info(f"Skipping {test_date} - only {len(day_players)} players")
            continue
        
        # Select lineup
        lineup = selector.select_lineup(day_players)
        
        if lineup and selector.validate_lineup(lineup):
            # Calculate stats
            total_salary = sum(p['salary'] for p in lineup)
            projected_points = sum(p['rolling_30_ppg'] for p in lineup)
            actual_points = sum(p['calculated_dk_fpts'] for p in lineup)
            
            logger.info(f"Lineup selected:")
            logger.info(f"  Total salary: ${total_salary:,}")
            logger.info(f"  Projected points: {projected_points:.1f}")
            logger.info(f"  Actual points: {actual_points:.1f}")
            logger.info(f"  Value efficiency: {projected_points/(total_salary/1000):.2f} pts per $1K")
            
            # Show lineup
            logger.info("  Players:")
            for player in lineup:
                logger.info(f"    {player['dk_position']}: {player['Name']} "
                           f"(${player['salary']:,}, {player['rolling_30_ppg']:.1f} proj, "
                           f"{player['calculated_dk_fpts']:.1f} actual)")
            
            results.append({
                'date': test_date,
                'total_salary': total_salary,
                'projected_points': projected_points,
                'actual_points': actual_points,
                'players': len(lineup)
            })
        else:
            logger.info("Failed to create valid lineup")
    
    # Summary
    if results:
        logger.info(f"\n=== Summary ({len(results)} successful lineups) ===")
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Average salary used: ${results_df['total_salary'].mean():,.0f}")
        logger.info(f"Average projected points: {results_df['projected_points'].mean():.1f}")
        logger.info(f"Average actual points: {results_df['actual_points'].mean():.1f}")
        
        # Projection accuracy
        proj_accuracy = np.corrcoef(results_df['projected_points'], results_df['actual_points'])[0, 1]
        logger.info(f"Projection accuracy: {proj_accuracy:.3f}")
        
        # Best lineup
        best_idx = results_df['actual_points'].idxmax()
        best_result = results_df.loc[best_idx]
        logger.info(f"Best lineup: {best_result['date']} with {best_result['actual_points']:.1f} points")
        
        # Show performance distribution
        logger.info(f"\nPerformance distribution:")
        logger.info(f"  Min: {results_df['actual_points'].min():.1f} points")
        logger.info(f"  Max: {results_df['actual_points'].max():.1f} points")
        logger.info(f"  Median: {results_df['actual_points'].median():.1f} points")
        logger.info(f"  Std: {results_df['actual_points'].std():.1f} points")
        
    else:
        logger.info("No successful lineups created")
    
    return results

def analyze_salary_performance():
    """Analyze how DKEntries salaries correlate with performance"""
    
    logger.info("\n=== DKEntries Salary Performance Analysis ===")
    
    # Load data
    dk_entries_salary_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    df = pd.read_csv(dk_entries_salary_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create salary tiers
    df['salary_tier'] = pd.cut(df['salary'], 
                               bins=[0, 4000, 6000, 8000, 12000], 
                               labels=['Budget', 'Mid', 'High', 'Elite'])
    
    # Analyze by salary tier
    logger.info("Performance by salary tier:")
    tier_stats = df.groupby('salary_tier').agg({
        'rolling_30_ppg': ['mean', 'std', 'count'],
        'calculated_dk_fpts': ['mean', 'std'],
        'salary': ['mean', 'min', 'max']
    }).round(2)
    
    for tier in ['Budget', 'Mid', 'High', 'Elite']:
        if tier in tier_stats.index:
            stats = tier_stats.loc[tier]
            logger.info(f"  {tier}: {stats[('rolling_30_ppg', 'count')]} players")
            logger.info(f"    Avg salary: ${stats[('salary', 'mean')]:,.0f}")
            logger.info(f"    Avg 30-day PPG: {stats[('rolling_30_ppg', 'mean')]:.1f}")
            logger.info(f"    Avg actual points: {stats[('calculated_dk_fpts', 'mean')]:.1f}")
    
    # Position analysis
    logger.info("\nPerformance by position:")
    pos_stats = df.groupby('position').agg({
        'rolling_30_ppg': 'mean',
        'calculated_dk_fpts': 'mean',
        'salary': 'mean'
    }).round(2)
    
    for pos in pos_stats.index:
        stats = pos_stats.loc[pos]
        logger.info(f"  {pos}: Avg salary ${stats['salary']:,.0f}, "
                   f"PPG {stats['rolling_30_ppg']:.1f}, "
                   f"Actual {stats['calculated_dk_fpts']:.1f}")
    
    # Value analysis (points per $1000)
    df['value_30day'] = df['rolling_30_ppg'] / (df['salary'] / 1000)
    df['value_actual'] = df['calculated_dk_fpts'] / (df['salary'] / 1000)
    
    logger.info(f"\nValue Analysis:")
    logger.info(f"Average value (30-day PPG per $1K): {df['value_30day'].mean():.2f}")
    logger.info(f"Average value (actual points per $1K): {df['value_actual'].mean():.2f}")
    
    # Best value players
    best_value_players = df.nlargest(10, 'value_30day')
    logger.info(f"\nBest value players (30-day PPG per $1K):")
    for _, player in best_value_players.iterrows():
        logger.info(f"  {player['Name']} ({player['position']}): "
                   f"{player['value_30day']:.2f} pts/$1K, "
                   f"${player['salary']:,}, {player['rolling_30_ppg']:.1f} PPG")

def main():
    """Main demo function"""
    
    logger.info("Starting DKEntries Lineup Selection Demo")
    
    # Run lineup selection demo
    results = demo_dk_entries_lineup_selection()
    
    # Analyze salary performance
    analyze_salary_performance()
    
    logger.info("\nDemo completed successfully!")
    
    return results

if __name__ == "__main__":
    main()
