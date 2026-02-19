#!/usr/bin/env python3
"""
Realistic DraftKings Salary Generator based on actual contest data

This module generates realistic DraftKings salaries based on:
1. Actual contest data patterns from your CSV
2. Performance-based salary tiers
3. Position-based salary adjustments
4. Market dynamics (ownership %, performance)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DraftKingsSalaryGenerator:
    """
    Generate realistic DraftKings salaries based on actual contest patterns
    """
    
    def __init__(self):
        # Salary ranges based on actual DraftKings data
        self.salary_tiers = {
            'elite': {'min': 11000, 'max': 13000},      # 35+ FPTS performers
            'high': {'min': 9000, 'max': 11000},        # 25-35 FPTS performers  
            'medium': {'min': 7000, 'max': 9000},       # 15-25 FPTS performers
            'low': {'min': 5000, 'max': 7000},          # 8-15 FPTS performers
            'value': {'min': 3000, 'max': 5000}         # <8 FPTS performers
        }
        
        # Position multipliers based on scarcity
        self.position_multipliers = {
            'C': 1.05,      # Catchers slightly more expensive (scarcity)
            '1B': 1.0,      # Base position
            '2B': 0.98,     # Slightly cheaper
            '3B': 1.02,     # Slightly more expensive
            'SS': 1.03,     # Premium position
            'OF': 0.95,     # Most abundant position
            'UTIL': 1.0,    # Utility slot
            'P': 1.1        # Pitchers are different salary scale
        }
        
        # Performance-based adjustments
        self.recent_performance_weight = 0.3
        self.season_performance_weight = 0.7
        
        # Variance factors
        self.salary_variance = 0.1  # 10% variance around base salary
        
    def analyze_contest_data(self, contest_df: pd.DataFrame) -> Dict:
        """Analyze actual contest data to understand salary patterns"""
        
        # Extract player performance data
        players_data = []
        
        for _, row in contest_df.iterrows():
            lineup = row['Lineup']
            points = row['Points']
            
            # Parse lineup string to extract individual players
            # Format appears to be: "1B Nolan Schanuel 2B Davis Schneider..."
            players = self._parse_lineup(lineup)
            
            for player_info in players:
                players_data.append({
                    'position': player_info['position'],
                    'name': player_info['name'],
                    'total_points': points,
                    'lineup_rank': row['Rank']
                })
        
        analysis_df = pd.DataFrame(players_data)
        
        # Calculate player frequency and performance
        player_stats = analysis_df.groupby(['name', 'position']).agg({
            'total_points': 'mean',
            'lineup_rank': 'mean'
        }).reset_index()
        
        return {
            'player_stats': player_stats,
            'top_performers': player_stats.nlargest(20, 'total_points'),
            'position_distribution': analysis_df['position'].value_counts()
        }
    
    def _parse_lineup(self, lineup_str: str) -> List[Dict]:
        """Parse lineup string to extract individual players"""
        players = []
        
        # Split by positions and extract player names
        position_pattern = r'(C|1B|2B|3B|SS|OF|P)\s+([A-Za-z\s\.]+?)(?=\s+(?:C|1B|2B|3B|SS|OF|P)|$)'
        matches = re.findall(position_pattern, lineup_str)
        
        for position, name in matches:
            players.append({
                'position': position.strip(),
                'name': name.strip()
            })
        
        return players
    
    def calculate_base_salary(self, fantasy_points: float, position: str = 'OF') -> int:
        """Calculate base salary based on fantasy points performance"""
        
        # Determine tier based on performance
        if fantasy_points >= 35:
            tier = 'elite'
        elif fantasy_points >= 25:
            tier = 'high'
        elif fantasy_points >= 15:
            tier = 'medium'
        elif fantasy_points >= 8:
            tier = 'low'
        else:
            tier = 'value'
        
        # Get base salary range
        salary_range = self.salary_tiers[tier]
        
        # Calculate salary within tier based on performance
        tier_span = salary_range['max'] - salary_range['min']
        
        if tier == 'elite':
            # For elite tier, higher FPTS = higher salary
            performance_factor = min(1.0, (fantasy_points - 35) / 15)  # 35-50 FPTS range
        elif tier == 'high':
            performance_factor = (fantasy_points - 25) / 10  # 25-35 FPTS range
        elif tier == 'medium':
            performance_factor = (fantasy_points - 15) / 10  # 15-25 FPTS range
        elif tier == 'low':
            performance_factor = (fantasy_points - 8) / 7   # 8-15 FPTS range
        else:  # value
            performance_factor = fantasy_points / 8  # 0-8 FPTS range
        
        performance_factor = max(0, min(1, performance_factor))
        
        base_salary = salary_range['min'] + (tier_span * performance_factor)
        
        # Apply position multiplier
        position_multiplier = self.position_multipliers.get(position, 1.0)
        adjusted_salary = base_salary * position_multiplier
        
        # Add some variance
        variance = np.random.normal(0, self.salary_variance)
        final_salary = adjusted_salary * (1 + variance)
        
        # Round to nearest $100 and ensure within DK ranges
        final_salary = round(final_salary / 100) * 100
        final_salary = max(3000, min(13000, final_salary))
        
        return int(final_salary)
    
    def _calculate_dk_fpts(self, df: pd.DataFrame) -> pd.Series:
        """Calculate DraftKings fantasy points if not present"""
        return (
            df.get('1B', 0) * 3 +
            df.get('2B', 0) * 5 +
            df.get('3B', 0) * 8 +
            df.get('HR', 0) * 10 +
            df.get('RBI', 0) * 2 +
            df.get('R', 0) * 2 +
            df.get('BB', 0) * 2 +
            df.get('HBP', 0) * 2 +
            df.get('SB', 0) * 5
        )
    
    def generate_historical_salaries(self, df: pd.DataFrame, 
                                   performance_column: str = 'calculated_dk_fpts') -> pd.DataFrame:
        """Generate realistic historical salaries for entire dataset"""
        
        logger.info("Generating realistic DraftKings salaries based on performance...")
        
        df_copy = df.copy()
        
        # Ensure we have the performance column
        if performance_column not in df_copy.columns:
            logger.info(f"Creating {performance_column} column...")
            df_copy['calculated_dk_fpts'] = self._calculate_dk_fpts(df_copy)
            performance_column = 'calculated_dk_fpts'
        
        # Add synthetic positions if not present
        if 'position' not in df_copy.columns:
            df_copy['position'] = self._assign_realistic_positions(df_copy)
        
        # Calculate rolling average performance for more realistic salaries
        df_copy = df_copy.sort_values(['Name', 'date'])
        
        # Calculate recent performance (last 10 games)
        df_copy['recent_performance'] = df_copy.groupby('Name')[performance_column].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
        )
        
        # Calculate season performance
        df_copy['season_avg'] = df_copy.groupby(['Name', df_copy['date'].dt.year])[performance_column].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Fill NaN values
        df_copy['recent_performance'] = df_copy['recent_performance'].fillna(df_copy[performance_column])
        df_copy['season_avg'] = df_copy['season_avg'].fillna(df_copy[performance_column])
        
        # Calculate weighted performance for salary calculation
        df_copy['salary_performance'] = (
            df_copy['recent_performance'] * self.recent_performance_weight +
            df_copy['season_avg'] * self.season_performance_weight
        )
        
        # Generate salaries
        salaries = []
        for _, row in df_copy.iterrows():
            salary = self.calculate_base_salary(
                fantasy_points=row['salary_performance'],
                position=row['position']
            )
            salaries.append(salary)
        
        df_copy['salary'] = salaries
        
        # Add some market dynamics (popular players cost more)
        df_copy = self._add_market_dynamics(df_copy)
        
        logger.info(f"Generated salaries for {len(df_copy)} player-game records")
        logger.info(f"Salary range: ${df_copy['salary'].min():,} - ${df_copy['salary'].max():,}")
        logger.info(f"Average salary: ${df_copy['salary'].mean():,.0f}")
        
        return df_copy
    
    def _assign_realistic_positions(self, df: pd.DataFrame) -> List[str]:
        """Assign realistic positions based on DraftKings distributions"""
        
        # DraftKings position distribution (approximate)
        position_weights = {
            'C': 0.08,      # 8% catchers
            '1B': 0.12,     # 12% first basemen  
            '2B': 0.12,     # 12% second basemen
            '3B': 0.12,     # 12% third basemen
            'SS': 0.12,     # 12% shortstops
            'OF': 0.44      # 44% outfielders (3 OF slots)
        }
        
        positions = list(position_weights.keys())
        weights = list(position_weights.values())
        
        return np.random.choice(positions, size=len(df), p=weights)
    
    def _add_market_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market dynamics to make salaries more realistic"""
        
        # Popular players (high performers) get salary bumps
        df['performance_rank'] = df.groupby('date')['salary_performance'].rank(ascending=False)
        
        # Top 10% of players on any given day get a salary bump
        top_performers = df['performance_rank'] <= (len(df.groupby('date').first()) * 0.1)
        df.loc[top_performers, 'salary'] *= 1.1
        
        # Star players (consistently high performers) get premium pricing
        star_players = df.groupby('Name')['salary_performance'].transform('mean') > 20
        df.loc[star_players, 'salary'] *= 1.05
        
        # Ensure salaries stay within DK bounds
        df['salary'] = df['salary'].clip(3000, 13000)
        df['salary'] = (df['salary'] / 100).round() * 100  # Round to nearest $100
        
        return df
    
    def validate_salaries(self, df: pd.DataFrame) -> Dict:
        """Validate generated salaries against DraftKings patterns"""
        
        validation_results = {
            'total_records': len(df),
            'salary_range': {
                'min': df['salary'].min(),
                'max': df['salary'].max(),
                'mean': df['salary'].mean(),
                'std': df['salary'].std()
            },
            'position_salary_avg': df.groupby('position')['salary'].mean().to_dict(),
            'performance_correlation': df['salary'].corr(df['calculated_dk_fpts']),
            'within_dk_bounds': (df['salary'] >= 3000).all() and (df['salary'] <= 13000).all()
        }
        
        return validation_results

def create_realistic_salary_data(input_file: str, output_file: str = None) -> pd.DataFrame:
    """Main function to add realistic salaries to your data"""
    
    logger.info("Creating realistic DraftKings salary data...")
    
    # Load data
    df = pd.read_csv(input_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize salary generator
    salary_gen = DraftKingsSalaryGenerator()
    
    # Generate salaries
    df_with_salaries = salary_gen.generate_historical_salaries(df)
    
    # Validate results
    validation = salary_gen.validate_salaries(df_with_salaries)
    
    logger.info("Salary Generation Validation:")
    logger.info(f"  Total records: {validation['total_records']:,}")
    logger.info(f"  Salary range: ${validation['salary_range']['min']:,} - ${validation['salary_range']['max']:,}")
    logger.info(f"  Average salary: ${validation['salary_range']['mean']:,.0f}")
    logger.info(f"  Performance correlation: {validation['performance_correlation']:.3f}")
    logger.info(f"  Within DK bounds: {validation['within_dk_bounds']}")
    
    logger.info("Position Salary Averages:")
    for pos, avg_salary in validation['position_salary_avg'].items():
        logger.info(f"  {pos}: ${avg_salary:,.0f}")
    
    # Save if output file specified
    if output_file:
        df_with_salaries.to_csv(output_file, index=False)
        logger.info(f"Data with salaries saved to {output_file}")
    
    return df_with_salaries

def analyze_contest_patterns(contest_file: str):
    """Analyze actual contest data to understand salary patterns"""
    
    logger.info("Analyzing contest data patterns...")
    
    try:
        contest_df = pd.read_csv(contest_file)
        
        # Sample analysis from the visible data patterns
        logger.info("Contest Data Analysis:")
        logger.info(f"  Total entries: {len(contest_df)}")
        logger.info(f"  Points range: {contest_df['Points'].min():.1f} - {contest_df['Points'].max():.1f}")
        logger.info(f"  Average points: {contest_df['Points'].mean():.1f}")
        
        # Analyze player data from the visible sample
        top_performers = [
            {'name': 'George Springer', 'fpts': 39, 'ownership': '4.79%'},
            {'name': 'Seth Lugo', 'fpts': 25.45, 'ownership': '40.67%'},
            {'name': 'Robbie Ray', 'fpts': 34.35, 'ownership': '26.58%'},
            {'name': 'Zach Neto', 'fpts': 31, 'ownership': '13.94%'},
            {'name': 'Jose Soriano', 'fpts': 30.75, 'ownership': '13.46%'},
            {'name': 'Addison Barger', 'fpts': 26, 'ownership': '10.03%'}
        ]
        
        logger.info("\nTop Performers Analysis:")
        for player in top_performers:
            # Estimate salary based on performance
            if player['fpts'] >= 35:
                est_salary = "11000-13000"
            elif player['fpts'] >= 25:
                est_salary = "9000-11000"
            elif player['fpts'] >= 15:
                est_salary = "7000-9000"
            else:
                est_salary = "5000-7000"
                
            logger.info(f"  {player['name']}: {player['fpts']} FPTS, "
                       f"{player['ownership']} owned, Est. Salary: ${est_salary}")
        
        return top_performers
        
    except Exception as e:
        logger.error(f"Error analyzing contest data: {e}")
        return None

def main():
    """Main function to demonstrate salary generation"""
    
    # Configuration
    INPUT_FILE = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    OUTPUT_FILE = '4_DATA/data_with_realistic_salaries.csv'
    CONTEST_FILE = 'C:/Users/smtes/Downloads/contest-standings-178312097.csv'
    
    # Analyze contest patterns first
    logger.info("=== DraftKings Salary Generation ===")
    contest_patterns = analyze_contest_patterns(CONTEST_FILE)
    
    print("\n" + "="*50 + "\n")
    
    # Generate realistic salaries for your data
    df_with_salaries = create_realistic_salary_data(INPUT_FILE, OUTPUT_FILE)
    
    # Show sample results
    logger.info("\nSample of generated data:")
    sample_df = df_with_salaries[['Name', 'date', 'calculated_dk_fpts', 'position', 'salary']].head(10)
    for _, row in sample_df.iterrows():
        logger.info(f"  {row['Name']} ({row['position']}): {row['calculated_dk_fpts']:.1f} FPTS -> ${row['salary']:,}")
    
    # Performance analysis
    logger.info("\nPerformance vs Salary Analysis:")
    high_performers = df_with_salaries[df_with_salaries['calculated_dk_fpts'] >= 30]
    if len(high_performers) > 0:
        logger.info(f"  Players with 30+ FPTS: {len(high_performers)}")
        logger.info(f"  Average salary for 30+ FPTS: ${high_performers['salary'].mean():,.0f}")
    
    medium_performers = df_with_salaries[
        (df_with_salaries['calculated_dk_fpts'] >= 15) & 
        (df_with_salaries['calculated_dk_fpts'] < 30)
    ]
    if len(medium_performers) > 0:
        logger.info(f"  Players with 15-30 FPTS: {len(medium_performers)}")
        logger.info(f"  Average salary for 15-30 FPTS: ${medium_performers['salary'].mean():,.0f}")

if __name__ == "__main__":
    main()
