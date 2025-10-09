#!/usr/bin/env python3
"""
Enhanced Salary Generator Using 30-Day PPG Averages

This module creates realistic DraftKings salaries based on:
1. 30-day rolling PPG (Points Per Game) averages
2. Position-based salary multipliers
3. Market dynamics and ownership patterns
4. Real contest data analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPGBasedSalaryGenerator:
    """Enhanced salary generator using 30-day PPG averages"""
    
    def __init__(self, contest_data_path: str = None):
        self.contest_data_path = contest_data_path
        self.salary_stats = {}
        self.position_multipliers = {}
        self.ppg_tiers = {}
        
        # Default DraftKings salary ranges by position
        self.position_salary_ranges = {
            'C': (3000, 11000),
            '1B': (3200, 12500),
            '2B': (3000, 11500),
            '3B': (3100, 12000),
            'SS': (3200, 12500),
            'OF': (3000, 13000),
            'P': (3500, 11000)  # Pitchers
        }
        
        # PPG-to-salary relationship parameters
        self.ppg_salary_base = {
            'C': 2800,
            '1B': 3000,
            '2B': 2900,
            '3B': 2950,
            'SS': 3000,
            'OF': 2850,
            'P': 3300
        }
        
        # PPG multipliers (how much salary increases per PPG point)
        self.ppg_multipliers = {
            'C': 450,    # Lower multiplier for catchers
            '1B': 500,   # Standard for power positions
            '2B': 480,   # Slightly lower for middle infield
            '3B': 490,   # Similar to 1B
            'SS': 495,   # Premium for SS
            'OF': 470,   # Slightly lower due to abundance
            'P': 380     # Different scale for pitchers
        }
        
    def load_contest_data(self):
        """Load contest data to analyze real salary patterns"""
        if self.contest_data_path and pd.io.common.file_exists(self.contest_data_path):
            logger.info("Loading contest data for salary analysis...")
            contest_df = pd.read_csv(self.contest_data_path)
            
            # Extract player data from contest
            player_data = []
            for _, row in contest_df.iterrows():
                if pd.notna(row.get('Player')) and pd.notna(row.get('FPTS')):
                    player_data.append({
                        'name': row['Player'],
                        'position': row.get('Roster Position', 'OF'),
                        'fpts': float(row['FPTS']),
                        'ownership': float(row.get('%Drafted', '0').replace('%', ''))
                    })
            
            if player_data:
                contest_players_df = pd.DataFrame(player_data)
                logger.info(f"Loaded {len(contest_players_df)} players from contest data")
                return contest_players_df
            
        logger.info("No contest data available, using default parameters")
        return None
    
    def calculate_30_day_ppg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 30-day rolling PPG averages for each player"""
        logger.info("Calculating 30-day PPG averages...")
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Sort by player and date
        df = df.sort_values(['Name', 'date'])
        
        # Calculate PPG for each player with 30-day rolling window
        enhanced_data = []
        
        for player_name in df['Name'].unique():
            player_data = df[df['Name'] == player_name].copy()
            
            # Calculate rolling averages
            player_data['ppg_30day'] = player_data['calculated_dk_fpts'].rolling(
                window=30, min_periods=5, center=False
            ).mean()
            
            # For early games with insufficient history, use available average
            player_data['ppg_30day'] = player_data['ppg_30day'].fillna(
                player_data['calculated_dk_fpts'].expanding(min_periods=1).mean()
            )
            
            # Calculate recent form (last 10 games)
            player_data['ppg_10day'] = player_data['calculated_dk_fpts'].rolling(
                window=10, min_periods=3, center=False
            ).mean()
            
            player_data['ppg_10day'] = player_data['ppg_10day'].fillna(
                player_data['calculated_dk_fpts'].expanding(min_periods=1).mean()
            )
            
            # Calculate season average up to each date
            player_data['season_ppg'] = player_data['calculated_dk_fpts'].expanding(
                min_periods=1
            ).mean()
            
            enhanced_data.append(player_data)
        
        result_df = pd.concat(enhanced_data, ignore_index=True)
        logger.info(f"Calculated PPG averages for {len(result_df)} player-game records")
        
        return result_df
    
    def assign_ppg_based_salaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign salaries based on 30-day PPG averages"""
        logger.info("Assigning salaries based on PPG averages...")
        
        df = df.copy()
        df['salary'] = 0
        
        for position in df['position'].unique():
            if position not in self.position_salary_ranges:
                continue
                
            pos_data = df[df['position'] == position].copy()
            
            # Get salary range for position
            min_salary, max_salary = self.position_salary_ranges[position]
            base_salary = self.ppg_salary_base.get(position, 3000)
            ppg_multiplier = self.ppg_multipliers.get(position, 400)
            
            # Calculate salary based on PPG
            pos_data['base_ppg_salary'] = (
                base_salary + 
                (pos_data['ppg_30day'] * ppg_multiplier)
            )
            
            # Add form adjustment (recent vs long-term performance)
            form_adjustment = (
                (pos_data['ppg_10day'] - pos_data['ppg_30day']) * 
                (ppg_multiplier * 0.3)  # 30% weight to recent form
            )
            pos_data['base_ppg_salary'] += form_adjustment
            
            # Add market variance (simulate market inefficiencies)
            market_variance = np.random.normal(0, ppg_multiplier * 0.15, len(pos_data))
            pos_data['base_ppg_salary'] += market_variance
            
            # Ensure salaries are within realistic ranges
            pos_data['salary'] = np.clip(
                pos_data['base_ppg_salary'],
                min_salary,
                max_salary
            )
            
            # Round to nearest $100 (DraftKings standard)
            pos_data['salary'] = (pos_data['salary'] / 100).round() * 100
            
            # Update main dataframe
            df.loc[df['position'] == position, 'salary'] = pos_data['salary']
            
            logger.info(f"  {position}: {len(pos_data)} players, "
                       f"salary range ${pos_data['salary'].min():.0f} - ${pos_data['salary'].max():.0f}, "
                       f"avg PPG {pos_data['ppg_30day'].mean():.2f}")
        
        return df
    
    def add_market_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market dynamics like star player premiums and value adjustments"""
        logger.info("Adding market dynamics...")
        
        df = df.copy()
        
        # Star player premium (top performers get salary boost)
        for position in df['position'].unique():
            pos_data = df[df['position'] == position]
            
            if len(pos_data) > 0:
                # Top 10% of performers get premium
                top_threshold = pos_data['ppg_30day'].quantile(0.9)
                star_players = pos_data['ppg_30day'] >= top_threshold
                
                # Add 5-15% premium for star players
                premium_multiplier = np.random.uniform(1.05, 1.15, star_players.sum())
                df.loc[(df['position'] == position) & star_players, 'salary'] *= premium_multiplier
        
        # Value player adjustments (some high-PPG players priced lower)
        # This simulates market inefficiencies
        value_adjustment_chance = 0.1  # 10% of players get value pricing
        value_players = np.random.random(len(df)) < value_adjustment_chance
        value_discount = np.random.uniform(0.85, 0.95, value_players.sum())
        df.loc[value_players, 'salary'] *= value_discount
        
        # Ensure salaries stay within bounds and round to nearest $100
        for position in df['position'].unique():
            if position in self.position_salary_ranges:
                min_sal, max_sal = self.position_salary_ranges[position]
                mask = df['position'] == position
                df.loc[mask, 'salary'] = np.clip(df.loc[mask, 'salary'], min_sal, max_sal)
                df.loc[mask, 'salary'] = (df.loc[mask, 'salary'] / 100).round() * 100
        
        return df
    
    def generate_enhanced_salaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to generate enhanced salaries using PPG"""
        logger.info("=== Enhanced PPG-Based Salary Generation ===")
        
        # Load contest data if available
        contest_data = self.load_contest_data()
        
        # Step 1: Calculate 30-day PPG averages
        df_with_ppg = self.calculate_30_day_ppg(df)
        
        # Step 2: Assign salaries based on PPG
        df_with_salaries = self.assign_ppg_based_salaries(df_with_ppg)
        
        # Step 3: Add market dynamics
        df_final = self.add_market_dynamics(df_with_salaries)
        
        # Step 4: Validate and report
        self.validate_salary_generation(df_final)
        
        return df_final
    
    def validate_salary_generation(self, df: pd.DataFrame):
        """Validate the generated salaries"""
        logger.info("=== Salary Generation Validation ===")
        
        total_players = len(df)
        logger.info(f"Total players with salaries: {total_players:,}")
        
        if total_players == 0:
            logger.error("No players have salaries assigned!")
            return
        
        # Overall salary stats
        logger.info(f"Salary range: ${df['salary'].min():.0f} - ${df['salary'].max():.0f}")
        logger.info(f"Average salary: ${df['salary'].mean():.0f}")
        logger.info(f"Median salary: ${df['salary'].median():.0f}")
        
        # PPG correlation
        if 'ppg_30day' in df.columns:
            correlation = df['salary'].corr(df['ppg_30day'])
            logger.info(f"Salary-PPG correlation: {correlation:.3f}")
        
        # Position breakdown
        logger.info("\nPosition breakdown:")
        position_stats = df.groupby('position').agg({
            'salary': ['count', 'mean', 'min', 'max'],
            'ppg_30day': 'mean' if 'ppg_30day' in df.columns else 'count'
        }).round(0)
        
        for position in position_stats.index:
            count = position_stats.loc[position, ('salary', 'count')]
            avg_sal = position_stats.loc[position, ('salary', 'mean')]
            min_sal = position_stats.loc[position, ('salary', 'min')]
            max_sal = position_stats.loc[position, ('salary', 'max')]
            
            if 'ppg_30day' in df.columns:
                avg_ppg = position_stats.loc[position, ('ppg_30day', 'mean')]
                logger.info(f"  {position}: {count} players, ${avg_sal:.0f} avg salary "
                           f"(${min_sal:.0f}-${max_sal:.0f}), {avg_ppg:.1f} avg PPG")
            else:
                logger.info(f"  {position}: {count} players, ${avg_sal:.0f} avg salary "
                           f"(${min_sal:.0f}-${max_sal:.0f})")
        
        logger.info("✅ Salary generation validation completed!")

def main():
    """Test the PPG-based salary generator"""
    
    # Test data paths
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    CONTEST_PATH = 'contest-standings-178312097.csv'
    OUTPUT_PATH = 'data_with_ppg_salaries.csv'
    
    try:
        logger.info("=== PPG-Based Salary Generator Test ===")
        
        # Load test data
        logger.info("Loading test data...")
        df = pd.read_csv(DATA_PATH, nrows=5000, low_memory=False)  # Test with subset
        
        # Initialize generator
        generator = PPGBasedSalaryGenerator(contest_data_path=CONTEST_PATH)
        
        # Generate salaries
        enhanced_df = generator.generate_enhanced_salaries(df)
        
        # Save results
        enhanced_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Enhanced data saved to {OUTPUT_PATH}")
        
        # Show sample results
        logger.info("\nSample results:")
        sample_cols = ['Name', 'position', 'date', 'calculated_dk_fpts', 'ppg_30day', 'salary']
        available_cols = [col for col in sample_cols if col in enhanced_df.columns]
        sample = enhanced_df[available_cols].head(10)
        logger.info(sample.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
