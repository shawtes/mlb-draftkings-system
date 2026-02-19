#!/usr/bin/env python3
"""
Synthetic Salary Generator for MLB DraftKings Data

This module creates realistic salary data based on player performance metrics.
Since historical salary data is not readily available, we'll generate synthetic
salaries that correlate with player performance and follow DraftKings patterns.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticSalaryGenerator:
    """
    Generate realistic DraftKings salaries based on player performance
    
    This class creates synthetic salary data that:
    1. Correlates with player performance metrics
    2. Follows realistic DraftKings salary ranges
    3. Includes position-based adjustments
    4. Accounts for seasonal trends
    5. Adds realistic variance
    """
    
    def __init__(self):
        # DraftKings salary ranges (based on typical patterns)
        self.salary_ranges = {
            'min_salary': 2000,
            'max_salary': 13000,
            'position_multipliers': {
                'C': 0.85,    # Catchers typically cheaper
                '1B': 1.0,    # Baseline
                '2B': 0.95,   # Slightly cheaper
                '3B': 1.05,   # Slightly more expensive
                'SS': 1.10,   # Premium position
                'OF': 1.0,    # Baseline
                'UTIL': 1.0   # Utility same as position
            },
            'performance_weights': {
                'calculated_dk_fpts': 0.4,    # Primary factor
                'AVG': 0.15,                   # Batting average
                'OBP': 0.15,                   # On-base percentage
                'SLG': 0.15,                   # Slugging percentage
                'HR': 0.10,                    # Home runs
                'RBI': 0.05                    # RBIs
            }
        }
        
        # Performance percentiles for salary bands
        self.salary_bands = {
            'superstar': {'min_pct': 95, 'salary_range': (11000, 13000)},
            'star': {'min_pct': 85, 'salary_range': (9000, 11000)},
            'solid': {'min_pct': 70, 'salary_range': (7000, 9000)},
            'average': {'min_pct': 50, 'salary_range': (5000, 7000)},
            'below_avg': {'min_pct': 30, 'salary_range': (3500, 5000)},
            'minimum': {'min_pct': 0, 'salary_range': (2000, 3500)}
        }
        
        self.scaler = StandardScaler()
        self.salary_model = None
        
    def calculate_performance_score(self, df):
        """Calculate a composite performance score for each player"""
        logger.info("Calculating performance scores...")
        
        # Ensure required columns exist
        required_cols = ['calculated_dk_fpts', 'AVG', 'OBP', 'SLG', 'HR', 'RBI']
        for col in required_cols:
            if col not in df.columns:
                if col == 'calculated_dk_fpts':
                    df[col] = self._calculate_dk_fpts(df)
                else:
                    df[col] = 0
                    logger.warning(f"Column {col} not found, setting to 0")
        
        # Calculate weighted performance score
        performance_score = np.zeros(len(df))
        for metric, weight in self.salary_ranges['performance_weights'].items():
            if metric in df.columns:
                # Normalize the metric (handle potential inf/nan values)
                metric_values = df[metric].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Only normalize if there's variation in the data
                if metric_values.std() > 0 and not metric_values.isna().all():
                    normalized = (metric_values - metric_values.mean()) / metric_values.std()
                    # Replace any remaining NaN with 0
                    normalized = normalized.fillna(0)
                else:
                    normalized = np.zeros(len(metric_values))
                
                performance_score += normalized.values * weight
        
        return performance_score
    
    def _calculate_dk_fpts(self, df):
        """Calculate DraftKings fantasy points"""
        return (
            df.get('1B', 0) * 3 + df.get('2B', 0) * 5 + df.get('3B', 0) * 8 +
            df.get('HR', 0) * 10 + df.get('RBI', 0) * 2 + df.get('R', 0) * 2 +
            df.get('BB', 0) * 2 + df.get('HBP', 0) * 2 + df.get('SB', 0) * 5
        )
    
    def assign_positions(self, df):
        """Assign realistic positions to players"""
        if 'position' not in df.columns:
            logger.info("Assigning positions to players...")
            
            # Position distribution based on typical MLB roster
            position_weights = {
                'C': 0.08,     # 8% catchers
                '1B': 0.12,    # 12% first basemen
                '2B': 0.12,    # 12% second basemen
                '3B': 0.12,    # 12% third basemen
                'SS': 0.12,    # 12% shortstops
                'OF': 0.44     # 44% outfielders (3 OF positions)
            }
            
            positions = []
            for pos, weight in position_weights.items():
                count = int(len(df) * weight)
                positions.extend([pos] * count)
            
            # Fill remaining with OF
            while len(positions) < len(df):
                positions.append('OF')
            
            # Shuffle and assign
            np.random.shuffle(positions)
            df['position'] = positions[:len(df)]
            
        return df
    
    def generate_base_salaries(self, df):
        """Generate base salaries based on performance"""
        logger.info("Generating base salaries...")
        
        # Calculate performance score
        df['performance_score'] = self.calculate_performance_score(df)
        
        # Assign positions
        df = self.assign_positions(df)
        
        # Calculate percentiles for salary bands
        performance_percentiles = df['performance_score'].rank(pct=True) * 100
        
        salaries = []
        for i, (_, row) in enumerate(df.iterrows()):
            pct = performance_percentiles.iloc[i]
            
            # Determine salary band
            salary_band = 'minimum'
            for band_name, band_info in self.salary_bands.items():
                if pct >= band_info['min_pct']:
                    salary_band = band_name
                    break
            
            # Get salary range for this band
            min_sal, max_sal = self.salary_bands[salary_band]['salary_range']
            
            # Base salary within the band
            base_salary = np.random.uniform(min_sal, max_sal)
            
            # Apply position multiplier
            position = row['position']
            multiplier = self.salary_ranges['position_multipliers'].get(position, 1.0)
            
            # Calculate final salary
            final_salary = base_salary * multiplier
            
            # Ensure within DraftKings bounds
            final_salary = max(self.salary_ranges['min_salary'], 
                             min(self.salary_ranges['max_salary'], final_salary))
            
            # Round to nearest $100 (typical DraftKings pattern)
            final_salary = round(final_salary / 100) * 100
            
            salaries.append(final_salary)
        
        df['salary'] = salaries
        return df
    
    def add_temporal_variance(self, df):
        """Add realistic temporal variance to salaries"""
        logger.info("Adding temporal variance to salaries...")
        
        if 'date' not in df.columns:
            logger.warning("No date column found, skipping temporal variance")
            return df
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Name', 'date'])
        
        # Add variance based on recent performance
        for name in df['Name'].unique():
            player_data = df[df['Name'] == name].copy()
            
            if len(player_data) > 1:
                # Calculate rolling performance
                rolling_perf = player_data['calculated_dk_fpts'].rolling(window=5, min_periods=1).mean()
                
                # Only add variance if we have valid rolling performance data
                if not rolling_perf.isna().all() and rolling_perf.std() > 0:
                    # Add variance based on hot/cold streaks
                    variance_factor = 1 + (rolling_perf - rolling_perf.mean()) / rolling_perf.std() * 0.1
                    variance_factor = np.clip(variance_factor.fillna(1.0), 0.85, 1.15)  # ±15% max variance
                else:
                    # No variance if no valid data
                    variance_factor = pd.Series([1.0] * len(player_data), index=player_data.index)
                
                # Apply variance to salary
                base_salary = player_data['salary'].iloc[0]  # Use first salary as base
                varied_salaries = base_salary * variance_factor
                
                # Ensure salaries don't go below minimum or above maximum
                varied_salaries = np.clip(varied_salaries, 
                                        self.salary_ranges['min_salary'], 
                                        self.salary_ranges['max_salary'])
                
                # Round to nearest $100
                varied_salaries = (varied_salaries / 100).round() * 100
                
                # Update dataframe
                df.loc[df['Name'] == name, 'salary'] = varied_salaries.values
        
        return df
    
    def add_market_trends(self, df):
        """Add market trends (injury news, hot streaks, etc.)"""
        logger.info("Adding market trend adjustments...")
        
        # Random market adjustments for some players
        num_adjustments = int(len(df) * 0.1)  # 10% of players get market adjustments
        adjustment_indices = np.random.choice(len(df), num_adjustments, replace=False)
        
        for idx in adjustment_indices:
            # Random adjustment between -20% and +20%
            adjustment = np.random.uniform(0.8, 1.2)
            df.iloc[idx, df.columns.get_loc('salary')] *= adjustment
            
            # Keep within bounds
            df.iloc[idx, df.columns.get_loc('salary')] = np.clip(
                df.iloc[idx, df.columns.get_loc('salary')],
                self.salary_ranges['min_salary'],
                self.salary_ranges['max_salary']
            )
            
            # Round to nearest $100
            df.iloc[idx, df.columns.get_loc('salary')] = (
                round(df.iloc[idx, df.columns.get_loc('salary')] / 100) * 100
            )
        
        return df
    
    def generate_synthetic_salaries(self, df):
        """Main method to generate synthetic salaries"""
        logger.info("Starting synthetic salary generation...")
        
        # Make a copy to avoid modifying original
        df_copy = df.copy()
        
        # Generate base salaries
        df_copy = self.generate_base_salaries(df_copy)
        
        # Add temporal variance
        df_copy = self.add_temporal_variance(df_copy)
        
        # Add market trends
        df_copy = self.add_market_trends(df_copy)
        
        logger.info("Synthetic salary generation completed!")
        logger.info(f"Salary range: ${df_copy['salary'].min():,.0f} - ${df_copy['salary'].max():,.0f}")
        logger.info(f"Average salary: ${df_copy['salary'].mean():,.0f}")
        
        return df_copy
    
    def validate_salaries(self, df):
        """Validate the generated salaries"""
        logger.info("Validating synthetic salaries...")
        
        # Check salary distribution
        salary_stats = df['salary'].describe()
        logger.info(f"Salary statistics:")
        logger.info(f"  Min: ${salary_stats['min']:,.0f}")
        logger.info(f"  Max: ${salary_stats['max']:,.0f}")
        logger.info(f"  Mean: ${salary_stats['mean']:,.0f}")
        logger.info(f"  Median: ${salary_stats['50%']:,.0f}")
        logger.info(f"  Std: ${salary_stats['std']:,.0f}")
        
        # Check position distribution
        if 'position' in df.columns:
            position_dist = df['position'].value_counts()
            logger.info(f"Position distribution:")
            for pos, count in position_dist.items():
                logger.info(f"  {pos}: {count} ({count/len(df)*100:.1f}%)")
        
        # Check salary vs performance correlation
        if 'calculated_dk_fpts' in df.columns:
            correlation = df['salary'].corr(df['calculated_dk_fpts'])
            logger.info(f"Salary vs Performance correlation: {correlation:.3f}")
        
        # Check for valid lineup construction
        sample_lineups = self.test_lineup_construction(df)
        logger.info(f"Sample lineups constructed: {len(sample_lineups)}")
        
        return True
    
    def test_lineup_construction(self, df, num_tests=10):
        """Test if we can construct valid lineups with the synthetic data"""
        valid_lineups = []
        
        for _ in range(num_tests):
            try:
                # Simple greedy lineup construction
                lineup = self.construct_greedy_lineup(df)
                if lineup is not None:
                    valid_lineups.append(lineup)
            except Exception as e:
                logger.warning(f"Error constructing test lineup: {e}")
        
        return valid_lineups
    
    def construct_greedy_lineup(self, df):
        """Construct a simple greedy lineup for testing"""
        SALARY_CAP = 50000
        POSITIONS = {'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
        
        lineup = []
        total_salary = 0
        
        # Sort by value (points per dollar)
        df_sorted = df.copy()
        df_sorted['value'] = df_sorted['calculated_dk_fpts'] / (df_sorted['salary'] / 1000)
        df_sorted = df_sorted.sort_values('value', ascending=False)
        
        # Fill required positions
        for position, count in POSITIONS.items():
            position_players = df_sorted[df_sorted['position'] == position]
            selected = 0
            
            for _, player in position_players.iterrows():
                if selected >= count:
                    break
                if total_salary + player['salary'] <= SALARY_CAP:
                    lineup.append(player)
                    total_salary += player['salary']
                    selected += 1
        
        return lineup if len(lineup) == sum(POSITIONS.values()) else None
    
    def plot_salary_distribution(self, df, save_path=None):
        """Plot salary distribution for analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Salary histogram
        axes[0, 0].hist(df['salary'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Salary Distribution')
        axes[0, 0].set_xlabel('Salary ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Salary by position
        if 'position' in df.columns:
            position_salary = df.groupby('position')['salary'].mean()
            axes[0, 1].bar(position_salary.index, position_salary.values)
            axes[0, 1].set_title('Average Salary by Position')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Average Salary ($)')
        
        # Salary vs Performance
        if 'calculated_dk_fpts' in df.columns:
            axes[1, 0].scatter(df['calculated_dk_fpts'], df['salary'], alpha=0.5)
            axes[1, 0].set_title('Salary vs Fantasy Points')
            axes[1, 0].set_xlabel('Fantasy Points')
            axes[1, 0].set_ylabel('Salary ($)')
        
        # Value distribution (points per $1000)
        if 'calculated_dk_fpts' in df.columns:
            df['value'] = df['calculated_dk_fpts'] / (df['salary'] / 1000)
            axes[1, 1].hist(df['value'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Value Distribution (Points per $1000)')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()

def main():
    """Main function to demonstrate synthetic salary generation"""
    
    # Configuration
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    OUTPUT_PATH = '4_DATA/data_with_synthetic_salaries.csv'
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(DATA_PATH, low_memory=False)
        logger.info(f"Loaded {len(df)} rows")
        
        # Initialize generator
        generator = SyntheticSalaryGenerator()
        
        # Generate synthetic salaries
        df_with_salaries = generator.generate_synthetic_salaries(df)
        
        # Validate results
        generator.validate_salaries(df_with_salaries)
        
        # Plot distribution
        generator.plot_salary_distribution(df_with_salaries, 
                                         save_path='salary_distribution.png')
        
        # Save results
        df_with_salaries.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Data with synthetic salaries saved to {OUTPUT_PATH}")
        
        # Show sample
        logger.info("\nSample data with synthetic salaries:")
        sample = df_with_salaries[['Name', 'date', 'position', 'salary', 'calculated_dk_fpts']].head(10)
        logger.info(sample.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()
