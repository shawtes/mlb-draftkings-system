#!/usr/bin/env python3
"""
DraftKings Salary Generator using DKEntries.csv as reference

This module generates realistic DraftKings salaries for historical data by:
1. Using DKEntries.csv as the reference for salary/PPG mapping
2. Calculating 30-day rolling PPG averages for each player
3. Mapping historical player performance to current DK salary patterns
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DKEntriesSalaryGenerator:
    """
    Generate realistic DraftKings salaries using DKEntries.csv as reference
    """
    
    def __init__(self, dk_entries_file: str):
        """
        Initialize with DKEntries.csv file
        
        Args:
            dk_entries_file: Path to DKEntries.csv file
        """
        self.dk_entries_file = dk_entries_file
        self.dk_entries = None
        self.salary_mapping = {}
        self.load_dk_entries()
        
    def load_dk_entries(self):
        """Load and process DKEntries.csv file"""
        try:
            logger.info(f"Loading DKEntries data from {self.dk_entries_file}")
            
            # Read the file with custom parsing to handle the complex format
            with open(self.dk_entries_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find lines that contain player data (have the player info pattern)
            player_lines = []
            header_found = False
            
            for line in lines:
                # Look for lines with position data (SP, RP, C, 1B, 2B, 3B, SS, OF)
                if any(pos in line for pos in ['SP,', 'RP,', 'C,', '1B,', '2B,', '3B,', 'SS,', 'OF,']):
                    # Split by comma and extract relevant columns
                    parts = line.strip().split(',')
                    if len(parts) >= 24:  # Ensure we have enough columns
                        # Extract the player data from the end of the line
                        # Format: Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame
                        try:
                            position = parts[15]  # Position column
                            name = parts[17]      # Name column
                            salary = parts[20]    # Salary column
                            avg_ppg = parts[23]   # AvgPointsPerGame column
                            
                            # Clean the data
                            position = position.strip()
                            name = name.strip()
                            salary = salary.strip()
                            avg_ppg = avg_ppg.strip()
                            
                            # Validate numeric values
                            if salary.isdigit() and avg_ppg.replace('.', '').isdigit():
                                player_lines.append({
                                    'Position': position,
                                    'Name': name,
                                    'Salary': int(salary),
                                    'AvgPointsPerGame': float(avg_ppg)
                                })
                        except (IndexError, ValueError):
                            continue
            
            # Create DataFrame from parsed data
            self.dk_entries = pd.DataFrame(player_lines)
            
            if len(self.dk_entries) == 0:
                raise ValueError("No player data found in DKEntries.csv")
            
            # Clean and prepare data
            self.dk_entries['PPG'] = pd.to_numeric(self.dk_entries['AvgPointsPerGame'], errors='coerce')
            self.dk_entries['Salary'] = pd.to_numeric(self.dk_entries['Salary'], errors='coerce')
            
            # Remove rows with missing data
            self.dk_entries = self.dk_entries.dropna(subset=['PPG', 'Salary'])
            
            logger.info(f"Loaded {len(self.dk_entries)} entries from DKEntries.csv")
            logger.info(f"Salary range: ${self.dk_entries['Salary'].min():,} - ${self.dk_entries['Salary'].max():,}")
            logger.info(f"PPG range: {self.dk_entries['PPG'].min():.1f} - {self.dk_entries['PPG'].max():.1f}")
            
            # Create salary mapping by position
            self.create_salary_mapping()
            
        except Exception as e:
            logger.error(f"Error loading DKEntries.csv: {e}")
            raise
    
    def create_salary_mapping(self):
        """Create salary mapping from PPG to salary by position"""
        logger.info("Creating salary mapping from DKEntries data...")
        
        # Group by position and create mappings
        for position in self.dk_entries['Position'].unique():
            pos_data = self.dk_entries[self.dk_entries['Position'] == position].copy()
            
            # Sort by PPG for mapping
            pos_data = pos_data.sort_values('PPG')
            
            # Create mapping dictionary
            self.salary_mapping[position] = {
                'ppg_values': pos_data['PPG'].values,
                'salary_values': pos_data['Salary'].values,
                'min_ppg': pos_data['PPG'].min(),
                'max_ppg': pos_data['PPG'].max(),
                'min_salary': pos_data['Salary'].min(),
                'max_salary': pos_data['Salary'].max()
            }
            
            logger.info(f"  {position}: {len(pos_data)} players, "
                       f"PPG {pos_data['PPG'].min():.1f}-{pos_data['PPG'].max():.1f}, "
                       f"Salary ${pos_data['Salary'].min():,}-${pos_data['Salary'].max():,}")
    
    def map_ppg_to_salary(self, ppg: float, position: str) -> int:
        """
        Map PPG to salary using DKEntries data
        
        Args:
            ppg: Points per game
            position: Player position
            
        Returns:
            Estimated salary
        """
        if position not in self.salary_mapping:
            # If position not found, use closest position
            available_positions = list(self.salary_mapping.keys())
            if 'OF' in available_positions:
                position = 'OF'  # Most common position
            else:
                position = available_positions[0]
        
        mapping = self.salary_mapping[position]
        
        # Handle edge cases
        if ppg <= mapping['min_ppg']:
            return int(mapping['min_salary'])
        elif ppg >= mapping['max_ppg']:
            return int(mapping['max_salary'])
        
        # Find closest PPG value and interpolate
        ppg_array = mapping['ppg_values']
        salary_array = mapping['salary_values']
        
        # Find insertion point
        idx = np.searchsorted(ppg_array, ppg)
        
        if idx == 0:
            return int(salary_array[0])
        elif idx == len(ppg_array):
            return int(salary_array[-1])
        else:
            # Linear interpolation between two closest points
            ppg_low, ppg_high = ppg_array[idx-1], ppg_array[idx]
            salary_low, salary_high = salary_array[idx-1], salary_array[idx]
            
            # Calculate interpolation factor
            factor = (ppg - ppg_low) / (ppg_high - ppg_low)
            estimated_salary = salary_low + factor * (salary_high - salary_low)
            
            return int(estimated_salary)
    
    def calculate_30_day_rolling_ppg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 30-day rolling PPG averages for each player
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with 30-day rolling PPG column
        """
        logger.info("Calculating 30-day rolling PPG averages...")
        
        df_copy = df.copy()
        
        # Ensure date column is datetime
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Calculate DK fantasy points if not present
        if 'calculated_dk_fpts' not in df_copy.columns:
            df_copy['calculated_dk_fpts'] = self._calculate_dk_fpts(df_copy)
        
        # Sort by player and date
        df_copy = df_copy.sort_values(['Name', 'date'])
        
        # Calculate 30-day rolling PPG
        df_copy['rolling_30_ppg'] = df_copy.groupby('Name')['calculated_dk_fpts'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean().shift(1)
        )
        
        # Fill NaN values with current game points (for first games)
        df_copy['rolling_30_ppg'] = df_copy['rolling_30_ppg'].fillna(df_copy['calculated_dk_fpts'])
        
        logger.info(f"Calculated rolling PPG for {len(df_copy)} records")
        logger.info(f"Rolling PPG range: {df_copy['rolling_30_ppg'].min():.1f} - {df_copy['rolling_30_ppg'].max():.1f}")
        
        return df_copy
    
    def _calculate_dk_fpts(self, df: pd.DataFrame) -> pd.Series:
        """Calculate DraftKings fantasy points"""
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
    
    def assign_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign positions to players based on DKEntries position distribution
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with position assignments
        """
        df_copy = df.copy()
        
        if 'position' not in df_copy.columns:
            # Get position distribution from DKEntries
            position_counts = self.dk_entries['Position'].value_counts()
            position_probs = position_counts / position_counts.sum()
            
            # Assign positions randomly based on DK distribution
            positions = np.random.choice(
                position_probs.index,
                size=len(df_copy),
                p=position_probs.values
            )
            
            df_copy['position'] = positions
            
            logger.info("Assigned positions based on DKEntries distribution:")
            for pos, count in df_copy['position'].value_counts().items():
                logger.info(f"  {pos}: {count} players ({count/len(df_copy)*100:.1f}%)")
        
        return df_copy
    
    def generate_historical_salaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate realistic historical salaries using DKEntries mapping
        
        Args:
            df: DataFrame with historical player data
            
        Returns:
            DataFrame with synthetic salaries
        """
        logger.info("Generating historical salaries using DKEntries reference...")
        
        # Calculate 30-day rolling PPG
        df_with_ppg = self.calculate_30_day_rolling_ppg(df)
        
        # Assign positions if not present
        df_with_positions = self.assign_positions(df_with_ppg)
        
        # Generate salaries based on rolling PPG
        salaries = []
        for _, row in df_with_positions.iterrows():
            salary = self.map_ppg_to_salary(row['rolling_30_ppg'], row['position'])
            salaries.append(salary)
        
        df_with_positions['salary'] = salaries
        
        # Add some realistic variance (±5%)
        variance = np.random.normal(0, 0.05, len(df_with_positions))
        df_with_positions['salary'] = df_with_positions['salary'] * (1 + variance)
        
        # Round to nearest $100 and ensure within DK bounds
        df_with_positions['salary'] = (df_with_positions['salary'] / 100).round() * 100
        df_with_positions['salary'] = df_with_positions['salary'].clip(3000, 13000)
        
        logger.info(f"Generated salaries for {len(df_with_positions)} records")
        logger.info(f"Salary range: ${df_with_positions['salary'].min():,} - ${df_with_positions['salary'].max():,}")
        logger.info(f"Average salary: ${df_with_positions['salary'].mean():,.0f}")
        
        return df_with_positions
    
    def validate_salaries(self, df: pd.DataFrame) -> Dict:
        """Validate generated salaries"""
        validation_results = {
            'total_records': len(df),
            'salary_range': {
                'min': df['salary'].min(),
                'max': df['salary'].max(),
                'mean': df['salary'].mean(),
                'std': df['salary'].std()
            },
            'position_salary_avg': df.groupby('position')['salary'].mean().to_dict(),
            'ppg_salary_correlation': df['salary'].corr(df['rolling_30_ppg']),
            'within_dk_bounds': (df['salary'] >= 3000).all() and (df['salary'] <= 13000).all(),
            'dk_entries_comparison': self._compare_to_dk_entries(df)
        }
        
        return validation_results
    
    def _compare_to_dk_entries(self, df: pd.DataFrame) -> Dict:
        """Compare generated salaries to DKEntries patterns"""
        comparison = {}
        
        for position in df['position'].unique():
            if position in self.salary_mapping:
                pos_data = df[df['position'] == position]
                dk_data = self.dk_entries[self.dk_entries['Position'] == position]
                
                comparison[position] = {
                    'generated_avg_salary': pos_data['salary'].mean(),
                    'dk_avg_salary': dk_data['Salary'].mean(),
                    'generated_avg_ppg': pos_data['rolling_30_ppg'].mean(),
                    'dk_avg_ppg': dk_data['PPG'].mean()
                }
        
        return comparison

def create_dk_entries_salary_data(input_file: str, dk_entries_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Main function to generate salaries using DKEntries.csv reference
    
    Args:
        input_file: Path to historical data CSV
        dk_entries_file: Path to DKEntries.csv
        output_file: Optional output file path
        
    Returns:
        DataFrame with generated salaries
    """
    logger.info("Creating DraftKings salaries using DKEntries reference...")
    
    # Load historical data
    df = pd.read_csv(input_file, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize salary generator
    salary_gen = DKEntriesSalaryGenerator(dk_entries_file)
    
    # Generate salaries
    df_with_salaries = salary_gen.generate_historical_salaries(df)
    
    # Validate results
    validation = salary_gen.validate_salaries(df_with_salaries)
    
    # Log validation results
    logger.info("\n=== Salary Generation Validation ===")
    logger.info(f"Total records: {validation['total_records']:,}")
    logger.info(f"Salary range: ${validation['salary_range']['min']:,} - ${validation['salary_range']['max']:,}")
    logger.info(f"Average salary: ${validation['salary_range']['mean']:,.0f}")
    logger.info(f"PPG-Salary correlation: {validation['ppg_salary_correlation']:.3f}")
    logger.info(f"Within DK bounds: {validation['within_dk_bounds']}")
    
    logger.info("\n=== Position Salary Averages ===")
    for pos, avg_salary in validation['position_salary_avg'].items():
        logger.info(f"{pos}: ${avg_salary:,.0f}")
    
    logger.info("\n=== Comparison to DKEntries ===")
    for pos, comparison in validation['dk_entries_comparison'].items():
        logger.info(f"{pos}:")
        logger.info(f"  Generated avg salary: ${comparison['generated_avg_salary']:,.0f}")
        logger.info(f"  DK avg salary: ${comparison['dk_avg_salary']:,.0f}")
        logger.info(f"  Generated avg PPG: {comparison['generated_avg_ppg']:.1f}")
        logger.info(f"  DK avg PPG: {comparison['dk_avg_ppg']:.1f}")
    
    # Save results
    if output_file:
        df_with_salaries.to_csv(output_file, index=False)
        logger.info(f"\nData with salaries saved to {output_file}")
    
    return df_with_salaries

def main():
    """Main function to demonstrate DKEntries salary generation"""
    
    # Configuration
    INPUT_FILE = 'C:\\Users\\smtes\\Downloads\\coinbase_ml_trader\\c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM\4_DATA\\merged_merged_fangraphs_logs_with_fpts_merged_fangraphs_data.csv'
    DK_ENTRIES_FILE = 'C:/Users/smtes/Downloads/DKEntries.csv'
    OUTPUT_FILE = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    
    # Generate salaries
    df_with_salaries = create_dk_entries_salary_data(INPUT_FILE, DK_ENTRIES_FILE, OUTPUT_FILE)
    
    # Show sample results
    logger.info("\n=== Sample Results ===")
    sample_df = df_with_salaries[['Name', 'date', 'calculated_dk_fpts', 'rolling_30_ppg', 'position', 'salary']].head(10)
    for _, row in sample_df.iterrows():
        logger.info(f"{row['Name']} ({row['position']}): "
                   f"Game: {row['calculated_dk_fpts']:.1f} FPTS, "
                   f"30-day PPG: {row['rolling_30_ppg']:.1f}, "
                   f"Salary: ${row['salary']:,}")
    
    # Performance analysis
    logger.info("\n=== Performance Analysis ===")
    high_performers = df_with_salaries[df_with_salaries['rolling_30_ppg'] >= 20]
    if len(high_performers) > 0:
        logger.info(f"Players with 20+ PPG: {len(high_performers)}")
        logger.info(f"Average salary for 20+ PPG: ${high_performers['salary'].mean():,.0f}")
    
    medium_performers = df_with_salaries[
        (df_with_salaries['rolling_30_ppg'] >= 10) & 
        (df_with_salaries['rolling_30_ppg'] < 20)
    ]
    if len(medium_performers) > 0:
        logger.info(f"Players with 10-20 PPG: {len(medium_performers)}")
        logger.info(f"Average salary for 10-20 PPG: ${medium_performers['salary'].mean():,.0f}")
    
    low_performers = df_with_salaries[df_with_salaries['rolling_30_ppg'] < 10]
    if len(low_performers) > 0:
        logger.info(f"Players with <10 PPG: {len(low_performers)}")
        logger.info(f"Average salary for <10 PPG: ${low_performers['salary'].mean():,.0f}")

if __name__ == "__main__":
    main()
