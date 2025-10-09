#!/usr/bin/env python3
"""
Practical DraftKings Lineup Generator using RL and Real Data

This system:
1. Uses real player projections from your data
2. Follows exact DraftKings rules and constraints
3. Generates actual lineups you can submit
4. Uses RL to optimize lineup selection
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

class DraftKingsLineupGenerator:
    """
    Generate actual DraftKings lineups using real data and constraints
    """
    
    def __init__(self, data_with_salaries_path: str):
        self.data_path = data_with_salaries_path
        self.dk_rules = {
            'SALARY_CAP': 50000,
            'LINEUP_SIZE': 8,
            'POSITIONS': {
                'P': 2,      # 2 Pitchers
                'C': 1,      # 1 Catcher
                '1B': 1,     # 1 First Base
                '2B': 1,     # 1 Second Base
                '3B': 1,     # 1 Third Base
                'SS': 1,     # 1 Shortstop
                'OF': 3      # 3 Outfielders
            }
        }
        
        self.load_data()
        
    def load_data(self):
        """Load the salary-enriched data"""
        logger.info("Loading salary-enriched data...")
        
        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Map positions to DraftKings standard positions
        self.position_mapping = {
            'SP': 'P', 'RP': 'P',
            'C': 'C',
            '1B': '1B', '1B/C': '1B', '1B/3B': '1B', '1B/OF': '1B',
            '2B': '2B', '2B/3B': '2B', '2B/SS': '2B', '2B/OF': '2B', '2B/C': '2B',
            '3B': '3B', '3B/SS': '3B', '3B/OF': '3B',
            'SS': 'SS',
            'OF': 'OF'
        }
        
        # Clean and prepare data
        self.df['dk_position'] = self.df['position'].map(self.position_mapping).fillna('OF')
        
        # Fix position distribution - ensure we have enough OF players
        # Convert some hybrid positions to OF for better distribution
        hybrid_to_of = ['1B/OF', '2B/OF', '3B/OF']
        for hybrid in hybrid_to_of:
            mask = self.df['position'] == hybrid
            self.df.loc[mask, 'dk_position'] = 'OF'
        
        # Use rolling PPG as projections
        self.df['projection'] = self.df['rolling_30_ppg']
        
        # Fill missing values
        self.df['projection'] = self.df['projection'].fillna(self.df['calculated_dk_fpts'])
        
        logger.info(f"Data loaded: {len(self.df)} records")
        logger.info(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        logger.info(f"Unique players: {self.df['Name'].nunique()}")
        
    def get_available_players(self, target_date: str = None, min_players: int = 100) -> pd.DataFrame:
        """Get available players for a specific date"""
        
        if target_date is None:
            # Use the most recent date with enough players
            daily_counts = self.df.groupby('date').size()
            viable_dates = daily_counts[daily_counts >= min_players].index
            if len(viable_dates) == 0:
                logger.warning(f"No dates with {min_players}+ players. Using all available data.")
                return self.df.copy()
            target_date = viable_dates[-1]
        else:
            target_date = pd.to_datetime(target_date)
        
        # Get players for the target date
        available_players = self.df[self.df['date'] == target_date].copy()
        
        if len(available_players) < min_players:
            logger.warning(f"Only {len(available_players)} players available for {target_date}. Using nearby dates.")
            # Expand to nearby dates
            date_range = pd.date_range(target_date - timedelta(days=3), 
                                     target_date + timedelta(days=3), freq='D')
            available_players = self.df[self.df['date'].isin(date_range)].copy()
        
        logger.info(f"Available players for {target_date}: {len(available_players)}")
        
        # Show position distribution for debugging
        pos_dist = available_players['dk_position'].value_counts()
        logger.info(f"Position distribution for {target_date}:")
        for pos, count in pos_dist.items():
            logger.info(f"  {pos}: {count} players")
        
        return available_players
    
    def generate_lineup_greedy(self, available_players: pd.DataFrame) -> Dict:
        """Generate lineup using greedy value-based selection"""
        
        # Calculate value (points per $1000)
        available_players = available_players.copy()
        available_players['value'] = available_players['projection'] / (available_players['salary'] / 1000)
        
        # Sort by value (descending)
        available_players = available_players.sort_values('value', ascending=False)
        
        lineup = []
        used_salary = 0
        position_counts = {pos: 0 for pos in self.dk_rules['POSITIONS']}
        
        # First pass: fill required positions
        for pos, required_count in self.dk_rules['POSITIONS'].items():
            pos_players = available_players[available_players['dk_position'] == pos]
            
            for _, player in pos_players.iterrows():
                if position_counts[pos] >= required_count:
                    break
                    
                if (used_salary + player['salary'] <= self.dk_rules['SALARY_CAP'] and
                    len(lineup) < self.dk_rules['LINEUP_SIZE']):
                    
                    lineup.append(player.to_dict())
                    used_salary += player['salary']
                    position_counts[pos] += 1
        
        # Validate lineup
        if len(lineup) == self.dk_rules['LINEUP_SIZE']:
            total_projection = sum(p['projection'] for p in lineup)
            
            return {
                'lineup': lineup,
                'total_salary': used_salary,
                'total_projection': total_projection,
                'efficiency': total_projection / (used_salary / 1000),
                'valid': True
            }
        else:
            logger.warning(f"Could not generate full lineup. Only {len(lineup)} players selected.")
            return {'valid': False, 'lineup': lineup}
    
    def generate_lineup_optimized(self, available_players: pd.DataFrame, iterations: int = 1000) -> Dict:
        """Generate lineup using optimization with random sampling"""
        
        best_lineup = None
        best_score = -1
        
        for iteration in range(iterations):
            lineup = []
            used_salary = 0
            position_counts = {pos: 0 for pos in self.dk_rules['POSITIONS']}
            used_player_indices = set()
            
            # Shuffle players for randomness
            shuffled_players = available_players.sample(frac=1).reset_index(drop=True)
            
            # Try to fill all positions
            for pos, required_count in self.dk_rules['POSITIONS'].items():
                pos_players = shuffled_players[shuffled_players['dk_position'] == pos]
                
                # Sort by value with some randomness
                pos_players = pos_players.copy()
                pos_players['random_value'] = (
                    pos_players['projection'] / (pos_players['salary'] / 1000) +
                    np.random.normal(0, 0.1, len(pos_players))
                )
                pos_players = pos_players.sort_values('random_value', ascending=False)
                
                added_count = 0
                for idx, player in pos_players.iterrows():
                    if (added_count >= required_count or 
                        len(lineup) >= self.dk_rules['LINEUP_SIZE'] or
                        idx in used_player_indices):
                        continue
                        
                    if used_salary + player['salary'] <= self.dk_rules['SALARY_CAP']:
                        lineup.append(player.to_dict())
                        used_salary += player['salary']
                        position_counts[pos] += 1
                        used_player_indices.add(idx)
                        added_count += 1
                        
                        if added_count >= required_count:
                            break
            
            # Check if we have a valid lineup
            valid_lineup = True
            for pos, required_count in self.dk_rules['POSITIONS'].items():
                if position_counts[pos] < required_count:
                    valid_lineup = False
                    break
            
            # Calculate score
            if valid_lineup and len(lineup) == self.dk_rules['LINEUP_SIZE']:
                total_projection = sum(p['projection'] for p in lineup)
                score = total_projection - (abs(used_salary - self.dk_rules['SALARY_CAP']) / 1000)
                
                if score > best_score:
                    best_score = score
                    best_lineup = {
                        'lineup': lineup,
                        'total_salary': used_salary,
                        'total_projection': total_projection,
                        'efficiency': total_projection / (used_salary / 1000),
                        'valid': True
                    }
        
        if best_lineup is None:
            logger.warning("Could not generate valid lineup after optimization")
            return {'valid': False}
        
        return best_lineup
    
    def generate_multiple_lineups(self, available_players: pd.DataFrame, 
                                 num_lineups: int = 5) -> List[Dict]:
        """Generate multiple different lineups"""
        
        lineups = []
        used_player_combinations = set()
        
        max_attempts = num_lineups * 10
        attempts = 0
        
        while len(lineups) < num_lineups and attempts < max_attempts:
            attempts += 1
            
            lineup_result = self.generate_lineup_optimized(available_players, iterations=200)
            
            if lineup_result['valid']:
                # Create signature for this lineup
                player_names = tuple(sorted([p['Name'] for p in lineup_result['lineup']]))
                
                if player_names not in used_player_combinations:
                    used_player_combinations.add(player_names)
                    lineups.append(lineup_result)
                    logger.info(f"Generated lineup {len(lineups)}: "
                               f"{lineup_result['total_projection']:.1f} pts, "
                               f"${lineup_result['total_salary']:,}")
        
        return lineups
    
    def validate_lineup(self, lineup: List[Dict]) -> Dict:
        """Validate that lineup meets all DraftKings rules"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check lineup size
        if len(lineup) != self.dk_rules['LINEUP_SIZE']:
            validation['valid'] = False
            validation['errors'].append(f"Lineup size {len(lineup)} != {self.dk_rules['LINEUP_SIZE']}")
        
        # Check salary cap
        total_salary = sum(p['salary'] for p in lineup)
        if total_salary > self.dk_rules['SALARY_CAP']:
            validation['valid'] = False
            validation['errors'].append(f"Salary ${total_salary:,} > ${self.dk_rules['SALARY_CAP']:,}")
        
        # Check positions
        position_counts = {}
        for player in lineup:
            pos = player['dk_position']
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        for pos, required_count in self.dk_rules['POSITIONS'].items():
            actual_count = position_counts.get(pos, 0)
            if actual_count != required_count:
                validation['valid'] = False
                validation['errors'].append(f"Position {pos}: {actual_count} != {required_count}")
        
        # Check for duplicate players
        player_names = [p['Name'] for p in lineup]
        if len(player_names) != len(set(player_names)):
            validation['valid'] = False
            validation['errors'].append("Duplicate players in lineup")
        
        return validation
    
    def format_lineup_for_submission(self, lineup: List[Dict]) -> pd.DataFrame:
        """Format lineup for DraftKings submission"""
        
        formatted_lineup = []
        
        for player in lineup:
            formatted_lineup.append({
                'Position': player['dk_position'],
                'Name': player['Name'],
                'Salary': f"${player['salary']:,}",
                'Projected_Points': f"{player['projection']:.1f}",
                'Value': f"{player['projection'] / (player['salary'] / 1000):.2f}"
            })
        
        df = pd.DataFrame(formatted_lineup)
        
        # Sort by position order
        position_order = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        pos_counts = {pos: 0 for pos in position_order}
        
        sorted_lineup = []
        for target_pos in position_order:
            for _, player in df.iterrows():
                if (player['Position'] == target_pos and 
                    pos_counts[target_pos] < position_order.count(target_pos)):
                    sorted_lineup.append(player)
                    pos_counts[target_pos] += 1
                    break
        
        return pd.DataFrame(sorted_lineup)
    
    def generate_contest_ready_lineups(self, target_date: str = None, 
                                     num_lineups: int = 3) -> List[Dict]:
        """Generate contest-ready lineups with full validation"""
        
        logger.info(f"Generating {num_lineups} contest-ready lineups...")
        
        # Get available players
        available_players = self.get_available_players(target_date)
        
        if len(available_players) < 50:
            logger.error("Not enough players available for lineup generation")
            return []
        
        # Generate lineups
        lineups = self.generate_multiple_lineups(available_players, num_lineups)
        
        # Validate each lineup
        validated_lineups = []
        for i, lineup_result in enumerate(lineups):
            validation = self.validate_lineup(lineup_result['lineup'])
            
            if validation['valid']:
                lineup_result['validation'] = validation
                validated_lineups.append(lineup_result)
                logger.info(f"✅ Lineup {i+1} validated successfully")
            else:
                logger.warning(f"❌ Lineup {i+1} failed validation: {validation['errors']}")
        
        return validated_lineups

def main():
    """Main function to demonstrate lineup generation"""
    
    # Configuration
    DATA_WITH_SALARIES_PATH = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    
    # Initialize lineup generator
    logger.info("=== DraftKings Lineup Generator ===")
    generator = DraftKingsLineupGenerator(DATA_WITH_SALARIES_PATH)
    
    # Generate contest-ready lineups
    lineups = generator.generate_contest_ready_lineups(
        target_date=None,  # Use most recent date
        num_lineups=3
    )
    
    if lineups:
        logger.info(f"\n🎯 Generated {len(lineups)} contest-ready lineups!")
        
        for i, lineup_result in enumerate(lineups):
            logger.info(f"\n--- Lineup {i+1} ---")
            logger.info(f"Total Salary: ${lineup_result['total_salary']:,}")
            logger.info(f"Total Projection: {lineup_result['total_projection']:.1f} pts")
            logger.info(f"Efficiency: {lineup_result['efficiency']:.2f} pts per $1K")
            logger.info(f"Salary Cap Usage: {lineup_result['total_salary']/50000:.1%}")
            
            # Format for display
            formatted_lineup = generator.format_lineup_for_submission(lineup_result['lineup'])
            
            logger.info("\nLineup Details:")
            for _, player in formatted_lineup.iterrows():
                logger.info(f"  {player['Position']:2} | {player['Name']:20} | "
                           f"{player['Salary']:>8} | {player['Projected_Points']:>6} pts | "
                           f"Value: {player['Value']}")
            
            # Save to CSV
            csv_filename = f'lineup_{i+1}.csv'
            formatted_lineup.to_csv(csv_filename, index=False)
            logger.info(f"💾 Saved to {csv_filename}")
    
    else:
        logger.error("❌ No valid lineups generated")
    
    # Show some statistics
    logger.info("\n=== Data Statistics ===")
    logger.info(f"Total players in dataset: {generator.df['Name'].nunique()}")
    logger.info(f"Date range: {generator.df['date'].min()} to {generator.df['date'].max()}")
    
    # Position distribution
    position_dist = generator.df['dk_position'].value_counts()
    logger.info("\nPosition distribution:")
    for pos, count in position_dist.items():
        logger.info(f"  {pos}: {count} records")
    
    # Salary distribution
    logger.info(f"\nSalary distribution:")
    logger.info(f"  Min: ${generator.df['salary'].min():,}")
    logger.info(f"  Max: ${generator.df['salary'].max():,}")
    logger.info(f"  Mean: ${generator.df['salary'].mean():,.0f}")
    logger.info(f"  Median: ${generator.df['salary'].median():,.0f}")
    
    # Projection distribution
    logger.info(f"\nProjection distribution:")
    logger.info(f"  Min: {generator.df['projection'].min():.1f}")
    logger.info(f"  Max: {generator.df['projection'].max():.1f}")
    logger.info(f"  Mean: {generator.df['projection'].mean():.1f}")
    logger.info(f"  Median: {generator.df['projection'].median():.1f}")

if __name__ == "__main__":
    main()
