#!/usr/bin/env python3
"""
PuLP-based DraftKings Lineup Optimizer

This module uses linear programming to generate optimal DraftKings lineups
following all constraints exactly. It can use real projections and data.
"""

import pandas as pd
import numpy as np
import pulp
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DraftKingsLineupOptimizer:
    """
    PuLP-based optimizer for DraftKings MLB lineups
    
    Uses linear programming to find optimal lineups subject to:
    - Salary cap constraint
    - Position requirements
    - Roster size constraint
    - Unique player constraint
    """
    
    def __init__(self, salary_cap: int = 50000):
        self.salary_cap = salary_cap
        
        # DraftKings MLB position requirements (10-player format with pitchers)
        self.position_requirements = {
            'P': 2,      # Pitcher (2 required)
            'C': 1,      # Catcher
            '1B': 1,     # First Base
            '2B': 1,     # Second Base
            '3B': 1,     # Third Base
            'SS': 1,     # Shortstop
            'OF': 3      # Outfielder (3 required)
        }
        
        self.lineup_size = 10
        
        # Position flexibility for multi-position eligibility
        self.utility_eligible = ['C', '1B', '2B', '3B', 'SS', 'OF']
        
    def prepare_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare player data for optimization
        
        Args:
            df: DataFrame with player data
            
        Returns:
            Cleaned DataFrame ready for optimization
        """
        # Required columns
        required_columns = ['Name', 'salary', 'position']
        
        # Check for projection column
        projection_columns = ['projected_points', 'rolling_30_ppg', 'calculated_dk_fpts']
        projection_col = None
        
        for col in projection_columns:
            if col in df.columns:
                projection_col = col
                break
        
        if projection_col is None:
            raise ValueError(f"No projection column found. Need one of: {projection_columns}")
        
        # Clean data
        clean_df = df[required_columns + [projection_col]].copy()
        clean_df = clean_df.dropna()
        
        # Standardize column names
        clean_df = clean_df.rename(columns={projection_col: 'projected_points'})
        
        # Clean position names
        clean_df['position'] = clean_df['position'].str.upper()
        
        # Map multi-position players to primary position
        position_mapping = {
            'SP': 'P', 'RP': 'P',  # All pitchers mapped to P
            '1B/C': '1B', '1B/3B': '1B', '1B/OF': '1B',
            '2B/3B': '2B', '2B/SS': '2B', '2B/OF': '2B', '2B/C': '2B',
            '3B/SS': '3B', '3B/OF': '3B',
            'SS/OF': 'SS',
            'C/1B': 'C', 'C/OF': 'C'
        }
        
        clean_df['position'] = clean_df['position'].map(position_mapping).fillna(clean_df['position'])
        
        # Filter to valid positions (including pitchers for 10-player format)
        valid_positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
        clean_df = clean_df[clean_df['position'].isin(valid_positions)]
        
        # Remove duplicates (keep best projection)
        clean_df = clean_df.sort_values('projected_points', ascending=False)
        clean_df = clean_df.drop_duplicates(subset=['Name'], keep='first')
        
        # Add player ID
        clean_df['player_id'] = range(len(clean_df))
        
        logger.info(f"Prepared {len(clean_df)} players for optimization")
        logger.info(f"Position distribution: {clean_df['position'].value_counts().to_dict()}")
        
        return clean_df
    
    def optimize_lineup(self, players_df: pd.DataFrame, 
                       objective: str = 'maximize_points',
                       excluded_players: List[str] = None,
                       forced_players: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Optimize lineup using PuLP
        
        Args:
            players_df: DataFrame with player data
            objective: 'maximize_points' or 'maximize_value'
            excluded_players: List of player names to exclude
            forced_players: List of player names to force include
            
        Returns:
            DataFrame with optimal lineup or None if infeasible
        """
        logger.info("Starting lineup optimization...")
        
        # Prepare data
        players = self.prepare_player_data(players_df)
        
        if len(players) < self.lineup_size:
            logger.error(f"Not enough players ({len(players)}) for lineup size ({self.lineup_size})")
            return None
        
        # Create optimization problem
        prob = pulp.LpProblem("DraftKings_Lineup", pulp.LpMaximize)
        
        # Decision variables - binary for each player
        player_vars = {}
        for idx, player in players.iterrows():
            player_vars[player['player_id']] = pulp.LpVariable(
                f"player_{player['player_id']}", 
                cat='Binary'
            )
        
        # Objective function
        if objective == 'maximize_points':
            prob += pulp.lpSum([
                player_vars[player['player_id']] * player['projected_points']
                for idx, player in players.iterrows()
            ])
        elif objective == 'maximize_value':
            prob += pulp.lpSum([
                player_vars[player['player_id']] * (player['projected_points'] / (player['salary'] / 1000))
                for idx, player in players.iterrows()
            ])
        
        # Constraints
        
        # 1. Salary cap constraint
        prob += pulp.lpSum([
            player_vars[player['player_id']] * player['salary']
            for idx, player in players.iterrows()
        ]) <= self.salary_cap
        
        # 2. Roster size constraint
        prob += pulp.lpSum([
            player_vars[player['player_id']]
            for idx, player in players.iterrows()
        ]) == self.lineup_size
        
        # 3. Position constraints
        for position, required_count in self.position_requirements.items():
            # Get players eligible for this position
            eligible_players = players[players['position'] == position]
            
            if len(eligible_players) < required_count:
                logger.warning(f"Not enough {position} players ({len(eligible_players)}) for requirement ({required_count})")
            
            prob += pulp.lpSum([
                player_vars[player['player_id']]
                for idx, player in eligible_players.iterrows()
            ]) >= required_count
        
        # 4. Excluded players constraint
        if excluded_players:
            excluded_ids = players[players['Name'].isin(excluded_players)]['player_id'].tolist()
            for player_id in excluded_ids:
                prob += player_vars[player_id] == 0
        
        # 5. Forced players constraint
        if forced_players:
            forced_ids = players[players['Name'].isin(forced_players)]['player_id'].tolist()
            for player_id in forced_ids:
                prob += player_vars[player_id] == 1
        
        # Solve the problem
        logger.info("Solving optimization problem...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Check solution status
        if prob.status == pulp.LpStatusOptimal:
            logger.info("Optimal solution found!")
            
            # Extract lineup
            selected_players = []
            for idx, player in players.iterrows():
                if player_vars[player['player_id']].varValue == 1:
                    selected_players.append(player)
            
            lineup_df = pd.DataFrame(selected_players)
            
            # Validate lineup
            if self.validate_lineup(lineup_df):
                return lineup_df
            else:
                logger.error("Generated lineup failed validation")
                return None
                
        else:
            logger.error(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
            return None
    
    def validate_lineup(self, lineup_df: pd.DataFrame) -> bool:
        """
        Validate that lineup meets all DraftKings constraints
        
        Args:
            lineup_df: DataFrame with lineup
            
        Returns:
            True if valid, False otherwise
        """
        logger.info("Validating lineup...")
        
        # Check roster size
        if len(lineup_df) != self.lineup_size:
            logger.error(f"Invalid lineup size: {len(lineup_df)} (expected {self.lineup_size})")
            return False
        
        # Check salary cap
        total_salary = lineup_df['salary'].sum()
        if total_salary > self.salary_cap:
            logger.error(f"Salary cap exceeded: ${total_salary:,} > ${self.salary_cap:,}")
            return False
        
        # Check position requirements
        position_counts = lineup_df['position'].value_counts()
        
        # Check each position requirement
        for position, required_count in self.position_requirements.items():
            if position == 'UTIL':
                continue
            
            actual_count = position_counts.get(position, 0)
            if actual_count < required_count:
                logger.error(f"Position {position} requirement not met: {actual_count} < {required_count}")
                return False
        
        # Check that we have exactly the right number of players
        total_required = sum(self.position_requirements.values())
        if len(lineup_df) != total_required:
            logger.error(f"Total players {len(lineup_df)} != required {total_required}")
            return False
        
        # Check for duplicate players
        if len(lineup_df['Name'].unique()) != len(lineup_df):
            logger.error("Duplicate players in lineup")
            return False
        
        logger.info("Lineup validation passed!")
        return True
    
    def generate_multiple_lineups(self, players_df: pd.DataFrame, 
                                 num_lineups: int = 5,
                                 diversity_threshold: float = 0.3) -> List[pd.DataFrame]:
        """
        Generate multiple diverse lineups
        
        Args:
            players_df: DataFrame with player data
            num_lineups: Number of lineups to generate
            diversity_threshold: Minimum difference between lineups (0-1)
            
        Returns:
            List of lineup DataFrames
        """
        logger.info(f"Generating {num_lineups} diverse lineups...")
        
        lineups = []
        excluded_combinations = []
        
        for i in range(num_lineups):
            logger.info(f"Generating lineup {i+1}/{num_lineups}")
            
            # Generate lineup
            lineup = self.optimize_lineup(players_df)
            
            if lineup is None:
                logger.warning(f"Could not generate lineup {i+1}")
                continue
            
            # Check diversity
            if self.is_diverse_lineup(lineup, lineups, diversity_threshold):
                lineups.append(lineup)
                logger.info(f"Lineup {i+1} accepted (diversity check passed)")
            else:
                logger.info(f"Lineup {i+1} rejected (too similar to existing)")
                
                # Add some players to exclusion list to force diversity
                if len(lineups) > 0:
                    last_lineup = lineups[-1]
                    common_players = set(lineup['Name']) & set(last_lineup['Name'])
                    if len(common_players) > 0:
                        excluded_player = list(common_players)[0]
                        lineup_with_exclusion = self.optimize_lineup(
                            players_df, 
                            excluded_players=[excluded_player]
                        )
                        if lineup_with_exclusion is not None:
                            lineups.append(lineup_with_exclusion)
                            logger.info(f"Lineup {i+1} regenerated with exclusion")
        
        logger.info(f"Generated {len(lineups)} diverse lineups")
        return lineups
    
    def is_diverse_lineup(self, lineup: pd.DataFrame, 
                         existing_lineups: List[pd.DataFrame],
                         threshold: float) -> bool:
        """
        Check if lineup is diverse enough from existing lineups
        
        Args:
            lineup: New lineup to check
            existing_lineups: List of existing lineups
            threshold: Minimum difference threshold (0-1)
            
        Returns:
            True if diverse enough, False otherwise
        """
        if not existing_lineups:
            return True
        
        lineup_players = set(lineup['Name'])
        
        for existing_lineup in existing_lineups:
            existing_players = set(existing_lineup['Name'])
            
            # Calculate overlap
            overlap = len(lineup_players & existing_players)
            overlap_ratio = overlap / len(lineup_players)
            
            # If overlap is too high, lineup is not diverse
            if overlap_ratio > (1 - threshold):
                return False
        
        return True
    
    def analyze_lineup(self, lineup_df: pd.DataFrame) -> Dict:
        """
        Analyze lineup and provide detailed statistics
        
        Args:
            lineup_df: DataFrame with lineup
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_salary': lineup_df['salary'].sum(),
            'total_projected_points': lineup_df['projected_points'].sum(),
            'salary_remaining': self.salary_cap - lineup_df['salary'].sum(),
            'avg_salary': lineup_df['salary'].mean(),
            'avg_projected_points': lineup_df['projected_points'].mean(),
            'salary_efficiency': lineup_df['projected_points'].sum() / (lineup_df['salary'].sum() / 1000),
            'position_breakdown': lineup_df['position'].value_counts().to_dict(),
            'players': []
        }
        
        # Add player details
        for _, player in lineup_df.iterrows():
            analysis['players'].append({
                'name': player['Name'],
                'position': player['position'],
                'salary': player['salary'],
                'projected_points': player['projected_points'],
                'value': player['projected_points'] / (player['salary'] / 1000)
            })
        
        return analysis
    
    def print_lineup_summary(self, lineup_df: pd.DataFrame):
        """Print a formatted summary of the lineup"""
        analysis = self.analyze_lineup(lineup_df)
        
        print("\n" + "="*60)
        print("DRAFTKINGS LINEUP SUMMARY")
        print("="*60)
        print(f"Total Salary: ${analysis['total_salary']:,} / ${self.salary_cap:,}")
        print(f"Remaining Salary: ${analysis['salary_remaining']:,}")
        print(f"Total Projected Points: {analysis['total_projected_points']:.1f}")
        print(f"Salary Efficiency: {analysis['salary_efficiency']:.2f} pts per $1K")
        print("\nLineup:")
        print("-" * 60)
        
        for player in analysis['players']:
            print(f"{player['position']:>3} | {player['name']:<20} | "
                  f"${player['salary']:>6,} | {player['projected_points']:>6.1f} pts | "
                  f"{player['value']:>5.2f} val")
        
        print("-" * 60)
        print(f"{'TOT':<3} | {'TEAM TOTAL':<20} | "
              f"${analysis['total_salary']:>6,} | {analysis['total_projected_points']:>6.1f} pts")
        print("="*60)

def main():
    """Main function to demonstrate the PuLP optimizer"""
    
    # Load data with DKEntries salaries
    data_file = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    
    try:
        logger.info("Loading player data...")
        df = pd.read_csv(data_file, low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        
        # Use recent data for demo
        recent_df = df[df['date'] >= '2025-01-01'].copy()
        
        # Get a sample date with good player availability
        date_counts = recent_df.groupby('date').size()
        good_dates = date_counts[date_counts >= 100].index
        
        if len(good_dates) == 0:
            logger.error("No dates with sufficient players found")
            return
        
        test_date = good_dates[-1]  # Use latest good date
        day_players = recent_df[recent_df['date'] == test_date].copy()
        
        logger.info(f"Testing with {len(day_players)} players from {test_date}")
        
        # Initialize optimizer
        optimizer = DraftKingsLineupOptimizer()
        
        # Generate optimal lineup
        logger.info("Generating optimal lineup...")
        optimal_lineup = optimizer.optimize_lineup(day_players, objective='maximize_points')
        
        if optimal_lineup is not None:
            optimizer.print_lineup_summary(optimal_lineup)
            
            # Generate multiple lineups
            logger.info("\nGenerating multiple diverse lineups...")
            multiple_lineups = optimizer.generate_multiple_lineups(day_players, num_lineups=3)
            
            for i, lineup in enumerate(multiple_lineups):
                print(f"\n{'='*20} LINEUP {i+1} {'='*20}")
                optimizer.print_lineup_summary(lineup)
        else:
            logger.error("Could not generate optimal lineup")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
