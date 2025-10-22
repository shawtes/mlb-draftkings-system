"""
NFL Probability Engine - Calculate probabilities based on historical player stats
"""
import numpy as np
import pandas as pd
import pickle
import os
from scipy import stats

class NFLProbabilityEngine:
    def __init__(self):
        self.historical_data = self.load_historical_data()
        self.player_stats = {}
        self.cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nfl_historical_cache"
        
        if self.historical_data is not None:
            print(f"✅ Loaded historical data: {len(self.historical_data)} player records")
            self.build_player_stats()
        else:
            print("⚠️  No historical data found, using projection-based probabilities")
    
    def load_historical_data(self):
        """Load historical NFL data from cache"""
        cache_path = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nfl_historical_cache/historical_3years.pkl"
        
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            print(f"⚠️  Error loading historical data: {e}")
        
        return None
    
    def build_player_stats(self):
        """Build player statistics from historical data"""
        if self.historical_data is None:
            return
        
        try:
            # Group by player and calculate stats
            for player_name in self.historical_data['Name'].unique():
                player_data = self.historical_data[self.historical_data['Name'] == player_name]
                
                self.player_stats[player_name] = {
                    'passing_yards_mean': player_data['PassingYards'].mean(),
                    'passing_yards_std': player_data['PassingYards'].std(),
                    'passing_tds_mean': player_data['PassingTouchdowns'].mean(),
                    'passing_tds_std': player_data['PassingTouchdowns'].std(),
                    'rushing_yards_mean': player_data['RushingYards'].mean(),
                    'rushing_yards_std': player_data['RushingYards'].std(),
                    'rushing_tds_mean': player_data['RushingTouchdowns'].mean(),
                    'rushing_tds_std': player_data['RushingTouchdowns'].std(),
                    'receiving_yards_mean': player_data['ReceivingYards'].mean(),
                    'receiving_yards_std': player_data['ReceivingYards'].std(),
                    'receptions_mean': player_data['Receptions'].mean(),
                    'receptions_std': player_data['Receptions'].std(),
                    'receiving_tds_mean': player_data['ReceivingTouchdowns'].mean(),
                    'receiving_tds_std': player_data['ReceivingTouchdowns'].std(),
                    'games_played': len(player_data)
                }
            
            print(f"✅ Built stats for {len(self.player_stats)} players")
        except Exception as e:
            print(f"⚠️  Error building player stats: {e}")
    
    def calculate_probability(self, player_data, stat_name, line):
        """Calculate probability that player goes OVER the line
        
        Uses PROJECTION as mean, but HISTORICAL std dev for realistic spread
        """
        try:
            # Try to get player name
            player_name = player_data.get('Name', '')
            
            # Map stat names to historical columns
            stat_mapping = {
                'PassingYards': 'passing_yards',
                'PassingTouchdowns': 'passing_tds',
                'RushingYards': 'rushing_yards',
                'RushingTouchdowns': 'rushing_tds',
                'ReceivingYards': 'receiving_yards',
                'Receptions': 'receptions',
                'ReceivingTouchdowns': 'receiving_tds'
            }
            
            # ALWAYS use projection as the mean (this is today's expected value)
            mean = player_data.get(stat_name, 0)
            
            # But get std dev from historical data for realistic variance
            if player_name in self.player_stats and stat_name in stat_mapping:
                hist_key = stat_mapping[stat_name]
                std = self.player_stats[player_name].get(f'{hist_key}_std', 0)
                
                # If historical std is NaN or 0, estimate from projection
                if pd.isna(std) or std == 0:
                    # Use 30% of projection as std dev for realistic spread
                    std = mean * 0.30 if mean > 0 else 1
            else:
                # No historical data - estimate std dev as 30% of projection
                std = mean * 0.30 if mean > 0 else 1
            
            # Calculate probability using normal distribution
            if std > 0 and mean > 0:
                z_score = (line - mean) / std
                probability = 1 - stats.norm.cdf(z_score)
            else:
                # If no stats, use 50/50
                probability = 0.5
            
            # Clamp between 1% and 99%
            return max(0.01, min(0.99, probability))
        
        except Exception as e:
            print(f"⚠️  Error calculating probability for {player_data.get('Name', 'Unknown')}: {e}")
            return 0.5
    
    def calculate_combined_probability(self, projection=None, line=None, prop_type=None, 
                                      position=None, multiplier=1.0, player_consistency=None,
                                      probabilities=None, **kwargs):
        """Calculate probability for a prop bet
        
        Can be called two ways:
        1. With projection/line/prop_type for single prop calculation
        2. With probabilities list for parlay calculation
        """
        # If called with probabilities list (for parlays)
        if probabilities is not None:
            combined = 1.0
            for prob in probabilities:
                if isinstance(prob, (int, float)):
                    combined *= prob
            return max(0.01, min(0.99, combined))
        
        # If called with projection/line (for single props)
        if projection is not None and line is not None:
            try:
                # Convert to float if needed
                projection_val = float(projection) if projection is not None else 0
                line_val = float(line) if line is not None else 0
                
                # Handle multiplier - might be string like '20x' or '3x'
                # NOTE: multiplier is for PAYOUT, not for projection
                # We DON'T multiply the projection by it
                if multiplier is None:
                    multiplier_val = 1.0
                elif isinstance(multiplier, str):
                    # Extract number from string like '20x' -> 20.0
                    multiplier_val = float(multiplier.replace('x', '').strip()) if 'x' in multiplier else 1.0
                else:
                    multiplier_val = float(multiplier)
                
                # Use projection as mean (NOT multiplied by payout multiplier!)
                mean = projection_val
                
                # Estimate std dev as 30% of mean for realistic spread  
                std = mean * 0.30 if mean > 0 else 1
                
                # Adjust std based on consistency if available
                if player_consistency is not None and player_consistency > 0:
                    # Lower consistency = higher std dev (more variable)
                    # Consistency is typically 0-1, where 1 is most consistent
                    consistency_factor = (2 - player_consistency) if player_consistency < 1 else 1
                    std *= consistency_factor
                
                # Calculate probability using normal distribution
                if std > 0 and mean > 0:
                    z_score = (line_val - mean) / std
                    probability = 1 - stats.norm.cdf(z_score)
                else:
                    probability = 0.5
                
                return {
                    'probability': max(0.01, min(0.99, probability)),
                    'mean': mean,
                    'std': std,
                    'z_score': (line_val - mean) / std if std > 0 else 0
                }
            except (ValueError, TypeError) as e:
                # Return fallback if conversion fails
                return {'probability': 0.5, 'mean': 0, 'std': 0, 'z_score': 0}
        
        # Fallback
        return {'probability': 0.5, 'mean': 0, 'std': 0, 'z_score': 0}
    
    def calculate_expected_value(self, probability, payout):
        """Calculate expected value"""
        return (probability * payout) - ((1 - probability) * 1.0)
    
    def get_player_history_summary(self, player_name):
        """Get summary statistics for a player"""
        if player_name in self.player_stats:
            return self.player_stats[player_name]
        return None
