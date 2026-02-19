#!/usr/bin/env python3
"""
RL Parlay Data Collector
Collects 3 years of historical NFL projections and actual outcomes for RL training
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

class RLDataCollector:
    def __init__(self, api_key: str):
        self.api = SportsDataNFLAPI(api_key)
        self.data_dir = "rl_training_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_historical_data(self, years: List[int] = [2022, 2023, 2024]) -> Dict:
        """
        Collect 3 years of historical data for RL training
        """
        print("🏈 Starting RL Data Collection for Parlay Training")
        print("=" * 60)
        
        all_data = {
            'projections': [],
            'actuals': [],
            'games': [],
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'years_collected': years,
                'total_weeks': 0,
                'total_games': 0,
                'total_players': 0
            }
        }
        
        for year in years:
            print(f"\n📅 Collecting {year} data...")
            year_data = self._collect_year_data(year)
            
            if year_data:
                all_data['projections'].extend(year_data['projections'])
                all_data['actuals'].extend(year_data['actuals'])
                all_data['games'].extend(year_data['games'])
                
                print(f"✅ {year}: {len(year_data['projections'])} projections, {len(year_data['actuals'])} actuals")
            else:
                print(f"❌ Failed to collect {year} data")
        
        # Update metadata
        all_data['metadata']['total_weeks'] = len(set([p['week'] for p in all_data['projections']]))
        all_data['metadata']['total_games'] = len(set([p['game_id'] for p in all_data['projections'] if 'game_id' in p]))
        all_data['metadata']['total_players'] = len(set([p['player_id'] for p in all_data['projections']]))
        
        # Save collected data
        self._save_collected_data(all_data)
        
        return all_data
    
    def _collect_year_data(self, year: int) -> Optional[Dict]:
        """Collect data for a specific year"""
        try:
            projections = []
            actuals = []
            games = []
            
            # Collect data for each week (1-18 for regular season)
            for week in range(1, 19):
                print(f"  📊 Week {week}...", end=" ")
                
                try:
                    # Get projections for the week
                    week_projections = self.api.get_player_projections_by_week(f"{year}REG", week)
                    if week_projections:
                        for proj in week_projections:
                            projections.append({
                                'year': year,
                                'week': week,
                                'player_id': proj.get('PlayerID'),
                                'player_name': proj.get('Name'),
                                'team': proj.get('Team'),
                                'position': proj.get('Position'),
                                'opponent': proj.get('Opponent'),
                                'game_id': proj.get('GameID'),
                                'projected_dk_points': proj.get('FantasyPointsDraftKings', 0),
                                'projected_passing_yds': proj.get('PassingYards', 0),
                                'projected_rushing_yds': proj.get('RushingYards', 0),
                                'projected_receiving_yds': proj.get('ReceivingYards', 0),
                                'projected_receptions': proj.get('Receptions', 0),
                                'projected_passing_tds': proj.get('PassingTouchdowns', 0),
                                'projected_rushing_tds': proj.get('RushingTouchdowns', 0),
                                'projected_receiving_tds': proj.get('ReceivingTouchdowns', 0),
                                'projected_interceptions': proj.get('Interceptions', 0),
                                'projected_fumbles': proj.get('FumblesLost', 0),
                                'salary': proj.get('Salary', 0),
                                'injury_status': proj.get('InjuryStatus', ''),
                                'weather': proj.get('Weather', ''),
                                'stadium': proj.get('Stadium', ''),
                                'surface': proj.get('Surface', ''),
                                'temperature': proj.get('Temperature', 0),
                                'wind_speed': proj.get('WindSpeed', 0),
                                'humidity': proj.get('Humidity', 0)
                            })
                    
                    # Get actual stats for the week
                    week_actuals = self.api.get_player_game_stats_by_week(f"{year}REG", week)
                    if week_actuals:
                        for actual in week_actuals:
                            actuals.append({
                                'year': year,
                                'week': week,
                                'player_id': actual.get('PlayerID'),
                                'player_name': actual.get('Name'),
                                'team': actual.get('Team'),
                                'position': actual.get('Position'),
                                'opponent': actual.get('Opponent'),
                                'game_id': actual.get('GameID'),
                                'actual_dk_points': actual.get('FantasyPointsDraftKings', 0),
                                'actual_passing_yds': actual.get('PassingYards', 0),
                                'actual_rushing_yds': actual.get('RushingYards', 0),
                                'actual_receiving_yds': actual.get('ReceivingYards', 0),
                                'actual_receptions': actual.get('Receptions', 0),
                                'actual_passing_tds': actual.get('PassingTouchdowns', 0),
                                'actual_rushing_tds': actual.get('RushingTouchdowns', 0),
                                'actual_receiving_tds': actual.get('ReceivingTouchdowns', 0),
                                'actual_interceptions': actual.get('Interceptions', 0),
                                'actual_fumbles': actual.get('FumblesLost', 0),
                                'game_date': actual.get('Date', ''),
                                'home_team': actual.get('HomeTeam', ''),
                                'away_team': actual.get('AwayTeam', ''),
                                'final_score_home': actual.get('HomeScore', 0),
                                'final_score_away': actual.get('AwayScore', 0),
                                'game_total': actual.get('Total', 0),
                                'spread': actual.get('Spread', 0)
                            })
                    
                    # Get game information
                    week_games = self.api.get_games_by_week(f"{year}REG", week)
                    if week_games:
                        for game in week_games:
                            games.append({
                                'year': year,
                                'week': week,
                                'game_id': game.get('GameID'),
                                'home_team': game.get('HomeTeam'),
                                'away_team': game.get('AwayTeam'),
                                'game_date': game.get('Date', ''),
                                'stadium': game.get('Stadium', ''),
                                'surface': game.get('Surface', ''),
                                'weather': game.get('Weather', ''),
                                'temperature': game.get('Temperature', 0),
                                'wind_speed': game.get('WindSpeed', 0),
                                'humidity': game.get('Humidity', 0),
                                'total': game.get('Total', 0),
                                'spread': game.get('Spread', 0),
                                'home_score': game.get('HomeScore', 0),
                                'away_score': game.get('AwayScore', 0)
                            })
                    
                    print("✅")
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"❌ Error in week {week}: {str(e)}")
                    continue
            
            return {
                'projections': projections,
                'actuals': actuals,
                'games': games
            }
            
        except Exception as e:
            print(f"❌ Error collecting {year} data: {str(e)}")
            return None
    
    def _save_collected_data(self, data: Dict):
        """Save collected data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save projections
        proj_df = pd.DataFrame(data['projections'])
        proj_file = os.path.join(self.data_dir, f"projections_{timestamp}.csv")
        proj_df.to_csv(proj_file, index=False)
        print(f"💾 Saved projections: {proj_file}")
        
        # Save actuals
        actual_df = pd.DataFrame(data['actuals'])
        actual_file = os.path.join(self.data_dir, f"actuals_{timestamp}.csv")
        actual_df.to_csv(actual_file, index=False)
        print(f"💾 Saved actuals: {actual_file}")
        
        # Save games
        games_df = pd.DataFrame(data['games'])
        games_file = os.path.join(self.data_dir, f"games_{timestamp}.csv")
        games_df.to_csv(games_file, index=False)
        print(f"💾 Saved games: {games_file}")
        
        # Save metadata
        metadata_file = os.path.join(self.data_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(data['metadata'], f, indent=2)
        print(f"💾 Saved metadata: {metadata_file}")
        
        # Save combined data
        combined_file = os.path.join(self.data_dir, f"combined_data_{timestamp}.json")
        with open(combined_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved combined data: {combined_file}")
    
    def create_training_dataset(self, data_files: List[str]) -> pd.DataFrame:
        """
        Create a training dataset by merging projections with actuals
        """
        print("🔄 Creating RL Training Dataset...")
        
        # Load projections and actuals
        projections_df = pd.read_csv(data_files[0])  # projections file
        actuals_df = pd.read_csv(data_files[1])      # actuals file
        games_df = pd.read_csv(data_files[2])        # games file
        
        # Merge projections with actuals
        merged_df = pd.merge(
            projections_df, 
            actuals_df, 
            on=['year', 'week', 'player_id'], 
            how='inner',
            suffixes=('_proj', '_actual')
        )
        
        # Add game context
        merged_df = pd.merge(
            merged_df,
            games_df,
            on=['year', 'week', 'game_id'],
            how='left'
        )
        
        # Calculate accuracy metrics
        merged_df['dk_points_accuracy'] = merged_df['actual_dk_points'] / (merged_df['projected_dk_points'] + 0.1)
        merged_df['passing_yds_accuracy'] = merged_df['actual_passing_yds'] / (merged_df['projected_passing_yds'] + 0.1)
        merged_df['rushing_yds_accuracy'] = merged_df['actual_rushing_yds'] / (merged_df['projected_rushing_yds'] + 0.1)
        merged_df['receiving_yds_accuracy'] = merged_df['actual_receiving_yds'] / (merged_df['projected_receiving_yds'] + 0.1)
        merged_df['receptions_accuracy'] = merged_df['actual_receptions'] / (merged_df['projected_receptions'] + 0.1)
        
        # Calculate prop hit rates (70% of projection as threshold)
        merged_df['dk_points_hit'] = (merged_df['actual_dk_points'] >= merged_df['projected_dk_points'] * 0.7).astype(int)
        merged_df['passing_yds_hit'] = (merged_df['actual_passing_yds'] >= merged_df['projected_passing_yds'] * 0.7).astype(int)
        merged_df['rushing_yds_hit'] = (merged_df['actual_rushing_yds'] >= merged_df['projected_rushing_yds'] * 0.7).astype(int)
        merged_df['receiving_yds_hit'] = (merged_df['actual_receiving_yds'] >= merged_df['projected_receiving_yds'] * 0.7).astype(int)
        merged_df['receptions_hit'] = (merged_df['actual_receptions'] >= merged_df['projected_receptions'] * 0.7).astype(int)
        
        # Add player consistency metrics
        player_stats = merged_df.groupby('player_id').agg({
            'dk_points_accuracy': ['mean', 'std'],
            'passing_yds_accuracy': ['mean', 'std'],
            'rushing_yds_accuracy': ['mean', 'std'],
            'receiving_yds_accuracy': ['mean', 'std'],
            'receptions_accuracy': ['mean', 'std'],
            'dk_points_hit': 'mean',
            'passing_yds_hit': 'mean',
            'rushing_yds_hit': 'mean',
            'receiving_yds_hit': 'mean',
            'receptions_hit': 'mean'
        }).reset_index()
        
        # Flatten column names
        player_stats.columns = ['player_id'] + [f"{col[0]}_{col[1]}" for col in player_stats.columns[1:]]
        
        # Merge player consistency back
        merged_df = pd.merge(merged_df, player_stats, on='player_id', how='left')
        
        print(f"✅ Created training dataset with {len(merged_df)} records")
        print(f"   Players: {merged_df['player_id'].nunique()}")
        print(f"   Games: {merged_df['game_id'].nunique()}")
        print(f"   Weeks: {merged_df['week'].nunique()}")
        
        return merged_df

def main():
    """Main function to collect historical data"""
    API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
    
    collector = RLDataCollector(API_KEY)
    
    # Collect 3 years of data
    years = [2022, 2023, 2024]
    data = collector.collect_historical_data(years)
    
    if data and len(data['projections']) > 0:
        print(f"\n🎉 Data Collection Complete!")
        print(f"   Total projections: {len(data['projections'])}")
        print(f"   Total actuals: {len(data['actuals'])}")
        print(f"   Total games: {len(data['games'])}")
        print(f"   Years: {data['metadata']['years_collected']}")
        print(f"   Weeks: {data['metadata']['total_weeks']}")
        print(f"   Players: {data['metadata']['total_players']}")
    else:
        print("❌ Data collection failed")

if __name__ == "__main__":
    main()
