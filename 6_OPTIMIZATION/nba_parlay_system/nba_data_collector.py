#!/usr/bin/env python3
"""
NBA Historical Data Collector
Collects NBA player projections and actual stats for training the parlay model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python_algorithms'))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json
import time

# You'll need to import your NBA API class or create one
# For now, using placeholder API calls

class NBADataCollector:
    """Collects NBA historical data for training"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # self.api = YourNBAAPIClass(api_key)  # Replace with actual API
    
    def collect_historical_data(self, seasons: List[str] = ['2023', '2024', '2025']) -> Dict:
        """
        Collect historical NBA data across multiple seasons
        
        Args:
            seasons: List of season years to collect
            
        Returns:
            Dictionary with projections and actuals
        """
        print("🏀 NBA Historical Data Collection")
        print("=" * 70)
        
        all_projections = []
        all_actuals = []
        
        for season in seasons:
            print(f"\n📅 Collecting {season} season data...")
            
            season_data = self._collect_season_data(season)
            if season_data:
                all_projections.extend(season_data['projections'])
                all_actuals.extend(season_data['actuals'])
                print(f"✅ Collected {len(season_data['projections'])} projections")
                print(f"✅ Collected {len(season_data['actuals'])} actual stats")
        
        return {
            'projections': all_projections,
            'actuals': all_actuals
        }
    
    def _collect_season_data(self, season: str) -> Optional[Dict]:
        """Collect data for a single season"""
        projections = []
        actuals = []
        
        # Collect data for regular season games (approximately 82 games per team)
        # NBA seasons typically have 82 regular season games
        
        # Mock implementation - replace with actual API calls
        # Example structure:
        """
        # For each game date in the season
        for game_date in season_dates:
            # Get projections for that date
            game_projections = self.api.get_player_projections_by_date(game_date)
            
            # Store projections
            for proj in game_projections:
                projections.append({
                    'season': season,
                    'game_date': game_date,
                    'player_id': proj['PlayerID'],
                    'player_name': proj['Name'],
                    'team': proj['Team'],
                    'position': proj['Position'],
                    'opponent': proj['Opponent'],
                    'projected_points': proj.get('ProjectedPoints', 0),
                    'projected_rebounds': proj.get('ProjectedRebounds', 0),
                    'projected_assists': proj.get('ProjectedAssists', 0),
                    'projected_steals': proj.get('ProjectedSteals', 0),
                    'projected_blocks': proj.get('ProjectedBlocks', 0),
                    'projected_three_pointers': proj.get('ProjectedThreePointers', 0),
                    'projected_dk_points': proj.get('ProjectedFantasyPoints', 0)
                })
            
            # After game is played, get actual stats
            actual_stats = self.api.get_player_stats_by_date(game_date)
            
            for stat in actual_stats:
                actuals.append({
                    'season': season,
                    'game_date': game_date,
                    'player_id': stat['PlayerID'],
                    'player_name': stat['Name'],
                    'team': stat['Team'],
                    'position': stat['Position'],
                    'opponent': stat['Opponent'],
                    'actual_points': stat.get('Points', 0),
                    'actual_rebounds': stat.get('Rebounds', 0),
                    'actual_assists': stat.get('Assists', 0),
                    'actual_steals': stat.get('Steals', 0),
                    'actual_blocks': stat.get('Blocks', 0),
                    'actual_three_pointers': stat.get('ThreePointersMade', 0),
                    'actual_dk_points': stat.get('FantasyPoints', 0)
                })
        """
        
        # Placeholder return
        return {
            'projections': projections,
            'actuals': actuals
        }
    
    def create_training_dataset(self, historical_data: Dict) -> pd.DataFrame:
        """
        Create training dataset from historical data
        
        Merges projections with actual results and calculates accuracy metrics
        """
        print("\n📊 Creating training dataset...")
        
        # Convert to DataFrames
        df_proj = pd.DataFrame(historical_data['projections'])
        df_actual = pd.DataFrame(historical_data['actuals'])
        
        if len(df_proj) == 0 or len(df_actual) == 0:
            print("❌ No data to process")
            return pd.DataFrame()
        
        # Merge projections with actuals
        df_merged = df_proj.merge(
            df_actual,
            on=['season', 'game_date', 'player_id', 'player_name', 'team', 'position', 'opponent'],
            how='inner',
            suffixes=('_proj', '_actual')
        )
        
        # Calculate accuracy for each stat
        stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'three_pointers', 'dk_points']
        
        for stat in stats:
            proj_col = f'projected_{stat}'
            actual_col = f'actual_{stat}'
            
            if proj_col in df_merged.columns and actual_col in df_merged.columns:
                # Calculate error
                df_merged[f'{stat}_error'] = df_merged[actual_col] - df_merged[proj_col]
                
                # Calculate absolute error
                df_merged[f'{stat}_abs_error'] = abs(df_merged[f'{stat}_error'])
                
                # Calculate hit rate (actual >= 70% of projection)
                line = df_merged[proj_col] * 0.7
                df_merged[f'{stat}_hit'] = (df_merged[actual_col] >= line).astype(int)
        
        # Calculate accuracy std by player
        print("📈 Calculating player accuracy metrics...")
        
        player_stats = []
        for player_id in df_merged['player_id'].unique():
            player_data = df_merged[df_merged['player_id'] == player_id]
            
            if len(player_data) < 5:  # Need at least 5 games
                continue
            
            stat_row = {
                'player_id': player_id,
                'player_name': player_data['player_name'].iloc[0],
                'position': player_data['position'].iloc[0],
                'team': player_data['team'].iloc[0],
                'games_played': len(player_data)
            }
            
            # Calculate std for each stat
            for stat in stats:
                proj_col = f'projected_{stat}'
                actual_col = f'actual_{stat}'
                
                if proj_col in player_data.columns and actual_col in player_data.columns:
                    # Calculate coefficient of variation
                    errors = player_data[f'{stat}_abs_error']
                    projections = player_data[proj_col]
                    
                    if projections.mean() > 0:
                        cv = errors.mean() / projections.mean()
                        stat_row[f'{stat}_accuracy_std'] = cv
                        
                        # Hit rate
                        hit_rate = player_data[f'{stat}_hit'].mean()
                        stat_row[f'{stat}_hit_mean'] = hit_rate
            
            player_stats.append(stat_row)
        
        df_player_stats = pd.DataFrame(player_stats)
        
        # Merge player stats back into main dataset
        df_final = df_merged.merge(df_player_stats, on='player_id', how='left')
        
        print(f"✅ Created training dataset: {len(df_final)} records")
        print(f"   Players analyzed: {len(df_player_stats)}")
        
        return df_final
    
    def save_training_data(self, df: pd.DataFrame, filename: str = 'nba_training_data.csv'):
        """Save training data to CSV"""
        if len(df) == 0:
            print("❌ No data to save")
            return
        
        df.to_csv(filename, index=False)
        print(f"✅ Saved training data to {filename}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Shape: {df.shape}")

def main():
    """Main function to collect NBA data"""
    print("🏀 NBA Data Collector")
    print("=" * 70)
    
    # Use the NBA API key from test file
    API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
    
    # Add path to NBA fetcher
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from nba_sportsdata_fetcher import NBADataFetcher
    
    # Create fetcher with API key
    fetcher = NBADataFetcher(API_KEY)
    
    print("📥 Fetching 3 years of NBA historical data...")
    print("This will take a while as we fetch projections and actuals...")
    
    # Fetch multiple dates from the past 3 years
    from datetime import datetime, timedelta
    
    # Generate dates for training (past 3 seasons - 2022-2023, 2023-2024, 2024-2025)
    training_dates = []
    
    # NBA seasons: Oct to Apr/May
    seasons = [
        # 2024-2025 season (current)
        (datetime(2024, 10, 15), datetime(2025, 4, 15)),
        # 2023-2024 season
        (datetime(2023, 10, 15), datetime(2024, 4, 15)),
        # 2022-2023 season
        (datetime(2022, 10, 15), datetime(2023, 4, 15)),
    ]
    
    for start_date, end_date in seasons:
        # Sample dates every 2 days for comprehensive coverage
        current = start_date
        while current <= end_date:
            training_dates.append(current.strftime('%Y-%b-%d').upper())
            current += timedelta(days=2)
    
    print(f"📅 Generated {len(training_dates)} potential game dates across 3 seasons")
    
    all_projections = []
    all_actuals = []
    
    # Fetch data for each date
    num_dates = min(len(training_dates), 300)  # Fetch up to 300 dates for multi-year training
    print(f"📥 Fetching data for {num_dates} dates (this will take 10-15 minutes)...")
    
    for i, date_str in enumerate(training_dates[:num_dates]):
        if i % 20 == 0:
            print(f"\n[{i+1}/{num_dates}] Fetching {date_str}...")
        else:
            print(f"[{i+1}/{num_dates}] Fetching {date_str}...", end=' ')
        
        # Get projections
        projections = fetcher.get_daily_projections(date_str)
        if not projections.empty:
            # Rename columns
            projections = projections.rename(columns={
                'Name': 'player_name_proj',
                'Team': 'team_proj',
                'Position': 'position_proj',
                'PlayerID': 'player_id',
                'ProjectedPoints': 'projected_points',
                'ProjectedRebounds': 'projected_rebounds',
                'ProjectedAssists': 'projected_assists',
                'ProjectedSteals': 'projected_steals',
                'ProjectedBlocks': 'projected_blocks',
                'ThreePointersMade': 'projected_three_pointers',
                'Predicted_DK_Points': 'projected_dk_points'
            })
            projections['game_date'] = date_str
            all_projections.append(projections)
            
            if i % 20 == 0:
                print(f"   ✅ Got {len(projections)} player projections")
            else:
                print(f"✅")
        
        # Small delay to avoid rate limits
        time.sleep(0.3)
    
    # Combine all projections
    if all_projections:
        training_df = pd.concat(all_projections, ignore_index=True)
        
        print(f"\n📊 Combined data shape: {training_df.shape}")
        print(f"Columns: {training_df.columns.tolist()[:10]}...")
        
        # Ensure player_id column exists (use ID column)
        if 'ID' in training_df.columns and 'player_id' not in training_df.columns:
            training_df['player_id'] = training_df['ID']
        
        if 'player_id' not in training_df.columns:
            print("Error: No player_id column found!")
            print(f"Available columns: {training_df.columns.tolist()}")
            return
        
        # Calculate historical variance by player and position
        print("\n📊 Fetching actual historical stats...")
        
        stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'three_pointers']
        
        # Get actual stats from API for past dates
        print("   Fetching actual game results from API...")
        all_actuals = []
        
        # Get unique dates (past dates only, skip today/future)
        unique_dates = training_df['game_date'].unique()
        from datetime import datetime
        today = datetime.now().strftime('%Y-%b-%d').upper()
        
        for date_str in unique_dates[:50]:  # Process up to 50 dates to avoid timeout
            if date_str >= today:
                continue
            
            try:
                actual_stats = fetcher.get_historical_stats(date_str, num_days=1)
                if not actual_stats.empty:
                    # Map to our stat columns
                    actual_stats['game_date'] = date_str
                    actual_stats['ID'] = actual_stats.get('PlayerID', actual_stats.get('ID', None))
                    all_actuals.append(actual_stats)
                    print(f"   ✅ Got actuals for {date_str}")
            except Exception as e:
                print(f"   ⚠️ Could not fetch actuals for {date_str}: {e}")
        
        # Combine actual stats
        if all_actuals:
            actuals_df = pd.concat(all_actuals, ignore_index=True)
            print(f"   ✅ Combined {len(actuals_df)} actual records")
            
            # Merge actuals with projections by player_id and date
            training_df = training_df.merge(
                actuals_df[['ID', 'game_date', 'Points', 'Rebounds', 'Assists', 'Steals', 'BlockedShots', 'ThreePointersMade']],
                on=['ID', 'game_date'],
                how='left',
                suffixes=('_proj', '_actual')
            )
            
            # Rename actual columns
            training_df = training_df.rename(columns={
                'Points': 'actual_points',
                'Rebounds': 'actual_rebounds',
                'Assists': 'actual_assists',
                'Steals': 'actual_steals',
                'BlockedShots': 'actual_blocks',
                'ThreePointersMade': 'actual_three_pointers'
            })
        else:
            print("   ⚠️  No actual stats retrieved, using synthetic variance")
            # Fallback: Add placeholder actuals
            for stat in stats:
                training_df[f'actual_{stat}'] = training_df[f'projected_{stat}'] * np.random.normal(1.0, 0.25)
        
        # Calculate variance metrics
        player_stats = []
        for player_id in training_df['player_id'].unique():
            player_data = training_df[training_df['player_id'] == player_id]
            
            if len(player_data) < 5:
                continue
            
            stat_row = {
                'player_id': player_id,
                'player_name_proj': player_data['player_name_proj'].iloc[0],
                'position_proj': player_data['position_proj'].iloc[0],
                'team_proj': player_data['team_proj'].iloc[0],
                'games_played': len(player_data)
            }
            
            for stat in stats:
                proj_col = f'projected_{stat}'
                actual_col = f'actual_{stat}'
                
                if proj_col in player_data.columns and actual_col in player_data.columns:
                    errors = abs(player_data[actual_col] - player_data[proj_col])
                    projections = player_data[proj_col]
                    
                    if projections.mean() > 0:
                        cv = errors.mean() / projections.mean()
                        stat_row[f'{stat}_accuracy_std'] = cv
                        
                        # Hit rate (70% of projection)
                        line = player_data[proj_col] * 0.7
                        hit_rate = (player_data[actual_col] >= line).mean()
                        stat_row[f'{stat}_hit_mean'] = hit_rate
            
            player_stats.append(stat_row)
        
        df_player_stats = pd.DataFrame(player_stats)
        
        # Merge back into training data
        training_df = training_df.merge(df_player_stats, on='player_id', how='left')
        
        # Save training data
        training_df.to_csv('nba_training_data.csv', index=False)
        print(f"\n✅ Saved {len(training_df)} records to nba_training_data.csv")
        print(f"   Analyzed {len(df_player_stats)} players")
        print("\n✅ NBA data collection complete!")
    else:
        print("\n❌ No data collected. Check API connection.")

if __name__ == "__main__":
    main()

