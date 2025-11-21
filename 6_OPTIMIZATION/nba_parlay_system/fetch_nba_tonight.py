#!/usr/bin/env python3
"""
Fetch NBA player projections for tonight's games (Oct 25, 2025)
"""

import requests
import pandas as pd
import os
from datetime import datetime

def fetch_nba_projections_tonight():
    """Fetch NBA player projections for tonight"""
    
    api_key = os.getenv('SPORTSDATA_API_KEY')
    if not api_key:
        print('❌ Please set SPORTSDATA_API_KEY environment variable')
        return None
    
    # Fetch projections for Oct 25, 2025
    date = '2025-10-25'
    
    url = f"https://api.sportsdata.io/api/nba/fantasy/json/PlayerGameProjectionStatsByDate/{date}"
    headers = {
        'Ocp-Apim-Subscription-Key': api_key
    }
    
    print(f"📡 Fetching NBA projections for {date}...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("❌ No data returned")
            return None
        
        print(f"✅ Retrieved {len(data)} player projections")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename columns to match what the GUI expects
        column_mapping = {
            'PlayerID': 'player_id',
            'Name': 'player_name_proj',
            'Team': 'team_proj',
            'Position': 'position_proj',
            'GameDate': 'game_date',
            'Points': 'projected_points',
            'Rebounds': 'projected_rebounds',
            'Assists': 'projected_assists',
            'Steals': 'projected_steals',
            'BlockedShots': 'projected_blocks',
            'ThreePointersMade': 'projected_three_pointers',
            'FantasyPoints': 'projected_dk_points'
        }
        
        # Only rename columns that exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)
        
        # Add required columns for parlay generation
        if 'projected_points' in df.columns:
            df['points_accuracy_std'] = 0.25
            df['points_hit_mean'] = 0.70
        if 'projected_rebounds' in df.columns:
            df['rebounds_accuracy_std'] = 0.20
            df['rebounds_hit_mean'] = 0.75
        if 'projected_assists' in df.columns:
            df['assists_accuracy_std'] = 0.20
            df['assists_hit_mean'] = 0.75
        if 'projected_steals' in df.columns:
            df['steals_accuracy_std'] = 0.25
            df['steals_hit_mean'] = 0.68
        if 'projected_blocks' in df.columns:
            df['blocks_accuracy_std'] = 0.30
            df['blocks_hit_mean'] = 0.65
        if 'projected_three_pointers' in df.columns:
            df['three_pointers_accuracy_std'] = 0.35
            df['three_pointers_hit_mean'] = 0.60
        
        # Save to CSV
        filename = 'nba_tonight_proj.csv'
        df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename}")
        
        # Show summary
        print(f"\n📊 Summary:")
        print(f"   Total players: {len(df)}")
        if 'team_proj' in df.columns:
            teams = df['team_proj'].unique()
            print(f"   Teams: {len(teams)} - {', '.join(sorted(teams)[:10])}")
        if 'position_proj' in df.columns:
            positions = df['position_proj'].unique()
            print(f"   Positions: {', '.join(sorted(positions))}")
        
        print(f"\n🏀 Top 10 projected scorers:")
        if 'projected_points' in df.columns:
            top_players = df.nlargest(10, 'projected_points')
            for _, player in top_players.iterrows():
                name = player.get('player_name_proj', 'Unknown')
                team = player.get('team_proj', 'N/A')
                pos = player.get('position_proj', 'N/A')
                pts = player['projected_points']
                print(f"   {name} ({pos}, {team}) - {pts:.1f} pts")
        
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response.status_code == 401:
            print("   Invalid API key")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    fetch_nba_projections_tonight()











