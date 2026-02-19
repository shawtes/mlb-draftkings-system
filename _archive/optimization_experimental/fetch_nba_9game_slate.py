#!/usr/bin/env python3
"""
Fetch NBA 9-Game Slate Data from SportsData.io API
Uses the existing nba_sportsdata_fetcher.py module
"""

import pandas as pd
from datetime import datetime
from nba_sportsdata_fetcher import NBADataFetcher

# API Key from the existing fetcher
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"

def main():
    """Main execution"""
    print(f"\n{'='*60}")
    print(f"🏀 NBA 9-Game Slate Fetcher")
    print(f"{'='*60}\n")
    
    # Use the existing NBADataFetcher
    fetcher = NBADataFetcher(API_KEY)
    
    # Try today's date (or specify: "2025-10-27")
    date = "2025-OCT-27"  # October 27th
    
    print(f"📅 Fetching data for: {date}")
    
    # Get projections (already has this data)
    proj_df = fetcher.get_daily_projections(date)
    
    if proj_df.empty:
        print("❌ No projection data found")
        return
    
    print(f"✅ Found {len(proj_df)} players")
    
    # Get DFS slate info to find game count
    slate_info = fetcher.get_dfs_slate_info(date)
    
    if isinstance(slate_info, list):
        print(f"\n📊 Found {len(slate_info)} slate(s)")
        for slate in slate_info:
            print(f"   - {slate.get('Operator', 'Unknown')}: {slate.get('DfsSlateID', 'Unknown')}")
    elif isinstance(slate_info, dict):
        print(f"📊 Slate info: {slate_info}")
    
    # Process and save
    # The projections already have FantasyPointsDraftKings
    if 'FantasyPointsDraftKings' in proj_df.columns:
        proj_df['Predicted_DK_Points'] = proj_df['FantasyPointsDraftKings']
    elif 'Predicted_DK_Points' in proj_df.columns:
        pass  # Already has it
    else:
        print("⚠️ No fantasy points found")
    
    # Need to add real salaries - fetch from DFS slate
    try:
        # Try to get salaries from DFS slates
        if isinstance(slate_info, list):
            for slate in slate_info:
                if slate.get('Operator') == 'DraftKings':
                    players = slate.get('DfsSlatePlayers', [])
                    if players:
                        # Create salary lookup
                        salary_lookup = {}
                        for p in players:
                            name = p.get('Name', '')
                            salary = p.get('OperatorSalary', 0)
                            if salary > 0:
                                salary_lookup[name] = salary
                        
                        # Apply salaries
                        proj_df['Salary'] = proj_df.apply(
                            lambda row: salary_lookup.get(row['Name'], 5000),
                            axis=1
                        )
                        print(f"✅ Applied salaries to {len(salary_lookup)} players")
                        break
    except Exception as e:
        print(f"⚠️ Could not get real salaries: {e}")
        proj_df['Salary'] = 5000
    
    # Count games
    if 'GameID' in proj_df.columns:
        num_games = proj_df['GameID'].nunique()
        print(f"\n📊 Games in slate: {num_games}")
        
        if num_games >= 9:
            print("✅ This is a 9+ game slate!")
        else:
            print(f"⚠️ This is only a {num_games}-game slate")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"nba_9game_slate_{timestamp}.csv"
    
    # Select only needed columns
    cols_to_keep = ['Name', 'Position', 'Team']
    
    if 'Salary' in proj_df.columns:
        cols_to_keep.append('Salary')
    if 'Predicted_DK_Points' in proj_df.columns:
        cols_to_keep.append('Predicted_DK_Points')
    if 'GameID' in proj_df.columns:
        cols_to_keep.append('GameID')
    if 'Opponent' in proj_df.columns:
        cols_to_keep.append('Opponent')
    if 'Game' in proj_df.columns:
        cols_to_keep.append('Game')
    
    final_df = proj_df[cols_to_keep].copy()
    final_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 Players: {len(final_df)}")
    
    if not final_df.empty:
        print(f"\n📋 Sample (first 5 rows):")
        display_cols = ['Name', 'Position', 'Team']
        if 'Salary' in final_df.columns:
            display_cols.append('Salary')
        if 'Predicted_DK_Points' in final_df.columns:
            display_cols.append('Predicted_DK_Points')
        print(final_df[display_cols].head())
        print(f"\n💡 Load this file into the genetic optimizer!")


if __name__ == "__main__":
    main()
