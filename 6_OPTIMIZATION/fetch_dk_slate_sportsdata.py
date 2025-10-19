#!/usr/bin/env python3
"""
Fetch NFL DraftKings slate data from sportsdata.io API
Creates player pools for both Cash Games and GPP contests
"""

import requests
import pandas as pd
import json
from datetime import datetime

# sportsdata.io API configuration
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
BASE_URL = "https://api.sportsdata.io/api/nfl/fantasy/json"

def fetch_dfs_slates(api_key, target_date=None):
    """Fetch available DFS slates"""
    # Format: 2024REG for regular season
    season = "2024REG"
    week = "7"
    
    url = f"{BASE_URL}/DfsSlatesByWeek/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    print(f"🔍 Fetching DFS slates for {season} Week {week}...")
    print(f"   URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        slates = response.json()
        
        print(f"✅ Found {len(slates)} slates")
        
        # Filter for DraftKings slates
        dk_slates = [s for s in slates if s.get('Operator') == 'DraftKings']
        
        print(f"📊 DraftKings slates:")
        for slate in dk_slates:
            print(f"   - {slate.get('SlateID')}: {slate.get('Name')} ({slate.get('NumberOfGames')} games)")
        
        return dk_slates
    
    except Exception as e:
        print(f"❌ Error fetching slates: {e}")
        return []

def fetch_players_by_slate(api_key, slate_id):
    """Fetch player projections for a specific slate"""
    url = f"{BASE_URL}/projections/json/DfsSlatesBySlateID/{{slate_id}}"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    print(f"\n🔍 Fetching players for slate {slate_id}...")
    
    try:
        response = requests.get(url.format(slate_id=slate_id), headers=headers)
        response.raise_for_status()
        slate_data = response.json()
        
        # Extract player projections
        players = slate_data.get('DfsSlateGames', [])
        
        player_list = []
        for game in players:
            game_info = f"{game.get('AwayTeam')}@{game.get('HomeTeam')}"
            
            # Get players from this game
            for player in game.get('DfsSlatePlayers', []):
                player_data = {
                    'Name': player.get('Name'),
                    'Position': player.get('Position'),
                    'Team': player.get('Team'),
                    'Salary': player.get('Salary'),
                    'DraftKingsPlayerID': player.get('PlayerID'),  # DK Player ID
                    'FantasyDataPlayerID': player.get('FantasyDataPlayerID'),  # sportsdata.io ID
                    'Projected_Points': player.get('ProjectedPoints'),
                    'Game': game_info,
                    'Opponent': player.get('Opponent')
                }
                player_list.append(player_data)
        
        print(f"✅ Found {len(player_list)} players")
        return player_list
    
    except Exception as e:
        print(f"❌ Error fetching players: {e}")
        return []

def fetch_projections(api_key, week):
    """Fetch detailed projections"""
    season = "2025REG"
    
    url = f"{BASE_URL}/PlayerGameProjectionStatsByWeek/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    
    print(f"\n🔍 Fetching detailed projections for {season} Week {week}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        projections = response.json()
        
        print(f"✅ Found {len(projections)} player projections")
        
        # Convert to dictionary for easier lookup
        proj_dict = {}
        for proj in projections:
            name = proj.get('Name')
            proj_dict[name] = {
                'Passing_Yards': proj.get('PassingYards', 0),
                'Passing_TDs': proj.get('PassingTouchdowns', 0),
                'Rushing_Yards': proj.get('RushingYards', 0),
                'Rushing_TDs': proj.get('RushingTouchdowns', 0),
                'Receiving_Yards': proj.get('ReceivingYards', 0),
                'Receiving_TDs': proj.get('ReceivingTouchdowns', 0),
                'Receptions': proj.get('Receptions', 0),
                'FantasyPoints': proj.get('FantasyPointsDraftKings', 0),
                'Ceiling': proj.get('FantasyPointsDraftKings', 0) * 1.2,  # Estimate
                'Floor': proj.get('FantasyPointsDraftKings', 0) * 0.7,  # Estimate
            }
        
        return proj_dict
    
    except Exception as e:
        print(f"❌ Error fetching projections: {e}")
        return {}

def create_player_pools(players_df, projections_dict, output_prefix):
    """Create Cash and GPP optimized player pools"""
    
    # Merge projections
    for idx, row in players_df.iterrows():
        name = row['Name']
        if name in projections_dict:
            proj = projections_dict[name]
            players_df.at[idx, 'Predicted_DK_Points'] = proj['FantasyPoints']
            players_df.at[idx, 'Ceiling'] = proj['Ceiling']
            players_df.at[idx, 'Floor'] = proj['Floor']
            players_df.at[idx, 'PassingYards'] = proj['Passing_Yards']
            players_df.at[idx, 'PassingTouchdowns'] = proj['Passing_TDs']
            players_df.at[idx, 'RushingYards'] = proj['Rushing_Yards']
            players_df.at[idx, 'RushingTouchdowns'] = proj['Rushing_TDs']
            players_df.at[idx, 'ReceivingYards'] = proj['Receiving_Yards']
            players_df.at[idx, 'ReceivingTouchdowns'] = proj['Receiving_TDs']
            players_df.at[idx, 'Receptions'] = proj['Receptions']
    
    # Calculate Value
    players_df['Value'] = players_df['Predicted_DK_Points'] / (players_df['Salary'] / 1000)
    
    # CASH GAME: Focus on floor, consistency
    cash_df = players_df.copy()
    cash_df['Cash_Score'] = (cash_df['Floor'] * 0.7) + (cash_df['Predicted_DK_Points'] * 0.3)
    cash_df = cash_df.sort_values('Cash_Score', ascending=False)
    
    # GPP: Focus on ceiling, upside
    gpp_df = players_df.copy()
    gpp_df['GPP_Score'] = (gpp_df['Ceiling'] * 0.6) + (gpp_df['Predicted_DK_Points'] * 0.4)
    gpp_df = gpp_df.sort_values('GPP_Score', ascending=False)
    
    # Save files
    cash_file = f"{output_prefix}_CASH.csv"
    gpp_file = f"{output_prefix}_GPP.csv"
    
    cash_df.to_csv(cash_file, index=False)
    gpp_df.to_csv(gpp_file, index=False)
    
    print(f"\n💾 Saved Cash Game pool: {cash_file}")
    print(f"💾 Saved GPP pool: {gpp_file}")
    
    return cash_df, gpp_df

def main():
    print("=" * 70)
    print("NFL DraftKings Slate Data Fetcher (sportsdata.io)")
    print("=" * 70)
    
    # Check if API key is set
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  Please set your sportsdata.io API key in the script!")
        print("Get your key at: https://sportsdata.io/")
        
        # Try to read from environment variable
        import os
        api_key = os.environ.get('SPORTSDATA_API_KEY')
        if not api_key:
            print("\nOr set environment variable: SPORTSDATA_API_KEY")
            return
    else:
        api_key = API_KEY
    
    # Fetch available slates
    slates = fetch_dfs_slates(api_key)
    
    if not slates:
        print("\n❌ No DraftKings slates found for today")
        return
    
    # Let user choose slate
    print("\n📋 Available slates:")
    for i, slate in enumerate(slates):
        print(f"   {i+1}. {slate.get('Name')} ({slate.get('NumberOfGames')} games) - SlateID: {slate.get('SlateID')}")
    
    # Auto-select first slate (or user can modify)
    selected_slate = slates[0]
    slate_id = selected_slate.get('SlateID')
    slate_name = selected_slate.get('Name', 'slate')
    
    print(f"\n🎯 Using slate: {slate_name} (ID: {slate_id})")
    
    # Fetch players
    players = fetch_players_by_slate(api_key, slate_id)
    
    if not players:
        print("\n❌ No players found for this slate")
        return
    
    # Convert to DataFrame
    players_df = pd.DataFrame(players)
    
    # Get current week (estimate)
    week = 7  # Adjust as needed
    
    # Fetch detailed projections
    projections = fetch_projections(api_key, week)
    
    # Create player pools
    output_prefix = f"nfl_week{week}_{slate_name.replace(' ', '_').replace('[', '').replace(']', '')}"
    
    create_player_pools(players_df, projections, output_prefix)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Slate: {slate_name}")
    print(f"   Total Players: {len(players_df)}")
    print(f"   Teams: {', '.join(sorted(players_df['Team'].unique()))}")
    print(f"   Positions: {', '.join(players_df['Position'].unique())}")
    
    print("\n✅ Files created successfully!")
    print("\n📝 Next steps:")
    print("   1. Load DKEntries.csv in optimizer")
    print(f"   2. Load {output_prefix}_CASH.csv for cash games")
    print(f"   3. Load {output_prefix}_GPP.csv for tournaments")

if __name__ == "__main__":
    main()

