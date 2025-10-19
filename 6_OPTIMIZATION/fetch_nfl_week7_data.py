#!/usr/bin/env python3
"""
Fetch NFL Week 7 data from sportsdata.io and create player pools for DraftKings
"""

import requests
import pandas as pd
from datetime import datetime

API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
BASE_URL = "https://api.sportsdata.io/api/nfl/fantasy/json"

def get_dk_slate_games():
    """Extract games from DKEntries.csv"""
    import re
    games = set()
    try:
        with open('DKEntries.csv', 'r') as f:
            content = f.read()
            # Find all matchups like "MIA@CLE"
            matches = re.findall(r'([A-Z]{2,3})@([A-Z]{2,3})', content)
            for away, home in matches:
                games.add((away, home))
        
        print(f"📋 DraftKings Slate Games:")
        for away, home in sorted(games):
            print(f"   {away}@{home}")
        print()
        
        return games
    except Exception as e:
        print(f"⚠️  Could not read DKEntries.csv: {e}")
        # Default to the 6 games we know
        return {('MIA', 'CLE'), ('NO', 'CHI'), ('LV', 'KC'), 
                ('PHI', 'MIN'), ('CAR', 'NYJ'), ('NE', 'TEN')}

def fetch_week7_projections():
    """Fetch Week 7 player projections"""
    season = "2025REG"  # Updated to 2025
    week = "7"
    
    url = f"{BASE_URL}/PlayerGameProjectionStatsByWeek/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"🔍 Fetching Week {week} projections...")
    print(f"   URL: {url}\n")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        projections = response.json()
        
        print(f"✅ Found {len(projections)} player projections")
        return projections
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def fetch_defense_projections():
    """Fetch Week 7 defense projections"""
    season = "2025REG"  # Updated to 2025
    week = "7"
    
    url = f"{BASE_URL}/FantasyDefenseProjectionsByGame/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"\n🔍 Fetching Week {week} defense projections...")
    print(f"   URL: {url}\n")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        defenses = response.json()
        
        print(f"✅ Found {len(defenses)} defense projections")
        return defenses
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def process_player_projections(projections, dk_player_ids):
    """Process player projections and merge with DK player IDs"""
    
    players_list = []
    
    for proj in projections:
        # Extract player info
        player_data = {
            'Name': proj.get('Name'),
            'Position': proj.get('Position'),
            'Team': proj.get('Team'),
            'Opponent': proj.get('Opponent'),
            
            # Stats
            'PassingYards': proj.get('PassingYards', 0),
            'PassingTouchdowns': proj.get('PassingTouchdowns', 0),
            'PassingInterceptions': proj.get('PassingInterceptions', 0),
            'RushingYards': proj.get('RushingYards', 0),
            'RushingTouchdowns': proj.get('RushingTouchdowns', 0),
            'ReceivingYards': proj.get('ReceivingYards', 0),
            'ReceivingTouchdowns': proj.get('ReceivingTouchdowns', 0),
            'Receptions': proj.get('Receptions', 0),
            'ReceivingTargets': proj.get('ReceivingTargets', 0),
            
            # Fantasy points
            'Predicted_DK_Points': proj.get('FantasyPointsDraftKings', 0),
            'FantasyPointsYahoo': proj.get('FantasyPointsYahoo', 0),
            'FantasyPointsFanDuel': proj.get('FantasyPointsFanDuel', 0),
            
            # Estimates for ceiling/floor
            'Ceiling': proj.get('FantasyPointsDraftKings', 0) * 1.3,
            'Floor': proj.get('FantasyPointsDraftKings', 0) * 0.6,
        }
        
        # Try to match with DK player ID
        name = player_data['Name']
        if name in dk_player_ids:
            player_data['ID'] = dk_player_ids[name]['id']
            player_data['Salary'] = dk_player_ids[name]['salary']
        else:
            player_data['ID'] = ''
            player_data['Salary'] = 0
        
        players_list.append(player_data)
    
    return pd.DataFrame(players_list)

def process_defense_projections(defenses, dk_player_ids):
    """Process defense projections"""
    
    defense_list = []
    
    for defense in defenses:
        team = defense.get('Team', '')
        defense_data = {
            'Name': f"{team} ",
            'Position': 'DST',
            'Team': team,
            'Opponent': defense.get('Opponent'),
            
            # Defense stats
            'Sacks': defense.get('Sacks', 0),
            'Interceptions': defense.get('Interceptions', 0),
            'FumblesRecovered': defense.get('FumblesRecovered', 0),
            'Touchdowns': defense.get('Touchdowns', 0),
            'PointsAllowed': defense.get('PointsAllowed', 0),
            
            # Fantasy points
            'Predicted_DK_Points': defense.get('FantasyPointsDraftKings', 0),
            'Ceiling': defense.get('FantasyPointsDraftKings', 0) * 1.4,
            'Floor': defense.get('FantasyPointsDraftKings', 0) * 0.5,
        }
        
        # Match with DK defense
        defense_name = f"{team} "
        if defense_name in dk_player_ids:
            defense_data['ID'] = dk_player_ids[defense_name]['id']
            defense_data['Salary'] = dk_player_ids[defense_name]['salary']
        else:
            defense_data['ID'] = ''
            defense_data['Salary'] = 0
        
        defense_list.append(defense_data)
    
    return pd.DataFrame(defense_list)

def main():
    print("=" * 70)
    print("NFL Week 7 Data Fetcher (sportsdata.io)")
    print("=" * 70)
    print()
    
    # Get actual games from DKEntries.csv
    dk_games = get_dk_slate_games()
    dk_teams = set()
    game_matchups = {}
    
    for away, home in dk_games:
        dk_teams.add(away)
        dk_teams.add(home)
        game_matchups[away] = home
        game_matchups[home] = away
    
    dk_teams = sorted(list(dk_teams))
    print(f"🎯 DK Slate Teams: {', '.join(dk_teams)} ({len(dk_teams)} teams)\n")
    
    # Load DK player IDs from the complete extracted file
    print("📂 Loading DraftKings player IDs...")
    try:
        dk_pool = pd.read_csv('nfl_week7_DK_PLAYER_POOL_COMPLETE.csv')
        dk_player_ids = {}
        for _, row in dk_pool.iterrows():
            dk_player_ids[row['Name']] = {
                'id': row['ID'],
                'salary': row['Salary']
            }
        print(f"✅ Loaded {len(dk_player_ids)} DK players\n")
    except Exception as e:
        print(f"⚠️  Could not load DK player pool: {e}")
        print("   Will create file without DK player IDs\n")
        dk_player_ids = {}
    
    # Fetch projections
    projections = fetch_week7_projections()
    defenses = fetch_defense_projections()
    
    if not projections:
        print("\n❌ Failed to fetch projections")
        return
    
    # Process players
    print("\n📊 Processing player data...")
    players_df = process_player_projections(projections, dk_player_ids)
    
    if defenses:
        defenses_df = process_defense_projections(defenses, dk_player_ids)
        # Combine players and defenses
        all_players_df = pd.concat([players_df, defenses_df], ignore_index=True)
    else:
        all_players_df = players_df
    
    # CRITICAL: Filter to only DK slate teams AND games
    print(f"\n🔍 Filtering to DK slate teams...")
    print(f"   Before filter: {len(all_players_df)} players")
    all_players_df = all_players_df[all_players_df['Team'].isin(dk_teams)]
    print(f"   After team filter: {len(all_players_df)} players")
    
    # Further filter by opponent to ensure correct games
    def is_correct_game(row):
        team = row['Team']
        opponent = row['Opponent']
        if pd.isna(opponent):
            return True  # Keep if no opponent data
        # Check if this matchup is in our slate
        expected_opponent = game_matchups.get(team, '')
        return opponent == expected_opponent or expected_opponent == ''
    
    all_players_df['correct_game'] = all_players_df.apply(is_correct_game, axis=1)
    filtered_df = all_players_df[all_players_df['correct_game']]
    
    print(f"   After game filter: {len(filtered_df)} players")
    print(f"   Removed {len(all_players_df) - len(filtered_df)} players from wrong games")
    
    all_players_df = filtered_df.drop(columns=['correct_game'])
    
    # Calculate Value
    all_players_df['Value'] = all_players_df.apply(
        lambda row: row['Predicted_DK_Points'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0,
        axis=1
    )
    
    # Remove players with 0 projections
    all_players_df = all_players_df[all_players_df['Predicted_DK_Points'] > 0]
    
    # Create CASH game pool (high floor players)
    cash_df = all_players_df.copy()
    cash_df['Cash_Score'] = (cash_df['Floor'] * 0.6) + (cash_df['Predicted_DK_Points'] * 0.4)
    cash_df = cash_df.sort_values('Cash_Score', ascending=False)
    
    # Create GPP pool (high ceiling players)
    gpp_df = all_players_df.copy()
    gpp_df['GPP_Score'] = (gpp_df['Ceiling'] * 0.7) + (gpp_df['Predicted_DK_Points'] * 0.3)
    gpp_df = gpp_df.sort_values('GPP_Score', ascending=False)
    
    # Save files
    cash_file = "nfl_week7_CASH_SPORTSDATA.csv"
    gpp_file = "nfl_week7_GPP_SPORTSDATA.csv"
    
    cash_df.to_csv(cash_file, index=False)
    gpp_df.to_csv(gpp_file, index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"\n📊 Summary:")
    print(f"   Total players: {len(all_players_df)}")
    print(f"   Teams: {', '.join(sorted(all_players_df['Team'].unique()))}")
    print(f"   With DK IDs: {(all_players_df['ID'] != '').sum()}")
    print(f"   With salaries: {(all_players_df['Salary'] > 0).sum()}")
    
    print(f"\n💾 Files created:")
    print(f"   📁 {cash_file} - Cash game optimized")
    print(f"   📁 {gpp_file} - GPP/Tournament optimized")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Load DKEntries.csv in optimizer")
    print(f"   2. Load {cash_file} for cash games")
    print(f"   3. Load {gpp_file} for tournaments")
    print(f"   4. Run optimization and fill entries!")

if __name__ == "__main__":
    main()

