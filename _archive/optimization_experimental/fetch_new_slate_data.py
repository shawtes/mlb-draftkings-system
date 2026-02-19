#!/usr/bin/env python3
"""
Fetch NFL data for the new slate (ATL@SF, TB@DET, HOU@SEA)
Apply contest learnings to create optimized player pool
"""

import requests
import pandas as pd
import re

API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
BASE_URL = "https://api.sportsdata.io/api/nfl/fantasy/json"

def extract_games_from_dk_file(dk_file_path):
    """Extract games from DKEntries file"""
    games = set()
    teams = set()
    
    with open(dk_file_path, 'r') as f:
        content = f.read()
        # Find patterns like "ATL@SF"
        matches = re.findall(r'([A-Z]{2,3})@([A-Z]{2,3})\s+\d{1,2}/\d{1,2}/\d{4}', content)
        for away, home in matches:
            games.add((away, home))
            teams.add(away)
            teams.add(home)
    
    return sorted(list(games)), sorted(list(teams))

def fetch_week7_projections():
    """Fetch Week 7 player projections"""
    season = "2025REG"
    week = "7"
    
    url = f"{BASE_URL}/PlayerGameProjectionStatsByWeek/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"🔍 Fetching Week {week} projections from API...")
    
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
    season = "2025REG"
    week = "7"
    
    url = f"{BASE_URL}/FantasyDefenseProjectionsByGame/{season}/{week}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"🔍 Fetching Week {week} defense projections...")
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        defenses = response.json()
        print(f"✅ Found {len(defenses)} defense projections")
        return defenses
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def load_dk_salaries(dk_file_path):
    """Load DK salaries and IDs from entries file"""
    print("📂 Loading DraftKings player data...")
    
    # Read file more carefully
    with open(dk_file_path, 'r') as f:
        lines = f.readlines()
    
    players = {}
    dst_mapping = {
        'Lions': 'DET',
        'Seahawks': 'SEA',
        '49ers': 'SF',
        'Falcons': 'ATL',
        'Texans': 'HOU',
        'Buccaneers': 'TB'
    }
    
    for line in lines:
        # Look for player data with Name + ID format
        match = re.search(r'([A-Za-z\s\.\-\']+)\s+\((\d+)\),([A-Za-z\s\.\-\']+),(\d+),[A-Z/]+,(\d+)', line)
        if match:
            name = match.group(3).strip()
            player_id = match.group(4)
            salary = int(match.group(5))
            
            players[name] = {
                'id': int(player_id),
                'salary': salary
            }
            
            # Also create mapping for DST by team abbreviation
            for dst_name, team_abbr in dst_mapping.items():
                if dst_name in name:
                    players[f"{team_abbr} "] = {
                        'id': int(player_id),
                        'salary': salary
                    }
                    players[team_abbr] = {
                        'id': int(player_id),
                        'salary': salary
                    }
    
    print(f"✅ Loaded {len(players)} DK players with salaries\n")
    return players

def apply_contest_learnings(df):
    """Apply learnings from Week 7 contest to adjust projections"""
    
    print("\n" + "="*100)
    print("APPLYING CONTEST LEARNINGS")
    print("="*100)
    
    df = df.copy()
    df['Adjusted_Projection'] = df['Predicted_DK_Points']
    df['Quality_Tier'] = 'Standard'
    
    # Elite players to boost (based on historical performance)
    elite_patterns = {
        'Christian McCaffrey': 1.25,
        'Bijan Robinson': 1.20,
        'Amon-Ra St': 1.20,
        'Jaxon Smith-Njigba': 1.15,
        'Jahmyr Gibbs': 1.15,
        'George Kittle': 1.15,
        'Baker Mayfield': 1.10,
        'Brock Purdy': 1.10
    }
    
    # Bust-prone players to penalize
    bust_patterns = {
        'Kirk Cousins': 0.60,  # Historically inconsistent
        'Brandon Aiyuk': 0.70,  # Injured/inconsistent
        'Joe Mixon': 0.70       # Injured
    }
    
    changes = []
    
    # Apply elite boosts
    for pattern, boost in elite_patterns.items():
        mask = df['Name'].str.contains(pattern, case=False, na=False)
        if mask.any():
            count = mask.sum()
            df.loc[mask, 'Adjusted_Projection'] *= boost
            df.loc[mask, 'Quality_Tier'] = 'Elite'
            changes.append(f"✅ BOOSTED {pattern}: {count} player(s) × {boost}")
    
    # Apply bust penalties
    for pattern, penalty in bust_patterns.items():
        mask = df['Name'].str.contains(pattern, case=False, na=False)
        if mask.any():
            count = mask.sum()
            df.loc[mask, 'Adjusted_Projection'] *= penalty
            df.loc[mask, 'Quality_Tier'] = 'Avoid'
            changes.append(f"❌ PENALIZED {pattern}: {count} player(s) × {penalty}")
    
    # Boost high-value plays (points per $1000)
    df['Value'] = df['Adjusted_Projection'] / (df['Salary'] / 1000)
    high_value_mask = df['Value'] > 4.0
    if high_value_mask.any():
        df.loc[high_value_mask, 'Adjusted_Projection'] *= 1.10
        df.loc[high_value_mask & (df['Quality_Tier'] == 'Standard'), 'Quality_Tier'] = 'Value'
        changes.append(f"💎 BOOSTED {high_value_mask.sum()} high-value players (>4.0x)")
    
    # Apply defensive learnings - boost strong matchups
    dst_mask = df['Position'] == 'DST'
    if dst_mask.any():
        # Boost DSTs against weak offenses
        df.loc[dst_mask, 'Adjusted_Projection'] *= 1.15
        changes.append(f"🛡️  BOOSTED all DST projections by 15% (lesson from Browns DST)")
    
    for change in changes:
        print(change)
    
    return df

def main():
    print("="*100)
    print("NFL NEW SLATE DATA FETCHER - CONTEST OPTIMIZED")
    print("="*100)
    
    dk_file = "/Users/sineshawmesfintesfaye/Downloads/DKEntries-2.csv"
    
    # Extract games
    games, teams = extract_games_from_dk_file(dk_file)
    
    print(f"\n🎯 DK Slate Teams: {', '.join(teams)}")
    print(f"📋 Games:")
    for away, home in games:
        print(f"   {away} @ {home}")
    
    # Load DK salaries
    dk_players = load_dk_salaries(dk_file)
    
    # Fetch API projections
    projections = fetch_week7_projections()
    defenses = fetch_defense_projections()
    
    if not projections:
        print("\n❌ Failed to fetch projections")
        return
    
    # Process players
    print(f"\n📊 Processing {len(projections)} player projections...")
    
    players_list = []
    for proj in projections:
        team = proj.get('Team')
        if team not in teams:
            continue  # Skip players not in this slate
        
        name = proj.get('Name')
        player_data = {
            'Position': proj.get('Position'),
            'Name': name,
            'Team': team,
            'Opponent': proj.get('Opponent'),
            'Predicted_DK_Points': proj.get('FantasyPointsDraftKings', 0),
            'Ceiling': proj.get('FantasyPointsDraftKings', 0) * 1.3,
            'Floor': proj.get('FantasyPointsDraftKings', 0) * 0.6,
            'ID': '',
            'Salary': 0
        }
        
        # Match with DK data
        if name in dk_players:
            player_data['ID'] = dk_players[name]['id']
            player_data['Salary'] = dk_players[name]['salary']
        
        if player_data['Predicted_DK_Points'] > 0:
            players_list.append(player_data)
    
    df = pd.DataFrame(players_list)
    
    # Process defenses
    if defenses:
        for defense in defenses:
            team = defense.get('Team')
            if team not in teams:
                continue
            
            defense_name = team
            defense_data = {
                'Position': 'DST',
                'Name': f"{team} ",
                'Team': team,
                'Opponent': defense.get('Opponent'),
                'Predicted_DK_Points': defense.get('FantasyPointsDraftKings', 0),
                'Ceiling': defense.get('FantasyPointsDraftKings', 0) * 1.4,
                'Floor': defense.get('FantasyPointsDraftKings', 0) * 0.5,
                'ID': '',
                'Salary': 0
            }
            
            # Try to match defense
            for dk_name, dk_data in dk_players.items():
                if team in dk_name or f"{team} " == dk_name:
                    defense_data['ID'] = dk_data['id']
                    defense_data['Salary'] = dk_data['salary']
                    break
            
            if defense_data['Predicted_DK_Points'] > 0:
                df = pd.concat([df, pd.DataFrame([defense_data])], ignore_index=True)
    
    print(f"✅ Processed {len(df)} players for this slate")
    
    # Apply contest learnings
    df = apply_contest_learnings(df)
    
    # Create GPP and Cash versions
    gpp_df = df.copy()
    gpp_df['GPP_Score'] = (gpp_df['Ceiling'] * 0.7) + (gpp_df['Adjusted_Projection'] * 0.3)
    gpp_df = gpp_df.sort_values('GPP_Score', ascending=False)
    
    cash_df = df.copy()
    cash_df['Cash_Score'] = (cash_df['Floor'] * 0.6) + (cash_df['Adjusted_Projection'] * 0.4)
    cash_df = cash_df.sort_values('Cash_Score', ascending=False)
    
    # Rename column for optimizer compatibility
    gpp_df['AvgPointsPerGame'] = gpp_df['Adjusted_Projection']
    cash_df['AvgPointsPerGame'] = cash_df['Adjusted_Projection']
    
    # Save files
    gpp_file = "nfl_new_slate_GPP_OPTIMIZED.csv"
    cash_file = "nfl_new_slate_CASH_OPTIMIZED.csv"
    
    gpp_df.to_csv(gpp_file, index=False)
    cash_df.to_csv(cash_file, index=False)
    
    print(f"\n{'='*100}")
    print("FILES CREATED")
    print(f"{'='*100}")
    print(f"✅ {gpp_file} - GPP/Tournament optimized with contest learnings")
    print(f"✅ {cash_file} - Cash game optimized with contest learnings")
    
    # Show top players
    print(f"\n{'='*100}")
    print("TOP 20 PLAYERS BY ADJUSTED PROJECTION")
    print(f"{'='*100}")
    top = gpp_df.nlargest(20, 'Adjusted_Projection')
    for i, (idx, row) in enumerate(top.iterrows(), 1):
        tier_emoji = "🔥" if row['Quality_Tier'] == 'Elite' else "💎" if row['Quality_Tier'] == 'Value' else "⚠️" if row['Quality_Tier'] == 'Avoid' else "📊"
        print(f"{i:2d}. {tier_emoji} {row['Name']:30s} ({row['Position']:3s}-{row['Team']:3s}) ${row['Salary']:5.0f} → {row['Adjusted_Projection']:5.1f} pts")
    
    print(f"\n💡 Ready to load into optimizer!")
    print(f"   Load: {gpp_file} for GPP contests")
    print(f"   Load: {cash_file} for cash games")

if __name__ == "__main__":
    main()

