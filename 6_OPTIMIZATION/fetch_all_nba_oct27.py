#!/usr/bin/env python3
"""
Fetch ALL NBA players for October 27th from API
"""

import requests
import pandas as pd
from datetime import datetime

# API Key
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

def fetch_all_nba_players(date="2025-OCT-27"):
    """Fetch all NBA player projections from API"""
    
    print(f"\n{'='*70}")
    print(f"🏀 Fetching ALL NBA Players for {date}")
    print(f"{'='*70}\n")
    
    # Fetch player projections
    endpoint = f"/PlayerGameProjectionStatsByDate/{date}"
    url = f"{BASE_URL}{endpoint}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"📥 Fetching from: {endpoint}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data)} player projections")
            return pd.DataFrame(data)
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def process_data(df):
    """Process and prepare data for optimizer"""
    
    print(f"\n{'='*70}")
    print(f"📊 Processing Data")
    print(f"{'='*70}\n")
    
    print(f"📋 Total players from API: {len(df)}")
    print(f"📋 Columns: {list(df.columns)[:10]}")
    
    # Create optimizer-ready DataFrame
    optimizer_df = pd.DataFrame()
    
    # Map columns
    if 'Name' in df.columns:
        optimizer_df['Name'] = df['Name']
    elif 'PlayerName' in df.columns:
        optimizer_df['Name'] = df['PlayerName']
    
    if 'Position' in df.columns:
        optimizer_df['Position'] = df['Position']
    
    if 'Team' in df.columns:
        optimizer_df['Team'] = df['Team']
    
    # Get salary
    if 'Salary' in df.columns:
        optimizer_df['Salary'] = df['Salary']
    elif 'SalaryDraftKings' in df.columns:
        optimizer_df['Salary'] = df['SalaryDraftKings']
    else:
        print("⚠️ No salary column found, will estimate from points")
    
    # Get predictions
    if 'FantasyPointsDraftKings' in df.columns:
        optimizer_df['Predicted_DK_Points'] = df['FantasyPointsDraftKings']
    elif 'Predicted_DK_Points' in df.columns:
        optimizer_df['Predicted_DK_Points'] = df['Predicted_DK_Points']
    else:
        # Calculate from stats if available
        optimizer_df['Predicted_DK_Points'] = 0
        if 'Points' in df.columns:
            optimizer_df['Predicted_DK_Points'] += df['Points']
        if 'Rebounds' in df.columns:
            optimizer_df['Predicted_DK_Points'] += df['Rebounds'] * 1.2
        if 'Assists' in df.columns:
            optimizer_df['Predicted_DK_Points'] += df['Assists'] * 1.5
        if 'Steals' in df.columns:
            optimizer_df['Predicted_DK_Points'] += df['Steals'] * 3
        if 'BlockedShots' in df.columns:
            optimizer_df['Predicted_DK_Points'] += df['BlockedShots'] * 3
        if 'Turnovers' in df.columns:
            optimizer_df['Predicted_DK_Points'] -= df['Turnovers'] * 1
    
    # Estimate salary if not present
    if 'Salary' not in optimizer_df.columns:
        optimizer_df['Salary'] = (optimizer_df['Predicted_DK_Points'] / 5 * 1000).astype(int).clip(3000, 12000)
        print("📊 Estimated salary from points")
    
    if 'Opponent' in df.columns:
        optimizer_df['Opponent'] = df['Opponent']
    elif 'OpponentTeam' in df.columns:
        optimizer_df['Opponent'] = df['OpponentTeam']
    
    return optimizer_df


def main():
    # Fetch all players
    df = fetch_all_nba_players()
    
    if df is None:
        print("\n❌ Could not fetch data from API")
        return
    
    # Process data
    processed_df = process_data(df)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nba_all_players_oct27_{timestamp}.csv"
    processed_df.to_csv(filename, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Saved ALL players to: {filename}")
    print(f"📊 Total players: {len(processed_df)}")
    print(f"{'='*70}\n")
    
    print("📋 Sample data:")
    print(processed_df.head(20).to_string(index=False))
    
    print(f"\n📊 Teams available:")
    print(processed_df['Team'].value_counts().to_string())


if __name__ == "__main__":
    main()

