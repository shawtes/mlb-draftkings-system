#!/usr/bin/env python3
"""
Fetch NBA data from API and filter using DK entries file
"""

import sys
import requests
import pandas as pd
from datetime import datetime

# API Key
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

def fetch_nba_data(date="2025-OCT-27"):
    """Fetch NBA projections from API"""
    
    print(f"\n{'='*70}")
    print(f"🏀 Fetching NBA Data for {date}")
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


def filter_by_dk_entries(df, dk_file="nba_tonight_FIXED.csv"):
    """Filter fetched data to only include players in DK entries"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Filtering by DK entries from: {dk_file}")
    print(f"{'='*70}\n")
    
    # Load DK-approved players
    dk_df = pd.read_csv(dk_file)
    dk_names = set(dk_df['Name'].str.strip().str.lower())
    
    print(f"📋 DK-approved players: {len(dk_names)}")
    print(f"📋 Fetched players: {len(df)}")
    
    # Filter
    df['Name_lower'] = df['Name'].str.strip().str.lower()
    filtered = df[df['Name_lower'].isin(dk_names)].copy()
    filtered = filtered.drop(columns=['Name_lower'])
    
    print(f"📋 Filtered players: {len(filtered)}")
    
    # Show removed players
    removed = df[~df['Name_lower'].isin(dk_names)]
    if len(removed) > 0:
        print(f"🗑️  Removed {len(removed)} players not in DK pool")
        print(f"🗑️  Examples: {', '.join(removed['Name'].head(10).tolist())}")
    
    # Prepare optimizer-ready format
    optimizer_df = pd.DataFrame()
    
    # Map columns
    if 'Name' in filtered.columns:
        optimizer_df['Name'] = filtered['Name']
    elif 'PlayerName' in filtered.columns:
        optimizer_df['Name'] = filtered['PlayerName']
    
    if 'Position' in filtered.columns:
        optimizer_df['Position'] = filtered['Position']
    
    if 'Team' in filtered.columns:
        optimizer_df['Team'] = filtered['Team']
    
    # Get salary and projections
    if 'FantasyPointsDraftKings' in filtered.columns:
        optimizer_df['Predicted_DK_Points'] = filtered['FantasyPointsDraftKings']
    elif 'Predicted_DK_Points' in filtered.columns:
        optimizer_df['Predicted_DK_Points'] = filtered['Predicted_DK_Points']
    
    if 'Salary' in filtered.columns:
        optimizer_df['Salary'] = filtered['Salary']
    elif 'SalaryDraftKings' in filtered.columns:
        optimizer_df['Salary'] = filtered['SalaryDraftKings']
    else:
        # Estimate from points
        optimizer_df['Salary'] = (optimizer_df['Predicted_DK_Points'] / 5 * 1000).astype(int).clip(3000, 12000)
    
    if 'Opponent' in filtered.columns:
        optimizer_df['Opponent'] = filtered['Opponent']
    
    return optimizer_df


def main():
    # Fetch fresh data
    df = fetch_nba_data()
    
    if df is None:
        print("\n❌ Could not fetch data from API")
        return
    
    # Filter by DK entries
    filtered_df = filter_by_dk_entries(df)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"nba_READY_{timestamp}.csv"
    filtered_df.to_csv(filename, index=False)
    
    print(f"\n{'='*70}")
    print(f"✅ Saved filtered data to: {filename}")
    print(f"📊 Total players: {len(filtered_df)}")
    print(f"{'='*70}\n")
    
    print("📋 Sample data:")
    print(filtered_df.head(10))


if __name__ == "__main__":
    main()

