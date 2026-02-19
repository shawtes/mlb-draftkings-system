#!/usr/bin/env python3
"""
Fetch clean NBA data from API and merge with DraftKings entries file
"""

import requests
import pandas as pd
from datetime import datetime
import sys

# API Configuration
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

def fetch_nba_projections(date="2025-OCT-27"):
    """Fetch NBA player projections from SportsData.io API"""
    
    print(f"\n{'='*70}")
    print(f"🏀 FETCHING NBA PROJECTIONS FROM API")
    print(f"{'='*70}\n")
    
    endpoint = f"/PlayerGameProjectionStatsByDate/{date}"
    url = f"{BASE_URL}{endpoint}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"📥 Endpoint: {endpoint}")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Retrieved {len(data)} player projections")
            return pd.DataFrame(data)
        else:
            print(f"❌ API Error: Status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def process_api_data(api_df):
    """Process API data to get projections"""
    
    print(f"\n{'='*70}")
    print(f"📊 PROCESSING API DATA")
    print(f"{'='*70}\n")
    
    # Create clean DataFrame
    processed = pd.DataFrame()
    
    # Map columns
    processed['Name'] = api_df['Name']
    processed['Team'] = api_df['Team']
    
    # Get DK fantasy points if available
    if 'FantasyPointsDraftKings' in api_df.columns:
        processed['Predicted_DK_Points'] = api_df['FantasyPointsDraftKings']
    else:
        # Calculate from stats
        processed['Predicted_DK_Points'] = 0
        if 'Points' in api_df.columns:
            processed['Predicted_DK_Points'] += api_df['Points']
        if 'Rebounds' in api_df.columns:
            processed['Predicted_DK_Points'] += api_df['Rebounds'] * 1.25
        if 'Assists' in api_df.columns:
            processed['Predicted_DK_Points'] += api_df['Assists'] * 1.5
        if 'Steals' in api_df.columns:
            processed['Predicted_DK_Points'] += api_df['Steals'] * 2
        if 'BlockedShots' in api_df.columns:
            processed['Predicted_DK_Points'] += api_df['BlockedShots'] * 2
        if 'Turnovers' in api_df.columns:
            processed['Predicted_DK_Points'] -= api_df['Turnovers'] * 0.5
    
    if 'Opponent' in api_df.columns:
        processed['Opponent'] = api_df['Opponent']
    elif 'OpponentTeam' in api_df.columns:
        processed['Opponent'] = api_df['OpponentTeam']
    
    print(f"✅ Processed {len(processed)} player projections")
    return processed


def load_dk_entries(dk_file):
    """Load DraftKings entries CSV file"""
    
    print(f"\n{'='*70}")
    print(f"📋 LOADING DRAFTKINGS ENTRIES FILE")
    print(f"{'='*70}\n")
    
    try:
        dk_df = pd.read_csv(dk_file)
        print(f"✅ Loaded DK entries: {len(dk_df)} players")
        
        # Show what columns we have
        print(f"📊 Columns: {list(dk_df.columns)}")
        
        # Check if we have the required columns
        required = ['Name']
        missing = [col for col in required if col not in dk_df.columns]
        if missing:
            print(f"⚠️ Missing columns: {missing}")
            return None
        
        return dk_df
        
    except Exception as e:
        print(f"❌ Failed to load DK entries: {e}")
        return None


def parse_dk_player_data(text_data):
    """
    Parse DK player data from text format:
    Name + ID	Name	ID	Roster Position	Salary	Game Info
    """
    
    print(f"\n{'='*70}")
    print(f"📋 PARSING DK PLAYER DATA")
    print(f"{'='*70}\n")
    
    players = []
    lines = text_data.strip().split('\n')
    
    for line in lines[1:]:  # Skip header
        parts = line.split('\t')
        if len(parts) >= 5:
            try:
                name = parts[1].strip()
                dk_id = parts[2].strip()
                roster_position = parts[3].strip()
                salary = parts[4].strip()
                
                # Clean salary (remove commas)
                salary = salary.replace(',', '')
                
                players.append({
                    'Name': name,
                    'DK_ID': dk_id,
                    'Roster_Position': roster_position,
                    'Salary': int(salary)
                })
            except:
                continue
    
    df = pd.DataFrame(players)
    print(f"✅ Parsed {len(df)} players with DK data")
    return df


def merge_data(api_df, dk_df):
    """Merge API projections with DK player data"""
    
    print(f"\n{'='*70}")
    print(f"🔗 MERGING API + DK DATA")
    print(f"{'='*70}\n")
    
    # Normalize names for matching
    api_df['Name_lower'] = api_df['Name'].str.strip().str.lower()
    dk_df['Name_lower'] = dk_df['Name'].str.strip().str.lower()
    
    # Merge on name
    merged = api_df.merge(
        dk_df[['Name_lower', 'DK_ID', 'Roster_Position', 'Salary']],
        on='Name_lower',
        how='inner'
    )
    
    # Drop the lowercase name column
    merged = merged.drop(columns=['Name_lower'])
    
    # Extract simple position from Roster_Position for backward compatibility
    merged['Position'] = merged['Roster_Position'].str.extract(r'(PG|SG|SF|PF|C)')[0]
    
    print(f"✅ Matched {len(merged)} players")
    print(f"📊 Teams: {sorted(merged['Team'].unique().tolist())}")
    
    # Show sample
    print(f"\n📋 Sample of merged data:")
    print(merged[['Name', 'Position', 'Roster_Position', 'Team', 'Salary', 'Predicted_DK_Points', 'DK_ID']].head(10).to_string(index=False))
    
    return merged


def save_output(df, output_file='nba_oct27_READY.csv'):
    """Save the final dataset"""
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING OUTPUT")
    print(f"{'='*70}\n")
    
    # Reorder columns
    cols = ['Name', 'DK_ID', 'Position', 'Roster_Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Opponent']
    # Only include columns that exist
    cols = [col for col in cols if col in df.columns]
    df = df[cols]
    
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} players to: {output_file}")
    
    # Summary
    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"   Total players: {len(df)}")
    print(f"   Teams: {len(df['Team'].unique())}")
    print(f"   Avg salary: ${df['Salary'].mean():.0f}")
    print(f"   Avg projected points: {df['Predicted_DK_Points'].mean():.2f}")
    
    print(f"\n✅ Ready for optimization!")


def main():
    """Main execution"""
    
    # Configuration
    date = "2025-OCT-27"
    dk_entries_file = "dk_player_pool_oct27.csv"  # DK player pool with IDs, positions, salaries
    output_file = "nba_oct27_READY.csv"
    
    print(f"\n🏀 NBA DATA FETCHER")
    print(f"{'='*70}")
    print(f"Date: {date}")
    print(f"DK Entries: {dk_entries_file}")
    print(f"Output: {output_file}")
    print(f"{'='*70}\n")
    
    # Step 1: Fetch from API
    api_df = fetch_nba_projections(date)
    if api_df is None:
        print("\n❌ Failed to fetch API data")
        sys.exit(1)
    
    # Step 2: Process API data
    processed_df = process_api_data(api_df)
    
    # Step 3: Load DK entries
    dk_df = load_dk_entries(dk_entries_file)
    if dk_df is None:
        print("\n❌ Failed to load DK entries")
        sys.exit(1)
    
    # Step 4: Merge
    final_df = merge_data(processed_df, dk_df)
    
    if len(final_df) == 0:
        print("\n❌ No players matched between API and DK entries")
        sys.exit(1)
    
    # Step 5: Save
    save_output(final_df, output_file)
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

