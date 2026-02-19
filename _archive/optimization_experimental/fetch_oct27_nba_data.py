#!/usr/bin/env python3
"""
Fetch NBA data for October 27th from the API
"""

import sys
import requests
import pandas as pd
from datetime import datetime

# API Key from existing files
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

def fetch_oct27_data():
    """Fetch October 27th NBA data"""
    
    print(f"\n{'='*60}")
    print(f"🏀 Fetching NBA Data for October 27th, 2025")
    print(f"{'='*60}\n")
    
    # Format date for API
    date_formatted = "2025-OCT-27"
    
    # Try different endpoints
    endpoints = [
        f"/PlayerGameProjectionStatsByDate/{date_formatted}",
        f"/DfsSlatesByDate/{date_formatted}",
        "/Players",
        "/Teams"
    ]
    
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    
    print(f"🔑 Using API Key: {API_KEY[:20]}...")
    print(f"📅 Target date: {date_formatted}")
    
    # Try to fetch projections
    endpoint = f"/PlayerGameProjectionStatsByDate/{date_formatted}"
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n📥 Attempting to fetch from: {url}")
    
    try:
        # Method 1: With header
        print("   Method 1: Header authentication")
        response = requests.get(url, headers=headers, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Success!")
            data = response.json()
            print(f"   Retrieved {len(data)} records")
            return data, endpoint
        
        # Method 2: With query parameter
        if response.status_code == 401:
            print("\n   Method 2: Query parameter authentication")
            params = {'key': API_KEY}
            response2 = requests.get(url, params=params, timeout=30)
            print(f"   Status: {response2.status_code}")
            
            if response2.status_code == 200:
                print("   ✅ Success!")
                data = response2.json()
                print(f"   Retrieved {len(data)} records")
                return data, endpoint
        
        # Method 3: Try with different header format
        if response.status_code == 401:
            print("\n   Method 3: Different header format")
            headers2 = {"Ocp-Apim-Subscription-Key": API_KEY, "Content-Type": "application/json"}
            response3 = requests.get(url, headers=headers2, timeout=30)
            print(f"   Status: {response3.status_code}")
            
            if response3.status_code == 200:
                print("   ✅ Success!")
                data = response3.json()
                print(f"   Retrieved {len(data)} records")
                return data, endpoint
        
        print(f"\n❌ Failed with status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # If all fails, try getting just players
    print(f"\n🔄 Trying alternative: Fetch all players")
    try:
        players_url = f"{BASE_URL}/Players"
        response = requests.get(players_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ Could fetch players, API key works!")
            players = response.json()
            print(f"   Retrieved {len(players)} players")
            
            # Now try date-specific data
            print(f"\n🔄 Retrying with season-specific endpoint...")
            
            # Try different endpoint formats
            alt_endpoints = [
                f"PlayerGameProjectionStatsByDate/{date_formatted}",
                f"PlayerGameProjectionStatsByDate/{date_formatted.lower()}",
                f"PlayerGameProjectionStatsByDate/2025-10-27",
            ]
            
            for alt_endpoint in alt_endpoints:
                test_url = f"{BASE_URL}/{alt_endpoint}"
                print(f"   Trying: {test_url}")
                r = requests.get(test_url, headers=headers, timeout=30)
                if r.status_code == 200:
                    print(f"   ✅ Success with: {alt_endpoint}")
                    return r.json(), alt_endpoint
                else:
                    print(f"   Status: {r.status_code}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    return None, None


def process_and_save(data, source_info):
    """Process fetched data and save to CSV"""
    
    if data is None:
        print("\n❌ No data to process")
        return None
    
    print(f"\n📊 Processing data from: {source_info}")
    
    try:
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df)} records")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Columns: {list(df.columns)[:10]}")
        
        # Check for key columns
        has_name = 'Name' in df.columns or 'PlayerName' in df.columns
        has_position = 'Position' in df.columns
        has_fantasy_points = 'FantasyPointsDraftKings' in df.columns
        
        print(f"\n📋 Key columns present:")
        print(f"   Name: {has_name}")
        print(f"   Position: {has_position}")
        print(f"   Fantasy Points: {has_fantasy_points}")
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nba_oct27_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved raw data to: {filename}")
        
        # If we have FantasyPointsDraftKings, create optimizer-ready version
        if has_name and has_fantasy_points:
            optimizer_df = pd.DataFrame()
            
            # Get name
            if 'Name' in df.columns:
                optimizer_df['Name'] = df['Name']
            elif 'PlayerName' in df.columns:
                optimizer_df['Name'] = df['PlayerName']
            
            # Get position
            if 'Position' in df.columns:
                optimizer_df['Position'] = df['Position']
            
            # Get team
            if 'Team' in df.columns:
                optimizer_df['Team'] = df['Team']
            
            # Get fantasy points
            if 'FantasyPointsDraftKings' in df.columns:
                optimizer_df['Predicted_DK_Points'] = df['FantasyPointsDraftKings']
            
            # Estimate salary
            if 'Predicted_DK_Points' in optimizer_df.columns:
                optimizer_df['Salary'] = (optimizer_df['Predicted_DK_Points'] / 5 * 1000).astype(int).clip(3000, 12000)
            
            # Save optimized version
            opt_filename = f"nba_oct27_FIXED.csv"
            optimizer_df.to_csv(opt_filename, index=False)
            print(f"💾 Saved optimizer-ready data to: {opt_filename}")
            print(f"\n📊 Players: {len(optimizer_df)}")
            print(f"📋 Sample:")
            print(optimizer_df.head())
            
            return optimizer_df
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    data, endpoint = fetch_oct27_data()
    
    if data:
        df = process_and_save(data, endpoint)
        if df is not None:
            print(f"\n{'='*60}")
            print("✅ Done! Data saved")
            print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("❌ Could not fetch data")
        print("\nPossible issues:")
        print("  1. API key may need activation")
        print("  2. Date format might be wrong")
        print("  3. No data available for that date")
        print("  4. Try a different date format")
        print(f"{'='*60}\n")

