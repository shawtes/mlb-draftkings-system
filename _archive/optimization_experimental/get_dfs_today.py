#!/usr/bin/env python3
"""
Super Simple: Get Today's DFS Data
===================================

Just run this script to get today's NFL DFS data as CSV!

Usage:
    python get_dfs_today.py
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

# Your API key (already configured)
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"

def main():
    print("\n" + "="*70)
    print("🏈 FETCHING TODAY'S NFL DFS DATA")
    print("="*70 + "\n")
    
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"📅 Date: {today}")
    
    # You'll need to specify the week - NFL weeks change weekly
    # For now, let's try to estimate or ask user
    print("\n⚠️  You need to know the NFL week number.")
    print("   Example: Week 8, Week 9, etc.")
    
    try:
        week_input = input("\nEnter NFL week number (or press Enter for week 8): ").strip()
        week = int(week_input) if week_input else 8
    except ValueError:
        print("⚠️  Invalid input, using week 8")
        week = 8
    
    season = "2025REG"  # Adjust if needed
    
    print(f"\n📊 Week: {week}")
    print(f"📊 Season: {season}")
    print(f"\n🚀 Fetching data...\n")
    
    # Initialize API
    api = SportsDataNFLAPI(API_KEY)
    
    # Export to CSV
    csv_file = api.export_dfs_players_to_csv(
        date=today,
        season=season,
        week=week,
        include_projections=True
    )
    
    if csv_file:
        print("\n" + "="*70)
        print("✅ SUCCESS! CSV FILE CREATED")
        print("="*70)
        print(f"\n📁 File: {csv_file}")
        print(f"\n💡 You can now:")
        print(f"   1. Open {csv_file}")
        print(f"   2. Load it in your optimizer")
        print(f"   3. Generate lineups!")
        print("\n" + "="*70)
    else:
        print("\n" + "="*70)
        print("❌ FAILED")
        print("="*70)
        print("\nPossible reasons:")
        print("  • No DFS slate available for today")
        print("  • Wrong week number")
        print("  • API connection issue")
        print("\nTry running: python fetch_todays_dfs.py --date YYYY-MM-DD --week X")

if __name__ == "__main__":
    main()



