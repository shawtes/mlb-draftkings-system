#!/usr/bin/env python3
"""
Fetch NFL Week 8 Data for DraftKings 13-Game Slate
==================================================

This script fetches NFL data for Week 8 using SportsData.io API
and formats it for the genetic algorithm optimizer.

For Week 8 (October 26-27, 2025), we'll fetch:
- DraftKings salaries for the 13-game slate
- Player projections for Week 8
- Injury reports
- Game information
"""

import sys
import os
sys.path.append('../python_algorithms')

from sportsdata_nfl_api import SportsDataNFLAPI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# API Key
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"

def fetch_week8_data():
    """
    Fetch all data needed for Week 8 NFL DFS
    """
    print("="*80)
    print("🏈 NFL WEEK 8 DATA FETCHER - 13 GAME SLATE")
    print("="*80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📅 Target: Week 8 (October 26-27, 2025)")
    print("="*80)
    
    api = SportsDataNFLAPI(API_KEY)
    
    # Week 8 dates - Sunday games are typically on October 26, 2025
    # Let's try both Saturday and Sunday to find the 13-game slate
    dates_to_try = [
        "2025-10-26",  # Sunday (most likely)
        "2025-10-25",  # Saturday (if there are Saturday games)
        "2025-10-27",  # Monday (if there are Monday games)
    ]
    
    slate_data = None
    selected_date = None
    
    # Try to find the 13-game slate
    for date in dates_to_try:
        print(f"\n🔍 Checking {date} for 13-game slate...")
        temp_slate = api.get_dfs_slates_by_date(date, save_to_file=False)
        
        if temp_slate:
            for slate in temp_slate:
                if slate.get('Operator', '').upper() == 'DRAFTKINGS':
                    games = slate.get('DfsSlateGames', [])
                    if len(games) >= 13:  # 13-game slate
                        slate_data = slate
                        selected_date = date
                        print(f"✅ Found 13-game DraftKings slate on {date}!")
                        print(f"   Games: {len(games)}")
                        break
            
            if slate_data:
                break
    
    if not slate_data:
        print("❌ Could not find 13-game DraftKings slate")
        print("   Trying to fetch any available slate...")
        
        # Fallback: try the most likely date
        slate_data = api.get_dfs_slates_by_date("2025-10-26", save_to_file=True)
        if slate_data:
            for slate in slate_data:
                if slate.get('Operator', '').upper() == 'DRAFTKINGS':
                    slate_data = slate
                    selected_date = "2025-10-26"
                    break
    
    if not slate_data:
        print("❌ No DraftKings slate found for Week 8")
        return None
    
    print(f"\n📊 Slate Information:")
    print(f"   Date: {selected_date}")
    print(f"   Operator: {slate_data.get('Operator', 'Unknown')}")
    print(f"   Slate ID: {slate_data.get('SlateID', 'Unknown')}")
    print(f"   Games: {len(slate_data.get('DfsSlateGames', []))}")
    print(f"   Players: {len(slate_data.get('DfsSlatePlayers', []))}")
    
    # Extract player data
    players = slate_data.get('DfsSlatePlayers', [])
    print(f"\n✅ Found {len(players)} players in DraftKings slate")
    
    # Create DataFrame
    df = pd.DataFrame(players)
    
    # Show position breakdown
    print(f"\n📊 Position Breakdown:")
    if 'OperatorPosition' in df.columns:
        pos_counts = df['OperatorPosition'].value_counts()
        for pos, count in pos_counts.items():
            avg_salary = df[df['OperatorPosition'] == pos]['OperatorSalary'].mean()
            print(f"   {pos}: {count} players (Avg: ${avg_salary:,.0f})")
    
    # Fetch projections for Week 8
    print(f"\n📊 Fetching Week 8 Projections...")
    projections = api.get_player_projections_by_week("2025REG", 8, save_to_file=True)
    
    if projections:
        print(f"✅ Loaded {len(projections)} player projections")
        proj_df = pd.DataFrame(projections)
        
        # Show top projected players by position
        print(f"\n🏆 Top Projected Players by Position:")
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            pos_proj = proj_df[proj_df['Position'] == pos].nlargest(3, 'FantasyPoints')
            if not pos_proj.empty:
                print(f"\n   {pos}:")
                for _, player in pos_proj.iterrows():
                    print(f"      {player['Name']}: {player['FantasyPoints']:.1f} pts")
    else:
        print("⚠️  No projections available yet")
        proj_df = None
    
    # Fetch injury data
    print(f"\n🏥 Fetching Week 8 Injury Reports...")
    injuries = api.get_injuries_by_week("2025REG", 8, save_to_file=True)
    
    if injuries:
        print(f"✅ Loaded {len(injuries)} injury records")
        
        # Show injury summary
        injury_df = pd.DataFrame(injuries)
        if 'InjuryStatus' in injury_df.columns:
            status_counts = injury_df['InjuryStatus'].value_counts()
            print(f"\n📊 Injury Status Summary:")
            for status, count in status_counts.items():
                print(f"   {status}: {count} players")
    else:
        print("⚠️  No injury data available yet")
    
    # Create optimizer-ready CSV
    print(f"\n🔧 Creating optimizer-ready CSV...")
    
    # Format for genetic algorithm optimizer
    optimizer_df = pd.DataFrame()
    
    # Map DraftKings columns to optimizer format
    optimizer_df['Name'] = df.get('OperatorPlayerName', df.get('PlayerName', ''))
    optimizer_df['PlayerID'] = df.get('PlayerID', '')
    optimizer_df['Position'] = df.get('OperatorPosition', '')
    optimizer_df['Team'] = df.get('Team', '')
    optimizer_df['Salary'] = df.get('OperatorSalary', 0)
    optimizer_df['OperatorPlayerID'] = df.get('OperatorPlayerID', '')
    optimizer_df['RosterSlots'] = df.get('OperatorRosterSlots', [])
    
    # Merge with projections if available
    if proj_df is not None:
        # Merge on PlayerID
        if 'PlayerID' in proj_df.columns:
            merged = optimizer_df.merge(
                proj_df[['PlayerID', 'FantasyPoints']],
                on='PlayerID',
                how='left'
            )
            optimizer_df['FantasyPoints'] = merged['FantasyPoints'].fillna(0)
        else:
            # Fallback to name matching
            proj_df['MatchName'] = proj_df['Name'].str.strip().str.upper()
            optimizer_df['MatchName'] = optimizer_df['Name'].str.strip().str.upper()
            
            merged = optimizer_df.merge(
                proj_df[['MatchName', 'FantasyPoints']],
                on='MatchName',
                how='left'
            )
            optimizer_df['FantasyPoints'] = merged['FantasyPoints'].fillna(0)
            optimizer_df = optimizer_df.drop(columns=['MatchName'])
        
        print(f"✅ Merged projections for {optimizer_df['FantasyPoints'].notna().sum()} players")
    else:
        optimizer_df['FantasyPoints'] = 0
        print("⚠️  No projections - FantasyPoints set to 0")
    
    # Position mapping for NFL
    position_map = {
        'DEF': 'DST',
        'D': 'DST',
    }
    optimizer_df['Position'] = optimizer_df['Position'].replace(position_map)
    
    # Filter to valid positions
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    before_filter = len(optimizer_df)
    optimizer_df = optimizer_df[optimizer_df['Position'].isin(valid_positions)]
    print(f"✅ Filtered to valid positions: {len(optimizer_df)}/{before_filter} players")
    
    # Data quality checks
    optimizer_df = optimizer_df.dropna(subset=['Name', 'Position', 'Salary'])
    optimizer_df = optimizer_df[optimizer_df['Salary'] > 0]
    print(f"✅ After data quality checks: {len(optimizer_df)} players")
    
    # Add value calculations
    optimizer_df['Value'] = optimizer_df.apply(
        lambda row: row['FantasyPoints'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0,
        axis=1
    )
    optimizer_df['PointsPerK'] = optimizer_df['Value']
    
    # Save to CSV
    output_filename = f"nfl_week8_draftkings_13game_slate.csv"
    optimizer_df.to_csv(output_filename, index=False)
    print(f"\n💾 Saved to: {output_filename}")
    
    # Show sample data
    print(f"\n📋 Top 10 by Salary:")
    top_salary = optimizer_df.nlargest(10, 'Salary')[
        ['Name', 'Position', 'Team', 'Salary', 'FantasyPoints', 'Value']
    ]
    print(top_salary.to_string(index=False))
    
    print(f"\n📋 Top 10 by Projection:")
    top_proj = optimizer_df.nlargest(10, 'FantasyPoints')[
        ['Name', 'Position', 'Team', 'Salary', 'FantasyPoints', 'Value']
    ]
    print(top_proj.to_string(index=False))
    
    # Show position breakdown
    print(f"\n📊 Final Position Breakdown:")
    pos_counts = optimizer_df['Position'].value_counts().sort_index()
    for pos, count in pos_counts.items():
        avg_salary = optimizer_df[optimizer_df['Position'] == pos]['Salary'].mean()
        avg_proj = optimizer_df[optimizer_df['Position'] == pos]['FantasyPoints'].mean()
        print(f"   {pos}: {count} players (Avg Salary: ${avg_salary:,.0f}, Avg Proj: {avg_proj:.1f})")
    
    print(f"\n" + "="*80)
    print("✅ WEEK 8 DATA READY!")
    print("="*80)
    print(f"📁 File: {output_filename}")
    print(f"📊 Players: {len(optimizer_df)}")
    print(f"🎯 Ready for genetic algorithm optimizer")
    print(f"\n🎮 Next steps:")
    print(f"   1. Open genetic_algo_nfl_optimizer.py")
    print(f"   2. Load {output_filename}")
    print(f"   3. Configure GPP settings")
    print(f"   4. Generate lineups!")
    
    return optimizer_df

if __name__ == "__main__":
    fetch_week8_data()
