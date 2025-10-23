#!/usr/bin/env python3
"""
Simple NFL API to DraftKings Formatter
======================================

Easy-to-use script that loads NFL data from API and formats it
for your DraftKings optimizer.

Usage:
    python simple_nfl_formatter.py
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the python_algorithms directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def format_nfl_data_for_optimizer(date, week, season='2025REG'):
    """
    Load NFL data from API and format for DraftKings optimizer
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           Simple NFL API to DraftKings Formatter       ║
    ║              Loads data and formats for optimizer         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize API
    api = SportsDataNFLAPI("1dd5e646265649af87e0d9cdb80d1c8c")
    
    print(f"🏈 Loading NFL Week {week} data for {date}")
    print("="*60)
    
    # Fetch DFS slate data
    print(f"📊 Fetching DFS slates...")
    slate_data = api.get_dfs_slates_by_date(date, save_to_file=False)
    
    if not slate_data:
        print("❌ Failed to fetch DFS slate data")
        return None
    
    # Find DraftKings slate
    dk_slate = None
    for slate in slate_data:
        if slate.get('Operator', '').upper() == 'DRAFTKINGS':
            dk_slate = slate
            break
    
    if not dk_slate:
        print("❌ No DraftKings slate found")
        return None
    
    print(f"✅ Found DraftKings slate with {len(dk_slate.get('DfsSlatePlayers', []))} players")
    
    # Fetch projections
    print(f"📊 Fetching projections...")
    projections = api.get_player_projections_by_week(season, week, save_to_file=False)
    
    if projections:
        print(f"✅ Loaded {len(projections)} player projections")
    else:
        print("⚠️ No projections available")
        projections = []
    
    # Format the data
    print(f"🔧 Formatting data...")
    
    # Extract player data
    players = dk_slate.get('DfsSlatePlayers', [])
    df = pd.DataFrame(players)
    
    # Create formatted DataFrame
    formatted_df = pd.DataFrame()
    
    # Map columns
    formatted_df['Name'] = df.get('OperatorPlayerName', df.get('PlayerName', ''))
    formatted_df['Position'] = df.get('OperatorPosition', '')
    formatted_df['Team'] = df.get('Team', '')
    formatted_df['Salary'] = df.get('OperatorSalary', 0)
    formatted_df['OperatorPlayerID'] = df.get('OperatorPlayerID', '')
    formatted_df['PlayerID'] = df.get('PlayerID', '')
    formatted_df['RosterSlots'] = df.get('OperatorRosterSlots', [])
    
    # Add projections
    if projections:
        proj_df = pd.DataFrame(projections)
        
        # Merge projections
        if 'PlayerID' in proj_df.columns:
            merged = formatted_df.merge(
                proj_df[['PlayerID', 'FantasyPoints']],
                on='PlayerID',
                how='left'
            )
            formatted_df['Predicted_DK_Points'] = merged['FantasyPoints'].fillna(0)
        else:
            # Fallback to name matching
            proj_df['MatchName'] = proj_df['Name'].str.strip().str.upper()
            formatted_df['MatchName'] = formatted_df['Name'].str.strip().str.upper()
            
            merged = formatted_df.merge(
                proj_df[['MatchName', 'FantasyPoints']],
                on='MatchName',
                how='left'
            )
            formatted_df['Predicted_DK_Points'] = merged['FantasyPoints'].fillna(0)
            formatted_df = formatted_df.drop(columns=['MatchName'])
    else:
        formatted_df['Predicted_DK_Points'] = 0
    
    # 🏈 CRITICAL: Handle DST projections specifically
    dst_players = formatted_df[formatted_df['Position'].isin(['DST', 'DEF', 'D'])]
    if len(dst_players) > 0:
        print(f"🏈 Processing {len(dst_players)} DST players...")
        
        # For DST players without projections, calculate estimated projections
        dst_no_proj = dst_players[dst_players['Predicted_DK_Points'] == 0]
        if len(dst_no_proj) > 0:
            print(f"   📊 {len(dst_no_proj)} DST players need estimated projections")
            
            # Calculate DST projections based on salary tiers
            for idx, dst_player in dst_no_proj.iterrows():
                salary = dst_player['Salary']
                
                # DST projection estimation based on salary
                if salary >= 4000:  # Premium DST
                    proj = 8.5
                elif salary >= 3000:  # Mid-tier DST
                    proj = 6.5
                elif salary >= 2000:  # Budget DST
                    proj = 4.5
                else:  # Minimum salary DST
                    proj = 3.0
                
                formatted_df.loc[idx, 'Predicted_DK_Points'] = proj
                print(f"   🎯 {dst_player['Name']}: ${salary} → {proj:.1f} pts")
        
        # Ensure all DST have projections > 0
        dst_zero_proj = formatted_df[(formatted_df['Position'].isin(['DST', 'DEF', 'D'])) & (formatted_df['Predicted_DK_Points'] <= 0)]
        if len(dst_zero_proj) > 0:
            print(f"   ⚠️ Setting minimum 3.0 projections for {len(dst_zero_proj)} DST players")
            formatted_df.loc[dst_zero_proj.index, 'Predicted_DK_Points'] = 3.0
    
    # Position mapping
    position_map = {'DEF': 'DST', 'D': 'DST'}
    formatted_df['Position'] = formatted_df['Position'].replace(position_map)
    
    # Filter to valid positions
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    formatted_df = formatted_df[formatted_df['Position'].isin(valid_positions)]
    
    # Data quality checks
    formatted_df = formatted_df.dropna(subset=['Name', 'Position', 'Salary'])
    formatted_df = formatted_df[formatted_df['Salary'] > 0]
    
    # Add optimizer columns
    formatted_df['Opponent'] = 'TBD'
    formatted_df['InjuryStatus'] = 'Active'
    formatted_df['GameInfo'] = 'TBD'
    
    # Calculate value
    formatted_df['Value'] = formatted_df.apply(
        lambda row: row['Predicted_DK_Points'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0,
        axis=1
    )
    formatted_df['PointsPerK'] = formatted_df['Value']
    
    # Add all optimizer columns (matching working format)
    formatted_df['PassingYards'] = 0.0
    formatted_df['PassingTouchdowns'] = 0.0
    formatted_df['PassingInterceptions'] = 0.0
    formatted_df['RushingYards'] = 0.0
    formatted_df['RushingTouchdowns'] = 0.0
    formatted_df['ReceivingYards'] = 0.0
    formatted_df['ReceivingTouchdowns'] = 0.0
    formatted_df['Receptions'] = 0.0
    formatted_df['ReceivingTargets'] = 0.0
    formatted_df['FantasyPointsYahoo'] = 0.0
    formatted_df['FantasyPointsFanDuel'] = 0.0
    formatted_df['Ceiling'] = formatted_df['Predicted_DK_Points'] * 1.3
    formatted_df['Floor'] = formatted_df['Predicted_DK_Points'] * 0.6
    formatted_df['ID'] = formatted_df['OperatorPlayerID']
    formatted_df['Sacks'] = 0.0
    formatted_df['Interceptions'] = 0.0
    formatted_df['FumblesRecovered'] = 0.0
    formatted_df['Touchdowns'] = 0.0
    formatted_df['PointsAllowed'] = 0.0
    formatted_df['Cash_Score'] = formatted_df['Predicted_DK_Points'] * 0.8
    
    print(f"✅ Formatted {len(formatted_df)} players")
    
    # Show summary
    print(f"\n📊 Data Summary:")
    print(f"   Total Players: {len(formatted_df)}")
    print(f"   Players with Projections: {len(formatted_df[formatted_df['Predicted_DK_Points'] > 0])}")
    print(f"   Position breakdown: {formatted_df['Position'].value_counts().to_dict()}")
    print(f"   Salary Range: ${formatted_df['Salary'].min():,} - ${formatted_df['Salary'].max():,}")
    
    # Show top projected players
    print(f"\n🏆 Top 10 Projected Players:")
    top_proj = formatted_df.nlargest(10, 'Predicted_DK_Points')[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Value']]
    print(top_proj.to_string(index=False))
    
    return formatted_df

def main():
    """
    Main function - asks for user input
    """
    print("🏈 NFL API to DraftKings Formatter")
    print("="*50)
    
    # Get user input
    date = input("Enter date (YYYY-MM-DD) [2025-10-26]: ").strip()
    if not date:
        date = "2025-10-26"
    
    week = input("Enter week number [8]: ").strip()
    if not week:
        week = 8
    else:
        week = int(week)
    
    season = input("Enter season [2025REG]: ").strip()
    if not season:
        season = "2025REG"
    
    # Format the data
    df = format_nfl_data_for_optimizer(date, week, season)
    
    if df is not None:
        # Save the file
        output_filename = f"nfl_week{week}_optimizer_ready_{date.replace('-', '')}.csv"
        df.to_csv(output_filename, index=False)
        
        print(f"\n💾 Saved to: {output_filename}")
        print(f"\n✅ Ready for your optimizer!")
        print(f"🎯 Load this file into your genetic algorithm optimizer")
    else:
        print("❌ Failed to format data")

if __name__ == "__main__":
    main()
