#!/usr/bin/env python3
"""
Load NFL Data from SportsData.io for Genetic Algorithm Optimizer
================================================================

This script fetches NFL data from SportsData.io and formats it
for use with the genetic_algo_nfl_optimizer.py

It creates a CSV file compatible with the optimizer's expected format.
"""

import sys
import os
sys.path.append('../python_algorithms')

from sportsdata_nfl_api import SportsDataNFLAPI
import pandas as pd
import numpy as np


def load_nfl_data_for_optimizer(
    api_key: str,
    date: str,
    season: str = None,
    week: int = None,
    use_projections: bool = True,
    output_filename: str = None,
    operator: str = "DraftKings"
) -> pd.DataFrame:
    """
    Fetch NFL data from SportsData.io and format for genetic optimizer
    Combines DFS slate data (for salaries) with projections
    
    Args:
        api_key: SportsData.io API key
        date: Date for DFS slates (e.g., "2025-10-20")
        season: Season (e.g., "2025REG") - for projections
        week: Week number - for projections
        use_projections: If True, use projections. If False, use actual stats.
        output_filename: Custom output filename
        operator: DFS operator (default "DraftKings")
    
    Returns:
        DataFrame formatted for the genetic optimizer
    """
    print("="*70)
    print("NFL DATA LOADER FOR GENETIC ALGORITHM OPTIMIZER")
    print("="*70)
    
    api = SportsDataNFLAPI(api_key)
    
    # Step 1: Fetch DFS slate data (contains actual DraftKings salaries)
    print(f"\n📊 Step 1: Fetching DFS Slates for {date}...")
    slate_data = api.get_dfs_slates_by_date(date, save_to_file=True)
    
    if not slate_data:
        print("❌ Failed to fetch DFS slate data")
        return None
    
    # Find the DraftKings slate
    dk_slate = None
    for slate in slate_data:
        if slate.get('Operator', '').upper() == operator.upper():
            dk_slate = slate
            break
    
    if not dk_slate:
        print(f"❌ No {operator} slate found for {date}")
        print(f"   Available operators: {[s.get('Operator') for s in slate_data]}")
        return None
    
    # Extract player salary data
    slate_players = dk_slate.get('DfsSlatePlayers', [])
    print(f"✅ Found {len(slate_players)} players in {operator} slate")
    
    # Create DataFrame from slate players (contains salaries)
    salary_df = pd.DataFrame(slate_players)
    
    # Step 2: Fetch projection data (if requested)
    projections_df = None
    if use_projections and season and week:
        print(f"\n📊 Step 2: Fetching PROJECTIONS for {season} Week {week}...")
        proj_data = api.get_player_projections_by_week(season, week)
        
        if proj_data:
            projections_df = pd.DataFrame(proj_data)
            print(f"✅ Loaded {len(projections_df)} player projections")
    
    # Step 3: Merge salary data with projections
    print(f"\n🔗 Step 3: Merging salary and projection data...")
    
    # Rename columns from DFS slate to match optimizer format
    optimizer_df = pd.DataFrame()
    
    # Map DFS slate columns (DraftKings uses 'Operator' prefix for their data)
    optimizer_df['Name'] = salary_df.get('OperatorPlayerName', salary_df.get('PlayerName', ''))
    optimizer_df['PlayerID'] = salary_df.get('PlayerID', '')
    optimizer_df['Position'] = salary_df.get('OperatorPosition', '')
    optimizer_df['Team'] = salary_df.get('Team', '')
    optimizer_df['Salary'] = salary_df.get('OperatorSalary', 0)
    optimizer_df['OperatorPlayerID'] = salary_df.get('OperatorPlayerID', '')
    optimizer_df['RosterSlots'] = salary_df.get('OperatorRosterSlots', [])
    
    # Merge with projections if available
    if projections_df is not None:
        # Merge on PlayerID or Name
        if 'PlayerID' in projections_df.columns:
            merged = optimizer_df.merge(
                projections_df[['PlayerID', 'FantasyPoints']],
                on='PlayerID',
                how='left'
            )
            optimizer_df['FantasyPoints'] = merged['FantasyPoints'].fillna(0)
        else:
            # Fallback to name matching
            projections_df['MatchName'] = projections_df['Name'].str.strip().str.upper()
            optimizer_df['MatchName'] = optimizer_df['Name'].str.strip().str.upper()
            
            merged = optimizer_df.merge(
                projections_df[['MatchName', 'FantasyPoints']],
                on='MatchName',
                how='left'
            )
            optimizer_df['FantasyPoints'] = merged['FantasyPoints'].fillna(0)
            optimizer_df = optimizer_df.drop(columns=['MatchName'])
        
        print(f"✅ Merged projections for {optimizer_df['FantasyPoints'].notna().sum()} players")
    else:
        # No projections available
        optimizer_df['FantasyPoints'] = 0
        print("⚠️  No projections available - FantasyPoints set to 0")
    
    # Ensure Position is properly formatted for NFL
    if 'Position' in optimizer_df.columns:
        position_map = {
            'DEF': 'DST',
            'D': 'DST',
        }
        optimizer_df['Position'] = optimizer_df['Position'].replace(position_map)
    
    # Filter to only valid positions for DraftKings Classic
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    before_filter = len(optimizer_df)
    optimizer_df = optimizer_df[optimizer_df['Position'].isin(valid_positions)]
    print(f"✅ Filtered to valid positions: {len(optimizer_df)}/{before_filter} players")
    
    # Remove players with missing critical data
    optimizer_df = optimizer_df.dropna(subset=['Name', 'Position', 'Salary'])
    optimizer_df = optimizer_df[optimizer_df['Salary'] > 0]
    
    print(f"✅ After data quality checks: {len(optimizer_df)} players")
    
    # Show position breakdown
    print("\n📊 Position Breakdown:")
    position_counts = optimizer_df['Position'].value_counts().sort_index()
    for pos, count in position_counts.items():
        avg_salary = optimizer_df[optimizer_df['Position'] == pos]['Salary'].mean()
        print(f"   {pos}: {count} players (Avg Salary: ${avg_salary:,.0f})")
    
    # Add value columns
    optimizer_df['Value'] = optimizer_df.apply(
        lambda row: row['FantasyPoints'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0,
        axis=1
    )
    optimizer_df['PointsPerK'] = optimizer_df['Value']
    
    # Save to CSV
    if output_filename is None:
        date_clean = date.replace('-', '')
        output_filename = f"nfl_optimizer_data_{date_clean}.csv"
    
    optimizer_df.to_csv(output_filename, index=False)
    print(f"\n💾 Saved optimizer-ready data to: {output_filename}")
    
    # Show sample data
    print("\n📋 Sample Data (Top 10 by salary):")
    sample = optimizer_df.nlargest(10, 'Salary')[
        ['Name', 'Position', 'Team', 'Salary', 'FantasyPoints', 'Value']
    ]
    print(sample.to_string(index=False))
    
    print("\n📋 Sample Data (Top 10 by projection):")
    sample2 = optimizer_df.nlargest(10, 'FantasyPoints')[
        ['Name', 'Position', 'Team', 'Salary', 'FantasyPoints', 'Value']
    ]
    print(sample2.to_string(index=False))
    
    return optimizer_df


def main():
    """
    Main function to load NFL data with DraftKings salaries
    """
    # API Key
    API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
    
    # Settings for upcoming week
    DATE = "2025-10-20"  # Sunday games - DFS slate date
    SEASON = "2025REG"
    WEEK = 7
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     NFL Data Loader for Genetic Algorithm Optimizer     ║
    ║              With DraftKings Salaries                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"📅 DFS Slate Date: {DATE}")
    print(f"📅 Season: {SEASON}")
    print(f"📅 Week: {WEEK}")
    print()
    
    # Load DFS slate data with projections
    print("Loading DraftKings salaries + projections...")
    df = load_nfl_data_for_optimizer(
        api_key=API_KEY,
        date=DATE,
        season=SEASON,
        week=WEEK,
        use_projections=True,
        output_filename=f"nfl_week{WEEK}_draftkings_optimizer.csv"
    )
    
    if df is not None:
        print("\n" + "="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print("\n📁 File ready for genetic optimizer:")
        print(f"   nfl_week{WEEK}_draftkings_optimizer.csv")
        print("\n🎯 Next steps:")
        print("   1. Open genetic_algo_nfl_optimizer.py")
        print("   2. Load the CSV file in the application")
        print("   3. Configure your settings (stacks, exposures, etc.)")
        print("   4. Click 'Generate Lineups'")
        print("\n💡 The data includes:")
        print(f"   • {len(df)} NFL players")
        print(f"   • Real DraftKings salaries from {DATE}")
        print(f"   • Fantasy point projections for Week {WEEK}")
        print(f"   • Positions: QB, RB, WR, TE, DST")
        print(f"   • Value calculations (Points per $1K)")


if __name__ == "__main__":
    main()

