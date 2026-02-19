#!/usr/bin/env python3
"""
Filter NBA data with REAL DraftKings salaries
Processes both projection and salary data
"""

import pandas as pd
from datetime import datetime

def process_nba_with_salaries():
    """Process NBA data combining projections and salaries"""
    
    print(f"\n{'='*60}")
    print(f"🏀 Processing NBA Data with Real Salaries")
    print(f"{'='*60}\n")
    
    # Read the projection data
    proj_file = "nba_tonight_6pm_2025OCT26.csv"
    print(f"📊 Reading projections: {proj_file}")
    proj_df = pd.read_csv(proj_file)
    print(f"✅ Loaded {len(proj_df)} player records")
    
    # Read the DraftKings salary data
    salary_file = "nba_draftkings_tonight_2025OCT26.csv"
    print(f"\n💰 Reading salaries: {salary_file}")
    
    try:
        salary_df = pd.read_csv(salary_file)
        print(f"✅ Loaded {len(salary_df)} DraftKings player records")
    except Exception as e:
        print(f"❌ Could not load salary file: {e}")
        print("⚠️ Will use projection data without real salaries")
        return process_without_salaries(proj_df)
    
    # Map columns
    salary_map = {
        'OperatorPlayerName': 'Name',
        'OperatorPosition': 'Position', 
        'OperatorSalary': 'Salary',
        'Team': 'Team'
    }
    
    print(f"\n📋 Salary file columns: {list(salary_df.columns)}")
    
    # Check for UTIL positions and extract base position
    if 'OperatorPosition' in salary_df.columns:
        print(f"\n🔍 Found positions column, checking for UTIL...")
        # Extract base position (PG/SG -> PG for eligibility)
        salary_df['Position'] = salary_df['OperatorPosition'].str.split('/').str[0]
        print(f"   Sample positions: {salary_df['Position'].unique()[:10]}")
    
    # Create optimized dataframe
    optimized_df = pd.DataFrame()
    
    # Player names
    if 'Name' in proj_df.columns:
        optimized_df['Name'] = proj_df['Name']
    else:
        print("❌ Missing 'Name' in projection data")
        return None
    
    # Positions
    if 'Position' in proj_df.columns:
        optimized_df['Position'] = proj_df['Position']
    
    # Teams
    if 'Team' in proj_df.columns:
        optimized_df['Team'] = proj_df['Team']
    
    # Opponent and game info
    if 'Opponent' in proj_df.columns:
        optimized_df['Opponent'] = proj_df['Opponent']
    
    if 'HomeOrAway' in proj_df.columns:
        optimized_df['HomeOrAway'] = proj_df['HomeOrAway']
        # Create Game string
        optimized_df['Game'] = optimized_df.apply(
            lambda x: f"{x['Team']}@{x['Opponent']}" if x.get('HomeOrAway') == 'AWAY'
            else f"{x['Opponent']}@{x['Team']}", axis=1
        )
    
    if 'GameID' in proj_df.columns:
        optimized_df['GameID'] = proj_df['GameID']
    
    # Projected DK Points
    if 'FantasyPointsDraftKings' in proj_df.columns:
        optimized_df['Predicted_DK_Points'] = proj_df['FantasyPointsDraftKings']
    elif 'Predicted_DK_Points' in proj_df.columns:
        optimized_df['Predicted_DK_Points'] = proj_df['Predicted_DK_Points']
    else:
        print("❌ No fantasy points found")
        return None
    
    # Merge with salary data
    print(f"\n🔗 Merging salary data...")
    
    # Create a lookup for salaries by name and team
    salary_lookup = {}
    
    if 'OperatorPlayerName' in salary_df.columns and 'OperatorSalary' in salary_df.columns:
        for _, row in salary_df.iterrows():
            name = row.get('OperatorPlayerName', '').strip()
            salary = row.get('OperatorSalary', 0)
            team = row.get('Team', '')
            
            if pd.notna(salary) and salary > 0:
                key = f"{name}_{team}"
                salary_lookup[key] = int(salary)
        
        print(f"✅ Created salary lookup with {len(salary_lookup)} players")
    
    # Apply salaries
    optimized_df['Salary'] = optimized_df.apply(
        lambda row: salary_lookup.get(f"{row.get('Name', '')}_{row.get('Team', '')}", 5000),
        axis=1
    )
    
    # Count how many got real salaries
    real_salary_count = (optimized_df['Salary'] != 5000).sum()
    print(f"✅ Applied real salaries to {real_salary_count} out of {len(optimized_df)} players")
    
    if real_salary_count == 0:
        print("⚠️ No real salaries found, using defaults")
        optimized_df['Salary'] = 5000
    else:
        print(f"💰 Salary range: ${optimized_df['Salary'].min():,} - ${optimized_df['Salary'].max():,}")
    
    # Filter valid players
    optimized_df = optimized_df[
        (optimized_df['Predicted_DK_Points'] > 0) &
        (optimized_df['Salary'] > 0)
    ].drop_duplicates(subset=['Name', 'Team'], keep='first')
    
    print(f"\n✅ Final processed players: {len(optimized_df)}")
    
    # Show game count
    if 'GameID' in optimized_df.columns:
        unique_games = optimized_df['GameID'].dropna().unique()
        print(f"📊 Games: {len(unique_games)}")
    
    # Show position breakdown
    if 'Position' in optimized_df.columns:
        print(f"\n📊 Position Breakdown:")
        pos_counts = optimized_df['Position'].value_counts().sort_index()
        for pos, count in pos_counts.items():
            print(f"   {pos}: {count}")
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"nba_9game_slate_with_salaries_{timestamp}.csv"
    optimized_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("✅ Ready for genetic optimizer!")
    print(f"{'='*60}\n")
    
    return optimized_df


def process_without_salaries(proj_df):
    """Fallback: process without salary data"""
    print("\n⚠️ Processing without salary data...")
    
    optimized_df = pd.DataFrame()
    
    if 'Name' in proj_df.columns:
        optimized_df['Name'] = proj_df['Name']
    if 'Position' in proj_df.columns:
        optimized_df['Position'] = proj_df['Position']
    if 'Team' in proj_df.columns:
        optimized_df['Team'] = proj_df['Team']
    
    if 'FantasyPointsDraftKings' in proj_df.columns:
        optimized_df['Predicted_DK_Points'] = proj_df['FantasyPointsDraftKings']
    
    optimized_df['Salary'] = 5000  # Default
    
    return optimized_df


if __name__ == "__main__":
    df = process_nba_with_salaries()
    
    if df is not None and not df.empty:
        print(f"\n📋 Sample (first 5 rows):")
        cols = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']
        print(df[cols].head())
        print(f"\n📊 Total: {len(df)} players")









