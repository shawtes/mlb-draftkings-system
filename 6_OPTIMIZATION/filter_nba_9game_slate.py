#!/usr/bin/env python3
"""
Filter and prepare NBA data for 9-game slate
Processes the nba_tonight_6pm_2025OCT26.csv file
"""

import pandas as pd
from datetime import datetime

def process_nba_slate(input_file, output_file=None):
    """
    Process NBA slate data for optimizer
    
    Args:
        input_file: Input CSV file
        output_file: Output filename (optional)
    """
    print(f"\n{'='*60}")
    print(f"🏀 Processing NBA Slate: {input_file}")
    print(f"{'='*60}\n")
    
    # Read the file
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} player records")
    
    # Check game count
    if 'GameID' in df.columns:
        unique_games = df['GameID'].dropna().unique()
        num_games = len(unique_games)
        print(f"📊 Found {num_games} games")
    
    # Show current structure
    print(f"\n📋 Available columns: {list(df.columns)[:15]}")
    
    # Process for optimizer format
    # Map columns to optimizer requirements
    optimized_df = pd.DataFrame()
    
    # Player info
    if 'Name' in df.columns:
        optimized_df['Name'] = df['Name']
    else:
        print("❌ Missing 'Name' column")
    
    if 'Position' in df.columns:
        optimized_df['Position'] = df['Position']
    else:
        print("❌ Missing 'Position' column")
    
    if 'Team' in df.columns:
        optimized_df['Team'] = df['Team']
    else:
        print("❌ Missing 'Team' column")
    
    # Salary (OperatorSalary in some formats)
    if 'Salary' in df.columns:
        optimized_df['Salary'] = df['Salary']
    elif 'OperatorSalary' in df.columns:
        optimized_df['Salary'] = df['OperatorSalary']
    else:
        print("⚠️ No salary column found, using defaults")
        optimized_df['Salary'] = 5000  # Default
    
    # Projected points
    if 'Predicted_DK_Points' in df.columns:
        optimized_df['Predicted_DK_Points'] = df['Predicted_DK_Points']
    elif 'FantasyPointsDraftKings' in df.columns:
        optimized_df['Predicted_DK_Points'] = df['FantasyPointsDraftKings']
    elif 'Fantasy_Points' in df.columns:
        optimized_df['Predicted_DK_Points'] = df['Fantasy_Points']
    else:
        print("⚠️ No predicted points found, calculating...")
        # Calculate DK points from stats
        optimized_df['Predicted_DK_Points'] = calculate_dk_points(df)
    
    # Game information
    if 'GameID' in df.columns:
        optimized_df['GameID'] = df['GameID']
    
    if 'Opponent' in df.columns:
        optimized_df['Opponent'] = df['Opponent']
    
    if 'HomeOrAway' in df.columns:
        optimized_df['HomeOrAway'] = df['HomeOrAway']
        
        # Create Game string
        optimized_df['Game'] = optimized_df.apply(
            lambda x: f"{x['Team']}@{x['Opponent']}" if x.get('HomeOrAway') == 'AWAY'
            else f"{x['Opponent']}@{x['Team']}", axis=1
        )
    
    # Remove duplicates
    optimized_df = optimized_df.drop_duplicates(subset=['Name', 'Team'], keep='first')
    
    # Remove players with 0 projected points or no salary
    optimized_df = optimized_df[optimized_df['Predicted_DK_Points'] > 0]
    optimized_df = optimized_df[optimized_df['Salary'] > 0]
    
    print(f"\n✅ Processed {len(optimized_df)} unique players")
    
    # Show game count after processing
    if 'GameID' in optimized_df.columns:
        unique_games = optimized_df['GameID'].dropna().unique()
        num_games = len(unique_games)
        print(f"📊 {num_games} games after filtering")
    
    # Show position breakdown
    if 'Position' in optimized_df.columns:
        print(f"\n📊 Position Breakdown:")
        pos_counts = optimized_df['Position'].value_counts().sort_index()
        for pos, count in pos_counts.items():
            print(f"   {pos}: {count}")
    
    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"nba_9game_slate_{timestamp}.csv"
    
    # Save
    optimized_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("✅ Ready for genetic optimizer!")
    print(f"   Load file: {output_file}")
    print(f"{'='*60}\n")
    
    return optimized_df


def calculate_dk_points(df):
    """
    Calculate DK points from stat columns
    DK Scoring: Points=1, 3PM=0.5, Reb=1.25, Ast=1.5, Stl=2, Blk=2, TO=-0.5
    """
    if 'FantasyPointsDraftKings' in df.columns:
        return df['FantasyPointsDraftKings']
    
    # Try to calculate from stats
    dk_points = pd.Series([0.0] * len(df))
    
    if 'Points' in df.columns:
        dk_points += df['Points'] * 1.0
    
    if 'ThreePointersHitMade' in df.columns:
        dk_points += df['ThreePointersHitMade'] * 0.5
    
    if 'Rebounds' in df.columns:
        dk_points += df['Rebounds'] * 1.25
    
    if 'Assists' in df.columns:
        dk_points += df['Assists'] * 1.5
    
    if 'Steals' in df.columns:
        dk_points += df['Steals'] * 2.0
    
    if 'BlockedShots' in df.columns:
        dk_points += df['BlockedShots'] * 2.0
    
    if 'Turnovers' in df.columns:
        dk_points -= df['Turnovers'] * 0.5
    
    return dk_points.fillna(0)


if __name__ == "__main__":
    # Process the current slate file
    input_file = "nba_tonight_6pm_2025OCT26.csv"
    
    df = process_nba_slate(input_file, output_file="nba_9game_slate_ready.csv")
    
    if df is not None and not df.empty:
        print(f"\n📋 Sample (first 5 rows):")
        cols = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']
        print(df[cols].head())
        print(f"\n📊 Total: {len(df)} players")





