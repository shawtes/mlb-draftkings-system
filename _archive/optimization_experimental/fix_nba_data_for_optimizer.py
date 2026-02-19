#!/usr/bin/env python3
"""
Fix NBA data for optimizer - Add required columns from projection data
"""

import pandas as pd
import numpy as np

def fix_nba_data():
    """Fix the NBA data file for the genetic optimizer"""
    
    print(f"\n{'='*60}")
    print(f"🏀 Fixing NBA Data for Optimizer")
    print(f"{'='*60}\n")
    
    # Read the existing file (check for Oct 27)
    input_file = "nba_tonight_6pm_2025OCT27.csv"
    if not pd.io.common.file_exists(input_file):
        input_file = "nba_tonight_6pm_2025OCT26.csv"
    print(f"📊 Reading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} records")
    
    # Check what we have
    print(f"\n📋 Available columns: {len(df.columns)}")
    print(f"   {list(df.columns)[:10]}")
    
    # Create optimized dataframe
    optimized = pd.DataFrame()
    
    # Required: Name
    if 'Name' in df.columns:
        optimized['Name'] = df['Name']
    else:
        print("❌ Missing Name column")
        return None
    
    # Required: Position
    if 'Position' in df.columns:
        optimized['Position'] = df['Position']
    else:
        print("❌ Missing Position column")
        return None
    
    # Required: Team
    if 'Team' in df.columns:
        optimized['Team'] = df['Team']
    else:
        print("❌ Missing Team column")
        return None
    
    # Required: Salary (we have to estimate or use defaults)
    if 'Salary' in df.columns:
        optimized['Salary'] = df['Salary']
    else:
        print("⚠️ No salary column found - estimating from fantasy points")
        # Estimate salary based on projection
        if 'FantasyPointsDraftKings' in df.columns:
            # Rough estimate: $1K per 5 fantasy points
            optimized['Salary'] = (df['FantasyPointsDraftKings'] / 5 * 1000).astype(int)
            # Cap between reasonable values
            optimized['Salary'] = optimized['Salary'].clip(3000, 12000)
        else:
            optimized['Salary'] = 6000  # Default
    
    # Required: Predicted_DK_Points
    if 'Predicted_DK_Points' in df.columns:
        optimized['Predicted_DK_Points'] = df['Predicted_DK_Points']
    elif 'FantasyPointsDraftKings' in df.columns:
        optimized['Predicted_DK_Points'] = df['FantasyPointsDraftKings']
    else:
        print("❌ No fantasy points found")
        return None
    
    # Optional: Game info
    if 'GameID' in df.columns:
        optimized['GameID'] = df['GameID']
    
    if 'Opponent' in df.columns and 'HomeOrAway' in df.columns:
        optimized['Opponent'] = df['Opponent']
        optimized['HomeOrAway'] = df['HomeOrAway']
        
        # Create Game string
        optimized['Game'] = optimized.apply(
            lambda x: f"{x['Team']}@{x['Opponent']}" if x['HomeOrAway'] == 'AWAY'
            else f"{x['Opponent']}@{x['Team']}", axis=1
        )
    
    # Filter valid players (remove zero points)
    optimized = optimized[optimized['Predicted_DK_Points'] > 0].copy()
    optimized = optimized[optimized['Salary'] > 0].copy()
    
    # Remove duplicates
    optimized = optimized.drop_duplicates(subset=['Name', 'Team'], keep='first')
    
    # Show game count
    if 'GameID' in optimized.columns:
        num_games = optimized['GameID'].nunique()
        print(f"\n📊 Games in slate: {num_games}")
    
    # Show position breakdown
    print(f"\n📊 Position Breakdown:")
    pos_counts = optimized['Position'].value_counts().sort_index()
    for pos, count in pos_counts.items():
        print(f"   {pos}: {count}")
    
    # Show salary range
    print(f"\n💰 Salary Range: ${optimized['Salary'].min():,} - ${optimized['Salary'].max():,}")
    print(f"   Average: ${optimized['Salary'].mean():.0f}")
    
    # Save
    output_file = "nba_tonight_FIXED.csv"
    optimized.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 Final players: {len(optimized)}")
    
    print(f"\n{'='*60}")
    print("✅ Data is ready for genetic optimizer!")
    print(f"   File: {output_file}")
    print(f"{'='*60}\n")
    
    # Show sample
    print(f"📋 Sample data:")
    print(optimized[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']].head(10))
    
    return optimized


if __name__ == "__main__":
    df = fix_nba_data()

