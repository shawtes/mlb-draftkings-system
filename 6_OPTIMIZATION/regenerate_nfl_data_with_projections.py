#!/usr/bin/env python3
"""
Quick script to regenerate NFL data with all detailed projections
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def regenerate_nfl_data():
    """Regenerate NFL data with all detailed projections"""
    
    print("🔄 Regenerating NFL data with detailed projections...")
    
    # Initialize API
    api = SportsDataNFLAPI("1dd5e646265649af87e0d9cdb80d1c8c")
    
    try:
        # Fetch projections for Week 8
        print("📊 Fetching NFL projections...")
        projections = api.get_player_projections_by_week("2025REG", 8)
        
        if not projections:
            print("❌ No projections found")
            return
        
        print(f"✅ Found {len(projections)} projections")
        
        # Convert to DataFrame
        proj_df = pd.DataFrame(projections)
        print(f"📋 Projection columns: {list(proj_df.columns)}")
        
        # Load existing data
        existing_file = "nfl_week8_DST_FIXED.csv"
        if os.path.exists(existing_file):
            df = pd.read_csv(existing_file)
            print(f"📁 Loaded existing data: {len(df)} players")
        else:
            print("❌ No existing data file found")
            return
        
        # Merge projections with existing data
        print("🔗 Merging projections with existing data...")
        
        # Try to merge by PlayerID first
        if 'PlayerID' in proj_df.columns and 'PlayerID' in df.columns:
            merged = df.merge(proj_df, on='PlayerID', how='left', suffixes=('', '_proj'))
            print("✅ Merged by PlayerID")
        else:
            # Fallback to name matching
            proj_df['MatchName'] = proj_df['Name'].str.strip().str.upper()
            df['MatchName'] = df['Name'].str.strip().str.upper()
            
            merged = df.merge(proj_df, on='MatchName', how='left', suffixes=('', '_proj'))
            merged = merged.drop(columns=['MatchName'])
            print("✅ Merged by Name")
        
        # Update projection columns
        projection_columns = [
            'PassingYards', 'PassingTouchdowns', 'PassingInterceptions',
            'RushingYards', 'RushingTouchdowns', 'ReceivingYards', 'ReceivingTouchdowns',
            'Receptions', 'ReceivingTargets', 'Sacks', 'Interceptions', 'FumblesRecovered',
            'Touchdowns', 'PointsAllowed'
        ]
        
        for col in projection_columns:
            if f'{col}_proj' in merged.columns:
                # Use projection data where available
                merged[col] = merged[f'{col}_proj'].fillna(merged[col])
                merged = merged.drop(columns=[f'{col}_proj'])
            elif col not in merged.columns:
                # Initialize with 0 if column doesn't exist
                merged[col] = 0
        
        # Update FantasyPoints if available
        if 'FantasyPoints_proj' in merged.columns:
            merged['Predicted_DK_Points'] = merged['FantasyPoints_proj'].fillna(merged['Predicted_DK_Points'])
            merged = merged.drop(columns=['FantasyPoints_proj'])
        
        # 🎯 ESTIMATE individual stat projections from DK points if still 0
        print("🎯 Estimating remaining projections from DK points...")
        
        for idx, player in merged.iterrows():
            dk_points = player.get('Predicted_DK_Points', 0)
            position = player.get('Position', '')
            
            if dk_points > 0:
                # QB projections
                if position == 'QB':
                    if merged.loc[idx, 'PassingYards'] == 0:
                        merged.loc[idx, 'PassingYards'] = dk_points * 25
                    if merged.loc[idx, 'PassingTouchdowns'] == 0:
                        merged.loc[idx, 'PassingTouchdowns'] = max(0.5, dk_points / 4)
                    if merged.loc[idx, 'RushingYards'] == 0:
                        merged.loc[idx, 'RushingYards'] = dk_points * 2
                    if merged.loc[idx, 'RushingTouchdowns'] == 0:
                        merged.loc[idx, 'RushingTouchdowns'] = max(0.5, dk_points / 8)
                
                # RB projections
                elif position == 'RB':
                    if merged.loc[idx, 'RushingYards'] == 0:
                        merged.loc[idx, 'RushingYards'] = dk_points * 8
                    if merged.loc[idx, 'RushingTouchdowns'] == 0:
                        merged.loc[idx, 'RushingTouchdowns'] = max(0.5, dk_points / 6)
                    if merged.loc[idx, 'ReceivingYards'] == 0:
                        merged.loc[idx, 'ReceivingYards'] = dk_points * 3
                    if merged.loc[idx, 'Receptions'] == 0:
                        merged.loc[idx, 'Receptions'] = max(0.5, dk_points / 2)
                
                # WR projections
                elif position == 'WR':
                    if merged.loc[idx, 'ReceivingYards'] == 0:
                        merged.loc[idx, 'ReceivingYards'] = dk_points * 6
                    if merged.loc[idx, 'ReceivingTouchdowns'] == 0:
                        merged.loc[idx, 'ReceivingTouchdowns'] = max(0.5, dk_points / 6)
                    if merged.loc[idx, 'Receptions'] == 0:
                        merged.loc[idx, 'Receptions'] = max(0.5, dk_points / 1.5)
                    if merged.loc[idx, 'ReceivingTargets'] == 0:
                        merged.loc[idx, 'ReceivingTargets'] = max(1, dk_points)
                
                # TE projections
                elif position == 'TE':
                    if merged.loc[idx, 'ReceivingYards'] == 0:
                        merged.loc[idx, 'ReceivingYards'] = dk_points * 5
                    if merged.loc[idx, 'ReceivingTouchdowns'] == 0:
                        merged.loc[idx, 'ReceivingTouchdowns'] = max(0.5, dk_points / 6)
                    if merged.loc[idx, 'Receptions'] == 0:
                        merged.loc[idx, 'Receptions'] = max(0.5, dk_points / 2)
                    if merged.loc[idx, 'ReceivingTargets'] == 0:
                        merged.loc[idx, 'ReceivingTargets'] = max(1, dk_points * 1.2)
                
                # DST projections
                elif position in ['DST', 'DEF', 'D']:
                    if merged.loc[idx, 'Sacks'] == 0:
                        merged.loc[idx, 'Sacks'] = max(0.5, dk_points / 2)
                    if merged.loc[idx, 'Interceptions'] == 0:
                        merged.loc[idx, 'Interceptions'] = max(0.5, dk_points / 3)
                    if merged.loc[idx, 'FumblesRecovered'] == 0:
                        merged.loc[idx, 'FumblesRecovered'] = max(0.5, dk_points / 4)
                    if merged.loc[idx, 'Touchdowns'] == 0:
                        merged.loc[idx, 'Touchdowns'] = max(0.5, dk_points / 6)
                    if merged.loc[idx, 'PointsAllowed'] == 0:
                        merged.loc[idx, 'PointsAllowed'] = max(10, 30 - dk_points * 2)
        
        # Save the updated data
        output_file = "nfl_week8_WITH_PROJECTIONS.csv"
        merged.to_csv(output_file, index=False)
        
        print(f"✅ Saved updated data to {output_file}")
        print(f"📊 Final data shape: {merged.shape}")
        
        # Show sample of projections
        print("\n📈 Sample projections:")
        sample_cols = ['Name', 'Position', 'Predicted_DK_Points', 'PassingYards', 'RushingYards', 'ReceivingYards', 'Receptions']
        available_cols = [col for col in sample_cols if col in merged.columns]
        print(merged[available_cols].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    regenerate_nfl_data()
