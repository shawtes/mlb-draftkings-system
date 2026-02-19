#!/usr/bin/env python3
"""
Diagnose Duplicate Lineups Issue
================================

This script will help identify why you're still getting duplicate lineups.
"""

import sys
import os
sys.path.insert(0, '.')

from optimizer01 import optimize_single_lineup
import pandas as pd
import numpy as np
import random
import time

def diagnose_duplicates():
    """Diagnose what's causing duplicate lineups"""
    
    print("🔍 DIAGNOSING DUPLICATE LINEUPS ISSUE")
    print("=" * 50)
    
    # Load CSV data - find the correct player data file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'backup' not in f]
    if not csv_files:
        print("❌ No CSV files found!")
        return
    
    # Find the correct player data file by checking columns
    player_data_file = None
    
    print(f"📊 Checking {len(csv_files)} CSV files for player data...")
    
    for csv_file in csv_files:
        try:
            df_test = pd.read_csv(csv_file)
            print(f"   {csv_file}: {list(df_test.columns)[:5]}...")
            
            # Check if this looks like player data (has Name, Team, Position/Pos, Salary)
            has_name = any(col in df_test.columns for col in ['Name', 'Player', 'name', 'player'])
            has_team = any(col in df_test.columns for col in ['Team', 'team', 'Club'])
            has_position = any(col in df_test.columns for col in ['Position', 'Pos', 'position', 'pos'])
            has_salary = any(col in df_test.columns for col in ['Salary', 'salary', 'Cost'])
            
            if has_name and has_team and has_position and has_salary:
                player_data_file = csv_file
                print(f"   ✅ Found player data file: {csv_file}")
                break
            else:
                print(f"   ❌ {csv_file} - Not player data (missing required columns)")
                
        except Exception as e:
            print(f"   ❌ {csv_file} - Error reading: {e}")
    
    if not player_data_file:
        print("❌ No suitable player data file found!")
        print("Expected columns: Name, Team, Position (or Pos), Salary")
        return
    
    print(f"🎯 Using player data file: {player_data_file}")
    df = pd.read_csv(player_data_file)
    
    # Apply column mapping if needed
    if 'Pos' in df.columns and 'Position' not in df.columns:
        df = df.rename(columns={'Pos': 'Position'})
        print("🔧 Renamed 'Pos' to 'Position'")
    if 'Predicted_Points' in df.columns and 'Predicted_DK_Points' not in df.columns:
        df = df.rename(columns={'Predicted_Points': 'Predicted_DK_Points'})
        print("🔧 Renamed 'Predicted_Points' to 'Predicted_DK_Points'")
    
    # Find prediction column if 'Predicted_DK_Points' doesn't exist
    if 'Predicted_DK_Points' not in df.columns:
        prediction_columns = ['My_Proj', 'ML_Prediction', 'PPG_Projection', 'Projection', 'Points']
        prediction_col = None
        
        for col in prediction_columns:
            if col in df.columns:
                prediction_col = col
                break
        
        if prediction_col:
            df = df.rename(columns={prediction_col: 'Predicted_DK_Points'})
            print(f"🔧 Using '{prediction_col}' as prediction column")
        else:
            print("❌ No prediction column found! Need points/projections for optimization.")
            return
    
    print(f"✅ Loaded {len(df)} players")
    print(f"📋 Final columns: {list(df.columns)}")
    
    # Test parameters
    team_projected_runs = {team: 5.0 for team in df['Team'].unique()}
    team_selections = None
    min_salary = 0
    
    print(f"\n🎯 TESTING LINEUP GENERATION...")
    print(f"Using {len(df)} players from {len(df['Team'].unique())} teams")
    
    # Generate multiple lineups and check for duplicates
    lineups = []
    lineup_hashes = set()
    
    num_tests = 10
    for i in range(num_tests):
        print(f"\n--- Test {i+1}/{num_tests} ---")
        
        # Test with different stack types
        stack_types = ["No Stacks", "4|2", "5|3"]
        
        for stack_type in stack_types:
            print(f"Testing {stack_type}...")
            
            # Generate lineup
            args = (df.copy(), stack_type, team_projected_runs, team_selections, min_salary)
            lineup, returned_stack = optimize_single_lineup(args)
            
            if lineup.empty:
                print(f"  ❌ {stack_type} failed to generate lineup")
                continue
                
            # Create lineup signature
            player_names = sorted(lineup['Name'].tolist())
            lineup_hash = '|'.join(player_names)
            
            # Check for duplicates
            if lineup_hash in lineup_hashes:
                print(f"  ❌ {stack_type} generated DUPLICATE lineup!")
                print(f"     Players: {player_names[:3]}...")
            else:
                print(f"  ✅ {stack_type} generated UNIQUE lineup")
                print(f"     Players: {player_names[:3]}...")
                lineup_hashes.add(lineup_hash)
                
            lineups.append({
                'stack_type': stack_type,
                'lineup': lineup,
                'hash': lineup_hash,
                'points': lineup['Predicted_DK_Points'].sum(),
                'salary': lineup['Salary'].sum()
            })
    
    # Analysis
    print(f"\n📊 DUPLICATE ANALYSIS:")
    print(f"Total lineups generated: {len(lineups)}")
    print(f"Unique lineups: {len(lineup_hashes)}")
    print(f"Duplicate lineups: {len(lineups) - len(lineup_hashes)}")
    
    if len(lineup_hashes) < len(lineups):
        print(f"\n⚠️ DUPLICATES FOUND! Here's the breakdown:")
        
        # Group by hash to find duplicates
        hash_counts = {}
        for lineup_data in lineups:
            hash_key = lineup_data['hash']
            if hash_key not in hash_counts:
                hash_counts[hash_key] = []
            hash_counts[hash_key].append(lineup_data)
        
        # Show duplicates
        for hash_key, lineup_list in hash_counts.items():
            if len(lineup_list) > 1:
                print(f"\n🔄 Duplicate lineup found {len(lineup_list)} times:")
                for i, lineup_data in enumerate(lineup_list):
                    print(f"   {i+1}. {lineup_data['stack_type']}: {lineup_data['points']:.1f} pts, ${lineup_data['salary']:.0f}")
    
    # Test randomness
    print(f"\n🎲 TESTING RANDOMNESS...")
    random_test_results = []
    
    for i in range(5):
        # Force different random seeds
        np.random.seed(int(time.time() * 1000000) % 2147483647 + i)
        random.seed(int(time.time() * 1000000) % 2147483647 + i)
        
        args = (df.copy(), "No Stacks", team_projected_runs, team_selections, min_salary)
        lineup, _ = optimize_single_lineup(args)
        
        if not lineup.empty:
            lineup_hash = '|'.join(sorted(lineup['Name'].tolist()))
            random_test_results.append(lineup_hash)
    
    unique_random = len(set(random_test_results))
    print(f"Random test: {unique_random}/5 unique lineups")
    
    if unique_random < 5:
        print(f"⚠️ RANDOMNESS ISSUE: Even with different seeds, getting duplicates")
        print(f"This suggests the optimization problem is too constrained")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if len(lineup_hashes) < len(lineups):
        print("1. 🔧 INCREASE DIVERSITY:")
        print("   - Set Min Unique to 4-6 players")
        print("   - Use more stack types (4|2, 5|3, 4|2|2, 3|3|2)")
        print("   - Check 'Disable Kelly Sizing' if available")
        
        print("\n2. 📊 CHECK DATA QUALITY:")
        print("   - Ensure you have enough players in each position")
        print("   - Verify salary ranges allow for variety")
        
        print("\n3. ⚙️ OPTIMIZER SETTINGS:")
        print("   - Generate more candidates (100+ lineups)")
        print("   - Use different min_salary settings")
        print("   - Try different team combinations")
    
    return len(lineup_hashes) >= len(lineups) * 0.8  # 80% unique is acceptable

if __name__ == "__main__":
    success = diagnose_duplicates()
    if success:
        print("\n🎯 DUPLICATE DIAGNOSIS COMPLETE - MOSTLY UNIQUE")
    else:
        print("\n⚠️ DUPLICATE DIAGNOSIS COMPLETE - SIGNIFICANT DUPLICATES FOUND")
        print("Apply the recommendations above to improve diversity.") 