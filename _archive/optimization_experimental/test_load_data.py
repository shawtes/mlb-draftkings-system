#!/usr/bin/env python3
"""
Test script to diagnose why the optimizer isn't loading data correctly
"""

import pandas as pd
import sys

print("="*80)
print("🔍 TESTING DATA LOAD FOR OPTIMIZER")
print("="*80)

# Test file
test_file = 'nfl_week7_1PM_SLATE_READY_FOR_OPTIMIZER.csv'

print(f"\n📁 Testing file: {test_file}")

try:
    # Load the CSV
    print("\n1️⃣ Loading CSV...")
    df = pd.read_csv(test_file)
    print(f"   ✅ Loaded successfully")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check required columns
    print("\n2️⃣ Checking required columns...")
    required = ['Name', 'Team', 'Position', 'Salary', 'Predicted_DK_Points']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"   ❌ Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    else:
        print(f"   ✅ All required columns present")
    
    # Check for empty data
    print("\n3️⃣ Checking data quality...")
    
    if df.empty:
        print(f"   ❌ DataFrame is EMPTY!")
        sys.exit(1)
    else:
        print(f"   ✅ DataFrame has {len(df)} rows")
    
    # Check each position
    print("\n4️⃣ Checking position groups...")
    positions = df['Position'].value_counts()
    for pos, count in positions.items():
        print(f"   {pos:3s}: {count:3d} players")
    
    if len(positions) == 0:
        print(f"   ❌ No positions found!")
        sys.exit(1)
    
    # Check for NFL positions
    nfl_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    found_nfl = [pos for pos in nfl_positions if pos in positions.index]
    
    if len(found_nfl) < 5:
        print(f"   ⚠️  Warning: Only found {len(found_nfl)} NFL positions")
        print(f"   Found: {found_nfl}")
        print(f"   Missing: {[pos for pos in nfl_positions if pos not in found_nfl]}")
    else:
        print(f"   ✅ All 5 NFL positions found")
    
    # Check projections
    print("\n5️⃣ Checking projections...")
    proj_col = 'Predicted_DK_Points'
    
    if df[proj_col].isna().all():
        print(f"   ❌ All projections are NaN!")
        sys.exit(1)
    
    valid_projs = df[proj_col].notna().sum()
    print(f"   ✅ {valid_projs}/{len(df)} players have valid projections")
    
    if valid_projs > 0:
        print(f"   Range: {df[proj_col].min():.1f} - {df[proj_col].max():.1f} pts")
        print(f"   Average: {df[proj_col].mean():.1f} pts")
    
    # Check salaries
    print("\n6️⃣ Checking salaries...")
    
    if df['Salary'].isna().all():
        print(f"   ❌ All salaries are NaN!")
        sys.exit(1)
    
    valid_salaries = df['Salary'].notna().sum()
    print(f"   ✅ {valid_salaries}/{len(df)} players have valid salaries")
    
    if valid_salaries > 0:
        print(f"   Range: ${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}")
        print(f"   Average: ${df['Salary'].mean():,.0f}")
    
    # Show sample data
    print("\n7️⃣ Sample data (top 5 by projection):")
    print("-"*80)
    top5 = df.nlargest(5, 'Predicted_DK_Points')[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']]
    print(top5.to_string(index=False))
    
    # Test position filtering (like optimizer does)
    print("\n8️⃣ Testing position filtering (simulating optimizer)...")
    
    position_groups = {
        'All Offense': df[df['Position'] != 'DST'],
        'QB': df[df['Position'] == 'QB'],
        'RB': df[df['Position'] == 'RB'],
        'WR': df[df['Position'] == 'WR'],
        'TE': df[df['Position'] == 'TE'],
        'DST': df[df['Position'] == 'DST']
    }
    
    for pos_name, pos_df in position_groups.items():
        if pos_name == 'All Offense':
            print(f"   {pos_name:12s}: {len(pos_df):3d} players (all non-DST)")
        else:
            print(f"   {pos_name:12s}: {len(pos_df):3d} players")
    
    # Final check
    print(f"\n{'='*80}")
    print("✅ ALL CHECKS PASSED")
    print("="*80)
    
    print(f"\n💡 This file should work in the optimizer!")
    print(f"\nNext steps:")
    print(f"1. Open optimizer: python3 genetic_algo_nfl_optimizer.py")
    print(f"2. Click 'Load Players'")
    print(f"3. Select: {test_file}")
    print(f"4. Tables should populate with {len(df)} players")
    
    print(f"\n⚠️  If tables are still empty, check:")
    print(f"   • Console output for errors")
    print(f"   • Make sure 'Predicted_DK_Points' column exists (not 'Fantasy_Points')")
    print(f"   • Try closing and reopening the optimizer")

except FileNotFoundError:
    print(f"   ❌ File not found: {test_file}")
    print(f"\n📁 Available CSV files:")
    import os
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in csv_files:
        print(f"   - {f}")
    sys.exit(1)

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*80)

