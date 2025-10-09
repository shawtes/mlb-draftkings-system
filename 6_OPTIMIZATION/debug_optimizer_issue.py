#!/usr/bin/env python3
"""
Debug DFS Optimizer Issue
========================

This script will help identify why the optimizer isn't generating multiple unique lineups.
"""

import sys
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_optimizer_issue():
    """Debug the main optimizer issues"""
    
    print("🔍 DFS OPTIMIZER DEBUGGING SCRIPT")
    print("=" * 50)
    
    # 1. Check imports
    print("\n1. 📦 CHECKING IMPORTS...")
    try:
        from optimizer01 import FantasyBaseballApp, optimize_single_lineup
        print("✅ Main optimizer imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 2. Check for data files
    print("\n2. 📊 CHECKING DATA FILES...")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files[:5]:  # Show first 5
        print(f"  - {csv_file}")
    
    if not csv_files:
        print("❌ No CSV files found! Need player data to test.")
        return False
    
    # 3. Test data loading
    print("\n3. 🔄 TESTING DATA LOADING...")
    try:
        df = pd.read_csv(csv_files[0])
        print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)[:10]}")
        
        # Check required columns
        required_cols = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}")
            print("Available columns containing 'Name':", [col for col in df.columns if 'name' in col.lower()])
            print("Available columns containing 'Position':", [col for col in df.columns if 'position' in col.lower()])
            print("Available columns containing 'Team':", [col for col in df.columns if 'team' in col.lower()])
            print("Available columns containing 'Salary':", [col for col in df.columns if 'salary' in col.lower()])
            print("Available columns containing 'Point':", [col for col in df.columns if 'point' in col.lower()])
        else:
            print("✅ All required columns found")
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False
    
    # 4. Test basic optimization
    print("\n4. 🎯 TESTING BASIC OPTIMIZATION...")
    try:
        # Prepare test data
        if missing_cols:
            print("⚠️ Cannot test optimization due to missing columns")
            return True
        
        # Test single lineup optimization
        df_test = df.head(50).copy()  # Use first 50 players for test
        
        # Mock team selections and other parameters
        team_projected_runs = {team: 5.0 for team in df_test['Team'].unique()}
        team_selections = None
        min_salary = 0
        
        print(f"Testing with {len(df_test)} players from {len(df_test['Team'].unique())} teams")
        
        # Test single lineup
        args = (df_test, "No Stacks", team_projected_runs, team_selections, min_salary)
        result_lineup, stack_type = optimize_single_lineup(args)
        
        if result_lineup.empty:
            print("❌ Single lineup optimization failed")
            print("This suggests the optimization problem is infeasible")
            
            # Check data quality
            print("\n🔍 DATA QUALITY CHECK:")
            print(f"Salary range: ${df_test['Salary'].min():.0f} - ${df_test['Salary'].max():.0f}")
            print(f"Points range: {df_test['Predicted_DK_Points'].min():.1f} - {df_test['Predicted_DK_Points'].max():.1f}")
            print(f"Unique positions: {df_test['Position'].unique()}")
            print(f"Players per position:")
            for pos in df_test['Position'].unique():
                count = len(df_test[df_test['Position'] == pos])
                print(f"  {pos}: {count} players")
                
            # Check if we have enough players for each position
            position_limits = {
                'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 
                'OF': 3, 'P': 2
            }
            
            print("\n⚠️ POSITION REQUIREMENTS CHECK:")
            for pos, required in position_limits.items():
                available = len(df_test[df_test['Position'] == pos])
                status = "✅" if available >= required else "❌"
                print(f"  {pos}: {available} available, {required} required {status}")
                
        else:
            print("✅ Single lineup optimization successful!")
            print(f"Generated lineup: {len(result_lineup)} players")
            print(f"Total salary: ${result_lineup['Salary'].sum():.0f}")
            print(f"Total points: {result_lineup['Predicted_DK_Points'].sum():.1f}")
            
    except Exception as e:
        print(f"❌ Optimization test error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Check stack settings
    print("\n5. 🏗️ CHECKING STACK SETTINGS...")
    try:
        # Test different stack types
        stack_types = ["No Stacks", "4|2", "5|3"]
        for stack_type in stack_types:
            print(f"Testing stack type: {stack_type}")
            args = (df_test, stack_type, team_projected_runs, team_selections, min_salary)
            result_lineup, returned_stack = optimize_single_lineup(args)
            
            if result_lineup.empty:
                print(f"  ❌ {stack_type} failed")
            else:
                print(f"  ✅ {stack_type} succeeded")
                
    except Exception as e:
        print(f"❌ Stack test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 DEBUGGING COMPLETE")
    print("\nIf you see errors above, those are the issues to fix.")
    print("If everything looks good, the issue might be in the GUI or user workflow.")
    
    return True

if __name__ == "__main__":
    debug_optimizer_issue() 