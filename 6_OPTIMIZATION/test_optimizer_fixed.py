#!/usr/bin/env python3
"""
Test Fixed Optimizer
===================

Simple test to confirm the optimizer is working after column fixes.
"""

import sys
import os
sys.path.insert(0, '.')

from optimizer01 import optimize_single_lineup
import pandas as pd

def test_optimizer():
    """Test the fixed optimizer"""
    
    print("🎯 TESTING FIXED OPTIMIZER")
    print("=" * 40)
    
    try:
        # Load the CSV file with all required columns
        df = pd.read_csv('optimizer_input_with_probabilities.csv')
        print(f"✅ Loaded CSV: {len(df)} rows")
        
        # Test basic optimization
        df_test = df.head(50).copy()
        team_projected_runs = {team: 5.0 for team in df_test['Team'].unique()}
        team_selections = None
        min_salary = 0
        
        print(f"🔍 Testing with {len(df_test)} players")
        
        # Test single lineup
        args = (df_test, 'No Stacks', team_projected_runs, team_selections, min_salary)
        result_lineup, stack_type = optimize_single_lineup(args)
        
        if result_lineup.empty:
            print("❌ Optimization failed")
            return False
        else:
            print("✅ OPTIMIZATION SUCCESSFUL!")
            print(f"   Players: {len(result_lineup)}")
            print(f"   Total salary: ${result_lineup['Salary'].sum():.0f}")
            print(f"   Total points: {result_lineup['Predicted_DK_Points'].sum():.1f}")
            print(f"   Positions: {dict(result_lineup['Position'].value_counts())}")
            
            # Test multiple lineups
            print(f"\n🎯 Testing multiple lineup generation...")
            lineups = []
            for i in range(3):
                args = (df_test, 'No Stacks', team_projected_runs, team_selections, min_salary)
                lineup, stack_type = optimize_single_lineup(args)
                if not lineup.empty:
                    lineups.append(lineup)
            
            if len(lineups) > 1:
                print(f"✅ Generated {len(lineups)} different lineups!")
                for i, lineup in enumerate(lineups):
                    points = lineup['Predicted_DK_Points'].sum()
                    salary = lineup['Salary'].sum()
                    print(f"   Lineup {i+1}: {points:.1f} pts, ${salary:.0f}")
                    
                # Check if lineups are different
                same_lineups = 0
                for i in range(len(lineups)):
                    for j in range(i+1, len(lineups)):
                        if set(lineups[i]['Name']) == set(lineups[j]['Name']):
                            same_lineups += 1
                
                if same_lineups == 0:
                    print("✅ All lineups are unique!")
                else:
                    print(f"⚠️ {same_lineups} duplicate lineups found")
                    
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎯 OPTIMIZER TEST COMPLETE")

if __name__ == "__main__":
    test_optimizer() 