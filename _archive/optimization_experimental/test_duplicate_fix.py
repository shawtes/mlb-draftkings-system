#!/usr/bin/env python3
"""
Test script to verify the duplicate lineup fix is working correctly.
This test will verify that the sequential optimization generates truly diverse lineups.
"""

import pandas as pd
import sys
import os

# Add the directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from optimizer01 import OptimizationWorker
from collections import defaultdict

def test_duplicate_fix():
    """Test that the sequential optimization fixes duplicate lineup generation"""
    
    print("🧪 TESTING DUPLICATE LINEUP FIX")
    print("=" * 60)
    
    # Load test data
    try:
        # Try multiple possible data file locations
        data_files = [
            os.path.join(os.path.dirname(current_dir), "4_DATA", "today_data.csv"),
            os.path.join(os.path.dirname(current_dir), "4_DATA", "data_with_dk_entries_salaries_final_cleaned_with_pitchers_dk_fpts.csv"),
            os.path.join(os.path.dirname(current_dir), "4_DATA", "merged_player_projections.csv"),
            os.path.join(current_dir, "today_data.csv"),
        ]
        
        df_players = None
        used_file = None
        
        for csv_file in data_files:
            if os.path.exists(csv_file):
                try:
                    df_players = pd.read_csv(csv_file)
                    used_file = csv_file
                    break
                except Exception as e:
                    print(f"⚠️ Error reading {csv_file}: {e}")
                    continue
                    
        if df_players is None:
            print(f"❌ Test data not found. Tried files:")
            for f in data_files:
                print(f"   - {f}")
            return False
            
        print(f"✅ Loaded {len(df_players)} players from {os.path.basename(used_file)}")
        
        # Check required columns
        required_columns = ['Name', 'Team', 'Position', 'Salary', 'Predicted_DK_Points']
        missing_columns = [col for col in required_columns if col not in df_players.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # Try to map common column names
            column_mappings = {
                'position': 'Position',
                'Pos': 'Position', 
                'salary': 'Salary',
                'DK_Salary': 'Salary',
                'calculated_dk_fpts': 'Predicted_DK_Points',
                'DK_Projection': 'Predicted_DK_Points',
                'rolling_30_ppg': 'Predicted_DK_Points'
            }
            
            for old_col, new_col in column_mappings.items():
                if old_col in df_players.columns and new_col not in df_players.columns:
                    df_players[new_col] = df_players[old_col]
                    print(f"   📋 Mapped {old_col} -> {new_col}")
                
        # Final check
        missing_columns = [col for col in required_columns if col not in df_players.columns]
        if missing_columns:
            print(f"❌ Still missing required columns: {missing_columns}")
            print(f"   Available columns: {list(df_players.columns)[:10]}...")  # Show first 10 only
            return False
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False
    
    # Test parameters
    num_lineups = 10  # Request 10 lineups
    test_team_selections = []  # Use all teams (empty = all teams)
    
    print(f"\n🎯 TEST CONFIGURATION:")
    print(f"   📊 Requesting: {num_lineups} lineups")
    print(f"   🏟️ Team selections: {'All teams' if not test_team_selections else test_team_selections}")
    print(f"   🔧 Expected: {num_lineups} UNIQUE lineups (no duplicates)")
    
    # Create optimization worker
    try:
        worker = OptimizationWorker(
            df_players=df_players,
            salary_cap=50000,
            position_limits={
                'P': 2, 'C': 1, '1B': 1, '2B': 1, 
                '3B': 1, 'SS': 1, 'OF': 3
            },
            included_players=[],  # Use all players
            stack_settings=['Mixed'],
            min_exposure=0,
            max_exposure=100,
            min_points=1,
            monte_carlo_iterations=100,
            num_lineups=num_lineups,
            team_selections=test_team_selections,
            min_unique=0,  # Disable min_unique filtering
            bankroll=1000,
            risk_tolerance='medium',
            disable_kelly=True,
            min_salary=None,
            use_advanced_quant=False,
            advanced_quant_params=None
        )
        
        print(f"✅ Created optimization worker")
        
    except Exception as e:
        print(f"❌ Error creating worker: {e}")
        return False
    
    # Run optimization
    try:
        print(f"\n🔥 RUNNING OPTIMIZATION...")
        results, team_exposure, stack_exposure = worker.optimize_lineups()
        
        print(f"✅ Optimization completed")
        print(f"📊 Generated {len(results)} lineup results")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return False
    
    # Analyze results for duplicates
    print(f"\n🔍 ANALYZING RESULTS FOR DUPLICATES...")
    
    if not results:
        print(f"❌ No lineups generated!")
        return False
    
    # Extract lineup signatures
    lineup_signatures = []
    lineup_analysis = []
    
    for i, result in results.items():
        if isinstance(result, dict) and 'lineup' in result:
            lineup_df = result['lineup']
            if not lineup_df.empty:
                # Create lineup signature from player names
                players = sorted(lineup_df['Name'].tolist())
                signature = tuple(players)
                lineup_signatures.append(signature)
                
                lineup_analysis.append({
                    'lineup_num': i + 1,
                    'points': result['total_points'],
                    'players': players,
                    'signature': signature
                })
                
                print(f"   Lineup {i+1}: {result['total_points']:.1f} pts - {players[:3]}...")
    
    # Check for duplicates
    print(f"\n📊 DUPLICATE ANALYSIS:")
    unique_signatures = set(lineup_signatures)
    total_lineups = len(lineup_signatures)
    unique_lineups = len(unique_signatures)
    duplicates = total_lineups - unique_lineups
    
    print(f"   🎯 Total lineups: {total_lineups}")
    print(f"   ✨ Unique lineups: {unique_lineups}")
    print(f"   🔄 Duplicate lineups: {duplicates}")
    
    if duplicates == 0:
        print(f"   ✅ SUCCESS: No duplicate lineups found!")
        success = True
    else:
        print(f"   ❌ FAILURE: Found {duplicates} duplicate lineups")
        
        # Show which lineups are duplicates
        signature_counts = {}
        for sig in lineup_signatures:
            signature_counts[sig] = signature_counts.get(sig, 0) + 1
        
        print(f"\n   🔍 DUPLICATE DETAILS:")
        for sig, count in signature_counts.items():
            if count > 1:
                players = list(sig)
                print(f"      {count}x duplicate: {players[:3]}...")
        
        success = False
    
    # Additional diversity metrics
    print(f"\n📈 DIVERSITY METRICS:")
    all_players_used = set()
    for analysis in lineup_analysis:
        all_players_used.update(analysis['players'])
    
    print(f"   👥 Total unique players used: {len(all_players_used)}")
    print(f"   📊 Expected max players: {total_lineups * 8} (if no overlap)")
    print(f"   🎯 Diversity ratio: {len(all_players_used) / (total_lineups * 8) * 100:.1f}%")
    
    # Test result summary
    print(f"\n" + "=" * 60)
    if success:
        print(f"🎉 TEST PASSED: Duplicate lineup fix is working correctly!")
        print(f"   ✅ Generated {unique_lineups} unique lineups out of {total_lineups} requested")
        print(f"   ✅ Sequential optimization successfully prevents duplicates")
    else:
        print(f"❌ TEST FAILED: Duplicate lineups still being generated")
        print(f"   🔧 The sequential optimization needs further debugging")
    
    return success

if __name__ == "__main__":
    test_duplicate_fix()
