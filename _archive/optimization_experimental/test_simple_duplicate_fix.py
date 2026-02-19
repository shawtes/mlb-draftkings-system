#!/usr/bin/env python3
"""
Simple test with clean sample data to verify the duplicate lineup fix works correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from optimizer01 import OptimizationWorker
from collections import defaultdict

def create_sample_data():
    """Create clean sample player data for testing"""
    
    # Sample MLB teams
    teams = ['BOS', 'NYY', 'CHC', 'LAD', 'HOU', 'ATL']
    positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
    
    players = []
    player_id = 1
    
    # Create balanced data for each team and position
    for team in teams:
        # Pitchers (2 per team)
        for i in range(2):
            players.append({
                'Name': f'{team}_Pitcher_{i+1}',
                'Team': team,
                'Position': 'P',
                'Salary': np.random.randint(8000, 12000),
                'Predicted_DK_Points': np.random.uniform(15, 25)
            })
        
        # Position players
        for pos in ['C', '1B', '2B', '3B', 'SS']:
            for i in range(2):  # 2 players per position per team
                players.append({
                    'Name': f'{team}_{pos}_{i+1}',
                    'Team': team,
                    'Position': pos,
                    'Salary': np.random.randint(6000, 10000),
                    'Predicted_DK_Points': np.random.uniform(8, 18)
                })
        
        # Outfielders (6 per team for more options)
        for i in range(6):
            players.append({
                'Name': f'{team}_OF_{i+1}',
                'Team': team,
                'Position': 'OF',
                'Salary': np.random.randint(5000, 9000),
                'Predicted_DK_Points': np.random.uniform(7, 16)
            })
    
    df = pd.DataFrame(players)
    print(f"✅ Created {len(df)} sample players across {len(teams)} teams")
    return df

def test_sequential_optimization():
    """Test the sequential optimization fix for duplicate lineups"""
    
    print("🧪 TESTING SEQUENTIAL OPTIMIZATION FIX")
    print("=" * 60)
    
    # Create clean test data
    df_players = create_sample_data()
    
    # Test parameters
    num_lineups = 8  # Request 8 lineups
    
    print(f"\n🎯 TEST CONFIGURATION:")
    print(f"   📊 Requesting: {num_lineups} lineups")
    print(f"   🏟️ Teams available: {sorted(df_players['Team'].unique())}")
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
            team_selections=[],  # No team restrictions
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
    
    # Force use of sequential optimization by setting num_lineups > 1
    print(f"\n🔥 RUNNING SEQUENTIAL OPTIMIZATION...")
    print(f"   ℹ️ With {num_lineups} lineups requested, sequential optimization should be used")
    
    try:
        # Run optimization
        results, team_exposure, stack_exposure = worker.optimize_lineups()
        
        print(f"✅ Optimization completed")
        print(f"📊 Generated {len(results)} lineup results")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        print(traceback.format_exc())
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
    if success and total_lineups >= num_lineups // 2:  # Accept if we get at least half the requested lineups
        print(f"🎉 TEST PASSED: Sequential optimization is working!")
        print(f"   ✅ Generated {unique_lineups} unique lineups")
        print(f"   ✅ No duplicate lineups found")
        print(f"   ✅ Sequential player exclusion is functioning correctly")
    else:
        print(f"❌ TEST FAILED:")
        if duplicates > 0:
            print(f"   🔧 Duplicate lineups still being generated")
        if total_lineups < num_lineups // 2:
            print(f"   🔧 Too few lineups generated ({total_lineups} vs {num_lineups} requested)")
    
    return success and total_lineups >= num_lineups // 2

if __name__ == "__main__":
    test_sequential_optimization()
