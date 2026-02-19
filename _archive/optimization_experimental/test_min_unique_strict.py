#!/usr/bin/env python3
"""
Test script to verify strict min_unique constraint enforcement
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimizer01 import *

def test_strict_min_unique():
    """Test that min_unique constraint is strictly enforced"""
    print("🔥 Testing strict min_unique constraint enforcement")
    print("=" * 60)
    
    # Load test data
    data_file = os.path.join('..', '4_DATA', 'merged_player_projections01.csv')
    if not os.path.exists(data_file):
        print(f"❌ Test data file not found: {data_file}")
        return
    
    df_players = pd.read_csv(data_file)
    
    # Map columns to expected names
    if 'ML_Prediction' in df_players.columns:
        df_players['Predicted_DK_Points'] = df_players['ML_Prediction']
    elif 'My_Proj' in df_players.columns:
        df_players['Predicted_DK_Points'] = df_players['My_Proj']
    elif 'PPG_Projection' in df_players.columns:
        df_players['Predicted_DK_Points'] = df_players['PPG_Projection']
    else:
        # Use first numeric column as fallback
        numeric_cols = df_players.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_players['Predicted_DK_Points'] = df_players[numeric_cols[0]]
        else:
            df_players['Predicted_DK_Points'] = 10  # Default value
    
    print(f"✅ Loaded {len(df_players)} players from {data_file}")
    
    # Check available teams and select valid ones
    available_teams = df_players['Team'].value_counts()
    print(f"✅ Available teams: {list(available_teams.index[:10])}")
    
    # Select teams with enough players for stacking
    teams_with_enough_players = available_teams[available_teams >= 4].index.tolist()
    
    if len(teams_with_enough_players) < 2:
        print("❌ Not enough teams with sufficient players for testing")
        return
        
    # Use the top teams for testing
    team_4_stack = teams_with_enough_players[:2]  # WAS, BOS
    team_2_stack = teams_with_enough_players[2:4]  # ARI, TOR
    
    print(f"🎯 Using 4-stack teams: {team_4_stack}")
    print(f"🎯 Using 2-stack teams: {team_2_stack}")
    
    # Test parameters
    salary_cap = 50000
    position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
    included_players = set()
    stack_settings = {'4|2': True}
    min_exposure = {}
    max_exposure = {}
    min_points = 1
    monte_carlo_iterations = 50
    num_lineups = 5
    team_selections = {'4': team_4_stack, '2': team_2_stack}
    min_salary = 45000
    
    # Test different min_unique values
    test_values = [0, 3, 5, 7]
    
    for min_unique in test_values:
        print(f"\n🎯 Testing min_unique = {min_unique}")
        print("-" * 40)
        
        # Create optimizer worker
        worker = OptimizationWorker(
            df_players=df_players,
            salary_cap=salary_cap,
            position_limits=position_limits,
            included_players=included_players,
            stack_settings=stack_settings,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            min_points=min_points,
            monte_carlo_iterations=monte_carlo_iterations,
            num_lineups=num_lineups,
            team_selections=team_selections,
            min_unique=min_unique,
            bankroll=1000,
            risk_tolerance='medium',
            disable_kelly=True,
            min_salary=min_salary
        )
        
        # Generate lineups
        try:
            results, _, _ = worker.optimize_lineups()
            
            if results and len(results) > 0:
                print(f"✅ Generated {len(results)} lineups")
                
                # Check actual min_unique constraint between all pairs
                lineups = []
                for i, result in enumerate(results):
                    if isinstance(result, dict) and 'lineup' in result:
                        lineups.append(result['lineup'])
                    elif isinstance(result, pd.DataFrame):
                        lineups.append(result)
                
                # Check pairwise unique players
                min_unique_found = float('inf')
                for i in range(len(lineups)):
                    for j in range(i + 1, len(lineups)):
                        players_i = set(lineups[i]['Name'].tolist())
                        players_j = set(lineups[j]['Name'].tolist())
                        unique_players = len(players_i.symmetric_difference(players_j))
                        min_unique_found = min(min_unique_found, unique_players)
                        
                        print(f"   Lineup {i+1} vs {j+1}: {unique_players} unique players")
                
                print(f"📊 Min unique found: {min_unique_found}, Required: {min_unique}")
                
                if min_unique == 0:
                    print("✅ PASS: min_unique=0 allows any overlap")
                elif min_unique_found >= min_unique:
                    print("✅ PASS: Constraint satisfied")
                else:
                    print("❌ FAIL: Constraint violated")
                    
            else:
                print("❌ No lineups generated")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    test_strict_min_unique()
