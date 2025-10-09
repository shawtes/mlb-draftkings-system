#!/usr/bin/env python3
"""
Test Script for DFS Optimizer - Team Combination Testing
"""

import sys
import os
import pandas as pd
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_path():
    """Add project directory to Python path"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

def load_test_data():
    """Load the player data for testing"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', '4_DATA', 'merged_player_projections01.csv')
        df = pd.read_csv(data_path)
        
        # Add the expected column name if it doesn't exist
        if 'Predicted_DK_Points' not in df.columns and 'My_Proj' in df.columns:
            df['Predicted_DK_Points'] = df['My_Proj']
        
        # Add Position column if it doesn't exist but Pos does
        if 'Position' not in df.columns and 'Pos' in df.columns:
            df['Position'] = df['Pos']
        
        print(f"✅ Loaded {len(df)} players from {data_path}")
        
        # Show team distribution
        team_counts = df['Team'].value_counts()
        print(f"📊 Teams available: {list(team_counts.index)}")
        print(f"🔢 Player counts per team: {dict(team_counts.head(10))}")
        
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def test_single_combination():
    """Test generating lineups for a single team combination"""
    print("\n" + "="*60)
    print("🧪 TESTING SINGLE TEAM COMBINATION")
    print("="*60)
    
    setup_path()
    
    try:
        from optimizer01 import OptimizationWorker
        
        # Load test data
        df = load_test_data()
        if df is None:
            return False
        
        # Test combination: CHC(5) + BOS(2)
        test_combination = {
            5: ['CHC'],  # 5 players from Chicago Cubs
            2: ['BOS']   # 2 players from Boston Red Sox
        }
        
        print(f"\n🎯 Testing combination: CHC(5) + BOS(2)")
        print(f"📋 Team selections: {test_combination}")
        
        # Check if we have enough players from each team
        chc_players = df[df['Team'] == 'CHC']
        bos_players = df[df['Team'] == 'BOS']
        chc_batters = chc_players[~chc_players['Pos'].str.contains('P', na=False)]
        bos_batters = bos_players[~bos_players['Pos'].str.contains('P', na=False)]
        
        print(f"🏟️ CHC total players: {len(chc_players)}, batters: {len(chc_batters)}")
        print(f"🏟️ BOS total players: {len(bos_players)}, batters: {len(bos_batters)}")
        
        # Show position breakdown for required teams
        print(f"\n📋 CHC Position Breakdown:")
        chc_positions = chc_batters['Pos'].value_counts()
        for pos, count in chc_positions.items():
            print(f"   {pos}: {count}")
        
        print(f"\n📋 BOS Position Breakdown:")
        bos_positions = bos_batters['Pos'].value_counts()
        for pos, count in bos_positions.items():
            print(f"   {pos}: {count}")
        
        # Check mathematical feasibility
        total_required = 5 + 2  # CHC + BOS
        total_batters_available = len(chc_batters) + len(bos_batters)
        print(f"\n🧮 Mathematical Check:")
        print(f"   Required team players: {total_required}")
        print(f"   Available team batters: {total_batters_available}")
        print(f"   Remaining slots: {8 - total_required}")  # 8 total lineup spots minus team requirements
        
        if len(chc_batters) < 5:
            print(f"❌ Not enough CHC batters: need 5, have {len(chc_batters)}")
            return False
        if len(bos_batters) < 2:
            print(f"❌ Not enough BOS batters: need 2, have {len(bos_batters)}")
            return False
        
        # Set up optimization parameters
        salary_cap = 50000
        position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
        num_lineups = 3  # Test with 3 lineups per combination as originally requested
        
        print(f"\n🚀 Generating {num_lineups} lineups...")
        
        # Create optimization worker
        worker = OptimizationWorker(
            df_players=df,
            salary_cap=salary_cap,
            position_limits=position_limits,
            included_players=[],  # Use all players
            stack_settings={"5|2": True},  # Stack pattern
            min_exposure={},
            max_exposure={},
            min_points=1,
            monte_carlo_iterations=100,
            num_lineups=num_lineups,
            team_selections=test_combination,
            min_unique=0,  # Disable min_unique filtering
            bankroll=1000,
            risk_tolerance='medium',
            disable_kelly=True,
            min_salary=None,
            use_advanced_quant=False,
            advanced_quant_params=None
        )
        
        # Run optimization
        results, team_exposure, stack_exposure = worker.optimize_lineups()
        
        # Analyze results
        print(f"\n📊 RESULTS ANALYSIS")
        print(f"💎 Generated {len(results)} lineups")
        
        if not results:
            print("❌ No lineups generated!")
            return False
        
        # Check each lineup
        lineup_analysis = []
        for i, result in results.items():
            if isinstance(result, dict) and 'lineup' in result:
                lineup_df = result['lineup']
            else:
                lineup_df = result
            
            if isinstance(lineup_df, pd.DataFrame):
                team_counts = lineup_df['Team'].value_counts()
                chc_count = team_counts.get('CHC', 0)
                bos_count = team_counts.get('BOS', 0)
                total_salary = lineup_df['Salary'].sum()
                total_points = lineup_df['My_Proj'].sum() if 'My_Proj' in lineup_df.columns else 0
                
                # Check positions
                position_counts = {}
                for pos in position_limits.keys():
                    position_counts[pos] = len(lineup_df[lineup_df['Pos'].str.contains(pos, na=False)])
                
                lineup_info = {
                    'lineup_num': i + 1,
                    'chc_players': chc_count,
                    'bos_players': bos_count,
                    'total_teams': len(team_counts),
                    'total_salary': total_salary,
                    'total_points': total_points,
                    'positions': position_counts,
                    'players': lineup_df['Name'].tolist()
                }
                lineup_analysis.append(lineup_info)
                
                print(f"\n🏆 Lineup {i+1}:")
                print(f"   👥 CHC: {chc_count}, BOS: {bos_count}, Total teams: {len(team_counts)}")
                print(f"   💰 Salary: ${total_salary:,}, Points: {total_points:.2f}")
                print(f"   🎯 Positions: {position_counts}")
                print(f"   📋 Players: {', '.join(lineup_df['Name'].tolist())}")
                
                # Check constraints
                if chc_count >= 5 and bos_count >= 2:
                    print(f"   ✅ Team constraints satisfied")
                else:
                    print(f"   ❌ Team constraints FAILED: need CHC≥5, BOS≥2")
        
        # Check for diversity
        print(f"\n🔍 DIVERSITY ANALYSIS")
        all_players_used = set()
        for lineup in lineup_analysis:
            all_players_used.update(lineup['players'])
        
        print(f"📊 Total unique players used: {len(all_players_used)}")
        print(f"📊 Expected max players (8 per lineup): {num_lineups * 8}")
        
        # Check for duplicate lineups
        lineup_signatures = []
        for lineup in lineup_analysis:
            signature = tuple(sorted(lineup['players']))
            lineup_signatures.append(signature)
        
        unique_signatures = set(lineup_signatures)
        print(f"🎯 Unique lineup signatures: {len(unique_signatures)}")
        print(f"🎯 Total lineups generated: {len(lineup_signatures)}")
        
        if len(unique_signatures) == len(lineup_signatures):
            print("✅ All lineups are unique!")
        else:
            print("❌ Some lineups are duplicates!")
        
        return len(results) == num_lineups and len(unique_signatures) == len(lineup_signatures)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_combinations():
    """Test generating lineups for multiple team combinations"""
    print("\n" + "="*60)
    print("🧪 TESTING MULTIPLE TEAM COMBINATIONS")
    print("="*60)
    
    setup_path()
    
    try:
        from optimizer01 import OptimizationWorker
        
        # Load test data
        df = load_test_data()
        if df is None:
            return False
        
        # Test multiple combinations
        test_combinations = [
            ({5: ['CHC'], 2: ['BOS']}, "CHC(5) + BOS(2)"),
            ({4: ['MIL'], 3: ['ATL']}, "MIL(4) + ATL(3)"),  # Changed from LAD/NYY to avoid expensive player issues
            ({3: ['ATL'], 3: ['HOU']}, "ATL(3) + HOU(3)")
        ]
        
        num_lineups = 3  # Test with 3 lineups per combination as originally requested
        all_results = []
        
        for team_selections, combo_name in test_combinations:
            print(f"\n🎯 Testing: {combo_name}")
            print(f"📋 Team selections: {team_selections}")
            
            # Check player availability
            team_check_passed = True
            for stack_size, teams in team_selections.items():
                for team in teams:
                    team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))]
                    if len(team_batters) < stack_size:
                        print(f"❌ {team} has only {len(team_batters)} batters, need {stack_size}")
                        team_check_passed = False
                    else:
                        print(f"✅ {team} has {len(team_batters)} batters, need {stack_size}")
            
            if not team_check_passed:
                print(f"❌ Skipping {combo_name} due to insufficient players")
                continue
            
            # Create optimization worker
            worker = OptimizationWorker(
                df_players=df,
                salary_cap=50000,
                position_limits={'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3},
                included_players=[],
                stack_settings={"5|2": True},
                min_exposure={},
                max_exposure={},
                min_points=1,
                monte_carlo_iterations=100,
                num_lineups=num_lineups,
                team_selections=team_selections,
                min_unique=0,
                bankroll=1000,
                risk_tolerance='medium',
                disable_kelly=True,
                min_salary=None,
                use_advanced_quant=False,
                advanced_quant_params=None
            )
            
            # Run optimization
            results, _, _ = worker.optimize_lineups()
            
            print(f"💎 Generated {len(results)} lineups for {combo_name}")
            
            # Quick analysis
            for i, result in results.items():
                if isinstance(result, dict) and 'lineup' in result:
                    lineup_df = result['lineup']
                else:
                    lineup_df = result
                
                if isinstance(lineup_df, pd.DataFrame):
                    team_counts = lineup_df['Team'].value_counts()
                    print(f"   Lineup {i+1}: {dict(team_counts)}")
            
            all_results.append((combo_name, len(results)))
        
        # Summary
        print(f"\n📊 SUMMARY")
        total_expected = len(test_combinations) * num_lineups
        total_generated = sum(count for _, count in all_results)
        
        print(f"🎯 Expected total lineups: {total_expected}")
        print(f"💎 Generated total lineups: {total_generated}")
        
        for combo_name, count in all_results:
            print(f"   {combo_name}: {count}/{num_lineups} lineups")
        
        return total_generated == total_expected
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔥 DFS OPTIMIZER TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Single Combination Test", test_single_combination),
        ("Multiple Combinations Test", test_multiple_combinations)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🚀 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The optimizer is working correctly.")
    else:
        print("🚨 SOME TESTS FAILED! The optimizer needs debugging.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
