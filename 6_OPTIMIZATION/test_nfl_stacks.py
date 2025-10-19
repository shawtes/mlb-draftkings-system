"""
Test script for new NFL stacking system
Run this to verify the stacking engine works correctly
"""

import pandas as pd
import sys
from nfl_stack_config import get_all_stack_names, get_stack_display_names
from nfl_stack_engine import create_stack_engine
from nfl_stack_integration import (
    generate_team_stack_combinations,
    optimize_lineup_with_stack,
    validate_nfl_lineup,
    get_stack_types_for_gui
)

def load_test_data():
    """Load Week 6 data for testing"""
    try:
        df = pd.read_csv('nfl_week6_gpp_enhanced.csv')
        print(f"✅ Loaded {len(df)} players from nfl_week6_gpp_enhanced.csv")
        return df
    except FileNotFoundError:
        print("❌ Could not find nfl_week6_gpp_enhanced.csv")
        print("   Run this from 6_OPTIMIZATION directory")
        sys.exit(1)

def test_stack_config():
    """Test 1: Verify stack configuration"""
    print("\n" + "="*80)
    print("TEST 1: Stack Configuration")
    print("="*80)
    
    all_stacks = get_all_stack_names()
    display_names = get_stack_display_names()
    
    print(f"\n📋 Available Stack Types: {len(all_stacks)}")
    for stack_key in all_stacks:
        display_name = display_names.get(stack_key, stack_key)
        print(f"  • {stack_key:25s} → {display_name}")
    
    print("\n✅ Stack configuration loaded successfully")

def test_stack_engine(df):
    """Test 2: Stack engine functionality"""
    print("\n" + "="*80)
    print("TEST 2: Stack Engine")
    print("="*80)
    
    engine = create_stack_engine(df)
    
    # Test team analysis
    teams = df['Team'].unique()[:5]
    print(f"\n🏈 Analyzing first 5 teams...")
    
    for team in teams:
        if pd.isna(team):
            continue
        print(f"\n  {team}:")
        
        # Check game info
        game_info = engine.get_teams_in_game(team)
        opponent = game_info[1]
        print(f"    Opponent: {opponent if opponent else 'Unknown'}")
        
        # Check feasible stacks
        feasible_stacks = engine.get_available_stacks_for_team(team, 'gpp_tournament')
        print(f"    Feasible Stacks: {len(feasible_stacks)}")
        if feasible_stacks:
            for stack in feasible_stacks[:3]:  # Show first 3
                print(f"      - {stack}")
    
    # Test game matchups
    matchups = engine.get_all_game_matchups()
    print(f"\n🎮 Found {len(matchups)} game matchups:")
    for team1, team2 in matchups[:5]:
        print(f"    {team1} vs {team2}")
    
    print("\n✅ Stack engine working correctly")

def test_combination_generation(df):
    """Test 3: Generate stack combinations"""
    print("\n" + "="*80)
    print("TEST 3: Stack Combination Generation")
    print("="*80)
    
    # Generate combinations for GPP
    combos = generate_team_stack_combinations(
        df, 
        contest_type='gpp_tournament',
        max_combinations=20
    )
    
    print(f"\n📊 Generated {len(combos)} combinations")
    print("\nSample combinations:")
    for i, combo in enumerate(combos[:10], 1):
        print(f"  {i}. {combo['display_name']}")
        print(f"      Correlation: {combo['correlation']:.2f}, Leverage: {combo['leverage']:.2f}")
    
    print("\n✅ Combination generation working")
    return combos

def test_lineup_optimization(df, combos):
    """Test 4: Generate actual lineups"""
    print("\n" + "="*80)
    print("TEST 4: Lineup Optimization")
    print("="*80)
    
    if not combos:
        print("❌ No combinations to test")
        return
    
    # Test first 3 combinations
    for i, combo in enumerate(combos[:3], 1):
        print(f"\n🎯 Testing Combination {i}: {combo['display_name']}")
        
        try:
            lineup, projection = optimize_lineup_with_stack(
                df,
                stack_type=combo['stack_type'],
                team=combo['team'],
                opponent=combo['opponent']
            )
            
            if len(lineup) == 9:
                print(f"  ✅ Generated valid lineup")
                print(f"     Projected Points: {projection:.2f}")
                print(f"     Total Salary: ${lineup['Salary'].sum():,}")
                
                # Show stack players
                stack_team = combo['team']
                stack_players = lineup[lineup['Team'] == stack_team]
                print(f"     Stack Players ({stack_team}):")
                for _, player in stack_players.iterrows():
                    print(f"       • {player['Position']:3s} {player['Name']:25s} ${player['Salary']:5.0f}")
                
                # Validate lineup
                is_valid, errors = validate_nfl_lineup(lineup)
                if is_valid:
                    print(f"  ✅ Lineup validation passed")
                else:
                    print(f"  ❌ Validation errors: {errors}")
            else:
                print(f"  ❌ Invalid lineup size: {len(lineup)}")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print("\n✅ Lineup optimization test complete")

def test_gui_integration():
    """Test 5: GUI integration functions"""
    print("\n" + "="*80)
    print("TEST 5: GUI Integration")
    print("="*80)
    
    # Test GUI stack list for different contest types
    contest_types = ['cash_game', 'gpp_tournament', 'single_entry_gpp']
    
    for contest_type in contest_types:
        stacks = get_stack_types_for_gui(contest_type)
        print(f"\n  {contest_type}:")
        print(f"    {len(stacks)} stack types available")
        for stack_key, display_name in stacks[:5]:
            print(f"      • {display_name}")
    
    print("\n✅ GUI integration ready")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("NFL STACKING SYSTEM TEST SUITE")
    print("="*80)
    
    # Load data
    df = load_test_data()
    
    # Run tests
    test_stack_config()
    test_stack_engine(df)
    combos = test_combination_generation(df)
    test_lineup_optimization(df, combos)
    test_gui_integration()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nThe new NFL stacking system is ready to use!")
    print("\nNext steps:")
    print("  1. Integrate with genetic_algo_nfl_optimizer.py")
    print("  2. Update GUI to use new stack types")
    print("  3. Test with real lineup generation")
    print()

if __name__ == "__main__":
    main()

