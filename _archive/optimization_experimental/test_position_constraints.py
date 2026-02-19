"""
Direct test of position constraints in advanced quantitative optimizer
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.insert(0, os.getcwd())

from advanced_quant_optimizer import AdvancedQuantitativeOptimizer

def test_position_constraints():
    """Test position constraints directly"""
    print("🔍 Testing Position Constraints in Advanced Quantitative Optimizer")
    print("=" * 70)
    
    # Load sample data
    data_path = "../4_DATA/merged_player_projections01.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} players from data file")
        
        # Fix column names
        if 'Pos' in df.columns:
            df['Position'] = df['Pos']
        if 'My_Proj' in df.columns:
            df['Predicted_DK_Points'] = df['My_Proj']
        
        # Clean and validate data
        df = df.dropna(subset=['Predicted_DK_Points', 'Position', 'Team'])
        df = df[df['Predicted_DK_Points'] > 0]
        
        print(f"✅ Data cleaned and validated, {len(df)} players remaining")
        
        # Check position distribution
        print("\n📊 Position Distribution:")
        position_counts = df['Position'].value_counts()
        print(position_counts)
        
        # Initialize advanced optimizer
        optimizer = AdvancedQuantitativeOptimizer(
            confidence_level=0.95,
            lookback_window=30,
            monte_carlo_sims=1000
        )
        
        # Test lineup generation
        print("\n🎯 Testing lineup generation...")
        
        # Generate 3 lineups to test
        lineups = optimizer.optimize_lineups(
            player_data=df,
            historical_data=df,
            num_lineups=3,
            optimization_strategy='sharpe',
            risk_tolerance=0.5
        )
        
        if not lineups:
            print("❌ No lineups generated!")
            return False
        
        print(f"✅ Generated {len(lineups)} lineups")
        
        # Expected DraftKings MLB positions
        expected_positions = {'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3, 'P': 2}
        
        # Test each lineup
        all_lineups_valid = True
        for i, lineup_result in enumerate(lineups):
            print(f"\n📋 Lineup {i+1} Analysis:")
            
            # Extract lineup data
            lineup_data = lineup_result.get('lineup', [])
            if not lineup_data:
                print(f"   ❌ No lineup data found!")
                all_lineups_valid = False
                continue
            
            # Count positions
            position_counts = {}
            for player in lineup_data:
                pos = player.get('position', 'Unknown')
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            print(f"   Positions found: {position_counts}")
            
            # Check each expected position
            lineup_valid = True
            for pos, expected_count in expected_positions.items():
                actual_count = position_counts.get(pos, 0)
                if actual_count != expected_count:
                    print(f"   ❌ Position {pos}: Expected {expected_count}, Got {actual_count}")
                    lineup_valid = False
                    all_lineups_valid = False
                else:
                    print(f"   ✅ Position {pos}: {actual_count} (correct)")
            
            # Check total players
            total_players = len(lineup_data)
            if total_players != 10:
                print(f"   ❌ Total players: Expected 10, Got {total_players}")
                lineup_valid = False
                all_lineups_valid = False
            else:
                print(f"   ✅ Total players: {total_players} (correct)")
            
            # Check salary cap
            total_salary = sum(player.get('salary', 0) for player in lineup_data)
            if total_salary > 50000:
                print(f"   ❌ Salary cap exceeded: ${total_salary}")
                lineup_valid = False
                all_lineups_valid = False
            else:
                print(f"   ✅ Salary: ${total_salary} (within cap)")
            
            # Show player details
            print(f"   👥 Players:")
            for player in lineup_data:
                print(f"      {player.get('name', 'Unknown')} ({player.get('position', 'Unknown')}) - ${player.get('salary', 0)}")
            
            if lineup_valid:
                print(f"   🎉 Lineup {i+1} PASSES all constraints!")
            else:
                print(f"   ❌ Lineup {i+1} FAILS constraints!")
        
        if all_lineups_valid:
            print("\n🎉 ALL LINEUPS PASS position constraints!")
            return True
        else:
            print("\n❌ Some lineups FAIL position constraints!")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_position_constraints()
    if success:
        print("\n🎉 POSITION CONSTRAINTS TEST PASSED!")
    else:
        print("\n❌ POSITION CONSTRAINTS TEST FAILED!")
