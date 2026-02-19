#!/usr/bin/env python3
"""
Test script to verify combination generation is working
"""

import sys
import os
import pandas as pd
from itertools import combinations
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from optimizer01 import OptimizationWorker
    print("✅ Successfully imported OptimizationWorker")
except ImportError as e:
    print(f"❌ Failed to import OptimizationWorker: {e}")
    sys.exit(1)

def test_combinations():
    """Test the combination generation functionality"""
    
    print("🔍 Testing DFS Combination Generation")
    print("=" * 50)
    
    # Try to load the player data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '4_DATA', 'merged_player_projections01.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Player data file not found: {data_path}")
        print("Please ensure the player data file exists")
        return False
    
    try:
        # Load and preprocess the data manually
        df_players = pd.read_csv(data_path)
        
        # Basic required columns
        basic_required = ['Name', 'Team', 'Pos', 'Salary']
        
        # Check for basic required columns
        missing_basic = [col for col in basic_required if col not in df_players.columns]
        if missing_basic:
            print(f"❌ Missing required columns: {missing_basic}")
            return False
        
        # Handle different prediction column names flexibly
        prediction_column = None
        possible_prediction_columns = [
            'Predicted_DK_Points',  # Standard expected name
            'My_Proj',              # Your CSV format
            'ML_Prediction',        # ML prediction column
            'PPG_Projection',       # PPG projection column
            'Projection',           # Generic projection
            'Points',               # Simple points
            'DK_Points',            # DraftKings points
            'Fantasy_Points'        # Fantasy points
        ]
        
        # Find the first available prediction column
        for col in possible_prediction_columns:
            if col in df_players.columns:
                prediction_column = col
                break
        
        if prediction_column is None:
            available_cols = list(df_players.columns)
            print(f"❌ No prediction column found. Available columns: {available_cols}. Expected one of: {possible_prediction_columns}")
            return False
        
        # Rename the prediction column to the standard name for consistency
        if prediction_column != 'Predicted_DK_Points':
            df_players = df_players.rename(columns={prediction_column: 'Predicted_DK_Points'})
            logging.info(f"Using '{prediction_column}' as prediction column, renamed to 'Predicted_DK_Points'")
        
        # Clean and validate data
        df_players = df_players.dropna(subset=['Name', 'Salary', 'Predicted_DK_Points'])
        df_players['Salary'] = pd.to_numeric(df_players['Salary'], errors='coerce')
        df_players['Predicted_DK_Points'] = pd.to_numeric(df_players['Predicted_DK_Points'], errors='coerce')
        
        # Remove rows with invalid salary or prediction values
        df_players = df_players.dropna(subset=['Salary', 'Predicted_DK_Points'])
        df_players = df_players[df_players['Salary'] > 0]
        df_players = df_players[df_players['Predicted_DK_Points'] > 0]
        
        print(f"✅ Loaded {len(df_players)} players from {data_path}")
        
        # Display basic info about the processed data
        print(f"📊 Data info:")
        print(f"   • Columns: {list(df_players.columns)}")
        if 'Team' in df_players.columns:
            teams = df_players['Team'].unique()
            print(f"   • Teams: {len(teams)} teams - {list(teams)[:10]}{'...' if len(teams) > 10 else ''}")
        
        # Check if we have the expected prediction column
        if 'Predicted_DK_Points' in df_players.columns:
            print(f"   • ✅ Prediction column 'Predicted_DK_Points' found (from '{prediction_column}')")
        else:
            print(f"   • ❌ Prediction column 'Predicted_DK_Points' missing")
            return False
        
        # Test a simple combination
        print("\n🧪 Testing a simple 2-team combination...")
        
        # Get some teams to test with
        if 'Team' in df_players.columns:
            available_teams = df_players['Team'].unique()
            if len(available_teams) >= 2:
                test_teams = available_teams[:2]
                print(f"   • Using teams: {test_teams}")
                
                # Set up basic parameters
                salary_cap = 50000
                position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
                
                # Create team selections for 4|2 stack
                team_selections = {
                    4: [test_teams[0]],  # 4 players from first team
                    2: [test_teams[1]]   # 2 players from second team
                }
                
                # Create stack settings
                stack_settings = {'4|2': True}
                
                print(f"   • Team selections: {team_selections}")
                print(f"   • Stack settings: {stack_settings}")
                
                # Create optimization worker
                worker = OptimizationWorker(
                    df_players=df_players,
                    salary_cap=salary_cap,
                    position_limits=position_limits,
                    included_players=None,
                    stack_settings=stack_settings,
                    min_exposure={},
                    max_exposure={},
                    min_points=1,
                    monte_carlo_iterations=100,                        num_lineups=5,  # Generate 5 lineups to test min_unique constraint
                        team_selections=team_selections,
                        min_unique=5,  # Test with min_unique=5 to force diversity
                        bankroll=1000,
                        risk_tolerance='medium',
                        disable_kelly=True,
                        min_salary=0,
                        use_advanced_quant=False,
                        advanced_quant_params={}
                )
                
                print("\n🚀 Running optimization...")
                
                # Run the optimization
                try:
                    results, _, _ = worker.optimize_lineups()
                    
                    if results and len(results) > 0:
                        print(f"✅ Successfully generated {len(results)} lineups!")
                        
                        # Test min_unique constraint by checking lineup diversity
                        print(f"\n🔍 Testing min_unique constraint (min_unique=5):")
                        
                        if len(results) > 1:
                            # Compare first two lineups to check diversity
                            first_result = list(results.values())[0] if isinstance(results, dict) else results[0]
                            second_result = list(results.values())[1] if isinstance(results, dict) else results[1]
                            
                            # Extract player names from both lineups
                            if isinstance(first_result, dict) and 'lineup' in first_result:
                                first_lineup = first_result['lineup']
                                second_lineup = second_result['lineup']
                            else:
                                first_lineup = first_result
                                second_lineup = second_result
                            
                            first_players = set(first_lineup['Name'].tolist())
                            second_players = set(second_lineup['Name'].tolist())
                            
                            # Calculate overlap
                            overlap = len(first_players.intersection(second_players))
                            unique_players = 10 - overlap  # Total players - overlapping players
                            
                            print(f"   📊 Lineup 1 players: {list(first_players)[:5]}...")
                            print(f"   📊 Lineup 2 players: {list(second_players)[:5]}...")
                            print(f"   📊 Overlapping players: {overlap}")
                            print(f"   📊 Unique players between lineups: {unique_players}")
                            
                            if unique_players >= 5:
                                print(f"   ✅ Min unique constraint satisfied! ({unique_players} >= 5)")
                            else:
                                print(f"   ⚠️ Min unique constraint NOT satisfied! ({unique_players} < 5)")
                        
                        # Show some details about the first lineup
                        first_result = list(results.values())[0] if isinstance(results, dict) else results[0]
                        
                        if isinstance(first_result, dict) and 'lineup' in first_result:
                            lineup_df = first_result['lineup']
                            total_salary = first_result.get('total_salary', 0)
                            total_points = first_result.get('total_points', 0)
                        else:
                            lineup_df = first_result
                            total_salary = lineup_df['Salary'].sum() if 'Salary' in lineup_df.columns else 0
                            total_points = lineup_df['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_df.columns else 0
                        
                        print(f"\n📋 Sample lineup:")
                        print(f"   • Total salary: ${total_salary:,.0f}")
                        print(f"   • Total points: {total_points:.2f}")
                        print(f"   • Players: {len(lineup_df)}")
                        
                        if hasattr(lineup_df, 'to_string'):
                            print(f"\n   Lineup details:")
                            # Show the standard columns
                            display_cols = ['Name', 'Pos', 'Team', 'Salary', 'Predicted_DK_Points']
                            # Only show columns that exist
                            actual_cols = [col for col in display_cols if col in lineup_df.columns]
                            print(lineup_df[actual_cols].to_string(index=False))
                        
                        print(f"\n✅ Combination generation is working correctly!")
                        return True
                    else:
                        print(f"❌ No lineups generated - optimization may have failed")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error during optimization: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print(f"❌ Not enough teams in data (need at least 2, found {len(available_teams)})")
                return False
        else:
            print("❌ No 'Team' column found in player data")
            return False
            
    except Exception as e:
        print(f"❌ Error loading player data: {e}")
        return False

if __name__ == "__main__":
    success = test_combinations()
    
    if success:
        print(f"\n🎉 Combination test completed successfully!")
        print(f"💡 To use combinations in the GUI:")
        print(f"   1. Launch the optimizer with: python launch_optimizer.py")
        print(f"   2. Load your player data")
        print(f"   3. Go to the 'Team Combinations' tab")
        print(f"   4. Select teams and stack patterns")
        print(f"   5. Generate combinations and lineups")
    else:
        print(f"\n❌ Combination test failed!")
        print(f"🔧 Check the error messages above for troubleshooting")
    
    input("\nPress Enter to exit...")
