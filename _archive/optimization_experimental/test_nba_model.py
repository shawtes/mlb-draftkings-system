"""
Test NBA Parlay Model on Historical Data
=========================================
Generates parlays for past dates and checks actual win rate
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_parlay_generator import NBAAdvancedParlayGenerator

def test_model_on_historical_data():
    """Test the model on a week of historical data"""
    
    print("🧪 Testing NBA Parlay Model on Historical Data")
    print("=" * 60)
    
    # Load training data
    print("\n📥 Loading historical data...")
    data = pd.read_csv('nba_training_data.csv')
    
    # Filter for dates with actual outcomes
    data_with_actuals = data.dropna(subset=['actual_points'])
    
    if len(data_with_actuals) == 0:
        print("❌ No data with actual outcomes found!")
        return
    
    print(f"✅ Found {len(data_with_actuals)} records with actual outcomes")
    
    # Get unique dates
    unique_dates = sorted(data_with_actuals['game_date'].unique())
    
    # Test on a single random day from the dataset
    import random
    test_dates = [random.choice(unique_dates)]
    
    print(f"\n📅 Testing on {len(test_dates)} dates:")
    for date in test_dates:
        print(f"   - {date}")
    
    all_results = []
    
    for date in test_dates:
        print(f"\n🎲 Testing {date}...")
        
        # Get projections for this date
        projections = data_with_actuals[data_with_actuals['game_date'] == date].copy()
        
        if len(projections) < 10:
            print(f"   ⚠️  Not enough data for {date}")
            continue
        
        # Ensure correct column names
        if 'position_proj' not in projections.columns and 'position_proj_x' in projections.columns:
            projections['position_proj'] = projections['position_proj_x']
        if 'player_name_proj' not in projections.columns and 'player_name_proj_x' in projections.columns:
            projections['player_name_proj'] = projections['player_name_proj_x']
        if 'team_proj' not in projections.columns and 'team_proj_x' in projections.columns:
            projections['team_proj'] = projections['team_proj_x']
        
        # Generate parlays
        generator = NBAAdvancedParlayGenerator(projections)
        
        # Generate 10 parlays
        parlays = []
        for i in range(10):
            parlay = generator.generate_parlay(max_legs=3)
            if parlay.legs:
                parlays.append(parlay)
        
        if not parlays:
            print(f"   ⚠️  Could not generate parlays for {date}")
            continue
        
        print(f"   ✅ Generated {len(parlays)} parlays")
        
        # Check each parlay against actual outcomes
        for parlay in parlays:
            winning_legs = 0
            total_legs = len(parlay.legs)
            
            for leg in parlay.legs:
                # Find the player's actual stat
                player_data = projections[projections['player_name_proj'] == leg.player_name]
                
                if len(player_data) == 0:
                    # Try alternative column name
                    player_data = projections[projections['player_name_proj_x'] == leg.player_name]
                
                if len(player_data) == 0:
                    winning_legs += 0  # Missing data = lose
                    continue
                
                player_data = player_data.iloc[0]
                
                # Get actual stat value
                actual_value = None
                prop_to_col = {
                    'points': 'actual_points',
                    'rebounds': 'actual_rebounds',
                    'assists': 'actual_assists',
                    'steals': 'actual_steals',
                    'blocks': 'actual_blocks',
                    'three_pointers': 'actual_three_pointers'
                }
                
                actual_col = prop_to_col.get(leg.prop_type)
                if actual_col and actual_col in player_data:
                    actual_value = player_data[actual_col]
                
                if actual_value is not None:
                    # Check if leg wins
                    if leg.bet_type == 'OVER':
                        if actual_value > leg.line:
                            winning_legs += 1
                    elif leg.bet_type == 'UNDER':
                        if actual_value < leg.line:
                            winning_legs += 1
            
            # Parlay wins if all legs win
            parlay_won = (winning_legs == total_legs)
            
            all_results.append({
                'date': date,
                'legs': total_legs,
                'won': parlay_won,
                'winning_legs': winning_legs,
                'predicted_hit_rate': parlay.combined_hit_rate,
                'odds': parlay.estimated_odds
            })
            
            if parlay_won:
                print(f"   ✅ Parlay won ({winning_legs}/{total_legs} legs)")
            else:
                print(f"   ❌ Parlay lost ({winning_legs}/{total_legs} legs)")
    
    # Calculate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        total_parlays = len(results_df)
        winning_parlays = results_df['won'].sum()
        win_rate = winning_parlays / total_parlays if total_parlays > 0 else 0
        
        avg_predicted = results_df['predicted_hit_rate'].mean()
        avg_odds = results_df['odds'].mean()
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Total Parlays: {total_parlays}")
        print(f"Winning Parlays: {winning_parlays}")
        print(f"Actual Win Rate: {win_rate:.1%}")
        print(f"Predicted Hit Rate: {avg_predicted:.1%}")
        print(f"Average Odds: +{avg_odds:.0f}")
        
        # Show breakdown by legs
        print("\n📈 Breakdown by Number of Legs:")
        for legs in sorted(results_df['legs'].unique()):
            leg_data = results_df[results_df['legs'] == legs]
            leg_win_rate = leg_data['won'].mean()
            print(f"   {legs}-leg parlays: {leg_win_rate:.1%} ({leg_data['won'].sum()}/{len(leg_data)})")
        
        print("\n" + "=" * 60)
        
        return results_df
    else:
        print("\n❌ No results to analyze")
        return None

if __name__ == "__main__":
    test_model_on_historical_data()

