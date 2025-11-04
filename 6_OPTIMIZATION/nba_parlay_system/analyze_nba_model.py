"""
Analyze NBA Parlay Model Performance
====================================
Deep dive into what's working and what's not
"""
import pandas as pd
import numpy as np
from datetime import datetime
from nba_parlay_generator import NBAAdvancedParlayGenerator

def analyze_model_performance():
    """Analyze the model's performance on historical data"""
    
    print("🔍 NBA Parlay Model Analysis")
    print("=" * 60)
    
    # Load data
    data = pd.read_csv('nba_training_data.csv')
    data_with_actuals = data.dropna(subset=['actual_points'])
    
    # Get recent dates
    unique_dates = sorted(data_with_actuals['game_date'].unique())
    test_dates = unique_dates[-7:]
    
    print(f"\n📅 Analyzing {len(test_dates)} dates")
    
    all_legs = []
    
    for date in test_dates:
        projections = data_with_actuals[data_with_actuals['game_date'] == date].copy()
        
        # Fix column names
        if 'position_proj' not in projections.columns and 'position_proj_x' in projections.columns:
            projections['position_proj'] = projections['position_proj_x']
        if 'player_name_proj' not in projections.columns and 'player_name_proj_x' in projections.columns:
            projections['player_name_proj'] = projections['player_name_proj_x']
        if 'team_proj' not in projections.columns and 'team_proj_x' in projections.columns:
            projections['team_proj'] = projections['team_proj_x']
        
        # Generate parlays
        generator = NBAAdvancedParlayGenerator(projections)
        
        for i in range(10):
            parlay = generator.generate_parlay(max_legs=3)
            if not parlay.legs:
                continue
            
            for leg in parlay.legs:
                # Find player data
                player_data = projections[projections['player_name_proj'] == leg.player_name]
                if len(player_data) == 0:
                    continue
                
                player_data = player_data.iloc[0]
                
                # Get actual value
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
                    
                    # Determine if won
                    won = False
                    if leg.bet_type == 'OVER' and actual_value > leg.line:
                        won = True
                    elif leg.bet_type == 'UNDER' and actual_value < leg.line:
                        won = True
                    
                    # Store leg info
                    all_legs.append({
                        'date': date,
                        'player': leg.player_name,
                        'prop': leg.prop_type,
                        'line': leg.line,
                        'bet_type': leg.bet_type,
                        'predicted_hit_rate': leg.hit_rate,
                        'actual_value': actual_value,
                        'won': won,
                        'projection': player_data.get(f'projected_{leg.prop_type}', 0),
                        'position': player_data.get('position_proj', ''),
                        'team': player_data.get('team_proj', '')
                    })
    
    legs_df = pd.DataFrame(all_legs)
    
    if len(legs_df) == 0:
        print("❌ No legs to analyze")
        return
    
    print(f"\n✅ Analyzed {len(legs_df)} individual legs")
    
    # Overall performance
    overall_win_rate = legs_df['won'].mean()
    avg_predicted = legs_df['predicted_hit_rate'].mean()
    
    print(f"\n📊 Overall Performance:")
    print(f"   Actual Win Rate: {overall_win_rate:.1%}")
    print(f"   Predicted Hit Rate: {avg_predicted:.1%}")
    print(f"   Gap: {abs(overall_win_rate - avg_predicted):.1%}")
    
    # Analysis by prop type
    print(f"\n🎯 Performance by Prop Type:")
    prop_performance = legs_df.groupby('prop').agg({
        'won': 'mean',
        'predicted_hit_rate': 'mean',
        'player': 'count'
    }).round(3)
    prop_performance.columns = ['Actual WR', 'Predicted WR', 'Count']
    prop_performance['Gap'] = abs(prop_performance['Actual WR'] - prop_performance['Predicted WR'])
    print(prop_performance.sort_values('Actual WR', ascending=False))
    
    # Analysis by bet type
    print(f"\n📈 Performance by Bet Type:")
    bet_performance = legs_df.groupby('bet_type').agg({
        'won': 'mean',
        'predicted_hit_rate': 'mean',
        'player': 'count'
    }).round(3)
    bet_performance.columns = ['Actual WR', 'Predicted WR', 'Count']
    bet_performance['Gap'] = abs(bet_performance['Actual WR'] - bet_performance['Predicted WR'])
    print(bet_performance)
    
    # Analysis by position
    print(f"\n👥 Performance by Position:")
    pos_performance = legs_df.groupby('position').agg({
        'won': 'mean',
        'predicted_hit_rate': 'mean',
        'player': 'count'
    }).round(3)
    pos_performance.columns = ['Actual WR', 'Predicted WR', 'Count']
    pos_performance['Gap'] = abs(pos_performance['Actual WR'] - pos_performance['Predicted WR'])
    print(pos_performance.sort_values('Actual WR', ascending=False))
    
    # Analysis by predicted hit rate ranges
    print(f"\n🎲 Performance by Predicted Hit Rate:")
    legs_df['hit_rate_bin'] = pd.cut(legs_df['predicted_hit_rate'], 
                                      bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
                                      labels=['<50%', '50-60%', '60-70%', '70-80%', '80%+'])
    hit_rate_performance = legs_df.groupby('hit_rate_bin').agg({
        'won': 'mean',
        'predicted_hit_rate': 'mean',
        'player': 'count'
    }).round(3)
    hit_rate_performance.columns = ['Actual WR', 'Predicted WR', 'Count']
    hit_rate_performance['Gap'] = abs(hit_rate_performance['Actual WR'] - hit_rate_performance['Predicted WR'])
    print(hit_rate_performance)
    
    # Check for overconfidence
    print(f"\n⚠️  Overconfidence Analysis:")
    overconfident = legs_df[legs_df['predicted_hit_rate'] > 0.7]
    if len(overconfident) > 0:
        actual_vs_predicted = overconfident.groupby('prop').agg({
            'won': 'mean',
            'predicted_hit_rate': 'mean'
        }).round(3)
        actual_vs_predicted.columns = ['Actual', 'Predicted']
        actual_vs_predicted['Overconfident'] = actual_vs_predicted['Predicted'] - actual_vs_predicted['Actual']
        print(actual_vs_predicted.sort_values('Overconfident', ascending=False))
    
    # Analyze line selection
    print(f"\n📏 Line Selection Analysis:")
    legs_df['line_offset'] = legs_df['line'] - legs_df['projection']
    
    # For OVER bets
    over_bets = legs_df[legs_df['bet_type'] == 'OVER']
    print(f"\n   OVER Bets:")
    print(f"   Average line offset: {over_bets['line_offset'].mean():.2f}")
    print(f"   Win rate: {over_bets['won'].mean():.1%}")
    
    # Show distribution
    offset_bins = pd.cut(over_bets['line_offset'], bins=[-5, -1, 0, 1, 5], labels=['Far Under', 'Under', 'Over', 'Far Over'])
    offset_perf = over_bets.groupby(offset_bins).agg({
        'won': 'mean',
        'player': 'count'
    }).round(3)
    offset_perf.columns = ['Win Rate', 'Count']
    print(offset_perf)
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    # Find best and worst props
    best_prop = prop_performance.sort_values('Actual WR', ascending=False).index[0]
    worst_prop = prop_performance.sort_values('Actual WR', ascending=True).index[0]
    
    print(f"   ✅ Best performing prop: {best_prop}")
    print(f"   ❌ Worst performing prop: {worst_prop}")
    
    # Find if OVER or UNDER is better
    best_bet_type = bet_performance.sort_values('Actual WR', ascending=False).index[0]
    print(f"   🎯 Best bet type: {best_bet_type}")
    
    # Find best position
    best_pos = pos_performance.sort_values('Actual WR', ascending=False).index[0]
    print(f"   👤 Best position: {best_pos}")
    
    # Hit rate gap
    avg_gap = prop_performance['Gap'].mean()
    if avg_gap > 0.1:
        print(f"   ⚠️  Large prediction gap ({avg_gap:.1%}) - model may be overconfident")
    
    return legs_df

if __name__ == "__main__":
    analyze_model_performance()







