import pandas as pd
import numpy as np
import sys
import os

# Add the system path to import from prediction modules
sys.path.append('c:/Users/smtes/Downloads/coinbase_ml_trader/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS')

def test_ten_day_rolling_constraint():
    """Test that predictions are constrained within 10-day rolling ranges"""
    
    # Create mock data with known 10-day rolling ranges
    test_data = {
        'Name': ['Player A', 'Player B', 'Player C', 'Player D'],
        'rolling_min_fpts_10': [5.0, 8.0, 2.0, np.nan],  # Player D has no 10-day data
        'rolling_max_fpts_10': [15.0, 20.0, 12.0, np.nan],
        '5_game_avg': [10.0, 14.0, 7.0, 9.0],
        'predicted_dk_fpts': [0.0, 0.0, 0.0, 0.0]  # Will be filled by constraint function
    }
    
    df = pd.DataFrame(test_data)
    
    # Test different raw predictions
    test_cases = [
        # Raw predictions that should be constrained to 10-day range
        np.array([25.0, 30.0, 25.0, 20.0]),  # All too high
        np.array([2.0, 3.0, 1.0, 2.0]),      # All too low
        np.array([10.0, 15.0, 7.0, 12.0]),   # All within range
        np.array([100.0, 5.0, 15.0, 5.0]),   # Mixed case
    ]
    
    # Import the constraint function
    from predction01 import apply_smart_prediction_constraints
    
    print("Testing 10-day rolling range constraints...")
    print("=" * 60)
    
    for i, raw_predictions in enumerate(test_cases):
        print(f"\nTest Case {i+1}: Raw predictions = {raw_predictions}")
        
        # Apply constraints
        constrained = apply_smart_prediction_constraints(raw_predictions, df)
        
        # Check results for each player
        for j, (_, row) in enumerate(df.iterrows()):
            player_name = row['Name']
            raw_pred = raw_predictions[j]
            constrained_pred = constrained[j]
            
            rolling_min = row['rolling_min_fpts_10']
            rolling_max = row['rolling_max_fpts_10']
            five_game_avg = row['5_game_avg']
            
            print(f"  {player_name}:")
            print(f"    Raw prediction: {raw_pred:.2f}")
            print(f"    Constrained prediction: {constrained_pred:.2f}")
            
            if not np.isnan(rolling_min) and not np.isnan(rolling_max):
                # Player has 10-day rolling data
                range_expansion = max(3.0, (rolling_max - rolling_min) * 0.2)
                expected_min = max(0, rolling_min - range_expansion)
                expected_max = rolling_max + range_expansion
                
                # Ensure minimum range
                min_range = 5.0
                current_range = expected_max - expected_min
                if current_range < min_range:
                    center = (expected_max + expected_min) / 2
                    expected_min = max(0, center - min_range / 2)
                    expected_max = center + min_range / 2
                
                print(f"    10-day range: {rolling_min:.2f} - {rolling_max:.2f}")
                print(f"    Expected constraint range: {expected_min:.2f} - {expected_max:.2f}")
                
                # Check if constraint is working properly
                if raw_pred > expected_max:
                    # Should be constrained but with soft scaling
                    excess = raw_pred - expected_max
                    expected_constrained = expected_max + np.log1p(excess) * 2
                    expected_constrained = min(expected_constrained, 45.0)  # Final cap
                    print(f"    Expected constrained (high): {expected_constrained:.2f}")
                elif raw_pred < expected_min:
                    # Should be constrained but with soft scaling
                    deficit = expected_min - raw_pred
                    expected_constrained = expected_min - np.log1p(deficit) * 2
                    expected_constrained = max(expected_constrained, 0.0)  # Final floor
                    print(f"    Expected constrained (low): {expected_constrained:.2f}")
                else:
                    print(f"    Within range - no constraint needed")
            else:
                # Player uses 5-game average fallback
                print(f"    Using 5-game average fallback: {five_game_avg:.2f}")
                if five_game_avg > 0:
                    expected_min = max(0, five_game_avg - 8)
                    expected_max = five_game_avg + 12
                else:
                    expected_min = 0
                    expected_max = 15
                print(f"    Fallback constraint range: {expected_min:.2f} - {expected_max:.2f}")
            
            print()

def test_realistic_scenarios():
    """Test with realistic MLB player scenarios"""
    
    print("\nTesting realistic MLB scenarios...")
    print("=" * 60)
    
    # Realistic MLB player profiles
    realistic_data = {
        'Name': [
            'Star Player', 'Consistent Player', 'Streaky Player', 
            'Rookie Player', 'Veteran Slump', 'Hot Streak Player'
        ],
        'rolling_min_fpts_10': [12.0, 6.0, 0.0, np.nan, 2.0, 8.0],
        'rolling_max_fpts_10': [28.0, 14.0, 25.0, np.nan, 8.0, 35.0],
        '5_game_avg': [20.0, 10.0, 12.0, 8.0, 5.0, 22.0],
        'predicted_dk_fpts': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    
    df = pd.DataFrame(realistic_data)
    
    # Test with realistic model predictions
    raw_predictions = np.array([45.0, 18.0, 30.0, 25.0, 15.0, 50.0])
    
    from predction01 import apply_smart_prediction_constraints
    
    constrained = apply_smart_prediction_constraints(raw_predictions, df)
    
    print("\nRealistic Scenario Results:")
    print("-" * 40)
    
    for i, (_, row) in enumerate(df.iterrows()):
        player_name = row['Name']
        raw_pred = raw_predictions[i]
        constrained_pred = constrained[i]
        
        rolling_min = row['rolling_min_fpts_10']
        rolling_max = row['rolling_max_fpts_10']
        
        print(f"{player_name}:")
        print(f"  Raw prediction: {raw_pred:.2f}")
        print(f"  Constrained prediction: {constrained_pred:.2f}")
        
        if not np.isnan(rolling_min) and not np.isnan(rolling_max):
            print(f"  10-day range: {rolling_min:.2f} - {rolling_max:.2f}")
            
            # Check if constraint is reasonable
            range_expansion = max(3.0, (rolling_max - rolling_min) * 0.2)
            expected_min = max(0, rolling_min - range_expansion)
            expected_max = rolling_max + range_expansion
            
            # Ensure minimum range
            min_range = 5.0
            current_range = expected_max - expected_min
            if current_range < min_range:
                center = (expected_max + expected_min) / 2
                expected_min = max(0, center - min_range / 2)
                expected_max = center + min_range / 2
            
            print(f"  Allowed range: {expected_min:.2f} - {expected_max:.2f}")
            
            # Validate constraint worked
            if raw_pred > expected_max:
                reduction = raw_pred - constrained_pred
                print(f"  Reduction applied: {reduction:.2f} points")
            elif raw_pred < expected_min:
                increase = constrained_pred - raw_pred
                print(f"  Increase applied: {increase:.2f} points")
            else:
                print(f"  No constraint needed (within range)")
        else:
            print(f"  Using 5-game average fallback")
        
        print()

def analyze_constraint_effectiveness():
    """Analyze how effective the constraints are at reducing outliers"""
    
    print("\nAnalyzing constraint effectiveness...")
    print("=" * 60)
    
    # Generate a large sample of test data
    np.random.seed(42)
    n_players = 100
    
    # Create diverse player profiles
    rolling_mins = np.random.uniform(0, 15, n_players)
    rolling_maxs = rolling_mins + np.random.uniform(5, 20, n_players)
    five_game_avgs = (rolling_mins + rolling_maxs) / 2 + np.random.normal(0, 2, n_players)
    
    # Set some players to have no 10-day data
    no_data_mask = np.random.choice([True, False], n_players, p=[0.2, 0.8])
    rolling_mins[no_data_mask] = np.nan
    rolling_maxs[no_data_mask] = np.nan
    
    test_data = {
        'Name': [f'Player_{i}' for i in range(n_players)],
        'rolling_min_fpts_10': rolling_mins,
        'rolling_max_fpts_10': rolling_maxs,
        '5_game_avg': five_game_avgs,
        'predicted_dk_fpts': np.zeros(n_players)
    }
    
    df = pd.DataFrame(test_data)
    
    # Generate problematic raw predictions (many outliers)
    raw_predictions = np.random.uniform(0, 100, n_players)
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_players, 10, replace=False)
    raw_predictions[outlier_indices] = np.random.uniform(80, 150, 10)
    
    from predction01 import apply_smart_prediction_constraints
    
    constrained = apply_smart_prediction_constraints(raw_predictions, df)
    
    # Analyze results
    print(f"Raw predictions statistics:")
    print(f"  Min: {raw_predictions.min():.2f}")
    print(f"  Max: {raw_predictions.max():.2f}")
    print(f"  Mean: {raw_predictions.mean():.2f}")
    print(f"  Std: {raw_predictions.std():.2f}")
    print(f"  Predictions > 45: {(raw_predictions > 45).sum()}")
    print(f"  Predictions > 30: {(raw_predictions > 30).sum()}")
    
    print(f"\nConstrained predictions statistics:")
    print(f"  Min: {constrained.min():.2f}")
    print(f"  Max: {constrained.max():.2f}")
    print(f"  Mean: {constrained.mean():.2f}")
    print(f"  Std: {constrained.std():.2f}")
    print(f"  Predictions > 45: {(constrained > 45).sum()}")
    print(f"  Predictions > 30: {(constrained > 30).sum()}")
    
    # Check how many predictions were significantly adjusted
    large_reductions = (raw_predictions - constrained) > 10
    large_increases = (constrained - raw_predictions) > 10
    
    print(f"\nAdjustment statistics:")
    print(f"  Large reductions (>10 points): {large_reductions.sum()}")
    print(f"  Large increases (>10 points): {large_increases.sum()}")
    print(f"  Predictions unchanged: {(np.abs(raw_predictions - constrained) < 0.1).sum()}")
    
    # Check players with 10-day data vs fallback
    has_10_day_data = ~(np.isnan(rolling_mins) | np.isnan(rolling_maxs))
    print(f"\nPlayer data availability:")
    print(f"  Players with 10-day data: {has_10_day_data.sum()}")
    print(f"  Players using fallback: {(~has_10_day_data).sum()}")

if __name__ == "__main__":
    test_ten_day_rolling_constraint()
    test_realistic_scenarios()
    analyze_constraint_effectiveness()
