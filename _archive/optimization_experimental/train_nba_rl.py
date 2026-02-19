#!/usr/bin/env python3
"""
Quick NBA RL Training Script
Simplified version for training the NBA RL model
"""

import pandas as pd
import numpy as np
import sys
import os

# Add path to RL system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rl_parlay_system'))

def train_nba_rl():
    """Train NBA RL model on historical data"""
    
    print("🏀 NBA RL Model Training")
    print("=" * 70)
    
    # Load training data
    training_file = 'nba_training_data.csv'
    if not os.path.exists(training_file):
        print(f"❌ Training data not found: {training_file}")
        print("   Run: python3 nba_data_collector.py first")
        return
    
    print(f"📥 Loading training data from {training_file}...")
    data = pd.read_csv(training_file)
    print(f"✅ Loaded {len(data)} records")
    
    # Filter to active players only
    if 'projected_points' in data.columns:
        data = data[data['projected_points'] > 0].copy()
        print(f"✅ Filtered to {len(data)} active players")
    
    # Check for required columns
    required_cols = ['player_id', 'player_name_proj', 'team_proj', 'position_proj',
                     'projected_points', 'projected_rebounds', 'projected_assists',
                     'projected_steals', 'projected_blocks', 'projected_three_pointers']
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
        print("   Using synthetic data instead")
        
        # Generate synthetic NBA data
        nba_data = []
        teams = data['team_proj'].unique() if 'team_proj' in data.columns else ['LAL', 'BOS', 'MIA']
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        
        for i in range(100):
            nba_data.append({
                'player_id': f'player_{i}',
                'player_name_proj': f'Player {i}',
                'team_proj': np.random.choice(teams),
                'position_proj': np.random.choice(positions),
                'projected_points': max(0, np.random.normal(15, 5)),
                'projected_rebounds': max(0, np.random.normal(6, 2)),
                'projected_assists': max(0, np.random.normal(5, 2)),
                'projected_steals': max(0, np.random.normal(1.2, 0.5)),
                'projected_blocks': max(0, np.random.normal(0.8, 0.4)),
                'projected_three_pointers': max(0, np.random.normal(2.5, 1.0)),
                'points_accuracy_std': 0.25,
                'rebounds_accuracy_std': 0.20,
                'assists_accuracy_std': 0.20,
                'steals_accuracy_std': 0.25,
                'blocks_accuracy_std': 0.30,
                'three_pointers_accuracy_std': 0.35,
                'points_hit_mean': 0.70,
                'rebounds_hit_mean': 0.75,
                'assists_hit_mean': 0.75,
                'steals_hit_mean': 0.68,
                'blocks_hit_mean': 0.65,
                'three_pointers_hit_mean': 0.60
            })
        
        data = pd.DataFrame(nba_data)
        print(f"✅ Generated {len(data)} synthetic players")
    
    # Split training and test data (90/10 split)
    split_idx = int(len(data) * 0.9)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Save split data
    train_data.to_csv('nba_train_split.csv', index=False)
    test_data.to_csv('nba_test_split.csv', index=False)
    
    print("\n✅ NBA RL training data prepared!")
    print("   Training file: nba_train_split.csv")
    print("   Test file: nba_test_split.csv")
    print("\n📝 Note: Full RL training requires GPU and several hours.")
    print("   For now, using the historical variance approach.")

if __name__ == "__main__":
    train_nba_rl()











