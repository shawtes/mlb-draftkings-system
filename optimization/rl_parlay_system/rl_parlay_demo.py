#!/usr/bin/env python3
"""
RL Parlay Demo
Demonstration script for the RL parlay generation system
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def create_demo_data():
    """Create demo data for testing the RL system"""
    print("🎯 Creating Demo Data for RL Parlay System")
    print("=" * 50)
    
    # Create sample player data
    np.random.seed(42)
    n_players = 100
    n_weeks = 18
    n_years = 3
    
    demo_data = []
    
    # Sample teams and positions
    teams = ['KC', 'BUF', 'MIA', 'NE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 'HOU', 
             'IND', 'JAX', 'TEN', 'DEN', 'LV', 'LAC', 'DAL', 'NYG', 'PHI', 'WAS',
             'CHI', 'DET', 'GB', 'MIN', 'ATL', 'CAR', 'NO', 'TB', 'ARI', 'LAR', 'SF', 'SEA']
    
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    
    # Create players
    for year in range(2022, 2025):
        for week in range(1, 19):
            for i in range(n_players):
                team = np.random.choice(teams)
                position = np.random.choice(positions)
                
                # Generate realistic projections based on position
                if position == 'QB':
                    dk_proj = np.random.normal(18, 5)
                    pass_yds = np.random.normal(250, 50)
                    rush_yds = np.random.normal(20, 10)
                    rec_yds = 0
                    receptions = 0
                elif position == 'RB':
                    dk_proj = np.random.normal(15, 4)
                    pass_yds = 0
                    rush_yds = np.random.normal(80, 25)
                    rec_yds = np.random.normal(30, 15)
                    receptions = np.random.normal(3, 1.5)
                elif position in ['WR', 'TE']:
                    dk_proj = np.random.normal(12, 3)
                    pass_yds = 0
                    rush_yds = np.random.normal(5, 5)
                    rec_yds = np.random.normal(60, 20)
                    receptions = np.random.normal(4, 1.5)
                else:  # K, DST
                    dk_proj = np.random.normal(8, 2)
                    pass_yds = 0
                    rush_yds = 0
                    rec_yds = 0
                    receptions = 0
                
                # Ensure positive values
                dk_proj = max(0, dk_proj)
                pass_yds = max(0, pass_yds)
                rush_yds = max(0, rush_yds)
                rec_yds = max(0, rec_yds)
                receptions = max(0, receptions)
                
                # Generate actual outcomes with some noise
                dk_actual = dk_proj * np.random.normal(1.0, 0.3)
                pass_actual = pass_yds * np.random.normal(1.0, 0.25)
                rush_actual = rush_yds * np.random.normal(1.0, 0.3)
                rec_actual = rec_yds * np.random.normal(1.0, 0.25)
                rec_actual_count = receptions * np.random.normal(1.0, 0.3)
                
                # Calculate accuracy metrics
                dk_accuracy = dk_actual / (dk_proj + 0.1)
                pass_accuracy = pass_actual / (pass_yds + 0.1)
                rush_accuracy = rush_actual / (rush_yds + 0.1)
                rec_accuracy = rec_actual / (rec_yds + 0.1)
                rec_count_accuracy = rec_actual_count / (receptions + 0.1)
                
                # Calculate hit rates (70% of projection as threshold)
                dk_hit = 1 if dk_actual >= dk_proj * 0.7 else 0
                pass_hit = 1 if pass_actual >= pass_yds * 0.7 else 0
                rush_hit = 1 if rush_actual >= rush_yds * 0.7 else 0
                rec_hit = 1 if rec_actual >= rec_yds * 0.7 else 0
                rec_count_hit = 1 if rec_actual_count >= receptions * 0.7 else 0
                
                player_data = {
                    'year': year,
                    'week': week,
                    'player_id': f"P{year}{week:02d}{i:03d}",
                    'player_name_proj': f"Player {i+1}",
                    'team_proj': team,
                    'position_proj': position,
                    'opponent': np.random.choice([t for t in teams if t != team]),
                    'game_id': f"G{year}{week:02d}{i//10:02d}",
                    'projected_dk_points': dk_proj,
                    'projected_passing_yds': pass_yds,
                    'projected_rushing_yds': rush_yds,
                    'projected_receiving_yds': rec_yds,
                    'projected_receptions': receptions,
                    'actual_dk_points': dk_actual,
                    'actual_passing_yds': pass_actual,
                    'actual_rushing_yds': rush_actual,
                    'actual_receiving_yds': rec_actual,
                    'actual_receptions': rec_actual_count,
                    'salary': np.random.randint(3000, 10000),
                    'injury_status': np.random.choice(['', 'Questionable', 'Probable']),
                    'temperature': np.random.randint(20, 80),
                    'wind_speed': np.random.randint(0, 20),
                    'humidity': np.random.randint(30, 90),
                    'surface': np.random.choice(['Grass', 'Turf']),
                    'stadium': f"Stadium {i%10}",
                    'weather': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Snow']),
                    'total': np.random.randint(40, 60),
                    'spread': np.random.randint(-14, 15),
                    'home_score': np.random.randint(10, 40),
                    'away_score': np.random.randint(10, 40),
                    'dk_points_accuracy_mean': dk_accuracy,
                    'dk_points_accuracy_std': 0.2,
                    'passing_yds_accuracy_mean': pass_accuracy,
                    'passing_yds_accuracy_std': 0.15,
                    'rushing_yds_accuracy_mean': rush_accuracy,
                    'rushing_yds_accuracy_std': 0.25,
                    'receiving_yds_accuracy_mean': rec_accuracy,
                    'receiving_yds_accuracy_std': 0.2,
                    'receptions_accuracy_mean': rec_count_accuracy,
                    'receptions_accuracy_std': 0.25,
                    'dk_points_hit_mean': dk_hit,
                    'passing_yds_hit_mean': pass_hit,
                    'rushing_yds_hit_mean': rush_hit,
                    'receiving_yds_hit_mean': rec_hit,
                    'receptions_hit_mean': rec_count_hit
                }
                
                demo_data.append(player_data)
    
    # Create DataFrame
    demo_df = pd.DataFrame(demo_data)
    
    # Save demo data
    demo_file = "rl_demo_data.csv"
    demo_df.to_csv(demo_file, index=False)
    
    print(f"✅ Demo data created: {demo_file}")
    print(f"   Records: {len(demo_df)}")
    print(f"   Players: {demo_df['player_id'].nunique()}")
    print(f"   Teams: {demo_df['team_proj'].nunique()}")
    print(f"   Years: {demo_df['year'].nunique()}")
    print(f"   Weeks: {demo_df['week'].nunique()}")
    
    return demo_df

def run_demo():
    """Run the RL parlay demo"""
    print("\n🚀 RL Parlay System Demo")
    print("=" * 50)
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Import RL components
    try:
        from rl_parlay_environment import ParlayEnvironment
        from rl_parlay_agent import PPOAgent
        
        print("\n🤖 Creating RL Environment...")
        env = ParlayEnvironment(demo_data)
        
        print("✅ Environment created successfully!")
        print(f"   State dimension: {env.observation_space.shape[0]}")
        print(f"   Action dimensions: {env.action_space.nvec}")
        
        print("\n🧠 Creating RL Agent...")
        state_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()
        agent = PPOAgent(state_dim, action_dims)
        
        print("✅ Agent created successfully!")
        print(f"   Device: {agent.device}")
        
        print("\n🎯 Testing Parlay Generation...")
        
        # Generate a few test parlays
        for i in range(3):
            print(f"\n--- Test Parlay {i+1} ---")
            parlay = agent.generate_parlay(env)
            
            print(f"Legs: {parlay['num_legs']}")
            print(f"Hit Rate: {parlay['combined_hit_rate']:.2%}")
            print(f"Odds: +{parlay['estimated_odds']:.0f}")
            print(f"Expected Value: ${parlay['expected_value']:.2f}")
            
            print("Legs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"  {j}. {leg['player']} ({leg['team']}) - {leg['prop']} O{leg['line']:.1f} ({leg['hit_rate']:.1%})")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python rl_parlay_trainer.py --api-key YOUR_KEY' to train on real data")
        print("2. Run 'python rl_parlay_gui.py' to use the GUI interface")
        print("3. Check RL_PARLAY_README.md for detailed documentation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements_rl.txt")
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    run_demo()
