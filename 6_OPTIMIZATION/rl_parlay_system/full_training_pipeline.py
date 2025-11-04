#!/usr/bin/env python3
"""
Full Training Pipeline for RL Parlay System
Trains on full dataset and tests on 2025 Week 7
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from rl_parlay_environment import ParlayEnvironment
from rl_parlay_agent import PPOAgent
from simple_parlay_generator import SimpleParlayGenerator

class FullTrainingPipeline:
    """Complete training and testing pipeline"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.training_data = None
        self.test_data = None
        self.agent = None
        
    def collect_full_dataset(self):
        """Collect comprehensive training data"""
        print("📊 Collecting Full Dataset for Training")
        print("=" * 50)
        
        # Import the data collector
        from rl_parlay_data_collector import RLDataCollector
        
        collector = RLDataCollector(self.api_key)
        
        # Collect 3 years of data
        years = [2022, 2023, 2024]
        print(f"🔄 Collecting data for years: {years}")
        
        data = collector.collect_historical_data(years)
        
        if data and len(data['projections']) > 0:
            print(f"✅ Data collection successful!")
            print(f"   Projections: {len(data['projections'])}")
            print(f"   Actuals: {len(data['actuals'])}")
            print(f"   Games: {len(data['games'])}")
            
            # Create training dataset
            self.training_data = collector.create_training_dataset([
                'rl_training_data/projections_*.csv',
                'rl_training_data/actuals_*.csv', 
                'rl_training_data/games_*.csv'
            ])
            
            print(f"✅ Training dataset created: {len(self.training_data)} records")
            return True
        else:
            print("❌ Data collection failed!")
            return False
    
    def collect_week7_2025_data(self):
        """Collect 2025 Week 7 data for testing"""
        print("\n📊 Collecting 2025 Week 7 Test Data")
        print("=" * 50)
        
        from rl_parlay_data_collector import RLDataCollector
        
        collector = RLDataCollector(self.api_key)
        
        # Collect Week 7 2025 data
        print("🔄 Collecting 2025 Week 7 data...")
        
        try:
            # Get projections for Week 7 2025
            projections = collector.api.get_player_projections_by_week('2025REG', 7)
            actuals = collector.api.get_player_game_stats_by_week('2025REG', 7)
            games = collector.api.get_games_by_week('2025REG', 7)
            
            if projections and actuals:
                # Create test dataset similar to training data
                test_projections = []
                for proj in projections:
                    test_projections.append({
                        'year': 2025,
                        'week': 7,
                        'player_id': proj.get('PlayerID'),
                        'player_name_proj': proj.get('Name'),
                        'team_proj': proj.get('Team'),
                        'position_proj': proj.get('Position'),
                        'opponent': proj.get('Opponent'),
                        'game_id': proj.get('GameID'),
                        'projected_dk_points': proj.get('FantasyPointsDraftKings', 0),
                        'projected_passing_yds': proj.get('PassingYards', 0),
                        'projected_rushing_yds': proj.get('RushingYards', 0),
                        'projected_receiving_yds': proj.get('ReceivingYards', 0),
                        'projected_receptions': proj.get('Receptions', 0),
                        'projected_passing_tds': proj.get('PassingTouchdowns', 0),
                        'projected_rushing_tds': proj.get('RushingTouchdowns', 0),
                        'projected_receiving_tds': proj.get('ReceivingTouchdowns', 0),
                        'projected_interceptions': proj.get('Interceptions', 0),
                        'projected_fumbles': proj.get('FumblesLost', 0),
                        'salary': proj.get('Salary', 0),
                        'injury_status': proj.get('InjuryStatus', ''),
                        'weather': proj.get('Weather', ''),
                        'stadium': proj.get('Stadium', ''),
                        'surface': proj.get('Surface', ''),
                        'temperature': proj.get('Temperature', 0),
                        'wind_speed': proj.get('WindSpeed', 0),
                        'humidity': proj.get('Humidity', 0)
                    })
                
                test_actuals = []
                for actual in actuals:
                    test_actuals.append({
                        'year': 2025,
                        'week': 7,
                        'player_id': actual.get('PlayerID'),
                        'player_name_actual': actual.get('Name'),
                        'team_actual': actual.get('Team'),
                        'position_actual': actual.get('Position'),
                        'opponent': actual.get('Opponent'),
                        'game_id': actual.get('GameID'),
                        'actual_dk_points': actual.get('FantasyPointsDraftKings', 0),
                        'actual_passing_yds': actual.get('PassingYards', 0),
                        'actual_rushing_yds': actual.get('RushingYards', 0),
                        'actual_receiving_yds': actual.get('ReceivingYards', 0),
                        'actual_receptions': actual.get('Receptions', 0),
                        'actual_passing_tds': actual.get('PassingTouchdowns', 0),
                        'actual_rushing_tds': actual.get('RushingTouchdowns', 0),
                        'actual_receiving_tds': actual.get('ReceivingTouchdowns', 0),
                        'actual_interceptions': actual.get('Interceptions', 0),
                        'actual_fumbles': actual.get('FumblesLost', 0),
                        'game_date': actual.get('Date', ''),
                        'home_team': actual.get('HomeTeam', ''),
                        'away_team': actual.get('AwayTeam', ''),
                        'final_score_home': actual.get('HomeScore', 0),
                        'final_score_away': actual.get('AwayScore', 0),
                        'game_total': actual.get('Total', 0),
                        'spread': actual.get('Spread', 0)
                    })
                
                # Merge projections with actuals
                proj_df = pd.DataFrame(test_projections)
                actual_df = pd.DataFrame(test_actuals)
                
                self.test_data = pd.merge(
                    proj_df, 
                    actual_df, 
                    on=['year', 'week', 'player_id'], 
                    how='inner',
                    suffixes=('_proj', '_actual')
                )
                
                # Add historical accuracy metrics (use training data averages)
                if self.training_data is not None:
                    player_stats = self.training_data.groupby('player_id').agg({
                        'dk_points_accuracy_mean': 'mean',
                        'dk_points_accuracy_std': 'mean',
                        'passing_yds_accuracy_mean': 'mean',
                        'passing_yds_accuracy_std': 'mean',
                        'rushing_yds_accuracy_mean': 'mean',
                        'rushing_yds_accuracy_std': 'mean',
                        'receiving_yds_accuracy_mean': 'mean',
                        'receiving_yds_accuracy_std': 'mean',
                        'receptions_accuracy_mean': 'mean',
                        'receptions_accuracy_std': 'mean',
                        'dk_points_hit_mean': 'mean',
                        'passing_yds_hit_mean': 'mean',
                        'rushing_yds_hit_mean': 'mean',
                        'receiving_yds_hit_mean': 'mean',
                        'receptions_hit_mean': 'mean'
                    }).reset_index()
                    
                    # Merge with test data
                    self.test_data = pd.merge(self.test_data, player_stats, on='player_id', how='left')
                    
                    # Fill missing values with defaults
                    for col in player_stats.columns:
                        if col != 'player_id':
                            self.test_data[col] = self.test_data[col].fillna(0.5)
                
                print(f"✅ Test data created: {len(self.test_data)} records")
                print(f"   Players: {self.test_data['player_id'].nunique()}")
                print(f"   Teams: {self.test_data['team_proj'].nunique()}")
                
                return True
            else:
                print("❌ No Week 7 2025 data available")
                return False
                
        except Exception as e:
            print(f"❌ Error collecting Week 7 data: {e}")
            return False
    
    def train_agent(self, num_episodes: int = 2000):
        """Train the RL agent on full dataset"""
        print("\n🤖 Training RL Agent on Full Dataset")
        print("=" * 50)
        
        if self.training_data is None:
            print("❌ No training data available")
            return False
        
        # Create environment
        env = ParlayEnvironment(self.training_data)
        print(f"✅ Environment created: {env.state_dim} state dim")
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()
        self.agent = PPOAgent(state_dim, action_dims)
        print(f"✅ Agent created: {self.agent.device}")
        
        # Training loop
        print(f"🏋️ Training for {num_episodes} episodes...")
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            episode_reward = self.agent.train_episode(env)
            
            if episode % 200 == 0:
                # Evaluate current policy
                eval_metrics = self.agent.evaluate(env, num_episodes=10)
                print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                      f"Eval Reward={eval_metrics['avg_reward']:.2f}, "
                      f"Hit Rate={eval_metrics['avg_hit_rate']:.2%}, "
                      f"Expected Value=${eval_metrics['avg_expected_value']:.2f}")
                
                # Save best model
                if eval_metrics['avg_reward'] > best_reward:
                    best_reward = eval_metrics['avg_reward']
                    self.agent.save_model("rl_models/full_trained_model.pth")
                    print(f"💾 New best model saved! Reward: {best_reward:.2f}")
        
        print(f"\n✅ Training completed! Best reward: {best_reward:.2f}")
        return True
    
    def test_on_week7_2025(self, num_parlays: int = 20):
        """Test trained agent on 2025 Week 7 data"""
        print("\n🧪 Testing on 2025 Week 7 Data")
        print("=" * 50)
        
        if self.test_data is None:
            print("❌ No test data available")
            return
        
        if self.agent is None:
            print("❌ No trained agent available")
            return
        
        # Create test environment
        test_env = ParlayEnvironment(self.test_data)
        print(f"✅ Test environment created: {len(self.test_data)} players")
        
        # Generate parlays
        print(f"🎲 Generating {num_parlays} parlays...")
        
        parlays = []
        for i in range(num_parlays):
            parlay = self.agent.generate_parlay(test_env)
            if parlay.legs:  # Only count parlays with legs
                parlays.append(parlay)
        
        # Display results
        print(f"\n📊 Generated {len(parlays)} parlays:")
        
        for i, parlay in enumerate(parlays[:10], 1):  # Show first 10
            print(f"\n--- Parlay {i} ---")
            print(f"Legs: {len(parlay.legs)}")
            print(f"Hit Rate: {parlay.combined_hit_rate:.2%}")
            print(f"Odds: +{parlay.estimated_odds:.0f}")
            print(f"Expected Value: ${parlay.expected_value:.2f}")
            
            print("Legs:")
            for j, leg in enumerate(parlay.legs, 1):
                print(f"  {j}. {leg.player_name} ({leg.team}) - {leg.prop_type} O{leg.line:.1f} ({leg.hit_rate:.1%})")
        
        # Summary statistics
        if parlays:
            stats = {
                'total_parlays': len(parlays),
                'avg_legs': np.mean([len(p.legs) for p in parlays]),
                'avg_hit_rate': np.mean([p.combined_hit_rate for p in parlays]),
                'avg_odds': np.mean([p.estimated_odds for p in parlays]),
                'avg_expected_value': np.mean([p.expected_value for p in parlays]),
                'max_hit_rate': max([p.combined_hit_rate for p in parlays]),
                'min_hit_rate': min([p.combined_hit_rate for p in parlays]),
                'profitable_parlays': len([p for p in parlays if p.expected_value > 0])
            }
            
            print(f"\n📈 Week 7 2025 Test Results:")
            print(f"   Total Parlays: {stats['total_parlays']}")
            print(f"   Average Legs: {stats['avg_legs']:.1f}")
            print(f"   Average Hit Rate: {stats['avg_hit_rate']:.2%}")
            print(f"   Average Odds: +{stats['avg_odds']:.0f}")
            print(f"   Average Expected Value: ${stats['avg_expected_value']:.2f}")
            print(f"   Hit Rate Range: {stats['min_hit_rate']:.1%} - {stats['max_hit_rate']:.1%}")
            print(f"   Profitable Parlays: {stats['profitable_parlays']} ({stats['profitable_parlays']/stats['total_parlays']*100:.1f}%)")
            
            # Save results
            results_file = f"rl_results/week7_2025_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("rl_results", exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"💾 Results saved: {results_file}")
        
        return parlays
    
    def compare_with_simple_generator(self, num_parlays: int = 20):
        """Compare RL agent with simple rule-based generator"""
        print("\n⚖️ Comparing RL Agent vs Simple Generator")
        print("=" * 50)
        
        if self.test_data is None:
            print("❌ No test data available")
            return
        
        # RL Agent results
        if self.agent is not None:
            test_env = ParlayEnvironment(self.test_data)
            rl_parlays = []
            for _ in range(num_parlays):
                parlay = self.agent.generate_parlay(test_env)
                if parlay.legs:
                    rl_parlays.append(parlay)
        else:
            rl_parlays = []
        
        # Simple generator results
        simple_gen = SimpleParlayGenerator(self.test_data)
        simple_parlays = simple_gen.generate_multiple_parlays(num_parlays, max_legs=4)
        
        # Compare results
        print(f"📊 Comparison Results ({num_parlays} parlays each):")
        
        if rl_parlays:
            rl_stats = {
                'avg_legs': np.mean([len(p.legs) for p in rl_parlays]),
                'avg_hit_rate': np.mean([p.combined_hit_rate for p in rl_parlays]),
                'avg_odds': np.mean([p.estimated_odds for p in rl_parlays]),
                'avg_expected_value': np.mean([p.expected_value for p in rl_parlays]),
                'profitable': len([p for p in rl_parlays if p.expected_value > 0])
            }
            
            print(f"\n🤖 RL Agent:")
            print(f"   Average Legs: {rl_stats['avg_legs']:.1f}")
            print(f"   Average Hit Rate: {rl_stats['avg_hit_rate']:.2%}")
            print(f"   Average Odds: +{rl_stats['avg_odds']:.0f}")
            print(f"   Average Expected Value: ${rl_stats['avg_expected_value']:.2f}")
            print(f"   Profitable Parlays: {rl_stats['profitable']} ({rl_stats['profitable']/len(rl_parlays)*100:.1f}%)")
        else:
            print(f"\n🤖 RL Agent: No parlays generated")
        
        if simple_parlays:
            simple_stats = {
                'avg_legs': np.mean([len(p.legs) for p in simple_parlays]),
                'avg_hit_rate': np.mean([p.combined_hit_rate for p in simple_parlays]),
                'avg_odds': np.mean([p.estimated_odds for p in simple_parlays]),
                'avg_expected_value': np.mean([p.expected_value for p in simple_parlays]),
                'profitable': len([p for p in simple_parlays if p.expected_value > 0])
            }
            
            print(f"\n📋 Simple Generator:")
            print(f"   Average Legs: {simple_stats['avg_legs']:.1f}")
            print(f"   Average Hit Rate: {simple_stats['avg_hit_rate']:.2%}")
            print(f"   Average Odds: +{simple_stats['avg_odds']:.0f}")
            print(f"   Average Expected Value: ${simple_stats['avg_expected_value']:.2f}")
            print(f"   Profitable Parlays: {simple_stats['profitable']} ({simple_stats['profitable']/len(simple_parlays)*100:.1f}%)")
        else:
            print(f"\n📋 Simple Generator: No parlays generated")
    
    def run_full_pipeline(self, num_episodes: int = 2000, num_test_parlays: int = 20):
        """Run the complete training and testing pipeline"""
        print("🚀 Full RL Parlay Training & Testing Pipeline")
        print("=" * 60)
        
        # Step 1: Collect training data
        if not self.collect_full_dataset():
            print("❌ Pipeline failed at data collection")
            return False
        
        # Step 2: Collect test data
        if not self.collect_week7_2025_data():
            print("❌ Pipeline failed at test data collection")
            return False
        
        # Step 3: Train agent
        if not self.train_agent(num_episodes):
            print("❌ Pipeline failed at training")
            return False
        
        # Step 4: Test on Week 7 2025
        self.test_on_week7_2025(num_test_parlays)
        
        # Step 5: Compare with simple generator
        self.compare_with_simple_generator(num_test_parlays)
        
        print("\n🎉 Full pipeline completed successfully!")
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full RL Parlay Training Pipeline")
    parser.add_argument("--api-key", type=str, required=True, help="SportsData.io API key")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--test-parlays", type=int, default=20, help="Number of test parlays")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = FullTrainingPipeline(args.api_key)
    
    # Run full pipeline
    success = pipeline.run_full_pipeline(args.episodes, args.test_parlays)
    
    if success:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed!")

if __name__ == "__main__":
    main()






