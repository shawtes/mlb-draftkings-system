#!/usr/bin/env python3
"""
RL Parlay Trainer
Main script to train the RL agent for parlay generation
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import argparse
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from .rl_parlay_data_collector import RLDataCollector
from .rl_parlay_environment import ParlayEnvironment
from .rl_parlay_agent import PPOAgent, train_rl_agent

class RLParlayTrainer:
    """Main trainer class for RL parlay generation"""
    
    def __init__(self, api_key: str, data_dir: str = "rl_training_data"):
        self.api_key = api_key
        self.data_dir = data_dir
        self.collector = RLDataCollector(api_key)
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs("rl_models", exist_ok=True)
        os.makedirs("rl_results", exist_ok=True)
    
    def collect_data(self, years: list = [2022, 2023, 2024], force_collect: bool = False):
        """Collect historical data for training"""
        print("📊 Data Collection Phase")
        print("=" * 40)
        
        # Check if data already exists
        existing_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if existing_files and not force_collect:
            print(f"✅ Found existing data files: {len(existing_files)}")
            print("   Use --force-collect to re-collect data")
            return existing_files
        
        # Collect new data
        print("🔄 Collecting historical data...")
        data = self.collector.collect_historical_data(years)
        
        if data and len(data['projections']) > 0:
            print(f"✅ Data collection successful!")
            print(f"   Projections: {len(data['projections'])}")
            print(f"   Actuals: {len(data['actuals'])}")
            print(f"   Games: {len(data['games'])}")
            return True
        else:
            print("❌ Data collection failed!")
            return False
    
    def prepare_training_data(self, data_files: list = None) -> pd.DataFrame:
        """Prepare training dataset from collected data"""
        print("\n🔄 Data Preparation Phase")
        print("=" * 40)
        
        if data_files is None:
            # Find the most recent data files
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No data files found. Run data collection first.")
            
            # Sort by modification time and get the most recent
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)
            
            # Find projections, actuals, and games files
            projections_file = None
            actuals_file = None
            games_file = None
            
            for file in csv_files:
                if 'projections' in file:
                    projections_file = os.path.join(self.data_dir, file)
                elif 'actuals' in file:
                    actuals_file = os.path.join(self.data_dir, file)
                elif 'games' in file:
                    games_file = os.path.join(self.data_dir, file)
            
            if not all([projections_file, actuals_file, games_file]):
                raise ValueError("Missing required data files. Run data collection first.")
            
            data_files = [projections_file, actuals_file, games_file]
        
        # Create training dataset
        training_data = self.collector.create_training_dataset(data_files)
        
        # Filter out invalid data
        training_data = training_data.dropna(subset=['player_id', 'projected_dk_points'])
        training_data = training_data[training_data['projected_dk_points'] > 0]
        
        print(f"✅ Training data prepared!")
        print(f"   Records: {len(training_data)}")
        print(f"   Players: {training_data['player_id'].nunique()}")
        print(f"   Games: {training_data['game_id'].nunique()}")
        print(f"   Weeks: {training_data['week'].nunique()}")
        
        # Save prepared data
        training_file = os.path.join(self.data_dir, "training_data_prepared.csv")
        training_data.to_csv(training_file, index=False)
        print(f"💾 Saved prepared data: {training_file}")
        
        return training_data
    
    def train_agent(self, training_data: pd.DataFrame, 
                   num_episodes: int = 1000,
                   model_name: str = None) -> PPOAgent:
        """Train the RL agent"""
        print("\n🤖 Agent Training Phase")
        print("=" * 40)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"rl_parlay_model_{timestamp}"
        
        model_path = os.path.join("rl_models", f"{model_name}.pth")
        
        # Train the agent
        agent = train_rl_agent(
            training_data=training_data,
            num_episodes=num_episodes,
            save_path=model_path
        )
        
        # Plot training progress
        plot_path = os.path.join("rl_results", f"{model_name}_training.png")
        agent.plot_training_progress(save_path=plot_path)
        
        # Final evaluation
        print("\n📊 Final Evaluation")
        print("-" * 20)
        env = ParlayEnvironment(training_data)
        eval_metrics = agent.evaluate(env, num_episodes=20)
        
        print(f"Average Reward: {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"Average Legs: {eval_metrics['avg_legs']:.1f}")
        print(f"Average Hit Rate: {eval_metrics['avg_hit_rate']:.2%}")
        print(f"Average Odds: +{eval_metrics['avg_odds']:.0f}")
        print(f"Average Expected Value: ${eval_metrics['avg_expected_value']:.2f}")
        
        # Save evaluation results
        results_path = os.path.join("rl_results", f"{model_name}_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        print(f"💾 Model saved: {model_path}")
        print(f"📊 Results saved: {results_path}")
        
        return agent
    
    def test_parlay_generation(self, agent: PPOAgent, training_data: pd.DataFrame, 
                             num_tests: int = 10):
        """Test parlay generation with trained agent"""
        print("\n🧪 Parlay Generation Testing")
        print("=" * 40)
        
        env = ParlayEnvironment(training_data)
        
        for i in range(num_tests):
            print(f"\n--- Test {i+1} ---")
            parlay = agent.generate_parlay(env)
            
            print(f"Legs: {parlay['num_legs']}")
            print(f"Hit Rate: {parlay['combined_hit_rate']:.2%}")
            print(f"Odds: +{parlay['estimated_odds']:.0f}")
            print(f"Expected Value: ${parlay['expected_value']:.2f}")
            
            print("Legs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"  {j}. {leg['player']} ({leg['team']}) - {leg['prop']} O{leg['line']:.1f} ({leg['hit_rate']:.1%})")
    
    def run_full_pipeline(self, years: list = [2022, 2023, 2024], 
                         num_episodes: int = 1000,
                         force_collect: bool = False):
        """Run the complete training pipeline"""
        print("🚀 RL Parlay Training Pipeline")
        print("=" * 50)
        
        # Step 1: Collect data
        data_collected = self.collect_data(years, force_collect)
        if not data_collected:
            print("❌ Pipeline failed at data collection")
            return None
        
        # Step 2: Prepare training data
        try:
            training_data = self.prepare_training_data()
        except Exception as e:
            print(f"❌ Pipeline failed at data preparation: {e}")
            return None
        
        # Step 3: Train agent
        try:
            agent = self.train_agent(training_data, num_episodes)
        except Exception as e:
            print(f"❌ Pipeline failed at training: {e}")
            return None
        
        # Step 4: Test parlay generation
        try:
            self.test_parlay_generation(agent, training_data)
        except Exception as e:
            print(f"❌ Pipeline failed at testing: {e}")
            return None
        
        print("\n🎉 Pipeline completed successfully!")
        return agent

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train RL agent for parlay generation")
    parser.add_argument("--api-key", type=str, required=True, help="SportsData.io API key")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024], 
                       help="Years to collect data for")
    parser.add_argument("--episodes", type=int, default=1000, 
                       help="Number of training episodes")
    parser.add_argument("--force-collect", action="store_true", 
                       help="Force data collection even if data exists")
    parser.add_argument("--model-name", type=str, 
                       help="Custom model name")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLParlayTrainer(args.api_key)
    
    # Run pipeline
    agent = trainer.run_full_pipeline(
        years=args.years,
        num_episodes=args.episodes,
        force_collect=args.force_collect
    )
    
    if agent:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
