#!/usr/bin/env python3
"""
RL Parlay System - Main Entry Point
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def main():
    """Main entry point for RL Parlay System"""
    parser = argparse.ArgumentParser(description="RL Parlay Generation System")
    parser.add_argument("command", choices=["demo", "train", "gui", "collect"], 
                       help="Command to run")
    parser.add_argument("--api-key", type=str, help="SportsData.io API key")
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024], 
                       help="Years to collect data for")
    parser.add_argument("--episodes", type=int, default=1000, 
                       help="Number of training episodes")
    parser.add_argument("--force-collect", action="store_true", 
                       help="Force data collection even if data exists")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        print("🎯 Running RL Parlay Demo...")
        from rl_parlay_demo import run_demo
        run_demo()
        
    elif args.command == "collect":
        if not args.api_key:
            print("❌ API key required for data collection")
            return
        print("📊 Collecting historical data...")
        from rl_parlay_trainer import RLParlayTrainer
        trainer = RLParlayTrainer(args.api_key)
        trainer.collect_data(args.years, args.force_collect)
        
    elif args.command == "train":
        if not args.api_key:
            print("❌ API key required for training")
            return
        print("🤖 Training RL agent...")
        from rl_parlay_trainer import RLParlayTrainer
        trainer = RLParlayTrainer(args.api_key)
        trainer.run_full_pipeline(args.years, args.episodes, args.force_collect)
        
    elif args.command == "gui":
        print("🎮 Starting RL Parlay GUI...")
        from rl_parlay_gui import main as gui_main
        gui_main()

if __name__ == "__main__":
    main()






