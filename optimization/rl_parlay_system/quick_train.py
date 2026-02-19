#!/usr/bin/env python3
"""
Quick training script for RL parlay system
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from rl_parlay_environment import ParlayEnvironment
from rl_parlay_agent import PPOAgent

def quick_train():
    """Quick training session to get a working model"""
    print("🚀 Quick Training Session")
    print("=" * 40)
    
    # Load demo data
    if not os.path.exists('rl_demo_data.csv'):
        print("❌ Demo data not found. Run demo first.")
        return
    
    demo_data = pd.read_csv('rl_demo_data.csv')
    print(f"✅ Loaded demo data: {len(demo_data)} records")
    
    # Create environment
    env = ParlayEnvironment(demo_data)
    print(f"✅ Environment created: {env.state_dim} state dim")
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    agent = PPOAgent(state_dim, action_dims)
    print(f"✅ Agent created: {agent.device}")
    
    # Quick training loop
    print("\n🏋️ Training agent...")
    for episode in range(50):  # Quick training
        state = env.reset()
        episode_reward = 0
        
        for step in range(4):  # Max 4 legs
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update every 10 episodes
        if episode % 10 == 0:
            loss = agent.update()
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Loss={loss_str}")
    
    # Final update
    agent.update()
    
    # Save model
    model_path = "rl_models/quick_trained_model.pth"
    os.makedirs("rl_models", exist_ok=True)
    agent.save_model(model_path)
    print(f"💾 Model saved: {model_path}")
    
    # Test the trained model
    print("\n🎯 Testing trained model...")
    for i in range(3):
        parlay = agent.generate_parlay(env)
        print(f"\n--- Test Parlay {i+1} ---")
        print(f"Legs: {parlay['num_legs']}")
        print(f"Hit Rate: {parlay['combined_hit_rate']:.2%}")
        print(f"Odds: +{parlay['estimated_odds']:.0f}")
        print(f"Expected Value: ${parlay['expected_value']:.2f}")
        
        if parlay['legs']:
            print("Legs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"  {j}. {leg['player']} ({leg['team']}) - {leg['prop']} O{leg['line']:.1f} ({leg['hit_rate']:.1%})")
        else:
            print("  No legs generated")
    
    print("\n🎉 Quick training completed!")
    return agent

if __name__ == "__main__":
    quick_train()
