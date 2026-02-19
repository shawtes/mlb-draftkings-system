#!/usr/bin/env python3
"""
Train and Test RL Parlay System
Uses existing data to train and test the system
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

def create_enhanced_training_data():
    """Create enhanced training data with more realistic patterns"""
    print("📊 Creating Enhanced Training Data")
    print("=" * 40)
    
    # Load existing demo data
    if not os.path.exists('rl_demo_data.csv'):
        print("❌ Demo data not found. Run main.py demo first.")
        return None
    
    data = pd.read_csv('rl_demo_data.csv')
    print(f"✅ Loaded demo data: {len(data)} records")
    
    # Enhance the data with more realistic patterns
    enhanced_data = data.copy()
    
    # Add more realistic hit rates based on position and prop type
    def calculate_realistic_hit_rate(row):
        position = row['position_proj']
        dk_proj = row['projected_dk_points']
        
        # Base hit rate by position
        if position == 'QB':
            base_rate = 0.65
        elif position == 'RB':
            base_rate = 0.70
        elif position in ['WR', 'TE']:
            base_rate = 0.68
        else:  # K, DST
            base_rate = 0.60
        
        # Adjust based on projection level
        if dk_proj > 20:
            base_rate += 0.1
        elif dk_proj > 15:
            base_rate += 0.05
        elif dk_proj < 5:
            base_rate -= 0.1
        
        return max(0.3, min(0.9, base_rate))
    
    # Calculate realistic hit rates
    enhanced_data['realistic_hit_rate'] = enhanced_data.apply(calculate_realistic_hit_rate, axis=1)
    
    # Update hit rate columns with more realistic values
    enhanced_data['dk_points_hit_mean'] = enhanced_data['realistic_hit_rate']
    enhanced_data['passing_yds_hit_mean'] = enhanced_data['realistic_hit_rate'] * 0.95
    enhanced_data['rushing_yds_hit_mean'] = enhanced_data['realistic_hit_rate'] * 0.90
    enhanced_data['receiving_yds_hit_mean'] = enhanced_data['realistic_hit_rate'] * 0.92
    enhanced_data['receptions_hit_mean'] = enhanced_data['realistic_hit_rate'] * 0.88
    
    # Add consistency metrics
    enhanced_data['dk_points_accuracy_mean'] = 1.0 + np.random.normal(0, 0.2, len(enhanced_data))
    enhanced_data['dk_points_accuracy_std'] = np.random.uniform(0.1, 0.4, len(enhanced_data))
    
    # Save enhanced data
    enhanced_data.to_csv('enhanced_training_data.csv', index=False)
    print(f"✅ Enhanced training data saved: {len(enhanced_data)} records")
    
    return enhanced_data

def train_agent_on_enhanced_data(data, num_episodes=1000):
    """Train RL agent on enhanced data"""
    print(f"\n🤖 Training RL Agent ({num_episodes} episodes)")
    print("=" * 40)
    
    # Create environment
    env = ParlayEnvironment(data)
    print(f"✅ Environment created: {env.state_dim} state dim")
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    agent = PPOAgent(state_dim, action_dims)
    print(f"✅ Agent created: {agent.device}")
    
    # Training loop with better reward structure
    print("🏋️ Training agent...")
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
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
        
        # Update every 50 episodes
        if episode % 50 == 0:
            loss = agent.update()
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Loss={loss_str}")
            
            # Evaluate every 200 episodes
            if episode % 200 == 0:
                eval_metrics = agent.evaluate(env, num_episodes=5)
                print(f"  Eval: Reward={eval_metrics['avg_reward']:.2f}, "
                      f"Hit Rate={eval_metrics['avg_hit_rate']:.2%}, "
                      f"Expected Value=${eval_metrics['avg_expected_value']:.2f}")
                
                # Save best model
                if eval_metrics['avg_reward'] > best_reward:
                    best_reward = eval_metrics['avg_reward']
                    agent.save_model("rl_models/enhanced_trained_model.pth")
                    print(f"  💾 New best model saved! Reward: {best_reward:.2f}")
    
    # Final update
    agent.update()
    print(f"✅ Training completed! Best reward: {best_reward:.2f}")
    
    return agent

def test_agent_performance(agent, data, num_tests=50):
    """Test agent performance"""
    print(f"\n🧪 Testing Agent Performance ({num_tests} tests)")
    print("=" * 40)
    
    env = ParlayEnvironment(data)
    
    # Generate test parlays
    parlays = []
    for i in range(num_tests):
        parlay = agent.generate_parlay(env)
        if parlay['legs']:  # Only count parlays with legs
            parlays.append(parlay)
    
    if not parlays:
        print("❌ No parlays generated!")
        return None
    
    # Calculate statistics
    stats = {
        'total_parlays': len(parlays),
        'avg_legs': np.mean([len(p['legs']) for p in parlays]),
        'avg_hit_rate': np.mean([p['combined_hit_rate'] for p in parlays]),
        'avg_odds': np.mean([p['estimated_odds'] for p in parlays]),
        'avg_expected_value': np.mean([p['expected_value'] for p in parlays]),
        'max_hit_rate': max([p['combined_hit_rate'] for p in parlays]),
        'min_hit_rate': min([p['combined_hit_rate'] for p in parlays]),
        'profitable_parlays': len([p for p in parlays if p['expected_value'] > 0]),
        'high_hit_rate_parlays': len([p for p in parlays if p['combined_hit_rate'] > 0.6])
    }
    
    # Display results
    print(f"📊 Test Results:")
    print(f"   Total Parlays: {stats['total_parlays']}")
    print(f"   Average Legs: {stats['avg_legs']:.1f}")
    print(f"   Average Hit Rate: {stats['avg_hit_rate']:.2%}")
    print(f"   Average Odds: +{stats['avg_odds']:.0f}")
    print(f"   Average Expected Value: ${stats['avg_expected_value']:.2f}")
    print(f"   Hit Rate Range: {stats['min_hit_rate']:.1%} - {stats['max_hit_rate']:.1%}")
    print(f"   Profitable Parlays: {stats['profitable_parlays']} ({stats['profitable_parlays']/stats['total_parlays']*100:.1f}%)")
    print(f"   High Hit Rate (>60%): {stats['high_hit_rate_parlays']} ({stats['high_hit_rate_parlays']/stats['total_parlays']*100:.1f}%)")
    
    # Show best parlays
    best_parlays = sorted(parlays, key=lambda x: x['expected_value'], reverse=True)[:5]
    print(f"\n🏆 Top 5 Parlays by Expected Value:")
    for i, parlay in enumerate(best_parlays, 1):
        print(f"  {i}. {len(parlay['legs'])} legs, {parlay['combined_hit_rate']:.1%} hit rate, +{parlay['estimated_odds']:.0f} odds, ${parlay['expected_value']:.2f} EV")
    
    # Save results
    results_file = f"rl_results/agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("rl_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Results saved: {results_file}")
    
    return stats

def compare_with_simple_generator(data, num_tests=50):
    """Compare RL agent with simple generator"""
    print(f"\n⚖️ Comparing RL Agent vs Simple Generator ({num_tests} tests each)")
    print("=" * 60)
    
    # Load trained agent
    if not os.path.exists("rl_models/enhanced_trained_model.pth"):
        print("❌ No trained model found. Train agent first.")
        return
    
    # Create environments
    env = ParlayEnvironment(data)
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    agent = PPOAgent(state_dim, action_dims)
    agent.load_model("rl_models/enhanced_trained_model.pth")
    
    simple_gen = SimpleParlayGenerator(data)
    
    # Generate parlays with both methods
    print("🎲 Generating parlays...")
    
    # RL Agent parlays
    rl_parlays = []
    for i in range(num_tests):
        parlay = agent.generate_parlay(env)
        if parlay['legs']:
            rl_parlays.append(parlay)
    
    # Simple generator parlays
    simple_parlays = simple_gen.generate_multiple_parlays(num_tests, max_legs=4)
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    
    if rl_parlays:
        rl_stats = {
            'avg_legs': np.mean([len(p['legs']) for p in rl_parlays]),
            'avg_hit_rate': np.mean([p['combined_hit_rate'] for p in rl_parlays]),
            'avg_odds': np.mean([p['estimated_odds'] for p in rl_parlays]),
            'avg_expected_value': np.mean([p['expected_value'] for p in rl_parlays]),
            'profitable': len([p for p in rl_parlays if p['expected_value'] > 0])
        }
        
        print(f"\n🤖 RL Agent ({len(rl_parlays)} parlays):")
        print(f"   Average Legs: {rl_stats['avg_legs']:.1f}")
        print(f"   Average Hit Rate: {rl_stats['avg_hit_rate']:.2%}")
        print(f"   Average Odds: +{rl_stats['avg_odds']:.0f}")
        print(f"   Average Expected Value: ${rl_stats['avg_expected_value']:.2f}")
        print(f"   Profitable Parlays: {rl_stats['profitable']} ({rl_stats['profitable']/len(rl_parlays)*100:.1f}%)")
    else:
        print(f"\n🤖 RL Agent: No parlays generated")
        rl_stats = None
    
    if simple_parlays:
        simple_stats = {
            'avg_legs': np.mean([len(p.legs) for p in simple_parlays]),
            'avg_hit_rate': np.mean([p.combined_hit_rate for p in simple_parlays]),
            'avg_odds': np.mean([p.estimated_odds for p in simple_parlays]),
            'avg_expected_value': np.mean([p.expected_value for p in simple_parlays]),
            'profitable': len([p for p in simple_parlays if p.expected_value > 0])
        }
        
        print(f"\n📋 Simple Generator ({len(simple_parlays)} parlays):")
        print(f"   Average Legs: {simple_stats['avg_legs']:.1f}")
        print(f"   Average Hit Rate: {simple_stats['avg_hit_rate']:.2%}")
        print(f"   Average Odds: +{simple_stats['avg_odds']:.0f}")
        print(f"   Average Expected Value: ${simple_stats['avg_expected_value']:.2f}")
        print(f"   Profitable Parlays: {simple_stats['profitable']} ({simple_stats['profitable']/len(simple_parlays)*100:.1f}%)")
    else:
        print(f"\n📋 Simple Generator: No parlays generated")
        simple_stats = None
    
    # Determine winner
    if rl_stats and simple_stats:
        if rl_stats['avg_expected_value'] > simple_stats['avg_expected_value']:
            print(f"\n🏆 RL Agent wins! (${rl_stats['avg_expected_value']:.2f} vs ${simple_stats['avg_expected_value']:.2f})")
        elif simple_stats['avg_expected_value'] > rl_stats['avg_expected_value']:
            print(f"\n🏆 Simple Generator wins! (${simple_stats['avg_expected_value']:.2f} vs ${rl_stats['avg_expected_value']:.2f})")
        else:
            print(f"\n🤝 It's a tie! (${rl_stats['avg_expected_value']:.2f} vs ${simple_stats['avg_expected_value']:.2f})")

def main():
    """Main function"""
    print("🚀 RL Parlay Training & Testing Pipeline")
    print("=" * 50)
    
    # Step 1: Create enhanced training data
    data = create_enhanced_training_data()
    if data is None:
        return
    
    # Step 2: Train agent
    agent = train_agent_on_enhanced_data(data, num_episodes=1000)
    
    # Step 3: Test agent performance
    test_agent_performance(agent, data, num_tests=50)
    
    # Step 4: Compare with simple generator
    compare_with_simple_generator(data, num_tests=50)
    
    print("\n🎉 Training and testing completed successfully!")

if __name__ == "__main__":
    main()
