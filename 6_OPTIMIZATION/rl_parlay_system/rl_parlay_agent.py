#!/usr/bin/env python3
"""
RL Parlay Agent
PPO-based agent for learning optimal parlay generation strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from collections import deque
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import os

class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, state_dim: int, action_dims: List[int], hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate heads for each action dimension
        self.player_head = nn.Linear(hidden_dim, action_dims[0])
        self.prop_type_head = nn.Linear(hidden_dim, action_dims[1])
        self.line_multiplier_head = nn.Linear(hidden_dim, action_dims[2])
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network"""
        features = self.feature_extractor(state)
        
        # Get action logits
        player_logits = self.player_head(features)
        prop_type_logits = self.prop_type_head(features)
        line_multiplier_logits = self.line_multiplier_head(features)
        
        # Get value
        value = self.value_head(features)
        
        return player_logits, prop_type_logits, line_multiplier_logits, value

class PPOAgent:
    """PPO Agent for parlay generation"""
    
    def __init__(self, state_dim: int, action_dims: List[int], 
                 lr: float = 3e-4, gamma: float = 0.99, 
                 eps_clip: float = 0.2, k_epochs: int = 4):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 Using device: {self.device}")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Experience buffer
        self.memory = deque(maxlen=10000)
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        self.episode_rewards = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            player_logits, prop_type_logits, line_multiplier_logits, value = self.policy(state_tensor)
            
            # Create distributions
            player_dist = Categorical(logits=player_logits)
            prop_type_dist = Categorical(logits=prop_type_logits)
            line_multiplier_dist = Categorical(logits=line_multiplier_logits)
            
            # Sample actions
            player_action = player_dist.sample()
            prop_type_action = prop_type_dist.sample()
            line_multiplier_action = line_multiplier_dist.sample()
            
            # Calculate log probabilities
            player_log_prob = player_dist.log_prob(player_action)
            prop_type_log_prob = prop_type_dist.log_prob(prop_type_action)
            line_multiplier_log_prob = line_multiplier_dist.log_prob(line_multiplier_action)
            
            total_log_prob = player_log_prob + prop_type_log_prob + line_multiplier_log_prob
            
            action = np.array([player_action.item(), prop_type_action.item(), line_multiplier_action.item()])
            
            return action, total_log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, 
                        done: bool, log_prob: float, value: float):
        """Store transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def update(self, batch_size: int = 64):
        """Update policy using PPO"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        
        # Convert to tensors
        states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t['done'] for t in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in batch]).to(self.device)
        old_values = torch.FloatTensor([t['value'] for t in batch]).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, dones)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(self.k_epochs):
            # Get current policy outputs
            player_logits, prop_type_logits, line_multiplier_logits, values = self.policy(states)
            
            # Create distributions
            player_dist = Categorical(logits=player_logits)
            prop_type_dist = Categorical(logits=prop_type_logits)
            line_multiplier_dist = Categorical(logits=line_multiplier_logits)
            
            # Calculate log probabilities
            player_log_probs = player_dist.log_prob(actions[:, 0])
            prop_type_log_probs = prop_type_dist.log_prob(actions[:, 1])
            line_multiplier_log_probs = line_multiplier_dist.log_prob(actions[:, 2])
            
            new_log_probs = player_log_probs + prop_type_log_probs + line_multiplier_log_probs
            
            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * value_loss
            total_loss += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Store metrics
        avg_loss = total_loss / self.k_epochs
        self.training_losses.append(avg_loss)
        
        return avg_loss
    
    def _calculate_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def train_episode(self, env, max_steps: int = 100) -> float:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action, log_prob, value = self.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            self.store_transition(state, action, reward, next_state, done, log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update policy
        loss = self.update()
        
        self.episode_rewards.append(episode_reward)
        return episode_reward
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict:
        """Evaluate current policy"""
        total_rewards = []
        parlay_metrics = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            while True:
                action, _, _ = self.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            
            # Get final parlay metrics
            parlay = env.get_current_parlay()
            parlay_metrics.append({
                'num_legs': len(parlay.legs),
                'hit_rate': parlay.combined_hit_rate,
                'odds': parlay.estimated_odds,
                'expected_value': parlay.expected_value
            })
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_legs': np.mean([m['num_legs'] for m in parlay_metrics]),
            'avg_hit_rate': np.mean([m['hit_rate'] for m in parlay_metrics]),
            'avg_odds': np.mean([m['odds'] for m in parlay_metrics]),
            'avg_expected_value': np.mean([m['expected_value'] for m in parlay_metrics])
        }
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"📂 Model loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot episode rewards
        if self.episode_rewards:
            ax1.plot(self.episode_rewards)
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
        
        # Plot training losses
        if self.training_losses:
            ax2.plot(self.training_losses)
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Training progress saved to {save_path}")
        else:
            plt.show()
    
    def generate_parlay(self, env) -> Dict:
        """Generate a parlay using current policy"""
        state = env.reset()
        
        while True:
            action, _, _ = self.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            
            if done:
                break
        
        parlay = env.get_current_parlay()
        
        return {
            'legs': [
                {
                    'player': leg.player_name,
                    'team': leg.team,
                    'position': leg.position,
                    'prop': leg.prop_type,
                    'line': leg.line,
                    'projection': leg.projection,
                    'hit_rate': leg.hit_rate,
                    'confidence': leg.confidence
                }
                for leg in parlay.legs
            ],
            'combined_hit_rate': parlay.combined_hit_rate,
            'estimated_odds': parlay.estimated_odds,
            'expected_value': parlay.expected_value,
            'num_legs': len(parlay.legs)
        }

def train_rl_agent(training_data: pd.DataFrame, 
                  num_episodes: int = 1000,
                  save_path: str = "rl_parlay_model.pth") -> PPOAgent:
    """Train the RL agent"""
    print("🚀 Starting RL Agent Training")
    print("=" * 50)
    
    # Create environment
    env = ParlayEnvironment(training_data)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec.tolist()
    agent = PPOAgent(state_dim, action_dims)
    
    # Training loop
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        episode_reward = agent.train_episode(env)
        
        if episode % 100 == 0:
            # Evaluate current policy
            eval_metrics = agent.evaluate(env, num_episodes=5)
            print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                  f"Eval Reward={eval_metrics['avg_reward']:.2f}, "
                  f"Hit Rate={eval_metrics['avg_hit_rate']:.2%}, "
                  f"Expected Value=${eval_metrics['avg_expected_value']:.2f}")
            
            # Save best model
            if eval_metrics['avg_reward'] > best_reward:
                best_reward = eval_metrics['avg_reward']
                agent.save_model(save_path)
    
    print(f"\n✅ Training complete! Best reward: {best_reward:.2f}")
    return agent

if __name__ == "__main__":
    # This would be called after data collection
    print("🤖 RL Parlay Agent - Ready for training!")
    print("   Run data collection first, then train the agent.")
