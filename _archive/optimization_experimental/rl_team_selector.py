import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import gymnasium as gym
from gymnasium import spaces
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for team selection"""
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_actions: int = 100):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, num_actions)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MLBTeamSelectionEnvironment(gym.Env):
    """
    Custom environment for MLB team selection
    - State: Player features, historical performance, current lineup
    - Action: Select/deselect players for lineup
    - Reward: Based on actual fantasy points scored
    """
    
    def __init__(self, data_df: pd.DataFrame, salary_cap: float = 50000, 
                 lineup_size: int = 8, position_constraints: Dict = None):
        super().__init__()
        
        self.data_df = data_df.copy()
        self.salary_cap = salary_cap
        self.lineup_size = lineup_size
        self.position_constraints = position_constraints or {
            'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
        }
        
        # Preprocess data
        self._preprocess_data()
        
        # Get unique dates for episodes
        self.dates = sorted(self.data_df['date'].unique())
        self.current_date_idx = 0
        
        # Action space: binary selection for each available player
        self.max_players = min(100, len(self.data_df['Name'].unique()))  # Cap at 100 for memory efficiency
        self.action_space = spaces.Discrete(self.max_players)
        
        # State space: simplified state representation
        self.feature_cols = self._get_feature_columns()
        # Fixed state size: player features + lineup state + metadata
        self.state_size = len(self.feature_cols) * 10 + self.lineup_size + 5  # Simplified size
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(self.state_size,), dtype=np.float32)
        
        # Initialize episode
        self.reset()
        
    def _preprocess_data(self):
        """Preprocess the data for RL training"""
        # Ensure required columns exist
        if 'salary' not in self.data_df.columns:
            self.data_df['salary'] = np.random.randint(3000, 12000, len(self.data_df))
        
        if 'position' not in self.data_df.columns:
            positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
            self.data_df['position'] = np.random.choice(positions, len(self.data_df))
        
        # Fill missing values
        self.data_df = self.data_df.fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        numeric_cols = self.data_df.select_dtypes(include=[np.number]).columns
        self.data_df[numeric_cols] = self.scaler.fit_transform(self.data_df[numeric_cols])
        
    def _get_feature_columns(self):
        """Get relevant feature columns for state representation"""
        feature_cols = [
            'calculated_dk_fpts', 'salary', 'AVG', 'OBP', 'SLG', 'HR', 'RBI', 'R', 'SB',
            'wOBA', 'wRC+', 'BABIP', 'ISO', 'BB', 'SO', 'PA', 'AB'
        ]
        return [col for col in feature_cols if col in self.data_df.columns]
    
    def reset(self, seed=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Select a random date for this episode
        self.current_date_idx = np.random.randint(0, len(self.dates))
        self.current_date = self.dates[self.current_date_idx]
        
        # Get players available for this date
        self.current_players = self.data_df[self.data_df['date'] == self.current_date].copy()
        
        if len(self.current_players) == 0:
            # If no players for this date, use a nearby date
            self.current_date_idx = min(self.current_date_idx + 1, len(self.dates) - 1)
            self.current_date = self.dates[self.current_date_idx]
            self.current_players = self.data_df[self.data_df['date'] == self.current_date].copy()
        
        # Initialize lineup
        self.current_lineup = []
        self.current_salary_used = 0
        self.steps_taken = 0
        self.max_steps = min(self.lineup_size * 2, len(self.current_players))
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state representation - simplified and robust"""
        # Get top 10 players by fantasy points for current date
        top_players = self.current_players.nlargest(10, 'calculated_dk_fpts')
        
        # Create fixed-size player features (10 players x features)
        player_features = np.zeros((10, len(self.feature_cols)))
        
        for i, (_, player) in enumerate(top_players.iterrows()):
            if i >= 10:
                break
            for j, col in enumerate(self.feature_cols):
                player_features[i, j] = player[col]
        
        # Flatten player features
        flattened_features = player_features.flatten()
        
        # Add current lineup state
        lineup_state = np.zeros(self.lineup_size)
        for i, player_idx in enumerate(self.current_lineup):
            if i < self.lineup_size:
                lineup_state[i] = 1.0
        
        # Add metadata
        metadata = [
            self.current_salary_used / self.salary_cap if self.salary_cap > 0 else 0,
            len(self.current_lineup) / self.lineup_size,
            self.steps_taken / self.max_steps,
            len(self.current_players) / 100,  # Normalize available players
            np.mean([p['calculated_dk_fpts'] for _, p in top_players.iterrows()]) / 50  # Normalize avg points
        ]
        
        # Combine all components
        full_state = np.concatenate([flattened_features, lineup_state, metadata])
        
        # Ensure exact state size
        if len(full_state) != self.state_size:
            # Pad or truncate to exact size
            if len(full_state) < self.state_size:
                full_state = np.pad(full_state, (0, self.state_size - len(full_state)))
            else:
                full_state = full_state[:self.state_size]
        
        return full_state.astype(np.float32)
    
    def step(self, action):
        """Take a step in the environment"""
        reward = 0
        done = False
        info = {}
        
        # Ensure action is within bounds of current available players
        if action >= len(self.current_players):
            # Invalid action - select random valid action
            action = np.random.randint(0, len(self.current_players))
        
        player_row = self.current_players.iloc[action]
        player_salary = player_row.get('salary', 5000)  # Default salary if missing
        player_fpts = player_row.get('calculated_dk_fpts', 0)
        
        # Check if player can be added to lineup
        if (len(self.current_lineup) < self.lineup_size and 
            self.current_salary_used + player_salary <= self.salary_cap and
            action not in self.current_lineup):
            
            # Add player to lineup
            self.current_lineup.append(action)
            self.current_salary_used += player_salary
            
            # Reward based on expected fantasy points (ensure positive)
            base_reward = max(0, player_fpts) * 0.1  # Scale reward and ensure non-negative
            
            # Bonus for efficient salary usage
            if player_salary > 0:
                salary_efficiency = max(0, player_fpts) / (player_salary / 1000)
                reward = base_reward + (salary_efficiency * 0.05)
            else:
                reward = base_reward
            
            # Additional bonus for players with good recent performance
            if player_fpts > 10:  # Good performance threshold
                reward += 2.0
            elif player_fpts > 5:  # Decent performance
                reward += 1.0
            
        else:
            # Penalty for invalid action
            reward = -0.5  # Reduced penalty
        
        self.steps_taken += 1
        
        # Check if episode is done
        if (len(self.current_lineup) >= self.lineup_size or 
            self.steps_taken >= self.max_steps):
            done = True
            
            # Final reward based on total lineup performance
            if len(self.current_lineup) > 0:
                total_fpts = sum(max(0, self.current_players.iloc[i]['calculated_dk_fpts']) 
                               for i in self.current_lineup)
                
                # Bonus for good lineup (scaled appropriately)
                lineup_bonus = total_fpts * 0.2
                reward += lineup_bonus
                
                # Bonus for lineup completion
                if len(self.current_lineup) == self.lineup_size:
                    reward += 10.0  # Completion bonus
                
                # Penalty for incomplete lineup
                if len(self.current_lineup) < self.lineup_size:
                    missing_players = self.lineup_size - len(self.current_lineup)
                    reward -= missing_players * 2.0  # Reduced penalty
            else:
                reward -= 5.0  # Penalty for no players selected
        
        next_state = self._get_state()
        
        return next_state, reward, done, False, info
    
    def get_lineup_performance(self):
        """Get actual performance of current lineup"""
        if not self.current_lineup:
            return 0
        
        total_fpts = sum(self.current_players.iloc[i]['calculated_dk_fpts'] 
                        for i in self.current_lineup)
        return total_fpts

class DQNAgent:
    """DQN Agent for team selection"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Neural networks - force CPU usage for stability
        self.device = torch.device("cpu")  # Force CPU to avoid CUDA issues
        self.q_network = DQN(state_size, 512, action_size).to(self.device)
        self.target_network = DQN(state_size, 512, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(memory_size)
        self.update_target_freq = 100
        self.learn_step = 0
        
    def act(self, state, valid_actions=None):
        """Select action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return np.random.choice(valid_actions)
            return np.random.randint(0, min(self.action_size, 100))  # Cap random actions
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        if valid_actions is not None and len(valid_actions) > 0:
            # Mask invalid actions
            masked_q_values = q_values.clone()
            valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.long)
            # Only consider valid actions
            if len(valid_actions_tensor) > 0:
                best_valid_idx = torch.argmax(masked_q_values[0, valid_actions_tensor])
                return valid_actions_tensor[best_valid_idx].item()
        
        return min(q_values.argmax().item(), 99)  # Cap at 99 for safety
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.learn_step += 1
        if self.learn_step % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step': self.learn_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step = checkpoint['learn_step']

class MLBRLTeamSelector:
    """Main class for MLB team selection using reinforcement learning"""
    
    def __init__(self, data_path: str, model_save_path: str = None):
        self.data_path = data_path
        self.model_save_path = model_save_path or "mlb_rl_model.pth"
        self.data_df = None
        self.env = None
        self.agent = None
        self.training_history = []
        
    def load_data(self):
        """Load and preprocess the MLB data"""
        logger.info("Loading MLB data...")
        
        # Load data
        self.data_df = pd.read_csv(self.data_path, low_memory=False)
        
        # Convert date column
        self.data_df['date'] = pd.to_datetime(self.data_df['date'], errors='coerce')
        self.data_df = self.data_df.dropna(subset=['date'])
        
        # Calculate fantasy points if not present
        if 'calculated_dk_fpts' not in self.data_df.columns:
            self.data_df['calculated_dk_fpts'] = self._calculate_dk_fpts(self.data_df)
        
        # Remove rows with missing critical data
        self.data_df = self.data_df.dropna(subset=['Name', 'calculated_dk_fpts'])
        
        logger.info(f"Data loaded: {len(self.data_df)} rows, {len(self.data_df['Name'].unique())} unique players")
        logger.info(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
        
    def _calculate_dk_fpts(self, df):
        """Calculate DraftKings fantasy points"""
        return (df.get('1B', 0) * 3 + df.get('2B', 0) * 5 + df.get('3B', 0) * 8 + 
                df.get('HR', 0) * 10 + df.get('RBI', 0) * 2 + df.get('R', 0) * 2 + 
                df.get('BB', 0) * 2 + df.get('HBP', 0) * 2 + df.get('SB', 0) * 5)
    
    def setup_environment(self):
        """Setup the RL environment"""
        logger.info("Setting up RL environment...")
        
        self.env = MLBTeamSelectionEnvironment(
            data_df=self.data_df,
            salary_cap=50000,
            lineup_size=8
        )
        
        # Initialize agent
        self.agent = DQNAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_space.n,
            lr=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        logger.info(f"Environment setup complete. State size: {self.env.state_size}, Action size: {self.env.action_space.n}")
    
    def train(self, episodes: int = 1000, save_freq: int = 100):
        """Train the RL agent"""
        logger.info(f"Starting training for {episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Get valid actions (available players not in lineup)
                valid_actions = [i for i in range(len(self.env.current_players)) 
                               if i not in self.env.current_lineup]
                
                if not valid_actions:
                    break
                
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done or truncated:
                    break
            
            # Train the agent
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                           f"Avg Length: {avg_length:.1f}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                self.agent.save(f"{self.model_save_path}_{episode}.pth")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        # Save final model
        self.agent.save(self.model_save_path)
        logger.info("Training completed!")
    
    def evaluate(self, num_episodes: int = 100):
        """Evaluate the trained agent"""
        logger.info("Evaluating trained agent...")
        
        # Load best model
        if os.path.exists(self.model_save_path):
            self.agent.load(self.model_save_path)
        
        # Set epsilon to 0 for evaluation (no exploration)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        evaluation_results = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            while True:
                valid_actions = [i for i in range(len(self.env.current_players)) 
                               if i not in self.env.current_lineup]
                
                if not valid_actions:
                    break
                
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done or truncated:
                    break
            
            # Get lineup performance
            lineup_performance = self.env.get_lineup_performance()
            evaluation_results.append({
                'episode': episode,
                'total_reward': total_reward,
                'lineup_performance': lineup_performance,
                'lineup_size': len(self.env.current_lineup),
                'salary_used': self.env.current_salary_used
            })
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        # Calculate evaluation metrics
        avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
        avg_performance = np.mean([r['lineup_performance'] for r in evaluation_results])
        avg_lineup_size = np.mean([r['lineup_size'] for r in evaluation_results])
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Average Reward: {avg_reward:.2f}")
        logger.info(f"Average Lineup Performance: {avg_performance:.2f}")
        logger.info(f"Average Lineup Size: {avg_lineup_size:.1f}")
        
        return evaluation_results
    
    def predict_optimal_lineup(self, date: str = None):
        """Predict optimal lineup for a given date"""
        if date is None:
            date = self.data_df['date'].max()
        else:
            date = pd.to_datetime(date)
        
        logger.info(f"Predicting optimal lineup for {date}")
        
        # Get players for the date
        available_players = self.data_df[self.data_df['date'] == date].copy()
        
        if len(available_players) == 0:
            logger.warning(f"No players found for date {date}")
            return None
        
        # Create temporary environment for prediction
        temp_env = MLBTeamSelectionEnvironment(
            data_df=available_players,
            salary_cap=50000,
            lineup_size=8
        )
        
        # Load trained model
        if os.path.exists(self.model_save_path):
            self.agent.load(self.model_save_path)
        
        # Set epsilon to 0 for prediction
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        # Run prediction
        state, _ = temp_env.reset()
        selected_players = []
        
        while len(selected_players) < 8:
            valid_actions = [i for i in range(len(temp_env.current_players)) 
                           if i not in temp_env.current_lineup]
            
            if not valid_actions:
                break
            
            action = self.agent.act(state, valid_actions)
            next_state, reward, done, truncated, info = temp_env.step(action)
            
            if action in temp_env.current_lineup:
                player_data = temp_env.current_players.iloc[action]
                selected_players.append({
                    'Name': player_data['Name'],
                    'Team': player_data.get('Team', 'N/A'),
                    'Position': player_data.get('position', 'N/A'),
                    'Salary': player_data.get('salary', 0),
                    'Predicted_Points': player_data['calculated_dk_fpts']
                })
            
            state = next_state
            
            if done or truncated:
                break
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        # Create lineup summary
        lineup_df = pd.DataFrame(selected_players)
        total_salary = lineup_df['Salary'].sum()
        total_points = lineup_df['Predicted_Points'].sum()
        
        logger.info(f"Optimal Lineup for {date}:")
        logger.info(f"Total Salary: ${total_salary:,.0f}")
        logger.info(f"Total Predicted Points: {total_points:.1f}")
        logger.info("\nLineup:")
        for _, player in lineup_df.iterrows():
            logger.info(f"{player['Name']} ({player['Position']}) - "
                       f"${player['Salary']:,.0f} - {player['Predicted_Points']:.1f} pts")
        
        return lineup_df
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        ax1.plot(self.training_history['episode_rewards'])
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(self.training_history['episode_lengths'])
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    """Main function to run the RL team selector"""
    # Configuration
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    MODEL_SAVE_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/mlb_rl_model.pth'
    
    # Initialize the RL team selector
    selector = MLBRLTeamSelector(DATA_PATH, MODEL_SAVE_PATH)
    
    # Load data
    selector.load_data()
    
    # Setup environment
    selector.setup_environment()
    
    # Train the model
    selector.train(episodes=2000, save_freq=200)
    
    # Evaluate the model
    evaluation_results = selector.evaluate(num_episodes=100)
    
    # Plot training history
    selector.plot_training_history()
    
    # Predict optimal lineup for latest date
    optimal_lineup = selector.predict_optimal_lineup()
    
    # Save optimal lineup
    if optimal_lineup is not None:
        output_path = '5_ENTRIES/rl_optimal_lineup.csv'
        optimal_lineup.to_csv(output_path, index=False)
        logger.info(f"Optimal lineup saved to {output_path}")

if __name__ == "__main__":
    main()
