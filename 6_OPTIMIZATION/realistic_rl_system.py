#!/usr/bin/env python3
"""
Realistic RL System for MLB DraftKings Team Selection

This system follows the real game-day workflow:
1. Get player projections using your existing prediction model
2. Use RL to learn optimal lineup selection from those projections
3. Follow exact DraftKings rules (same as optimizer)
4. Validate performance using walk-forward methodology
"""

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

# Import PuLP for optimization
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not available. Using greedy optimization.")

# Import your existing prediction system
try:
    from predction01 import predict_unseen_data, create_synthetic_rows_for_all_players
except ImportError:
    print("Warning: Could not import prediction system. Using mock predictions.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DraftKings Rules (same as your optimizer)
DK_RULES = {
    'SALARY_CAP': 50000,
    'LINEUP_SIZE': 8,
    'POSITIONS': {
        'C': 1,      # Catcher
        '1B': 1,     # First Base  
        '2B': 1,     # Second Base
        '3B': 1,     # Third Base
        'SS': 1,     # Shortstop
        'OF': 3,     # Outfielder
        'UTIL': 1    # Utility (any position)
    },
    'SCORING': {
        '1B': 3,     # Single
        '2B': 5,     # Double
        '3B': 8,     # Triple
        'HR': 10,    # Home Run
        'RBI': 2,    # Run Batted In
        'R': 2,      # Run
        'BB': 2,     # Walk
        'HBP': 2,    # Hit by Pitch
        'SB': 5      # Stolen Base
    }
}

class PredictionBasedRLEnvironment(gym.Env):
    """
    RL Environment that uses actual predictions (like game day)
    
    This environment:
    1. Gets player projections from your prediction model
    2. Creates lineup selection scenarios
    3. Rewards based on actual performance vs projections
    """
    
    def __init__(self, prediction_model_path: str, data_path: str):
        super().__init__()
        
        self.prediction_model_path = prediction_model_path
        self.data_path = data_path
        self.current_predictions = None
        self.current_actual_performance = None
        self.current_lineup = []
        self.current_salary_used = 0
        self.episode_date = None
        
        # Load historical data for validation
        self.load_historical_data()
        
        # Get available dates for episodes
        self.available_dates = sorted(self.historical_data['date'].unique())
        
        # Action space: select player index from available predictions
        self.max_players_per_day = 100  # Reasonable limit
        self.action_space = spaces.Discrete(self.max_players_per_day)
        
        # State space: player projections + lineup state + constraints
        self.state_size = self.max_players_per_day * 4 + 10  # projections + lineup info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_size,), 
            dtype=np.float32
        )
        
    def load_historical_data(self):
        """Load historical data for validation"""
        logger.info("Loading historical data...")
        self.historical_data = pd.read_csv(self.data_path, low_memory=False)
        self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
        
        # Calculate actual fantasy points if not present
        if 'calculated_dk_fpts' not in self.historical_data.columns:
            self.historical_data['calculated_dk_fpts'] = self._calculate_dk_fpts(self.historical_data)
        
        # Add synthetic position and salary data
        self._add_synthetic_game_data()
        
        logger.info(f"Historical data loaded: {len(self.historical_data)} rows")
        
    def _calculate_dk_fpts(self, df):
        """Calculate DraftKings fantasy points using official scoring"""
        return (
            df.get('1B', 0) * DK_RULES['SCORING']['1B'] +
            df.get('2B', 0) * DK_RULES['SCORING']['2B'] +
            df.get('3B', 0) * DK_RULES['SCORING']['3B'] +
            df.get('HR', 0) * DK_RULES['SCORING']['HR'] +
            df.get('RBI', 0) * DK_RULES['SCORING']['RBI'] +
            df.get('R', 0) * DK_RULES['SCORING']['R'] +
            df.get('BB', 0) * DK_RULES['SCORING']['BB'] +
            df.get('HBP', 0) * DK_RULES['SCORING']['HBP'] +
            df.get('SB', 0) * DK_RULES['SCORING']['SB']
        )
    
    def _add_synthetic_game_data(self):
        """Add synthetic salary and position data (like DraftKings would have)"""
        # Generate realistic salaries based on performance
        performance_quantiles = self.historical_data['calculated_dk_fpts'].quantile([0.2, 0.4, 0.6, 0.8])
        
        def assign_salary(fpts):
            if fpts <= performance_quantiles[0.2]:
                return np.random.randint(3000, 5000)
            elif fpts <= performance_quantiles[0.4]:
                return np.random.randint(4500, 6500)
            elif fpts <= performance_quantiles[0.6]:
                return np.random.randint(6000, 8500)
            elif fpts <= performance_quantiles[0.8]:
                return np.random.randint(8000, 11000)
            else:
                return np.random.randint(10000, 13000)
        
        if 'salary' not in self.historical_data.columns:
            self.historical_data['salary'] = self.historical_data['calculated_dk_fpts'].apply(assign_salary)
        
        # Assign positions
        if 'position' not in self.historical_data.columns:
            positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
            self.historical_data['position'] = np.random.choice(positions, len(self.historical_data))
    
    def get_predictions_for_date(self, date: datetime) -> pd.DataFrame:
        """Get predictions for a specific date using your prediction model"""
        try:
            # Use your existing prediction system
            predictions = predict_unseen_data(
                self.data_path,
                self.prediction_model_path,
                date.strftime('%Y-%m-%d')
            )
            
            if predictions is not None and len(predictions) > 0:
                # Add synthetic salary and position data for prediction day
                self._add_synthetic_game_data_to_predictions(predictions)
                return predictions
            else:
                return self._create_mock_predictions(date)
                
        except Exception as e:
            logger.warning(f"Could not get predictions for {date}: {e}")
            return self._create_mock_predictions(date)
    
    def _add_synthetic_game_data_to_predictions(self, predictions):
        """Add salary and position data to predictions"""
        # Generate salaries based on predicted performance
        if 'salary' not in predictions.columns:
            predictions['salary'] = predictions['predicted_dk_fpts'].apply(
                lambda x: int(3000 + (x / 50.0) * 8000)  # Scale 0-50 pts to 3000-11000 salary
            )
        
        # Assign positions
        if 'position' not in predictions.columns:
            positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
            predictions['position'] = np.random.choice(positions, len(predictions))
    
    def _create_mock_predictions(self, date: datetime) -> pd.DataFrame:
        """Create mock predictions when real predictions aren't available"""
        # Get players who played around this date
        nearby_data = self.historical_data[
            (self.historical_data['date'] >= date - timedelta(days=7)) &
            (self.historical_data['date'] <= date + timedelta(days=7))
        ]
        
        if len(nearby_data) == 0:
            # Use all available players
            nearby_data = self.historical_data.sample(min(50, len(self.historical_data)))
        
        # Create mock predictions
        mock_predictions = nearby_data.groupby('Name').agg({
            'calculated_dk_fpts': 'mean',
            'salary': 'mean',
            'position': 'first'
        }).reset_index()
        
        mock_predictions['predicted_dk_fpts'] = mock_predictions['calculated_dk_fpts'] + np.random.normal(0, 3, len(mock_predictions))
        mock_predictions['predicted_dk_fpts'] = np.clip(mock_predictions['predicted_dk_fpts'], 0, 50)
        
        return mock_predictions.head(self.max_players_per_day)
    
    def reset(self, seed=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Select random date for episode
        self.episode_date = random.choice(self.available_dates)
        
        # Get predictions for this date
        self.current_predictions = self.get_predictions_for_date(self.episode_date)
        
        # Get actual performance for this date
        self.current_actual_performance = self.historical_data[
            self.historical_data['date'] == self.episode_date
        ].copy()
        
        # Reset lineup
        self.current_lineup = []
        self.current_salary_used = 0
        self.steps_taken = 0
        self.max_steps = DK_RULES['LINEUP_SIZE'] * 2
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros(self.state_size)
        
        # Player predictions (projected points, salary, position encoding)
        for i, (_, player) in enumerate(self.current_predictions.iterrows()):
            if i >= self.max_players_per_day:
                break
                
            base_idx = i * 4
            state[base_idx] = player['predicted_dk_fpts']  # Projected points
            state[base_idx + 1] = player['salary'] / 1000  # Salary (scaled)
            state[base_idx + 2] = 1 if i in self.current_lineup else 0  # Selected
            state[base_idx + 3] = self._encode_position(player['position'])  # Position
        
        # Lineup constraints state
        constraint_start = self.max_players_per_day * 4
        state[constraint_start] = len(self.current_lineup) / DK_RULES['LINEUP_SIZE']  # Lineup fill
        state[constraint_start + 1] = self.current_salary_used / DK_RULES['SALARY_CAP']  # Salary usage
        state[constraint_start + 2] = (DK_RULES['SALARY_CAP'] - self.current_salary_used) / DK_RULES['SALARY_CAP']  # Remaining salary
        
        # Position constraints
        position_counts = self._get_position_counts()
        state[constraint_start + 3] = position_counts.get('C', 0)
        state[constraint_start + 4] = position_counts.get('1B', 0)
        state[constraint_start + 5] = position_counts.get('2B', 0)
        state[constraint_start + 6] = position_counts.get('3B', 0)
        state[constraint_start + 7] = position_counts.get('SS', 0)
        state[constraint_start + 8] = position_counts.get('OF', 0)
        state[constraint_start + 9] = position_counts.get('UTIL', 0)
        
        return state.astype(np.float32)
    
    def _encode_position(self, position):
        """Encode position as number"""
        position_map = {'C': 1, '1B': 2, '2B': 3, '3B': 4, 'SS': 5, 'OF': 6, 'UTIL': 7}
        return position_map.get(position, 0)
    
    def _get_position_counts(self):
        """Get current position counts in lineup"""
        counts = {}
        for player_idx in self.current_lineup:
            if player_idx < len(self.current_predictions):
                pos = self.current_predictions.iloc[player_idx]['position']
                counts[pos] = counts.get(pos, 0) + 1
        return counts
    
    def _is_valid_action(self, action):
        """Check if action is valid given DraftKings constraints"""
        if action >= len(self.current_predictions) or action in self.current_lineup:
            return False
        
        player = self.current_predictions.iloc[action]
        
        # Check salary constraint
        if self.current_salary_used + player['salary'] > DK_RULES['SALARY_CAP']:
            return False
        
        # Check lineup size
        if len(self.current_lineup) >= DK_RULES['LINEUP_SIZE']:
            return False
        
        # Check position constraints
        position_counts = self._get_position_counts()
        player_pos = player['position']
        
        if player_pos in DK_RULES['POSITIONS']:
            if position_counts.get(player_pos, 0) >= DK_RULES['POSITIONS'][player_pos]:
                return False
        
        return True
    
    def step(self, action):
        """Take a step in the environment"""
        reward = 0
        done = False
        info = {}
        
        # Ensure action is within bounds
        action = min(action, len(self.current_predictions) - 1)
        
        if self._is_valid_action(action):
            # Valid action - add player to lineup
            player = self.current_predictions.iloc[action]
            self.current_lineup.append(action)
            self.current_salary_used += player['salary']
            
            # Reward based on projected performance
            projected_points = player['predicted_dk_fpts']
            reward += projected_points * 0.1  # Scale reward
            
            # Salary efficiency bonus
            salary_efficiency = projected_points / (player['salary'] / 1000)
            reward += salary_efficiency * 0.05
            
            # Position filling bonus
            if len(self.current_lineup) == DK_RULES['LINEUP_SIZE']:
                reward += 10  # Completion bonus
            
        else:
            # Invalid action - penalty
            reward = -5.0
        
        self.steps_taken += 1
        
        # Check if episode is done
        if (len(self.current_lineup) >= DK_RULES['LINEUP_SIZE'] or 
            self.steps_taken >= self.max_steps):
            done = True
            
            # Final reward based on actual performance
            actual_total_points = self._calculate_actual_lineup_performance()
            projected_total_points = self._calculate_projected_lineup_performance()
            
            # Reward for accuracy (actual vs projected)
            accuracy_bonus = 20 - abs(actual_total_points - projected_total_points)
            reward += max(0, accuracy_bonus)
            
            # Reward for absolute performance
            reward += actual_total_points * 0.2
            
            info['actual_points'] = actual_total_points
            info['projected_points'] = projected_total_points
            info['lineup_size'] = len(self.current_lineup)
            info['salary_used'] = self.current_salary_used
        
        next_state = self._get_state()
        return next_state, reward, done, False, info
    
    def _calculate_actual_lineup_performance(self):
        """Calculate actual performance of current lineup"""
        total_points = 0
        for player_idx in self.current_lineup:
            if player_idx < len(self.current_predictions):
                player_name = self.current_predictions.iloc[player_idx]['Name']
                actual_performance = self.current_actual_performance[
                    self.current_actual_performance['Name'] == player_name
                ]
                if len(actual_performance) > 0:
                    total_points += actual_performance['calculated_dk_fpts'].iloc[0]
        return total_points
    
    def _calculate_projected_lineup_performance(self):
        """Calculate projected performance of current lineup"""
        total_points = 0
        for player_idx in self.current_lineup:
            if player_idx < len(self.current_predictions):
                total_points += self.current_predictions.iloc[player_idx]['predicted_dk_fpts']
        return total_points
    
    def get_current_lineup_info(self):
        """Get detailed info about current lineup"""
        lineup_info = []
        for player_idx in self.current_lineup:
            if player_idx < len(self.current_predictions):
                player = self.current_predictions.iloc[player_idx]
                lineup_info.append({
                    'Name': player['Name'],
                    'Position': player['position'],
                    'Salary': player['salary'],
                    'Projected_Points': player['predicted_dk_fpts']
                })
        return lineup_info

class RealisticRLAgent:
    """RL Agent that learns from predictions vs actual performance"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 100
        self.learn_step = 0
        
    def _build_network(self):
        """Build neural network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def act(self, state, valid_actions=None):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            if valid_actions:
                return np.random.choice(valid_actions)
            return np.random.randint(0, self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        if valid_actions:
            # Mask invalid actions
            masked_q_values = q_values.clone()
            invalid_actions = [i for i in range(self.action_size) if i not in valid_actions]
            if invalid_actions:
                masked_q_values[0, invalid_actions] = -float('inf')
            return masked_q_values.argmax().item()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train the network"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
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

class RealisticMLBRLSystem:
    """Complete system that uses predictions + PuLP/RL for lineup optimization"""
    
    def __init__(self, data_path: str, prediction_model_path: str):
        self.data_path = data_path
        self.prediction_model_path = prediction_model_path
        
        # Initialize PuLP optimizer
        self.pulp_optimizer = PuLPLineupOptimizer()
        
        # Initialize environment
        self.env = PredictionBasedRLEnvironment(prediction_model_path, data_path)
        
        # Initialize agent
        self.agent = RealisticRLAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_space.n
        )
        
        self.training_history = []
        
    def generate_lineup_with_pulp(self, players_data: List[Dict], 
                                 use_projections: bool = True) -> pd.DataFrame:
        """
        Generate lineup using PuLP optimization
        
        Args:
            players_data: List of player dictionaries with keys:
                         'name', 'position', 'salary', 'projected_points', 'actual_points'
            use_projections: Whether to optimize for projected or actual points
            
        Returns:
            DataFrame with optimal lineup
        """
        # Convert to DataFrame
        players_df = pd.DataFrame(players_data)
        
        # Determine optimization metric
        maximize_metric = 'projected_points' if use_projections else 'actual_points'
        
        # Use PuLP optimizer
        optimal_lineup = self.pulp_optimizer.optimize_lineup(
            players_df, 
            use_projections=use_projections,
            maximize_metric=maximize_metric
        )
        
        return optimal_lineup
    
    def generate_lineup_with_rl(self, date: str) -> pd.DataFrame:
        """
        Generate lineup using RL agent (original method)
        """
        logger.info(f"Generating lineup with RL for {date}")
        
        # Get predictions for the date
        target_date = pd.to_datetime(date)
        predictions = self.env.get_predictions_for_date(target_date)
        
        if len(predictions) == 0:
            logger.warning(f"No predictions available for {date}")
            return None
        
        # Set up temporary episode
        self.env.episode_date = target_date
        self.env.current_predictions = predictions
        self.env.current_lineup = []
        self.env.current_salary_used = 0
        
        # Use trained agent to select lineup
        state = self.env._get_state()
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration for prediction
        
        lineup_info = []
        
        for step in range(DK_RULES['LINEUP_SIZE']):
            # Get valid actions
            valid_actions = []
            for i in range(len(self.env.current_predictions)):
                if self.env._is_valid_action(i):
                    valid_actions.append(i)
            
            if not valid_actions:
                break
            
            action = self.agent.act(state, valid_actions)
            next_state, reward, done, _, info = self.env.step(action)
            state = next_state
            
            if done:
                lineup_info = self.env.get_current_lineup_info()
                break
        
        # Restore epsilon
        self.agent.epsilon = original_epsilon
        
        return pd.DataFrame(lineup_info)
    
    def generate_optimal_lineup(self, players_data: List[Dict] = None, 
                              date: str = None, 
                              method: str = 'pulp') -> pd.DataFrame:
        """
        Generate optimal lineup using specified method
        
        Args:
            players_data: List of player data (required for PuLP method)
            date: Date for RL method
            method: 'pulp' or 'rl'
            
        Returns:
            DataFrame with optimal lineup
        """
        if method == 'pulp':
            if players_data is None:
                raise ValueError("players_data required for PuLP method")
            return self.generate_lineup_with_pulp(players_data)
        
        elif method == 'rl':
            if date is None:
                raise ValueError("date required for RL method")
            return self.generate_lineup_with_rl(date)
        
        else:
            raise ValueError("method must be 'pulp' or 'rl'")
    
    def compare_methods(self, players_data: List[Dict], date: str) -> Dict:
        """
        Compare PuLP vs RL lineup generation methods
        """
        results = {}
        
        # Generate lineup with PuLP
        try:
            pulp_lineup = self.generate_lineup_with_pulp(players_data)
            results['pulp'] = {
                'lineup': pulp_lineup,
                'total_salary': pulp_lineup['salary'].sum(),
                'total_projected': pulp_lineup['projected_points'].sum(),
                'players': len(pulp_lineup)
            }
        except Exception as e:
            logger.error(f"PuLP method failed: {e}")
            results['pulp'] = None
        
        # Generate lineup with RL
        try:
            rl_lineup = self.generate_lineup_with_rl(date)
            if rl_lineup is not None and len(rl_lineup) > 0:
                results['rl'] = {
                    'lineup': rl_lineup,
                    'total_salary': rl_lineup['Salary'].sum(),
                    'total_projected': rl_lineup['Projected_Points'].sum(),
                    'players': len(rl_lineup)
                }
            else:
                results['rl'] = None
        except Exception as e:
            logger.error(f"RL method failed: {e}")
            results['rl'] = None
        
        return results
        
    def train(self, episodes=1000):
        """Train the RL agent"""
        logger.info(f"Starting training for {episodes} episodes...")
        
        episode_rewards = []
        episode_actual_points = []
        episode_projected_points = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            while True:
                # Get valid actions
                valid_actions = []
                for i in range(len(self.env.current_predictions)):
                    if self.env._is_valid_action(i):
                        valid_actions.append(i)
                
                if not valid_actions:
                    break
                
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, _, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    episode_rewards.append(total_reward)
                    if 'actual_points' in info:
                        episode_actual_points.append(info['actual_points'])
                        episode_projected_points.append(info['projected_points'])
                    break
            
            # Train agent
            if len(self.agent.memory) > 32:
                self.agent.replay(32)
            
            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_actual = np.mean(episode_actual_points[-100:]) if episode_actual_points else 0
                avg_projected = np.mean(episode_projected_points[-100:]) if episode_projected_points else 0
                
                logger.info(f"Episode {episode}")
                logger.info(f"  Avg Reward: {avg_reward:.2f}")
                logger.info(f"  Avg Actual Points: {avg_actual:.2f}")
                logger.info(f"  Avg Projected Points: {avg_projected:.2f}")
                logger.info(f"  Epsilon: {self.agent.epsilon:.3f}")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_actual_points': episode_actual_points,
            'episode_projected_points': episode_projected_points
        }
        
        logger.info("Training completed!")
    
    def predict_lineup(self, date: str, method: str = 'pulp'):
        """Predict optimal lineup for a specific date using specified method"""
        logger.info(f"Predicting lineup for {date} using {method}")
        
        # Get predictions for the date
        target_date = pd.to_datetime(date)
        predictions = self.env.get_predictions_for_date(target_date)
        
        if len(predictions) == 0:
            logger.warning(f"No predictions available for {date}")
            return None
        
        if method == 'pulp':
            # Convert predictions to player data format
            players_data = []
            for _, row in predictions.iterrows():
                players_data.append({
                    'name': row['Name'],
                    'position': row.get('position', 'UTIL'),
                    'salary': row.get('salary', 5000),
                    'projected_points': row.get('predicted_dk_fpts', 0),
                    'actual_points': row.get('calculated_dk_fpts', 0)
                })
            
            return self.generate_lineup_with_pulp(players_data)
        
        else:
            # Use RL method
            return self.generate_lineup_with_rl(date)
    
    def evaluate_predictions_vs_actual(self, num_episodes=50):
        """Evaluate how well the system predicts vs actual performance"""
        logger.info("Evaluating predictions vs actual performance...")
        
        results = []
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            
            while True:
                valid_actions = []
                for i in range(len(self.env.current_predictions)):
                    if self.env._is_valid_action(i):
                        valid_actions.append(i)
                
                if not valid_actions:
                    break
                
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, _, info = self.env.step(action)
                state = next_state
                
                if done:
                    results.append({
                        'episode': episode,
                        'date': self.env.episode_date,
                        'actual_points': info.get('actual_points', 0),
                        'projected_points': info.get('projected_points', 0),
                        'lineup_size': info.get('lineup_size', 0),
                        'salary_used': info.get('salary_used', 0)
                    })
                    break
        
        self.agent.epsilon = original_epsilon
        
        # Calculate metrics
        results_df = pd.DataFrame(results)
        
        avg_actual = results_df['actual_points'].mean()
        avg_projected = results_df['projected_points'].mean()
        prediction_accuracy = 1 - abs(avg_actual - avg_projected) / max(avg_actual, 1)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Average Actual Points: {avg_actual:.2f}")
        logger.info(f"  Average Projected Points: {avg_projected:.2f}")
        logger.info(f"  Prediction Accuracy: {prediction_accuracy:.2%}")
        
        return results_df

class PuLPLineupOptimizer:
    """
    PuLP-based lineup optimizer that integrates with RL system
    """
    
    def __init__(self):
        self.pulp_available = PULP_AVAILABLE
        
    def optimize_lineup(self, players_df: pd.DataFrame, 
                       use_projections: bool = True,
                       maximize_metric: str = 'projected_points') -> pd.DataFrame:
        """
        Optimize lineup using PuLP with DraftKings constraints
        
        Args:
            players_df: DataFrame with player data
            use_projections: Whether to use projected points or actual points
            maximize_metric: Column to maximize ('projected_points' or 'actual_points')
            
        Returns:
            DataFrame with optimal lineup
        """
        if not self.pulp_available:
            logger.warning("PuLP not available, using greedy optimization")
            return self._greedy_optimize(players_df, maximize_metric)
        
        try:
            # Create optimization problem
            prob = pulp.LpProblem("DraftKings_Lineup", pulp.LpMaximize)
            
            # Create binary variables for each player
            player_vars = {}
            for i, row in players_df.iterrows():
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
            
            # Objective: Maximize projected/actual points
            if maximize_metric in players_df.columns:
                prob += pulp.lpSum([player_vars[i] * players_df.loc[i, maximize_metric] 
                                  for i in players_df.index])
            else:
                logger.warning(f"Metric {maximize_metric} not found, using first numeric column")
                numeric_cols = players_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    prob += pulp.lpSum([player_vars[i] * players_df.loc[i, numeric_cols[0]] 
                                      for i in players_df.index])
            
            # Constraint: Total salary <= salary cap
            if 'salary' in players_df.columns:
                prob += pulp.lpSum([player_vars[i] * players_df.loc[i, 'salary'] 
                                  for i in players_df.index]) <= DK_RULES['SALARY_CAP']
            
            # Constraint: Exactly 8 players
            prob += pulp.lpSum([player_vars[i] for i in players_df.index]) == DK_RULES['LINEUP_SIZE']
            
            # Position constraints
            if 'position' in players_df.columns:
                position_groups = players_df.groupby('position')
                
                # Standard position constraints (minimum required)
                for position, required_count in DK_RULES['POSITIONS'].items():
                    if position == 'UTIL':
                        continue  # Handle UTIL separately
                    
                    if position in position_groups.groups:
                        position_players = position_groups.get_group(position).index
                        prob += pulp.lpSum([player_vars[i] for i in position_players]) >= required_count
                
                # Maximum position constraints (prevent overloading)
                for position in position_groups.groups:
                    if position == 'OF':
                        # Allow up to 4 OF (3 OF + 1 UTIL)
                        position_players = position_groups.get_group(position).index
                        prob += pulp.lpSum([player_vars[i] for i in position_players]) <= 4
                    elif position != 'UTIL':
                        # Allow up to 2 of each other position (1 position + 1 UTIL)
                        position_players = position_groups.get_group(position).index
                        prob += pulp.lpSum([player_vars[i] for i in position_players]) <= 2
            
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract solution
            if prob.status == pulp.LpStatusOptimal:
                selected_players = []
                for i in players_df.index:
                    if player_vars[i].value() == 1:
                        selected_players.append(i)
                
                optimal_lineup = players_df.loc[selected_players].copy()
                
                logger.info(f"PuLP optimization successful!")
                logger.info(f"Lineup size: {len(optimal_lineup)}")
                logger.info(f"Total salary: ${optimal_lineup['salary'].sum():,}")
                logger.info(f"Total {maximize_metric}: {optimal_lineup[maximize_metric].sum():.1f}")
                
                return optimal_lineup
            else:
                logger.warning(f"PuLP optimization failed with status: {prob.status}")
                return self._greedy_optimize(players_df, maximize_metric)
                
        except Exception as e:
            logger.error(f"PuLP optimization error: {e}")
            return self._greedy_optimize(players_df, maximize_metric)
    
    def _greedy_optimize(self, players_df: pd.DataFrame, maximize_metric: str) -> pd.DataFrame:
        """
        Greedy optimization fallback when PuLP fails
        """
        logger.info("Using greedy optimization fallback")
        
        # Calculate value efficiency (points per dollar)
        df_work = players_df.copy()
        
        if maximize_metric in df_work.columns and 'salary' in df_work.columns:
            df_work['value'] = df_work[maximize_metric] / df_work['salary']
        else:
            df_work['value'] = np.random.random(len(df_work))  # Random selection
        
        # Sort by value
        df_work = df_work.sort_values('value', ascending=False)
        
        # Greedy selection
        selected_players = []
        total_salary = 0
        position_counts = {}
        
        for _, player in df_work.iterrows():
            if len(selected_players) >= DK_RULES['LINEUP_SIZE']:
                break
            
            # Check salary constraint
            if total_salary + player['salary'] > DK_RULES['SALARY_CAP']:
                continue
            
            # Check position constraints
            player_pos = player.get('position', 'UTIL')
            current_count = position_counts.get(player_pos, 0)
            max_count = DK_RULES['POSITIONS'].get(player_pos, 1)
            
            if current_count < max_count:
                selected_players.append(player.name)
                total_salary += player['salary']
                position_counts[player_pos] = current_count + 1
        
        if len(selected_players) > 0:
            return players_df.loc[selected_players]
        else:
            # Return top players if no valid lineup found
            return players_df.head(min(DK_RULES['LINEUP_SIZE'], len(players_df)))

def main():
    """Main function to demonstrate the realistic RL system with PuLP integration"""
    
    # Configuration
    DATA_PATH = '5_ENTRIES/data_with_dk_entries_salaries.csv'
    PREDICTION_MODEL_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    
    # Initialize system
    logger.info("Initializing Realistic MLB RL System with PuLP integration...")
    rl_system = RealisticMLBRLSystem(DATA_PATH, PREDICTION_MODEL_PATH)
    
    # Load sample data for demonstration
    logger.info("Loading sample data...")
    sample_data = pd.read_csv(DATA_PATH, nrows=100)
    
    # Create sample players data
    players_data = []
    for _, row in sample_data.iterrows():
        players_data.append({
            'name': row.get('Name', f'Player_{len(players_data)}'),
            'position': row.get('position', 'OF'),
            'salary': int(row.get('salary', 5000)),
            'projected_points': row.get('rolling_30_ppg', 10),
            'actual_points': row.get('calculated_dk_fpts', 8)
        })
    
    # Test PuLP method
    logger.info("\n=== Testing PuLP Lineup Optimization ===")
    try:
        pulp_lineup = rl_system.generate_lineup_with_pulp(players_data[:50])  # Use first 50 players
        
        if pulp_lineup is not None and len(pulp_lineup) > 0:
            logger.info("PuLP Optimal Lineup:")
            logger.info(f"Total Salary: ${pulp_lineup['salary'].sum():,.0f}")
            logger.info(f"Total Projected Points: {pulp_lineup['projected_points'].sum():.1f}")
            logger.info(f"Players: {len(pulp_lineup)}")
            
            logger.info("\nLineup Details:")
            for _, player in pulp_lineup.iterrows():
                logger.info(f"  {player['name']} ({player['position']}) - "
                           f"${player['salary']:,.0f} - {player['projected_points']:.1f} pts")
        else:
            logger.warning("PuLP optimization failed")
    except Exception as e:
        logger.error(f"PuLP method error: {e}")
    
    # Test RL method
    logger.info("\n=== Testing RL Lineup Generation ===")
    try:
        rl_lineup = rl_system.predict_lineup('2025-07-02', method='rl')
        
        if rl_lineup is not None and len(rl_lineup) > 0:
            logger.info("RL Generated Lineup:")
            logger.info(f"Total Salary: ${rl_lineup['Salary'].sum():,.0f}")
            logger.info(f"Total Projected Points: {rl_lineup['Projected_Points'].sum():.1f}")
            logger.info(f"Players: {len(rl_lineup)}")
            
            logger.info("\nLineup Details:")
            for _, player in rl_lineup.iterrows():
                logger.info(f"  {player['Name']} ({player['Position']}) - "
                           f"${player['Salary']:,.0f} - {player['Projected_Points']:.1f} pts")
        else:
            logger.warning("RL method failed")
    except Exception as e:
        logger.error(f"RL method error: {e}")
    
    # Compare methods
    logger.info("\n=== Comparing PuLP vs RL Methods ===")
    try:
        comparison = rl_system.compare_methods(players_data[:30], '2025-07-02')
        
        if comparison.get('pulp'):
            pulp_result = comparison['pulp']
            logger.info(f"PuLP Method: {pulp_result['players']} players, "
                       f"${pulp_result['total_salary']:,.0f} salary, "
                       f"{pulp_result['total_projected']:.1f} projected points")
        
        if comparison.get('rl'):
            rl_result = comparison['rl']
            logger.info(f"RL Method: {rl_result['players']} players, "
                       f"${rl_result['total_salary']:,.0f} salary, "
                       f"{rl_result['total_projected']:.1f} projected points")
    except Exception as e:
        logger.error(f"Comparison error: {e}")
    
    # Train RL agent (optional)
    logger.info("\n=== Training RL Agent (Optional) ===")
    train_rl = input("Train RL agent? (y/n): ").lower() == 'y'
    
    if train_rl:
        logger.info("Training RL agent...")
        rl_system.train(episodes=100)  # Short training for demo
        
        # Test trained agent
        logger.info("Testing trained RL agent...")
        trained_lineup = rl_system.predict_lineup('2025-07-02', method='rl')
        
        if trained_lineup is not None and len(trained_lineup) > 0:
            logger.info("Trained RL Agent Lineup:")
            logger.info(f"Total Salary: ${trained_lineup['Salary'].sum():,.0f}")
            logger.info(f"Total Projected Points: {trained_lineup['Projected_Points'].sum():.1f}")
    
    logger.info("\n=== System Demonstration Complete ===")
    logger.info("The system now supports both PuLP and RL methods for lineup optimization!")
    logger.info("Use method='pulp' for constraint-based optimization")
    logger.info("Use method='rl' for machine learning-based optimization")

if __name__ == "__main__":
    main()
