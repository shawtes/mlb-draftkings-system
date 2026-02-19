#!/usr/bin/env python3
"""
RL Parlay Environment
Defines the environment for reinforcement learning parlay generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces
import random
from dataclasses import dataclass

@dataclass
class ParlayLeg:
    """Represents a single parlay leg"""
    player_id: str
    player_name: str
    team: str
    position: str
    prop_type: str  # 'passing_yds', 'rushing_yds', 'receiving_yds', 'receptions', 'dk_points'
    line: float
    projection: float
    confidence: float
    hit_rate: float

@dataclass
class Parlay:
    """Represents a complete parlay"""
    legs: List[ParlayLeg]
    estimated_odds: float
    combined_hit_rate: float
    expected_value: float

class ParlayEnvironment(gym.Env):
    """
    RL Environment for generating optimal parlays
    """
    
    def __init__(self, training_data: pd.DataFrame, max_legs: int = 4):
        super().__init__()
        
        self.training_data = training_data
        self.max_legs = max_legs
        
        # Define action space
        # Action format: [player_idx, prop_type, line_multiplier]
        # player_idx: index in available players (0 to n_players-1)
        # prop_type: 0=passing_yds, 1=rushing_yds, 2=receiving_yds, 3=receptions, 4=dk_points
        # line_multiplier: 0.5, 0.6, 0.7, 0.8, 0.9 (70% of projection = 0.7)
        
        self.n_players = len(training_data)
        self.n_prop_types = 5
        self.n_line_multipliers = 5
        
        # Action space: [player_idx, prop_type, line_multiplier]
        self.action_space = spaces.MultiDiscrete([
            self.n_players,      # player selection
            self.n_prop_types,   # prop type
            self.n_line_multipliers  # line multiplier
        ])
        
        # State space: [player_features, game_context, parlay_state]
        # Player features: 20 features per player
        # Game context: 10 features
        # Parlay state: 10 features
        self.state_dim = (self.n_players * 20) + 10 + 10
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Select a random game/week for this episode
        self.current_game = self.training_data.sample(1).iloc[0]
        self.current_week = self.current_game['week']
        self.current_year = self.current_game['year']
        
        # Get available players for this game
        self.available_players = self.training_data[
            (self.training_data['week'] == self.current_week) & 
            (self.training_data['year'] == self.current_year)
        ].copy()
        
        # Initialize parlay
        self.current_parlay = []
        self.episode_reward = 0.0
        self.step_count = 0
        self.max_steps = self.max_legs
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        player_idx, prop_type, line_multiplier = action
        
        # Check if action is valid
        if player_idx >= len(self.available_players):
            return self._get_state(), -1.0, True, {"error": "Invalid player index"}
        
        if self.step_count >= self.max_steps:
            return self._get_state(), 0.0, True, {"error": "Max steps reached"}
        
        # Get player and create parlay leg
        player = self.available_players.iloc[player_idx]
        leg = self._create_parlay_leg(player, prop_type, line_multiplier)
        
        # Add leg to parlay
        self.current_parlay.append(leg)
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(leg)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.step_count >= self.max_steps or len(self.current_parlay) >= self.max_legs
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            "parlay_legs": len(self.current_parlay),
            "current_leg": leg,
            "episode_reward": self.episode_reward
        }
        
        return next_state, reward, done, info
    
    def _create_parlay_leg(self, player: pd.Series, prop_type: int, line_multiplier: int) -> ParlayLeg:
        """Create a parlay leg from player data and action"""
        prop_types = ['passing_yds', 'rushing_yds', 'receiving_yds', 'receptions', 'dk_points']
        line_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        prop_type_name = prop_types[prop_type]
        multiplier = line_multipliers[line_multiplier]
        
        # Get projection for the prop type
        if prop_type_name == 'passing_yds':
            projection = player['projected_passing_yds']
        elif prop_type_name == 'rushing_yds':
            projection = player['projected_rushing_yds']
        elif prop_type_name == 'receiving_yds':
            projection = player['projected_receiving_yds']
        elif prop_type_name == 'receptions':
            projection = player['projected_receptions']
        else:  # dk_points
            projection = player['projected_dk_points']
        
        # Calculate line and confidence
        line = projection * multiplier
        confidence = self._calculate_confidence(player, prop_type_name)
        hit_rate = self._calculate_hit_rate(player, prop_type_name, multiplier)
        
        return ParlayLeg(
            player_id=player['player_id'],
            player_name=player['player_name_proj'],
            team=player['team_proj'],
            position=player['position_proj'],
            prop_type=prop_type_name,
            line=line,
            projection=projection,
            confidence=confidence,
            hit_rate=hit_rate
        )
    
    def _calculate_confidence(self, player: pd.Series, prop_type: str) -> float:
        """Calculate confidence score for a prop based on player history"""
        # Use historical accuracy and consistency
        if prop_type == 'passing_yds':
            accuracy = player.get('passing_yds_accuracy_mean', 1.0)
            consistency = 1.0 - player.get('passing_yds_accuracy_std', 0.3)
        elif prop_type == 'rushing_yds':
            accuracy = player.get('rushing_yds_accuracy_mean', 1.0)
            consistency = 1.0 - player.get('rushing_yds_accuracy_std', 0.3)
        elif prop_type == 'receiving_yds':
            accuracy = player.get('receiving_yds_accuracy_mean', 1.0)
            consistency = 1.0 - player.get('receiving_yds_accuracy_std', 0.3)
        elif prop_type == 'receptions':
            accuracy = player.get('receptions_accuracy_mean', 1.0)
            consistency = 1.0 - player.get('receptions_accuracy_std', 0.3)
        else:  # dk_points
            accuracy = player.get('dk_points_accuracy_mean', 1.0)
            consistency = 1.0 - player.get('dk_points_accuracy_std', 0.3)
        
        # Combine accuracy and consistency
        confidence = (accuracy * 0.6 + consistency * 0.4)
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_hit_rate(self, player: pd.Series, prop_type: str, multiplier: float) -> float:
        """Calculate historical hit rate for a prop at given multiplier"""
        # Get historical hit rate for this prop type
        if prop_type == 'passing_yds':
            base_hit_rate = player.get('passing_yds_hit_mean', 0.5)
        elif prop_type == 'rushing_yds':
            base_hit_rate = player.get('rushing_yds_hit_mean', 0.5)
        elif prop_type == 'receiving_yds':
            base_hit_rate = player.get('receiving_yds_hit_mean', 0.5)
        elif prop_type == 'receptions':
            base_hit_rate = player.get('receptions_hit_mean', 0.5)
        else:  # dk_points
            base_hit_rate = player.get('dk_points_hit_mean', 0.5)
        
        # Adjust hit rate based on line multiplier
        # Lower multiplier (easier line) = higher hit rate
        adjusted_hit_rate = base_hit_rate * (1.0 - (0.7 - multiplier) * 0.3)
        return min(max(adjusted_hit_rate, 0.0), 1.0)
    
    def _calculate_reward(self, leg: ParlayLeg) -> float:
        """Calculate reward for adding a parlay leg"""
        # Base reward for adding a leg
        base_reward = 0.1
        
        # Reward for high hit rate
        hit_rate_reward = leg.hit_rate * 0.3
        
        # Reward for high confidence
        confidence_reward = leg.confidence * 0.2
        
        # Penalty for too many legs (diminishing returns)
        leg_count_penalty = -0.05 * len(self.current_parlay)
        
        # Reward for prop type diversity
        prop_types = [l.prop_type for l in self.current_parlay]
        diversity_reward = 0.1 if len(set(prop_types)) > 1 else 0.0
        
        # Reward for team diversity
        teams = [l.team for l in self.current_parlay]
        team_diversity_reward = 0.1 if len(set(teams)) > 1 else 0.0
        
        total_reward = (base_reward + hit_rate_reward + confidence_reward + 
                       leg_count_penalty + diversity_reward + team_diversity_reward)
        
        return total_reward
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Player features (20 per player)
        player_features = []
        for _, player in self.available_players.iterrows():
            features = [
                player.get('projected_dk_points', 0),
                player.get('projected_passing_yds', 0),
                player.get('projected_rushing_yds', 0),
                player.get('projected_receiving_yds', 0),
                player.get('projected_receptions', 0),
                player.get('salary', 0),
                player.get('dk_points_accuracy_mean', 1.0),
                player.get('dk_points_accuracy_std', 0.3),
                player.get('passing_yds_accuracy_mean', 1.0),
                player.get('passing_yds_accuracy_std', 0.3),
                player.get('rushing_yds_accuracy_mean', 1.0),
                player.get('rushing_yds_accuracy_std', 0.3),
                player.get('receiving_yds_accuracy_mean', 1.0),
                player.get('receiving_yds_accuracy_std', 0.3),
                player.get('receptions_accuracy_mean', 1.0),
                player.get('receptions_accuracy_std', 0.3),
                player.get('dk_points_hit_mean', 0.5),
                player.get('passing_yds_hit_mean', 0.5),
                player.get('rushing_yds_hit_mean', 0.5),
                player.get('receiving_yds_hit_mean', 0.5)
            ]
            player_features.extend(features)
        
        # Pad or truncate to fixed size
        max_players = 50  # Maximum players per game
        if len(player_features) < max_players * 20:
            player_features.extend([0] * (max_players * 20 - len(player_features)))
        else:
            player_features = player_features[:max_players * 20]
        
        # Game context features (10)
        game_context = [
            self.current_week / 18.0,  # Week normalized
            self.current_year - 2020,  # Year offset
            self.current_game.get('temperature', 70) / 100.0,  # Temperature normalized
            self.current_game.get('wind_speed', 0) / 20.0,  # Wind speed normalized
            self.current_game.get('humidity', 50) / 100.0,  # Humidity normalized
            1.0 if self.current_game.get('surface', '') == 'Grass' else 0.0,  # Surface type
            1.0 if 'Dome' in str(self.current_game.get('stadium', '')) else 0.0,  # Indoor/outdoor
            self.current_game.get('total', 45) / 60.0,  # Game total normalized
            abs(self.current_game.get('spread', 0)) / 14.0,  # Spread normalized
            1.0 if self.current_game.get('weather', '') == 'Clear' else 0.0  # Weather
        ]
        
        # Parlay state features (10)
        parlay_state = [
            len(self.current_parlay) / self.max_legs,  # Current leg count
            sum([l.hit_rate for l in self.current_parlay]) / max(len(self.current_parlay), 1),  # Avg hit rate
            sum([l.confidence for l in self.current_parlay]) / max(len(self.current_parlay), 1),  # Avg confidence
            len(set([l.prop_type for l in self.current_parlay])) / 5.0,  # Prop diversity
            len(set([l.team for l in self.current_parlay])) / 2.0,  # Team diversity
            sum([l.projection for l in self.current_parlay]) / 100.0,  # Total projection
            sum([l.line for l in self.current_parlay]) / 100.0,  # Total line
            self.episode_reward,  # Current reward
            self.step_count / self.max_steps,  # Progress
            1.0 if len(self.current_parlay) > 0 else 0.0  # Has legs
        ]
        
        # Combine all features
        state = np.array(player_features + game_context + parlay_state, dtype=np.float32)
        
        return state
    
    def get_current_parlay(self) -> Parlay:
        """Get the current parlay with calculated metrics"""
        if not self.current_parlay:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Calculate combined hit rate
        combined_hit_rate = 1.0
        for leg in self.current_parlay:
            combined_hit_rate *= leg.hit_rate
        
        # Calculate estimated odds
        estimated_odds = 1.0 / combined_hit_rate - 1.0
        
        # Calculate expected value (assuming $100 bet)
        expected_value = (combined_hit_rate * estimated_odds * 100) - 100
        
        return Parlay(
            legs=self.current_parlay,
            estimated_odds=estimated_odds,
            combined_hit_rate=combined_hit_rate,
            expected_value=expected_value
        )
    
    def render(self, mode='human'):
        """Render the current state"""
        parlay = self.get_current_parlay()
        
        print(f"\n🏈 Current Parlay ({len(parlay.legs)} legs):")
        print(f"   Combined Hit Rate: {parlay.combined_hit_rate:.2%}")
        print(f"   Estimated Odds: +{parlay.estimated_odds:.0f}")
        print(f"   Expected Value: ${parlay.expected_value:.2f}")
        
        for i, leg in enumerate(parlay.legs, 1):
            print(f"   {i}. {leg.player_name} ({leg.team}) - {leg.prop_type} O{leg.line:.1f} ({leg.hit_rate:.1%})")
