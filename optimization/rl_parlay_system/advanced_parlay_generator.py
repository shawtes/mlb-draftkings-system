#!/usr/bin/env python3
"""
Advanced Parlay Generator with Historical Statistics - IMPROVED VERSION
Uses research-backed improvements to increase win rate
"""

import pandas as pd
import numpy as np
import random
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy import stats

@dataclass
class ParlayLeg:
    """Represents a single parlay leg"""
    player_id: str
    player_name: str
    team: str
    position: str
    prop_type: str
    line: float
    projection: float
    std_dev: float
    hit_rate: float

@dataclass
class Parlay:
    """Represents a complete parlay"""
    legs: List[ParlayLeg]
    estimated_odds: float
    combined_hit_rate: float
    expected_value: float

class AdvancedParlayGenerator:
    """Improved parlay generator with research-based optimizations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prop_types = ['passing_yds', 'rushing_yds', 'receiving_yds', 'receptions', 'dk_points']
        # More conservative line multipliers (60-65% range)
        self.line_multipliers = [0.50, 0.55, 0.60, 0.65, 0.70]
        # Prop preference priority (higher = safer)
        self.prop_priority = {
            'receptions': 3,
            'receiving_yds': 2,
            'rushing_yds': 2,
            'dk_points': 2,
            'passing_yds': 1
        }
    
    def generate_parlay(self, max_legs: int = 4) -> Parlay:
        """Generate a single parlay with improved logic"""
        # Get available players
        current_week = random.choice(self.data['week'].unique())
        current_year = random.choice(self.data['year'].unique())
        
        available_players = self.data[
            (self.data['week'] == current_week) & 
            (self.data['year'] == current_year)
        ].copy()
        
        if len(available_players) == 0:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Filter for players with good projections
        good_players = available_players[available_players['projected_dk_points'] > 5].copy()
        
        if len(good_players) == 0:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Generate parlay legs
        legs = []
        selected_players = set()
        selected_teams = set()
        qb_count = 0  # Limit QB props
        
        for _ in range(min(max_legs, len(good_players))):
            # Prefer 2-3 leg parlays
            if len(legs) >= 2 and random.random() < 0.3:
                break
            
            # Select a random player
            remaining_players = good_players[~good_players['player_id'].isin(selected_players)]
            if len(remaining_players) == 0:
                break
            
            player = remaining_players.sample(1).iloc[0]
            selected_players.add(player['player_id'])
            
            # Select prop type (prefer safer props)
            prop_type = self._select_prop_type_improved(player['position_proj'], qb_count)
            
            if prop_type == 'passing_yds':
                qb_count += 1
            
            # Get projection
            projection = self._get_projection(player, prop_type)
            
            if projection <= 0:
                continue
            
            # Use conservative line multiplier (60-65% range)
            line_multiplier = random.choices(
                self.line_multipliers,
                weights=[0.1, 0.2, 0.35, 0.25, 0.1]  # Favor 0.60-0.65
            )[0]
            
            line = projection * line_multiplier
            
            # Calculate hit rate with improved variance estimates
            hit_rate = self._calculate_hit_rate_improved(player, prop_type, line, projection)
            
            # Reject if hit rate too low
            if hit_rate < 0.55:
                continue
            
            # Get std for the leg
            std = projection * self._get_default_cv_improved(player['position_proj'], prop_type)
            
            # Create parlay leg
            leg = ParlayLeg(
                player_id=player['player_id'],
                player_name=player['player_name_proj'],
                team=player['team_proj'],
                position=player['position_proj'],
                prop_type=prop_type,
                line=line,
                projection=projection,
                std_dev=std,
                hit_rate=hit_rate
            )
            
            legs.append(leg)
            selected_teams.add(player['team_proj'])
        
        if not legs:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Calculate parlay metrics
        combined_hit_rate = 1.0
        for leg in legs:
            combined_hit_rate *= leg.hit_rate
        
        estimated_odds = 1.0 / combined_hit_rate - 1.0
        expected_value = (combined_hit_rate * estimated_odds * 100) - 100
        
        return Parlay(
            legs=legs,
            estimated_odds=estimated_odds,
            combined_hit_rate=combined_hit_rate,
            expected_value=expected_value
        )
    
    def _select_prop_type_improved(self, position: str, qb_count: int) -> str:
        """Improved prop selection with priority system"""
        # Limit QB props to 1 per parlay
        if qb_count >= 1 and position == 'QB':
            return 'dk_points'
        
        if position == 'QB':
            return random.choice(['passing_yds', 'dk_points'])
        elif position == 'RB':
            # Prefer receptions and rushing yards
            options = ['receptions', 'rushing_yds', 'receiving_yds', 'dk_points']
            weights = [0.3, 0.3, 0.2, 0.2]
            return random.choices(options, weights=weights)[0]
        elif position in ['WR', 'TE']:
            # Prefer receptions over receiving yards
            options = ['receptions', 'receiving_yds', 'dk_points']
            weights = [0.4, 0.3, 0.3]
            return random.choices(options, weights=weights)[0]
        else:
            return 'dk_points'
    
    def _get_projection(self, player: pd.Series, prop_type: str) -> float:
        """Get projection for specific prop type"""
        if prop_type == 'passing_yds':
            return player.get('projected_passing_yds', 0)
        elif prop_type == 'rushing_yds':
            return player.get('projected_rushing_yds', 0)
        elif prop_type == 'receiving_yds':
            return player.get('projected_receiving_yds', 0)
        elif prop_type == 'receptions':
            return player.get('projected_receptions', 0)
        else:
            return player.get('projected_dk_points', 0)
    
    def _calculate_hit_rate_improved(self, player: pd.Series, prop_type: str, line: float, projection: float) -> float:
        """Calculate hit rate with improved variance estimates (1.5-2x multiplier)"""
        mean = projection
        
        # Get historical std with multiplier for better estimates
        if prop_type == 'passing_yds':
            accuracy_std = player.get('passing_yds_accuracy_std', 0)
            std = projection * max(0.40, abs(accuracy_std)) * 1.5  # 1.5x multiplier
        elif prop_type == 'rushing_yds':
            accuracy_std = player.get('rushing_yds_accuracy_std', 0)
            std = projection * max(0.40, abs(accuracy_std)) * 1.5
        elif prop_type == 'receiving_yds':
            accuracy_std = player.get('receiving_yds_accuracy_std', 0)
            std = projection * max(0.45, abs(accuracy_std)) * 1.5
        elif prop_type == 'receptions':
            accuracy_std = player.get('receptions_accuracy_std', 0)
            std = projection * max(0.40, abs(accuracy_std)) * 1.5
        else:
            accuracy_std = player.get('dk_points_accuracy_std', 0)
            std = projection * max(0.35, abs(accuracy_std)) * 1.5
        
        # If no std, use improved default CV
        if std == 0:
            cv = self._get_default_cv_improved(player.get('position_proj', ''), prop_type)
            std = projection * cv
        
        # Calculate probability using normal distribution
        if std > 0:
            hit_rate = 1 - stats.norm.cdf(line, loc=mean, scale=std)
        else:
            hit_rate = player.get('dk_points_hit_mean', 0.5)
        
        # Reasonable bounds
        return max(0.35, min(0.80, hit_rate))
    
    def _get_default_cv_improved(self, position: str, prop_type: str) -> float:
        """Improved CV estimates (higher variance = more conservative)"""
        cv_map = {
            'QB': {
                'passing_yds': 0.45,  # Increased from 0.35
                'rushing_yds': 0.60,
                'dk_points': 0.40
            },
            'RB': {
                'rushing_yds': 0.50,
                'receiving_yds': 0.55,
                'dk_points': 0.45
            },
            'WR': {
                'receiving_yds': 0.55,
                'receptions': 0.45,
                'dk_points': 0.45
            },
            'TE': {
                'receiving_yds': 0.50,
                'receptions': 0.40,
                'dk_points': 0.40
            }
        }
        return cv_map.get(position, {}).get(prop_type, 0.45)
    
    def generate_multiple_parlays(self, num_parlays: int = 5, max_legs: int = 4) -> List[Parlay]:
        """Generate multiple parlays"""
        parlays = []
        for _ in range(num_parlays):
            parlay = self.generate_parlay(max_legs)
            if parlay.legs:
                parlays.append(parlay)
        return parlays

def main():
    """Main function"""
    print("🎯 Improved Parlay Generator (Research-Based Optimizations)")
    print("=" * 60)
    
    if not os.path.exists('enhanced_training_data.csv'):
        print("❌ Enhanced training data not found")
        return
    
    data = pd.read_csv('enhanced_training_data.csv')
    print(f"✅ Loaded data: {len(data)} records")
    
    generator = AdvancedParlayGenerator(data)
    
    print("\n🎲 Generating improved parlays...")
    parlays = generator.generate_multiple_parlays(num_parlays=5, max_legs=4)
    
    for i, parlay in enumerate(parlays, 1):
        print(f"\n--- Parlay {i} ---")
        print(f"Legs: {len(parlay.legs)} | Hit Rate: {parlay.combined_hit_rate:.1%} | Odds: +{parlay.estimated_odds:.0f}")
        for j, leg in enumerate(parlay.legs, 1):
            print(f"  {j}. {leg.player_name} ({leg.team}) - {leg.prop_type} O{leg.line:.1f} ({leg.hit_rate:.0%})")
    
    if parlays:
        avg_hit_rate = np.mean([p.combined_hit_rate for p in parlays])
        print(f"\n📊 Average Hit Rate: {avg_hit_rate:.1%}")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
