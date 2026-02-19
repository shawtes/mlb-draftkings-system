#!/usr/bin/env python3
"""
Simple Parlay Generator
A working parlay generator that creates realistic parlays based on projections
"""

import pandas as pd
import numpy as np
import random
import os
from typing import List, Dict, Any
from dataclasses import dataclass

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
    hit_rate: float

@dataclass
class Parlay:
    """Represents a complete parlay"""
    legs: List[ParlayLeg]
    estimated_odds: float
    combined_hit_rate: float
    expected_value: float

class SimpleParlayGenerator:
    """Simple parlay generator using rule-based approach"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prop_types = ['passing_yds', 'rushing_yds', 'receiving_yds', 'receptions', 'dk_points']
        self.line_multipliers = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    def generate_parlay(self, max_legs: int = 4) -> Parlay:
        """Generate a single parlay"""
        # Get available players for current week
        current_week = random.choice(self.data['week'].unique())
        current_year = random.choice(self.data['year'].unique())
        
        available_players = self.data[
            (self.data['week'] == current_week) & 
            (self.data['year'] == current_year)
        ].copy()
        
        if len(available_players) == 0:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Filter for players with good projections
        good_players = available_players[available_players['projected_dk_points'] > 5]
        
        if len(good_players) == 0:
            return Parlay(legs=[], estimated_odds=0.0, combined_hit_rate=0.0, expected_value=0.0)
        
        # Generate parlay legs
        legs = []
        selected_players = set()
        selected_teams = set()
        
        for _ in range(min(max_legs, len(good_players))):
            # Select a random player (avoid duplicates)
            remaining_players = good_players[~good_players['player_id'].isin(selected_players)]
            if len(remaining_players) == 0:
                break
            
            player = remaining_players.sample(1).iloc[0]
            selected_players.add(player['player_id'])
            
            # Select prop type based on position
            prop_type = self._select_prop_type(player['position_proj'])
            
            # Get projection for the prop type
            projection = self._get_projection(player, prop_type)
            
            if projection <= 0:
                continue
            
            # Select line multiplier (prefer safer lines)
            line_multiplier = random.choices(
                self.line_multipliers, 
                weights=[0.1, 0.2, 0.4, 0.2, 0.1]  # Favor 0.7 multiplier
            )[0]
            
            line = projection * line_multiplier
            
            # Calculate hit rate based on historical accuracy
            hit_rate = self._calculate_hit_rate(player, prop_type, line_multiplier)
            
            # Create parlay leg
            leg = ParlayLeg(
                player_id=player['player_id'],
                player_name=player['player_name_proj'],
                team=player['team_proj'],
                position=player['position_proj'],
                prop_type=prop_type,
                line=line,
                projection=projection,
                hit_rate=hit_rate
            )
            
            legs.append(leg)
            
            # Add team diversity (prefer different teams)
            selected_teams.add(player['team_proj'])
            if len(selected_teams) >= 2 and len(legs) >= 2:
                break
        
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
    
    def _select_prop_type(self, position: str) -> str:
        """Select appropriate prop type based on position"""
        if position == 'QB':
            return random.choice(['passing_yds', 'dk_points'])
        elif position == 'RB':
            return random.choice(['rushing_yds', 'receiving_yds', 'receptions', 'dk_points'])
        elif position in ['WR', 'TE']:
            return random.choice(['receiving_yds', 'receptions', 'dk_points'])
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
        else:  # dk_points
            return player.get('projected_dk_points', 0)
    
    def _calculate_hit_rate(self, player: pd.Series, prop_type: str, line_multiplier: float) -> float:
        """Calculate hit rate for a prop"""
        # Base hit rate from historical data
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
        
        # Adjust based on line multiplier (easier line = higher hit rate)
        adjusted_hit_rate = base_hit_rate * (1.0 + (0.7 - line_multiplier) * 0.3)
        
        # Ensure reasonable bounds
        return max(0.3, min(0.9, adjusted_hit_rate))
    
    def generate_multiple_parlays(self, num_parlays: int = 5, max_legs: int = 4) -> List[Parlay]:
        """Generate multiple parlays"""
        parlays = []
        for _ in range(num_parlays):
            parlay = self.generate_parlay(max_legs)
            if parlay.legs:  # Only add parlays with legs
                parlays.append(parlay)
        return parlays

def main():
    """Main function to demonstrate the parlay generator"""
    print("🎯 Simple Parlay Generator Demo")
    print("=" * 40)
    
    # Load demo data
    if not os.path.exists('rl_demo_data.csv'):
        print("❌ Demo data not found. Run main.py demo first.")
        return
    
    data = pd.read_csv('rl_demo_data.csv')
    print(f"✅ Loaded data: {len(data)} records")
    
    # Create generator
    generator = SimpleParlayGenerator(data)
    
    # Generate parlays
    print("\n🎲 Generating parlays...")
    parlays = generator.generate_multiple_parlays(num_parlays=5, max_legs=4)
    
    # Display results
    for i, parlay in enumerate(parlays, 1):
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
        avg_legs = np.mean([len(p.legs) for p in parlays])
        avg_hit_rate = np.mean([p.combined_hit_rate for p in parlays])
        avg_odds = np.mean([p.estimated_odds for p in parlays])
        avg_ev = np.mean([p.expected_value for p in parlays])
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Average Legs: {avg_legs:.1f}")
        print(f"   Average Hit Rate: {avg_hit_rate:.2%}")
        print(f"   Average Odds: +{avg_odds:.0f}")
        print(f"   Average Expected Value: ${avg_ev:.2f}")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()
