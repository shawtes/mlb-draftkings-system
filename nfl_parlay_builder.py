#!/usr/bin/env python3
"""
NFL Parlay Builder using SportsData.io Data
==========================================

This script creates NFL prop bet parlays using the latest sportsdata.io projections.
It analyzes player projections and builds optimal parlay combinations for NFL Week 7.

Usage:
    python3 nfl_parlay_builder.py
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from itertools import combinations
import json

class NFLParlayBuilder:
    """
    NFL Parlay Builder using SportsData.io projections
    """
    
    def __init__(self, data_file=None):
        self.data_file = data_file or '6_OPTIMIZATION/nfl_week7_CASH_SPORTSDATA.csv'
        self.df = None
        self.prop_bets = []
        self.parlays = []
        
    def load_data(self):
        """Load NFL data from sportsdata.io"""
        try:
            print(f"📊 Loading NFL data from: {self.data_file}")
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.df)} players")
            print(f"📋 Teams: {sorted(self.df['Team'].unique())}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_prop_bets(self):
        """Create prop bets from player projections"""
        print("\n🎯 Creating NFL Prop Bets...")
        
        self.prop_bets = []
        
        for _, player in self.df.iterrows():
            name = player['Name']
            team = player['Team']
            position = player['Position']
            opponent = player['Opponent']
            
            # QB Props
            if position == 'QB':
                # Passing Yards
                if 'PassingYards' in player and not pd.isna(player['PassingYards']):
                    yards = player['PassingYards']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Passing Yards',
                        'line': round(yards),
                        'projection': yards,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Passing TDs
                if 'PassingTouchdowns' in player and not pd.isna(player['PassingTouchdowns']):
                    tds = player['PassingTouchdowns']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Passing TDs',
                        'line': round(tds, 1),
                        'projection': tds,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Rushing Yards
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    rush_yards = player['RushingYards']
                    if rush_yards > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Rushing Yards',
                            'line': round(rush_yards),
                            'projection': rush_yards,
                            'position': position,
                            'type': 'over_under'
                        })
            
            # RB Props
            elif position == 'RB':
                # Rushing Yards
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    yards = player['RushingYards']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Rushing Yards',
                        'line': round(yards),
                        'projection': yards,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Rushing TDs
                if 'RushingTouchdowns' in player and not pd.isna(player['RushingTouchdowns']):
                    tds = player['RushingTouchdowns']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Rushing TDs',
                        'line': round(tds, 1),
                        'projection': tds,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    rec_yards = player['ReceivingYards']
                    if rec_yards > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receiving Yards',
                            'line': round(rec_yards),
                            'projection': rec_yards,
                            'position': position,
                            'type': 'over_under'
                        })
            
            # WR Props
            elif position == 'WR':
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Receiving Yards',
                        'line': round(yards),
                        'projection': yards,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Receiving TDs
                if 'ReceivingTouchdowns' in player and not pd.isna(player['ReceivingTouchdowns']):
                    tds = player['ReceivingTouchdowns']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Receiving TDs',
                        'line': round(tds, 1),
                        'projection': tds,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Receptions
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Receptions',
                        'line': round(rec, 1),
                        'projection': rec,
                        'position': position,
                        'type': 'over_under'
                    })
            
            # TE Props
            elif position == 'TE':
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Receiving Yards',
                        'line': round(yards),
                        'projection': yards,
                        'position': position,
                        'type': 'over_under'
                    })
                
                # Receptions
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    self.prop_bets.append({
                        'player': name,
                        'team': team,
                        'opponent': opponent,
                        'prop': 'Receptions',
                        'line': round(rec, 1),
                        'projection': rec,
                        'position': position,
                        'type': 'over_under'
                    })
        
        print(f"✅ Created {len(self.prop_bets)} prop bets")
        return self.prop_bets
    
    def calculate_probability(self, projection, line, prop_type='over_under'):
        """Calculate probability of hitting over/under based on projection"""
        if prop_type == 'over_under':
            # Simple probability calculation based on projection vs line
            # This is a simplified model - in reality you'd use more sophisticated methods
            diff = projection - line
            
            # Avoid division by zero
            if line == 0:
                return 0.5
            
            if diff > 0:
                # Projection is above line - higher chance of over
                prob = min(0.85, 0.5 + (diff / abs(line)) * 0.3)
            else:
                # Projection is below line - higher chance of under
                prob = max(0.15, 0.5 - (abs(diff) / abs(line)) * 0.3)
            
            return prob
        return 0.5
    
    def build_parlays(self, max_legs=4, min_probability=0.15):
        """Build optimal parlay combinations"""
        print(f"\n🎲 Building Parlays (Max {max_legs} legs)...")
        
        # Filter high-probability props
        eligible_props = []
        for prop in self.prop_bets:
            prob = self.calculate_probability(prop['projection'], prop['line'])
            if prob >= min_probability:
                prop['probability'] = prob
                prop['edge'] = prob - 0.5  # Simple edge calculation
                eligible_props.append(prop)
        
        print(f"📊 {len(eligible_props)} eligible props (prob >= {min_probability})")
        
        # Sort by probability * edge
        eligible_props.sort(key=lambda x: x['probability'] * (1 + x['edge']), reverse=True)
        
        self.parlays = []
        
        # Generate parlays of different sizes
        for leg_count in range(2, min(max_legs + 1, len(eligible_props) + 1)):
            for parlay_combo in combinations(eligible_props, leg_count):
                parlay_legs = list(parlay_combo)
                
                # Calculate combined probability (assuming independence)
                combined_prob = 1.0
                for leg in parlay_legs:
                    combined_prob *= leg['probability']
                
                # Calculate payout multiplier (simplified)
                payout_multiplier = 1.0
                for leg in parlay_legs:
                    implied_odds = 1 / leg['probability'] if leg['probability'] > 0 else 1
                    payout_multiplier *= implied_odds
                
                # Calculate expected value
                expected_value = combined_prob * payout_multiplier
                
                # Only include profitable parlays
                if expected_value > 1.0:
                    self.parlays.append({
                        'legs': parlay_legs,
                        'leg_count': leg_count,
                        'combined_probability': combined_prob,
                        'payout_multiplier': payout_multiplier,
                        'expected_value': expected_value,
                        'risk_reward_ratio': expected_value / leg_count
                    })
        
        # Sort by expected value
        self.parlays.sort(key=lambda x: x['expected_value'], reverse=True)
        print(f"✅ Generated {len(self.parlays)} profitable parlays")
        
        return self.parlays
    
    def display_top_parlays(self, top_n=10):
        """Display top parlay recommendations"""
        print(f"\n🏆 TOP {top_n} PARLAY RECOMMENDATIONS")
        print("=" * 80)
        
        for i, parlay in enumerate(self.parlays[:top_n], 1):
            print(f"\n{i}. {parlay['leg_count']}-Leg Parlay")
            print(f"   Combined Probability: {parlay['combined_probability']:.1%}")
            print(f"   Payout: {parlay['payout_multiplier']:.1f}x")
            print(f"   Expected Value: {parlay['expected_value']:.2f}")
            print(f"   Risk/Reward: {parlay['risk_reward_ratio']:.2f}")
            print("   Legs:")
            
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"     {j}. {leg['player']} ({leg['team']}) - {leg['prop']} {leg['line']} "
                      f"(Proj: {leg['projection']:.1f}, Prob: {leg['probability']:.1%})")
    
    def save_parlays(self, filename=None):
        """Save parlay recommendations to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"nfl_parlays_week7_{timestamp}.csv"
        
        parlay_data = []
        for i, parlay in enumerate(self.parlays, 1):
            parlay_row = {
                'parlay_id': f"Parlay_{i}",
                'leg_count': parlay['leg_count'],
                'combined_probability': parlay['combined_probability'],
                'payout_multiplier': parlay['payout_multiplier'],
                'expected_value': parlay['expected_value'],
                'risk_reward_ratio': parlay['risk_reward_ratio']
            }
            
            # Add individual leg details
            for j, leg in enumerate(parlay['legs'], 1):
                parlay_row[f'leg_{j}_player'] = leg['player']
                parlay_row[f'leg_{j}_team'] = leg['team']
                parlay_row[f'leg_{j}_prop'] = leg['prop']
                parlay_row[f'leg_{j}_line'] = leg['line']
                parlay_row[f'leg_{j}_projection'] = leg['projection']
                parlay_row[f'leg_{j}_probability'] = leg['probability']
            
            parlay_data.append(parlay_row)
        
        parlay_df = pd.DataFrame(parlay_data)
        parlay_df.to_csv(filename, index=False)
        print(f"💾 Parlays saved to: {filename}")
        
        return filename
    
    def get_team_analysis(self):
        """Analyze teams for stacking opportunities"""
        print(f"\n🏈 TEAM ANALYSIS")
        print("=" * 50)
        
        # Group by team
        team_analysis = {}
        for prop in self.prop_bets:
            team = prop['team']
            if team not in team_analysis:
                team_analysis[team] = []
            team_analysis[team].append(prop)
        
        # Sort teams by number of props
        sorted_teams = sorted(team_analysis.items(), key=lambda x: len(x[1]), reverse=True)
        
        for team, props in sorted_teams[:5]:  # Top 5 teams
            print(f"\n{team} - {len(props)} props:")
            for prop in props[:3]:  # Top 3 props per team
                prob = self.calculate_probability(prop['projection'], prop['line'])
                print(f"  • {prop['player']} - {prop['prop']} {prop['line']} "
                      f"(Proj: {prop['projection']:.1f}, Prob: {prob:.1%})")

def main():
    """Main function"""
    print("🏈 NFL PARLAY BUILDER - SPORTSDATA.IO")
    print("=" * 50)
    
    # Initialize builder
    builder = NFLParlayBuilder()
    
    # Load data
    if not builder.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create prop bets
    builder.create_prop_bets()
    
    # Build parlays
    builder.build_parlays(max_legs=4, min_probability=0.15)
    
    # Display results
    builder.display_top_parlays(10)
    
    # Team analysis
    builder.get_team_analysis()
    
    # Save results
    builder.save_parlays()
    
    print(f"\n✅ NFL Parlay Builder Complete!")
    print(f"📊 Total Props: {len(builder.prop_bets)}")
    print(f"🎲 Total Parlays: {len(builder.parlays)}")

if __name__ == "__main__":
    main()
