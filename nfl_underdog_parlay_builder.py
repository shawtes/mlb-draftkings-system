#!/usr/bin/env python3
"""
NFL Underdog Parlay Builder using SportsData.io Data
===================================================

This script creates NFL prop bet parlays using the latest sportsdata.io projections.
It's specifically designed for NFL data and converts from the MLB version.

Usage:
    python3 nfl_underdog_parlay_builder.py
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from itertools import combinations
import json

class NFLUnderdogParlayBuilder:
    """
    NFL Underdog Parlay Builder using SportsData.io projections
    """
    
    def __init__(self, data_file=None):
        self.data_file = data_file or '6_OPTIMIZATION/nfl_week7_CASH_SPORTSDATA.csv'
        self.df = None
        self.prop_bets = []
        self.parlays = []
        self.power_plays = []
        self.insurance_plays = []
        
        # NFL-specific multipliers (different from MLB)
        self.power_play_multipliers = {
            '3x': 3.0,
            '6x': 6.0, 
            '10x': 10.0,
            '20x': 20.0,
            '40x': 40.0
        }
    
    def load_data(self):
        """Load NFL data from sportsdata.io"""
        try:
            print(f"🏈 Loading NFL data from: {self.data_file}")
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.df)} players")
            print(f"📋 Teams: {sorted(self.df['Team'].unique())}")
            print(f"📊 Positions: {self.df['Position'].value_counts().to_dict()}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_nfl_prop_bets(self):
        """Create NFL-specific prop bets from player projections"""
        print("\n🎯 Creating NFL Prop Bets...")
        
        self.prop_bets = []
        
        for _, player in self.df.iterrows():
            name = player['Name']
            team = player['Team']
            position = player['Position']
            opponent = player['Opponent']
            
            # Skip if missing essential data
            if pd.isna(name) or pd.isna(team) or pd.isna(position):
                continue
            
            # QB Props
            if position == 'QB':
                # Passing Yards
                if 'PassingYards' in player and not pd.isna(player['PassingYards']):
                    yards = player['PassingYards']
                    if yards > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Passing Yards',
                            'line': round(yards),
                            'projection': yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Passing Yards', yards)
                        })
                
                # Passing TDs
                if 'PassingTouchdowns' in player and not pd.isna(player['PassingTouchdowns']):
                    tds = player['PassingTouchdowns']
                    if tds > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Passing TDs',
                            'line': round(tds, 1),
                            'projection': tds,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Passing TDs', tds)
                        })
                
                # Rushing Yards
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    rush_yards = player['RushingYards']
                    if rush_yards > 5:  # Only include meaningful rushing props
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Rushing Yards',
                            'line': round(rush_yards),
                            'projection': rush_yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Rushing Yards', rush_yards)
                        })
            
            # RB Props
            elif position == 'RB':
                # Rushing Yards
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    yards = player['RushingYards']
                    if yards > 10:  # Only meaningful rushing props
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Rushing Yards',
                            'line': round(yards),
                            'projection': yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Rushing Yards', yards)
                        })
                
                # Rushing TDs
                if 'RushingTouchdowns' in player and not pd.isna(player['RushingTouchdowns']):
                    tds = player['RushingTouchdowns']
                    if tds > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Rushing TDs',
                            'line': round(tds, 1),
                            'projection': tds,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Rushing TDs', tds)
                        })
                
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    rec_yards = player['ReceivingYards']
                    if rec_yards > 5:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receiving Yards',
                            'line': round(rec_yards),
                            'projection': rec_yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receiving Yards', rec_yards)
                        })
            
            # WR Props
            elif position == 'WR':
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    if yards > 10:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receiving Yards',
                            'line': round(yards),
                            'projection': yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receiving Yards', yards)
                        })
                
                # Receiving TDs
                if 'ReceivingTouchdowns' in player and not pd.isna(player['ReceivingTouchdowns']):
                    tds = player['ReceivingTouchdowns']
                    if tds > 0:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receiving TDs',
                            'line': round(tds, 1),
                            'projection': tds,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receiving TDs', tds)
                        })
                
                # Receptions
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    if rec > 2:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receptions',
                            'line': round(rec, 1),
                            'projection': rec,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receptions', rec)
                        })
            
            # TE Props
            elif position == 'TE':
                # Receiving Yards
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    if yards > 5:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receiving Yards',
                            'line': round(yards),
                            'projection': yards,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receiving Yards', yards)
                        })
                
                # Receptions
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    if rec > 1:
                        self.prop_bets.append({
                            'player': name,
                            'team': team,
                            'opponent': opponent,
                            'prop': 'Receptions',
                            'line': round(rec, 1),
                            'projection': rec,
                            'position': position,
                            'type': 'over_under',
                            'multiplier': self._get_multiplier_for_prop('Receptions', rec)
                        })
        
        print(f"✅ Created {len(self.prop_bets)} NFL prop bets")
        return self.prop_bets
    
    def _get_multiplier_for_prop(self, prop_type, value):
        """Get appropriate multiplier for NFL prop type"""
        if 'TDs' in prop_type:
            if value >= 2.0:
                return '20x'
            elif value >= 1.5:
                return '10x'
            elif value >= 1.0:
                return '6x'
            else:
                return '3x'
        elif 'Yards' in prop_type:
            if value >= 100:
                return '20x'
            elif value >= 75:
                return '10x'
            elif value >= 50:
                return '6x'
            else:
                return '3x'
        elif 'Receptions' in prop_type:
            if value >= 8:
                return '20x'
            elif value >= 6:
                return '10x'
            elif value >= 4:
                return '6x'
            else:
                return '3x'
        else:
            return '3x'
    
    def calculate_nfl_probability(self, projection, line, prop_type='over_under'):
        """Calculate probability for NFL props based on projection vs line"""
        if prop_type == 'over_under':
            diff = projection - line
            
            # Avoid division by zero
            if line == 0:
                return 0.5
            
            # NFL-specific probability calculation
            if 'TDs' in prop_type:
                # TD props are more volatile
                if diff > 0:
                    prob = min(0.80, 0.5 + (diff / abs(line)) * 0.4)
                else:
                    prob = max(0.20, 0.5 - (abs(diff) / abs(line)) * 0.4)
            elif 'Yards' in prop_type:
                # Yard props are more predictable
                if diff > 0:
                    prob = min(0.85, 0.5 + (diff / abs(line)) * 0.3)
                else:
                    prob = max(0.15, 0.5 - (abs(diff) / abs(line)) * 0.3)
            elif 'Receptions' in prop_type:
                # Reception props are moderately predictable
                if diff > 0:
                    prob = min(0.82, 0.5 + (diff / abs(line)) * 0.35)
                else:
                    prob = max(0.18, 0.5 - (abs(diff) / abs(line)) * 0.35)
            else:
                # Default calculation
                if diff > 0:
                    prob = min(0.80, 0.5 + (diff / abs(line)) * 0.3)
                else:
                    prob = max(0.20, 0.5 - (abs(diff) / abs(line)) * 0.3)
            
            return prob
        return 0.5
    
    def create_power_plays(self, min_probability=0.60):
        """Create NFL power plays (high-probability individual bets)"""
        print(f"\n⚡ Creating NFL Power Plays (Min Prob: {min_probability:.0%})...")
        
        self.power_plays = []
        
        for prop in self.prop_bets:
            prob = self.calculate_nfl_probability(prop['projection'], prop['line'])
            if prob >= min_probability:
                expected_return = prob * prop['multiplier']
                kelly_fraction = (prob * prop['multiplier'] - 1) / (prop['multiplier'] - 1)
                
                self.power_plays.append({
                    'player': prop['player'],
                    'team': prop['team'],
                    'opponent': prop['opponent'],
                    'prop': prop['prop'],
                    'line': prop['line'],
                    'projection': prop['projection'],
                    'probability': prob,
                    'multiplier': prop['multiplier'],
                    'expected_return': expected_return,
                    'kelly_fraction': max(0, kelly_fraction),
                    'confidence': min(0.95, prob * 1.2)  # Boost confidence for NFL
                })
        
        # Sort by expected return
        self.power_plays.sort(key=lambda x: x['expected_return'], reverse=True)
        print(f"✅ Created {len(self.power_plays)} NFL power plays")
        
        return self.power_plays
    
    def create_insurance_plays(self, power_plays):
        """Create insurance plays to hedge power plays"""
        print(f"\n🛡️ Creating NFL Insurance Plays...")
        
        self.insurance_plays = []
        
        for power_play in power_plays[:5]:  # Top 5 power plays
            # Create under bet for each power play
            insurance_prop = {
                'player': power_play['player'],
                'team': power_play['team'],
                'opponent': power_play['opponent'],
                'prop': power_play['prop'],
                'line': power_play['line'],
                'projection': power_play['projection'],
                'probability': 1 - power_play['probability'],  # Opposite probability
                'multiplier': '3x',  # Conservative multiplier
                'expected_return': (1 - power_play['probability']) * 3,
                'kelly_fraction': 0.05,  # Small hedge
                'confidence': 0.6,
                'type': 'insurance'
            }
            
            self.insurance_plays.append(insurance_prop)
        
        print(f"✅ Created {len(self.insurance_plays)} insurance plays")
        return self.insurance_plays
    
    def build_nfl_parlays(self, max_legs=4, min_probability=0.20):
        """Build optimal NFL parlay combinations"""
        print(f"\n🎲 Building NFL Parlays (Max {max_legs} legs)...")
        
        # Filter high-probability props
        eligible_props = []
        for prop in self.prop_bets:
            prob = self.calculate_nfl_probability(prop['projection'], prop['line'])
            if prob >= min_probability:
                prop['probability'] = prob
                prop['edge'] = prob - 0.5
                eligible_props.append(prop)
        
        print(f"📊 {len(eligible_props)} eligible props (prob >= {min_probability:.0%})")
        
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
        print(f"✅ Generated {len(self.parlays)} profitable NFL parlays")
        
        return self.parlays
    
    def display_nfl_analysis(self):
        """Display comprehensive NFL analysis"""
        print(f"\n🏈 NFL UNDERDOG PARLAY ANALYSIS")
        print("=" * 80)
        
        # Power Plays
        print(f"\n⚡ TOP 5 POWER PLAYS:")
        print("-" * 50)
        for i, play in enumerate(self.power_plays[:5], 1):
            print(f"{i}. {play['player']} ({play['team']}) - {play['prop']} {play['line']}")
            print(f"   Projection: {play['projection']:.1f} | Prob: {play['probability']:.1%}")
            print(f"   Multiplier: {play['multiplier']} | Expected Return: {play['expected_return']:.2f}")
            print(f"   Kelly Fraction: {play['kelly_fraction']:.1%}")
        
        # Top Parlays
        print(f"\n🎲 TOP 5 PARLAYS:")
        print("-" * 50)
        for i, parlay in enumerate(self.parlays[:5], 1):
            print(f"{i}. {parlay['leg_count']}-Leg Parlay")
            print(f"   Combined Probability: {parlay['combined_probability']:.1%}")
            print(f"   Payout: {parlay['payout_multiplier']:.1f}x")
            print(f"   Expected Value: {parlay['expected_value']:.2f}")
            print("   Legs:")
            for j, leg in enumerate(parlay['legs'], 1):
                print(f"     {j}. {leg['player']} ({leg['team']}) - {leg['prop']} {leg['line']} "
                      f"(Prob: {leg['probability']:.1%})")
        
        # Team Analysis
        print(f"\n🏈 TEAM ANALYSIS:")
        print("-" * 50)
        team_analysis = {}
        for prop in self.prop_bets:
            team = prop['team']
            if team not in team_analysis:
                team_analysis[team] = []
            team_analysis[team].append(prop)
        
        sorted_teams = sorted(team_analysis.items(), key=lambda x: len(x[1]), reverse=True)
        for team, props in sorted_teams[:5]:
            print(f"\n{team} - {len(props)} props:")
            for prop in props[:3]:
                prob = self.calculate_nfl_probability(prop['projection'], prop['line'])
                print(f"  • {prop['player']} - {prop['prop']} {prop['line']} "
                      f"(Proj: {prop['projection']:.1f}, Prob: {prob:.1%})")
    
    def save_nfl_results(self, filename=None):
        """Save NFL parlay results to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"nfl_underdog_parlays_week7_{timestamp}.csv"
        
        # Save power plays
        power_plays_df = pd.DataFrame(self.power_plays)
        power_plays_file = filename.replace('.csv', '_power_plays.csv')
        power_plays_df.to_csv(power_plays_file, index=False)
        print(f"💾 Power plays saved to: {power_plays_file}")
        
        # Save parlays
        parlay_data = []
        for i, parlay in enumerate(self.parlays, 1):
            parlay_row = {
                'parlay_id': f"NFL_Parlay_{i}",
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
        print(f"💾 NFL parlays saved to: {filename}")
        
        return filename

def main():
    """Main function"""
    print("🏈 NFL UNDERDOG PARLAY BUILDER - SPORTSDATA.IO")
    print("=" * 60)
    
    # Initialize builder
    builder = NFLUnderdogParlayBuilder()
    
    # Load data
    if not builder.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Create prop bets
    builder.create_nfl_prop_bets()
    
    # Create power plays
    builder.create_power_plays(min_probability=0.60)
    
    # Create insurance plays
    builder.create_insurance_plays(builder.power_plays)
    
    # Build parlays
    builder.build_nfl_parlays(max_legs=4, min_probability=0.20)
    
    # Display analysis
    builder.display_nfl_analysis()
    
    # Save results
    builder.save_nfl_results()
    
    print(f"\n✅ NFL Underdog Parlay Builder Complete!")
    print(f"📊 Total Props: {len(builder.prop_bets)}")
    print(f"⚡ Power Plays: {len(builder.power_plays)}")
    print(f"🛡️ Insurance Plays: {len(builder.insurance_plays)}")
    print(f"🎲 Total Parlays: {len(builder.parlays)}")

if __name__ == "__main__":
    main()



