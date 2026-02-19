#!/usr/bin/env python3
"""
NBA Parlay Model Trainer
Trains and evaluates the NBA parlay generator using historical data
"""

import pandas as pd
import numpy as np
import os
from nba_parlay_generator import NBAAdvancedParlayGenerator, NBAParlay

class NBAModelTrainer:
    """Trains and evaluates the NBA parlay model"""
    
    def __init__(self, training_data_path: str):
        """Initialize trainer with training data"""
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(f"Training data not found: {training_data_path}")
        
        self.data = pd.read_csv(training_data_path)
        print(f"✅ Loaded training data: {len(self.data)} records")
        
        # Initialize generator
        self.generator = NBAAdvancedParlayGenerator(self.data)
    
    def train_and_evaluate(self, num_test_parlays: int = 100, test_season: str = '2025'):
        """
        Train and evaluate the model
        
        Args:
            num_test_parlays: Number of parlays to generate for testing
            test_season: Season to test on
        """
        print("\n🏀 NBA Parlay Model Training & Evaluation")
        print("=" * 70)
        
        # Split data into training and testing (use Season column)
        season_col = 'Season' if 'Season' in self.data.columns else 'season'
        if season_col not in self.data.columns:
            # If no season column, split by game_date
            self.data['year'] = pd.to_datetime(self.data['game_date'], errors='coerce').dt.year
            season_col = 'year'
        
        train_data = self.data[self.data[season_col] != test_season].copy()
        test_data = self.data[self.data[season_col] == test_season].copy()
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {len(train_data)} records")
        print(f"   Testing: {len(test_data)} records")
        
        if len(test_data) == 0:
            print("⚠️  No test data available. Using all data.")
            test_data = self.data.copy()
        
        # Generate test parlays
        print(f"\n🎲 Generating {num_test_parlays} test parlays...")
        
        test_parlays = []
        for i in range(num_test_parlays):
            parlay = self.generator.generate_parlay(max_legs=4)
            if parlay.legs:
                test_parlays.append(parlay)
        
        print(f"✅ Generated {len(test_parlays)} valid parlays")
        
        # Evaluate against actual results
        results = self._evaluate_parlays(test_parlays, test_data)
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _evaluate_parlays(self, parlays: List[NBAParlay], test_data: pd.DataFrame) -> Dict:
        """Evaluate parlays against actual results"""
        print("\n📊 Evaluating parlays against actual results...")
        
        winning_parlays = 0
        winning_legs = 0
        total_legs = 0
        
        prop_results = {
            'points': {'won': 0, 'total': 0},
            'rebounds': {'won': 0, 'total': 0},
            'assists': {'won': 0, 'total': 0},
            'steals': {'won': 0, 'total': 0},
            'blocks': {'won': 0, 'total': 0},
            'three_pointers': {'won': 0, 'total': 0}
        }
        
        for parlay in parlays:
            parlay_won = True
            legs_won = 0
            
            for leg in parlay.legs:
                # Find matching actual result
                matching_data = test_data[
                    (test_data['player_name'] == leg.player_name) &
                    (test_data['team'] == leg.team)
                ]
                
                if len(matching_data) > 0:
                    player_data = matching_data.iloc[0]
                    
                    # Get actual stat
                    actual_col = f'actual_{leg.prop_type}'
                    if actual_col in player_data:
                        actual_value = player_data[actual_col]
                        
                        # Check if leg hit
                        if leg.bet_type == 'OVER':
                            leg_hit = actual_value > leg.line
                        else:  # UNDER
                            leg_hit = actual_value < leg.line
                        
                        legs_won += int(leg_hit)
                        total_legs += 1
                        
                        # Track by prop type
                        if leg.prop_type in prop_results:
                            prop_results[leg.prop_type]['total'] += 1
                            prop_results[leg.prop_type]['won'] += int(leg_hit)
                        
                        if not leg_hit:
                            parlay_won = False
                else:
                    parlay_won = False
            
            winning_legs += legs_won
            if parlay_won:
                winning_parlays += 1
        
        return {
            'total_parlays': len(parlays),
            'winning_parlays': winning_parlays,
            'win_rate': winning_parlays / len(parlays) if len(parlays) > 0 else 0,
            'total_legs': total_legs,
            'winning_legs': winning_legs,
            'leg_hit_rate': winning_legs / total_legs if total_legs > 0 else 0,
            'prop_results': prop_results
        }
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nOverall Performance:")
        print(f"   Total Parlays: {results['total_parlays']}")
        print(f"   Winning Parlays: {results['winning_parlays']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Leg Hit Rate: {results['leg_hit_rate']:.1%}")
        
        print(f"\nPerformance by Prop Type:")
        for prop_type, stats in results['prop_results'].items():
            if stats['total'] > 0:
                hit_rate = stats['won'] / stats['total']
                print(f"   {prop_type.title()}: {stats['won']}/{stats['total']} ({hit_rate:.1%})")
        
        # Compare to NFL model
        print(f"\n🎯 Comparison to NFL Model:")
        print(f"   NFL Model Win Rate: 86.7%")
        print(f"   NBA Model Win Rate: {results['win_rate']:.1%}")
        
        if results['win_rate'] >= 0.80:
            print(f"   ✅ Excellent performance!")
        elif results['win_rate'] >= 0.70:
            print(f"   ✅ Good performance")
        elif results['win_rate'] >= 0.60:
            print(f"   ⚠️  Moderate performance - needs tuning")
        else:
            print(f"   ❌ Poor performance - model needs adjustment")
    
    def generate_sample_parlays(self, num_parlays: int = 10, max_legs: int = 4):
        """Generate sample parlays for review"""
        print(f"\n🎲 Generating {num_parlays} sample parlays...")
        
        parlays = []
        for i in range(num_parlays):
            parlay = self.generator.generate_parlay(max_legs=max_legs)
            if parlay.legs:
                parlays.append(parlay)
        
        print(f"\n📋 Sample Parlays:")
        print("=" * 70)
        
        for idx, parlay in enumerate(parlays, 1):
            print(f"\nParlay #{idx} ({len(parlay.legs)} legs)")
            print(f"Hit Rate: {parlay.combined_hit_rate:.1%} | Odds: +{parlay.estimated_odds:.0f}")
            print("-" * 70)
            
            for leg_idx, leg in enumerate(parlay.legs, 1):
                prop_display = leg.prop_type.replace('_', ' ').title()
                print(f"  {leg_idx}. {leg.player_name} ({leg.team})")
                print(f"     {prop_display} {leg.bet_type} {leg.line:.1f} ({leg.hit_rate:.0%})")
        
        return parlays

def main():
    """Main training function"""
    print("🏀 NBA Parlay Model Trainer")
    print("=" * 70)
    
    # Check if training data exists
    training_file = 'nba_training_data.csv'
    
    if not os.path.exists(training_file):
        print(f"❌ Training data not found: {training_file}")
        print("   Please run nba_data_collector.py first")
        return
    
    # Create trainer
    trainer = NBAModelTrainer(training_file)
    
    # Train and evaluate
    results = trainer.train_and_evaluate(num_test_parlays=100, test_season='2025')
    
    # Generate sample parlays
    trainer.generate_sample_parlays(num_parlays=10, max_legs=4)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()

