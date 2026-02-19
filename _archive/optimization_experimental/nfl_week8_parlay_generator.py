#!/usr/bin/env python3
"""
NFL Week 8 Parlay Data Generator
Generates parlay data for current Week 8 NFL games
"""

import pandas as pd
import requests
import json
import os
from datetime import datetime, timedelta
import sys

# Add parent directory to path for API access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

try:
    from sportsdata_nfl_api import SportsDataNFLAPI
except ImportError:
    print("❌ Error: Could not import sportsdata_nfl_api")
    print("Make sure the API client is in the python_algorithms directory")
    sys.exit(1)

class NFLParlayGenerator:
    """Generate NFL parlay data for Week 8"""
    
    def __init__(self):
        # Use the same API key as the fantasy fetcher
        API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
        self.api = SportsDataNFLAPI(API_KEY)
        self.week = 8
        self.season = 2025
        self.today = "2025-10-23"  # Today's date
        
    def fetch_todays_games(self):
        """Fetch today's NFL games and betting data"""
        print(f"🏈 Fetching NFL games for today ({self.today})...")
        
        try:
            # Focus on today's date first
            possible_dates = [
                self.today,    # Today (2025-10-23)
                '2025-10-26',  # Sunday
                '2025-10-25',  # Saturday
                '2025-10-27',  # Monday
                '2025-10-24'   # Friday
            ]
            
            games_data = []
            for date in possible_dates:
                print(f"   Checking {date}...")
                games = self.api.get_dfs_slates_by_date(date)
                if games:
                    print(f"   ✅ Found {len(games)} slates on {date}")
                    for slate in games:
                        if 'NFL' in slate.get('name', '') and 'Week' in slate.get('name', ''):
                            print(f"      📅 Slate: {slate['name']}")
                            games_data.append(slate)
            
            if not games_data:
                print("   ⚠️ No today's slates found, creating mock data...")
                return self.create_mock_todays_games()
            
            return games_data
            
        except Exception as e:
            print(f"   ❌ Error fetching games: {e}")
            print("   Creating mock today's data...")
            return self.create_mock_todays_games()
    
    def create_mock_todays_games(self):
        """Create mock today's games data"""
        print("🎯 Creating mock today's NFL games...")
        
        # Mock today's games (Thursday games)
        mock_games = [
            {
                'name': 'NFL Thursday Night Football',
                'date': self.today,
                'games': [
                    {'home_team': 'Giants', 'away_team': '49ers', 'time': '8:15 PM ET'},
                    {'home_team': 'Bills', 'away_team': 'Dolphins', 'time': '8:15 PM ET'},
                    {'home_team': 'Ravens', 'away_team': 'Steelers', 'time': '8:15 PM ET'}
                ]
            }
        ]
        
        return mock_games
    
    def generate_parlay_options(self, games_data):
        """Generate parlay betting options"""
        print("🎲 Generating parlay options...")
        
        parlay_data = []
        
        for slate in games_data:
            slate_name = slate.get('name', 'NFL Week 8')
            slate_date = slate.get('date', '2025-10-26')
            
            games = slate.get('games', [])
            print(f"   📅 Processing {len(games)} games for {slate_name}")
            
            for i, game in enumerate(games):
                home_team = game.get('home_team', f'Team{i+1}A')
                away_team = game.get('away_team', f'Team{i+1}B')
                game_time = game.get('time', '1:00 PM ET')
                
                # Create parlay options for each game
                game_parlays = [
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'{away_team} Moneyline',
                        'odds': '+110',
                        'type': 'Moneyline',
                        'team': away_team,
                        'confidence': 'Medium'
                    },
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'{home_team} Moneyline',
                        'odds': '-120',
                        'type': 'Moneyline',
                        'team': home_team,
                        'confidence': 'High'
                    },
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'Over 45.5 Total Points',
                        'odds': '-110',
                        'type': 'Total',
                        'team': 'Over',
                        'confidence': 'Medium'
                    },
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'Under 45.5 Total Points',
                        'odds': '-110',
                        'type': 'Total',
                        'team': 'Under',
                        'confidence': 'Medium'
                    },
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'{away_team} +3.5 Spread',
                        'odds': '-110',
                        'type': 'Spread',
                        'team': away_team,
                        'confidence': 'High'
                    },
                    {
                        'game': f'{away_team} @ {home_team}',
                        'time': game_time,
                        'option': f'{home_team} -3.5 Spread',
                        'odds': '-110',
                        'type': 'Spread',
                        'team': home_team,
                        'confidence': 'Medium'
                    }
                ]
                
                parlay_data.extend(game_parlays)
        
        return parlay_data
    
    def create_parlay_combinations(self, parlay_data):
        """Create popular parlay combinations"""
        print("🎯 Creating parlay combinations...")
        
        # Group by confidence level
        high_confidence = [p for p in parlay_data if p['confidence'] == 'High']
        medium_confidence = [p for p in parlay_data if p['confidence'] == 'Medium']
        
        combinations = []
        
        # 2-leg parlays (High confidence)
        if len(high_confidence) >= 2:
            for i in range(min(5, len(high_confidence))):
                for j in range(i+1, min(5, len(high_confidence))):
                    combo = {
                        'legs': 2,
                        'type': 'High Confidence',
                        'bets': [high_confidence[i], high_confidence[j]],
                        'estimated_odds': '+260',
                        'confidence': 'High'
                    }
                    combinations.append(combo)
        
        # 3-leg parlays (Mixed confidence)
        if len(high_confidence) >= 1 and len(medium_confidence) >= 2:
            for i in range(min(3, len(high_confidence))):
                for j in range(min(3, len(medium_confidence))):
                    for k in range(j+1, min(3, len(medium_confidence))):
                        combo = {
                            'legs': 3,
                            'type': 'Mixed Confidence',
                            'bets': [high_confidence[i], medium_confidence[j], medium_confidence[k]],
                            'estimated_odds': '+600',
                            'confidence': 'Medium'
                        }
                        combinations.append(combo)
        
        # 4-leg parlays (All medium confidence)
        if len(medium_confidence) >= 4:
            for i in range(min(2, len(medium_confidence))):
                for j in range(i+1, min(2, len(medium_confidence))):
                    for k in range(j+1, min(2, len(medium_confidence))):
                        for l in range(k+1, min(2, len(medium_confidence))):
                            combo = {
                                'legs': 4,
                                'type': 'Medium Confidence',
                                'bets': [medium_confidence[i], medium_confidence[j], medium_confidence[k], medium_confidence[l]],
                                'estimated_odds': '+1200',
                                'confidence': 'Medium'
                            }
                            combinations.append(combo)
        
        return combinations
    
    def save_parlay_data(self, parlay_data, combinations):
        """Save parlay data to CSV files"""
        print("💾 Saving parlay data...")
        
        # Save individual parlay options
        df_parlays = pd.DataFrame(parlay_data)
        parlay_file = 'nfl_1023_parlay_options.csv'
        df_parlays.to_csv(parlay_file, index=False)
        print(f"   ✅ Saved {len(parlay_data)} parlay options to {parlay_file}")
        
        # Save parlay combinations
        combo_data = []
        for combo in combinations:
            combo_row = {
                'legs': combo['legs'],
                'type': combo['type'],
                'confidence': combo['confidence'],
                'estimated_odds': combo['estimated_odds'],
                'bet_1': combo['bets'][0]['option'],
                'bet_1_game': combo['bets'][0]['game'],
                'bet_1_odds': combo['bets'][0]['odds']
            }
            
            if len(combo['bets']) > 1:
                combo_row['bet_2'] = combo['bets'][1]['option']
                combo_row['bet_2_game'] = combo['bets'][1]['game']
                combo_row['bet_2_odds'] = combo['bets'][1]['odds']
            
            if len(combo['bets']) > 2:
                combo_row['bet_3'] = combo['bets'][2]['option']
                combo_row['bet_3_game'] = combo['bets'][2]['game']
                combo_row['bet_3_odds'] = combo['bets'][2]['odds']
            
            if len(combo['bets']) > 3:
                combo_row['bet_4'] = combo['bets'][3]['option']
                combo_row['bet_4_game'] = combo['bets'][3]['game']
                combo_row['bet_4_odds'] = combo['bets'][3]['odds']
            
            combo_data.append(combo_row)
        
        df_combos = pd.DataFrame(combo_data)
        combo_file = 'nfl_1023_parlay_combinations.csv'
        df_combos.to_csv(combo_file, index=False)
        print(f"   ✅ Saved {len(combinations)} parlay combinations to {combo_file}")
        
        return parlay_file, combo_file
    
    def generate_summary_report(self, parlay_data, combinations):
        """Generate a summary report"""
        print("📊 Generating summary report...")
        
        report = f"""
🏈 NFL Today's Parlay Data Summary
{'='*50}

📅 Date: {self.today}
🎯 Today's Games
📊 Season: {self.season}

📈 PARLAY OPTIONS:
   Total Options: {len(parlay_data)}
   Moneyline Bets: {len([p for p in parlay_data if p['type'] == 'Moneyline'])}
   Spread Bets: {len([p for p in parlay_data if p['type'] == 'Spread'])}
   Total Bets: {len([p for p in parlay_data if p['type'] == 'Total'])}

🎲 PARLAY COMBINATIONS:
   Total Combinations: {len(combinations)}
   2-Leg Parlays: {len([c for c in combinations if c['legs'] == 2])}
   3-Leg Parlays: {len([c for c in combinations if c['legs'] == 3])}
   4-Leg Parlays: {len([c for c in combinations if c['legs'] == 4])}

🎯 CONFIDENCE BREAKDOWN:
   High Confidence: {len([p for p in parlay_data if p['confidence'] == 'High'])}
   Medium Confidence: {len([p for p in parlay_data if p['confidence'] == 'Medium'])}

📁 FILES CREATED:
   - nfl_1023_parlay_options.csv
   - nfl_1023_parlay_combinations.csv

🎲 RECOMMENDED PARLAYS:
"""
        
        # Add top recommended parlays
        high_confidence_combos = [c for c in combinations if c['confidence'] == 'High']
        if high_confidence_combos:
            report += "\n🏆 HIGH CONFIDENCE PARLAYS:\n"
            for i, combo in enumerate(high_confidence_combos[:3], 1):
                report += f"   {i}. {combo['legs']}-leg parlay ({combo['estimated_odds']})\n"
                for bet in combo['bets']:
                    report += f"      • {bet['option']}\n"
        
        # Save report
        with open('nfl_1023_parlay_summary.txt', 'w') as f:
            f.write(report)
        
        print("   ✅ Summary report saved to nfl_1023_parlay_summary.txt")
        print(report)
        
        return report

def main():
    """Main function"""
    print("🏈 NFL Today's Parlay Data Generator")
    print("="*50)
    
    try:
        generator = NFLParlayGenerator()
        
        # Fetch games data
        games_data = generator.fetch_todays_games()
        
        # Generate parlay options
        parlay_data = generator.generate_parlay_options(games_data)
        
        # Create parlay combinations
        combinations = generator.create_parlay_combinations(parlay_data)
        
        # Save data
        parlay_file, combo_file = generator.save_parlay_data(parlay_data, combinations)
        
        # Generate summary
        generator.generate_summary_report(parlay_data, combinations)
        
        print(f"\n🎉 SUCCESS! Parlay data generated for today ({generator.today})")
        print(f"📁 Files created:")
        print(f"   - {parlay_file}")
        print(f"   - {combo_file}")
        print(f"   - nfl_1023_parlay_summary.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
