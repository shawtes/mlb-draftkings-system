#!/usr/bin/env python3
"""
Create Vikings vs Chargers parlay based on fantasy projections
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def create_vikings_chargers_parlay():
    """Create parlay legs for Vikings vs Chargers game"""
    
    # Initialize API
    api = SportsDataNFLAPI('1dd5e646265649af87e0d9cdb80d1c8c')
    
    # Fetch projections for Week 8
    projections = api.get_player_projections_by_week('2025REG', 8)
    
    if not projections:
        print('❌ No projections found')
        return
    
    proj_df = pd.DataFrame(projections)
    
    # Get Vikings and Chargers data
    vikings = proj_df[proj_df['Team'].str.contains('MIN|Minnesota', case=False, na=False)]
    chargers = proj_df[proj_df['Team'].str.contains('LAC|Chargers', case=False, na=False)]
    
    print('🏈 VIKINGS vs CHARGERS PARLAY RECOMMENDATIONS')
    print('='*60)
    print('📅 Week 8 - Based on Fantasy Projections')
    print()
    
    sample_lines = []
    
    # VIKINGS PLAYER PROPS
    print('🟣 VIKINGS PLAYER PROPS:')
    
    # Justin Jefferson
    jj = vikings[vikings['Name'].str.contains('Justin Jefferson', case=False, na=False)]
    if len(jj) > 0:
        jj_proj = jj.iloc[0]
        receiving_yds = jj_proj.get('ReceivingYards', 0)
        receptions = jj_proj.get('Receptions', 0)
        dk_pts = jj_proj.get('FantasyPointsDraftKings', 0)
        
        if receiving_yds > 0:
            line = round(receiving_yds * 0.85 / 5) * 5  # Round to nearest 5
            sample_lines.append({
                'player': 'Justin Jefferson',
                'team': 'Vikings',
                'prop': f'Receiving Yards',
                'line': f'O{line}',
                'projection': receiving_yds,
                'confidence': 'HIGH'
            })
            print(f'  ✅ Justin Jefferson Receiving Yards O{line} (Proj: {receiving_yds:.0f} yds)')
        
        if receptions > 0:
            line = round(receptions * 0.85 * 2) / 2  # Round to nearest 0.5
            sample_lines.append({
                'player': 'Justin Jefferson',
                'team': 'Vikings',
                'prop': f'Receptions',
                'line': f'O{line}',
                'projection': receptions,
                'confidence': 'HIGH'
            })
            print(f'  ✅ Justin Jefferson Receptions O{line} (Proj: {receptions:.1f} rec)')
    
    # Carson Wentz
    wentz = vikings[vikings['Name'].str.contains('Carson Wentz', case=False, na=False)]
    if len(wentz) > 0:
        wentz_proj = wentz.iloc[0]
        passing_yds = wentz_proj.get('PassingYards', 0)
        dk_pts = wentz_proj.get('FantasyPointsDraftKings', 0)
        
        if passing_yds > 0:
            line = round(passing_yds * 0.85 / 10) * 10  # Round to nearest 10
            sample_lines.append({
                'player': 'Carson Wentz',
                'team': 'Vikings',
                'prop': f'Passing Yards',
                'line': f'O{line}',
                'projection': passing_yds,
                'confidence': 'MEDIUM'
            })
            print(f'  ✅ Carson Wentz Passing Yards O{line} (Proj: {passing_yds:.0f} yds)')
    
    # Jordan Addison
    addison = vikings[vikings['Name'].str.contains('Jordan Addison', case=False, na=False)]
    if len(addison) > 0:
        addison_proj = addison.iloc[0]
        receiving_yds = addison_proj.get('ReceivingYards', 0)
        receptions = addison_proj.get('Receptions', 0)
        
        if receiving_yds > 0:
            line = round(receiving_yds * 0.85 / 5) * 5
            sample_lines.append({
                'player': 'Jordan Addison',
                'team': 'Vikings',
                'prop': f'Receiving Yards',
                'line': f'O{line}',
                'projection': receiving_yds,
                'confidence': 'MEDIUM'
            })
            print(f'  ✅ Jordan Addison Receiving Yards O{line} (Proj: {receiving_yds:.0f} yds)')
    
    print()
    print('🔵 CHARGERS PLAYER PROPS:')
    
    # Justin Herbert
    herbert = chargers[chargers['Name'].str.contains('Justin Herbert', case=False, na=False)]
    if len(herbert) > 0:
        herbert_proj = herbert.iloc[0]
        passing_yds = herbert_proj.get('PassingYards', 0)
        dk_pts = herbert_proj.get('FantasyPointsDraftKings', 0)
        
        if passing_yds > 0:
            line = round(passing_yds * 0.85 / 10) * 10
            sample_lines.append({
                'player': 'Justin Herbert',
                'team': 'Chargers',
                'prop': f'Passing Yards',
                'line': f'O{line}',
                'projection': passing_yds,
                'confidence': 'HIGH'
            })
            print(f'  ✅ Justin Herbert Passing Yards O{line} (Proj: {passing_yds:.0f} yds)')
    
    # Keenan Allen
    allen = chargers[chargers['Name'].str.contains('Keenan Allen', case=False, na=False)]
    if len(allen) > 0:
        allen_proj = allen.iloc[0]
        receiving_yds = allen_proj.get('ReceivingYards', 0)
        receptions = allen_proj.get('Receptions', 0)
        
        if receiving_yds > 0:
            line = round(receiving_yds * 0.85 / 5) * 5
            sample_lines.append({
                'player': 'Keenan Allen',
                'team': 'Chargers',
                'prop': f'Receiving Yards',
                'line': f'O{line}',
                'projection': receiving_yds,
                'confidence': 'HIGH'
            })
            print(f'  ✅ Keenan Allen Receiving Yards O{line} (Proj: {receiving_yds:.0f} yds)')
        
        if receptions > 0:
            line = round(receptions * 0.85 * 2) / 2
            sample_lines.append({
                'player': 'Keenan Allen',
                'team': 'Chargers',
                'prop': f'Receptions',
                'line': f'O{line}',
                'projection': receptions,
                'confidence': 'HIGH'
            })
            print(f'  ✅ Keenan Allen Receptions O{line} (Proj: {receptions:.1f} rec)')
    
    # Ladd McConkey
    mcconkey = chargers[chargers['Name'].str.contains('Ladd McConkey', case=False, na=False)]
    if len(mcconkey) > 0:
        mcconkey_proj = mcconkey.iloc[0]
        receiving_yds = mcconkey_proj.get('ReceivingYards', 0)
        receptions = mcconkey_proj.get('Receptions', 0)
        
        if receiving_yds > 0:
            line = round(receiving_yds * 0.85 / 5) * 5
            sample_lines.append({
                'player': 'Ladd McConkey',
                'team': 'Chargers',
                'prop': f'Receiving Yards',
                'line': f'O{line}',
                'projection': receiving_yds,
                'confidence': 'MEDIUM'
            })
            print(f'  ✅ Ladd McConkey Receiving Yards O{line} (Proj: {receiving_yds:.0f} yds)')
    
    print()
    print('🎯 RECOMMENDED PARLAYS:')
    print()
    
    # Create parlay combinations
    high_confidence = [line for line in sample_lines if line['confidence'] == 'HIGH']
    
    if len(high_confidence) >= 3:
        print('🏆 HIGH CONFIDENCE 3-LEG PARLAY (+600):')
        for i, line in enumerate(high_confidence[:3]):
            print(f'  {i+1}. {line["player"]} ({line["team"]}) - {line["prop"]} {line["line"]}')
        print(f'     Projected Payout: $700 for $100 bet')
        print()
    
    if len(high_confidence) >= 2:
        print('🎯 SAFE 2-LEG PARLAY (+260):')
        for i, line in enumerate(high_confidence[:2]):
            print(f'  {i+1}. {line["player"]} ({line["team"]}) - {line["prop"]} {line["line"]}')
        print(f'     Projected Payout: $360 for $100 bet')
        print()
    
    # Create a mixed parlay
    if len(sample_lines) >= 4:
        print('🎲 MIXED CONFIDENCE 4-LEG PARLAY (+1000):')
        mixed_lines = sample_lines[:4]
        for i, line in enumerate(mixed_lines):
            print(f'  {i+1}. {line["player"]} ({line["team"]}) - {line["prop"]} {line["line"]} ({line["confidence"]})')
        print(f'     Projected Payout: $1100 for $100 bet')
        print()
    
    print('📊 BETTING STRATEGY:')
    print('  • Use 2-3% of bankroll per parlay')
    print('  • Focus on HIGH confidence props for safer plays')
    print('  • Mix in 1-2 MEDIUM confidence for higher payouts')
    print('  • Consider same-game parlays for better odds')
    print()
    print('⚠️  DISCLAIMER: These are projections, not guarantees!')
    print('   Always bet responsibly and within your means.')

if __name__ == "__main__":
    create_vikings_chargers_parlay()






