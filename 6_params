#!/usr/bin/env python3
"""
Get Vikings opponent data for parlay creation
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def get_vikings_opponent_data():
    """Get Vikings opponent data"""
    
    # Initialize API
    api = SportsDataNFLAPI('1dd5e646265649af87e0d9cdb80d1c8c')
    
    # Fetch projections for Week 8
    projections = api.get_player_projections_by_week('2025REG', 8)
    
    if not projections:
        print('❌ No projections found')
        return None
    
    proj_df = pd.DataFrame(projections)
    
    # Get Vikings data
    vikings = proj_df[proj_df['Team'].str.contains('MIN|Minnesota', case=False, na=False)]
    
    if len(vikings) > 0:
        # Check who Vikings are playing against
        opponent = vikings.iloc[0].get('Opponent', 'Unknown')
        print(f'🏈 Vikings are playing against: {opponent}')
        
        # Look for the opponent team
        opponent_team = proj_df[proj_df['Team'].str.contains(opponent, case=False, na=False)]
        print(f'📊 Opponent players found: {len(opponent_team)}')
        
        if len(opponent_team) > 0:
            print(f'\n🔵 {opponent.upper()} PLAYERS:')
            for _, player in opponent_team.iterrows():
                name = player['Name']
                pos = player['Position']
                dk_pts = player.get('FantasyPointsDraftKings', 0)
                passing_yds = player.get('PassingYards', 0)
                rushing_yds = player.get('RushingYards', 0)
                receiving_yds = player.get('ReceivingYards', 0)
                receptions = player.get('Receptions', 0)
                
                if dk_pts > 5:  # Only show significant players
                    print(f'  {name} ({pos}): {dk_pts:.1f} DK pts')
                    if passing_yds > 0:
                        print(f'    Passing: {passing_yds:.0f} yds')
                    if rushing_yds > 0:
                        print(f'    Rushing: {rushing_yds:.0f} yds')
                    if receiving_yds > 0:
                        print(f'    Receiving: {receiving_yds:.0f} yds, {receptions:.1f} rec')
        
        return vikings, opponent_team, opponent
    
    return None, None, None

if __name__ == "__main__":
    vikings, opponent, opponent_name = get_vikings_opponent_data()






