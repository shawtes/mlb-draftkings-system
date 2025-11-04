#!/usr/bin/env python3
"""
Get Vikings and Lions Week 8 data for parlay creation
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def get_vikings_lions_data():
    """Get Vikings and Lions Week 8 projections"""
    
    # Initialize API
    api = SportsDataNFLAPI('1dd5e646265649af87e0d9cdb80d1c8c')
    
    print('🏈 Fetching Week 8 data for Vikings and Lions...')
    
    # Fetch projections for Week 8
    projections = api.get_player_projections_by_week('2025REG', 8)
    
    if not projections:
        print('❌ No projections found')
        return None, None
    
    proj_df = pd.DataFrame(projections)
    
    # Look for Vikings and Lions
    vikings = proj_df[proj_df['Team'].str.contains('MIN|Minnesota', case=False, na=False)]
    lions = proj_df[proj_df['Team'].str.contains('DET|Detroit', case=False, na=False)]
    
    print(f'📊 Vikings players found: {len(vikings)}')
    print(f'📊 Lions players found: {len(lions)}')
    
    if len(vikings) > 0:
        print(f'\n🟣 VIKINGS PLAYERS:')
        for _, player in vikings.iterrows():
            name = player['Name']
            pos = player['Position']
            dk_pts = player.get('FantasyPointsDraftKings', 0)
            passing_yds = player.get('PassingYards', 0)
            rushing_yds = player.get('RushingYards', 0)
            receiving_yds = player.get('ReceivingYards', 0)
            receptions = player.get('Receptions', 0)
            
            if dk_pts > 0:
                print(f'  {name} ({pos}): {dk_pts:.1f} DK pts')
                if passing_yds > 0:
                    print(f'    Passing: {passing_yds:.0f} yds')
                if rushing_yds > 0:
                    print(f'    Rushing: {rushing_yds:.0f} yds')
                if receiving_yds > 0:
                    print(f'    Receiving: {receiving_yds:.0f} yds, {receptions:.1f} rec')
    
    if len(lions) > 0:
        print(f'\n🔵 LIONS PLAYERS:')
        for _, player in lions.iterrows():
            name = player['Name']
            pos = player['Position']
            dk_pts = player.get('FantasyPointsDraftKings', 0)
            passing_yds = player.get('PassingYards', 0)
            rushing_yds = player.get('RushingYards', 0)
            receiving_yds = player.get('ReceivingYards', 0)
            receptions = player.get('Receptions', 0)
            
            if dk_pts > 0:
                print(f'  {name} ({pos}): {dk_pts:.1f} DK pts')
                if passing_yds > 0:
                    print(f'    Passing: {passing_yds:.0f} yds')
                if rushing_yds > 0:
                    print(f'    Rushing: {rushing_yds:.0f} yds')
                if receiving_yds > 0:
                    print(f'    Receiving: {receiving_yds:.0f} yds, {receptions:.1f} rec')
    
    return vikings, lions

if __name__ == "__main__":
    vikings, lions = get_vikings_lions_data()






