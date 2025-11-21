#!/usr/bin/env python3
"""
Analyze NBA slate data to identify game counts
"""

import pandas as pd
import sys

def analyze_nba_file(filename):
    """Analyze NBA data file"""
    print(f"\n{'='*60}")
    print(f"🏀 Analyzing: {filename}")
    print(f"{'='*60}\n")
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded {len(df)} player records")
        
        # Get unique game IDs
        if 'GameID' in df.columns:
            game_ids = df['GameID'].unique()
            num_games = len([gid for gid in game_ids if pd.notna(gid)])
            
            print(f"\n📊 Game Count: {num_games} games")
            
            # Show game details
            if 'Team' in df.columns and 'Opponent' in df.columns and 'HomeOrAway' in df.columns:
                print(f"\n🏀 Game Breakdown:")
                game_teams = {}
                for _, row in df.iterrows():
                    gid = row.get('GameID')
                    team = row.get('Team')
                    opp = row.get('Opponent')
                    home_away = row.get('HomeOrAway', '')
                    
                    if pd.notna(gid) and pd.notna(team) and pd.notna(opp):
                        if gid not in game_teams:
                            if home_away == 'HOME':
                                game_teams[gid] = f"{team} vs {opp}"
                            else:
                                game_teams[gid] = f"{opp} vs {team}"
                
                for gid, matchup in sorted(game_teams.items()):
                    players = len(df[df['GameID'] == gid].drop_duplicates('Name'))
                    print(f"   Game {gid}: {matchup} ({players} players)")
            
            # Position breakdown
            if 'Position' in df.columns:
                print(f"\n📊 Position Breakdown:")
                pos_counts = df['Position'].value_counts().sort_index()
                for pos, count in pos_counts.items():
                    print(f"   {pos}: {count}")
            
            # Summary by games
            if num_games == 9:
                print(f"\n✅ This is a 9-game slate!")
            elif num_games > 9:
                print(f"\n⚠️ This is a {num_games}-game slate (larger than 9)")
            else:
                print(f"\n⚠️ This is only a {num_games}-game slate (less than 9)")
        
        # Check required columns for optimizer
        required_cols = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points']
        print(f"\n📋 Optimizer Columns:")
        for col in required_cols:
            status = "✅" if col in df.columns else "❌"
            print(f"   {status} {col}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Check command line args
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "nba_tonight_6pm_2025OCT26.csv"
    
    df = analyze_nba_file(filename)
    
    if df is not None:
        print(f"\n{'='*60}")
        print("💡 To use with genetic optimizer:")
        print(f"   python \"nba_sportsdata.io_gentic algo.py\"")
        print(f"   Then load: {filename}")
        print(f"{'='*60}\n")









