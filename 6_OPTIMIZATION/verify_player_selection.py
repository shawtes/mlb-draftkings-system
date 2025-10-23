#!/usr/bin/env python3
"""
Tool to verify which players ended up in lineups vs which players should have been selected.
This helps identify if unwanted players leaked into the optimization.
"""

import pandas as pd
import sys

def verify_lineup_players(lineups_csv, expected_players_txt=None):
    """
    Verify which players are in the lineups.
    
    Args:
        lineups_csv: Path to optimized lineups CSV
        expected_players_txt: Optional text file with one player name per line (your intended selections)
    """
    print("=" * 80)
    print("LINEUP PLAYER VERIFICATION TOOL")
    print("=" * 80)
    
    # Load lineups
    df_lineups = pd.read_csv(lineups_csv)
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    
    # Extract all unique players
    all_players = set()
    for pos in positions:
        if pos in df_lineups.columns:
            players = df_lineups[pos].dropna().unique()
            all_players.update([str(p).strip() for p in players if str(p) != 'nan'])
    
    all_players = sorted(all_players)
    
    print(f"\n📊 LINEUPS FILE: {lineups_csv}")
    print(f"   Total lineups: {len(df_lineups)}")
    print(f"   Unique players: {len(all_players)}")
    
    # Count appearances
    player_counts = {}
    for player in all_players:
        count = 0
        for pos in positions:
            if pos in df_lineups.columns:
                count += (df_lineups[pos] == player).sum()
        player_counts[player] = count
    
    # Sort by appearances
    sorted_players = sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 MOST USED PLAYERS:")
    for player, count in sorted_players[:10]:
        print(f"   {count:3d}x - {player}")
    
    # If expected players provided, compare
    if expected_players_txt:
        try:
            with open(expected_players_txt, 'r') as f:
                expected = set(line.strip() for line in f if line.strip())
            
            actual = set(all_players)
            
            unexpected = actual - expected
            missing = expected - actual
            
            print(f"\n{'='*80}")
            print(f"COMPARISON WITH EXPECTED SELECTIONS")
            print(f"{'='*80}")
            print(f"   Expected players: {len(expected)}")
            print(f"   Actual players: {len(actual)}")
            print(f"   Match: {len(actual & expected)}")
            
            if unexpected:
                print(f"\n⚠️  UNEXPECTED PLAYERS IN LINEUPS ({len(unexpected)}):")
                for player in sorted(unexpected):
                    count = player_counts[player]
                    print(f"   {count:3d}x - {player}")
            
            if missing:
                print(f"\n❌ SELECTED PLAYERS NOT IN LINEUPS ({len(missing)}):")
                for player in sorted(missing)[:20]:  # Show first 20
                    print(f"   - {player}")
                if len(missing) > 20:
                    print(f"   ... and {len(missing) - 20} more")
        
        except FileNotFoundError:
            print(f"\n⚠️  Expected players file not found: {expected_players_txt}")
    
    else:
        print(f"\n💡 TIP: To verify against your intended selections:")
        print(f"   1. Create a text file with one player name per line")
        print(f"   2. Run: python3 verify_player_selection.py lineups.csv expected_players.txt")
    
    # Save all players to file for easy review
    output_file = lineups_csv.replace('.csv', '_players_list.txt')
    with open(output_file, 'w') as f:
        f.write("PLAYERS IN LINEUPS (sorted by usage)\n")
        f.write("=" * 80 + "\n\n")
        for player, count in sorted_players:
            f.write(f"{count:3d}x - {player}\n")
    
    print(f"\n💾 Full player list saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 verify_player_selection.py <lineups.csv> [expected_players.txt]")
        print("\nExample:")
        print("  python3 verify_player_selection.py optimized_lineups3333333.csv")
        print("  python3 verify_player_selection.py optimized_lineups3333333.csv my_selections.txt")
        sys.exit(1)
    
    lineups_csv = sys.argv[1]
    expected_txt = sys.argv[2] if len(sys.argv) > 2 else None
    
    verify_lineup_players(lineups_csv, expected_txt)

