#!/usr/bin/env python3
"""
Calculate actual fantasy points for optimized lineups using contest standings data
"""

import pandas as pd
import sys

def load_contest_standings(file_path):
    """Load contest standings and extract player fantasy points"""
    print(f"Loading contest standings from: {file_path}")
    
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Extract player name and FPTS columns
    if 'Player' not in df.columns or 'FPTS' not in df.columns:
        print(f"ERROR: Required columns not found. Available columns: {df.columns.tolist()}")
        return None
    
    # Create player -> FPTS mapping (take max if duplicates)
    player_fpts = {}
    for _, row in df.iterrows():
        player_name = str(row['Player']).strip()
        fpts = row['FPTS']
        
        if pd.notna(player_name) and pd.notna(fpts):
            try:
                fpts_val = float(fpts)
                # Keep highest score if player appears multiple times
                if player_name not in player_fpts or fpts_val > player_fpts[player_name]:
                    player_fpts[player_name] = fpts_val
            except (ValueError, TypeError):
                continue
    
    print(f"✓ Loaded {len(player_fpts)} unique players with fantasy points")
    return player_fpts


def calculate_lineup_scores(lineups_file, player_fpts):
    """Calculate total fantasy points for each lineup"""
    print(f"\nLoading lineups from: {lineups_file}")
    
    # Read lineups CSV
    df_lineups = pd.read_csv(lineups_file)
    
    # Position columns in NBA DraftKings format
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    
    # Verify columns exist
    if not all(pos in df_lineups.columns for pos in positions):
        print(f"ERROR: Expected positions {positions}")
        print(f"Found columns: {df_lineups.columns.tolist()}")
        return None
    
    print(f"✓ Loaded {len(df_lineups)} lineups")
    
    # Calculate scores for each lineup
    results = []
    
    for idx, row in df_lineups.iterrows():
        lineup_num = idx + 1
        total_fpts = 0
        players_found = 0
        players_missing = []
        player_scores = []
        
        for pos in positions:
            player_name = str(row[pos]).strip()
            
            # Skip empty slots
            if not player_name or player_name == '' or player_name == 'nan':
                players_missing.append(f"{pos}: EMPTY")
                continue
            
            # Look up player's actual fantasy points
            if player_name in player_fpts:
                fpts = player_fpts[player_name]
                total_fpts += fpts
                players_found += 1
                player_scores.append(f"{player_name} ({pos}): {fpts}")
            else:
                players_missing.append(f"{pos}: {player_name}")
        
        results.append({
            'Lineup': lineup_num,
            'Total_FPTS': round(total_fpts, 2),
            'Players_Found': players_found,
            'Players_Missing': len(players_missing),
            'Complete': players_found == 8,
            'Missing_Players': ', '.join(players_missing) if players_missing else 'None',
            'Player_Scores': ' | '.join(player_scores)
        })
    
    return pd.DataFrame(results)


def main():
    # File paths
    contest_file = '/Users/sineshawmesfintesfaye/Downloads/contest-standings-183850292.csv'
    lineups_file = '/Users/sineshawmesfintesfaye/Downloads/optimized_lineupsmnmjmjnjm.csv'
    output_file = '/Users/sineshawmesfintesfaye/Downloads/lineup_scores_calculated.csv'
    
    print("=" * 80)
    print("NBA LINEUP FANTASY POINTS CALCULATOR")
    print("=" * 80)
    
    # Load contest standings (player fantasy points)
    player_fpts = load_contest_standings(contest_file)
    if player_fpts is None:
        return 1
    
    # Calculate lineup scores
    results_df = calculate_lineup_scores(lineups_file, player_fpts)
    if results_df is None:
        return 1
    
    # Display results
    print("\n" + "=" * 80)
    print("LINEUP SCORES SUMMARY")
    print("=" * 80)
    
    # Sort by total points (descending)
    results_df_sorted = results_df.sort_values('Total_FPTS', ascending=False)
    
    # Summary statistics
    complete_lineups = results_df[results_df['Complete'] == True]
    incomplete_lineups = results_df[results_df['Complete'] == False]
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total Lineups: {len(results_df)}")
    print(f"   Complete Lineups (8/8 players): {len(complete_lineups)}")
    print(f"   Incomplete Lineups: {len(incomplete_lineups)}")
    
    if len(complete_lineups) > 0:
        print(f"\n   Average Score (Complete): {complete_lineups['Total_FPTS'].mean():.2f}")
        print(f"   Best Score: {complete_lineups['Total_FPTS'].max():.2f} (Lineup {complete_lineups.loc[complete_lineups['Total_FPTS'].idxmax(), 'Lineup']})")
        print(f"   Worst Score: {complete_lineups['Total_FPTS'].min():.2f} (Lineup {complete_lineups.loc[complete_lineups['Total_FPTS'].idxmin(), 'Lineup']})")
    
    # Show top 10 lineups
    print(f"\n🏆 TOP 10 LINEUPS BY FANTASY POINTS:")
    print("-" * 80)
    for _, row in results_df_sorted.head(10).iterrows():
        status = "✓" if row['Complete'] else "⚠"
        print(f"{status} Lineup {row['Lineup']:3d}: {row['Total_FPTS']:6.2f} pts  ({row['Players_Found']}/8 players)")
    
    # Show lineups with missing players
    if len(incomplete_lineups) > 0:
        print(f"\n⚠️  LINEUPS WITH MISSING PLAYERS:")
        print("-" * 80)
        for _, row in incomplete_lineups.head(10).iterrows():
            print(f"   Lineup {row['Lineup']:3d}: {row['Total_FPTS']:6.2f} pts  (Missing: {row['Missing_Players']})")
    
    # Save detailed results
    results_df_sorted.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Also create a simple summary file
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df = results_df_sorted[['Lineup', 'Total_FPTS', 'Players_Found', 'Complete']].copy()
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("✓ CALCULATION COMPLETE")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

