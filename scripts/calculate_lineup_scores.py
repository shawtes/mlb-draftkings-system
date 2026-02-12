#!/usr/bin/env python3
"""
Calculate total fantasy points for lineups by mapping player IDs to contest standings
"""

import pandas as pd
import sys

def extract_player_scores(contest_file):
    """Extract player ID to FPTS mapping from contest standings"""
    print(f"Reading contest standings from: {contest_file}")
    
    # Read the contest standings - it has a complex structure
    df = pd.read_csv(contest_file)
    
    # The FPTS column contains scores, and we need to extract player IDs from the lineup
    # Let's look at the structure more carefully
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Create a mapping of player ID to points
    # We'll need to parse the Lineup column to extract player IDs
    player_scores = {}
    
    # The contest file format seems to have Player and FPTS columns
    if 'Player' in df.columns and 'FPTS' in df.columns:
        # Simple case - direct mapping
        for _, row in df.iterrows():
            player_name = str(row['Player']).strip()
            fpts = row['FPTS']
            if pd.notna(fpts):
                player_scores[player_name] = float(fpts)
    
    print(f"Extracted scores for {len(player_scores)} players")
    return player_scores

def extract_player_ids_from_dksalaries(salaries_file):
    """Extract player ID to name mapping from DKSalaries file"""
    try:
        df = pd.read_csv(salaries_file)
        player_mapping = {}
        for _, row in df.iterrows():
            if 'ID' in df.columns and 'Name' in df.columns:
                player_id = str(row['ID']).strip()
                player_name = str(row['Name']).strip()
                player_mapping[player_id] = player_name
        print(f"Loaded {len(player_mapping)} player ID mappings from salaries file")
        return player_mapping
    except Exception as e:
        print(f"Could not load DKSalaries file: {e}")
        return {}

def calculate_lineup_totals(entries_file, contest_file, player_pool_file=None):
    """Calculate total fantasy points for each lineup"""
    
    # First, let's read the contest standings to understand its structure
    print("\n" + "="*80)
    print("ANALYZING CONTEST STANDINGS FILE")
    print("="*80)
    
    contest_df = pd.read_csv(contest_file)
    print(f"\nContest file columns: {contest_df.columns.tolist()}")
    print(f"Contest file shape: {contest_df.shape}")
    print(f"\nFirst row sample:")
    print(contest_df.head(1).to_string())
    
    # Load player ID to name mapping from player pool
    player_id_to_name = {}
    
    if player_pool_file:
        print(f"\nLoading player mappings from: {player_pool_file}")
        try:
            pool_df = pd.read_csv(player_pool_file)
            for _, row in pool_df.iterrows():
                if 'ID' in pool_df.columns and 'Name' in pool_df.columns:
                    player_id = str(int(row['ID'])) if pd.notna(row['ID']) else None
                    player_name = str(row['Name']).strip() if pd.notna(row['Name']) else None
                    if player_id and player_name:
                        player_id_to_name[player_id] = player_name
            print(f"Loaded {len(player_id_to_name)} player ID to name mappings")
        except Exception as e:
            print(f"Error loading player pool: {e}")
    
    # Extract player scores from contest standings
    # The contest file has multiple rows per entry, with each row showing one player
    print("\n" + "="*80)
    print("EXTRACTING PLAYER SCORES")
    print("="*80)
    
    player_scores = {}
    
    # Group by EntryId and collect player scores
    if 'Player' in contest_df.columns and 'FPTS' in contest_df.columns:
        for _, row in contest_df.iterrows():
            player_name = str(row['Player']).strip()
            fpts = row['FPTS']
            
            if pd.notna(fpts) and player_name:
                try:
                    fpts_value = float(fpts)
                    # Keep the highest score if we see the same player multiple times
                    if player_name not in player_scores or fpts_value > player_scores[player_name]:
                        player_scores[player_name] = fpts_value
                except (ValueError, TypeError):
                    pass
    
    print(f"Extracted fantasy points for {len(player_scores)} unique players")
    
    if player_scores:
        # Show sample scores
        sample_players = list(player_scores.items())[:5]
        print("\nSample player scores:")
        for name, score in sample_players:
            print(f"  {name}: {score} FPTS")
    
    # Now map player IDs to names and calculate totals
    print("\n" + "="*80)
    print("CALCULATING LINEUP TOTALS")
    print("="*80)
    
    entries_df = pd.read_csv(entries_file)
    
    # Create reverse mapping from name to player ID
    name_to_id = {name: pid for pid, name in player_id_to_name.items()}
    
    results = []
    
    for idx, row in entries_df.iterrows():
        entry_id = row['Entry ID']
        
        # Collect all player IDs from this lineup
        player_ids = []
        for col in ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']:
            if col in row and pd.notna(row[col]):
                player_ids.append(str(int(row[col])))
        
        # Calculate total points
        total_points = 0
        player_details = []
        missing_players = []
        
        for player_id in player_ids:
            # Try to find this player's score
            player_name = player_id_to_name.get(player_id, None)
            
            if player_name and player_name in player_scores:
                points = player_scores[player_name]
                total_points += points
                player_details.append(f"{player_name}: {points}")
            else:
                # Try to match by ID in player names (some files use player IDs)
                missing_players.append(player_id)
        
        results.append({
            'Entry ID': entry_id,
            'Total FPTS': round(total_points, 2),
            'Players Found': len(player_details),
            'Players Missing': len(missing_players),
            'Details': ' | '.join(player_details) if player_details else 'No players matched'
        })
    
    results_df = pd.DataFrame(results)
    
    # Display results
    print(f"\nProcessed {len(results_df)} lineups")
    print("\n" + "="*80)
    print("LINEUP SCORES")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save to file
    output_file = entries_file.replace('.csv', '_with_scores.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    # Show summary statistics
    if results_df['Players Found'].sum() > 0:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Average Total FPTS: {results_df['Total FPTS'].mean():.2f}")
        print(f"Max Total FPTS: {results_df['Total FPTS'].max():.2f}")
        print(f"Min Total FPTS: {results_df['Total FPTS'].min():.2f}")
        print(f"Total Players Matched: {results_df['Players Found'].sum()}")
        print(f"Total Players Missing: {results_df['Players Missing'].sum()}")
    else:
        print("\n⚠️  WARNING: No players were matched. The player ID mapping may be incorrect.")
        print("Please ensure you have the DKSalaries file in the same directory as the contest standings.")

if __name__ == "__main__":
    entries_file = "/Users/sineshawmesfintesfaye/Downloads/my_favorites_entries.csv"
    contest_file = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/contest-standings-183480502.csv"
    
    # First, load the player pool to get ID to name mapping
    player_pool_file = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nfl_week7_DK_PLAYER_POOL_COMPLETE.csv"
    print("="*80)
    print("LOADING PLAYER ID MAPPING")
    print("="*80)
    pool_df = pd.read_csv(player_pool_file)
    print(f"Loaded {len(pool_df)} players from player pool")
    print(f"Columns: {pool_df.columns.tolist()}")
    
    calculate_lineup_totals(entries_file, contest_file, player_pool_file)

