"""
NBA DraftKings Slate Optimizer
Fetches projections from SportsData.io API and filters to only players in your DK slate
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_sportsdata_fetcher import NBADataFetcher
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
DK_ENTRIES_FILE = "/Users/sineshawmesfintesfaye/Downloads/DKEntries-3.csv"
OUTPUT_DIR = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION"

print("🏀 NBA DraftKings Slate Optimizer")
print("=" * 70)
print(f"API Key: {API_KEY[:20]}...")
print(f"DK Slate: {DK_ENTRIES_FILE}")
print("=" * 70)

# ============================================================================
# STEP 1: Load DraftKings Slate Players
# ============================================================================

print("\n📋 Step 1: Loading DraftKings Slate Players...")

# Read the DK file - player pool starts at line 8
dk_df = pd.read_csv(DK_ENTRIES_FILE, skiprows=7)

# Extract relevant columns
player_pool = dk_df[['ID', 'Name', 'Position', 'Roster Position', 'Salary', 'Game Info', 'TeamAbbrev', 'AvgPointsPerGame']].copy()

# Clean up column names
player_pool.columns = ['DK_ID', 'Name', 'DK_Position', 'Roster_Position', 'Salary', 'Game_Info', 'Team', 'DK_AvgPoints']

# Convert to proper types
player_pool['DK_ID'] = player_pool['DK_ID'].astype(str)
player_pool['Salary'] = pd.to_numeric(player_pool['Salary'], errors='coerce')
player_pool['DK_AvgPoints'] = pd.to_numeric(player_pool['DK_AvgPoints'], errors='coerce')

# Remove any rows with missing critical data
player_pool = player_pool.dropna(subset=['DK_ID', 'Name', 'Salary'])

print(f"✅ Loaded {len(player_pool)} players from DraftKings slate")
print(f"   Salary range: ${player_pool['Salary'].min():,.0f} - ${player_pool['Salary'].max():,.0f}")
print(f"   Teams: {player_pool['Team'].nunique()} teams")

# ============================================================================
# STEP 2: Fetch API Projections
# ============================================================================

print("\n📡 Step 2: Fetching Projections from SportsData.io API...")

fetcher = NBADataFetcher(API_KEY)
today = datetime.now().strftime('%Y-%b-%d').upper()

# Get today's projections
api_projections = fetcher.get_daily_projections(today)

if api_projections.empty:
    print("❌ No API projections available for today")
    print(f"   Falling back to DK average points as projections...")
    
    # Use DK average points as projections
    final_df = player_pool.copy()
    final_df['Projected_DK_Points'] = final_df['DK_AvgPoints']
    final_df['Ceiling'] = final_df['DK_AvgPoints'] * 1.3
    final_df['Floor'] = final_df['DK_AvgPoints'] * 0.7
    final_df['Source'] = 'DK_AvgPoints'
    
else:
    print(f"✅ Fetched {len(api_projections)} player projections from API")
    
    # ========================================================================
    # STEP 3: Match API Projections to DK Slate Players
    # ========================================================================
    
    print("\n🔗 Step 3: Matching API Projections to DK Slate...")
    
    # Create a name matching dictionary (handle slight name differences)
    def normalize_name(name):
        """Normalize player names for matching"""
        if pd.isna(name):
            return ""
        name = str(name).strip().upper()
        # Remove Jr., Sr., III, etc.
        name = name.replace(' JR.', '').replace(' SR.', '').replace(' III', '').replace(' II', '')
        name = name.replace('.', '').replace("'", '')
        return name
    
    # Normalize names in both dataframes
    player_pool['Name_Normalized'] = player_pool['Name'].apply(normalize_name)
    api_projections['Name_Normalized'] = api_projections['Name'].apply(normalize_name)
    
    # Check what columns are available
    print(f"   Available API columns: {list(api_projections.columns)}")
    
    # Select only columns that exist
    merge_columns = ['Name_Normalized', 'Team', 'Predicted_DK_Points']
    optional_columns = ['Ceiling', 'Floor', 'Minutes', 'IsInjured', 'InjuryStatus', 
                       'Minutes_Projected', 'Injury_Status']
    
    for col in optional_columns:
        if col in api_projections.columns:
            merge_columns.append(col)
    
    # Merge on normalized name and team
    merged_df = player_pool.merge(
        api_projections[merge_columns],
        on=['Name_Normalized', 'Team'],
        how='left'
    )
    
    # For players without API projections, use DK average points
    merged_df['Projected_DK_Points'] = merged_df['Predicted_DK_Points'].fillna(merged_df['DK_AvgPoints'])
    
    # Add Ceiling and Floor if not present
    if 'Ceiling' not in merged_df.columns:
        merged_df['Ceiling'] = merged_df['Projected_DK_Points'] * 1.3
    else:
        merged_df['Ceiling'] = merged_df['Ceiling'].fillna(merged_df['DK_AvgPoints'] * 1.3)
    
    if 'Floor' not in merged_df.columns:
        merged_df['Floor'] = merged_df['Projected_DK_Points'] * 0.7
    else:
        merged_df['Floor'] = merged_df['Floor'].fillna(merged_df['DK_AvgPoints'] * 0.7)
    
    # Add Minutes if not present
    if 'Minutes' not in merged_df.columns:
        if 'Minutes_Projected' in merged_df.columns:
            merged_df['Minutes'] = merged_df['Minutes_Projected']
        else:
            merged_df['Minutes'] = 30
    else:
        merged_df['Minutes'] = merged_df['Minutes'].fillna(30)
    
    # Add injury info if not present
    if 'IsInjured' not in merged_df.columns:
        if 'Injury_Status' in merged_df.columns:
            merged_df['IsInjured'] = merged_df['Injury_Status'].notna() & (merged_df['Injury_Status'] != '')
            merged_df['InjuryStatus'] = merged_df['Injury_Status']
        else:
            merged_df['IsInjured'] = False
            merged_df['InjuryStatus'] = ''
    else:
        merged_df['IsInjured'] = merged_df['IsInjured'].fillna(False)
        merged_df['InjuryStatus'] = merged_df['InjuryStatus'].fillna('')
    
    # Mark source of projections
    merged_df['Source'] = merged_df['Predicted_DK_Points'].notna().apply(
        lambda x: 'API_Projection' if x else 'DK_AvgPoints'
    )
    
    final_df = merged_df
    
    matched_count = (final_df['Source'] == 'API_Projection').sum()
    print(f"✅ Matched {matched_count}/{len(final_df)} players to API projections")
    print(f"   Using DK averages for remaining {len(final_df) - matched_count} players")

# ============================================================================
# STEP 4: Add Value Metrics
# ============================================================================

print("\n💰 Step 4: Calculating Value Metrics...")

# Points per $1000 of salary
final_df['Value'] = final_df['Projected_DK_Points'] / (final_df['Salary'] / 1000)
final_df['Ceiling_Value'] = final_df['Ceiling'] / (final_df['Salary'] / 1000)

# Ownership estimate (simple heuristic based on salary and projection)
final_df['Est_Ownership'] = (
    (final_df['Salary'] / final_df['Salary'].max()) * 0.3 +
    (final_df['Projected_DK_Points'] / final_df['Projected_DK_Points'].max()) * 0.7
) * 100

print(f"✅ Calculated value metrics for all players")

# ============================================================================
# STEP 5: Export Optimized Player Pool
# ============================================================================

print("\n💾 Step 5: Exporting Player Pool...")

# Prepare final columns for export
export_df = final_df[[
    'DK_ID', 'Name', 'DK_Position', 'Roster_Position', 'Team', 'Salary', 
    'Projected_DK_Points', 'Ceiling', 'Floor', 'Value', 'Ceiling_Value',
    'Minutes', 'IsInjured', 'InjuryStatus', 'Est_Ownership', 'Source', 'Game_Info'
]].copy()

# Sort by projected points (descending)
export_df = export_df.sort_values('Projected_DK_Points', ascending=False)

# Export to CSV
output_file = os.path.join(OUTPUT_DIR, f"nba_slate_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
export_df.to_csv(output_file, index=False)

print(f"✅ Exported optimized player pool to:")
print(f"   {output_file}")

# ============================================================================
# STEP 6: Display Top Players by Position
# ============================================================================

print("\n" + "=" * 70)
print("🌟 TOP PLAYERS BY PROJECTED POINTS")
print("=" * 70)

# Get primary position for grouping
def get_primary_position(roster_pos):
    """Extract primary position from roster position string"""
    if pd.isna(roster_pos):
        return "UTIL"
    positions = str(roster_pos).split('/')
    return positions[0]

export_df['Primary_Position'] = export_df['Roster_Position'].apply(get_primary_position)

# Show top 5 overall
print("\n🏆 TOP 5 OVERALL:")
print("-" * 70)
top5 = export_df.head(5)
for idx, player in top5.iterrows():
    injury_flag = " 🤕" if player['IsInjured'] else ""
    print(f"{player['Name']:25s} | {player['Primary_Position']:5s} | {player['Team']:3s} | "
          f"${player['Salary']:5,.0f} | {player['Projected_DK_Points']:5.1f} pts | "
          f"Value: {player['Value']:.2f}{injury_flag}")

# Show top 3 by each core position
for position in ['PG', 'SG', 'SF', 'PF', 'C']:
    pos_players = export_df[export_df['Primary_Position'] == position].head(3)
    if not pos_players.empty:
        print(f"\n{position}:")
        print("-" * 70)
        for idx, player in pos_players.iterrows():
            injury_flag = " 🤕" if player['IsInjured'] else ""
            print(f"{player['Name']:25s} | ${player['Salary']:5,.0f} | "
                  f"{player['Projected_DK_Points']:5.1f} pts | "
                  f"Value: {player['Value']:.2f}{injury_flag}")

# ============================================================================
# STEP 7: Value Plays (Punt Options)
# ============================================================================

print("\n" + "=" * 70)
print("💎 TOP VALUE PLAYS (Best Points per $1000)")
print("=" * 70)

# Filter for players under $6000 with good value
value_plays = export_df[
    (export_df['Salary'] < 6000) & 
    (export_df['Projected_DK_Points'] > 20)
].sort_values('Value', ascending=False).head(10)

for idx, player in value_plays.iterrows():
    injury_flag = " 🤕" if player['IsInjured'] else ""
    print(f"{player['Name']:25s} | {player['Primary_Position']:5s} | ${player['Salary']:5,.0f} | "
          f"{player['Projected_DK_Points']:5.1f} pts | Value: {player['Value']:.2f}{injury_flag}")

# ============================================================================
# STEP 8: Injury Report
# ============================================================================

injured_players = export_df[export_df['IsInjured'] == True]
if not injured_players.empty:
    print("\n" + "=" * 70)
    print("🤕 INJURY REPORT")
    print("=" * 70)
    for idx, player in injured_players.iterrows():
        print(f"{player['Name']:25s} | {player['Team']:3s} | "
              f"Status: {player['InjuryStatus']}")
else:
    print("\n✅ No injured players in slate")

# ============================================================================
# SUMMARY STATS
# ============================================================================

print("\n" + "=" * 70)
print("📊 SLATE SUMMARY")
print("=" * 70)
print(f"Total Players: {len(export_df)}")
print(f"Total Salary Cap: $50,000")
print(f"Average Salary: ${export_df['Salary'].mean():,.0f}")
print(f"Average Projection: {export_df['Projected_DK_Points'].mean():.1f} pts")
print(f"Highest Projection: {export_df['Projected_DK_Points'].max():.1f} pts ({export_df.loc[export_df['Projected_DK_Points'].idxmax(), 'Name']})")
print(f"API Projections: {(export_df['Source'] == 'API_Projection').sum()}")
print(f"DK Averages Used: {(export_df['Source'] == 'DK_AvgPoints').sum()}")

print("\n" + "=" * 70)
print("✅ OPTIMIZATION COMPLETE!")
print("=" * 70)
print("\nNext Steps:")
print("1. Load the exported CSV into your lineup optimizer")
print("2. Consider injury updates before finalizing lineups")
print("3. Use value plays to free up salary for studs")
print("4. Check game totals for high-scoring game stacks")
print("=" * 70)

