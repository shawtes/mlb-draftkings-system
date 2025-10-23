#!/usr/bin/env python3
"""
Add NBA positions (PG, SG, SF, PF, C, G, F, UTIL) to lineups in lineup.txt.md
IMPROVED VERSION - Ensures all 8 positions are filled
"""

import pandas as pd
import re

# Load the player pool to get position eligibility
player_pool_path = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_131902.csv"
try:
    df_players = pd.read_csv(player_pool_path)
    print(f"✅ Loaded player pool: {len(df_players)} players")
except:
    print(f"⚠️ Could not load {player_pool_path}, trying alternative...")
    player_pool_path = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv"
    df_players = pd.read_csv(player_pool_path)
    print(f"✅ Loaded player pool: {len(df_players)} players")

# Create a player position lookup dictionary
player_positions = {}
for _, row in df_players.iterrows():
    name = row['Name']
    # Get position eligibility (e.g., "PG/G/UTIL" or "PF/C/F/UTIL")
    if 'DK_Position' in row:
        positions = str(row['DK_Position']).split('/')
    elif 'Roster_Position' in row:
        positions = str(row['Roster_Position']).split('/')
    elif 'Position' in row:
        positions = str(row['Position']).split('/')
    else:
        positions = ['UTIL']
    
    player_positions[name] = positions
    print(f"  {name}: {positions}")

def is_eligible(player, position):
    """Check if a player is eligible for a position"""
    if player not in player_positions:
        return True  # Assume eligible if not found
    
    eligible = player_positions[player]
    
    # Direct match
    if position in eligible:
        return True
    
    # Check flex eligibility
    if position == 'G' and any(p in ['PG', 'SG'] for p in eligible):
        return True
    if position == 'F' and any(p in ['SF', 'PF'] for p in eligible):
        return True
    if position == 'UTIL':  # Everyone is eligible for UTIL
        return True
    
    return False

def assign_lineup_positions(players):
    """
    Assign players to NBA lineup positions: PG, SG, SF, PF, C, G, F, UTIL
    Ensures all 8 positions are filled
    """
    if len(players) != 8:
        print(f"⚠️ Warning: Expected 8 players, got {len(players)}")
    
    assigned = {}
    remaining_players = list(players)
    
    print(f"\n🔄 Assigning positions for: {players}")
    
    # First pass: Assign players to their strict positions (PG, SG, SF, PF, C)
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for player in remaining_players[:]:
            if is_eligible(player, pos):
                assigned[pos] = player
                remaining_players.remove(player)
                print(f"  ✅ {pos}: {player}")
                break
        if pos not in assigned:
            print(f"  ⚠️ No {pos} found, will try flex positions")
    
    # Second pass: Fill G position (eligible: PG or SG)
    if 'G' not in assigned:
        for player in remaining_players[:]:
            if is_eligible(player, 'G'):
                assigned['G'] = player
                remaining_players.remove(player)
                print(f"  ✅ G: {player}")
                break
    
    # Third pass: Fill F position (eligible: SF or PF)
    if 'F' not in assigned:
        for player in remaining_players[:]:
            if is_eligible(player, 'F'):
                assigned['F'] = player
                remaining_players.remove(player)
                print(f"  ✅ F: {player}")
                break
    
    # Fourth pass: Fill UTIL with anyone remaining
    if 'UTIL' not in assigned and remaining_players:
        assigned['UTIL'] = remaining_players[0]
        remaining_players.remove(remaining_players[0])
        print(f"  ✅ UTIL: {assigned['UTIL']}")
    
    # Ensure we have all 8 positions
    expected_positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    for pos in expected_positions:
        if pos not in assigned and remaining_players:
            assigned[pos] = remaining_players[0]
            remaining_players.remove(remaining_players[0])
            print(f"  🔧 {pos} (fallback): {assigned[pos]}")
    
    print(f"  ✅ Assigned {len(assigned)}/8 positions\n")
    return assigned

def format_lineup_with_positions(lineup_text, lineup_number, is_gpp=False):
    """
    Parse a lineup and add position labels
    """
    lines = lineup_text.strip().split('\n')
    
    # Extract player names (skip header and total lines)
    players = []
    player_lines = {}
    for line in lines:
        # Skip headers and totals
        if 'Name' in line or 'Total' in line or '===' in line or '---' in line or not line.strip():
            continue
        # Extract player name (first column, may have spaces)
        # More flexible pattern to catch names like "D'Angelo Russell" or "Karl-Anthony Towns"
        parts = line.strip().split()
        if len(parts) >= 3:
            # Find where the team abbreviation is (2-4 letter all caps)
            team_idx = -1
            for i, part in enumerate(parts):
                if len(part) in [2, 3, 4] and part.isupper() and part.isalpha():
                    team_idx = i
                    break
            
            if team_idx > 0:
                player_name = ' '.join(parts[:team_idx]).strip()
                players.append(player_name)
                player_lines[player_name] = line
    
    if not players or len(players) == 0:
        print(f"⚠️ No players found in lineup {lineup_number}")
        return lineup_text
    
    print(f"\n📋 Processing Lineup {lineup_number}: {len(players)} players")
    
    # Assign positions
    position_assignment = assign_lineup_positions(players)
    
    # Rebuild lineup with positions
    result_lines = []
    if is_gpp:
        result_lines.append(f"🎯 GPP Lineup {lineup_number} (with positions):")
        result_lines.append("Pos  Name                 Team  Salary  Ceiling  Est_Ownership")
    else:
        result_lines.append(f"💵 Cash Lineup {lineup_number} (with positions):")
        result_lines.append("Pos  Name                 Team  Salary  Floor  Projected_DK_Points")
    result_lines.append("-" * 70)
    
    # Add players in position order
    nba_positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    for pos in nba_positions:
        if pos in position_assignment:
            player_name = position_assignment[pos]
            if player_name in player_lines:
                result_lines.append(f"{pos:4} {player_lines[player_name].strip()}")
            else:
                result_lines.append(f"{pos:4} {player_name} [data not found]")
    
    # Add total line
    for line in lines:
        if 'Total' in line:
            result_lines.append(line)
            break
    
    return '\n'.join(result_lines)

# Read the lineup file
with open('/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/lineup.txt.md', 'r') as f:
    content = f.read()

# Find and process all lineups
output_lines = []
lines = content.split('\n')

i = 0
cash_count = 0
gpp_count = 0

while i < len(lines):
    line = lines[i]
    
    # Check for Cash lineup
    if line.startswith('💵 Cash Lineup'):
        lineup_num = re.search(r'Lineup (\d+)', line)
        if lineup_num:
            cash_count += 1
            num = lineup_num.group(1)
            # Collect the lineup (next ~10 lines until we hit Total or next lineup)
            lineup_block = []
            i += 1
            while i < len(lines):
                if lines[i].startswith('💵') or lines[i].startswith('🎯') or lines[i].startswith('======'):
                    break
                lineup_block.append(lines[i])
                if 'Total' in lines[i]:
                    i += 1
                    break
                i += 1
            
            # Format with positions
            formatted = format_lineup_with_positions('\n'.join(lineup_block), num, is_gpp=False)
            output_lines.append(formatted)
            output_lines.append('')
        else:
            output_lines.append(line)
            i += 1
    
    # Check for GPP lineup
    elif line.startswith('🎯 GPP Lineup'):
        lineup_num = re.search(r'Lineup (\d+)', line)
        if lineup_num:
            gpp_count += 1
            num = lineup_num.group(1)
            # Collect the lineup
            lineup_block = []
            i += 1
            while i < len(lines):
                if lines[i].startswith('💵') or lines[i].startswith('🎯') or lines[i].startswith('======'):
                    break
                lineup_block.append(lines[i])
                if 'Total' in lines[i]:
                    i += 1
                    break
                i += 1
            
            # Format with positions
            formatted = format_lineup_with_positions('\n'.join(lineup_block), num, is_gpp=True)
            output_lines.append(formatted)
            output_lines.append('')
        else:
            output_lines.append(line)
            i += 1
    else:
        output_lines.append(line)
        i += 1

# Write the updated content
output_path = '/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/lineups_with_positions.txt'
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"\n✅ Added positions to all lineups!")
print(f"📄 Processed {cash_count} Cash lineups and {gpp_count} GPP lineups")
print(f"💾 Saved to: {output_path}")
print(f"\n🏀 Lineup positions: PG, SG, SF, PF, C, G, F, UTIL")

