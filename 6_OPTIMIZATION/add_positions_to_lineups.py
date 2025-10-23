#!/usr/bin/env python3
"""
Add NBA positions (PG, SG, SF, PF, C, G, F, UTIL) to lineups in lineup.txt.md
"""

import pandas as pd
import re

# Load the player pool to get position eligibility
player_pool_path = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv"
df_players = pd.read_csv(player_pool_path)

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

def assign_lineup_positions(players):
    """
    Assign players to NBA lineup positions: PG, SG, SF, PF, C, G, F, UTIL
    """
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    assigned = {}
    remaining_players = list(players)
    
    # First pass: Assign players to their primary positions
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for player in remaining_players[:]:
            if player in player_positions:
                eligible_positions = player_positions[player]
                # Check if player is eligible for this specific position
                if pos in eligible_positions:
                    assigned[pos] = player
                    remaining_players.remove(player)
                    break
    
    # Second pass: Fill G position (eligible: PG or SG)
    if 'G' not in assigned:
        for player in remaining_players[:]:
            if player in player_positions:
                eligible_positions = player_positions[player]
                if 'PG' in eligible_positions or 'SG' in eligible_positions or 'G' in eligible_positions:
                    assigned['G'] = player
                    remaining_players.remove(player)
                    break
    
    # Third pass: Fill F position (eligible: SF or PF)
    if 'F' not in assigned:
        for player in remaining_players[:]:
            if player in player_positions:
                eligible_positions = player_positions[player]
                if 'SF' in eligible_positions or 'PF' in eligible_positions or 'F' in eligible_positions:
                    assigned['F'] = player
                    remaining_players.remove(player)
                    break
    
    # Fourth pass: Fill UTIL with anyone remaining
    if 'UTIL' not in assigned and remaining_players:
        assigned['UTIL'] = remaining_players[0]
        remaining_players.remove(remaining_players[0])
    
    return assigned

def format_lineup_with_positions(lineup_text, lineup_number, is_gpp=False):
    """
    Parse a lineup and add position labels
    """
    lines = lineup_text.strip().split('\n')
    
    # Extract player names (skip header and total lines)
    players = []
    for line in lines:
        # Skip headers and totals
        if 'Name' in line or 'Total' in line or '===' in line or '---' in line:
            continue
        # Extract player name (first column, may have spaces)
        match = re.match(r'^\s*([A-Za-z\'\.\-\s]+?)\s+([\w]{2,4})\s+\d', line)
        if match:
            player_name = match.group(1).strip()
            players.append(player_name)
    
    if not players:
        return lineup_text
    
    # Assign positions
    position_assignment = assign_lineup_positions(players)
    
    # Rebuild lineup with positions
    result_lines = []
    if is_gpp:
        result_lines.append(f"🎯 GPP Lineup {lineup_number} (with positions):")
        result_lines.append("Pos  Name                 Team  Salary  Ceiling  Est_Ownership")
        result_lines.append("-" * 70)
    else:
        result_lines.append(f"💵 Cash Lineup {lineup_number} (with positions):")
        result_lines.append("Pos  Name                 Team  Salary  Floor  Projected_DK_Points")
        result_lines.append("-" * 70)
    
    # Add players in position order
    nba_positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    for pos in nba_positions:
        if pos in position_assignment:
            player_name = position_assignment[pos]
            # Find the original line for this player
            for line in lines:
                if player_name in line and 'Name' not in line and 'Total' not in line:
                    # Add position at the start
                    result_lines.append(f"{pos:4} {line.strip()}")
                    break
    
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
while i < len(lines):
    line = lines[i]
    
    # Check for Cash lineup
    if line.startswith('💵 Cash Lineup'):
        lineup_num = re.search(r'Lineup (\d+)', line)
        if lineup_num:
            num = lineup_num.group(1)
            # Collect the lineup (next ~10 lines until we hit Total or empty line)
            lineup_block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('💵') and not lines[i].startswith('🎯') and not lines[i].startswith('==='):
                lineup_block.append(lines[i])
                if 'Total' in lines[i]:
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
            num = lineup_num.group(1)
            # Collect the lineup
            lineup_block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('💵') and not lines[i].startswith('🎯') and not lines[i].startswith('==='):
                lineup_block.append(lines[i])
                if 'Total' in lines[i]:
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

print(f"✅ Added positions to all lineups!")
print(f"📄 Saved to: {output_path}")
print(f"\n🏀 Lineup positions: PG, SG, SF, PF, C, G, F, UTIL")

