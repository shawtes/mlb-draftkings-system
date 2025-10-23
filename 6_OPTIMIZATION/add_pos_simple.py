#!/usr/bin/env python3
"""
Simple solution: Add position labels (PG, SG, SF, PF, C, G, F, UTIL) to each lineup
in order, based on the 8 players shown.
"""

with open('/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/lineup.txt.md', 'r') as f:
    lines = f.readlines()

output = []
positions = ['PG  ', 'SG  ', 'SF  ', 'PF  ', 'C   ', 'G   ', 'F   ', 'UTIL']
pos_idx = 0
in_lineup = False

for i, line in enumerate(lines):
    # Check if we're starting a new lineup
    if '💵 Cash Lineup' in line or '🎯 GPP Lineup' in line:
        in_lineup = True
        pos_idx = 0
        output.append(line)
        continue
    
    # Check if this is a header line with "Name"
    if in_lineup and 'Name' in line and 'Team' in line:
        # Add "Pos" to the header
        output.append('Pos  ' + line)
        continue
    
    # Check if this is a player line (has numbers for salary/points)
    if in_lineup and line.strip() and not line.startswith('Total') and not line.startswith('===') and not line.startswith('---'):
        # Check if line has digits (salary) - simple heuristic
        if any(char.isdigit() for char in line) and pos_idx < 8:
            # Prepend position
            output.append(positions[pos_idx] + line)
            pos_idx += 1
            continue
    
    # Check if lineup is ending
    if 'Total' in line:
        in_lineup = False
        pos_idx = 0
    
    output.append(line)

# Write output
with open('/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/lineups_with_positions.txt', 'w') as f:
    f.writelines(output)

print("✅ Added positions to all lineups!")
print("📄 Saved to: lineups_with_positions.txt")
print("🏀 Positions: PG, SG, SF, PF, C, G, F, UTIL")

