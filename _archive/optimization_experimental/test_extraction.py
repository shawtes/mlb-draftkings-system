#!/usr/bin/env python3
"""Test DK entries extraction"""
import re

def extract_player_ids(file_path):
    """Extract player IDs from DKEntries.csv"""
    player_map = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # Look for pattern: "Player Name (ID)"
            matches = re.findall(r'([A-Za-z][A-Za-z\s\.\-\']+)\s*\((\d{6,})\)', line)
            
            for name_part, id_part in matches:
                name_part = name_part.strip()
                id_part = id_part.strip()
                
                # Validation
                if (len(name_part) > 3 and 
                    len(name_part.split()) >= 2 and 
                    len(id_part) >= 6):
                    
                    if name_part not in player_map:
                        player_map[name_part] = id_part
                        if len(player_map) <= 10:
                            print(f"✅ {name_part} -> {id_part}")
    
    print(f"\n🎯 Total extracted: {len(player_map)} players")
    return player_map

if __name__ == "__main__":
    dk_file = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/DKEntries.csv"
    player_map = extract_player_ids(dk_file)
    
    # Test lookups
    test_names = ["De'Von Achane", "Patrick Mahomes", "Saquon Barkley", "Jonathan Taylor"]
    print("\n🔍 Testing lookups:")
    for name in test_names:
        if name in player_map:
            print(f"  ✅ {name} -> {player_map[name]}")
        else:
            print(f"  ❌ {name} -> NOT FOUND")


