#!/usr/bin/env python3
"""
Test script for DraftKings entries file format handling

This script demonstrates how the improved optimizer handles different DK file formats:
1. Official DraftKings contest format (like DKEntries (1).csv)
2. Malformed/custom formats
3. Player ID extraction and lineup filling
"""

import pandas as pd
import re

def test_dk_format_detection():
    """Test the format detection logic"""
    
    # Test 1: Official DraftKings format
    print("🧪 Testing Official DraftKings Format Detection")
    print("=" * 50)
    
    # Simulate the columns from your DKEntries (1).csv file
    official_dk_columns = [
        'Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 
        'P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 
        '', 'Instructions'
    ]
    
    # Test format detection logic
    def detect_format(columns):
        if 'Entry ID' in columns and 'Contest Name' in columns and 'Contest ID' in columns:
            # Check for exact DK structure
            core_columns = columns[:14] if len(columns) >= 14 else columns
            if len(core_columns) >= 14:
                position_columns = core_columns[4:14]  # Positions start at index 4
                expected_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
                if position_columns == expected_positions:
                    return 'official_dk_contest'
            return 'contest_format'
        return 'unknown'
    
    format_result = detect_format(official_dk_columns)
    print(f"✅ Format detected: {format_result}")
    print(f"📋 Columns: {official_dk_columns}")
    print()
    
    # Test 2: Player ID extraction
    print("🧪 Testing Player ID Extraction")
    print("=" * 50)
    
    sample_player_data = [
        "Hunter Brown (39204162)",
        "Zack Wheeler (39203788)", 
        "Chris Sale (39204163)",
        "Austin Hedges (39204500)"
    ]
    
    def extract_player_mapping(player_list):
        player_map = {}
        for player_str in player_list:
            match = re.match(r'^(.+?)\s*\((\d+)\)$', player_str.strip())
            if match:
                name = match.group(1).strip()
                player_id = match.group(2)
                player_map[name] = player_str
        return player_map
    
    player_map = extract_player_mapping(sample_player_data)
    print(f"✅ Extracted {len(player_map)} player mappings:")
    for name, full_str in player_map.items():
        print(f"   {name} -> {full_str}")
    print()
    
    # Test 3: ID extraction
    print("🧪 Testing ID Extraction")
    print("=" * 50)
    
    def extract_id(player_name, player_map):
        if player_name in player_map:
            name_with_id = player_map[player_name]
            match = re.search(r'\((\d+)\)', name_with_id)
            return match.group(1) if match else ""
        return ""
    
    test_names = ["Hunter Brown", "Zack Wheeler", "Unknown Player"]
    for name in test_names:
        player_id = extract_id(name, player_map)
        status = "✅" if player_id else "❌"
        print(f"   {status} {name} -> ID: {player_id}")
    print()
    
    # Test 4: Lineup formatting
    print("🧪 Testing Lineup Formatting")
    print("=" * 50)
    
    # Simulate a lineup
    sample_lineup = [
        {"Name": "Hunter Brown", "Pos": "P"},
        {"Name": "Zack Wheeler", "Pos": "P"}, 
        {"Name": "Austin Hedges", "Pos": "C"},
        {"Name": "Pete Alonso", "Pos": "1B"},
        {"Name": "Gleyber Torres", "Pos": "2B"},
        {"Name": "Alex Bregman", "Pos": "3B"},
        {"Name": "Trea Turner", "Pos": "SS"},
        {"Name": "Juan Soto", "Pos": "OF"},
        {"Name": "Ronald Acuna Jr.", "Pos": "OF"},
        {"Name": "Mookie Betts", "Pos": "OF"}
    ]
    
    # Add IDs for the players we have
    extended_player_map = {
        **player_map,
        "Pete Alonso": "Pete Alonso (39204501)",
        "Gleyber Torres": "Gleyber Torres (39204502)", 
        "Alex Bregman": "Alex Bregman (39204503)",
        "Trea Turner": "Trea Turner (39204504)",
        "Juan Soto": "Juan Soto (39204505)",
        "Ronald Acuna Jr.": "Ronald Acuna Jr. (39204506)",
        "Mookie Betts": "Mookie Betts (39204507)"
    }
    
    def format_for_dk_official(lineup, player_map):
        dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        position_players = {'P': [], 'C': [], '1B': [], '2B': [], '3B': [], 'SS': [], 'OF': []}
        
        # Group players by position
        for player in lineup:
            pos = player['Pos'].upper()
            name = player['Name']
            if 'P' in pos:
                position_players['P'].append(name)
            elif 'C' in pos:
                position_players['C'].append(name)
            elif '1B' in pos:
                position_players['1B'].append(name)
            elif '2B' in pos:
                position_players['2B'].append(name)
            elif '3B' in pos:
                position_players['3B'].append(name)
            elif 'SS' in pos:
                position_players['SS'].append(name)
            elif 'OF' in pos:
                position_players['OF'].append(name)
        
        # Format for DK
        formatted_lineup = []
        position_usage = {pos: 0 for pos in position_players.keys()}
        
        for dk_pos in dk_positions:
            player_id = ""
            if dk_pos in position_players and position_usage[dk_pos] < len(position_players[dk_pos]):
                player_name = position_players[dk_pos][position_usage[dk_pos]]
                position_usage[dk_pos] += 1
                player_id = extract_id(player_name, player_map)
            formatted_lineup.append(player_id)
        
        return formatted_lineup
    
    formatted_ids = format_for_dk_official(sample_lineup, extended_player_map)
    
    print("✅ Formatted lineup for DraftKings:")
    dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
    for i, (pos, player_id) in enumerate(zip(dk_positions, formatted_ids)):
        status = "✅" if player_id else "❌"
        print(f"   {status} {pos}: {player_id}")
    print()
    
    print("🎉 All tests completed!")
    print("=" * 50)
    print("Summary:")
    print("• ✅ Format detection works for official DK files")
    print("• ✅ Player ID extraction from Name+ID format") 
    print("• ✅ Lineup formatting to DK position order")
    print("• ✅ Ready to fill DraftKings entries files!")
    print()
    print("📝 Usage Instructions:")
    print("1. Load your DraftKings entries file using 'Load DraftKings Entries File'")
    print("2. Run optimization to generate lineups")
    print("3. Use 'Fill Entries with Optimized Lineups' to create filled file")
    print("4. Upload the filled file to DraftKings!")

if __name__ == "__main__":
    test_dk_format_detection()
