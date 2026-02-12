#!/usr/bin/env python3
"""
Test script to verify 3-stack constraint is working
"""

import json
import sys
import os

# Add the server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

# Import the adapter
from makrov_cli_adapter import main
import io
from contextlib import redirect_stderr, redirect_stdout

# Create test data with players from different teams
test_players = [
    # Team LAL players (need at least 3)
    {"name": "LeBron James", "position": "SF", "team": "LAL", "salary": 10000, "projection": 50.0},
    {"name": "Anthony Davis", "position": "PF", "team": "LAL", "salary": 9500, "projection": 48.0},
    {"name": "Russell Westbrook", "position": "PG", "team": "LAL", "salary": 8000, "projection": 40.0},
    {"name": "LAL Player 4", "position": "SG", "team": "LAL", "salary": 6000, "projection": 30.0},
    {"name": "LAL Player 5", "position": "C", "team": "LAL", "salary": 5000, "projection": 25.0},
    
    # Team BOS players
    {"name": "Jayson Tatum", "position": "SF", "team": "BOS", "salary": 9000, "projection": 45.0},
    {"name": "Jaylen Brown", "position": "SG", "team": "BOS", "salary": 8500, "projection": 42.0},
    {"name": "BOS Player 3", "position": "PG", "team": "BOS", "salary": 7000, "projection": 35.0},
    
    # Team GSW players
    {"name": "Stephen Curry", "position": "PG", "team": "GSW", "salary": 10000, "projection": 50.0},
    {"name": "Klay Thompson", "position": "SG", "team": "GSW", "salary": 7500, "projection": 38.0},
    {"name": "Draymond Green", "position": "PF", "team": "GSW", "salary": 6500, "projection": 32.0},
    
    # More players to fill positions
    {"name": "Player X1", "position": "PG", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X2", "position": "SG", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X3", "position": "SF", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X4", "position": "PF", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X5", "position": "C", "team": "MIA", "salary": 5000, "projection": 20.0},
]

# Test input with 3-stack for LAL
test_input = {
    "players": test_players,
    "numLineups": 3,
    "minSalary": 48000,
    "maxSalary": 50000,
    "stackSettings": {
        "enabled": True,
        "teams": ["LAL"],
        "minPlayersPerTeam": 3,
        "maxPlayersPerTeam": 3
    },
    "teamSelections": {
        "3": ["LAL"],  # 3-stack for LAL
        "all": ["LAL"]
    },
    "maxExposure": 100
}

# Capture stderr to see debug output
stderr_capture = io.StringIO()
stdout_capture = io.StringIO()

# Mock sys.argv
original_argv = sys.argv
sys.argv = ['test', json.dumps(test_input)]

try:
    with redirect_stderr(stderr_capture), redirect_stdout(stdout_capture):
        main()
finally:
    sys.argv = original_argv

# Get outputs
stderr_output = stderr_capture.getvalue()
stdout_output = stdout_capture.getvalue()

print("=" * 80)
print("STDERR OUTPUT (Debug logs):")
print("=" * 80)
print(stderr_output)

print("\n" + "=" * 80)
print("STDOUT OUTPUT (JSON result):")
print("=" * 80)
print(stdout_output)

# Parse and check results
try:
    result = json.loads(stdout_output)
    if result.get('success'):
        lineups = result.get('lineups', [])
        print(f"\n✅ Generated {len(lineups)} lineups")
        
        # Check each lineup for 3-stack constraint
        for i, lineup in enumerate(lineups):
            players = lineup.get('players', [])
            team_counts = {}
            for player in players:
                team = player.get('team', 'UNK')
                team_counts[team] = team_counts.get(team, 0) + 1
            
            print(f"\nLineup {i+1}:")
            print(f"  Team distribution: {team_counts}")
            
            # Check if LAL has at least 3 players
            lal_count = team_counts.get('LAL', 0)
            if lal_count >= 3:
                print(f"  ✅ PASS: LAL has {lal_count} players (required: 3)")
            else:
                print(f"  ❌ FAIL: LAL has {lal_count} players (required: 3)")
    else:
        print(f"\n❌ Optimization failed: {result.get('error', 'Unknown error')}")
except json.JSONDecodeError as e:
    print(f"\n❌ Failed to parse JSON output: {e}")
    print(f"Raw output: {stdout_output[:500]}")





