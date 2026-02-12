#!/usr/bin/env python3
"""
Test to simulate what the web UI sends
"""

import json
import sys
import os

# Simulate the exact format the web UI sends
web_input = {
    "players": [
        {"name": "Player 1", "position": "PG", "team": "LAL", "salary": 8000, "projection": 40.0},
        {"name": "Player 2", "position": "SG", "team": "LAL", "salary": 7000, "projection": 35.0},
        {"name": "Player 3", "position": "SF", "team": "LAL", "salary": 6000, "projection": 30.0},
        {"name": "Player 4", "position": "PF", "team": "LAL", "salary": 5000, "projection": 25.0},
        {"name": "Player 5", "position": "C", "team": "LAL", "salary": 5000, "projection": 20.0},
        {"name": "Player 6", "position": "PG", "team": "BOS", "salary": 8000, "projection": 38.0},
        {"name": "Player 7", "position": "SG", "team": "BOS", "salary": 7000, "projection": 33.0},
        {"name": "Player 8", "position": "SF", "team": "BOS", "salary": 6000, "projection": 28.0},
        {"name": "Player 9", "position": "PF", "team": "BOS", "salary": 5000, "projection": 23.0},
        {"name": "Player 10", "position": "C", "team": "BOS", "salary": 5000, "projection": 18.0},
        {"name": "Player 11", "position": "PG", "team": "GSW", "salary": 9000, "projection": 45.0},
        {"name": "Player 12", "position": "SG", "team": "GSW", "salary": 7500, "projection": 40.0},
        {"name": "Player 13", "position": "SF", "team": "MIA", "salary": 5000, "projection": 20.0},
        {"name": "Player 14", "position": "PF", "team": "MIA", "salary": 5000, "projection": 20.0},
        {"name": "Player 15", "position": "C", "team": "MIA", "salary": 5000, "projection": 20.0},
        {"name": "Player 16", "position": "PG", "team": "MIA", "salary": 5000, "projection": 20.0},
    ],
    "numLineups": 10,
    "minSalary": 48000,
    "maxSalary": 50000,
    "stackSettings": {
        "enabled": True,
        "teams": ["LAL", "BOS"],
        "minPlayersPerTeam": 3,
        "maxPlayersPerTeam": 3
    },
    "teamSelections": {
        "3": ["LAL", "BOS"],  # This is what the frontend sends
        "all": ["LAL", "BOS"]
    },
    "maxExposure": 100
}

# Test the parsing logic
print("=" * 80)
print("Testing team_selections parsing:")
print("=" * 80)
print(f"Input teamSelections: {web_input['teamSelections']}")
print()

# Simulate the parsing logic from makrov_cli_adapter.py
stack_type = "3"
team_selections = web_input['teamSelections']

teams_for_stack = []
if stack_type in team_selections:
    teams_for_stack = team_selections[stack_type]
    print(f"✅ Found using key '{stack_type}': {teams_for_stack}")
elif str(stack_type) in team_selections:
    teams_for_stack = team_selections[str(stack_type)]
    print(f"✅ Found using key '{str(stack_type)}': {teams_for_stack}")
elif int(stack_type) in team_selections if stack_type.isdigit() else False:
    teams_for_stack = team_selections[int(stack_type)]
    print(f"✅ Found using key {int(stack_type)}: {teams_for_stack}")
elif 'all' in team_selections:
    teams_for_stack = team_selections['all']
    print(f"✅ Found using key 'all': {teams_for_stack}")

print()
print(f"Teams for stack: {teams_for_stack}")
print(f"Multiple teams? {len(teams_for_stack) > 1 if teams_for_stack else False}")
print()

if teams_for_stack and len(teams_for_stack) > 1:
    print("✅ Multiple teams detected - should generate lineups for EACH team")
    print(f"   Will generate ~{10 // len(teams_for_stack)} lineups per team")
else:
    print("❌ Only single team or no teams - will use binary variable logic")





