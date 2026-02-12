#!/usr/bin/env python3
"""
Test script to verify multi-stack-size support works
(e.g., 3-stack from LAL + 2-stack from BOS simultaneously)
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from makrov_cli_adapter import main
import io
from contextlib import redirect_stderr, redirect_stdout

# Create test data with players from different teams
test_players = [
    # Team LAL players (need at least 3 for 3-stack)
    {"name": "LeBron James", "position": "SF", "team": "LAL", "salary": 10000, "projection": 50.0},
    {"name": "Anthony Davis", "position": "PF", "team": "LAL", "salary": 9500, "projection": 48.0},
    {"name": "Russell Westbrook", "position": "PG", "team": "LAL", "salary": 8000, "projection": 40.0},
    {"name": "LAL Player 4", "position": "SG", "team": "LAL", "salary": 6000, "projection": 30.0},
    {"name": "LAL Player 5", "position": "C", "team": "LAL", "salary": 5000, "projection": 25.0},

    # Team BOS players (need at least 2 for 2-stack)
    {"name": "Jayson Tatum", "position": "SF", "team": "BOS", "salary": 9000, "projection": 45.0},
    {"name": "Jaylen Brown", "position": "SG", "team": "BOS", "salary": 8500, "projection": 42.0},
    {"name": "BOS Player 3", "position": "PG", "team": "BOS", "salary": 7000, "projection": 35.0},
    {"name": "BOS Player 4", "position": "PF", "team": "BOS", "salary": 6000, "projection": 28.0},
    {"name": "BOS Player 5", "position": "C", "team": "BOS", "salary": 5000, "projection": 22.0},

    # Team GSW players
    {"name": "Stephen Curry", "position": "PG", "team": "GSW", "salary": 10000, "projection": 50.0},
    {"name": "Klay Thompson", "position": "SG", "team": "GSW", "salary": 7500, "projection": 38.0},
    {"name": "Draymond Green", "position": "PF", "team": "GSW", "salary": 6500, "projection": 32.0},

    # Filler players
    {"name": "Player X1", "position": "PG", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X2", "position": "SG", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X3", "position": "SF", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X4", "position": "PF", "team": "MIA", "salary": 5000, "projection": 20.0},
    {"name": "Player X5", "position": "C", "team": "MIA", "salary": 5000, "projection": 20.0},
]

# Test: 3-stack from LAL AND 2-stack from BOS simultaneously
test_input = {
    "players": test_players,
    "numLineups": 6,
    "minSalary": 48000,
    "maxSalary": 50000,
    "stackSettings": {
        "enabled": True,
        "teams": ["LAL", "BOS"],
        "minPlayersPerTeam": 2,
        "maxPlayersPerTeam": 3
    },
    "teamSelections": {
        "3": ["LAL"],    # 3-stack for LAL
        "2": ["BOS"],    # 2-stack for BOS
        "all": []
    },
    "maxExposure": 100
}

stderr_capture = io.StringIO()
stdout_capture = io.StringIO()

original_argv = sys.argv
sys.argv = ['test', json.dumps(test_input)]

try:
    with redirect_stderr(stderr_capture), redirect_stdout(stdout_capture):
        main()
finally:
    sys.argv = original_argv

stderr_output = stderr_capture.getvalue()
stdout_output = stdout_capture.getvalue()

print("=" * 80)
print("MULTI-STACK-SIZE TEST: 3-stack LAL + 2-stack BOS")
print("=" * 80)

# Parse and check results
try:
    result = json.loads(stdout_output)
    if result.get('success'):
        lineups = result.get('lineups', [])
        warnings = result.get('warnings', [])
        print(f"\nGenerated {len(lineups)} lineups (requested: 6)")

        if warnings:
            print(f"\nWarnings: {warnings}")

        lal_3stack_count = 0
        bos_2stack_count = 0

        for i, lineup in enumerate(lineups):
            players = lineup.get('players', [])
            team_counts = {}
            for player in players:
                team = player.get('team', 'UNK')
                team_counts[team] = team_counts.get(team, 0) + 1

            strategy = lineup.get('strategy', '')
            lal_count = team_counts.get('LAL', 0)
            bos_count = team_counts.get('BOS', 0)

            print(f"\nLineup {i+1} ({strategy}):")
            print(f"  Team distribution: {team_counts}")

            if '3' in strategy:
                if lal_count >= 3:
                    print(f"  PASS: LAL has {lal_count} players (required: 3)")
                    lal_3stack_count += 1
                else:
                    print(f"  FAIL: LAL has {lal_count} players (required: 3)")
            elif '2' in strategy:
                if bos_count >= 2:
                    print(f"  PASS: BOS has {bos_count} players (required: 2)")
                    bos_2stack_count += 1
                else:
                    print(f"  FAIL: BOS has {bos_count} players (required: 2)")

        print(f"\n{'=' * 80}")
        print(f"RESULTS:")
        print(f"  LAL 3-stack lineups: {lal_3stack_count}")
        print(f"  BOS 2-stack lineups: {bos_2stack_count}")

        if lal_3stack_count > 0 and bos_2stack_count > 0:
            print(f"\n  ALL TESTS PASSED! Both stack sizes were generated.")
        elif lal_3stack_count > 0:
            print(f"\n  PARTIAL: LAL 3-stacks generated but no BOS 2-stacks")
        elif bos_2stack_count > 0:
            print(f"\n  PARTIAL: BOS 2-stacks generated but no LAL 3-stacks")
        else:
            print(f"\n  FAIL: Neither stack size constraint was enforced")
    else:
        print(f"\nOptimization failed: {result.get('error', 'Unknown error')}")
except json.JSONDecodeError as e:
    print(f"\nFailed to parse JSON output: {e}")
