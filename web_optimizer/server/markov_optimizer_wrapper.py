#!/usr/bin/env python3
"""
Full Markov Chain Optimizer Wrapper
Uses the complete optimization logic from makrovchain_optimizer.py
"""

import sys
import json
import os
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import traceback

# Add the optimization directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'optimization'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Import required modules from makrovchain_optimizer
try:
    import pulp
    PULP_AVAILABLE = True
except:
    PULP_AVAILABLE = False
    print("⚠️ PuLP not available", file=sys.stderr)

try:
    from nba_markov_probabilities import apply_markov_adjustments
    MARKOV_AVAILABLE = True
except:
    MARKOV_AVAILABLE = False
    print("⚠️ Markov module not available", file=sys.stderr)

# NBA DraftKings position requirements
POSITION_LIMITS = {
    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
    'G': 1,   # Guard (PG or SG)
    'F': 1,   # Forward (SF or PF)
    'UTIL': 1  # Utility (any position)
}

SALARY_CAP = 50000
GUARD_POSITIONS = ['PG', 'SG']
FORWARD_POSITIONS = ['SF', 'PF']
ALL_POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']


def optimize_single_lineup_pulp(df_players, stack_type='No Stacks', team_selections=None, 
                                  min_salary=48000, max_salary=50000, used_player_ids=None):
    """
    Optimize a single lineup using PuLP (Linear Programming)
    This is the core optimization from makrovchain_optimizer.py
    """
    
    if not PULP_AVAILABLE:
        return None
    
    if used_player_ids is None:
        used_player_ids = set()
    
    # Create the optimization problem
    prob = pulp.LpProblem("NBA_DFS_Lineup", pulp.LpMaximize)
    
    # Decision variables: 1 if player is selected, 0 otherwise
    player_vars = {}
    for idx, row in df_players.iterrows():
        player_id = row.get('id', idx)
        if player_id not in used_player_ids:
            player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    if len(player_vars) < 8:
        return None
    
    # Objective: Maximize total projected points
    proj_col = 'Predicted_DK_Points_MarkovBlend' if 'Predicted_DK_Points_MarkovBlend' in df_players.columns else 'Predicted_DK_Points'
    prob += pulp.lpSum([
        player_vars[idx] * df_players.loc[idx, proj_col]
        for idx in player_vars
    ])
    
    # Constraint 1: Select exactly 8 players
    prob += pulp.lpSum([player_vars[idx] for idx in player_vars]) == 8
    
    # Constraint 2: Salary cap
    prob += pulp.lpSum([
        player_vars[idx] * df_players.loc[idx, 'Salary']
        for idx in player_vars
    ]) <= max_salary
    
    # Constraint 3: Minimum salary
    prob += pulp.lpSum([
        player_vars[idx] * df_players.loc[idx, 'Salary']
        for idx in player_vars
    ]) >= min_salary
    
    # Position constraints
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        eligible = [idx for idx in player_vars if pos in str(df_players.loc[idx, 'Position'])]
        prob += pulp.lpSum([player_vars[idx] for idx in eligible]) >= POSITION_LIMITS[pos]
    
    # Guard constraint (PG or SG for G slot)
    guard_eligible = [idx for idx in player_vars 
                     if any(g in str(df_players.loc[idx, 'Position']) for g in GUARD_POSITIONS)]
    prob += pulp.lpSum([player_vars[idx] for idx in guard_eligible]) >= (
        POSITION_LIMITS['PG'] + POSITION_LIMITS['SG'] + POSITION_LIMITS['G']
    )
    
    # Forward constraint (SF or PF for F slot)
    forward_eligible = [idx for idx in player_vars 
                       if any(f in str(df_players.loc[idx, 'Position']) for f in FORWARD_POSITIONS)]
    prob += pulp.lpSum([player_vars[idx] for idx in forward_eligible]) >= (
        POSITION_LIMITS['SF'] + POSITION_LIMITS['PF'] + POSITION_LIMITS['F']
    )
    
    # Team stacking constraints
    if stack_type != 'No Stacks' and team_selections:
        if stack_type in ['2', '3', '4', '5']:
            stack_size = int(stack_type)
            teams = team_selections.get(stack_type, team_selections.get('all', []))
            if teams:
                # Create binary variable for each team
                team_vars = {}
                for team in teams:
                    team_vars[team] = pulp.LpVariable(f"team_{team}_selected", cat='Binary')
                    team_players = [idx for idx in player_vars 
                                  if df_players.loc[idx, 'Team'] == team]
                    
                    # If team is selected, must have at least stack_size players
                    prob += pulp.lpSum([player_vars[idx] for idx in team_players]) >= stack_size * team_vars[team]
                
                # At least one team must be selected for stacking
                prob += pulp.lpSum([team_vars[team] for team in teams]) >= 1
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        return None
    
    # Extract selected players
    selected_indices = [idx for idx in player_vars if player_vars[idx].varValue > 0.5]
    
    if len(selected_indices) != 8:
        return None
    
    return selected_indices


def assign_roster_positions(selected_players):
    """
    Assign roster positions (PG, SG, SF, PF, C, G, F, UTIL) to selected players
    """
    roster = {}
    used_players = set()
    
    # Fill core positions first
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for player in selected_players:
            if player['id'] not in used_players and pos in player['position']:
                roster[pos] = player.copy()
                roster[pos]['rosterPosition'] = pos
                used_players.add(player['id'])
                break
    
    # Fill G position (any remaining guard)
    for player in selected_players:
        if player['id'] not in used_players:
            if any(g in player['position'] for g in ['PG', 'SG']):
                roster['G'] = player.copy()
                roster['G']['rosterPosition'] = 'G'
                used_players.add(player['id'])
                break
    
    # Fill F position (any remaining forward)
    for player in selected_players:
        if player['id'] not in used_players:
            if any(f in player['position'] for f in ['SF', 'PF']):
                roster['F'] = player.copy()
                roster['F']['rosterPosition'] = 'F'
                used_players.add(player['id'])
                break
    
    # Fill UTIL position (any remaining player)
    for player in selected_players:
        if player['id'] not in used_players:
            roster['UTIL'] = player.copy()
            roster['UTIL']['rosterPosition'] = 'UTIL'
            used_players.add(player['id'])
            break
    
    return list(roster.values())


def optimize_lineups(players_data, num_lineups=10, min_salary=48000, max_salary=50000,
                     unique_players=3, max_exposure=100, stack_settings=None):
    """
    Main optimization function using the full makrovchain_optimizer.py logic
    """
    
    print("🏀 Starting Full Markov Chain Optimization", file=sys.stderr)
    
    # Convert to DataFrame
    df = pd.DataFrame(players_data)
    
    # Validate required columns
    required_cols = ['id', 'name', 'position', 'salary', 'projection']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Rename columns to match makrovchain_optimizer format
    df['Name'] = df['name']
    df['Position'] = df['position']
    df['Salary'] = df['salary']
    df['Predicted_DK_Points'] = df['projection']
    df['Team'] = df.get('team', 'UNK')
    
    # Filter out players with no projection
    df = df[df['Predicted_DK_Points'] > 0].copy()
    
    if len(df) < 8:
        raise ValueError("Not enough players with projections")
    
    print(f"📊 Working with {len(df)} players", file=sys.stderr)
    
    # Apply Markov Chain adjustments (DISABLED to match desktop behavior)
    # The desktop version may not be applying Markov successfully, leading to higher projections
    USE_MARKOV = False  # Set to True to enable Markov blending
    
    if USE_MARKOV and MARKOV_AVAILABLE:
        try:
            cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_historical_cache"
            print("🔮 Applying Markov Chain probability adjustments...", file=sys.stderr)
            
            df = apply_markov_adjustments(
                df_players=df,
                history_df=None,
                cache_dir=cache_dir,
                blend_alpha=0.25,  # 25% Markov, 75% original
                min_games=30,
                player_thresholds=(20.0, 25.0, 30.0),
            )
            
            if 'Predicted_DK_Points_MarkovBlend' in df.columns:
                print("✅ Markov adjustments applied!", file=sys.stderr)
                markov_cols = [c for c in df.columns if 'MC_' in c]
                print(f"   Added columns: {markov_cols}", file=sys.stderr)
            
        except Exception as e:
            print(f"⚠️ Markov adjustment error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    else:
        print("ℹ️ Using raw projections (Markov disabled)", file=sys.stderr)
    
    # Determine stack types
    stack_types = ['No Stacks']
    team_selections = {}
    
    if stack_settings and isinstance(stack_settings, dict):
        if stack_settings.get('enabled'):
            stack_type = stack_settings.get('type', 'No Stacks')
            if stack_type != 'No Stacks':
                # Extract stack size (e.g., "3 Players Same Team" -> "3")
                import re
                match = re.search(r'(\d+)', stack_type)
                if match:
                    stack_types = [match.group(1)]
                    # Get unique teams
                    teams = df['Team'].unique().tolist()
                    team_selections[match.group(1)] = teams
                    print(f"🎯 Stack mode: {stack_type} from teams: {teams}", file=sys.stderr)
    
    print(f"🎲 Generating {num_lineups} lineups using PuLP optimization", file=sys.stderr)
    
    lineups = []
    exposure_tracker = {}
    used_lineup_keys = set()
    
    # Generate lineups with randomization for diversity
    attempts = 0
    max_attempts = num_lineups * 50  # Increased attempts
    
    while len(lineups) < num_lineups and attempts < max_attempts:
        attempts += 1
        
        # Rotate through stack types
        stack_type = stack_types[len(lineups) % len(stack_types)]
        
        # Track which players to exclude (over-exposed)
        max_lineup_exposure = int((max_exposure / 100.0) * num_lineups)
        excluded_ids = {pid for pid, count in exposure_tracker.items() 
                       if count >= max_lineup_exposure}
        
        # Add randomization to force diversity
        # Randomly exclude some highly-used players to create variety
        if len(lineups) > 0:
            # After first lineup, randomly exclude 1-3 top players occasionally
            if np.random.random() < 0.4:  # 40% chance
                top_exposed = sorted(exposure_tracker.items(), key=lambda x: x[1], reverse=True)[:3]
                num_to_exclude = np.random.randint(1, min(3, len(top_exposed)) + 1)
                for pid, _ in top_exposed[:num_to_exclude]:
                    excluded_ids.add(pid)
        
        # Optimize lineup using PuLP
        selected_indices = optimize_single_lineup_pulp(
            df, 
            stack_type=stack_type,
            team_selections=team_selections,
            min_salary=min_salary,
            max_salary=max_salary,
            used_player_ids=excluded_ids
        )
        
        if selected_indices is None:
            continue
        
        # Get selected players
        selected_players = []
        for idx in selected_indices:
            player = df.loc[idx].to_dict()
            # Convert NaN to None for JSON serialization
            player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
            selected_players.append(player)
        
        # Check uniqueness
        lineup_key = '|'.join(sorted([p['id'] for p in selected_players]))
        
        if lineup_key in used_lineup_keys:
            continue
        
        # Relaxed uniqueness check - only for first 5 lineups, then relax further
        min_unique = unique_players if len(lineups) < 5 else max(2, unique_players - 1)
        
        is_unique = True
        for existing_key in used_lineup_keys:
            existing_ids = set(existing_key.split('|'))
            new_ids = set(lineup_key.split('|'))
            common = len(existing_ids & new_ids)
            if common > (8 - min_unique):
                is_unique = False
                break
        
        if not is_unique:
            continue
        
        # Assign roster positions
        roster_players = assign_roster_positions(selected_players)
        
        if len(roster_players) != 8:
            continue
        
        # Calculate totals
        total_salary = sum(p['Salary'] for p in roster_players)
        proj_col = 'Predicted_DK_Points_MarkovBlend' if 'Predicted_DK_Points_MarkovBlend' in df.columns else 'Predicted_DK_Points'
        total_projection = sum(p.get(proj_col, p['Predicted_DK_Points']) for p in roster_players)
        
        # Update exposure
        for player in roster_players:
            exposure_tracker[player['id']] = exposure_tracker.get(player['id'], 0) + 1
        
        # Add lineup
        used_lineup_keys.add(lineup_key)
        lineups.append({
            'players': roster_players,
            'totalSalary': int(total_salary),
            'totalProjection': round(float(total_projection), 2),
            'strategy': f'pulp_stack_{stack_type}'
        })
        
        print(f"  ✓ Lineup {len(lineups)}: {total_projection:.1f} pts, ${total_salary}", file=sys.stderr)
    
    print(f"✅ Generated {len(lineups)} optimized lineups", file=sys.stderr)
    
    return lineups


def main():
    """Main CLI entry point"""
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input data provided"}))
        sys.exit(1)
    
    try:
        # Read input from command line argument
        input_data = json.loads(sys.argv[1])
        
        players = input_data.get('players', [])
        num_lineups = input_data.get('numLineups', 10)
        min_salary = input_data.get('minSalary', 48000)
        max_salary = input_data.get('maxSalary', 50000)
        unique_players = input_data.get('uniquePlayers', 3)
        max_exposure = input_data.get('maxExposure', 100)
        stack_settings = input_data.get('stackSettings', {})
        
        # Run optimization
        lineups = optimize_lineups(
            players,
            num_lineups,
            min_salary,
            max_salary,
            unique_players,
            max_exposure,
            stack_settings
        )
        
        # Output result
        output = {
            "success": True,
            "lineups": lineups,
            "summary": {
                "totalLineups": len(lineups),
                "avgProjection": round(np.mean([l['totalProjection'] for l in lineups]), 2) if lineups else 0,
                "avgSalary": round(np.mean([l['totalSalary'] for l in lineups]), 2) if lineups else 0,
                "topProjection": round(max([l['totalProjection'] for l in lineups]), 2) if lineups else 0,
            }
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

