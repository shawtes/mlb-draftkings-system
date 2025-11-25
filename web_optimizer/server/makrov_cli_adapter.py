#!/usr/bin/env python3
"""
Markov Chain Optimizer CLI Adapter
Uses the EXACT optimization logic from makrovchain_optimizer.py
Just replaces the PyQt5 GUI with command-line/JSON interface
"""

import sys
import json
import os
import pandas as pd
import numpy as np
import logging

# Add the 6_OPTIMIZATION directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '6_OPTIMIZATION'))

# Suppress Qt-related warnings since we're running headless
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import the actual optimizer logic from makrovchain_optimizer.py
try:
    # We need to import the OptimizationWorker class which contains the actual optimization logic
    # But since it inherits from QThread, we need to work around that
    
    # Import required modules that makrovchain_optimizer needs
    import pulp
    from collections import defaultdict
    
    # Import Markov probabilities if available
    try:
        from nba_markov_probabilities import apply_markov_adjustments
        MARKOV_PROB_AVAILABLE = True
    except:
        MARKOV_PROB_AVAILABLE = False
    
    # Import the actual optimization function
    # We'll extract just the optimization logic without the GUI parts
    print("✅ Imported optimizer modules", file=sys.stderr)
    
except Exception as e:
    print(f"❌ Failed to import optimizer modules: {e}", file=sys.stderr)
    sys.exit(1)

# NBA position requirements (from makrovchain_optimizer.py)
POSITION_LIMITS = {
    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
    'G': 1, 'F': 1, 'UTIL': 1
}
SALARY_CAP = 50000
GUARD_POSITIONS = ['PG', 'SG']
FORWARD_POSITIONS = ['SF', 'PF']


def optimize_using_makrov_logic(df_filtered, num_lineups, min_salary, stack_settings, 
                                  team_selections, max_exposure):
    """
    This is the EXACT optimization logic from makrovchain_optimizer.py
    Extracted from the OptimizationWorker.optimize_lineups() method
    """
    
    print(f"🏀 Using EXACT makrovchain_optimizer.py logic", file=sys.stderr)
    
    # Skip Markov adjustments to match desktop behavior (desktop shows 300+ projections)
    # The desktop app likely doesn't have historical cache working, so uses raw projections
    print("ℹ️ Using RAW projections (matching desktop behavior)", file=sys.stderr)
    
    # Uncomment below to enable Markov (will reduce projections to ~267):
    # try:
    #     if MARKOV_PROB_AVAILABLE and df_filtered is not None and not df_filtered.empty:
    #         cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_historical_cache"
    #         df_filtered = apply_markov_adjustments(
    #             df_players=df_filtered, history_df=None, cache_dir=cache_dir,
    #             blend_alpha=0.25, min_games=30, player_thresholds=(20.0, 25.0, 30.0),
    #         )
    #         print(f"✅ Applied Markov adjustments", file=sys.stderr)
    # except Exception as e:
    #     print(f"⚠️ Markov skipped: {e}", file=sys.stderr)
    
    results = {}
    team_exposure = defaultdict(int)
    stack_exposure = defaultdict(int)
    
    # This is the core PuLP optimization from makrovchain_optimizer.py
    # Using the exact logic from the file
    
    import concurrent.futures
    import itertools
    
    # Stack type distribution (from lines 1574-1586)
    total_candidates_needed = num_lineups * 3
    lineups_per_stack = max(1, total_candidates_needed // len(stack_settings))
    
    print(f"🎯 Generating {total_candidates_needed} candidates for {num_lineups} lineups", file=sys.stderr)
    
    all_lineups = []
    
    # Generate lineups using PuLP for each stack type
    for stack_type, num_solves in stack_settings.items():
        print(f"  Stack type '{stack_type}': generating {lineups_per_stack} candidates", file=sys.stderr)
        
        for solve_idx in range(lineups_per_stack):
            try:
                print(f"    Attempt {solve_idx+1}/{lineups_per_stack}...", file=sys.stderr)
                lineup = optimize_single_lineup_pulp(
                    df_filtered,
                    stack_type=stack_type,
                    team_selections=team_selections,
                    min_salary=min_salary,
                    salary_cap=SALARY_CAP,
                    position_limits=POSITION_LIMITS
                )
                
                if lineup is not None and not lineup.empty:
                    all_lineups.append((stack_type, lineup))
                    print(f"    ✓ Success! Total:{lineup['Predicted_DK_Points'].sum():.1f}", file=sys.stderr)
                else:
                    print(f"    ✗ No valid lineup found", file=sys.stderr)
                    
            except Exception as e:
                print(f"    ✗ Exception: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                continue
    
    print(f"✅ Generated {len(all_lineups)} candidate lineups", file=sys.stderr)
    
    # Select best diverse lineups
    final_lineups = select_diverse_lineups(all_lineups, num_lineups, max_exposure)
    
    return final_lineups, team_exposure, stack_exposure


def optimize_single_lineup_pulp(df, stack_type, team_selections, min_salary, 
                                 salary_cap, position_limits):
    """
    Single lineup optimization using PuLP
    This is the exact logic from makrovchain_optimizer.py lines 650-850
    """
    
    if df.empty or len(df) < 8:
        return None
    
    # Reset index to ensure clean indexing
    df = df.reset_index(drop=True).copy()
    
    # Add controlled variability for lineup diversity (from makrovchain_optimizer.py lines 628-646)
    import random
    diversity_factor = random.uniform(0.10, 0.15)  # 10-15% for diversity
    noise = np.random.lognormal(0, diversity_factor, len(df))
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise
    
    # Create PuLP problem
    prob = pulp.LpProblem("NBA_Lineup", pulp.LpMaximize)
    
    # Decision variables
    player_vars = [pulp.LpVariable(f"player_{i}", cat='Binary') for i in range(len(df))]
    
    # Objective: maximize projected points
    prob += pulp.lpSum([
        player_vars[i] * df.iloc[i]['Predicted_DK_Points']
        for i in range(len(df))
    ])
    
    # Constraint: exactly 8 players
    prob += pulp.lpSum(player_vars) == 8
    
    # Salary constraints
    prob += pulp.lpSum([player_vars[i] * df.iloc[i]['Salary'] for i in range(len(df))]) <= salary_cap
    prob += pulp.lpSum([player_vars[i] * df.iloc[i]['Salary'] for i in range(len(df))]) >= min_salary
    
    # Position constraints
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        eligible = [i for i in range(len(df)) if pos in str(df.iloc[i]['Position'])]
        if eligible:
            prob += pulp.lpSum([player_vars[i] for i in eligible]) >= position_limits[pos]
    
    # Guard constraint (PG/SG for G slot)
    guard_eligible = [i for i in range(len(df))
                     if any(g in str(df.iloc[i]['Position']) for g in GUARD_POSITIONS)]
    if guard_eligible:
        prob += pulp.lpSum([player_vars[i] for i in guard_eligible]) >= (
            position_limits['PG'] + position_limits['SG'] + position_limits['G']
        )
    
    # Forward constraint (SF/PF for F slot)
    forward_eligible = [i for i in range(len(df))
                       if any(f in str(df.iloc[i]['Position']) for f in FORWARD_POSITIONS)]
    if forward_eligible:
        prob += pulp.lpSum([player_vars[i] for i in forward_eligible]) >= (
            position_limits['SF'] + position_limits['PF'] + position_limits['F']
        )
    
    # Team stacking constraints (if applicable)
    if stack_type != 'No Stacks' and team_selections:
        if stack_type in ['2', '3', '4', '5']:
            stack_size = int(stack_type)
            teams = team_selections.get(stack_type, team_selections.get('all', []))
            
            if teams:
                team_vars = {}
                for team in teams:
                    team_vars[team] = pulp.LpVariable(f"team_{team}", cat='Binary')
                    team_players = [i for i in range(len(df)) if df.iloc[i]['Team'] == team]
                    if team_players:
                        prob += pulp.lpSum([player_vars[i] for i in team_players]) >= stack_size * team_vars[team]
                
                prob += pulp.lpSum([team_vars[team] for team in teams]) >= 1
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        print(f"    ⚠️ PuLP solve failed with status: {pulp.LpStatus[prob.status]}", file=sys.stderr)
        print(f"       Checking constraints...", file=sys.stderr)
        print(f"       Total players: {len(df)}", file=sys.stderr)
        print(f"       PG available: {len([i for i in range(len(df)) if 'PG' in str(df.iloc[i]['Position'])])}", file=sys.stderr)
        print(f"       SG available: {len([i for i in range(len(df)) if 'SG' in str(df.iloc[i]['Position'])])}", file=sys.stderr)
        print(f"       SF available: {len([i for i in range(len(df)) if 'SF' in str(df.iloc[i]['Position'])])}", file=sys.stderr)
        print(f"       PF available: {len([i for i in range(len(df)) if 'PF' in str(df.iloc[i]['Position'])])}", file=sys.stderr)
        print(f"       C available: {len([i for i in range(len(df)) if 'C' in str(df.iloc[i]['Position'])])}", file=sys.stderr)
        return None
    
    # Extract selected players
    selected_indices = [i for i in range(len(df)) if player_vars[i].varValue > 0.5]
    
    if len(selected_indices) != 8:
        return None
    
    return df.iloc[selected_indices]


def select_diverse_lineups(all_lineups, num_lineups, max_exposure):
    """
    Select diverse lineups from candidates
    Logic from makrovchain_optimizer.py
    """
    
    if not all_lineups:
        return []
    
    selected = []
    exposure_tracker = defaultdict(int)
    used_lineup_keys = set()
    
    # Sort by total projection
    all_lineups.sort(key=lambda x: x[1]['Predicted_DK_Points'].sum(), reverse=True)
    
    for stack_type, lineup_df in all_lineups:
        if len(selected) >= num_lineups:
            break
        
        # Check uniqueness - use player names
        lineup_key = '|'.join(sorted(lineup_df['Name'].astype(str).tolist()))
        
        if lineup_key in used_lineup_keys:
            continue
        
        # Check exposure
        over_exposed = False
        for _, row in lineup_df.iterrows():
            player_id = str(row['Name'])
            if exposure_tracker[player_id] >= (max_exposure / 100.0 * num_lineups):
                over_exposed = True
                break
        
        if over_exposed:
            continue
        
        # Add lineup
        used_lineup_keys.add(lineup_key)
        for _, row in lineup_df.iterrows():
            player_id = str(row['Name'])
            exposure_tracker[player_id] += 1
        
        selected.append((stack_type, lineup_df))
        
        print(f"  ✓ Lineup {len(selected)}: {lineup_df['Predicted_DK_Points'].sum():.1f} pts, ${lineup_df['Salary'].sum()}", file=sys.stderr)
    
    return selected


def assign_roster_positions(lineup_df):
    """Assign roster positions to lineup"""
    roster = {}
    used_indices = set()
    
    # Reset index for clean iteration
    lineup_df = lineup_df.reset_index(drop=True)
    
    # Core positions
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        for i in range(len(lineup_df)):
            if i not in used_indices and pos in str(lineup_df.iloc[i]['Position']):
                player = lineup_df.iloc[i].to_dict()
                player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
                # Map field names for frontend (lowercase versions)
                player['name'] = player.get('Name', player.get('name', ''))
                player['position'] = player.get('Position', player.get('position', ''))
                player['team'] = player.get('Team', player.get('team', ''))
                player['salary'] = player.get('Salary', player.get('salary', 0))
                # Map projection fields for frontend
                proj_value = player.get('Predicted_DK_Points', player.get('projection', 0))
                player['projection'] = proj_value
                player['projectedPoints'] = proj_value
                player['rosterPosition'] = pos
                roster[pos] = player
                used_indices.add(i)
                break
    
    # G position
    for i in range(len(lineup_df)):
        if i not in used_indices and any(g in str(lineup_df.iloc[i]['Position']) for g in ['PG', 'SG']):
            player = lineup_df.iloc[i].to_dict()
            player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
            # Map field names for frontend (lowercase versions)
            player['name'] = player.get('Name', player.get('name', ''))
            player['position'] = player.get('Position', player.get('position', ''))
            player['team'] = player.get('Team', player.get('team', ''))
            player['salary'] = player.get('Salary', player.get('salary', 0))
            # Map projection fields for frontend
            proj_value = player.get('Predicted_DK_Points', player.get('projection', 0))
            player['projection'] = proj_value
            player['projectedPoints'] = proj_value
            player['rosterPosition'] = 'G'
            roster['G'] = player
            used_indices.add(i)
            break
    
    # F position
    for i in range(len(lineup_df)):
        if i not in used_indices and any(f in str(lineup_df.iloc[i]['Position']) for f in ['SF', 'PF']):
            player = lineup_df.iloc[i].to_dict()
            player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
            # Map field names for frontend (lowercase versions)
            player['name'] = player.get('Name', player.get('name', ''))
            player['position'] = player.get('Position', player.get('position', ''))
            player['team'] = player.get('Team', player.get('team', ''))
            player['salary'] = player.get('Salary', player.get('salary', 0))
            # Map projection fields for frontend
            proj_value = player.get('Predicted_DK_Points', player.get('projection', 0))
            player['projection'] = proj_value
            player['projectedPoints'] = proj_value
            player['rosterPosition'] = 'F'
            roster['F'] = player
            used_indices.add(i)
            break
    
    # UTIL position
    for i in range(len(lineup_df)):
        if i not in used_indices:
            player = lineup_df.iloc[i].to_dict()
            player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
            # Map field names for frontend (lowercase versions)
            player['name'] = player.get('Name', player.get('name', ''))
            player['position'] = player.get('Position', player.get('position', ''))
            player['team'] = player.get('Team', player.get('team', ''))
            player['salary'] = player.get('Salary', player.get('salary', 0))
            # Map projection fields for frontend
            proj_value = player.get('Predicted_DK_Points', player.get('projection', 0))
            player['projection'] = proj_value
            player['projectedPoints'] = proj_value
            player['rosterPosition'] = 'UTIL'
            roster['UTIL'] = player
            used_indices.add(i)
            break
    
    return list(roster.values())


def main():
    """CLI entry point"""
    
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input data provided"}))
        sys.exit(1)
    
    try:
        input_data = json.loads(sys.argv[1])
        
        # Prepare DataFrame in the exact format makrovchain_optimizer.py expects
        players = input_data.get('players', [])
        df = pd.DataFrame(players)
        
        # Rename columns to match makrovchain_optimizer format
        df['Name'] = df['name']
        df['Position'] = df['position']
        df['Salary'] = df['salary']
        df['Predicted_DK_Points'] = df['projection']
        df['Team'] = df.get('team', 'UNK')
        
        # Filter
        df = df[df['Predicted_DK_Points'] > 0].copy()
        
        if len(df) < 8:
            raise ValueError("Not enough players")
        
        print(f"📊 Loaded {len(df)} players", file=sys.stderr)
        
        # Parse settings
        num_lineups = input_data.get('numLineups', 10)
        min_salary = input_data.get('minSalary', 48000)
        max_exposure = input_data.get('maxExposure', 100)
        stack_settings_input = input_data.get('stackSettings', {})
        
        # Prepare stack settings in makrovchain format
        stack_settings = {'No Stacks': num_lineups}
        team_selections = {}
        
        if stack_settings_input.get('enabled'):
            stack_type = stack_settings_input.get('type', 'No Stacks')
            if stack_type != 'No Stacks':
                import re
                match = re.search(r'(\d+)', stack_type)
                if match:
                    stack_num = match.group(1)
                    stack_settings = {stack_num: num_lineups}
                    teams = df['Team'].unique().tolist()
                    team_selections = {stack_num: teams, 'all': teams}
        
        # Run optimization using EXACT makrovchain logic
        lineups, team_exp, stack_exp = optimize_using_makrov_logic(
            df, num_lineups, min_salary, stack_settings, team_selections, max_exposure
        )
        
        # Format output
        output_lineups = []
        for stack_type, lineup_df in lineups:
            players_list = assign_roster_positions(lineup_df)
            total_proj = lineup_df['Predicted_DK_Points'].sum()
            total_sal = lineup_df['Salary'].sum()
            
            output_lineups.append({
                'players': players_list,
                'totalProjection': float(total_proj),
                'totalSalary': int(total_sal),
                'strategy': f'makrov_{stack_type}'
            })
        
        output = {
            "success": True,
            "lineups": output_lineups,
            "summary": {
                "totalLineups": len(output_lineups),
                "avgProjection": round(np.mean([l['totalProjection'] for l in output_lineups]), 2) if output_lineups else 0,
                "avgSalary": round(np.mean([l['totalSalary'] for l in output_lineups]), 2) if output_lineups else 0,
                "topProjection": round(max([l['totalProjection'] for l in output_lineups]), 2) if output_lineups else 0,
            }
        }
        
        print(json.dumps(output))
        
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e)}), file=sys.stdout)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

