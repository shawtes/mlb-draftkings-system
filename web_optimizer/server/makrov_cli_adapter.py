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

# Add the optimization directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'optimization'))

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
NBA_POSITION_LIMITS = {
    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
    'G': 1, 'F': 1, 'UTIL': 1
}
NBA_ROSTER_SIZE = 8
NBA_SALARY_CAP = 50000
GUARD_POSITIONS = ['PG', 'SG']
FORWARD_POSITIONS = ['SF', 'PF']

# MLB position requirements (DraftKings Classic)
MLB_POSITION_LIMITS = {
    'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
}
MLB_ROSTER_SIZE = 10
MLB_SALARY_CAP = 50000

# Legacy alias
POSITION_LIMITS = NBA_POSITION_LIMITS
SALARY_CAP = 50000


def optimize_using_makrov_logic(df_filtered, num_lineups, min_salary, stack_settings,
                                  team_selections, max_exposure, team_exposures=None,
                                  player_exposures=None, sport='NBA'):
    """
    This is the EXACT optimization logic from makrovchain_optimizer.py
    Extracted from the OptimizationWorker.optimize_lineups() method
    Supports NBA (8 players) and MLB (10 players) via sport parameter.
    """

    # Select sport-specific configuration
    if sport == 'MLB':
        position_limits = MLB_POSITION_LIMITS
        roster_size = MLB_ROSTER_SIZE
        salary_cap = MLB_SALARY_CAP
        sport_emoji = '⚾'
    else:
        position_limits = NBA_POSITION_LIMITS
        roster_size = NBA_ROSTER_SIZE
        salary_cap = NBA_SALARY_CAP
        sport_emoji = '🏀'

    print(f"{sport_emoji} Using EXACT makrovchain_optimizer.py logic ({sport}, {roster_size} players)", file=sys.stderr)
    
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

    # ACADEMIC PIPELINE (Hunter et al., 2016; Bertsimas & Tsitsiklis, 1997):
    # 1. Solve ILP with IMMUTABLE projections
    # 2. Add EXCLUSION CONSTRAINT from previous lineup (at least 1 different player)
    # 3. Re-solve → next diverse lineup
    # 4. Repeat until we have enough candidates
    # 5. Select best diverse subset respecting exposure limits

    # Generate 2x candidates for portfolio selection (Haugh & Singal, 2021)
    total_candidates_needed = num_lineups * 2
    lineups_per_stack = max(1, total_candidates_needed // max(1, len(stack_settings)))

    print(f"🎯 Generating {total_candidates_needed} candidates for {num_lineups} lineups (ILP + exclusion constraints)", file=sys.stderr)

    all_lineups = []

    print(f"🎯 Stack settings: {stack_settings}", file=sys.stderr)
    print(f"🎯 Team selections: {team_selections}", file=sys.stderr)

    for stack_type, num_solves in stack_settings.items():
        print(f"  Stack type '{stack_type}': generating {lineups_per_stack} candidates via iterative ILP", file=sys.stderr)

        # Build team list for this stack type
        teams_for_stack = []
        if stack_type in team_selections:
            teams_for_stack = team_selections[stack_type]
        elif str(stack_type) in team_selections:
            teams_for_stack = team_selections[str(stack_type)]
        elif 'all' in team_selections:
            teams_for_stack = team_selections['all']

        print(f"    🔍 Teams for stack '{stack_type}': {teams_for_stack} (count: {len(teams_for_stack) if teams_for_stack else 0})", file=sys.stderr)

        if teams_for_stack and len(teams_for_stack) > 1 and stack_type != 'No Stacks':
            if team_exposures is None:
                team_exposures = {}

            lineups_per_team = max(1, lineups_per_stack // len(teams_for_stack))
            remaining_lineups = lineups_per_stack - (lineups_per_team * len(teams_for_stack))

            for team_idx, team in enumerate(teams_for_stack):
                base_count = lineups_per_team + (1 if team_idx < remaining_lineups else 0)

                exp = team_exposures.get(team, {})
                max_exp = exp.get('max', 100)
                min_exp = exp.get('min', 0)
                max_lineups_for_team = max(1, int(num_lineups * max_exp / 100))
                min_lineups_for_team = max(0, int(num_lineups * min_exp / 100))
                team_lineup_count = min(max_lineups_for_team, max(min_lineups_for_team, base_count))

                print(f"    🏀 Generating {team_lineup_count} lineups for {team} {stack_type}-stack (iterative ILP)", file=sys.stderr)

                team_specific_selections = {
                    stack_type: [team],
                    str(stack_type): [team],
                    'all': [team]
                }

                # Iterative ILP with exclusion constraints for this team
                exclusion_sets = []
                for solve_idx in range(team_lineup_count):
                    try:
                        lineup = optimize_single_lineup_pulp(
                            df_filtered,
                            stack_type=stack_type,
                            team_selections=team_specific_selections,
                            min_salary=min_salary,
                            salary_cap=salary_cap,
                            position_limits=position_limits,
                            exclusion_sets=exclusion_sets,
                            sport=sport
                        )

                        if lineup is not None and not lineup.empty:
                            all_lineups.append((stack_type, lineup))
                            # Add this lineup's indices to exclusion sets for next solve
                            exclusion_sets.append(set(lineup.index.tolist()))
                            print(f"      ✓ Lineup {solve_idx+1}: {lineup['Predicted_DK_Points'].sum():.1f} pts (excl: {len(exclusion_sets)} prev)", file=sys.stderr)
                        else:
                            print(f"      ✗ No valid lineup (exhausted diversity space)", file=sys.stderr)
                            break  # No more diverse lineups possible

                    except Exception as e:
                        print(f"      ✗ Exception: {e}", file=sys.stderr)
                        break
        else:
            # Single team or no teams — iterative ILP with exclusion constraints
            exclusion_sets = []
            for solve_idx in range(lineups_per_stack):
                try:
                    lineup = optimize_single_lineup_pulp(
                        df_filtered,
                        stack_type=stack_type,
                        team_selections=team_selections,
                        min_salary=min_salary,
                        salary_cap=salary_cap,
                        position_limits=position_limits,
                        exclusion_sets=exclusion_sets,
                        sport=sport
                    )

                    if lineup is not None and not lineup.empty:
                        all_lineups.append((stack_type, lineup))
                        exclusion_sets.append(set(lineup.index.tolist()))
                        print(f"    ✓ Lineup {solve_idx+1}: {lineup['Predicted_DK_Points'].sum():.1f} pts (excl: {len(exclusion_sets)} prev)", file=sys.stderr)
                    else:
                        print(f"    ✗ No valid lineup (exhausted diversity space)", file=sys.stderr)
                        break

                except Exception as e:
                    print(f"    ✗ Exception: {e}", file=sys.stderr)
                    break

    print(f"✅ Generated {len(all_lineups)} candidate lineups via iterative ILP", file=sys.stderr)

    # Generate additional candidates with min-exposure players locked in (ILP constraint)
    # This is the academically correct way to enforce minimum exposure — hard constraint in the ILP
    if player_exposures:
        min_exp_players = {name: cfg for name, cfg in player_exposures.items() if cfg.get('min', 0) > 0}
        if min_exp_players:
            print(f"🔒 Generating locked candidates for {len(min_exp_players)} min-exposure players", file=sys.stderr)
            for player_name, exp_cfg in min_exp_players.items():
                required_count = max(1, int(exp_cfg['min'] / 100.0 * num_lineups))
                # Find player index in df
                player_indices = df_filtered.index[df_filtered['Name'].astype(str) == player_name].tolist()
                if not player_indices:
                    print(f"  ⚠️ Player '{player_name}' not found in data, skipping lock", file=sys.stderr)
                    continue
                print(f"  🔒 {player_name}: generating {required_count} locked lineups (minExp={exp_cfg['min']}%)", file=sys.stderr)
                # Use 'all' teams for locked lineups (no specific stack type)
                lock_exclusion_sets = []
                for lock_idx in range(required_count):
                    try:
                        lineup = optimize_single_lineup_pulp(
                            df_filtered,
                            stack_type='No Stacks',
                            team_selections=team_selections,
                            min_salary=min_salary,
                            salary_cap=salary_cap,
                            position_limits=position_limits,
                            exclusion_sets=lock_exclusion_sets,
                            locked_player_indices=player_indices,
                            sport=sport
                        )
                        if lineup is not None and not lineup.empty:
                            all_lineups.append(('min-exposure-lock', lineup))
                            lock_exclusion_sets.append(set(lineup.index.tolist()))
                            print(f"    ✓ Locked lineup {lock_idx+1}: {lineup['Predicted_DK_Points'].sum():.1f} pts", file=sys.stderr)
                        else:
                            print(f"    ✗ Could not generate locked lineup for {player_name}", file=sys.stderr)
                            break
                    except Exception as e:
                        print(f"    ✗ Exception generating locked lineup: {e}", file=sys.stderr)
                        break
            print(f"✅ Total candidates after min-exposure locks: {len(all_lineups)}", file=sys.stderr)

    # Select best diverse lineups from candidate pool (Haugh & Singal, 2021 — portfolio selection)
    final_lineups = select_diverse_lineups(all_lineups, num_lineups, max_exposure,
                                            player_exposures=player_exposures)

    return final_lineups, team_exposure, stack_exposure


def optimize_single_lineup_pulp(df, stack_type, team_selections, min_salary,
                                 salary_cap, position_limits, exclusion_sets=None,
                                 locked_player_indices=None, sport='NBA'):
    """
    Single lineup optimization using PuLP ILP.

    ACADEMIC BASIS: Hunter, Vielma, Zaman (2016) "Picking Winners in DFS Using IP"
    - Projections are IMMUTABLE objective coefficients (Bertsimas & Tsitsiklis, 1997)
    - Diversity via EXCLUSION CONSTRAINTS, not projection perturbation
    - Each solve maximizes: SUM(p_i * x_i) subject to salary, position, stack constraints

    Args:
        exclusion_sets: list of sets of player indices from previous lineups.
                        For each set S, we add constraint: SUM(x_i for i in S) <= |S| - 1
                        This guarantees at least 1 different player from each prior lineup.
        locked_player_indices: list of player indices that MUST be included (x_i >= 1).
                               Used for min-exposure enforcement.
        sport: 'NBA' or 'MLB' — determines roster size and position constraints.
    """

    roster_size = MLB_ROSTER_SIZE if sport == 'MLB' else NBA_ROSTER_SIZE
    min_players_needed = roster_size

    if df.empty or len(df) < min_players_needed:
        return None

    # Reset index to ensure clean indexing
    df = df.reset_index(drop=True).copy()

    # IMMUTABLE PROJECTIONS — no noise, no modification (Bertsimas & Tsitsiklis, 1997)
    # Diversity is achieved through exclusion constraints below

    # Create PuLP problem
    prob = pulp.LpProblem(f"{sport}_Lineup", pulp.LpMaximize)

    # Decision variables
    player_vars = [pulp.LpVariable(f"player_{i}", cat='Binary') for i in range(len(df))]

    # Objective: maximize projected points
    prob += pulp.lpSum([
        player_vars[i] * df.iloc[i]['Predicted_DK_Points']
        for i in range(len(df))
    ])

    # Constraint: exactly roster_size players
    prob += pulp.lpSum(player_vars) == roster_size

    # Salary constraints
    prob += pulp.lpSum([player_vars[i] * df.iloc[i]['Salary'] for i in range(len(df))]) <= salary_cap
    prob += pulp.lpSum([player_vars[i] * df.iloc[i]['Salary'] for i in range(len(df))]) >= min_salary

    if sport == 'MLB':
        # MLB position constraints: P>=2, C>=1, 1B>=1, 2B>=1, 3B>=1, SS>=1, OF>=3
        # No flex positions — simpler constraint set than NBA
        for pos, required in position_limits.items():
            eligible = [i for i in range(len(df)) if pos in str(df.iloc[i]['Position'])]
            if eligible:
                prob += pulp.lpSum([player_vars[i] for i in eligible]) >= required
            else:
                print(f"    ⚠️ No eligible players for MLB position {pos}", file=sys.stderr)
    else:
        # NBA position constraints (original logic)
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
    
    # Team stacking constraints (if applicable) - MATCHING DESKTOP LOGIC
    print(f"    🔍 Stack constraint check: stack_type={stack_type} (type: {type(stack_type)}), team_selections keys={list(team_selections.keys()) if team_selections else 'None'}", file=sys.stderr)
    if team_selections:
        print(f"    🔍 Full team_selections: {team_selections}", file=sys.stderr)
    
    if stack_type != 'No Stacks' and team_selections:
        if stack_type in ['2', '3', '4', '5']:
            stack_size = int(stack_type)
            # Get teams for this stack size
            teams = team_selections.get(stack_type, team_selections.get(str(stack_size), team_selections.get('all', [])))
            print(f"    🔍 Found {len(teams) if teams else 0} teams for {stack_size}-stack: {teams[:3] if teams else 'None'}...", file=sys.stderr)
            
            if teams:
                # Filter to only teams with enough players (matching desktop logic)
                valid_teams = []
                for team in teams:
                    team_players_idx = df[df['Team'] == team].index.tolist()
                    if len(team_players_idx) >= stack_size:
                        valid_teams.append(team)
                
                if not valid_teams:
                    # No valid teams, skip stack constraint
                    print(f"    ⚠️ No valid teams for {stack_size}-stack, skipping constraint", file=sys.stderr)
                    pass
                elif len(valid_teams) == 1:
                    # Single team: enforce directly (EXACT desktop logic)
                    selected_team = valid_teams[0]
                    team_players_idx = df[df['Team'] == selected_team].index.tolist()
                    print(f"    🎯 ENFORCING: At least {stack_size} players from {selected_team} (found {len(team_players_idx)} available)", file=sys.stderr)
                    print(f"    🎯 Team players indices: {team_players_idx[:5]}... (total: {len(team_players_idx)})", file=sys.stderr)
                    
                    # CRITICAL: Use >= to enforce minimum (matching desktop line 984)
                    constraint = pulp.lpSum([player_vars[idx] for idx in team_players_idx]) >= stack_size
                    prob += constraint
                    print(f"    ✅ Added constraint: sum(players from {selected_team}) >= {stack_size}", file=sys.stderr)
                    
                    # Limit all other teams to stack_size - 1 (matching desktop logic)
                    all_teams = df['Team'].unique().tolist()
                    for other_team in all_teams:
                        if other_team != selected_team:
                            other_team_players_idx = df[df['Team'] == other_team].index.tolist()
                            if len(other_team_players_idx) >= stack_size:
                                prob += pulp.lpSum([player_vars[idx] for idx in other_team_players_idx]) <= stack_size - 1
                                print(f"    🚫 LIMITING: {other_team} to max {stack_size - 1} players", file=sys.stderr)
                else:
                    # Multiple teams: use binary variables to ensure exactly ONE team gets the stack
                    # MATCHING DESKTOP LOGIC (lines 1002-1056)
                    team_binary_vars = {}
                    for team in valid_teams:
                        team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{stack_size}", cat='Binary')
                    
                    print(f"    🎯 Multiple teams selected ({len(valid_teams)}): {valid_teams}", file=sys.stderr)
                    
                    # At least one team must be selected
                    prob += pulp.lpSum([team_binary_vars[team] for team in valid_teams]) >= 1
                    
                    # ONLY ONE team can have the stack (prevent multiple 3+ player stacks)
                    prob += pulp.lpSum([team_binary_vars[team] for team in valid_teams]) <= 1
                    print(f"    ✅ ENFORCING: Exactly ONE team will have {stack_size}+ players", file=sys.stderr)
                    
                    # If a team is selected (binary = 1), enforce at least 'stack_size' players
                    for team in valid_teams:
                        team_players_idx = df[df['Team'] == team].index.tolist()
                        if team_players_idx and len(team_players_idx) >= stack_size:
                            # If team is selected (binary = 1), enforce at least 'stack_size' players
                            prob += pulp.lpSum([player_vars[idx] for idx in team_players_idx]) >= stack_size * team_binary_vars[team]
                            print(f"    ✅ If {team} selected, must have >= {stack_size} players", file=sys.stderr)
                    
                    # ADDITIONAL CONSTRAINT: Prevent ALL other teams from having large stacks
                    # This ensures ONLY ONE team (the stacked team) has stack_size+ players
                    all_teams = df['Team'].unique().tolist()
                    for other_team in all_teams:
                        if other_team in valid_teams:
                            # For valid teams, they can only have stack_size+ if their binary var is 1
                            other_team_players_idx = df[df['Team'] == other_team].index.tolist()
                            if len(other_team_players_idx) >= stack_size:
                                # If NOT selected (binary = 0), limit to stack_size-1
                                # If selected (binary = 1), allow up to roster_size (Big-M relaxation)
                                prob += pulp.lpSum([player_vars[idx] for idx in other_team_players_idx]) <= stack_size - 1 + (roster_size - stack_size + 1) * team_binary_vars[other_team]
                                print(f"    🚫 LIMITING: {other_team} to max {stack_size-1} unless selected (binary=1, then max {roster_size})", file=sys.stderr)
                        else:
                            # For non-valid teams, always limit to stack_size-1
                            other_team_players_idx = df[df['Team'] == other_team].index.tolist()
                            if len(other_team_players_idx) >= stack_size:
                                prob += pulp.lpSum([player_vars[idx] for idx in other_team_players_idx]) <= stack_size - 1
                                print(f"    🚫 LIMITING: {other_team} (not in valid teams) to max {stack_size-1} players", file=sys.stderr)
    
    # EXCLUSION CONSTRAINTS for diversity (Hunter et al., 2016)
    # For each previous lineup, require at least 1 different player
    if exclusion_sets:
        for ex_idx, prev_set in enumerate(exclusion_sets):
            prob += pulp.lpSum([player_vars[i] for i in prev_set if i < len(player_vars)]) <= len(prev_set) - 1, f"exclude_lineup_{ex_idx}"

    # LOCKED PLAYER CONSTRAINTS for min-exposure enforcement
    # Force specific players into the lineup (x_i >= 1)
    if locked_player_indices:
        for lock_idx in locked_player_indices:
            if lock_idx < len(player_vars):
                prob += player_vars[lock_idx] >= 1, f"lock_player_{lock_idx}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status != pulp.LpStatusOptimal:
        print(f"    ⚠️ PuLP solve failed with status: {pulp.LpStatus[prob.status]}", file=sys.stderr)
        print(f"       Checking constraints...", file=sys.stderr)
        print(f"       Total players: {len(df)}", file=sys.stderr)
        if sport == 'MLB':
            for pos in ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']:
                count = len([i for i in range(len(df)) if pos in str(df.iloc[i]['Position'])])
                print(f"       {pos} available: {count}", file=sys.stderr)
        else:
            for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                count = len([i for i in range(len(df)) if pos in str(df.iloc[i]['Position'])])
                print(f"       {pos} available: {count}", file=sys.stderr)
        return None

    # Extract selected players
    selected_indices = [i for i in range(len(df)) if player_vars[i].varValue > 0.5]

    if len(selected_indices) != roster_size:
        return None
    
    # Debug: Check team distribution
    selected_df = df.iloc[selected_indices]
    if stack_type != 'No Stacks' and team_selections:
        team_counts = selected_df['Team'].value_counts()
        print(f"    📊 Team distribution in lineup: {dict(team_counts)}", file=sys.stderr)
        if stack_type in ['2', '3', '4', '5']:
            stack_size = int(stack_type)
            max_team_count = team_counts.max()
            max_team = team_counts.idxmax()
            print(f"    📊 Max team: {max_team} with {max_team_count} players (required: {stack_size})", file=sys.stderr)
            if max_team_count < stack_size:
                print(f"    ❌ ERROR: Constraint violation! Expected at least {stack_size} players from one team, but max is {max_team_count} from {max_team}", file=sys.stderr)
                # This should never happen if constraints are working - return None to force retry
                return None
            else:
                print(f"    ✅ Constraint satisfied: {max_team} has {max_team_count} players (>= {stack_size})", file=sys.stderr)
    
    return selected_df


def select_diverse_lineups(all_lineups, num_lineups, max_exposure, player_exposures=None):
    """
    Select diverse lineups from candidates.
    Enforces per-player min AND max exposure (from player_exposures) with fallback to global max_exposure.

    Two-phase selection:
      Phase 1: Satisfy min-exposure constraints by selecting best lineups containing required players
      Phase 2: Fill remaining slots with highest-projection lineups (respecting max exposure)
    """

    if not all_lineups:
        return []

    selected = []
    exposure_tracker = defaultdict(int)
    used_lineup_keys = set()

    # Sort by total projection
    all_lineups.sort(key=lambda x: x[1]['Predicted_DK_Points'].sum(), reverse=True)

    def _check_max_exposure(lineup_df):
        """Return (ok, blocking_player) — checks no player exceeds their max exposure."""
        for _, row in lineup_df.iterrows():
            player_name = str(row['Name'])
            exp_config = (player_exposures or {}).get(player_name, {})
            player_max = exp_config.get('max', max_exposure)
            max_appearances = max(1, int(player_max / 100.0 * num_lineups))
            if exposure_tracker[player_name] >= max_appearances:
                return False, player_name
        return True, None

    def _add_lineup(stack_type, lineup_df, phase_label):
        """Add a lineup to selected, update exposure tracker."""
        lineup_key = '|'.join(sorted(lineup_df['Name'].astype(str).tolist()))
        if lineup_key in used_lineup_keys:
            return False
        ok, blocker = _check_max_exposure(lineup_df)
        if not ok:
            return False
        used_lineup_keys.add(lineup_key)
        for _, row in lineup_df.iterrows():
            exposure_tracker[str(row['Name'])] += 1
        selected.append((stack_type, lineup_df))
        print(f"  ✓ [{phase_label}] Lineup {len(selected)}: {lineup_df['Predicted_DK_Points'].sum():.1f} pts, ${lineup_df['Salary'].sum()}", file=sys.stderr)
        return True

    # ── Phase 1: Satisfy min-exposure requirements ──
    min_exp_needs = {}
    if player_exposures:
        for player_name, exp_config in player_exposures.items():
            min_exp = exp_config.get('min', 0)
            if min_exp > 0:
                required_count = max(1, int(min_exp / 100.0 * num_lineups))
                min_exp_needs[player_name] = required_count

    if min_exp_needs:
        print(f"  📌 Phase 1: Satisfying min exposure for {len(min_exp_needs)} players", file=sys.stderr)
        for player_name, required_count in min_exp_needs.items():
            print(f"     {player_name}: need {required_count} lineups", file=sys.stderr)

        # Keep iterating until all min requirements met or we run out of candidates
        for stack_type, lineup_df in all_lineups:
            if len(selected) >= num_lineups:
                break
            # Check if any unsatisfied min-exp player is in this lineup
            lineup_players = set(lineup_df['Name'].astype(str).tolist())
            has_needed_player = False
            for pname, needed in min_exp_needs.items():
                if pname in lineup_players and exposure_tracker.get(pname, 0) < needed:
                    has_needed_player = True
                    break
            if not has_needed_player:
                continue
            _add_lineup(stack_type, lineup_df, 'min-exp')

        # Report on min exposure satisfaction
        for player_name, required_count in min_exp_needs.items():
            actual = exposure_tracker.get(player_name, 0)
            status = '✅' if actual >= required_count else '⚠️'
            print(f"  {status} Min exposure: {player_name} in {actual}/{required_count} lineups", file=sys.stderr)

    # ── Phase 2: Fill remaining slots with best projection lineups ──
    remaining = num_lineups - len(selected)
    if remaining > 0:
        print(f"  📌 Phase 2: Filling {remaining} remaining slots by projection", file=sys.stderr)
        for stack_type, lineup_df in all_lineups:
            if len(selected) >= num_lineups:
                break
            _add_lineup(stack_type, lineup_df, 'proj')

    return selected


def _map_player_for_frontend(row, roster_position):
    """Map a DataFrame row to a frontend-compatible player dict."""
    player = row.to_dict()
    player = {k: (None if pd.isna(v) else v) for k, v in player.items()}
    player['name'] = player.get('Name', player.get('name', ''))
    player['position'] = player.get('Position', player.get('position', ''))
    player['team'] = player.get('Team', player.get('team', ''))
    player['salary'] = player.get('Salary', player.get('salary', 0))
    proj_value = player.get('Predicted_DK_Points', player.get('projection', 0))
    player['projection'] = proj_value
    player['projectedPoints'] = proj_value
    player['rosterPosition'] = roster_position

    # Pass through quant profile fields (snake_case → camelCase for JS frontend)
    _QUANT_FIELD_MAP = {
        'garch_volatility': 'garchVolatility',
        'garch_conditional_volatility': 'garchConditionalVolatility',
        'volatility_regime': 'volatilityRegime',
        'bull_regime': 'bullRegime',
        'regime_strength': 'regimeStrength',
        'momentum_regime': 'momentumRegime',
        'consistency_regime': 'consistencyRegime',
        'entropy': 'entropy',
        'hurst_exponent': 'hurstExponent',
        'rolling_sharpe': 'rollingSharpe',
        'avg_player_correlation': 'avgPlayerCorrelation',
        'correlation_volatility': 'correlationVolatility',
        'evt_return_level': 'evtReturnLevel',
        'exceedance_prob': 'exceedanceProb',
    }
    for snake, camel in _QUANT_FIELD_MAP.items():
        val = player.get(snake)
        if val is not None:
            player[camel] = val

    return player


def assign_roster_positions(lineup_df, sport='NBA'):
    """Assign roster positions to lineup. Routes to sport-specific logic."""
    if sport == 'MLB':
        return assign_roster_positions_mlb(lineup_df)
    return assign_roster_positions_nba(lineup_df)


def assign_roster_positions_nba(lineup_df):
    """Assign NBA roster positions (PG/SG/SF/PF/C/G/F/UTIL)."""
    roster = {}
    used_indices = set()

    lineup_df = lineup_df.reset_index(drop=True)

    def assign_first_match(roster_pos, eligible_positions=None):
        """Find first unused player matching eligible positions and assign to roster_pos."""
        for i in range(len(lineup_df)):
            if i in used_indices:
                continue
            if eligible_positions is not None:
                pos_str = str(lineup_df.iloc[i]['Position'])
                if not any(p in pos_str for p in eligible_positions):
                    continue
            roster[roster_pos] = _map_player_for_frontend(lineup_df.iloc[i], roster_pos)
            used_indices.add(i)
            return

    # Core positions
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        assign_first_match(pos, [pos])

    # G slot (PG or SG)
    assign_first_match('G', GUARD_POSITIONS)
    # F slot (SF or PF)
    assign_first_match('F', FORWARD_POSITIONS)
    # UTIL slot (any remaining)
    assign_first_match('UTIL')

    return list(roster.values())


def assign_roster_positions_mlb(lineup_df):
    """Assign MLB roster positions (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)."""
    roster = {}
    used_indices = set()

    lineup_df = lineup_df.reset_index(drop=True)

    def assign_first_match(roster_pos, eligible_positions):
        """Find first unused player matching eligible positions and assign to roster_pos."""
        for i in range(len(lineup_df)):
            if i in used_indices:
                continue
            pos_str = str(lineup_df.iloc[i]['Position'])
            if not any(p in pos_str for p in eligible_positions):
                continue
            roster[roster_pos] = _map_player_for_frontend(lineup_df.iloc[i], roster_pos)
            used_indices.add(i)
            return

    # Pitchers — P1 and P2 (DK uses "P" for both slots)
    assign_first_match('P', ['P'])
    assign_first_match('P', ['P'])  # second pitcher goes to same roster key—use unique keys
    # Re-do: need unique keys for the dict
    roster.clear()
    used_indices.clear()

    # MLB DK Classic roster slots
    mlb_slots = [
        ('P', ['P', 'SP', 'RP']),
        ('P', ['P', 'SP', 'RP']),
        ('C', ['C']),
        ('1B', ['1B']),
        ('2B', ['2B']),
        ('3B', ['3B']),
        ('SS', ['SS']),
        ('OF', ['OF']),
        ('OF', ['OF']),
        ('OF', ['OF']),
    ]

    result_players = []
    for roster_pos, eligible in mlb_slots:
        for i in range(len(lineup_df)):
            if i in used_indices:
                continue
            pos_str = str(lineup_df.iloc[i]['Position'])
            if not any(p in pos_str for p in eligible):
                continue
            result_players.append(_map_player_for_frontend(lineup_df.iloc[i], roster_pos))
            used_indices.add(i)
            break

    # Any remaining unassigned players get UTIL-like assignment
    for i in range(len(lineup_df)):
        if i not in used_indices:
            result_players.append(_map_player_for_frontend(lineup_df.iloc[i], 'UTIL'))
            used_indices.add(i)

    return result_players


def monte_carlo_lineup(lineup_df, num_sims=2000):
    """
    Monte Carlo simulation for lineup evaluation (Markowitz, 1952; Michaud, 1998).

    Uses GARCH-derived volatility when available (from training pipeline quant profile),
    and Cholesky-decomposed correlation for same-team players.

    This is POST-OPTIMIZATION evaluation — projections are NEVER modified for optimization.
    """
    n_players = len(lineup_df)
    projections = lineup_df['Predicted_DK_Points'].values.astype(float)

    # --- StdDev: use GARCH if available, else flat 30% of projection ---
    std_devs = np.zeros(n_players)
    garch_used = 0
    for i in range(n_players):
        row = lineup_df.iloc[i]
        garch_val = None
        for col in ['garch_conditional_volatility', 'garch_volatility']:
            if col in row.index:
                v = row[col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) > 0:
                    garch_val = float(v)
                    break
        if garch_val is not None:
            # GARCH value is a return-scale volatility; scale by projection
            std_devs[i] = garch_val * projections[i]
            garch_used += 1
        else:
            std_devs[i] = projections[i] * 0.30

    # Clamp stdDev: minimum 5% of projection, maximum 80% of projection
    std_devs = np.clip(std_devs, projections * 0.05, projections * 0.80)

    # --- Correlation matrix: same-team players get correlated outcomes ---
    teams = lineup_df['Team'].values if 'Team' in lineup_df.columns else [None] * n_players
    corr_matrix = np.eye(n_players)

    for i in range(n_players):
        for j in range(i + 1, n_players):
            if teams[i] is not None and teams[j] is not None and teams[i] == teams[j]:
                # Same-team correlation from avg_player_correlation if available
                rho_i = 0.0
                rho_j = 0.0
                if 'avg_player_correlation' in lineup_df.columns:
                    vi = lineup_df.iloc[i]['avg_player_correlation']
                    vj = lineup_df.iloc[j]['avg_player_correlation']
                    if vi is not None and not (isinstance(vi, float) and np.isnan(vi)):
                        rho_i = float(vi)
                    if vj is not None and not (isinstance(vj, float) and np.isnan(vj)):
                        rho_j = float(vj)
                rho = max(0.20, (rho_i + rho_j) / 2.0) if (rho_i + rho_j) > 0 else 0.25
                rho = min(rho, 0.60)  # cap at 0.60
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

    # --- Cholesky decomposition for correlated sampling ---
    try:
        L = np.linalg.cholesky(corr_matrix)
        correlated = True
    except np.linalg.LinAlgError:
        # Matrix not positive-definite — fall back to independent sampling
        L = np.eye(n_players)
        correlated = False

    # --- Run simulations ---
    # Generate all random numbers at once for efficiency (num_sims x n_players)
    Z_independent = np.random.standard_normal((num_sims, n_players))
    Z_correlated = Z_independent @ L.T  # Apply Cholesky factor

    # Scale: outcome = projection + stdDev * correlated_Z, clamped >= 0
    outcomes = projections[np.newaxis, :] + std_devs[np.newaxis, :] * Z_correlated
    outcomes = np.maximum(outcomes, 0.0)
    scores = outcomes.sum(axis=1)

    scores.sort()
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    total_proj = float(projections.sum())

    # VaR at 95% confidence (5th percentile)
    var_idx = max(0, int(num_sims * 0.05))
    value_at_risk = float(scores[var_idx])

    # CVaR (Expected Shortfall) — avg of outcomes below VaR
    cvar = float(np.mean(scores[:var_idx + 1])) if var_idx > 0 else value_at_risk

    # Sharpe ratio (mean / stdDev)
    sharpe = mean / std if std > 0 else 0

    # Ceiling probability: P(lineup > 1.3x projection)
    ceiling_target = total_proj * 1.3
    ceiling_prob = float(np.mean(scores >= ceiling_target))

    return {
        'simMean': round(mean, 2),
        'simStdDev': round(std, 2),
        'sharpeRatio': round(sharpe, 3),
        'valueAtRisk': round(value_at_risk, 2),
        'conditionalVaR': round(cvar, 2),
        'ceilingProbability': round(ceiling_prob, 4),
        'percentiles': {
            'p10': round(float(scores[int(num_sims * 0.10)]), 2),
            'p25': round(float(scores[int(num_sims * 0.25)]), 2),
            'p50': round(float(scores[int(num_sims * 0.50)]), 2),
            'p75': round(float(scores[int(num_sims * 0.75)]), 2),
            'p90': round(float(scores[int(num_sims * 0.90)]), 2),
        },
        'simulations': num_sims,
        'garchPlayersUsed': garch_used,
        'correlatedSampling': correlated,
    }


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

        # Map camelCase quant fields from server to snake_case for Python path
        _CAMEL_TO_SNAKE = {
            'garchVolatility': 'garch_volatility',
            'garchConditionalVolatility': 'garch_conditional_volatility',
            'volatilityRegime': 'volatility_regime',
            'bullRegime': 'bull_regime',
            'regimeStrength': 'regime_strength',
            'momentumRegime': 'momentum_regime',
            'consistencyRegime': 'consistency_regime',
            'entropy': 'entropy',
            'hurstExponent': 'hurst_exponent',
            'rollingSharpe': 'rolling_sharpe',
            'avgPlayerCorrelation': 'avg_player_correlation',
            'correlationVolatility': 'correlation_volatility',
            'evtReturnLevel': 'evt_return_level',
            'exceedanceProb': 'exceedance_prob',
        }
        quant_found = 0
        for camel, snake in _CAMEL_TO_SNAKE.items():
            if camel in df.columns:
                df[snake] = pd.to_numeric(df[camel], errors='coerce')
                quant_found += 1
        if quant_found > 0:
            print(f"📊 Mapped {quant_found} quant fields from server (camelCase → snake_case)", file=sys.stderr)

        # Detect sport (default NBA for backward compatibility)
        sport = input_data.get('sport', 'NBA').upper()
        if sport not in ('NBA', 'MLB'):
            sport = 'NBA'

        # Filter
        df = df[df['Predicted_DK_Points'] > 0].copy()

        min_players_needed = MLB_ROSTER_SIZE if sport == 'MLB' else NBA_ROSTER_SIZE
        if len(df) < min_players_needed:
            raise ValueError(f"Not enough players for {sport} (need {min_players_needed}, got {len(df)})")

        sport_emoji = '⚾' if sport == 'MLB' else '🏀'
        print(f"{sport_emoji} Loaded {len(df)} players for {sport}", file=sys.stderr)

        # Parse settings
        num_lineups = input_data.get('numLineups', 10)
        default_min_salary = 45000 if sport == 'MLB' else 48000
        min_salary = input_data.get('minSalary', default_min_salary)
        max_exposure = input_data.get('maxExposure', 100)
        stack_settings_input = input_data.get('stackSettings', {})
        team_selections_input = input_data.get('teamSelections', {})
        team_exposures = input_data.get('teamExposures', {})
        player_exposures = input_data.get('playerExposures', {})

        print(f"📊 Team exposures received: {team_exposures}", file=sys.stderr)
        if player_exposures:
            print(f"📊 Per-player exposures received for {len(player_exposures)} players:", file=sys.stderr)
            for pname, pcfg in player_exposures.items():
                print(f"   {pname}: min={pcfg.get('min',0)}%, max={pcfg.get('max',100)}%", file=sys.stderr)
        else:
            print(f"📊 No per-player exposures (all defaults)", file=sys.stderr)

        # Parse requestedStackSizes if available (explicit contract from frontend)
        requested_stack_sizes = stack_settings_input.get('requestedStackSizes', [])
        if requested_stack_sizes:
            print(f"Using explicit requestedStackSizes: {requested_stack_sizes}", file=sys.stderr)

        # Prepare stack settings in makrovchain format
        stack_settings = {'No Stacks': num_lineups}
        team_selections = {}
        warnings = []

        if stack_settings_input.get('enabled'):
            # FIRST: Check teamSelections directly to find which stack sizes have teams selected
            # teamSelections format: {2: ['LAL'], 3: ['BOS'], 'all': [...]}
            selected_stack_sizes = []

            # If requestedStackSizes is provided, use it as the source of truth
            if requested_stack_sizes:
                for ss in requested_stack_sizes:
                    try:
                        stack_size = int(ss)
                        if stack_size >= 2:
                            # Find teams for this stack size from teamSelections
                            teams = team_selections_input.get(str(stack_size), team_selections_input.get(stack_size, []))
                            if isinstance(teams, list) and len(teams) > 0:
                                selected_stack_sizes.append((stack_size, teams))
                            else:
                                # Use 'all' teams as fallback
                                all_teams = team_selections_input.get('all', [])
                                if all_teams:
                                    selected_stack_sizes.append((stack_size, all_teams))
                    except (ValueError, TypeError):
                        pass
                print(f"✅ Built stack sizes from requestedStackSizes: {[(s, len(t)) for s, t in selected_stack_sizes]}", file=sys.stderr)
            else:
                # Fall back to parsing teamSelections keys
                for key, teams in team_selections_input.items():
                    if key != 'all' and isinstance(teams, list) and len(teams) > 0:
                        try:
                            stack_size = int(key)
                            if stack_size >= 2:
                                selected_stack_sizes.append((stack_size, teams))
                        except (ValueError, TypeError):
                            pass
            
            # Process ALL stack sizes that have teams selected (not just the largest)
            if selected_stack_sizes:
                stack_settings = {}
                warnings = []

                # Distribute lineups across stack sizes
                lineups_per_size = max(1, num_lineups // len(selected_stack_sizes))
                remaining = num_lineups - (lineups_per_size * len(selected_stack_sizes))

                # Sort by stack size descending so larger stacks get remainder
                selected_stack_sizes.sort(key=lambda x: x[0], reverse=True)

                for idx, (stack_size, selected_teams) in enumerate(selected_stack_sizes):
                    stack_num = str(stack_size)
                    count = lineups_per_size + (1 if idx < remaining else 0)

                    # Filter to only teams that have enough players
                    valid_teams = []
                    insufficient_teams = []
                    for team in selected_teams:
                        team_players = df[df['Team'] == team]
                        if len(team_players) >= stack_size:
                            valid_teams.append(team)
                        else:
                            insufficient_teams.append((team, len(team_players)))

                    if insufficient_teams:
                        for team, player_count in insufficient_teams:
                            warnings.append(f"Team {team} only has {player_count} players, need {stack_size} for {stack_size}-stack")

                    if valid_teams:
                        stack_settings[stack_num] = count
                        team_selections[stack_num] = valid_teams
                        print(f"✅ Using {len(valid_teams)} teams for {stack_size}-stack ({count} lineups): {valid_teams}", file=sys.stderr)
                    else:
                        warnings.append(f"No valid teams found for {stack_size}-stack, skipping")
                        print(f"⚠️ No valid teams found for {stack_size}-stack, skipping", file=sys.stderr)

                # If no stack sizes had valid teams, fall back to No Stacks
                if not stack_settings:
                    stack_settings = {'No Stacks': num_lineups}
                    print(f"⚠️ No valid stack sizes, falling back to No Stacks", file=sys.stderr)

                if warnings:
                    print(f"⚠️ Warnings: {warnings}", file=sys.stderr)
            else:
                # Fallback: Determine stack size from minPlayersPerTeam or maxPlayersPerTeam
                min_players = stack_settings_input.get('minPlayersPerTeam', 0)
                max_players = stack_settings_input.get('maxPlayersPerTeam', 0)
                stack_size = max(min_players, max_players) if (min_players > 0 or max_players > 0) else 0
                
                if stack_size >= 2:
                    stack_num = str(stack_size)
                    stack_settings = {stack_num: num_lineups}
                    
                    # Use teams from stackSettings or 'all' key
                    selected_teams = []
                    if 'all' in team_selections_input:
                        selected_teams = team_selections_input['all']
                    elif stack_settings_input.get('teams'):
                        selected_teams = stack_settings_input['teams']
                    else:
                        selected_teams = df['Team'].unique().tolist()
                    
                    # Filter to only teams that have enough players
                    valid_teams = []
                    for team in selected_teams:
                        team_players = df[df['Team'] == team]
                        if len(team_players) >= stack_size:
                            valid_teams.append(team)
                    
                    if valid_teams:
                        team_selections = {stack_num: valid_teams, 'all': valid_teams}
                        print(f"✅ Using {len(valid_teams)} teams for {stack_size}-stack (fallback): {valid_teams[:5]}...", file=sys.stderr)
                    else:
                        print(f"⚠️ No valid teams found for {stack_size}-stack, using all teams", file=sys.stderr)
                        all_teams = df['Team'].unique().tolist()
                        team_selections = {stack_num: all_teams, 'all': all_teams}
        
        # Feasibility validation: warn if a team lacks enough players for its stack size
        for stack_size_key, teams in team_selections.items():
            if stack_size_key == 'all':
                continue
            try:
                fv_stack_size = int(stack_size_key)
            except (ValueError, TypeError):
                continue
            for team in teams:
                team_player_count = len(df[df['Team'] == team])
                if team_player_count < fv_stack_size:
                    warnings.append(
                        f"{team} has only {team_player_count} players but needs {fv_stack_size} for {fv_stack_size}-stack"
                    )

        if warnings:
            print(f"⚠️ Feasibility warnings: {warnings}", file=sys.stderr)

        # Run optimization using EXACT makrovchain logic
        print(f"🚀 Starting {sport} optimization with stack_settings={stack_settings}, team_selections={team_selections}", file=sys.stderr)
        lineups, team_exp, stack_exp = optimize_using_makrov_logic(
            df, num_lineups, min_salary, stack_settings, team_selections, max_exposure,
            team_exposures=team_exposures, player_exposures=player_exposures, sport=sport
        )
        
        # Format output with Monte Carlo evaluation (Markowitz, 1952; Michaud, 1998)
        output_lineups = []
        for stack_type, lineup_df in lineups:
            players_list = assign_roster_positions(lineup_df, sport=sport)
            total_proj = lineup_df['Predicted_DK_Points'].sum()
            total_sal = lineup_df['Salary'].sum()

            # Monte Carlo simulation per lineup (post-optimization evaluation)
            mc_metrics = monte_carlo_lineup(lineup_df, num_sims=2000)

            output_lineups.append({
                'players': players_list,
                'totalProjection': float(total_proj),
                'totalSalary': int(total_sal),
                'strategy': f'ilp_{stack_type}',
                'quantMetrics': mc_metrics
            })
        
        # Lineup count validation
        if len(output_lineups) < num_lineups:
            warnings.append(f"Only generated {len(output_lineups)} of {num_lineups} requested lineups. Stack constraints may be too restrictive.")

        output = {
            "success": True,
            "lineups": output_lineups,
            "warnings": warnings,
            "summary": {
                "totalLineups": len(output_lineups),
                "requestedLineups": num_lineups,
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

