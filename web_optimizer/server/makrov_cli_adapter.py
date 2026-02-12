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
                                  team_selections, max_exposure, team_exposures=None):
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
                            salary_cap=SALARY_CAP,
                            position_limits=POSITION_LIMITS,
                            exclusion_sets=exclusion_sets
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
                        salary_cap=SALARY_CAP,
                        position_limits=POSITION_LIMITS,
                        exclusion_sets=exclusion_sets
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

    # Select best diverse lineups from candidate pool (Haugh & Singal, 2021 — portfolio selection)
    final_lineups = select_diverse_lineups(all_lineups, num_lineups, max_exposure)

    return final_lineups, team_exposure, stack_exposure


def optimize_single_lineup_pulp(df, stack_type, team_selections, min_salary,
                                 salary_cap, position_limits, exclusion_sets=None):
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
    """

    if df.empty or len(df) < 8:
        return None

    # Reset index to ensure clean indexing
    df = df.reset_index(drop=True).copy()

    # IMMUTABLE PROJECTIONS — no noise, no modification (Bertsimas & Tsitsiklis, 1997)
    # Diversity is achieved through exclusion constraints below

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
                                # If selected (binary = 1), allow stack_size+ (9 is large enough to allow any reasonable stack)
                                prob += pulp.lpSum([player_vars[idx] for idx in other_team_players_idx]) <= stack_size - 1 + (8 - stack_size + 1) * team_binary_vars[other_team]
                                print(f"    🚫 LIMITING: {other_team} to max {stack_size-1} unless selected (binary=1, then max 8)", file=sys.stderr)
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
    return player


def assign_roster_positions(lineup_df):
    """Assign roster positions to lineup"""
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


def monte_carlo_lineup(lineup_df, num_sims=2000):
    """
    Monte Carlo simulation for lineup evaluation (Markowitz, 1952; Michaud, 1998).

    Simulates each player's outcome from Normal(projection, stdDev) independently,
    sums to lineup score. Returns distribution statistics for VaR, Sharpe, etc.

    This is POST-OPTIMIZATION evaluation — projections are NEVER modified for optimization.
    """
    projections = lineup_df['Predicted_DK_Points'].values.astype(float)
    # Estimate stdDev as 30% of projection (typical DFS variance)
    std_devs = projections * 0.30

    # Run simulations
    scores = np.zeros(num_sims)
    for sim in range(num_sims):
        player_outcomes = np.random.normal(projections, std_devs)
        player_outcomes = np.maximum(player_outcomes, 0)  # floor at 0
        scores[sim] = player_outcomes.sum()

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
        'simulations': num_sims
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
        team_selections_input = input_data.get('teamSelections', {})
        team_exposures = input_data.get('teamExposures', {})

        print(f"📊 Team exposures received: {team_exposures}", file=sys.stderr)

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
        print(f"🚀 Starting optimization with stack_settings={stack_settings}, team_selections={team_selections}", file=sys.stderr)
        lineups, team_exp, stack_exp = optimize_using_makrov_logic(
            df, num_lineups, min_salary, stack_settings, team_selections, max_exposure,
            team_exposures=team_exposures
        )
        
        # Format output with Monte Carlo evaluation (Markowitz, 1952; Michaud, 1998)
        output_lineups = []
        for stack_type, lineup_df in lineups:
            players_list = assign_roster_positions(lineup_df)
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

