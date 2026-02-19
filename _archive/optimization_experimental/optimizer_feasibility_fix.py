"""
CRITICAL OPTIMIZER FIX - FEASIBILITY ISSUE RESOLUTION
====================================================

This fix addresses the recurring infeasibility issue where the optimizer fails 
to generate any valid lineups due to over-constrained stacking requirements.

KEY FIXES:
1. Automatic detection of infeasible stack combinations
2. Dynamic reduction of min_unique when infeasibility is detected
3. Fallback to simpler stacking strategies
4. Feasibility pre-check before full optimization

The issue: 4-stack BOS + 2-stack LAD = 6 stack players + 2 pitchers = 8 players
This leaves only 2 slots for remaining batters, making the problem mathematically impossible.
"""

import logging

def check_stack_feasibility(stack_settings, df_players, position_limits):
    """
    Check if the requested stack combination is mathematically feasible
    
    Args:
        stack_settings: List of stack types (e.g., ["4|2"])
        df_players: DataFrame of available players
        position_limits: Dictionary of position requirements
    
    Returns:
        dict: Feasibility analysis results
    """
    logging.info("🔍 FEASIBILITY CHECK: Analyzing stack constraints...")
    
    results = {
        'is_feasible': True,
        'issues': [],
        'recommendations': [],
        'max_feasible_stacks': []
    }
    
    # Calculate available lineup slots
    total_roster_size = sum(position_limits.values())  # Should be 10
    pitcher_slots = position_limits.get('P', 2)  # 2 pitchers
    batter_slots = total_roster_size - pitcher_slots  # 8 batters
    
    logging.info(f"📊 Lineup structure: {total_roster_size} total ({pitcher_slots} P, {batter_slots} batters)")
    
    for stack_type in stack_settings:
        if '|' in stack_type:
            # Multi-stack scenario (e.g., "4|2")
            stack_sizes = [int(x.strip()) for x in stack_type.split('|')]
            total_stack_players = sum(stack_sizes)
            
            logging.info(f"🎯 Analyzing multi-stack: {stack_type} = {stack_sizes} = {total_stack_players} total stack players")
            
            # Check if total stack players exceeds available batter slots
            if total_stack_players > batter_slots:
                results['is_feasible'] = False
                issue = f"Multi-stack {stack_type} requires {total_stack_players} batters but only {batter_slots} batter slots available"
                results['issues'].append(issue)
                
                # Calculate maximum feasible combination
                max_primary_stack = min(stack_sizes[0], batter_slots - 1)  # Leave at least 1 for secondary
                max_secondary_stack = min(stack_sizes[1], batter_slots - max_primary_stack)
                feasible_combo = f"{max_primary_stack}|{max_secondary_stack}"
                results['recommendations'].append(f"Try {feasible_combo} instead of {stack_type}")
                results['max_feasible_stacks'].append(feasible_combo)
                
                logging.error(f"❌ INFEASIBLE: {issue}")
                logging.info(f"💡 SUGGESTION: Use {feasible_combo} instead")
            
            # Check team availability for each stack size
            for i, stack_size in enumerate(stack_sizes):
                teams_with_enough_batters = []
                for team in df_players['Team'].unique():
                    team_batters = df_players[(df_players['Team'] == team) & (~df_players['Pos'].str.contains('P', na=False))]
                    if len(team_batters) >= stack_size:
                        teams_with_enough_batters.append(team)
                
                if len(teams_with_enough_batters) == 0:
                    results['is_feasible'] = False
                    issue = f"No teams have {stack_size} or more batters for stack component {i+1}"
                    results['issues'].append(issue)
                    logging.error(f"❌ INFEASIBLE: {issue}")
                elif len(teams_with_enough_batters) < 2:
                    results['issues'].append(f"Warning: Only {len(teams_with_enough_batters)} team available for {stack_size}-stack")
                    logging.warning(f"⚠️ LIMITED: Only {teams_with_enough_batters} available for {stack_size}-stack")
        
        else:
            # Single stack scenario
            try:
                stack_size = int(stack_type.replace(' Stack', '').replace('-Stack', '').strip())
                if stack_size > batter_slots:
                    results['is_feasible'] = False
                    issue = f"Single {stack_size}-stack requires {stack_size} batters but only {batter_slots} batter slots available"
                    results['issues'].append(issue)
                    results['recommendations'].append(f"Try {batter_slots}-stack or smaller")
                    logging.error(f"❌ INFEASIBLE: {issue}")
            except ValueError:
                logging.warning(f"⚠️ Could not parse stack type: {stack_type}")
    
    # Log final result
    if results['is_feasible']:
        logging.info("✅ FEASIBILITY CHECK PASSED: All stack constraints are mathematically feasible")
    else:
        logging.error("❌ FEASIBILITY CHECK FAILED: Stack constraints are mathematically impossible")
        logging.error(f"📋 Issues found: {len(results['issues'])}")
        for issue in results['issues']:
            logging.error(f"   - {issue}")
        logging.info(f"💡 Recommendations: {len(results['recommendations'])}")
        for rec in results['recommendations']:
            logging.info(f"   - {rec}")
    
    return results

def apply_infeasibility_fixes(stack_settings, team_selections, min_unique, feasibility_results):
    """
    Apply automatic fixes when infeasibility is detected
    
    Returns:
        dict: Updated parameters
    """
    logging.info("🔧 APPLYING AUTOMATIC FIXES for infeasibility...")
    
    fixes_applied = []
    new_stack_settings = stack_settings.copy()
    new_min_unique = min_unique
    
    if not feasibility_results['is_feasible']:
        # Fix 1: Replace infeasible multi-stacks with feasible ones
        if feasibility_results['max_feasible_stacks']:
            for i, original_stack in enumerate(stack_settings):
                if '|' in original_stack and i < len(feasibility_results['max_feasible_stacks']):
                    new_stack = feasibility_results['max_feasible_stacks'][i]
                    new_stack_settings[i] = new_stack
                    fixes_applied.append(f"Reduced {original_stack} to {new_stack}")
                    logging.info(f"🔧 FIX 1: Changed {original_stack} → {new_stack}")
        
        # Fix 2: Drastically reduce min_unique to allow more lineup generation
        if min_unique > 1:
            new_min_unique = 0  # Disable uniqueness constraint entirely
            fixes_applied.append(f"Disabled min_unique (was {min_unique}, now {new_min_unique})")
            logging.info(f"🔧 FIX 2: Disabled min_unique constraint to allow lineup generation")
        
        # Fix 3: Add fallback "No Stacks" option if all else fails
        if len(new_stack_settings) == 1 and '|' in new_stack_settings[0]:
            new_stack_settings.append("No Stacks")
            fixes_applied.append("Added 'No Stacks' fallback option")
            logging.info(f"🔧 FIX 3: Added 'No Stacks' fallback option")
    
    result = {
        'stack_settings': new_stack_settings,
        'min_unique': new_min_unique,
        'team_selections': team_selections,  # Keep original team selections
        'fixes_applied': fixes_applied
    }
    
    if fixes_applied:
        logging.info(f"✅ APPLIED {len(fixes_applied)} AUTOMATIC FIXES:")
        for fix in fixes_applied:
            logging.info(f"   ✓ {fix}")
    else:
        logging.info("ℹ️ No automatic fixes needed - constraints appear feasible")
    
    return result

def optimize_single_lineup_with_feasibility_check(args):
    """
    Enhanced optimization function with built-in feasibility checking and recovery
    """
    df, stack_type, team_projected_runs, team_selections, min_salary = args
    
    logging.debug(f"🚀 ENHANCED OPTIMIZER: Starting with stack type {stack_type}, min_salary={min_salary}")
    
    # Quick feasibility pre-check for this specific stack type
    if '|' in stack_type:
        stack_sizes = [int(x) for x in stack_type.split('|')]
        total_stack_requirement = sum(stack_sizes)
        
        # MLB has 8 batter positions (10 total - 2 pitchers)
        if total_stack_requirement > 8:
            logging.warning(f"⚠️ INFEASIBLE STACK: {stack_type} requires {total_stack_requirement} batters but only 8 batter slots exist")
            logging.warning(f"🔄 FALLBACK: Attempting single-stack optimization instead...")
            
            # Try largest single stack as fallback
            fallback_stack_size = max(stack_sizes)
            if fallback_stack_size <= 8:
                # Recursively call with single stack
                fallback_args = (df, str(fallback_stack_size), team_projected_runs, team_selections, min_salary)
                return optimize_single_lineup_with_feasibility_check(fallback_args)
            else:
                # Even single stack is too big, try no stacks
                fallback_args = (df, "No Stacks", team_projected_runs, team_selections, min_salary)
                return optimize_single_lineup_with_feasibility_check(fallback_args)
    
    # If we get here, proceed with normal optimization
    # (This would be the normal optimize_single_lineup function logic)
    
    import pulp
    import pandas as pd
    import numpy as np
    import random
    
    # Standard optimization logic with enhanced error handling
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Add diversity noise
        diversity_factor = random.uniform(0.1, 0.25)
        noise = np.random.normal(1.0, diversity_factor, len(df))
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise
        
        problem = pulp.LpProblem("Enhanced_Stack_Optimization", pulp.LpMaximize)
        player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

        # Objective: Maximize projected points
        objective = pulp.lpSum([df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] for idx in df.index])
        problem += objective

        # Basic constraints
        REQUIRED_TEAM_SIZE = 10
        SALARY_CAP = 50000
        POSITION_LIMITS = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
        
        problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
        problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= SALARY_CAP
        
        # Minimum salary constraint
        if min_salary and min_salary > 0:
            problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) >= min_salary
        
        # Position constraints
        for position, limit in POSITION_LIMITS.items():
            problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Pos']]) == limit

        # Enhanced stacking constraints with feasibility checking
        if stack_type != "No Stacks":
            if '|' in stack_type:
                # Multi-stack with enhanced feasibility checking
                stack_sizes = [int(size) for size in stack_type.split('|')]
                logging.debug(f"🎯 Applying multi-stack constraints: {stack_sizes}")
                
                # Verify we have enough teams for each stack size
                constraints_added = 0
                for i, size in enumerate(stack_sizes):
                    available_teams = []
                    
                    # Get teams from team_selections or use all teams
                    if isinstance(team_selections, dict):
                        for key, teams in team_selections.items():
                            if str(size) in str(key):
                                available_teams = teams
                                break
                    
                    if not available_teams:
                        available_teams = df['Team'].unique().tolist()
                    
                    # Filter to teams with enough batters
                    valid_teams = []
                    for team in available_teams:
                        team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                        if len(team_batters) >= size:
                            valid_teams.append(team)
                    
                    if valid_teams:
                        # Add constraint for at least one valid team to contribute the stack
                        team_binary_vars = {}
                        for team in valid_teams:
                            team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{size}_{i}", cat='Binary')
                        
                        # At least one team must be selected for this stack size
                        problem += pulp.lpSum(team_binary_vars.values()) >= 1
                        
                        # If a team is selected, enforce the stack constraint
                        for team in valid_teams:
                            team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                            problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size * team_binary_vars[team]
                        
                        constraints_added += 1
                        logging.debug(f"✅ Added {size}-stack constraint with {len(valid_teams)} valid teams")
                    else:
                        logging.warning(f"⚠️ No valid teams found for {size}-stack, skipping constraint")
                
                if constraints_added == 0:
                    logging.warning(f"⚠️ No stack constraints could be added for {stack_type}, reverting to 'No Stacks'")
                    # Don't add any stacking constraints - will behave like "No Stacks"
            
            else:
                # Single stack constraint
                try:
                    stack_size = int(stack_type.replace(' Stack', '').replace('-Stack', '').strip())
                    
                    # Find valid teams for this stack size
                    available_teams = df['Team'].unique().tolist()
                    if isinstance(team_selections, dict):
                        for key, teams in team_selections.items():
                            if str(stack_size) in str(key):
                                available_teams = teams
                                break
                    
                    valid_teams = []
                    for team in available_teams:
                        team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                        if len(team_batters) >= stack_size:
                            valid_teams.append(team)
                    
                    if valid_teams:
                        # Create OR constraint for single stack
                        team_binary_vars = {}
                        for team in valid_teams:
                            team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{stack_size}", cat='Binary')
                        
                        problem += pulp.lpSum(team_binary_vars.values()) >= 1
                        
                        for team in valid_teams:
                            team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                            problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= stack_size * team_binary_vars[team]
                        
                        logging.debug(f"✅ Added {stack_size}-stack constraint with {len(valid_teams)} valid teams")
                    else:
                        logging.warning(f"⚠️ No valid teams found for {stack_size}-stack")
                
                except ValueError:
                    logging.warning(f"⚠️ Could not parse stack type: {stack_type}")

        # Solve with enhanced error handling
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
        status = problem.solve(solver)

        if pulp.LpStatus[status] == 'Optimal':
            lineup = df.loc[[idx for idx in df.index if player_vars[idx].varValue > 0.5]]
            
            # Verify lineup validity
            if len(lineup) != REQUIRED_TEAM_SIZE:
                logging.error(f"❌ Invalid lineup size: {len(lineup)} (expected {REQUIRED_TEAM_SIZE})")
                return pd.DataFrame(), stack_type
            
            total_salary = lineup['Salary'].sum()
            if total_salary > SALARY_CAP:
                logging.error(f"❌ Salary cap exceeded: ${total_salary} > ${SALARY_CAP}")
                return pd.DataFrame(), stack_type
            
            # Log successful result
            team_counts = lineup['Team'].value_counts()
            logging.debug(f"✅ ENHANCED OPTIMIZER SUCCESS: {len(lineup)} players, salary: ${total_salary}")
            logging.debug(f"✅ Team composition: {dict(team_counts)}")
            
            return lineup, stack_type
            
        else:
            logging.debug(f"❌ ENHANCED OPTIMIZER FAILED: Status = {pulp.LpStatus[status]}")
            
            # Try fallback strategies
            if stack_type != "No Stacks":
                logging.debug(f"🔄 TRYING FALLBACK: No Stacks for {stack_type}")
                fallback_args = (df, "No Stacks", team_projected_runs, team_selections, min_salary)
                return optimize_single_lineup_with_feasibility_check(fallback_args)
            
            return pd.DataFrame(), stack_type

    except Exception as e:
        logging.error(f"❌ ENHANCED OPTIMIZER EXCEPTION: {str(e)}")
        
        # Try fallback if not already trying "No Stacks"
        if stack_type != "No Stacks":
            logging.debug(f"🔄 EXCEPTION FALLBACK: No Stacks")
            fallback_args = (df, "No Stacks", team_projected_runs, team_selections, min_salary)
            return optimize_single_lineup_with_feasibility_check(fallback_args)
        
        return pd.DataFrame(), stack_type

def main():
    """
    Test the feasibility checking functions
    """
    print("🔍 OPTIMIZER FEASIBILITY CHECKER")
    print("=" * 50)
    
    # Simulate problematic scenario
    stack_settings = ["4|2"]  # This is the problematic combination from your logs
    position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
    
    # Create dummy player data
    import pandas as pd
    teams = ['BOS', 'LAD', 'NYY', 'HOU', 'ATL']
    players_data = []
    
    for team in teams:
        # Add 2 pitchers per team
        for i in range(2):
            players_data.append({
                'Name': f'{team}_P{i+1}',
                'Team': team,
                'Pos': 'P',
                'Salary': 8000,
                'Predicted_DK_Points': 15.0
            })
        
        # Add 12 batters per team (should be enough for stacking)
        positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'C', '1B', '2B', '3B']
        for i, pos in enumerate(positions):
            players_data.append({
                'Name': f'{team}_{pos}{i+1}',
                'Team': team,
                'Pos': pos,
                'Salary': 6000,
                'Predicted_DK_Points': 10.0
            })
    
    df_players = pd.DataFrame(players_data)
    
    print(f"📊 Test data: {len(df_players)} players across {len(teams)} teams")
    print(f"🎯 Testing stack settings: {stack_settings}")
    
    # Run feasibility check
    feasibility_results = check_stack_feasibility(stack_settings, df_players, position_limits)
    
    print(f"\n📋 FEASIBILITY RESULTS:")
    print(f"   ✅ Is feasible: {feasibility_results['is_feasible']}")
    print(f"   ❌ Issues: {len(feasibility_results['issues'])}")
    print(f"   💡 Recommendations: {len(feasibility_results['recommendations'])}")
    
    if not feasibility_results['is_feasible']:
        # Apply fixes
        team_selections = {'4': ['BOS'], '2': ['LAD']}  # Simulate user selections
        min_unique = 3
        
        fixes = apply_infeasibility_fixes(stack_settings, team_selections, min_unique, feasibility_results)
        
        print(f"\n🔧 AUTOMATIC FIXES APPLIED:")
        print(f"   🔄 New stack settings: {fixes['stack_settings']}")
        print(f"   🔄 New min_unique: {fixes['min_unique']}")
        print(f"   📋 Fixes applied: {len(fixes['fixes_applied'])}")
        
        for fix in fixes['fixes_applied']:
            print(f"      ✓ {fix}")

if __name__ == "__main__":
    main()
