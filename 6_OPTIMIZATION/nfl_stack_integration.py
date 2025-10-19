"""
Integration layer for NFL stacking system with existing optimizers
This module bridges the new NFL stack engine with the genetic algorithm optimizer
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from nfl_stack_config import (
    NFL_STACK_TYPES,
    RECOMMENDED_STACKS_BY_CONTEST,
    get_stack_display_names,
    get_all_stack_names
)
from nfl_stack_engine import NFLStackEngine, create_stack_engine, get_stack_display_string

logger = logging.getLogger(__name__)

# ============================================================================
# GUI INTEGRATION - Stack Selection Lists
# ============================================================================

def get_stack_types_for_gui(contest_type: str = None) -> List[Tuple[str, str]]:
    """
    Get list of (stack_key, display_name) tuples for GUI dropdowns
    
    Args:
        contest_type: Optional filter by contest type
        
    Returns:
        List of (key, name) tuples sorted by popularity
    """
    display_names = get_stack_display_names()
    
    if contest_type and contest_type in RECOMMENDED_STACKS_BY_CONTEST:
        # Filter to recommended stacks for this contest type
        recommended_keys = RECOMMENDED_STACKS_BY_CONTEST[contest_type]
        filtered = [(key, display_names[key]) for key in recommended_keys if key in display_names]
        # Add remaining stacks at the end
        remaining = [(key, name) for key, name in display_names.items() 
                    if key not in recommended_keys]
        return filtered + remaining
    else:
        # Return all stacks
        return [(key, display_names[key]) for key in display_names.keys()]

def get_default_stack_type(contest_type: str = 'gpp_tournament') -> str:
    """
    Get the default/recommended stack type for a contest
    
    Args:
        contest_type: Contest type identifier
        
    Returns:
        Stack type key (default: 'qb_wr')
    """
    recommended = RECOMMENDED_STACKS_BY_CONTEST.get(contest_type, [])
    if recommended:
        return recommended[0]
    return 'qb_wr'

# ============================================================================
# TEAM COMBINATION GENERATION
# ============================================================================

def generate_team_stack_combinations(df_players: pd.DataFrame, 
                                    contest_type: str = 'gpp_tournament',
                                    selected_stack_types: List[str] = None,
                                    max_combinations: int = 100) -> List[Dict[str, Any]]:
    """
    Generate all feasible team/stack combinations for lineup optimization
    
    Args:
        df_players: DataFrame with player pool
        contest_type: Contest type for filtering stacks
        selected_stack_types: Optional list of specific stack types to use
        max_combinations: Maximum number of combinations to generate
        
    Returns:
        List of combination dicts with keys: stack_type, team, opponent, display_name
    """
    engine = create_stack_engine(df_players)
    
    # Determine which stack types to use
    if selected_stack_types:
        stack_types_to_use = selected_stack_types
    else:
        stack_types_to_use = RECOMMENDED_STACKS_BY_CONTEST.get(
            contest_type, 
            ['qb_wr', 'qb_2wr', 'qb_wr_te']
        )
    
    combinations = []
    teams = df_players['Team'].unique()
    teams = [t for t in teams if pd.notna(t)]
    
    logger.info(f"Generating combinations for {len(teams)} teams with stack types: {stack_types_to_use}")
    
    for team in teams:
        opponent_info = engine.get_teams_in_game(team)
        opponent = opponent_info[1]
        
        for stack_type in stack_types_to_use:
            # Check feasibility
            is_feasible, msg = engine.validate_stack_feasibility(stack_type, team, opponent)
            
            if is_feasible:
                display_name = get_stack_display_string(stack_type, team, opponent)
                combinations.append({
                    'stack_type': stack_type,
                    'team': team,
                    'opponent': opponent,
                    'display_name': display_name,
                    'correlation': NFL_STACK_TYPES[stack_type].get('correlation', 0.0),
                    'leverage': NFL_STACK_TYPES[stack_type].get('leverage', 1.0)
                })
            else:
                logger.debug(f"  Skipping {stack_type} for {team}: {msg}")
            
            # Stop if we've hit the limit
            if len(combinations) >= max_combinations:
                break
        
        if len(combinations) >= max_combinations:
            break
    
    logger.info(f"Generated {len(combinations)} feasible team/stack combinations")
    return combinations

# ============================================================================
# OPTIMIZATION INTEGRATION
# ============================================================================

def apply_nfl_stack_to_optimization(problem, player_vars: dict, df_players: pd.DataFrame,
                                   stack_type: str, team: str, opponent: Optional[str] = None):
    """
    Apply NFL stacking constraints to a PuLP optimization problem
    
    This is the main function called during lineup optimization
    
    Args:
        problem: PuLP LpProblem instance
        player_vars: Dict mapping DataFrame index to PuLP binary variable
        df_players: DataFrame with all players
        stack_type: Stack type key from nfl_stack_config
        team: Primary team for the stack
        opponent: Opponent team (required for game stacks)
    """
    engine = NFLStackEngine(df_players)
    
    # Validate stack first
    is_feasible, msg = engine.validate_stack_feasibility(stack_type, team, opponent)
    if not is_feasible:
        logger.error(f"Stack validation failed: {msg}")
        return
    
    # Apply stack constraints
    engine.apply_stack_constraints_pulp(problem, player_vars, stack_type, team, opponent)
    
    logger.info(f"Applied {stack_type} stack constraints for {team}" + 
               (f" vs {opponent}" if opponent else ""))

def optimize_lineup_with_stack(df_players: pd.DataFrame,
                              stack_type: str,
                              team: str,
                              opponent: Optional[str] = None,
                              salary_cap: int = 50000,
                              min_salary: int = 48000,
                              position_limits: Dict[str, int] = None) -> Tuple[pd.DataFrame, float]:
    """
    Complete lineup optimization with NFL stack constraints
    
    This is a standalone optimization function that can be called independently
    
    Args:
        df_players: DataFrame with player pool
        stack_type: Stack type key
        team: Primary team
        opponent: Opponent team (if game stack)
        salary_cap: Maximum salary (default 50000 for DraftKings)
        min_salary: Minimum salary spend
        position_limits: Dict of position -> count requirements
        
    Returns:
        (lineup_df, total_projection) tuple
    """
    import pulp
    
    # Default NFL position limits
    if position_limits is None:
        position_limits = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'FLEX': 1,  # Will be handled specially
            'DST': 1
        }
    
    # Create optimization problem
    problem = pulp.LpProblem("NFL_DFS_Stack", pulp.LpMaximize)
    
    # Create binary variables for each player
    player_vars = {}
    for idx in df_players.index:
        player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    # Objective: Maximize fantasy points
    projection_col = None
    for col in ['Fantasy_Points', 'FantasyPoints', 'Predicted_DK_Points', 'Projection']:
        if col in df_players.columns:
            projection_col = col
            break
    
    if projection_col is None:
        raise ValueError("No projection column found in data")
    
    problem += pulp.lpSum([
        player_vars[idx] * df_players.at[idx, projection_col]
        for idx in df_players.index
    ])
    
    # Constraint: Salary cap
    problem += pulp.lpSum([
        player_vars[idx] * df_players.at[idx, 'Salary']
        for idx in df_players.index
    ]) <= salary_cap
    
    # Constraint: Minimum salary
    problem += pulp.lpSum([
        player_vars[idx] * df_players.at[idx, 'Salary']
        for idx in df_players.index
    ]) >= min_salary
    
    # Constraint: Exactly 9 players
    problem += pulp.lpSum([player_vars[idx] for idx in df_players.index]) == 9
    
    # Position constraints (excluding FLEX for now)
    for pos, limit in position_limits.items():
        if pos == 'FLEX':
            continue  # Handle FLEX separately
        
        eligible_players = df_players[df_players['Position'] == pos].index
        problem += pulp.lpSum([player_vars[idx] for idx in eligible_players]) == limit
    
    # FLEX constraint: Total RB + WR + TE = 7 (which includes the 2 RB, 3 WR, 1 TE + 1 FLEX)
    flex_eligible = df_players[df_players['Position'].isin(['RB', 'WR', 'TE'])].index
    problem += pulp.lpSum([player_vars[idx] for idx in flex_eligible]) == 7
    
    # Apply NFL stack constraints
    apply_nfl_stack_to_optimization(problem, player_vars, df_players, stack_type, team, opponent)
    
    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract lineup
    selected_indices = [idx for idx in df_players.index if pulp.value(player_vars[idx]) == 1]
    lineup = df_players.loc[selected_indices].copy()
    
    total_projection = lineup[projection_col].sum() if len(lineup) > 0 else 0.0
    
    return lineup, total_projection

# ============================================================================
# BATCH OPTIMIZATION FOR MULTIPLE LINEUPS
# ============================================================================

def generate_multiple_lineups_with_stacks(df_players: pd.DataFrame,
                                         stack_combinations: List[Dict[str, Any]],
                                         lineups_per_combination: int = 5,
                                         max_total_lineups: int = 100,
                                         diversity_factor: float = 0.15) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Generate multiple diverse lineups using different stack combinations
    
    Args:
        df_players: Player pool DataFrame
        stack_combinations: List of combination dicts from generate_team_stack_combinations
        lineups_per_combination: Number of lineups to generate per stack combo
        max_total_lineups: Maximum total lineups to return
        diversity_factor: Amount of randomness to add (0.0-1.0)
        
    Returns:
        List of (lineup_df, metadata_dict) tuples
    """
    import numpy as np
    
    all_lineups = []
    logger.info(f"Generating lineups for {len(stack_combinations)} stack combinations")
    
    for combo in stack_combinations:
        stack_type = combo['stack_type']
        team = combo['team']
        opponent = combo['opponent']
        
        for i in range(lineups_per_combination):
            try:
                # Add some randomness to projections for diversity
                df_modified = df_players.copy()
                if diversity_factor > 0:
                    proj_col = None
                    for col in ['Fantasy_Points', 'FantasyPoints']:
                        if col in df_modified.columns:
                            proj_col = col
                            break
                    
                    if proj_col:
                        noise = np.random.uniform(1 - diversity_factor, 1 + diversity_factor, len(df_modified))
                        df_modified[proj_col] = df_modified[proj_col] * noise
                
                # Generate lineup
                lineup, projection = optimize_lineup_with_stack(
                    df_modified, stack_type, team, opponent
                )
                
                if len(lineup) == 9:  # Valid lineup
                    metadata = {
                        'stack_type': stack_type,
                        'team': team,
                        'opponent': opponent,
                        'display_name': combo['display_name'],
                        'projection': projection,
                        'iteration': i + 1
                    }
                    all_lineups.append((lineup, metadata))
                    
                    if len(all_lineups) >= max_total_lineups:
                        break
                        
            except Exception as e:
                logger.error(f"Error generating lineup for {combo['display_name']}: {str(e)}")
                continue
        
        if len(all_lineups) >= max_total_lineups:
            break
    
    logger.info(f"Generated {len(all_lineups)} total lineups")
    return all_lineups

# ============================================================================
# LINEUP VALIDATION
# ============================================================================

def validate_nfl_lineup(lineup: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that a lineup meets all NFL DFS requirements
    
    Args:
        lineup: DataFrame of 9 players
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check lineup size
    if len(lineup) != 9:
        errors.append(f"Lineup must have 9 players, has {len(lineup)}")
    
    # Check positions
    required_positions = {
        'QB': 1,
        'RB': (2, 3),  # 2-3 RBs (2 + possible FLEX)
        'WR': (3, 4),  # 3-4 WRs (3 + possible FLEX)
        'TE': (1, 2),  # 1-2 TEs (1 + possible FLEX)
        'DST': 1
    }
    
    for pos, count in required_positions.items():
        actual_count = len(lineup[lineup['Position'] == pos])
        
        if isinstance(count, tuple):
            min_count, max_count = count
            if actual_count < min_count or actual_count > max_count:
                errors.append(f"{pos}: Expected {min_count}-{max_count}, got {actual_count}")
        else:
            if actual_count != count:
                errors.append(f"{pos}: Expected {count}, got {actual_count}")
    
    # Check salary
    total_salary = lineup['Salary'].sum()
    if total_salary > 50000:
        errors.append(f"Salary ${total_salary} exceeds cap of $50,000")
    if total_salary < 48000:
        errors.append(f"Salary ${total_salary} below minimum of $48,000")
    
    # Check for duplicates
    if lineup['Name'].duplicated().any():
        errors.append("Lineup contains duplicate players")
    
    is_valid = len(errors) == 0
    return (is_valid, errors)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def format_lineup_for_draftkings(lineup: pd.DataFrame) -> pd.DataFrame:
    """
    Format lineup for DraftKings CSV upload
    
    Args:
        lineup: DataFrame of 9 players
        
    Returns:
        Single-row DataFrame in DraftKings format
    """
    # Sort players by position for DraftKings format
    position_order = {'QB': 0, 'RB': 1, 'WR': 3, 'TE': 6, 'DST': 8}
    lineup_sorted = lineup.copy()
    lineup_sorted['_sort'] = lineup_sorted['Position'].map(position_order).fillna(99)
    lineup_sorted = lineup_sorted.sort_values('_sort')
    
    # Create DraftKings row
    dk_row = {}
    dk_positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
    
    pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0}
    
    for i, pos_label in enumerate(dk_positions):
        if pos_label == 'FLEX':
            # FLEX is the "extra" RB, WR, or TE
            flex_players = lineup_sorted[lineup_sorted['Position'].isin(['RB', 'WR', 'TE'])]
            # Get the one not yet assigned
            for _, player in flex_players.iterrows():
                p_pos = player['Position']
                if pos_counts[p_pos] >= (2 if p_pos == 'RB' else (3 if p_pos == 'WR' else 1)):
                    dk_row[pos_label] = player['Name']
                    break
        else:
            # Regular position
            pos_players = lineup_sorted[lineup_sorted['Position'] == pos_label]
            if pos_counts[pos_label] < len(pos_players):
                player = pos_players.iloc[pos_counts[pos_label]]
                dk_row[pos_label] = player['Name']
                pos_counts[pos_label] += 1
    
    return pd.DataFrame([dk_row])

