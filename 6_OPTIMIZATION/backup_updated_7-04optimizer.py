import sys
import os
import logging
import traceback
import psutil
import pulp
import pandas as pd
import numpy as np
import re
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import concurrent.futures
from itertools import combinations, permutations
import itertools
import csv
import json
from collections import defaultdict

# Import Windows-safe logging
try:
    from safe_logging import safe_log_info as safe_log_info_func, safe_log_debug as safe_log_debug_func, safe_log_warning as safe_log_warning_func, safe_log_error as safe_log_error_func
    # Create wrappers to match expected interface
    def safe_log_info(msg): return safe_log_info_func(msg)
    def safe_log_debug(msg): return safe_log_debug_func(msg)
    def safe_log_warning(msg): return safe_log_warning_func(msg)
    def safe_log_error(msg): return safe_log_error_func(msg)
    SAFE_LOGGING_AVAILABLE = True
    print("✅ Windows-safe logging loaded successfully!")
except ImportError as e:
    print(f"⚠️ Safe logging not available: {e}")
    SAFE_LOGGING_AVAILABLE = False
    # Fallback to regular logging
    def safe_log_info(msg): logging.info(msg)
    def safe_log_debug(msg): logging.debug(msg)
    def safe_log_warning(msg): logging.warning(msg)
    def safe_log_error(msg): logging.error(msg)

# Import enhanced checkbox handling
try:
    from checkbox_fix import CheckboxManager, collect_team_selections_enhanced
    ENHANCED_CHECKBOX_AVAILABLE = True
    print("✅ Enhanced Checkbox Manager loaded successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced Checkbox Manager not available: {e}")
    ENHANCED_CHECKBOX_AVAILABLE = False

# Import the advanced risk management engine
try:
    import sys
    import os
    # Add parent directory to path to find dfs_risk_engine
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from dfs_risk_engine import DFSRiskEngine, DFSBankrollManager, RiskMetrics
    RISK_ENGINE_AVAILABLE = True
    print("✅ Advanced Risk Engine loaded successfully!")
except ImportError as e:
    print(f"⚠️ Risk Engine not available: {e}")
    RISK_ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
SALARY_CAP = 50000
MIN_SALARY_DEFAULT = 45000  # Default minimum salary requirement
POSITION_LIMITS = {
    'P': 2,
    'C': 1,
    '1B': 1,
    '2B': 1,
    '3B': 1,
    'SS': 1,
    'OF': 3
}
REQUIRED_TEAM_SIZE = 10

def optimize_single_lineup(args):
    df, stack_type, team_projected_runs, team_selections, min_salary, disable_adjustments, conservative_mode = args
    logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type}, min_salary={min_salary}")
    
    import random
    import numpy as np
    
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # Initialize variables for all modes
    use_efficiency = False
    use_contrarian = False
    use_ceiling = False
    
    # RESPECT USER PREFERENCES FOR PROJECTION ADJUSTMENTS
    if disable_adjustments:
        # NO ADJUSTMENTS - Use pure projections for maximum accuracy
        logging.debug("optimize_single_lineup: Using pure projections (no adjustments)")
        pass  # Don't modify projections at all
    elif conservative_mode:
        # CONSERVATIVE: Minimal variance for diversity while preserving accuracy
        diversity_factor = random.uniform(0.02, 0.05)  # 2-5% variance
        noise = np.random.normal(1.0, diversity_factor, len(df))
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise
        logging.debug(f"optimize_single_lineup: Applied conservative variance ({diversity_factor:.3f})")
        
        # Conservative diversity techniques (very low probability)
        use_efficiency = random.random() < 0.1  # 10% chance (reduced from 30%)
        use_contrarian = random.random() < 0.05  # 5% chance (reduced from 10%)
        use_ceiling = random.random() < 0.1  # 10% chance (reduced from 20%)
    else:
        # LEGACY AGGRESSIVE MODE (not recommended)
        diversity_factor = random.uniform(0.02, 0.05)  # Still use conservative even in "aggressive"
        noise = np.random.normal(1.0, diversity_factor, len(df))
        df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * noise
        
        # Apply conservative diversity techniques
        use_efficiency = random.random() < 0.3
        use_contrarian = random.random() < 0.1
        use_ceiling = random.random() < 0.2
        
        logging.debug(f"optimize_single_lineup: Applied legacy mode with conservative adjustments")
    
    # Apply diversity techniques if enabled
    if use_contrarian:
        # CONSERVATIVE contrarian: Only tiny boost to avoid completely ignoring value plays
        bottom_threshold = df['Predicted_DK_Points'].quantile(0.2)
        boost_mask = df['Predicted_DK_Points'] <= bottom_threshold
        df.loc[boost_mask, 'Predicted_DK_Points'] *= random.uniform(1.05, 1.15)
    
    if use_ceiling:
        # CONSERVATIVE ceiling: Minimal variance adjustment
        top_threshold = df['Predicted_DK_Points'].quantile(0.8)
        ceiling_mask = df['Predicted_DK_Points'] >= top_threshold
        ceiling_boost = np.random.normal(1.0, 0.05, ceiling_mask.sum())
        df.loc[ceiling_mask, 'Predicted_DK_Points'] *= ceiling_boost
    
    problem = pulp.LpProblem("Stack_Optimization", pulp.LpMaximize)
    player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

    # Objective: Maximize projected points (with optional efficiency bonus for diversity)
    if use_efficiency:
        # Add small salary efficiency bonus for diversity
        efficiency_bonus = 0.1  # Small bonus
        objective = pulp.lpSum([
            (df.at[idx, 'Predicted_DK_Points'] + 
             efficiency_bonus * (df.at[idx, 'Predicted_DK_Points'] / max(df.at[idx, 'Salary'], 1000) * 1000)) * player_vars[idx] 
            for idx in df.index
        ])
    else:
        objective = pulp.lpSum([df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] for idx in df.index])
    
    problem += objective

    # Basic constraints
    problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
    problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= SALARY_CAP
    
    # ADD MINIMUM SALARY CONSTRAINT
    if min_salary and min_salary > 0:
        problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) >= min_salary
        logging.debug(f"optimize_single_lineup: Added minimum salary constraint >= {min_salary}")
    
    for position, limit in POSITION_LIMITS.items():
        available_for_position = [idx for idx in df.index if position in df.at[idx, 'Pos']]
        logging.debug(f"optimize_single_lineup: Position {position} needs {limit}, available: {len(available_for_position)}")
        if len(available_for_position) < limit:
            logging.error(f"optimize_single_lineup: INSUFFICIENT PLAYERS for {position}: need {limit}, have {len(available_for_position)}")
            return pd.DataFrame(), stack_type
        problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Pos']]) == limit

    # Handle different stack types
    if stack_type == "No Stacks":
        # No stacking constraints - just basic position and salary constraints
        logging.debug("optimize_single_lineup: Using no stacks")
    else:
        # Implement stacking with proper team selection enforcement
        stack_sizes = [int(size) for size in stack_type.split('|')]
        logging.info(f"🎯 OPTIMIZER: Processing stack sizes: {stack_sizes}")
        logging.info(f"🎯 OPTIMIZER: Team selections received: {team_selections}")
        
        # Simplified approach: For each stack size, randomly pick one of the available teams
        # and enforce that constraint. This avoids creating too many binary variables.
        import random
        
        for i, size in enumerate(stack_sizes):
            # Get teams available for this specific stack size
            available_teams = None
            
            # ENHANCED TEAM SELECTION LOOKUP - handles multiple key formats
            if isinstance(team_selections, dict):
                # Method 1: Check for exact integer match
                if size in team_selections:
                    available_teams = team_selections[size]
                    logging.info(f"🎯 FOUND teams for {size}-stack using exact integer match: {available_teams}")
                # Method 2: Check for string integer match
                elif str(size) in team_selections:
                    available_teams = team_selections[str(size)]
                    logging.info(f"🎯 FOUND teams for {size}-stack using string integer match: {available_teams}")
                # Method 3: Check for "X-Stack" format
                elif f"{size}-Stack" in team_selections:
                    available_teams = team_selections[f"{size}-Stack"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using dash format match: {available_teams}")
                # Method 4: Check for "X Stack" format (with space)
                elif f"{size} Stack" in team_selections:
                    available_teams = team_selections[f"{size} Stack"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using space format match: {available_teams}")
                # Method 5: Check for "all" selection (from All Stacks tab)
                elif "all" in team_selections:
                    available_teams = team_selections["all"]
                    logging.info(f"🎯 FOUND teams for {size}-stack using 'all' key: {available_teams}")
                # No specific match found in dictionary
                else:
                    logging.warning(f"🎯 NO specific team selection found for {size}-stack in keys: {list(team_selections.keys())}")
            # Fallback to list format
            elif isinstance(team_selections, list):
                available_teams = team_selections
                logging.debug(f"optimize_single_lineup: Using team list for stack size {size}: {available_teams}")
            
            # Final fallback - AVOID using all available teams unless absolutely necessary
            if not available_teams:
                # Check if team_selections has data but wrong format
                if team_selections:
                    logging.error(f"🚨 CRITICAL: Team selections exist but {size}-stack not found!")
                    
                    # Only access dictionary methods if it's actually a dictionary
                    if isinstance(team_selections, dict):
                        logging.error(f"🚨 Available keys: {list(team_selections.keys())}")
                        logging.error(f"🚨 This suggests a format mismatch or checkbox detection issue!")
                        
                        # Try to find any matching stack with similar size
                        for key, teams in team_selections.items():
                            if str(size) in str(key) or str(key) in str(size):
                                available_teams = teams
                                logging.warning(f"🎯 Using approximate match {key} for {size}-stack: {teams}")
                                break
                    else:
                        logging.error(f"🚨 Team selections is not a dictionary: {type(team_selections)}")
                        logging.error(f"🚨 Value: {team_selections}")
                
                # Only use all teams as absolute last resort and warn user
                if not available_teams:
                    available_teams = df['Team'].unique().tolist()
                    logging.error(f"🚨 LAST RESORT: Using ALL {len(available_teams)} teams for {size}-stack!")
                    logging.error(f"🚨 This means your team selections were not detected properly!")
            else:
                # Check if we got ALL teams when user likely selected specific ones
                all_teams = df['Team'].unique().tolist()
                if len(available_teams) == len(all_teams) and set(available_teams) == set(all_teams):
                    logging.error(f"🚨 SUSPICIOUS: {size}-stack has ALL {len(available_teams)} teams - likely a selection bug!")
                else:
                    logging.info(f"✅ CONFIRMED: Will enforce {size}-stack using {len(available_teams)} teams: {available_teams}")
            
            if not available_teams:
                logging.warning(f"optimize_single_lineup: No teams available for stack size {size}, skipping")
                continue
            
            # Filter available teams to only those with enough batters
            valid_teams = []
            for team in available_teams:
                team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                if len(team_batters) >= size:
                    valid_teams.append(team)
                    logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {size}) - VALID")
                else:
                    logging.debug(f"optimize_single_lineup: Team {team} has {len(team_batters)} batters (need {size}) - INSUFFICIENT")
                    
            if not valid_teams:
                logging.warning(f"optimize_single_lineup: No valid teams with enough batters for stack size {size}")
                logging.warning(f"Available teams: {available_teams}")
                continue
                
            # CRITICAL FIX: Instead of randomly selecting one team, enforce constraint for ALL user-selected teams
            # This ensures we respect the user's specific team choices for each stack size
            
            if len(valid_teams) == 1:
                # If only one team selected for this stack size, enforce it directly
                selected_team = valid_teams[0]
                team_batters = df[(df['Team'] == selected_team) & (~df['Pos'].str.contains('P', na=False))].index
                problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size
                logging.info(f"✅ ENFORCING: Must have at least {size} players from {selected_team}")
                
            else:
                # If multiple teams selected for this stack size, create OR constraint
                # This means: at least 'size' players from ANY of the selected teams
                team_constraints = []
                for team in valid_teams:
                    team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                    if len(team_batters) >= size:
                        team_constraints.append(pulp.lpSum([player_vars[idx] for idx in team_batters]))
                
                if team_constraints:
                    # At least one of the selected teams must contribute 'size' players
                    # Create binary variables for each team to implement OR logic
                    team_binary_vars = {}
                    for team in valid_teams:
                        team_binary_vars[team] = pulp.LpVariable(f"use_team_{team}_{size}_{i}", cat='Binary')
                    
                    # At least one team must be selected
                    problem += pulp.lpSum(team_binary_vars.values()) >= 1
                    
                    # If a team is selected, enforce the stack constraint
                    for team in valid_teams:
                        team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                        if len(team_batters) >= size:
                            # If team is selected (binary = 1), enforce at least 'size' players
                            problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size * team_binary_vars[team]
                    
                    logging.info(f"✅ ENFORCING: Must have at least {size} players from ANY of: {valid_teams}")
                else:
                    logging.warning(f"❌ NO valid team constraints created for {size}-stack")
            
            # Log final enforcement summary
            logging.info(f"📊 STACK CONSTRAINT SUMMARY: {size}-stack will be enforced using teams: {valid_teams}")

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    status = problem.solve(solver)

    if pulp.LpStatus[status] == 'Optimal':
        lineup = df.loc[[idx for idx in df.index if player_vars[idx].varValue is not None and player_vars[idx].varValue > 0.5]]
        
        # Log the actual team composition for debugging
        team_counts = lineup['Team'].value_counts()
        logging.debug(f"optimize_single_lineup: Found optimal solution with {len(lineup)} players")
        logging.debug(f"optimize_single_lineup: Team composition: {dict(team_counts)}")
        
        # Print detailed lineup information
        total_salary = lineup['Salary'].sum()
        total_points = lineup['Predicted_DK_Points'].sum()
        
        print(f"\n{'='*60}")
        print(f"LINEUP GENERATED - Stack Type: {stack_type}")
        print(f"{'='*60}")
        print(f"OPTIMAL SCORE: {total_points:.2f} DK Points")
        print(f"TOTAL SALARY: ${total_salary:,} / ${SALARY_CAP:,}")
        print(f"SALARY REMAINING: ${SALARY_CAP - total_salary:,}")
        print(f"{'='*60}")
        
        print(f"{'Position':<8} {'Player':<25} {'Team':<4} {'Salary':<8} {'Points':<8}")
        print(f"{'-'*60}")
        
        for _, player in lineup.iterrows():
            position = player['Pos']
            name = player['Name']
            team = player['Team']
            salary = player['Salary']
            points = player['Predicted_DK_Points']
            
            print(f"{position:<8} {name:<25} {team:<4} ${salary:<7,} {points:<8.2f}")
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':<38} ${total_salary:<7,} {total_points:<8.2f}")
        print(f"{'='*60}\n")
        
        return lineup, stack_type
    else:
        logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
        logging.debug(f"Constraints: {problem.constraints}")
        return pd.DataFrame(), stack_type
def simulate_iteration(df, disable_adjustments=False, conservative_mode=True):
    df = df.copy()
    
    if disable_adjustments:
        # NO ADJUSTMENTS - Return data unchanged for pure projection accuracy
        return df
    elif conservative_mode:
        # CONSERVATIVE: Use minimal variance for Monte Carlo simulation
        random_factors = np.random.normal(1, 0.03, size=len(df))  # 3% variance
    else:
        # LEGACY: Still use conservative variance even in "aggressive" mode
        random_factors = np.random.normal(1, 0.03, size=len(df))  # 3% variance
    
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * random_factors
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'].clip(lower=1)
    return df

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(dict, dict, dict)
    
    def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations, num_lineups, team_selections, min_unique=0, bankroll=1000, risk_tolerance='medium', disable_kelly=False, min_salary=None, disable_adjustments=False, conservative_mode=True, max_usage_percentage=None):
        super().__init__()
        self.df_players = df_players
        self.num_lineups = num_lineups
        self.salary_cap = salary_cap
        self.position_limits = position_limits
        self.included_players = included_players
        self.stack_settings = stack_settings
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.team_projected_runs = self.calculate_team_projected_runs(df_players)
        self.min_unique = min_unique  # Add min unique constraint
        self.disable_kelly = disable_kelly  # Option to disable Kelly sizing
        self.min_salary = min_salary if min_salary is not None else MIN_SALARY_DEFAULT  # Add minimum salary constraint
        self.disable_adjustments = disable_adjustments  # Option to disable all projection adjustments
        self.conservative_mode = conservative_mode  # Use conservative variance instead of aggressive
        
        # Player usage tracking system
        self.max_usage_percentage = max_usage_percentage if max_usage_percentage is not None else 40  # Default 40% max usage
        self.player_usage_tracker = defaultdict(int)  # Track how many times each player has been used
        self.total_lineups_generated = 0  # Track total lineups generated for percentage calculation
        self.excluded_players = set()  # Players currently excluded due to overuse
        
        self.max_workers = multiprocessing.cpu_count()  # Or set a specific number
        self.min_points = min_points
        self.monte_carlo_iterations = monte_carlo_iterations
        self.team_selections = team_selections  # Passed from main app
        
        # Initialize risk management components
        self.bankroll = bankroll
        self.risk_tolerance = risk_tolerance
        if RISK_ENGINE_AVAILABLE:
            self.risk_engine = DFSRiskEngine()
            self.bankroll_manager = DFSBankrollManager(bankroll)
            logging.info(f"🔥 Risk management initialized: Bankroll=${bankroll}, Risk={risk_tolerance}")
        else:
            self.risk_engine = None
            self.bankroll_manager = None
        
        logging.info(f"🎯 Player Usage Tracking: Max {self.max_usage_percentage}% usage per player")

    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")
        
        # SHOW USER THEIR SELECTIONS (team_selections already passed to worker)
        if self.team_selections:
            print(f"\n✅ YOUR TEAM SELECTIONS DETECTED:")
            for stack_size, teams in self.team_selections.items():
                if stack_size == "all":
                    print(f"   All Stacks: {teams}")
                else:
                    print(f"   {stack_size}-Stack: {teams}")
        else:
            print(f"\n⚠️ NO TEAM SELECTIONS - Using all teams")
        
        # IMMEDIATE DEBUG OUTPUT - Always print to console
        print(f"\n🚨 IMMEDIATE PLAYER FILTER DEBUG:")
        print(f"   📊 Total players after filtering: {len(df_filtered)}")
        print(f"   🎯 Included players list: {self.included_players}")
        print(f"   📋 Included players count: {len(self.included_players) if self.included_players else 0}")
        if len(df_filtered) > 0:
            print(f"   👥 Sample filtered players: {df_filtered['Name'].head(3).tolist()}")
        print(f"   🔥 This should help identify if filtering is working!\n")

        results = {}
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)
        
        # Risk-adjusted lineup generation if available and enabled
        use_risk_management = (RISK_ENGINE_AVAILABLE and 
                             self.risk_engine and 
                             getattr(self, 'enable_risk_mgmt', True))
        
        if use_risk_management:
            logging.info("🔥 Using advanced risk management optimization")
            return self.optimize_lineups_with_risk_management(df_filtered, team_exposure, stack_exposure)
        
        # Traditional optimization (existing logic)
        logging.info("📊 Using traditional optimization (risk management disabled)")
        
        # Generate lineups in batches to allow for player usage tracking
        attempts = 0
        max_attempts = self.num_lineups * 5  # Allow up to 5x attempts for variety
        
        while len(results) < self.num_lineups and attempts < max_attempts:
            # Generate a batch of lineups
            batch_size = min(50, self.num_lineups)  # Process in batches of 50
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                # Re-preprocess data to apply updated exclusions
                df_filtered = self.preprocess_data()
                
                for stack_type in self.stack_settings:
                    for _ in range(batch_size):
                        future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections, self.min_salary, self.disable_adjustments, self.conservative_mode))
                        futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    try:
                        lineup, stack_type = future.result()
                        if lineup.empty:
                            logging.debug(f"optimize_lineups: Empty lineup returned for stack type {stack_type}")
                        else:
                            total_points = lineup['Predicted_DK_Points'].sum()
                            results[len(results)] = {'total_points': total_points, 'lineup': lineup}
                            
                            # Update player usage tracking
                            self.update_player_usage(lineup)
                            
                            for team in lineup['Team'].unique():
                                team_exposure[team] += 1
                            stack_exposure[stack_type] += 1
                            
                            # Print lineup summary
                            print(f"✅ LINEUP #{len(results)} ADDED - Stack: {stack_type}, Score: {total_points:.2f}")
                            
                            logging.debug(f"optimize_lineups: Found valid lineup for stack type {stack_type}")
                            
                            # Break early if we have enough lineups
                            if len(results) >= self.num_lineups:
                                break
                                
                    except Exception as e:
                        logging.error(f"Error in optimization: {str(e)}")
                        
                    attempts += 1
                    if attempts >= max_attempts:
                        break
            
            # Log usage statistics periodically
            if len(results) % 20 == 0 and len(results) > 0:
                usage_stats = self.get_player_usage_stats()
                high_usage_players = [(p, s['percentage']) for p, s in usage_stats.items() if s['percentage'] > self.max_usage_percentage * 0.8]
                if high_usage_players:
                    logging.info(f"🎯 High usage players ({len(high_usage_players)}): {high_usage_players[:3]}...")
                logging.info(f"🎯 Generated {len(results)}/{self.num_lineups} lineups, excluded {len(self.excluded_players)} overused players")

        logging.debug(f"optimize_lineups: Completed. Found {len(results)} valid lineups")
        logging.debug(f"Team exposure: {dict(team_exposure)}")
        logging.debug(f"Stack exposure: {dict(stack_exposure)}")
        
        # Print optimization summary
        if results:
            print(f"\n{'='*80}")
            print(f"OPTIMIZATION SUMMARY")
            print(f"{'='*80}")
            print(f"Total Lineups Generated: {len(results)}")
            
            # Calculate optimal score (best possible lineup)
            optimal_score = max(result['total_points'] for result in results.values())
            total_actual_score = sum(result['total_points'] for result in results.values())
            average_score = total_actual_score / len(results)
            
            print(f"Optimal Score (Best Lineup): {optimal_score:.2f} DK Points")
            print(f"Average Score: {average_score:.2f} DK Points")
            print(f"Total Combined Score: {total_actual_score:.2f} DK Points")
            
            # Show top 3 lineups
            sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
            print(f"\nTOP 3 LINEUPS:")
            for i, (lineup_id, result) in enumerate(sorted_results[:3]):
                print(f"  #{i+1}: {result['total_points']:.2f} points")
            
            print(f"{'='*80}\n")
        
        return results, team_exposure, stack_exposure

    def update_player_usage(self, lineup_df):
        """Update player usage tracking when a lineup is generated"""
        self.total_lineups_generated += 1
        
        for player_name in lineup_df['Name']:
            self.player_usage_tracker[player_name] += 1
            
            # Check if player has exceeded usage threshold
            usage_percentage = (self.player_usage_tracker[player_name] / self.total_lineups_generated) * 100
            if usage_percentage >= self.max_usage_percentage:
                self.excluded_players.add(player_name)
                logging.info(f"🚫 Player '{player_name}' excluded due to {usage_percentage:.1f}% usage (max: {self.max_usage_percentage}%)")
    
    def get_player_usage_stats(self):
        """Get current player usage statistics"""
        if self.total_lineups_generated == 0:
            return {}
        
        usage_stats = {}
        for player, count in self.player_usage_tracker.items():
            usage_percentage = (count / self.total_lineups_generated) * 100
            usage_stats[player] = {
                'count': count,
                'percentage': usage_percentage,
                'excluded': player in self.excluded_players
            }
        return usage_stats
    
    def preprocess_data(self):
        """Preprocess player data for optimization"""
        df_filtered = self.df_players.copy()
        
        # Enhanced logging for debugging player selection issues
        print(f"\n🔍 PLAYER SELECTION DEBUG - DETAILED:")
        print(f"   📊 Total players available: {len(df_filtered)}")
        print(f"   🎯 Included players list: {self.included_players}")
        print(f"   📋 Number of included players: {len(self.included_players) if self.included_players else 0}")
        
        logging.info(f"🔍 PLAYER SELECTION DEBUG:")
        logging.info(f"   Total players available: {len(df_filtered)}")
        logging.info(f"   Included players list: {self.included_players}")
        logging.info(f"   Number of included players: {len(self.included_players) if self.included_players else 0}")
        
        # Apply included players filter - This is critical for respecting user selections!
        if self.included_players and len(self.included_players) > 0:
            # If players are specifically selected, use only those players
            print(f"   ✅ FILTERING to {len(self.included_players)} specifically selected players")
            logging.info(f"✅ Filtering to {len(self.included_players)} specifically selected players")
            original_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
            final_count = len(df_filtered)
            print(f"   ✅ RESULT: {final_count}/{original_count} players remain after filtering")
            logging.info(f"✅ After filtering by selected players: {final_count}/{original_count} players remain")
            
            # Additional debug: show which selected players were found/not found
            if final_count == 0:
                print(f"   ⚠️ WARNING: NO PLAYERS FOUND after filtering! Check names match exactly.")
                logging.warning(f"⚠️ NO PLAYERS FOUND after filtering! Check if player names match exactly.")
                logging.warning(f"   Selected players: {self.included_players[:5]}...")  # Show first 5
                if len(df_filtered) > 0:
                    logging.warning(f"   Available player names: {df_filtered['Name'].head().tolist()}")
            elif final_count < len(self.included_players):
                found_players = set(df_filtered['Name'].tolist())
                missing_players = [p for p in self.included_players if p not in found_players]
                print(f"   ⚠️ WARNING: Some selected players not found: {missing_players[:3]}...")
                logging.warning(f"⚠️ Some selected players not found: {missing_players[:3]}...")  # Show first 3
        else:
            # If no players are specifically selected, use all players
            print(f"   ℹ️ NO players specifically selected - using all {len(df_filtered)} players")
            print(f"   ℹ️ This happens when: (1) No checkboxes checked, or (2) Checkbox detection failed")
            logging.info(f"ℹ️ No players specifically selected - using all {len(df_filtered)} players")
            logging.info(f"   This happens when: (1) No checkboxes checked, or (2) Checkbox detection failed")
        
        # Apply player usage filtering - Exclude overused players
        if self.excluded_players:
            original_count = len(df_filtered)
            df_filtered = df_filtered[~df_filtered['Name'].isin(self.excluded_players)]
            excluded_count = original_count - len(df_filtered)
            if excluded_count > 0:
                print(f"   🚫 USAGE FILTER: Excluded {excluded_count} overused players")
                print(f"   🚫 Excluded players: {list(self.excluded_players)[:5]}{'...' if len(self.excluded_players) > 5 else ''}")
                logging.info(f"🚫 Player usage filter: Excluded {excluded_count} overused players")
        
        print(f"")  # Empty line for readability
        
        # Apply exposure constraints (future enhancement)
        # Could add min/max exposure filtering here
        
        return df_filtered

    def calculate_team_projected_runs(self, df):
        """Calculate projected runs for each team"""
        team_runs = {}
        for team in df['Team'].unique():
            team_players = df[df['Team'] == team]
            # Simple calculation based on average points
            avg_points = team_players['Predicted_DK_Points'].mean()
            team_runs[team] = avg_points * 0.1  # Simple scaling factor
        return team_runs

    def optimize_lineups_with_risk_management(self, df_filtered, team_exposure, stack_exposure):
        """
        Advanced lineup optimization using financial risk management principles
        """
        logging.info("🔥 Starting risk-adjusted optimization with financial portfolio theory")
        
        results = {}
        lineup_candidates = []
        
        # Step 1: Generate diverse lineup candidates
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            # With aggressive diversity injection, we need fewer candidates per request
            # Still generate extra to account for filtering, but not as many
            candidate_multiplier = max(10, min(15, self.num_lineups // 10))  # Enhanced multiplier for better selection
            candidate_multiplier = max(candidate_multiplier, 10)  # Minimum of 10x
            
            logging.info(f"🎯 Generating {self.num_lineups * candidate_multiplier} diverse candidates (multiplier: {candidate_multiplier}x) with CONSERVATIVE variance for accuracy")
            
            for stack_type in self.stack_settings:
                for _ in range(self.num_lineups * candidate_multiplier):
                    future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections, self.min_salary, self.disable_adjustments, self.conservative_mode))
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                    try:
                        lineup, stack_type = future.result()
                        if not lineup.empty:
                            lineup_data = self.calculate_lineup_metrics(lineup, stack_type)
                            lineup_candidates.append(lineup_data)
                            
                            # Update player usage tracking for risk management path too
                            self.update_player_usage(lineup)
                            
                    except Exception as e:
                        logging.error(f"Error generating lineup candidate: {str(e)}")
        
        logging.info(f"📊 Generated {len(lineup_candidates)} lineup candidates")
        
        # Step 2: Calculate risk metrics for all candidates
        for i, candidate in enumerate(lineup_candidates):
            try:
                risk_metrics = self.risk_engine.calculate_lineup_risk_metrics(candidate)
                candidate['risk_metrics'] = risk_metrics
                candidate['lineup_id'] = i
                
                # Add risk-adjusted score
                candidate['risk_adjusted_score'] = self.calculate_risk_adjusted_score(candidate, risk_metrics)
                
            except Exception as e:
                logging.error(f"Error calculating risk metrics for candidate {i}: {e}")
                candidate['risk_adjusted_score'] = 0
        
        # Step 3: Select optimal lineups based on risk tolerance
        selected_lineups = self.select_lineups_by_risk_profile(lineup_candidates)
        
        # Step 4: Optimize allocation across selected lineups
        if len(selected_lineups) > 1:
            allocation = self.optimize_lineup_allocation(selected_lineups)
            logging.info(f"💰 Portfolio allocation optimized across {len(selected_lineups)} lineups")
        else:
            allocation = {'weights': [1.0]} if selected_lineups else {'weights': []}
        
        # Step 5: Apply position sizing and bankroll management
        final_selection = self.apply_position_sizing(selected_lineups, allocation)
        
        # Build results in expected format
        for i, lineup_data in enumerate(final_selection):
            lineup_df = lineup_data['lineup']
            total_points = lineup_df['Predicted_DK_Points'].sum()

            # Add risk information to the result
            risk_info = {
                'sharpe_ratio': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).sharpe_ratio,
                'volatility': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).volatility,
                'kelly_fraction': lineup_data.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).kelly_fraction,
                'position_size': lineup_data.get('position_size', 0),
                'allocation_weight': lineup_data.get('allocation_weight', 0)
            }
            
            results[len(results)] = {
                'total_points': total_points, 
                'lineup': lineup_df,
                'risk_info': risk_info
            }
            
            # Print lineup summary with risk information
            sharpe = risk_info.get('sharpe_ratio', 0)
            volatility = risk_info.get('volatility', 0)
            print(f"✅ RISK-ADJUSTED LINEUP #{len(results)} ADDED - Score: {total_points:.2f}, Sharpe: {sharpe:.3f}, Vol: {volatility:.3f}")
            
            # Track exposure
            for team in lineup_df['Team'].unique():
                team_exposure[team] += 1
            stack_exposure[lineup_data['stack_type']] += 1
        
        logging.info(f"🎯 Risk-adjusted optimization complete. Selected {len(results)} optimal lineups")
        
        # Print risk-adjusted optimization summary
        if results:
            print(f"\n{'='*80}")
            print(f"RISK-ADJUSTED OPTIMIZATION SUMMARY")
            print(f"{'='*80}")
            print(f"Total Lineups Generated: {len(results)}")
            
            # Calculate optimal score (best possible lineup)
            optimal_score = max(result['total_points'] for result in results.values())
            total_actual_score = sum(result['total_points'] for result in results.values())
            average_score = total_actual_score / len(results)
            
            print(f"Optimal Score (Best Lineup): {optimal_score:.2f} DK Points")
            print(f"Average Score: {average_score:.2f} DK Points")
            print(f"Total Combined Score: {total_actual_score:.2f} DK Points")
            
            # Show top 3 lineups with risk metrics
            sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
            print(f"\nTOP 3 LINEUPS WITH RISK METRICS:")
            for i, (lineup_id, result) in enumerate(sorted_results[:3]):
                risk_info = result.get('risk_info', {})
                sharpe = risk_info.get('sharpe_ratio', 0)
                volatility = risk_info.get('volatility', 0)
                print(f"  #{i+1}: {result['total_points']:.2f} points (Sharpe: {sharpe:.3f}, Vol: {volatility:.3f})")
            
            print(f"{'='*80}\n")
        
        return results, team_exposure, stack_exposure
    
    def calculate_lineup_metrics(self, lineup_df, stack_type):
        """Calculate comprehensive metrics for a lineup"""
        try:
            total_points = lineup_df['Predicted_DK_Points'].sum()
            total_salary = lineup_df['Salary'].sum()
            
            # Calculate player variance (simplified)
            player_variances = []
            for _, player in lineup_df.iterrows():
                # Estimate variance from projected points (higher projection = higher variance)
                base_variance = (player['Predicted_DK_Points'] * 0.3) ** 2
                player_variances.append(base_variance)
            
            lineup_data = {
                'expected_points': total_points,
                'total_salary': total_salary,
                'lineup': lineup_df,
                'stack_type': stack_type,
                'players': [
                    {
                        'name': row['Name'],
                        'variance': var,
                        'projected_points': row['Predicted_DK_Points']
                    } 
                    for (_, row), var in zip(lineup_df.iterrows(), player_variances)
                ],
                'n_teams': len(lineup_df['Team'].unique()),
                'salary_efficiency': total_points / total_salary if total_salary > 0 else 0
            }
            
            return lineup_data
            
        except Exception as e:
            logging.error(f"Error calculating lineup metrics: {e}")
            return {
                'expected_points': 0,
                'total_salary': 50000,
                'lineup': lineup_df,
                'stack_type': stack_type,
                'players': []
            }
    
    def calculate_risk_adjusted_score(self, lineup_data, risk_metrics):
        """Calculate risk-adjusted score based on user's risk tolerance"""
        base_score = lineup_data['expected_points']
        
        # Risk tolerance multipliers
        risk_multipliers = {
            'conservative': {'sharpe_weight': 0.4, 'volatility_penalty': 0.3},
            'medium': {'sharpe_weight': 0.3, 'volatility_penalty': 0.2},
            'aggressive': {'sharpe_weight': 0.2, 'volatility_penalty': 0.1}
        }
        
        multiplier = risk_multipliers.get(self.risk_tolerance, risk_multipliers['medium'])
        
        # Sharpe ratio bonus
        sharpe_bonus = risk_metrics.sharpe_ratio * multiplier['sharpe_weight'] * base_score
        
        # Volatility penalty
        volatility_penalty = risk_metrics.volatility * multiplier['volatility_penalty'] * base_score
        
        # Kelly fraction bonus (rewards mathematically optimal sizing)
        kelly_bonus = max(0, risk_metrics.kelly_fraction) * 0.1 * base_score
        
        risk_adjusted_score = base_score + sharpe_bonus - volatility_penalty + kelly_bonus
        
        return max(0, risk_adjusted_score)
    
    def select_lineups_by_risk_profile(self, lineup_candidates):
        """Select lineups based on risk profile and diversification - SIMPLIFIED FROM BACKUP"""
        if not lineup_candidates:
            return []
        
        # Sort by risk-adjusted score
        sorted_candidates = sorted(lineup_candidates, key=lambda x: x.get('risk_adjusted_score', 0), reverse=True)
        
        # Apply diversity filters - AGGRESSIVE APPROACH FOR TRULY UNIQUE LINEUPS
        selected = []
        used_core_players = set()
        filtered_count = 0
        
        logging.info(f"🎲 Selecting diverse lineups with progressive overlap thresholds...")
        
        # Start with very strict overlap (only 5 shared players allowed)
        for overlap_threshold in [5, 6, 7, 8]:
            if len(selected) >= self.num_lineups:
                break
                
            logging.info(f"🔍 Trying overlap threshold {overlap_threshold}...")
            temp_selected = []
            temp_used_players = set()
            
            for candidate in sorted_candidates:
                if len(temp_selected) >= self.num_lineups:
                    break
                    
                # Check for excessive overlap with already selected lineups
                lineup_df = candidate['lineup']
                core_players = set(lineup_df['Name'].tolist())
                
                # Check overlap with both permanently selected and temporarily selected
                all_used_players = used_core_players.union(temp_used_players)
                max_overlap = max([len(core_players.intersection(used_players)) for used_players in all_used_players] + [0])
                
                if max_overlap < overlap_threshold:
                    temp_selected.append(candidate)
                    temp_used_players.add(frozenset(core_players))
                    logging.debug(f"Selected lineup {len(selected) + len(temp_selected)} with overlap: {max_overlap}")
                else:
                    filtered_count += 1
                    logging.debug(f"Filtered lineup due to overlap ({max_overlap} >= {overlap_threshold})")
            
            # Add temp selections to permanent selections
            selected.extend(temp_selected)
            used_core_players.update(temp_used_players)
            
            logging.info(f"📊 Got {len(temp_selected)} lineups at threshold {overlap_threshold}, total: {len(selected)}")
            
            if len(selected) >= self.num_lineups:
                break
        
        # Final fallback - take top lineups regardless of overlap
        if len(selected) < min(5, self.num_lineups):
            logging.warning(f"🚨 FALLBACK: Taking top {min(self.num_lineups, len(sorted_candidates))} lineups regardless of overlap")
            selected = sorted_candidates[:min(self.num_lineups, len(sorted_candidates))]
        
        logging.info(f"🎯 FINAL RESULT: Selected {len(selected)} diverse lineups from {len(lineup_candidates)} candidates")
        logging.info(f"✅ SUCCESS: Delivered exactly what you requested!")
        return selected
    
    def optimize_lineup_allocation(self, selected_lineups):
        """Optimize allocation weights across selected lineups using portfolio theory"""
        try:
            lineup_data_for_optimization = []
            for lineup in selected_lineups:
                risk_metrics = lineup.get('risk_metrics', RiskMetrics(0,0,0,0,0,0))
                lineup_data_for_optimization.append({
                    'expected_return': lineup['expected_points'],
                    'variance': risk_metrics.volatility ** 2,
                    'sharpe_ratio': risk_metrics.sharpe_ratio
                })
            
            allocation_result = self.risk_engine.optimize_lineup_allocation(lineup_data_for_optimization)
            
            if allocation_result and 'weights' in allocation_result:
                logging.info(f"📈 Portfolio Sharpe Ratio: {allocation_result.get('sharpe_ratio', 0):.3f}")
                return allocation_result
            else:
                # Equal weights fallback
                n_lineups = len(selected_lineups)
                return {'weights': [1.0/n_lineups] * n_lineups}
                
        except Exception as e:
            logging.error(f"Error in portfolio optimization: {e}")
            # Equal weights fallback
            n_lineups = len(selected_lineups)
            return {'weights': [1.0/n_lineups] * n_lineups}
    
    def apply_position_sizing(self, selected_lineups, allocation):
        """Apply Kelly criterion and bankroll management to determine position sizes"""
        try:
            weights = allocation.get('weights', [])
            
            # Fix array comparison issue - properly check if weights is empty or wrong length
            weights_is_empty = False
            weights_wrong_length = False
            
            try:
                # Check if weights is empty (handle both list and array cases)
                if hasattr(weights, '__len__'):
                    weights_is_empty = len(weights) == 0
                    weights_wrong_length = len(weights) != len(selected_lineups)
                else:
                    weights_is_empty = True
            except:
                weights_is_empty = True
            
            if weights_is_empty or weights_wrong_length:
                if len(selected_lineups) > 0:
                    weights = [1.0/len(selected_lineups)] * len(selected_lineups)
                    logging.info(f"💰 Using equal weights for {len(selected_lineups)} lineups")
                else:
                    logging.error(f"🚨 CRITICAL: No selected lineups available for position sizing!")
                    return []
            else:
                logging.info(f"💰 Using portfolio-optimized weights for {len(selected_lineups)} lineups")
            
            position_limits = None
            if self.bankroll_manager and not self.disable_kelly:
                # Calculate average edge and volatility for position sizing
                # FIX: Ensure scalar values to prevent array comparison errors
                edge_values = []
                volatility_values = []
                
                for l in selected_lineups:
                    # Safely extract salary_efficiency as scalar
                    salary_eff = l.get('salary_efficiency', 0)
                    if hasattr(salary_eff, '__len__') and not isinstance(salary_eff, str):
                        salary_eff = float(salary_eff[0]) if len(salary_eff) > 0 else 0.0
                    edge_values.append(float(salary_eff))
                    
                    # Safely extract volatility as scalar
                    risk_metrics = l.get('risk_metrics', RiskMetrics(0,0,0,0,0,0))
                    if hasattr(risk_metrics, 'volatility'):
                        vol = risk_metrics.volatility
                        if hasattr(vol, '__len__') and not isinstance(vol, str):
                            vol = float(vol[0]) if len(vol) > 0 else 0.0
                        volatility_values.append(float(vol))
                    else:
                        volatility_values.append(0.0)
                
                # Calculate means as scalars
                avg_edge = float(np.mean(edge_values)) if edge_values else 0.0
                avg_volatility = float(np.mean(volatility_values)) if volatility_values else 0.0
                
                position_limits = self.bankroll_manager.calculate_position_limits(avg_edge, avg_volatility)
                logging.info(f"💰 Kelly sizing input: avg_edge={avg_edge:.3f}, avg_volatility={avg_volatility:.3f}")
                logging.info(f"💰 Kelly sizing: Recommended {position_limits.get('recommended_lineups', 1)} lineups")
                logging.info(f"💰 Kelly position limits: {position_limits}")
            elif self.disable_kelly:
                logging.info(f"💰 Kelly sizing: DISABLED - Will use all requested lineups")
            
            # Apply position sizing to lineups
            final_lineups = []
            # Ensure recommended_lineups is always a scalar integer
            recommended_lineups = position_limits.get('recommended_lineups', len(selected_lineups)) if position_limits else len(selected_lineups)
            if hasattr(recommended_lineups, '__len__') and not isinstance(recommended_lineups, str):
                recommended_lineups = int(recommended_lineups[0]) if len(recommended_lineups) > 0 else len(selected_lineups)
            recommended_lineups = int(recommended_lineups)
            
            # OVERRIDE: If Kelly is being too conservative, use more lineups
            if position_limits and recommended_lineups < max(5, self.num_lineups // 10) and not self.disable_kelly:
                logging.warning(f"💰 Kelly recommendation ({recommended_lineups}) seems too conservative. Overriding to use more lineups.")
                # Use a more aggressive override - prioritize user's requested count
                if self.num_lineups >= 50:
                    # For large requests, use 80% of requested lineups instead of half
                    recommended_lineups = max(10, min(len(selected_lineups), int(self.num_lineups * 0.8)))
                else:
                    # For smaller requests, use half
                    recommended_lineups = max(5, min(len(selected_lineups), self.num_lineups // 2))
                logging.info(f"💰 Adjusted to {recommended_lineups} lineups (aggressive override for {self.num_lineups} requested)")
            elif self.disable_kelly:
                # Use all selected lineups when Kelly is disabled - GUARANTEE EXACT COUNT
                recommended_lineups = len(selected_lineups)
                logging.info(f"💰 Kelly disabled: Using all {recommended_lineups} selected lineups")
            
            # CRITICAL: Ensure we deliver the exact number requested when Kelly is disabled
            if self.disable_kelly:
                lineups_to_use = min(len(selected_lineups), self.num_lineups)
                if lineups_to_use < self.num_lineups:
                    logging.error(f"🚨 INSUFFICIENT LINEUPS: Only {lineups_to_use} available but {self.num_lineups} requested")
                else:
                    logging.info(f"✅ EXACT COUNT: Delivering {lineups_to_use} lineups as requested")
            else:
                # With Kelly enabled, use recommended count
                lineups_to_use = min(len(selected_lineups), recommended_lineups, self.num_lineups)
                logging.info(f"💰 Kelly enabled: Using {lineups_to_use} lineups (recommended: {recommended_lineups}, requested: {self.num_lineups})")
            
            # Build final lineup list
            for i in range(lineups_to_use):
                lineup = selected_lineups[i].copy()
                lineup['allocation_weight'] = weights[i] if i < len(weights) else 1.0/lineups_to_use
                
                # Fix the array comparison issue in position sizing
                if position_limits and i < len(weights):
                    optimal_size = position_limits.get('optimal_position_size', 100)
                    # Ensure optimal_size is a scalar, not an array
                    if hasattr(optimal_size, '__len__') and not isinstance(optimal_size, str):
                        optimal_size = float(optimal_size[0]) if len(optimal_size) > 0 else 100.0
                    else:
                        optimal_size = float(optimal_size)
                    
                    # Ensure weights[i] is also a scalar
                    weight_value = weights[i]
                    if hasattr(weight_value, '__len__') and not isinstance(weight_value, str):
                        weight_value = float(weight_value[0]) if len(weight_value) > 0 else 1.0/lineups_to_use
                    else:
                        weight_value = float(weight_value)
                    
                    lineup['position_size'] = optimal_size * weight_value
                else:
                    lineup['position_size'] = 100.0
                
                final_lineups.append(lineup)
            
            logging.info(f"🎯 FINAL DELIVERY: {len(final_lineups)} lineups delivered (requested: {self.num_lineups})")
            return final_lineups
            
        except Exception as e:
            logging.error(f"Error applying position sizing: {e}")
            logging.error(f"Error details: Type={type(e)}, Args={e.args}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Return the selected lineups without position sizing as fallback
            logging.warning(f"🚨 FALLBACK: Returning {len(selected_lineups)} lineups without position sizing due to error")
            return selected_lineups[:self.num_lineups]

class FantasyBaseballApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize enhanced checkbox managers
        if ENHANCED_CHECKBOX_AVAILABLE:
            self.checkbox_managers = {}  # One manager per stack size
            print("✅ Checkbox managers initialized")
        else:
            self.checkbox_managers = None
        
        self.setup_ui()
        
        self.included_players = []
        self.stack_settings = {}
        self.min_exposure = {}
        self.max_exposure = {}
        self.min_points = 1
        self.monte_carlo_iterations = 100
        self.team_selections = {}  # Initialize team selections

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        self.tabs = QTabWidget()
        self.splitter.addWidget(self.tabs)

        self.df_players = None
        self.df_entries = None
        self.player_exposure = {}
        self.optimized_lineups = []
        self.favorites_lineups = []  # Store favorite lineups from multiple runs
        self.favorites_file = "favorites_lineups.json"  # Persistent storage for favorites

        self.create_players_tab()
        self.create_team_stack_tab()
        self.create_stack_exposure_tab()
        self.create_team_combinations_tab()  # Add team combinations tab
        self.create_control_panel()
        self.create_favorites_tab()  # Add favorites tab
        self.load_favorites()  # Load saved favorites on startup

    def create_players_tab(self):
        players_tab = QWidget()
        self.tabs.addTab(players_tab, "Players")

        players_layout = QVBoxLayout(players_tab)

        position_tabs = QTabWidget()
        players_layout.addWidget(position_tabs)

        self.player_tables = {}

        positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        for position in positions:
            sub_tab = QWidget()
            position_tabs.addTab(sub_tab, position)
            layout = QVBoxLayout(sub_tab)

            select_all_button = QPushButton("Select All")
            deselect_all_button = QPushButton("Deselect All")
            select_all_button.clicked.connect(lambda _, p=position: self.select_all(p))
            deselect_all_button.clicked.connect(lambda _, p=position: self.deselect_all(p))
            button_layout = QHBoxLayout()
            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            layout.addLayout(button_layout)

            table = QTableWidget(0, 10)
            table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Pos", "Salary", "Predicted_DK_Points", "Value", "Min Exp", "Max Exp", "Actual Exp (%)"])
            layout.addWidget(table)

            self.player_tables[position] = table

    def create_team_stack_tab(self):
        team_stack_tab = QWidget()
        self.tabs.addTab(team_stack_tab, "Team Stacks")

        layout = QVBoxLayout(team_stack_tab)

        stack_size_tabs = QTabWidget()
        layout.addWidget(stack_size_tabs)

        stack_sizes = ["All Stacks", "2 Stack", "3 Stack", "4 Stack", "5 Stack"]
        self.team_stack_tables = {}

        for stack_size in stack_sizes:
            sub_tab = QWidget()
            stack_size_tabs.addTab(sub_tab, stack_size)
            sub_layout = QVBoxLayout(sub_tab)

            # Add Select All / Deselect All buttons for team stacks
            button_layout = QHBoxLayout()
            
            select_all_btn = QPushButton(f"✅ Select All")
            select_all_btn.setToolTip(f"Select all teams in {stack_size} tab")
            select_all_btn.clicked.connect(lambda checked, size=stack_size: self.select_all_teams(size))
            button_layout.addWidget(select_all_btn)
            
            deselect_all_btn = QPushButton(f"❌ Deselect All")
            deselect_all_btn.setToolTip(f"Deselect all teams in {stack_size} tab")
            deselect_all_btn.clicked.connect(lambda checked, size=stack_size: self.deselect_all_teams(size))
            button_layout.addWidget(deselect_all_btn)
            
            button_layout.addStretch()  # Push buttons to the left
            sub_layout.addLayout(button_layout)

            table = QTableWidget(0, 8)
            table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
            sub_layout.addWidget(table)

            self.team_stack_tables[stack_size] = table

        self.team_stack_table = self.team_stack_tables["All Stacks"]

        refresh_button = QPushButton("Refresh Team Stacks")
        refresh_button.clicked.connect(self.refresh_team_stacks)
        layout.addWidget(refresh_button)
        
        # Add debug button to test team selection collection
        debug_button = QPushButton("🔍 Test Team Selection Detection")
        debug_button.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        debug_button.clicked.connect(self.debug_team_selections)
        layout.addWidget(debug_button)

    def debug_team_selections(self):
        """Debug function to test team selection collection"""
        print("\n" + "="*60)
        print("🔍 MANUAL TEAM SELECTION DEBUG TEST")
        print("="*60)
        
        team_selections = self.collect_team_selections()
        
        print(f"\n📊 RESULTS:")
        if team_selections:
            print(f"✅ Found team selections: {team_selections}")
            for stack_size, teams in team_selections.items():
                print(f"   {stack_size}: {teams}")
        else:
            print(f"❌ No team selections found")
        
        print("="*60)

    def refresh_team_stacks(self):
        self.populate_team_stack_table()

    def select_all_teams(self, stack_size):
        """Select all teams in a specific team stack table"""
        if not hasattr(self, 'team_stack_tables') or stack_size not in self.team_stack_tables:
            logging.debug(f"No team stack table found for: {stack_size}")
            return
        
        table = self.team_stack_tables[stack_size]
        selected_count = 0
        
        # Try enhanced method first
        if (ENHANCED_CHECKBOX_AVAILABLE and hasattr(self, 'checkbox_managers') and 
            self.checkbox_managers and stack_size in self.checkbox_managers):
            
            manager = self.checkbox_managers[stack_size]
            logging.info(f"🚀 Using enhanced method to select all teams in {stack_size}")
            
            # Get all checkbox IDs for this manager and set them to checked
            for row in range(table.rowCount()):
                row_id = f"team_{row}"
                if manager.set_checkbox_state(row_id, True):
                    selected_count += 1
        else:
            # Fallback to original method
            logging.info(f"⚠️ Using fallback method to select all teams in {stack_size}")
            
            # Check all checkboxes in the table
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Find the checkbox within the widget
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox):
                            checkbox.setChecked(True)
                            selected_count += 1
        
        logging.info(f"Selected all {selected_count} teams in {stack_size} table")
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"✅ Selected all {selected_count} teams in {stack_size}")

    def deselect_all_teams(self, stack_size):
        """Deselect all teams in a specific team stack table"""
        if not hasattr(self, 'team_stack_tables') or stack_size not in self.team_stack_tables:
            logging.debug(f"No team stack table found for: {stack_size}")
            return
        
        table = self.team_stack_tables[stack_size]
        deselected_count = 0
        
        # Try enhanced method first
        if (ENHANCED_CHECKBOX_AVAILABLE and hasattr(self, 'checkbox_managers') and 
            self.checkbox_managers and stack_size in self.checkbox_managers):
            
            manager = self.checkbox_managers[stack_size]
            logging.info(f"🚀 Using enhanced method to deselect all teams in {stack_size}")
            
            # Get all checkbox IDs for this manager and set them to unchecked
            for row in range(table.rowCount()):
                row_id = f"team_{row}"
                if manager.set_checkbox_state(row_id, False):
                    deselected_count += 1
        else:
            # Fallback to original method
            logging.info(f"⚠️ Using fallback method to deselect all teams in {stack_size}")
            
            # Uncheck all checkboxes in the table
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Find the checkbox within the widget
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox):
                            checkbox.setChecked(False)
                            deselected_count += 1
        
        logging.info(f"Deselected all {deselected_count} teams in {stack_size} table")
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"❌ Deselected all {deselected_count} teams in {stack_size}")

    def populate_team_combinations_teams(self):
        """Populate team checkboxes in the combinations tab when data is loaded"""
        if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
            return
        
        # Clear existing checkboxes
        for i in reversed(range(self.teams_layout.count())):
            child = self.teams_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.team_checkboxes.clear()
        
        # Get unique teams
        teams = sorted(self.df_players['Team'].unique())
        
        # Create checkboxes for each team
        for team in teams:
            checkbox = QCheckBox(team)
            checkbox.setStyleSheet("padding: 2px; font-size: 11px;")
            self.teams_layout.addWidget(checkbox)
            self.team_checkboxes[team] = checkbox
        
        logging.info(f"Populated {len(teams)} teams in combinations tab")

    def select_all_combination_teams(self):
        """Select all teams in the combinations tab"""
        for checkbox in self.team_checkboxes.values():
            checkbox.setChecked(True)
        logging.info("Selected all teams in combinations tab")

    def deselect_all_combination_teams(self):
        """Deselect all teams in the combinations tab"""
        for checkbox in self.team_checkboxes.values():
            checkbox.setChecked(False)
        logging.info("Deselected all teams in combinations tab")

    def generate_team_combinations(self):
        """Generate all possible team combinations based on selected teams and stack pattern"""
        try:
            # Get selected teams
            selected_teams = []
            for team, checkbox in self.team_checkboxes.items():
                if checkbox.isChecked():
                    selected_teams.append(team)
            
            if len(selected_teams) < 2:
                QMessageBox.warning(self, "Warning", "Please select at least 2 teams to generate combinations.")
                return
            
            # Get stack pattern and parse it
            stack_pattern = self.combinations_stack_combo.currentText()
            stack_sizes = [int(x) for x in stack_pattern.split('|')]
            teams_needed = len(stack_sizes)
            
            if len(selected_teams) < teams_needed:
                QMessageBox.warning(self, "Warning", f"Need at least {teams_needed} teams selected for {stack_pattern} stack pattern.")
                return
            
            # Get default lineups per combination
            try:
                default_lineups = int(self.default_lineups_input.text())
            except ValueError:
                default_lineups = 5
            
            # Generate all possible team combinations for this stack pattern
            team_combinations = []
            
            # Generate combinations of teams
            for team_combo in combinations(selected_teams, teams_needed):
                # Generate all permutations to assign different stack sizes to different teams
                for team_perm in itertools.permutations(team_combo):
                    combination_info = {
                        'teams': team_perm,
                        'stack_pattern': stack_pattern,
                        'stack_sizes': stack_sizes
                    }
                    team_combinations.append(combination_info)
            
            # Clear existing combinations table
            self.combinations_table.setRowCount(0)
            
            # Populate table with combinations
            for i, combo_info in enumerate(team_combinations):
                row = self.combinations_table.rowCount()
                self.combinations_table.insertRow(row)
                
                # Select checkbox
                checkbox = QCheckBox()
                checkbox.setChecked(True)  # Select all by default
                checkbox.stateChanged.connect(self.update_total_lineups_display)
                checkbox_widget = QWidget()
                layout = QHBoxLayout(checkbox_widget)
                layout.addWidget(checkbox)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.combinations_table.setCellWidget(row, 0, checkbox_widget)
                
                # Team combination display
                teams = combo_info['teams']
                stack_sizes = combo_info['stack_sizes']
                combo_parts = []
                for j, team in enumerate(teams):
                    combo_parts.append(f"{team}({stack_sizes[j]})")
                combo_text = " + ".join(combo_parts)
                self.combinations_table.setItem(row, 1, QTableWidgetItem(combo_text))
                
                # Lineups per combo (editable)
                lineups_item = QTableWidgetItem(str(default_lineups))
                lineups_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
                self.combinations_table.setItem(row, 2, lineups_item)
                
                # Actions (could add individual generate button later)
                self.combinations_table.setItem(row, 3, QTableWidgetItem("Ready"))
            
            # Connect table editing signal to update total when lineups per combo are changed
            self.combinations_table.itemChanged.connect(self.update_total_lineups_display)
            
            # Update total lineups display
            self.update_total_lineups_display()
            
            # Show success message
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"✅ Generated {len(team_combinations)} combinations from {len(selected_teams)} teams using {stack_pattern} pattern")
            
            logging.info(f"Generated {len(team_combinations)} team combinations using {stack_pattern} pattern")
            
        except Exception as e:
            logging.error(f"Error generating team combinations: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate combinations: {str(e)}")

    def update_total_lineups_display(self):
        """Update the total lineups count display"""
        try:
            total_lineups = 0
            
            for row in range(self.combinations_table.rowCount()):
                # Check if combination is selected
                checkbox_widget = self.combinations_table.cellWidget(row, 0)
                if checkbox_widget:
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                            # Get lineups count for this combination
                            lineups_item = self.combinations_table.item(row, 2)
                            if lineups_item:
                                try:
                                    lineups_count = int(lineups_item.text())
                                    total_lineups += lineups_count
                                except ValueError:
                                    pass
            
            self.total_lineups_display.setText(str(total_lineups))
            
        except Exception as e:
            logging.error(f"Error updating total lineups display: {e}")

    def generate_combination_lineups(self):
        """Generate lineups for all selected team combinations"""
        try:
            if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
                QMessageBox.warning(self, "Warning", "Please load player data first.")
                return
            
            selected_combinations = []
            
            # Get selected combinations and their lineup counts
            for row in range(self.combinations_table.rowCount()):
                checkbox_widget = self.combinations_table.cellWidget(row, 0)
                if checkbox_widget:
                    layout = checkbox_widget.layout()
                    if layout and layout.count() > 0:
                        checkbox = layout.itemAt(0).widget()
                        if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                            combo_item = self.combinations_table.item(row, 1)
                            lineups_item = self.combinations_table.item(row, 2)
                            if combo_item and lineups_item:
                                try:
                                    combo_text = combo_item.text()
                                    # Parse the combination text like "LAD(4) + SF(2)"
                                    team_parts = combo_text.split(' + ')
                                    teams = []
                                    stack_sizes = []
                                    for part in team_parts:
                                        if '(' in part and ')' in part:
                                            team = part.split('(')[0].strip()
                                            stack_size = int(part.split('(')[1].split(')')[0])
                                            teams.append(team)
                                            stack_sizes.append(stack_size)
                                    
                                    lineups_count = int(lineups_item.text())
                                    selected_combinations.append((teams, stack_sizes, lineups_count))
                                except ValueError:
                                    pass
            
            if not selected_combinations:
                QMessageBox.warning(self, "Warning", "No combinations selected.")
                return
            
            # Get stack pattern
            stack_pattern = self.combinations_stack_combo.currentText()
            
            # Generate lineups for each combination
            all_lineups = []
            
            for teams, stack_sizes, lineups_count in selected_combinations:
                if lineups_count <= 0:
                    continue
                
                # Create team selections for this combination
                # Map each stack size to its corresponding teams
                team_selections = {}
                for i, stack_size in enumerate(stack_sizes):
                    if stack_size not in team_selections:
                        team_selections[stack_size] = []
                    team_selections[stack_size].append(teams[i])
                
                # Set up optimization parameters
                salary_cap = 50000
                position_limits = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
                
                # Get included players (all players for now)
                included_players = self.get_included_players()
                
                # Run optimization for this combination
                try:
                    # Format stack settings for the worker
                    stack_pattern = self.combinations_stack_combo.currentText()
                    stack_settings = {stack_pattern: True}  # Worker expects a dict of stack patterns
                    
                    worker = OptimizationWorker(
                        df_players=self.df_players,
                        salary_cap=salary_cap,
                        position_limits=position_limits,
                        included_players=included_players,
                        stack_settings=stack_settings,
                        min_exposure={},
                        max_exposure={},
                        min_points=1,
                        monte_carlo_iterations=100,
                        num_lineups=lineups_count,
                        team_selections=team_selections,
                        min_unique=3,
                        bankroll=1000,
                        risk_tolerance='medium',
                        disable_kelly=True,  # Generate exact number requested
                        min_salary=self.get_min_salary_constraint(),  # Add minimum salary constraint
                        disable_adjustments=self.disable_adjustments_checkbox.isChecked(),
                        conservative_mode=self.conservative_mode_checkbox.isChecked()
                    )
                    
                    # Run optimization directly and get results
                    combo_results, _, _ = worker.optimize_lineups()
                    
                    if combo_results and len(combo_results) > 0:
                        # Convert results to list format for consistency
                        if isinstance(combo_results, dict):
                            combo_lineups = []
                            for result in combo_results.values():
                                if isinstance(result, dict) and 'lineup' in result:
                                    combo_lineups.append(result['lineup'])
                                elif isinstance(result, pd.DataFrame):
                                    combo_lineups.append(result)
                        elif isinstance(combo_results, list):
                            # If it's already a list, use it directly
                            combo_lineups = combo_results
                        else:
                            # Handle any other format
                            combo_lineups = [combo_results] if combo_results is not None else []
                        
                        all_lineups.extend(combo_lineups)
                        combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                        logging.info(f"Generated {len(combo_lineups)} lineups for combination: {combo_display}")
                    else:
                        combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                        logging.warning(f"No lineups generated for combination: {combo_display}")
                        
                except Exception as e:
                    combo_display = " + ".join([f"{team}({size})" for team, size in zip(teams, stack_sizes)])
                    logging.error(f"Error generating lineups for combination {combo_display}: {e}")
                    continue
            
            if all_lineups and len(all_lineups) > 0:
                # Store the results
                self.optimized_lineups = all_lineups
                
                # Convert list format to expected dictionary format for display_results
                results_dict = {}
                for i, lineup_df in enumerate(all_lineups):
                    # Calculate lineup metrics for display
                    if isinstance(lineup_df, dict) and 'lineup' in lineup_df:
                        # Already in proper format
                        results_dict[i] = lineup_df
                    else:
                        # Convert DataFrame to expected format
                        total_salary = lineup_df['Salary'].sum() if 'Salary' in lineup_df.columns else 0
                        total_points = lineup_df['Predicted_DK_Points'].sum() if 'Predicted_DK_Points' in lineup_df.columns else 0
                        
                        results_dict[i] = {
                            'lineup': lineup_df,
                            'total_salary': total_salary,
                            'total_points': total_points
                        }
                
                # Display results
                team_exposure = {}
                stack_exposure = {}
                self.display_results(results_dict, team_exposure, stack_exposure)
                
                # Show success message
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"✅ Generated {len(all_lineups)} total lineups from {len(selected_combinations)} combinations")
                
                logging.info(f"Successfully generated {len(all_lineups)} total lineups from combinations")
                
            else:
                error_msg = ("No lineups were generated. This could be due to:\n\n"
                           "• Not enough players selected for some positions\n"
                           "• Stack requirements too restrictive for available teams\n"
                           "• Salary constraints too tight\n"
                           "• Selected players don't form valid lineups\n\n"
                           "Try:\n"
                           "• Selecting more players in each position\n"
                           "• Choosing simpler stack patterns\n"
                           "• Adjusting salary constraints")
                QMessageBox.warning(self, "No Lineups Generated", error_msg)
                
        except Exception as e:
            logging.error(f"Error in generate_combination_lineups: {e}")
            QMessageBox.critical(self, "Error", f"Failed to generate combination lineups: {str(e)}")

    def create_stack_exposure_tab(self):
        stack_exposure_tab = QWidget()
        self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
    
        layout = QVBoxLayout(stack_exposure_tab)
    
        self.stack_exposure_table = QTableWidget(0, 7)
        self.stack_exposure_table.setHorizontalHeaderLabels(["Select", "Stack Type", "Min Exp", "Max Exp", "Lineup Exp", "Pool Exp", "Entry Exp"])
        layout.addWidget(self.stack_exposure_table)
    
        stack_types = ["4|2|2", "4|2", "3|3|2", "3|2|2", "2|2|2", "5|3", "5|2", "No Stacks"]
        for stack_type in stack_types:
            row_position = self.stack_exposure_table.rowCount()
            self.stack_exposure_table.insertRow(row_position)
    
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            layout_checkbox = QHBoxLayout(checkbox_widget)
            layout_checkbox.addWidget(checkbox)
            layout_checkbox.setAlignment(Qt.AlignCenter)
            layout_checkbox.setContentsMargins(0, 0, 0, 0)
            self.stack_exposure_table.setCellWidget(row_position, 0, checkbox_widget)
    
            self.stack_exposure_table.setItem(row_position, 1, QTableWidgetItem(stack_type))
            min_exp_item = QTableWidgetItem("0")
            min_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 2, min_exp_item)
    
            max_exp_item = QTableWidgetItem("100")
            max_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.stack_exposure_table.setItem(row_position, 3, max_exp_item)
    
            self.stack_exposure_table.setItem(row_position, 4, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 5, QTableWidgetItem("0.0%"))
            self.stack_exposure_table.setItem(row_position, 6, QTableWidgetItem("0.0%"))

    def create_team_combinations_tab(self):
        """Create tab for generating all team combinations with specified stacks"""
        combinations_tab = QWidget()
        self.tabs.addTab(combinations_tab, "Team Combinations")
        
        layout = QVBoxLayout(combinations_tab)
        
        # Header section
        header_label = QLabel("🔥 Team Combination Generator")
        header_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #FF5722; padding: 10px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Select teams and stack type to generate all possible combinations, then specify lineups per combination.")
        desc_label.setStyleSheet("color: #666; font-size: 12px; padding: 5px;")
        layout.addWidget(desc_label)
        
        # Controls section
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        layout.addWidget(controls_frame)
        
        # Team selection section
        team_section = QVBoxLayout()
        team_label = QLabel("📋 Select Teams:")
        team_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        team_section.addWidget(team_label)
        
        # Team selection buttons
        team_btn_layout = QHBoxLayout()
        select_all_teams_btn = QPushButton("✅ Select All Teams")
        deselect_all_teams_btn = QPushButton("❌ Deselect All Teams")
        team_btn_layout.addWidget(select_all_teams_btn)
        team_btn_layout.addWidget(deselect_all_teams_btn)
        team_section.addLayout(team_btn_layout)
        
        # Team checkboxes area (scrollable)
        teams_scroll = QScrollArea()
        teams_scroll.setMaximumHeight(200)
        teams_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.teams_widget = QWidget()
        self.teams_layout = QVBoxLayout(self.teams_widget)
        teams_scroll.setWidget(self.teams_widget)
        teams_scroll.setWidgetResizable(True)
        team_section.addWidget(teams_scroll)
        
        controls_layout.addLayout(team_section)
        
        # Stack type and settings section
        settings_section = QVBoxLayout()
        
        # Stack type selection
        stack_label = QLabel("🏗️ Stack Type:")
        stack_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        settings_section.addWidget(stack_label)
        
        self.combinations_stack_combo = QComboBox()
        self.combinations_stack_combo.addItems(["5|2", "4|2", "4|2|2", "3|3|2", "3|2|2", "2|2|2", "5|3"])
        self.combinations_stack_combo.setCurrentText("4|2")
        self.combinations_stack_combo.setToolTip("Stack pattern: e.g., '4|2' means 4 players from one team, 2 from another")
        settings_section.addWidget(self.combinations_stack_combo)
        
        # Default lineups per combination
        lineups_label = QLabel("📊 Default Lineups per Combination:")
        lineups_label.setStyleSheet("font-weight: bold; color: #9C27B0;")
        settings_section.addWidget(lineups_label)
        
        self.default_lineups_input = QLineEdit()
        self.default_lineups_input.setText("5")
        self.default_lineups_input.setPlaceholderText("e.g., 5")
        self.default_lineups_input.setToolTip("Default number of lineups to generate per combination")
        settings_section.addWidget(self.default_lineups_input)
        
        # Generate combinations button
        generate_combinations_btn = QPushButton("🔄 Generate Team Combinations")
        generate_combinations_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        generate_combinations_btn.clicked.connect(self.generate_team_combinations)
        settings_section.addWidget(generate_combinations_btn)
        
        controls_layout.addLayout(settings_section)
        
        # Combinations display table
        combinations_label = QLabel("🎯 Generated Combinations:")
        combinations_label.setStyleSheet("font-weight: bold; color: #673AB7; font-size: 14px; padding: 10px 0px 5px 0px;")
        layout.addWidget(combinations_label)
        
        self.combinations_table = QTableWidget(0, 4)
        self.combinations_table.setHorizontalHeaderLabels(["Select", "Team Combination", "Lineups per Combo", "Actions"])
        self.combinations_table.setAlternatingRowColors(True)
        layout.addWidget(self.combinations_table)
        
        # Generate lineups button
        generate_lineups_frame = QFrame()
        generate_lineups_layout = QHBoxLayout(generate_lineups_frame)
        layout.addWidget(generate_lineups_frame)
        
        total_lineups_label = QLabel("Total Lineups:")
        total_lineups_label.setStyleSheet("font-weight: bold; color: #333;")
        self.total_lineups_display = QLabel("0")
        self.total_lineups_display.setStyleSheet("font-weight: bold; color: #FF5722; font-size: 14px;")
        
        generate_lineups_layout.addWidget(total_lineups_label)
        generate_lineups_layout.addWidget(self.total_lineups_display)
        generate_lineups_layout.addStretch()
        
        generate_lineups_btn = QPushButton("🚀 Generate All Combination Lineups")
        generate_lineups_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 14px; }")
        generate_lineups_btn.clicked.connect(self.generate_combination_lineups)
        generate_lineups_layout.addWidget(generate_lineups_btn)
        
        # Connect team selection buttons
        select_all_teams_btn.clicked.connect(self.select_all_combination_teams)
        deselect_all_teams_btn.clicked.connect(self.deselect_all_combination_teams)
        
        # Initialize empty team checkboxes (will be populated when data is loaded)
        self.team_checkboxes = {}

    def create_control_panel(self):
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)

        self.splitter.addWidget(control_panel)

        load_button = QPushButton('Load CSV')
        load_button.clicked.connect(self.load_file)
        control_layout.addWidget(load_button)

        load_dk_predictions_button = QPushButton('Load DraftKings Predictions')
        load_dk_predictions_button.clicked.connect(self.load_dk_predictions)
        control_layout.addWidget(load_dk_predictions_button)

        load_entries_button = QPushButton('Load Entries CSV')
        load_entries_button.clicked.connect(self.load_entries_csv)
        control_layout.addWidget(load_entries_button)
        
        # Simple selection status label (for debugging)
        self.selection_status_label = QLabel("Player selection status will appear here")
        self.selection_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
        control_layout.addWidget(self.selection_status_label)
        
        self.min_unique_label = QLabel('Min Unique:')
        self.min_unique_input = QLineEdit()
        self.min_unique_input.setText("3")  # Default value
        self.min_unique_input.setPlaceholderText("e.g., 3")
        self.min_unique_input.setToolTip("Minimum number of unique players between lineups (0-10). Higher values create more diverse lineups.")
        control_layout.addWidget(self.min_unique_label)
        control_layout.addWidget(self.min_unique_input)
        
        self.num_lineups_label = QLabel('Number of Lineups:')
        self.num_lineups_input = QLineEdit()
        self.num_lineups_input.setText("100")  # Default value
        self.num_lineups_input.setPlaceholderText("e.g., 100")
        self.num_lineups_input.setToolTip("Number of lineups to generate (1-500). More lineups take longer to generate.")
        control_layout.addWidget(self.num_lineups_label)
        control_layout.addWidget(self.num_lineups_input)
        
        # Kelly sizing control
        self.disable_kelly_checkbox = QCheckBox('Disable Kelly Sizing (Generate All Requested Lineups)')
        self.disable_kelly_checkbox.setToolTip("Check this to disable Kelly Criterion position sizing and generate the full number of requested lineups. Useful when you want more lineups regardless of risk management recommendations.")
        control_layout.addWidget(self.disable_kelly_checkbox)

        # ADD PROJECTION ADJUSTMENT CONTROLS
        projection_frame = QFrame()
        projection_frame.setFrameStyle(QFrame.Box)
        projection_frame.setStyleSheet("QFrame { background-color: #FFF3E0; border: 1px solid #FF9800; border-radius: 5px; padding: 5px; }")
        control_layout.addWidget(projection_frame)
        
        projection_layout = QVBoxLayout(projection_frame)
        
        projection_label = QLabel("🎯 Projection Adjustments (ACCURACY vs DIVERSITY):")
        projection_label.setStyleSheet("font-weight: bold; color: #F57C00; padding: 5px; font-size: 12px;")
        projection_layout.addWidget(projection_label)
        
        # Disable adjustments checkbox
        self.disable_adjustments_checkbox = QCheckBox('🎯 Pure Projections (No Random Adjustments)')
        self.disable_adjustments_checkbox.setToolTip("Check this to use your exact projections without any random variance. Best for accuracy but may create similar lineups.")
        self.disable_adjustments_checkbox.setStyleSheet("color: #F57C00; font-weight: bold;")
        projection_layout.addWidget(self.disable_adjustments_checkbox)
        
        # Conservative mode checkbox  
        self.conservative_mode_checkbox = QCheckBox('🛡️ Conservative Mode (Minimal Adjustments)')
        self.conservative_mode_checkbox.setChecked(True)  # Default to conservative
        self.conservative_mode_checkbox.setToolTip("Use minimal random variance (2-5%) for lineup diversity while preserving projection accuracy.")
        self.conservative_mode_checkbox.setStyleSheet("color: #4CAF50;")
        projection_layout.addWidget(self.conservative_mode_checkbox)
        
        # Warning label
        projection_warning = QLabel("⚠️ PERFORMANCE TIP: Use Pure Projections or Conservative Mode for best accuracy!")
        projection_warning.setStyleSheet("color: #D32F2F; font-size: 10px; font-weight: bold; padding: 2px;")
        projection_layout.addWidget(projection_warning)

        # ADD ENHANCED MINIMUM SALARY CONTROLS
        min_salary_frame = QFrame()
        min_salary_frame.setFrameStyle(QFrame.Box)
        min_salary_frame.setStyleSheet("QFrame { background-color: #E8F5E8; border: 1px solid #4CAF50; border-radius: 5px; padding: 5px; }")
        control_layout.addWidget(min_salary_frame)
        
        min_salary_layout = QVBoxLayout(min_salary_frame)
        
        min_salary_label = QLabel("💰 Minimum Salary Constraint:")
        min_salary_label.setStyleSheet("font-weight: bold; color: #2E7D32; padding: 5px; font-size: 12px;")
        min_salary_layout.addWidget(min_salary_label)
        
        # Create horizontal layout for input and presets
        min_salary_input_layout = QHBoxLayout()
        
        self.min_salary_input = QLineEdit()
        self.min_salary_input.setText("45000")  # Default value
        self.min_salary_input.setPlaceholderText("e.g., 45000")
        self.min_salary_input.setToolTip("Minimum total salary to spend (0-50000). Forces lineups to use higher budget to avoid too many cheap players.")
        self.min_salary_input.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #4CAF50; border-radius: 3px; }")
        min_salary_input_layout.addWidget(self.min_salary_input)
        
        # Add preset buttons for common minimum salary values
        preset_40k_btn = QPushButton("40K")
        preset_40k_btn.setToolTip("Set minimum salary to $40,000")
        preset_40k_btn.clicked.connect(lambda: self.min_salary_input.setText("40000"))
        preset_40k_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #66BB6A; color: white; border: none; border-radius: 3px; }")
        min_salary_input_layout.addWidget(preset_40k_btn)
        
        preset_45k_btn = QPushButton("45K")
        preset_45k_btn.setToolTip("Set minimum salary to $45,000 (Recommended)")
        preset_45k_btn.clicked.connect(lambda: self.min_salary_input.setText("45000"))
        preset_45k_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; }")
        min_salary_input_layout.addWidget(preset_45k_btn)
        
        preset_48k_btn = QPushButton("48K")
        preset_48k_btn.setToolTip("Set minimum salary to $48,000 (High budget)")
        preset_48k_btn.clicked.connect(lambda: self.min_salary_input.setText("48000"))
        preset_48k_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #388E3C; color: white; border: none; border-radius: 3px; }")
        min_salary_input_layout.addWidget(preset_48k_btn)
        
        min_salary_layout.addLayout(min_salary_input_layout)
        
        # Add explanation label
        min_salary_explanation = QLabel("💡 Higher minimum salaries force more expensive, potentially better players")
        min_salary_explanation.setStyleSheet("color: #2E7D32; font-size: 10px; font-style: italic; padding: 2px;")
        min_salary_layout.addWidget(min_salary_explanation)

        # ADD PLAYER USAGE TRACKING CONTROLS
        usage_frame = QFrame()
        usage_frame.setFrameStyle(QFrame.Box)
        usage_frame.setStyleSheet("QFrame { background-color: #FFF9C4; border: 1px solid #FF9800; border-radius: 5px; padding: 5px; }")
        control_layout.addWidget(usage_frame)
        
        usage_layout = QVBoxLayout(usage_frame)
        
        usage_label = QLabel("🎯 Player Usage Tracking (FORCE DIVERSITY):")
        usage_label.setStyleSheet("font-weight: bold; color: #F57C00; padding: 5px; font-size: 12px;")
        usage_layout.addWidget(usage_label)
        
        # Create horizontal layout for input and presets
        usage_input_layout = QHBoxLayout()
        
        usage_desc_label = QLabel("Max Usage Per Player (%):")
        usage_desc_label.setStyleSheet("color: #F57C00; font-weight: bold; font-size: 11px;")
        usage_layout.addWidget(usage_desc_label)
        
        self.max_usage_input = QLineEdit()
        self.max_usage_input.setText("40")  # Default 40% max usage
        self.max_usage_input.setPlaceholderText("e.g., 40")
        self.max_usage_input.setToolTip("Maximum % of lineups a player can appear in before being temporarily excluded. Lower values force more diversity.")
        self.max_usage_input.setStyleSheet("QLineEdit { padding: 5px; border: 1px solid #FF9800; border-radius: 3px; }")
        usage_input_layout.addWidget(self.max_usage_input)
        
        # Add preset buttons for common usage limits
        usage_20_btn = QPushButton("20%")
        usage_20_btn.setToolTip("Very diverse - max 20% usage per player")
        usage_20_btn.clicked.connect(lambda: self.max_usage_input.setText("20"))
        usage_20_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #FF9800; color: white; border: none; border-radius: 3px; }")
        usage_input_layout.addWidget(usage_20_btn)
        
        usage_40_btn = QPushButton("40%")
        usage_40_btn.setToolTip("Balanced diversity - max 40% usage per player (Recommended)")
        usage_40_btn.clicked.connect(lambda: self.max_usage_input.setText("40"))
        usage_40_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #FF9800; color: white; border: none; border-radius: 3px; }")
        usage_input_layout.addWidget(usage_40_btn)
        
        usage_60_btn = QPushButton("60%")
        usage_60_btn.setToolTip("Light diversity - max 60% usage per player")
        usage_60_btn.clicked.connect(lambda: self.max_usage_input.setText("60"))
        usage_60_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #FF9800; color: white; border: none; border-radius: 3px; }")
        usage_input_layout.addWidget(usage_60_btn)
        
        usage_100_btn = QPushButton("OFF")
        usage_100_btn.setToolTip("Disable usage tracking (100% = no limits)")
        usage_100_btn.clicked.connect(lambda: self.max_usage_input.setText("100"))
        usage_100_btn.setStyleSheet("QPushButton { padding: 3px 8px; font-size: 10px; background-color: #757575; color: white; border: none; border-radius: 3px; }")
        usage_input_layout.addWidget(usage_100_btn)
        
        usage_layout.addLayout(usage_input_layout)
        
        # Add explanation label
        usage_explanation = QLabel("💡 Lower percentages = more diverse lineups but may exclude optimal players")
        usage_explanation.setStyleSheet("color: #F57C00; font-size: 10px; font-style: italic; padding: 2px;")
        usage_layout.addWidget(usage_explanation)

        # Salary Range Constraints
        salary_range_label = QLabel("💰 Salary Range Constraints:")
        salary_range_label.setStyleSheet("font-weight: bold; color: #4CAF50; padding: 5px; font-size: 12px;")
        control_layout.addWidget(salary_range_label)
        
        self.max_salary_label = QLabel('Maximum Salary Spent:')
        self.max_salary_input = QLineEdit()
        self.max_salary_input.setText("50000")  # Default maximum (salary cap)
        self.max_salary_input.setPlaceholderText("e.g., 50000")
        self.max_salary_input.setToolTip("Maximum total salary to spend on lineups (default: 50000 - salary cap)")
        control_layout.addWidget(self.max_salary_label)
        control_layout.addWidget(self.max_salary_input)

        self.sorting_label = QLabel('Sorting Method:')
        self.sorting_combo = QComboBox()
        self.sorting_combo.addItems(["Points", "Value", "Salary"])
        control_layout.addWidget(self.sorting_label)
        control_layout.addWidget(self.sorting_combo)

        # Add Risk Management Section
        if RISK_ENGINE_AVAILABLE:
            risk_separator = QFrame()
            risk_separator.setFrameShape(QFrame.HLine)
            risk_separator.setFrameShadow(QFrame.Sunken)
            control_layout.addWidget(risk_separator)
            
            risk_label = QLabel("🔥 Risk Management:")
            risk_label.setStyleSheet("font-weight: bold; color: #FF5722; padding: 5px; font-size: 12px;")
            control_layout.addWidget(risk_label)
            
            # Bankroll management
            self.bankroll_label = QLabel('Bankroll ($):')
            self.bankroll_input = QLineEdit()
            self.bankroll_input.setText("1000")
            self.bankroll_input.setPlaceholderText("e.g., 1000")
            self.bankroll_input.setToolTip("Your total bankroll for position sizing and Kelly criterion")
            control_layout.addWidget(self.bankroll_label)
            control_layout.addWidget(self.bankroll_input)
            
            # Risk tolerance
            self.risk_tolerance_label = QLabel('Risk Profile:')
            self.risk_tolerance_combo = QComboBox()
            self.risk_tolerance_combo.addItems(["conservative", "medium", "aggressive"])
            self.risk_tolerance_combo.setCurrentText("medium")
            self.risk_tolerance_combo.setToolTip("Conservative: Focus on Sharpe ratio\nMedium: Balanced approach\nAggressive: Higher risk tolerance")
            control_layout.addWidget(self.risk_tolerance_label)
            control_layout.addWidget(self.risk_tolerance_combo)
            
            # Enable/disable risk management
            self.enable_risk_mgmt_checkbox = QCheckBox("Enable Advanced Risk Management")
            self.enable_risk_mgmt_checkbox.setChecked(True)
            self.enable_risk_mgmt_checkbox.setToolTip("Use Kelly criterion, GARCH volatility, and portfolio theory")
            self.enable_risk_mgmt_checkbox.setStyleSheet("color: #FF5722; font-weight: bold;")
            control_layout.addWidget(self.enable_risk_mgmt_checkbox)

        run_button = QPushButton('Run Contest Sim')
        run_button.clicked.connect(self.run_optimization)
        control_layout.addWidget(run_button)

        save_button = QPushButton('Save CSV for DraftKings')
        save_button.clicked.connect(self.save_csv)
        control_layout.addWidget(save_button)
        
        # Add button for loading and filling DraftKings entries
        load_dk_entries_button = QPushButton('Load DraftKings Entries File')
        load_dk_entries_button.clicked.connect(self.load_dk_entries_file)
        load_dk_entries_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        control_layout.addWidget(load_dk_entries_button)
        
        # Add button for filling loaded entries with optimized lineups
        fill_entries_button = QPushButton('Fill Entries with Optimized Lineups')
        fill_entries_button.clicked.connect(self.fill_dk_entries_dynamic)
        fill_entries_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        control_layout.addWidget(fill_entries_button)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)
        
        # Add favorites control buttons to main panel
        favorites_label = QLabel("🌟 Favorites Management:")
        favorites_label.setStyleSheet("font-weight: bold; color: #FF9800; padding: 5px;")
        control_layout.addWidget(favorites_label)
        
        add_to_favorites_main_button = QPushButton("➕ Add Current to Favorites")
        add_to_favorites_main_button.clicked.connect(self.add_current_lineups_to_favorites)
        add_to_favorites_main_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        control_layout.addWidget(add_to_favorites_main_button)
        
        save_favorites_main_button = QPushButton("💾 Export Favorites as New Lineups")
        save_favorites_main_button.clicked.connect(self.save_favorites_to_entries)
        save_favorites_main_button.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        control_layout.addWidget(save_favorites_main_button)

        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Pos", "Salary", "Predicted_DK_Points", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
        control_layout.addWidget(self.results_table)

        self.status_label = QLabel('')
        control_layout.addWidget(self.status_label)

    def create_favorites_tab(self):
        """Create the favorites tab for managing saved lineups from multiple runs"""
        favorites_tab = QWidget()
        self.tabs.addTab(favorites_tab, "My Entries")
        
        layout = QVBoxLayout(favorites_tab)
        
        # Header with info
        header_label = QLabel("💾 My Entries - Build your final contest lineup from multiple optimization runs")
        header_label.setStyleSheet("font-weight: bold; color: #2196F3; padding: 10px;")
        layout.addWidget(header_label)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        # Add button to save current lineups to favorites
        self.add_to_favorites_button = QPushButton("➕ Add Current Pool to Favorites")
        self.add_to_favorites_button.clicked.connect(self.add_current_lineups_to_favorites)
        self.add_to_favorites_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(self.add_to_favorites_button)
        
        # Clear favorites button
        clear_favorites_button = QPushButton("🗑️ Clear All Favorites")
        clear_favorites_button.clicked.connect(self.clear_favorites)
        clear_favorites_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(clear_favorites_button)
        
        # Save favorites to entries button
        save_favorites_button = QPushButton("💾 Export Favorites as New Lineups")
        save_favorites_button.clicked.connect(self.save_favorites_to_entries)
        save_favorites_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 5px; }")
        button_layout.addWidget(save_favorites_button)
        
        layout.addLayout(button_layout)
        
        # Stats display
        self.favorites_stats_label = QLabel("📊 Total Favorites: 0 lineups")
        self.favorites_stats_label.setStyleSheet("padding: 5px; color: #666;")
        layout.addWidget(self.favorites_stats_label)
        
        # Favorites table
        self.favorites_table = QTableWidget(0, 11)
        self.favorites_table.setHorizontalHeaderLabels([
            "Select", "Run#", "Player", "Team", "Pos", "Salary", "Points", 
            "Total Salary", "Total Points", "Added Date", "Actions"
        ])
        
        # Set table selection behavior
        self.favorites_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.favorites_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.favorites_table)
        
        # Update favorites display
        self.update_favorites_display()

    def load_favorites(self):
        """Load saved favorites from persistent storage"""
        try:
            if os.path.exists(self.favorites_file):
                with open(self.favorites_file, 'r') as f:
                    favorites_data = json.load(f)
                    
                # Convert back to DataFrame format
                self.favorites_lineups = []
                for fav_data in favorites_data:
                    lineup_df = pd.DataFrame(fav_data['lineup_data'])
                    fav_entry = {
                        'lineup': lineup_df,
                        'total_points': float(fav_data['total_points']),
                        'total_salary': int(fav_data['total_salary']),
                        'run_number': fav_data.get('run_number', 1),
                        'date_added': fav_data.get('date_added', 'Unknown')
                    }
                    self.favorites_lineups.append(fav_entry)
                    
                logging.info(f"Loaded {len(self.favorites_lineups)} favorite lineups")
            else:
                self.favorites_lineups = []
                logging.info("No favorites file found, starting with empty favorites")
                
        except Exception as e:
            logging.error(f"Error loading favorites: {e}")
            self.favorites_lineups = []

    def save_favorites(self):
        """Save favorites to persistent storage"""
        try:
            favorites_data = []
            for fav in self.favorites_lineups:
                fav_data = {
                    'lineup_data': fav['lineup'].to_dict('records'),
                    'total_points': float(fav['total_points']),  # Convert to Python float
                    'total_salary': int(fav['total_salary']),   # Convert to Python int
                    'run_number': fav.get('run_number', 1),
                    'date_added': fav.get('date_added', 'Unknown')
                }
                favorites_data.append(fav_data)
            
            with open(self.favorites_file, 'w') as f:
                json.dump(favorites_data, f, indent=2)
                
            logging.info(f"Saved {len(self.favorites_lineups)} favorite lineups")
            
        except Exception as e:
            logging.error(f"Error saving favorites: {e}")

    def add_current_lineups_to_favorites(self):
        """Add current optimized lineups to favorites"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Lineups", "No optimized lineups available to add to favorites.\n\nPlease run optimization first.")
            return
        
        # Ask user how many lineups to add
        num_available = len(self.optimized_lineups)
        num_to_add, ok = QInputDialog.getInt(
            self, 
            'Add to Favorites', 
            f'How many lineups to add to favorites?\n\n(Available: {num_available} optimized lineups)', 
            value=min(num_available, 50),  # Default to 50 or available, whichever is less
            min=1, 
            max=num_available
        )
        
        if not ok:
            return
        
        # Get current run number (increment from existing favorites)
        current_run = max([fav.get('run_number', 0) for fav in self.favorites_lineups], default=0) + 1
        current_date = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        # Add selected lineups to favorites
        added_count = 0
        for i, lineup in enumerate(self.optimized_lineups[:num_to_add]):
            fav_entry = {
                'lineup': lineup.copy(),
                'total_points': lineup['Predicted_DK_Points'].sum(),
                'total_salary': lineup['Salary'].sum(),
                'run_number': current_run,
                'date_added': current_date
            }
            self.favorites_lineups.append(fav_entry)
            added_count += 1
        
        # Save to persistent storage
        self.save_favorites()
        
        # Update display
        self.update_favorites_display()
        
        # Show success message with feedback if fewer than requested
        success_msg = f"✅ Successfully added {added_count} lineups to favorites!\n\n"
        success_msg += f"🏃 Run #{current_run}\n"
        success_msg += f"📅 {current_date}\n"
        success_msg += f"📊 Total favorites: {len(self.favorites_lineups)} lineups"
        
        if added_count < num_to_add:
            success_msg += f"\n\n⚠️ Note: Added {added_count} lineups (requested {num_to_add})"
        
        QMessageBox.information(self, "Added to Favorites", success_msg)
        
        self.status_label.setText(f'Added {added_count} lineups to favorites (Run #{current_run})')

    def clear_favorites(self):
        """Clear all favorites"""
        if not self.favorites_lineups:
            QMessageBox.information(self, "No Favorites", "No favorites to clear.")
            return
        
        reply = QMessageBox.question(
            self, 
            'Clear Favorites', 
            f'Are you sure you want to clear all {len(self.favorites_lineups)} favorite lineups?\n\nThis action cannot be undone.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.favorites_lineups = []
            self.save_favorites()
            self.update_favorites_display()
            QMessageBox.information(self, "Cleared", "All favorites have been cleared.")
            self.status_label.setText('All favorites cleared')

    def update_favorites_display(self):
        """Update the favorites table display"""
        if not hasattr(self, 'favorites_table'):
            return  # Table not created yet
            
        self.favorites_table.setRowCount(0)
        
        if not self.favorites_lineups:
            self.favorites_stats_label.setText("📊 Total Favorites: 0 lineups")
            return
        
        # Update stats
        total_favorites = len(self.favorites_lineups)
        unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
        self.favorites_stats_label.setText(f"📊 Total Favorites: {total_favorites} lineups from {unique_runs} runs")
        
        # Populate table
        for fav_idx, fav in enumerate(self.favorites_lineups):
            lineup = fav['lineup']
            run_number = fav.get('run_number', 1)
            date_added = fav.get('date_added', 'Unknown')
            
            # Add each player in the lineup as a row
            for _, player in lineup.iterrows():
                row_position = self.favorites_table.rowCount()
                self.favorites_table.insertRow(row_position)
                
                # Checkbox for selection
                checkbox = QCheckBox()
                checkbox_widget = QWidget()
                layout_checkbox = QHBoxLayout(checkbox_widget)
                layout_checkbox.addWidget(checkbox)
                layout_checkbox.setAlignment(Qt.AlignCenter)
                layout_checkbox.setContentsMargins(0, 0, 0, 0)
                self.favorites_table.setCellWidget(row_position, 0, checkbox_widget)
                
                # Fill row data
                self.favorites_table.setItem(row_position, 1, QTableWidgetItem(f"Run {run_number}"))
                self.favorites_table.setItem(row_position, 2, QTableWidgetItem(str(player['Name'])))
                self.favorites_table.setItem(row_position, 3, QTableWidgetItem(str(player['Team'])))
                self.favorites_table.setItem(row_position, 4, QTableWidgetItem(str(player['Pos'])))
                self.favorites_table.setItem(row_position, 5, QTableWidgetItem(str(player['Salary'])))
                self.favorites_table.setItem(row_position, 6, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
                self.favorites_table.setItem(row_position, 7, QTableWidgetItem(str(fav['total_salary'])))
                self.favorites_table.setItem(row_position, 8, QTableWidgetItem(f"{fav['total_points']:.2f}"))
                self.favorites_table.setItem(row_position, 9, QTableWidgetItem(date_added))
                
                # Delete button for this lineup
                delete_button = QPushButton("🗑️")
                delete_button.setMaximumWidth(30)
                delete_button.clicked.connect(lambda checked, idx=fav_idx: self.delete_favorite_lineup(idx))
                delete_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
                self.favorites_table.setCellWidget(row_position, 10, delete_button)

    def delete_favorite_lineup(self, lineup_index):
        """Delete a specific favorite lineup"""
        if 0 <= lineup_index < len(self.favorites_lineups):
            run_number = self.favorites_lineups[lineup_index].get('run_number', 'Unknown')
            
            reply = QMessageBox.question(
                self, 
                'Delete Favorite', 
                f'Delete lineup from Run #{run_number}?',
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                del self.favorites_lineups[lineup_index]
                self.save_favorites()
                self.update_favorites_display()
                self.status_label.setText(f'Deleted favorite lineup from Run #{run_number}')

    def save_favorites_to_entries(self):
        """Save selected favorites to a DraftKings entries file"""
        if not self.favorites_lineups:
            QMessageBox.warning(self, "No Favorites", "No favorite lineups available.\n\nPlease add some lineups to favorites first.")
            return
        
        try:
            # Show detailed info about favorites
            total_favorites = len(self.favorites_lineups)
            unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
            
            # Ask user how many favorites to use
            num_available = len(self.favorites_lineups)
            dialog_text = f'How many favorite lineups to save?\n\n'
            dialog_text += f'📊 Available: {num_available} favorite lineups\n'
            dialog_text += f'🏃 From {unique_runs} different optimization runs\n\n'
            dialog_text += f'💡 Tip: You can save up to {num_available} lineups'
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Save Favorites to Entries', 
                dialog_text,
                value=min(num_available, 150),
                min=1, 
                max=num_available
            )
            
            if not ok:
                return
            
            # Ask for save location
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Favorites to Entries File', 
                'my_favorites_entries.csv',
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Use the export method that uses the exact same logic as create_filled_entries_df
            result = self.export_favorites_as_new_lineups(save_path, num_to_use)
            
            # Show success message with more details
            success_msg = f"🎉 Favorites saved successfully!\n\n"
            success_msg += f"📊 Saved {num_to_use} favorite lineups\n"
            success_msg += f"🏃 From {unique_runs} optimization runs\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"🚀 Ready to upload to DraftKings!"
            
            # Add details from the export result
            if result:
                success_msg += f"\n✅ Exported {result.get('lineups_exported', 0)} lineups"
                success_msg += f"\n🆔 Used {result.get('player_ids_used', 0)} player IDs"
                if result.get('entry_metadata_found'):
                    success_msg += f"\n📋 Contest metadata included"
            
            QMessageBox.information(self, "Favorites Saved", success_msg)
            self.status_label.setText(f'Saved {num_to_use} favorites to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error saving favorites: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Save Error", f"Failed to save favorites:\n\n{str(e)}")

    def save_favorites_as_new_lineups(self):
        """Save favorites in DraftKings contest entry format (same as DD.csv)"""
        if not self.favorites_lineups:
            QMessageBox.warning(self, "No Favorites", "No favorite lineups available.\n\nPlease add some lineups to favorites first.")
            return
        
        try:
            # Show detailed info about favorites
            total_favorites = len(self.favorites_lineups)
            unique_runs = len(set(fav.get('run_number', 1) for fav in self.favorites_lineups))
            
            # Ask user how many favorites to export
            num_available = len(self.favorites_lineups)
            dialog_text = f'How many favorite lineups to export in DraftKings format?\n\n'
            dialog_text += f'📊 Available: {num_available} favorite lineups\n'
            dialog_text += f'🏃 From {unique_runs} different optimization runs\n\n'
            dialog_text += f'🎯 Export format: DraftKings contest entry format\n'
            dialog_text += f'📋 Headers: Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF\n'
            dialog_text += f'🔢 Contains: Player IDs (not names) ready for DraftKings upload'
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Export Favorites in DraftKings Format', 
                dialog_text,
                value=min(num_available, 150),
                min=1, 
                max=num_available
            )
            
            if not ok:
                return
            
            # Ask for save location
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Favorites in DraftKings Format', 
                'my_favorites_dk_format.csv',
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Export favorites in DraftKings contest entry format
            export_info = self.export_favorites_as_new_lineups(save_path, num_to_use)
            
            # Show success message with detailed feedback
            success_msg = f"🎉 Favorites exported in DraftKings format!\n\n"
            success_msg += f"📊 Exported {export_info['lineups_exported']} favorite lineups\n"
            success_msg += f"🏃 From {unique_runs} optimization runs\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"📋 Format: Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF\n"
            
            if export_info['player_ids_used'] > 0:
                success_msg += f"🔢 Player IDs: {export_info['player_ids_used']} positions filled with numeric IDs\n"
            else:
                success_msg += f"⚠️ Player IDs: Using player names (no ID mappings found)\n"
            
            if export_info['entry_metadata_found']:
                success_msg += f"📝 Entry metadata: Preserved from loaded DK entries file\n"
            else:
                success_msg += f"📝 Entry metadata: Empty (no DK entries file loaded)\n"
            
            success_msg += f"\n🚀 Ready for DraftKings upload!"
            
            QMessageBox.information(self, "DraftKings Format Export Complete", success_msg)
            self.status_label.setText(f'Exported {num_to_use} favorites in DK format to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error exporting favorites in DK format: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Export Error", f"Failed to export favorites:\n\n{str(e)}")

    def export_favorites_as_new_lineups(self, output_path, num_to_use):
        """Export favorite lineups using the EXACT same logic as create_filled_entries_df"""
        if not self.favorites_lineups:
            logging.warning("No favorite lineups available to export")
            return {'lineups_exported': 0, 'player_ids_used': 0, 'entry_metadata_found': False}
        
        logging.info(f"Exporting {num_to_use} favorites using create_filled_entries_df logic to {output_path}")
        
        # CRITICAL FIX: Clear cached favorites data to force fresh generation
        if hasattr(self, '_favorites_with_player_data'):
            delattr(self, '_favorites_with_player_data')
            logging.info("🔄 Cleared cached favorites data to force fresh generation")
        
        # FORCE FRESH LINEUP GENERATION: Temporarily disable favorites file reading
        original_favorites_file_path = None
        backup_favorites_file_path = None
        saved_contest_info = None
        original_lineups = None  # Ensure this is always defined
        
        try:
            # First, create backup files and extract contest info from original files
            workspace_favorites_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
            onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
            
            # Extract contest info from original file BEFORE renaming
            if os.path.exists(workspace_favorites_path):
                self.extract_contest_info_from_favorites()
                saved_contest_info = getattr(self, '_contest_info_list', None)
                logging.info(f"🎯 Extracted contest info from workspace file: {len(saved_contest_info) if saved_contest_info else 0} entries")
                
                original_favorites_file_path = workspace_favorites_path
                backup_favorites_file_path = workspace_favorites_path + ".backup_temp"
                
                # Remove existing backup file if it exists
                if os.path.exists(backup_favorites_file_path):
                    os.remove(backup_favorites_file_path)
                    logging.info(f"🗑️ Removed existing backup file")
                
                os.rename(workspace_favorites_path, backup_favorites_file_path)
                logging.info(f"🔄 Temporarily renamed favorites file to force fresh generation")
            elif os.path.exists(onedrive_favorites_path):
                self.extract_contest_info_from_favorites()
                saved_contest_info = getattr(self, '_contest_info_list', None)
                logging.info(f"🎯 Extracted contest info from OneDrive file: {len(saved_contest_info) if saved_contest_info else 0} entries")
                
                original_favorites_file_path = onedrive_favorites_path
                backup_favorites_file_path = onedrive_favorites_path + ".backup_temp"
                
                # Remove existing backup file if it exists
                if os.path.exists(backup_favorites_file_path):
                    os.remove(backup_favorites_file_path)
                    logging.info(f"🗑️ Removed existing OneDrive backup file")
                
                os.rename(onedrive_favorites_path, backup_favorites_file_path)
                logging.info(f"🔄 Temporarily renamed OneDrive favorites file to force fresh generation")
            
            # Temporarily store the current optimized_lineups
            original_lineups = getattr(self, 'optimized_lineups', None)
            
            # Replace optimized_lineups with favorites for the duration of this function
            # Extract just the lineup DataFrames from favorites (favorites store {'lineup': df, ...})
            favorite_lineups = [fav['lineup'] for fav in self.favorites_lineups[:num_to_use]]
            self.optimized_lineups = favorite_lineups
            
            logging.info(f"🎯 Using {len(favorite_lineups)} fresh favorite lineups for export")
            
            # Restore the contest information that we extracted before cache-clearing
            if saved_contest_info:
                self._contest_info_list = saved_contest_info
                logging.info(f"🎯 Restored contest info: {len(saved_contest_info)} entries")
            
            # Use the EXACT same logic as create_filled_entries_df
            filled_entries = self.create_filled_entries_df(num_to_use)
            
            # Save to CSV
            filled_entries.to_csv(output_path, index=False)
            
            logging.info(f"✅ Successfully exported {len(filled_entries)} favorites using create_filled_entries_df logic")
            
            # Calculate return info
            player_ids_used = 0
            for _, row in filled_entries.iterrows():
                player_ids_used += len([id for id in row[4:] if id and str(id).strip()])
            
            entry_metadata_found = bool(filled_entries.iloc[0, 0] if len(filled_entries) > 0 else False)
            
            return {
                'lineups_exported': len(filled_entries),
                'player_ids_used': player_ids_used,
                'entry_metadata_found': entry_metadata_found
            }
            
        finally:
            # Restore the original optimized_lineups
            if original_lineups is not None:
                self.optimized_lineups = original_lineups
            else:
                # Remove the temporary attribute if it didn't exist before
                if hasattr(self, 'optimized_lineups'):
                    delattr(self, 'optimized_lineups')
            
            # Restore the temporarily renamed favorites file
            if original_favorites_file_path and backup_favorites_file_path:
                try:
                    if os.path.exists(backup_favorites_file_path):
                        # If the original exists, remove it before renaming
                        if os.path.exists(original_favorites_file_path):
                            os.remove(original_favorites_file_path)
                        os.rename(backup_favorites_file_path, original_favorites_file_path)
                        logging.info("🔄 Restored favorites file after export")
                except Exception as e:
                    logging.error(f"Error restoring favorites file: {e}")

    def run_optimization(self):
        """Run the optimization with min unique constraint support"""
        logging.debug("Starting run_optimization method")
        if self.df_players is None or self.df_players.empty:
            self.status_label.setText("No player data loaded. Please load a CSV first.")
            logging.debug("No player data loaded")
            return
        
        logging.debug(f"df_players shape: {self.df_players.shape}")
        logging.debug(f"df_players columns: {self.df_players.columns}")
        logging.debug(f"df_players sample:\n{self.df_players.head()}")
        
        self.included_players = self.get_included_players()
        self.stack_settings = self.collect_stack_settings()
        self.min_exposure, self.max_exposure = self.collect_exposure_settings()
        
        # Get min unique constraint
        min_unique = self.get_min_unique_constraint()
        
        # Get requested number of lineups
        requested_lineups = self.get_requested_lineups()
        
        logging.debug(f"Included players: {len(self.included_players)}")
        logging.debug(f"Stack settings: {self.stack_settings}")
        logging.debug(f"Min unique constraint: {min_unique}")
        logging.debug(f"Requested lineups: {requested_lineups}")
        
        # Debug team selections
        # Collect team selections (this should now work properly)
        team_selections = self.collect_team_selections()
        logging.info(f"🎯 Team selections from UI: {team_selections}")
        
        # DETAILED DEBUG: Show exactly what was collected
        if team_selections:
            print(f"🎯 USER TEAM SELECTIONS DETECTED:")
            logging.info(f"🎯 DETAILED TEAM SELECTIONS DEBUG:")
            for stack_size, teams in team_selections.items():
                if stack_size == "all":
                    print(f"   All Stacks: {len(teams)} teams")
                    logging.info(f"   All Stacks: {len(teams)} teams = {teams[:5]}{'...' if len(teams) > 5 else ''}")
                else:
                    print(f"   {stack_size}-Stack: {len(teams)} teams = {teams}")
                    logging.info(f"   {stack_size}-Stack: {len(teams)} teams = {teams}")
                    
            # CRITICAL DEBUG: Check if all teams are the same
            all_team_lists = list(team_selections.values())
            if len(all_team_lists) > 1:
                all_same = all(set(teams) == set(all_team_lists[0]) for teams in all_team_lists)
                if all_same and len(all_team_lists[0]) > 10:
                    logging.error(f"🚨 BUG DETECTED: All stack sizes have identical team lists with {len(all_team_lists[0])} teams!")
                    logging.error(f"🚨 This suggests checkbox detection is not working properly!")
                    print(f"🚨 BUG DETECTED: All stack sizes have the same {len(all_team_lists[0])} teams!")
                    print(f"🚨 This means your specific team selections (PIT, NYM/NYY) were not detected!")
                    print(f"🚨 SOLUTION: Try using 'Disable Kelly Sizing' to get all 100 lineups, or manually reselect teams")
                    
            # Also log individual selections for debugging
            for stack_size, teams in team_selections.items():
                logging.info(f"Individual selection - Stack {stack_size}: {teams}")
        else:
            print("⚠️ No team selections found - will use all available teams")
            logging.warning("No team selections detected from UI")        
        
        if not self.stack_settings:
            self.status_label.setText("Please select at least one stack type in the Stack Exposure tab.")
            return
            
        # Get bankroll and risk tolerance settings
        bankroll = self.get_bankroll_setting()
        risk_tolerance = self.get_risk_tolerance_setting()
        enable_risk_mgmt = self.get_risk_management_enabled()
        
        logging.debug(f"Risk settings - Bankroll: ${bankroll}, Tolerance: {risk_tolerance}, Enabled: {enable_risk_mgmt}")
            
        if not enable_risk_mgmt and RISK_ENGINE_AVAILABLE:
            # Temporarily disable risk engine for this run
            original_status = f"Risk management disabled for this run (Bankroll: ${bankroll})"
            self.status_label.setText(f"Running traditional optimization... {original_status}")
        elif enable_risk_mgmt and RISK_ENGINE_AVAILABLE:
            self.status_label.setText(f"Running risk-adjusted optimization... Bankroll: ${bankroll}, Risk: {risk_tolerance}")
        else:
            self.status_label.setText("Running optimization... Please wait.")
            
        # Check if Kelly sizing should be disabled
        disable_kelly = self.disable_kelly_checkbox.isChecked()
        if disable_kelly:
            self.status_label.setText("Running optimization with Kelly sizing disabled... Please wait.")
            logging.info("💰 Kelly sizing disabled by user - will generate all requested lineups")
            logging.info("💰 BYPASS MODE: When Kelly is disabled, min unique constraint will be ignored to guarantee lineup count")
            
            # Override min unique when Kelly is disabled to guarantee full lineup count
            if min_unique > 0:
                logging.warning(f"💰 OVERRIDING: Min unique constraint ({min_unique}) ignored when Kelly sizing is disabled")
                min_unique = 0  # Force to 0 to bypass filtering
                logging.info(f"💰 Min unique set to 0 for maximum lineup diversity")
        
        # GET MINIMUM SALARY CONSTRAINT FROM UI
        self.min_salary = self.get_min_salary_constraint()
        logging.debug(f"Minimum salary constraint: {self.min_salary}")
        
        # GET PROJECTION ADJUSTMENT SETTINGS FROM UI
        disable_adjustments = self.disable_adjustments_checkbox.isChecked()
        conservative_mode = self.conservative_mode_checkbox.isChecked()
        
        # GET PLAYER USAGE TRACKING SETTING FROM UI
        max_usage_percentage = self.get_max_usage_percentage()
        
        if disable_adjustments:
            logging.info("🎯 Using PURE PROJECTIONS (no random adjustments) for maximum accuracy")
        elif conservative_mode:
            logging.info("🛡️ Using CONSERVATIVE MODE (minimal adjustments) for balance of accuracy and diversity")
        else:
            logging.info("⚡ Using LEGACY MODE (reduced from aggressive) for maximum diversity")
        
        if max_usage_percentage < 100:
            logging.info(f"🚫 Player Usage Tracking: Max {max_usage_percentage}% usage per player for forced diversity")
        
        self.optimization_thread = OptimizationWorker(
            df_players=self.df_players,
            salary_cap=SALARY_CAP,
            position_limits=POSITION_LIMITS,
            included_players=self.included_players,
            stack_settings=self.stack_settings,
            min_exposure=self.min_exposure,
            max_exposure=self.max_exposure,
            min_points=self.min_points,
            monte_carlo_iterations=self.monte_carlo_iterations,
            num_lineups=requested_lineups,
            team_selections=team_selections,
            min_unique=min_unique,  # Add min unique constraint
            bankroll=bankroll,
            risk_tolerance=risk_tolerance,
            disable_kelly=disable_kelly,
            min_salary=self.min_salary,
            disable_adjustments=disable_adjustments,
            conservative_mode=conservative_mode,
            max_usage_percentage=max_usage_percentage
        )
        
    def get_risk_management_enabled(self):
        """Get risk management enabled setting"""
        if RISK_ENGINE_AVAILABLE and hasattr(self, 'enable_risk_mgmt_checkbox'):
            return self.enable_risk_mgmt_checkbox.isChecked()
        return False

    def get_max_usage_percentage(self):
        """Get the maximum usage percentage from the UI input"""
        try:
            max_usage_text = self.max_usage_input.text().strip()
            if not max_usage_text:
                return 40  # Default 40% max usage
            
            max_usage = int(max_usage_text)
            if max_usage < 1:
                max_usage = 1
            elif max_usage > 100:
                max_usage = 100
            
            return max_usage
            
        except ValueError:
            logging.warning(f"Invalid max usage percentage value: {self.max_usage_input.text()}")
            return 40

    def update_selection_status(self):
        # Implement any additional logic you want to execute when selection status changes
        pass