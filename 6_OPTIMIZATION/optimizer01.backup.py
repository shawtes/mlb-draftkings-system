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
from itertools import combinations
import csv
import json
from collections import defaultdict

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

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
SALARY_CAP = 50000
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
    df, stack_type, team_projected_runs, team_selections = args
    logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type}")
    
    problem = pulp.LpProblem("Stack_Optimization", pulp.LpMaximize)
    player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

    # Objective: Maximize projected points
    problem += pulp.lpSum([df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] for idx in df.index])

    # Basic constraints
    problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
    problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= SALARY_CAP
    for position, limit in POSITION_LIMITS.items():
        problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Pos']]) == limit

    # Handle different stack types
    if stack_type == "No Stacks":
        # No stacking constraints - just basic position and salary constraints
        logging.debug("optimize_single_lineup: Using no stacks")
    else:
        # Implement stacking with proper team selection enforcement
        stack_sizes = [int(size) for size in stack_type.split('|')]
        logging.debug(f"optimize_single_lineup: Stack sizes: {stack_sizes}")
        logging.debug(f"optimize_single_lineup: Team selections: {team_selections}")
        
        # Simplified approach: For each stack size, randomly pick one of the available teams
        # and enforce that constraint. This avoids creating too many binary variables.
        import random
        
        for i, size in enumerate(stack_sizes):
            # Get teams available for this specific stack size
            if isinstance(team_selections, dict) and size in team_selections:
                available_teams = team_selections[size]
            elif isinstance(team_selections, list):
                available_teams = team_selections
            else:
                # Fallback to all teams in data
                available_teams = df['Team'].unique().tolist()
            
            logging.debug(f"optimize_single_lineup: Stack {i+1} (size {size}) - Available teams: {available_teams}")
            
            if not available_teams:
                logging.debug(f"optimize_single_lineup: No teams available for stack size {size}, skipping")
                continue
            
            # Filter available teams to only those with enough batters
            valid_teams = []
            for team in available_teams:
                team_batters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P', na=False))].index
                if len(team_batters) >= size:
                    valid_teams.append(team)
                    
            if not valid_teams:
                logging.debug(f"optimize_single_lineup: No valid teams with enough batters for stack size {size}")
                continue
                
            # Randomly select one team from valid teams for this stack
            selected_team = random.choice(valid_teams)
            team_batters = df[(df['Team'] == selected_team) & (~df['Pos'].str.contains('P', na=False))].index
            
            # Add constraint: we must have at least 'size' batters from this team
            problem += pulp.lpSum([player_vars[idx] for idx in team_batters]) >= size
            
            logging.debug(f"optimize_single_lineup: Enforcing {size}-stack from team {selected_team} (has {len(team_batters)} available batters)")

    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    status = problem.solve(solver)

    if pulp.LpStatus[status] == 'Optimal':
        lineup = df.loc[[idx for idx in df.index if player_vars[idx].varValue > 0.5]]
        
        # Log the actual team composition for debugging
        team_counts = lineup['Team'].value_counts()
        logging.debug(f"optimize_single_lineup: Found optimal solution with {len(lineup)} players")
        logging.debug(f"optimize_single_lineup: Team composition: {dict(team_counts)}")
        
        return lineup, stack_type
    else:
        logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
        logging.debug(f"Constraints: {problem.constraints}")
        return pd.DataFrame(), stack_type
def simulate_iteration(df):
    random_factors = np.random.normal(1, 0.1, size=len(df))
    df = df.copy()
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * random_factors
    df['Predicted_DK_Points'] = df['Predicted_DK_Points'].clip(lower=1)
    return df

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(dict, dict, dict)
    
    def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations, num_lineups, team_selections, min_unique=0, bankroll=1000, risk_tolerance='medium'):
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

    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")
        
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for stack_type in self.stack_settings:
                for _ in range(self.num_lineups):
                    future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections))
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    lineup, stack_type = future.result()
                    if lineup.empty:
                        logging.debug(f"optimize_lineups: Empty lineup returned for stack type {stack_type}")
                    else:
                        total_points = lineup['Predicted_DK_Points'].sum()
                        results[len(results)] = {'total_points': total_points, 'lineup': lineup}
                        for team in lineup['Team'].unique():
                            team_exposure[team] += 1
                        stack_exposure[stack_type] += 1
                        logging.debug(f"optimize_lineups: Found valid lineup for stack type {stack_type}")
                except Exception as e:
                    logging.error(f"Error in optimization: {str(e)}")

        logging.debug(f"optimize_lineups: Completed. Found {len(results)} valid lineups")
        logging.debug(f"Team exposure: {dict(team_exposure)}")
        logging.debug(f"Stack exposure: {dict(stack_exposure)}")
        
        return results, team_exposure, stack_exposure

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
            # Generate more candidates than needed for better selection
            candidate_multiplier = 3
            
            for stack_type in self.stack_settings:
                for _ in range(self.num_lineups * candidate_multiplier):
                    future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections))
                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    lineup, stack_type = future.result()
                    if not lineup.empty:
                        lineup_data = self.calculate_lineup_metrics(lineup, stack_type)
                        lineup_candidates.append(lineup_data)
                        
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
            
            # Track exposure
            for team in lineup_df['Team'].unique():
                team_exposure[team] += 1
            stack_exposure[lineup_data['stack_type']] += 1
        
        logging.info(f"🎯 Risk-adjusted optimization complete. Selected {len(results)} optimal lineups")
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
        """Select lineups based on risk profile and diversification"""
        if not lineup_candidates:
            return []
        
        # Sort by risk-adjusted score
        sorted_candidates = sorted(lineup_candidates, key=lambda x: x.get('risk_adjusted_score', 0), reverse=True)
        
        # Apply diversity filters
        selected = []
        used_core_players = set()
        
        for candidate in sorted_candidates:
            if len(selected) >= self.num_lineups:
                break
                
            # Check for excessive overlap with already selected lineups
            lineup_df = candidate['lineup']
            core_players = set(lineup_df['Name'].tolist())
            
            # Allow some overlap but not too much
            overlap_threshold = 7  # Max 7 overlapping players
            max_overlap = max([len(core_players.intersection(used_players)) for used_players in used_core_players] + [0])
            
            if max_overlap < overlap_threshold:
                selected.append(candidate)
                used_core_players.add(frozenset(core_players))
                logging.debug(f"Selected lineup with risk score: {candidate.get('risk_adjusted_score', 0):.2f}")
        
        logging.info(f"🎲 Selected {len(selected)} diverse lineups from {len(lineup_candidates)} candidates")
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
            if not weights or len(weights) != len(selected_lineups):
                weights = [1.0/len(selected_lineups)] * len(selected_lineups)
            
            position_limits = None
            if self.bankroll_manager:
                # Calculate average edge and volatility for position sizing
                avg_edge = np.mean([l.get('salary_efficiency', 0) for l in selected_lineups])
                avg_volatility = np.mean([l.get('risk_metrics', RiskMetrics(0,0,0,0,0,0)).volatility for l in selected_lineups])
                
                position_limits = self.bankroll_manager.calculate_position_limits(avg_edge, avg_volatility)
                logging.info(f"💰 Kelly sizing: Recommended {position_limits.get('recommended_lineups', 1)} lineups")
            
            # Apply position sizing to lineups
            final_lineups = []
            recommended_lineups = position_limits.get('recommended_lineups', len(selected_lineups)) if position_limits else len(selected_lineups)
            
            # Limit to recommended number of lineups
            lineups_to_use = min(len(selected_lineups), recommended_lineups, self.num_lineups)
            
            for i in range(lineups_to_use):
                lineup = selected_lineups[i].copy()
                lineup['allocation_weight'] = weights[i] if i < len(weights) else 1.0/lineups_to_use
                lineup['position_size'] = position_limits.get('optimal_position_size', 100) * weights[i] if position_limits and i < len(weights) else 100
                final_lineups.append(lineup)
            
            return final_lineups
            
        except Exception as e:
            logging.error(f"Error applying position sizing: {e}")
            return selected_lineups[:self.num_lineups]

class FantasyBaseballApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)
        self.setup_ui()
        
        self.included_players = []
        self.stack_settings = {}
        self.min_exposure = {}
        self.max_exposure = {}
        self.min_points = 1
        self.monte_carlo_iterations = 100

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

    def refresh_team_stacks(self):
        self.populate_team_stack_table()

    def select_all_teams(self, stack_size):
        """Select all teams in a specific team stack table"""
        if not hasattr(self, 'team_stack_tables') or stack_size not in self.team_stack_tables:
            logging.debug(f"No team stack table found for: {stack_size}")
            return
        
        table = self.team_stack_tables[stack_size]
        selected_count = 0
        
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
            success_msg += f"� Format: Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF\n"
            
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
                os.rename(workspace_favorites_path, backup_favorites_file_path)
                logging.info(f"🔄 Temporarily renamed favorites file to force fresh generation")
            elif os.path.exists(onedrive_favorites_path):
                self.extract_contest_info_from_favorites()
                saved_contest_info = getattr(self, '_contest_info_list', None)
                logging.info(f"🎯 Extracted contest info from OneDrive file: {len(saved_contest_info) if saved_contest_info else 0} entries")
                
                original_favorites_file_path = onedrive_favorites_path
                backup_favorites_file_path = onedrive_favorites_path + ".backup_temp"
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
        team_selections = self.collect_team_selections()
        logging.debug(f"Team selections from UI: {team_selections}")        
        
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
            risk_tolerance=risk_tolerance
        )
        
        # Pass the enable flag to the worker
        if hasattr(self.optimization_thread, 'risk_engine') and not enable_risk_mgmt:
            self.optimization_thread.risk_engine = None  # Disable for this run
        self.optimization_thread.optimization_done.connect(self.display_results)
        logging.debug("Starting optimization thread")
        self.optimization_thread.start()
        
        self.status_label.setText("Running optimization... Please wait.")

    def get_requested_lineups(self):
        """Get the requested number of lineups from the UI input"""
        try:
            num_lineups_text = self.num_lineups_input.text().strip()
            if not num_lineups_text:
                return 100  # Default
            
            num_lineups = int(num_lineups_text)
            if num_lineups < 1:
                num_lineups = 1
            elif num_lineups > 500:
                num_lineups = 500  # Max 500 lineups
            
            return num_lineups
            
        except ValueError:
            logging.warning(f"Invalid number of lineups value: {self.num_lineups_input.text()}")
            return 100

    def get_min_unique_constraint(self):
        """Get the min unique constraint from the UI input"""
        try:
            min_unique_text = self.min_unique_input.text().strip()
            if not min_unique_text:
                return 0  # Default: no constraint
            
            min_unique = int(min_unique_text)
            if min_unique < 0:
                min_unique = 0
            elif min_unique > 10:
                min_unique = 10  # Max 10 unique players per lineup
            
            return min_unique
            
        except ValueError:
            logging.warning(f"Invalid min unique value: {self.min_unique_input.text()}")
            return 0

    def get_included_players(self):
        """Get the list of included players from the UI with better checkbox detection"""
        included_players = []
        total_checkboxes = 0
        checked_checkboxes = 0
        
        # IMMEDIATE DEBUG OUTPUT
        print(f"\n🔍 CHECKBOX DEBUG - IMPROVED DETECTION:")
        
        if not hasattr(self, 'player_tables') or not self.player_tables:
            print(f"   ❌ No player tables available")
            logging.debug("No player tables available")
            return included_players
        
        print(f"   📋 Found {len(self.player_tables)} player tables")
        
        # Check ALL position tables for selected players with improved detection
        for position_name, table in self.player_tables.items():
            position_checked = 0
            position_total = 0
            
            print(f"   🔍 Checking {position_name} table ({table.rowCount()} rows)...")
            
            for row in range(table.rowCount()):
                # Multiple methods to find the checkbox
                checkbox = None
                
                # Method 1: Check if there's a checkbox widget directly
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    # Look for QCheckBox in the widget
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if not checkbox:
                        # Maybe the widget itself is a QCheckBox
                        if isinstance(checkbox_widget, QCheckBox):
                            checkbox = checkbox_widget
                        else:
                            # Look through all children
                            for child in checkbox_widget.findChildren(QCheckBox):
                                checkbox = child
                                break
                
                # Method 2: Check if the table item itself has checkbox
                if not checkbox:
                    table_item = table.item(row, 0)
                    if table_item and table_item.checkState() is not None:
                        # This item has a checkbox state
                        total_checkboxes += 1
                        position_total += 1
                        
                        if table_item.checkState() == Qt.Checked:
                            checked_checkboxes += 1
                            position_checked += 1
                            # Get player name from column 1
                            name_item = table.item(row, 1)
                            if name_item:
                                player_name = name_item.text().strip()
                                if player_name and player_name not in included_players:
                                    included_players.append(player_name)
                        continue
                
                # Method 3: Use the found checkbox widget
                if checkbox:
                    total_checkboxes += 1
                    position_total += 1
                    
                    if checkbox.isChecked():
                        checked_checkboxes += 1
                        position_checked += 1
                        # Get player name from column 1
                        name_item = table.item(row, 1)
                        if name_item:
                            player_name = name_item.text().strip()
                            if player_name and player_name not in included_players:
                                included_players.append(player_name)
            
            print(f"     ✅ {position_name}: {position_checked}/{position_total} players selected")
            logging.debug(f"Position {position_name}: {position_checked}/{position_total} players selected")
        
        print(f"   📊 TOTAL: {checked_checkboxes}/{total_checkboxes} checkboxes checked")
        print(f"   🎯 FINAL: {len(included_players)} unique players selected")
        
        # Show warning if too many players selected
        if len(included_players) > 150:
            print(f"   ⚠️  WARNING: {len(included_players)} players selected!")
            print(f"      This is a lot of players. The optimizer can pick from any of these.")
            print(f"      Consider being more selective if you want specific players.")
        elif len(included_players) > 50:
            print(f"   � INFO: {len(included_players)} players selected")
            print(f"      The optimizer will choose the best lineups from these players.")
        
        if len(included_players) > 0:
            print(f"   �👥 FIRST 5 PLAYERS: {included_players[:5]}")
        
        print(f"")
        
        # Update the status label
        if hasattr(self, 'selection_status_label'):
            if len(included_players) == 0:
                self.selection_status_label.setText("No players selected")
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px;")
            elif len(included_players) <= 50:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Good size)")
                self.selection_status_label.setStyleSheet("color: #4CAF50; font-size: 10px;")
            elif len(included_players) <= 150:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Large pool)")
                self.selection_status_label.setStyleSheet("color: #FF9800; font-size: 10px;")
            else:
                self.selection_status_label.setText(f"{len(included_players)} players selected (Very large!)")
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px; font-weight: bold;")
        
        logging.info(f"🎯 CHECKBOX SELECTION SUMMARY:")
        logging.info(f"   Total checkboxes found: {total_checkboxes}")
        logging.info(f"   Checkboxes checked: {checked_checkboxes}")
        logging.info(f"   Final included players: {len(included_players)}")
        
        if len(included_players) > 0:
            logging.info(f"   First 5 selected players: {included_players[:5]}")
        
        return included_players

    def collect_stack_settings(self):
        """Collect stack settings from the UI"""
        stack_settings = []
        
        if not hasattr(self, 'stack_exposure_table') or not self.stack_exposure_table:
            logging.debug("No stack exposure table available")
            return ["No Stacks"]  # Default
        
        # Check which stack types are selected
        for row in range(self.stack_exposure_table.rowCount()):
            checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    stack_type_item = self.stack_exposure_table.item(row, 1)
                    if stack_type_item:
                        stack_settings.append(stack_type_item.text())
        
        if not stack_settings:
            stack_settings = ["No Stacks"]  # Default if nothing selected
        
        logging.debug(f"Found stack settings: {stack_settings}")
        return stack_settings

    def collect_exposure_settings(self):
        """Collect min and max exposure settings from the UI"""
        min_exposure = {}
        max_exposure = {}
        
        if not hasattr(self, 'player_tables') or not self.player_tables:
            logging.debug("No player tables available for exposure settings")
            return min_exposure, max_exposure
        
        # For now, return empty dicts as exposure constraints aren't fully implemented
        logging.debug("Exposure settings collection not fully implemented")
        return min_exposure, max_exposure

    def collect_team_selections(self):
        """Collect team selections from the team stack UI"""
        team_selections = {}
        
        if not hasattr(self, 'team_stack_tables') or not self.team_stack_tables:
            logging.debug("No team stack tables available")
            return team_selections
        
        # For now, return empty dict - this can be enhanced later
        logging.debug("Team selections collection not fully implemented")
        return team_selections

    def display_results(self, results, team_exposure, stack_exposure):
        """Display optimization results with unique constraint filtering and risk information"""
        logging.debug(f"display_results: Received {len(results)} results")
        self.results_table.setRowCount(0)
        
        # Check if we have risk information in results
        has_risk_info = any('risk_info' in lineup_data for lineup_data in results.values())
        
        # Update table headers to include risk info if available
        if has_risk_info:
            self.results_table.setColumnCount(13)
            self.results_table.setHorizontalHeaderLabels([
                "Player", "Team", "Pos", "Salary", "Predicted_DK_Points", 
                "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)",
                "Sharpe Ratio", "Kelly %", "Risk Score", "Position $"
            ])
        else:
            self.results_table.setColumnCount(9)
            self.results_table.setHorizontalHeaderLabels([
                "Player", "Team", "Pos", "Salary", "Predicted_DK_Points", 
                "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"
            ])
        
        # Get requested number of lineups from UI
        requested_lineups = self.get_requested_lineups()
        
        # Apply min unique filtering if specified
        min_unique = self.get_min_unique_constraint()
        if min_unique > 0:
            results = self.filter_lineups_by_uniqueness(results, min_unique)
            logging.debug(f"After min unique filtering ({min_unique}): {len(results)} results")
        
        total_lineups = len(results)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)

        self.optimized_lineups = []
        for _, lineup_data in sorted_results:
            self.add_lineup_to_results(lineup_data, total_lineups, has_risk_info)
            self.optimized_lineups.append(lineup_data['lineup'])

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.refresh_team_stacks()
        
        # Create status message with risk information
        status_message = self.create_status_message(total_lineups, requested_lineups, min_unique, has_risk_info, results)
        self.status_label.setText(status_message)
        
        # Show detailed feedback dialog for significant differences
        if total_lineups < requested_lineups * 0.8:  # Less than 80% of requested
            self.show_optimization_feedback(total_lineups, requested_lineups, min_unique, has_risk_info)
    
    def create_status_message(self, total_lineups, requested_lineups, min_unique, has_risk_info, results):
        """Create comprehensive status message"""
        unique_msg = f" (Min unique: {min_unique})" if min_unique > 0 else ""
        risk_msg = ""
        
        if has_risk_info and results:
            # Calculate average risk metrics
            risk_infos = [lineup_data.get('risk_info', {}) for lineup_data in results.values()]
            avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for r in risk_infos if r])
            avg_kelly = np.mean([r.get('kelly_fraction', 0) for r in risk_infos if r]) * 100
            total_position = sum([r.get('position_size', 0) for r in risk_infos if r])
            
            risk_msg = f" | 🔥 Avg Sharpe: {avg_sharpe:.3f}, Kelly: {avg_kelly:.1f}%, Total: ${total_position:.0f}"
        
        if total_lineups < requested_lineups:
            return f"⚠️ Generated {total_lineups}/{requested_lineups} lineups{unique_msg}{risk_msg}"
        else:
            return f"✅ Generated {total_lineups} lineups{unique_msg}{risk_msg}"
    
    def show_optimization_feedback(self, total_lineups, requested_lineups, min_unique, has_risk_info):
        """Show detailed optimization feedback dialog"""
        risk_text = "\n🔥 Risk-adjusted optimization was used" if has_risk_info else ""
        
        QMessageBox.information(
            self,
            "Lineup Generation Results",
            f"🎯 Optimization Results:\n\n"
            f"📊 Requested: {requested_lineups} lineups\n"
            f"✅ Generated: {total_lineups} lineups\n\n"
            f"{'🔄 Min unique constraint limited results' if min_unique > 0 else '⚡ Limited by available player combinations'}"
            f"{risk_text}\n\n"
            f"💡 Tip: Try reducing min unique constraint or adjusting stack settings for more lineups."
        )

    def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
        """Update exposure statistics in all UI tabs"""
        # Update stack exposure in the stack exposure table
        if hasattr(self, 'stack_exposure_table') and self.stack_exposure_table:
            for row in range(self.stack_exposure_table.rowCount()):
                stack_type_item = self.stack_exposure_table.item(row, 1)
                if stack_type_item:
                    stack_type = stack_type_item.text()
                    exposure_count = stack_exposure.get(stack_type, 0)
                    exposure_percentage = (exposure_count / total_lineups * 100) if total_lineups > 0 else 0
                    
                    # Update lineup exposure (column 4)
                    self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
                    
                    # Pool exposure and Entry exposure can be the same for now (columns 5 and 6)
                    self.stack_exposure_table.setItem(row, 5, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
                    self.stack_exposure_table.setItem(row, 6, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
        
        # Update team exposure in team stack tables
        if hasattr(self, 'team_stack_tables') and self.team_stack_tables:
            for stack_size_name, table in self.team_stack_tables.items():
                for row in range(table.rowCount()):
                    team_item = table.item(row, 1)  # Teams column
                    if team_item:
                        team_name = team_item.text()
                        exposure_count = team_exposure.get(team_name, 0)
                        exposure_percentage = (exposure_count / total_lineups * 100) if total_lineups > 0 else 0
                        
                        # Update actual exposure (column 7)
                        table.setItem(row, 7, QTableWidgetItem(f"{exposure_percentage:.1f}%"))
        
        # Update player exposure in player tables (this is already handled in add_lineup_to_results)
        logging.debug(f"Updated exposure in all tabs: {total_lineups} lineups, {len(team_exposure)} teams, {len(stack_exposure)} stacks")

    def filter_lineups_by_uniqueness(self, results, min_unique):
        """Filter lineups to ensure minimum number of unique players between consecutive lineups"""
        if min_unique <= 0 or len(results) <= 1:
            return results
        
        filtered_results = {}
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)
        
        if sorted_results:
            # Always keep the first (best) lineup
            first_key, first_data = sorted_results[0]
            filtered_results[0] = first_data
            kept_lineups = [set(first_data['lineup']['Name'].tolist())]
        
        kept_count = 1
        for key, lineup_data in sorted_results[1:]:
            current_players = set(lineup_data['lineup']['Name'].tolist())
            
            # Check uniqueness against all previously kept lineups
            is_unique_enough = True
            for kept_lineup_players in kept_lineups:
                unique_players = len(current_players - kept_lineup_players)
                if unique_players < min_unique:
                    is_unique_enough = False
                    break
            
            if is_unique_enough:
                filtered_results[kept_count] = lineup_data
                kept_lineups.append(current_players)
                kept_count += 1
        
        logging.debug(f"Min unique filtering: kept {len(filtered_results)} out of {len(results)} lineups")
        return filtered_results

    def add_lineup_to_results(self, lineup_data, total_lineups, has_risk_info=False):
        """Add a lineup to the results table with optional risk information"""
        total_points = lineup_data['total_points']
        lineup = lineup_data['lineup']
        total_salary = lineup['Salary'].sum()
        risk_info = lineup_data.get('risk_info', {})

        for _, player in lineup.iterrows():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            
            # Basic lineup information
            self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Pos'])))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
            self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
            self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

            # Exposure calculation
            player_name = player['Name']
            if player_name in self.player_exposure:
                self.player_exposure[player_name] += 1
            else:
                self.player_exposure[player_name] = 1

            exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
            self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
            self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))
            
            # Add risk information if available
            if has_risk_info and risk_info:
                sharpe_ratio = risk_info.get('sharpe_ratio', 0)
                kelly_fraction = risk_info.get('kelly_fraction', 0) * 100  # Convert to percentage
                volatility = risk_info.get('volatility', 0)
                position_size = risk_info.get('position_size', 0)
                
                # Color code Sharpe ratio for easy interpretation
                sharpe_item = QTableWidgetItem(f"{sharpe_ratio:.3f}")
                if sharpe_ratio > 1.0:
                    sharpe_item.setBackground(QColor(76, 175, 80))  # Green for excellent
                elif sharpe_ratio > 0.5:
                    sharpe_item.setBackground(QColor(255, 193, 7))  # Yellow for good
                elif sharpe_ratio < 0:
                    sharpe_item.setBackground(QColor(244, 67, 54))  # Red for poor
                
                self.results_table.setItem(row_position, 9, sharpe_item)
                self.results_table.setItem(row_position, 10, QTableWidgetItem(f"{kelly_fraction:.1f}%"))
                self.results_table.setItem(row_position, 11, QTableWidgetItem(f"{volatility:.2f}"))
                self.results_table.setItem(row_position, 12, QTableWidgetItem(f"${position_size:.0f}"))

    def load_file(self):
        """Load player data from CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open CSV File', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_players = self.load_players(file_path)
                self.populate_player_tables()
                self.status_label.setText(f'Players loaded: {len(self.df_players)} players')
            except Exception as e:
                self.status_label.setText(f'Error loading file: {str(e)}')
                logging.error(f"Error loading file: {e}")

    def load_players(self, file_path):
        """Load players from CSV file with error handling and flexible column mapping"""
        try:
            df = pd.read_csv(file_path)
            
            # Basic required columns
            basic_required = ['Name', 'Team', 'Pos', 'Salary']
            
            # Check for basic required columns
            missing_basic = [col for col in basic_required if col not in df.columns]
            if missing_basic:
                raise ValueError(f"Missing required columns: {missing_basic}")
            
            # Handle different prediction column names flexibly
            prediction_column = None
            possible_prediction_columns = [
                'Predicted_DK_Points',  # Standard expected name
                'My_Proj',              # Your CSV format
                'ML_Prediction',        # ML prediction column
                'PPG_Projection',       # PPG projection column
                'Projection',           # Generic projection
                'Points',               # Simple points
                'DK_Points',            # DraftKings points
                'Fantasy_Points'        # Fantasy points
            ]
            
            # Find the first available prediction column
            for col in possible_prediction_columns:
                if col in df.columns:
                    prediction_column = col
                    break
            
            if prediction_column is None:
                available_cols = list(df.columns)
                raise ValueError(f"No prediction column found. Available columns: {available_cols}. Expected one of: {possible_prediction_columns}")
            
            # Rename the prediction column to the standard name for consistency
            if prediction_column != 'Predicted_DK_Points':
                df = df.rename(columns={prediction_column: 'Predicted_DK_Points'})
                logging.info(f"Using '{prediction_column}' as prediction column, renamed to 'Predicted_DK_Points'")
            
            # Clean and validate data
            df = df.dropna(subset=['Name', 'Salary', 'Predicted_DK_Points'])
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            df['Predicted_DK_Points'] = pd.to_numeric(df['Predicted_DK_Points'], errors='coerce')
            
            # Remove rows with invalid salary or prediction values
            df = df.dropna(subset=['Salary', 'Predicted_DK_Points'])
            df = df[df['Salary'] > 0]
            df = df[df['Predicted_DK_Points'] > 0]
            
            logging.info(f"Successfully loaded {len(df)} players from {file_path}")
            logging.info(f"Using prediction column: {prediction_column}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading players: {str(e)}")

    def populate_player_tables(self):
        """Populate player tables with loaded data"""
        if self.df_players is None or self.df_players.empty:
            return
        
        # Clear existing player exposure data
        self.player_exposure = {}
        
        # Group players by position
        position_groups = {
            'All Batters': self.df_players[~self.df_players['Pos'].str.contains('P', na=False)],
            'C': self.df_players[self.df_players['Pos'].str.contains('C', na=False)],
            '1B': self.df_players[self.df_players['Pos'].str.contains('1B', na=False)],
            '2B': self.df_players[self.df_players['Pos'].str.contains('2B', na=False)],
            '3B': self.df_players[self.df_players['Pos'].str.contains('3B', na=False)],
            'SS': self.df_players[self.df_players['Pos'].str.contains('SS', na=False)],
            'OF': self.df_players[self.df_players['Pos'].str.contains('OF', na=False)],
            'P': self.df_players[self.df_players['Pos'].str.contains('P', na=False)]
        }
        
        # Populate each table
        for position, table in self.player_tables.items():
            if position in position_groups:
                df_pos = position_groups[position]
                self.populate_position_table(table, df_pos)
        
        # Also populate team stack tables when player data is loaded
        self.populate_team_stack_table()
        
        # Update selection status after all tables are populated
        self.update_selection_status()

    def populate_position_table(self, table, df_pos):
        """Populate a specific position table with player data"""
        table.setRowCount(len(df_pos))
        
        for row_idx, (_, player) in enumerate(df_pos.iterrows()):
            # Checkbox for inclusion
            checkbox = QCheckBox()
            # Create unique identifier for each checkbox
            player_name = player['Name']
            checkbox.setObjectName(f"player_checkbox_{row_idx}_{player_name}")
            
            # Enhanced signal connection with logging for individual changes
            def create_checkbox_handler(name, pos_name, cb):
                def handle_checkbox_change(state):
                    checked = (state == 2)  # Qt.Checked = 2
                    action = "SELECTED" if checked else "DESELECTED"
                    print(f"🔄 CHECKBOX CHANGE: {action} {name} ({pos_name})")
                    logging.info(f"Player {action}: {name} in {pos_name} position")
                    
                    # Update selection status
                    self.update_selection_status()
                    
                    # If this was a deselection, show additional info
                    if not checked:
                        included_players = self.get_included_players_quick()
                        print(f"   📊 Remaining selected: {len(included_players)} players")
                        
                return handle_checkbox_change
            
            # Connect the enhanced signal handler
            checkbox.stateChanged.connect(create_checkbox_handler(player_name, table.objectName() if hasattr(table, 'objectName') else "Unknown", checkbox))
            
            # Store checkbox reference for debugging/access
            if not hasattr(self, '_player_checkboxes'):
                self._player_checkboxes = {}
            self._player_checkboxes[player_name] = checkbox
            
            checkbox_widget = QWidget()
            layout = QHBoxLayout(checkbox_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            table.setCellWidget(row_idx, 0, checkbox_widget)
            
            # Player data
            table.setItem(row_idx, 1, QTableWidgetItem(str(player['Name'])))
            table.setItem(row_idx, 2, QTableWidgetItem(str(player['Team'])))
            table.setItem(row_idx, 3, QTableWidgetItem(str(player['Pos'])))
            table.setItem(row_idx, 4, QTableWidgetItem(str(player['Salary'])))
            table.setItem(row_idx, 5, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
            
            # Value calculation
            value = player['Predicted_DK_Points'] / (player['Salary'] / 1000) if player['Salary'] > 0 else 0
            table.setItem(row_idx, 6, QTableWidgetItem(f"{value:.2f}"))
            
            # Exposure controls
            min_exp_spinbox = QSpinBox()
            min_exp_spinbox.setRange(0, 100)
            min_exp_spinbox.setValue(0)
            table.setCellWidget(row_idx, 7, min_exp_spinbox)
            
            max_exp_spinbox = QSpinBox()
            max_exp_spinbox.setRange(0, 100)
            max_exp_spinbox.setValue(100)
            table.setCellWidget(row_idx, 8, max_exp_spinbox)
            
            # Actual exposure (will be updated after optimization)
            actual_exp_label = QLabel("0.00%")
            table.setCellWidget(row_idx, 9, actual_exp_label)

    def populate_team_stack_table(self):
        """Populate team stack tables"""
        if self.df_players is None or self.df_players.empty:
            return
        
        # Get unique teams and their projected runs
        teams = self.df_players['Team'].unique()
        
        for stack_size_name, table in self.team_stack_tables.items():
            table.setRowCount(len(teams))
            
            for row_idx, team in enumerate(teams):
                # Checkbox for selection
                checkbox = QCheckBox()
                checkbox_widget = QWidget()
                layout = QHBoxLayout(checkbox_widget)
                layout.addWidget(checkbox)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                table.setCellWidget(row_idx, 0, checkbox_widget)
                
                # Team data
                table.setItem(row_idx, 1, QTableWidgetItem(str(team)))
                
                # Calculate team stats
                team_players = self.df_players[self.df_players['Team'] == team]
                avg_salary = team_players['Salary'].mean()
                avg_points = team_players['Predicted_DK_Points'].mean()
                total_points = team_players['Predicted_DK_Points'].sum()
                player_count = len(team_players)
                
                # Set proper columns based on headers: ["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"]
                table.setItem(row_idx, 2, QTableWidgetItem("Active"))  # Status
                table.setItem(row_idx, 3, QTableWidgetItem("--"))  # Time (placeholder)
                table.setItem(row_idx, 4, QTableWidgetItem(f"{total_points:.1f}"))  # Proj Runs (using total points)
                table.setItem(row_idx, 5, QTableWidgetItem("0"))  # Min exposure
                table.setItem(row_idx, 6, QTableWidgetItem("100"))  # Max exposure
                table.setItem(row_idx, 7, QTableWidgetItem("0.00%"))  # Actual exposure

    def select_all(self, position):
        """Select all players in a specific position table"""
        if not hasattr(self, 'player_tables') or position not in self.player_tables:
            logging.debug(f"No table found for position: {position}")
            return
        
        table = self.player_tables[position]
        
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
        
        logging.debug(f"Selected all players in {position} table")

    def deselect_all(self, position):
        """Deselect all players in a specific position table"""
        if not hasattr(self, 'player_tables') or position not in self.player_tables:
            logging.debug(f"No table found for position: {position}")
            return
        
        table = self.player_tables[position]
        
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
        
        logging.debug(f"Deselected all players in {position} table")

    def load_entries_csv(self):
        """Load entries CSV for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Entries CSV', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_entries = pd.read_csv(file_path)
                self.status_label.setText(f'Entries loaded: {len(self.df_entries)} entries')
            except Exception as e:
                self.status_label.setText(f'Error loading entries: {str(e)}')
                logging.error(f"Error loading entries: {e}")

    def save_csv(self):
        """Save optimized lineups to CSV"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Data", "No optimized lineups to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Optimized Lineups', 'optimized_lineups.csv', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.save_lineups_to_dk_format(file_path)
                self.status_label.setText(f'Lineups saved to: {file_path}')
            except Exception as e:
                self.status_label.setText(f'Error saving: {str(e)}')
                logging.error(f"Error saving lineups: {e}")

    def save_lineups_to_dk_format(self, output_path):
        """Save lineups in DraftKings format"""
        dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(dk_positions)
            
            for lineup in self.optimized_lineups:
                dk_lineup = self.format_lineup_for_dk(lineup, dk_positions)
                writer.writerow(dk_lineup)

    def format_lineup_for_dk(self, lineup, dk_positions):
        """Group players by position and assign to DraftKings positions"""
        position_players = {
            'P': [],
            'C': [],
            '1B': [],
            '2B': [],
            '3B': [],
            'SS': [],
            'OF': []
        }
        
        for _, player in lineup.iterrows():
            pos = str(player['Pos']).upper()
            name = str(player['Name'])
            
            # Handle pitcher designations
            if 'P' in pos or 'SP' in pos or 'RP' in pos:
                position_players['P'].append(name)
            elif 'C' in pos:
                position_players['C'].append(name)
            elif '1B' in pos:
                position_players['1B'].append(name)
            elif '2B' in pos:
                position_players['2B'].append(name)
            elif '3B' in pos:
                position_players['3B'].append(name)
            elif 'SS' in pos:
                position_players['SS'].append(name)
            elif 'OF' in pos:
                position_players['OF'].append(name)
        
        # Assign players to DK positions
        dk_lineup = []
        position_usage = {pos: 0 for pos in position_players.keys()}
        
        for dk_pos in dk_positions:
            if dk_pos in position_players and position_usage[dk_pos] < len(position_players[dk_pos]):
                dk_lineup.append(position_players[dk_pos][position_usage[dk_pos]])
                position_usage[dk_pos] += 1
            else:
                # Find any remaining player that can fill this position
                assigned = False
                for pos, players in position_players.items():
                    if position_usage[pos] < len(players):
                        dk_lineup.append(players[position_usage[pos]])
                        position_usage[pos] += 1
                        assigned = True
                        break
                if not assigned:
                    dk_lineup.append("")

        return dk_lineup

    def load_dk_predictions(self):
        """Load DraftKings predictions from CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open DraftKings Predictions CSV', '', 'CSV Files (*.csv);;All Files (*)'
        )
        if file_path:
            try:
                self.df_players = self.load_players(file_path)
                self.populate_player_tables()
                self.populate_team_stack_table()
                self.status_label.setText(f'DraftKings predictions loaded: {len(self.df_players)} players')
            except Exception as e:
                self.status_label.setText(f'Error loading DraftKings predictions: {str(e)}')
                logging.error(f"Error loading DraftKings predictions: {e}")

    def load_dk_entries_file(self):
        """Load a DraftKings entries file to be filled with optimized lineups"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load DraftKings Entries File', '', 'CSV Files (*.csv);;All Files (*)'
        )
        
        if not file_path:
            return
        
        try:
            # Try multiple approaches to read the CSV file with better error handling
            self.dk_entries_df = None
            read_success = False
            error_details = []
            
            # Method 1: Try standard pandas read with error handling for bad lines
            try:
                self.dk_entries_df = pd.read_csv(file_path, on_bad_lines='skip')
                read_success = True
                logging.info("Successfully read CSV with standard method (skipping bad lines)")
            except Exception as e1:
                error_details.append(f"Standard read with skip bad lines: {str(e1)}")
                logging.debug(f"Standard read failed: {e1}")
            
            # Method 2: Try with different separators
            if not read_success:
                for separator in [',', ';', '\t']:
                    try:
                        self.dk_entries_df = pd.read_csv(file_path, sep=separator, on_bad_lines='skip')
                        read_success = True
                        logging.info(f"Successfully read CSV with separator '{separator}'")
                        break
                    except Exception as e2:
                        error_details.append(f"Separator '{separator}': {str(e2)}")
                        continue
            
            # Method 3: Try reading with no header and detect format
            if not read_success:
                try:
                    # Read first few lines to understand structure
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[:10]
                    
                    # Look for a line that might be the header
                    header_line_idx = 0
                    for i, line in enumerate(lines):
                        if any(pos in line.upper() for pos in ['P', 'C', '1B', '2B', 'SS', 'OF', 'ENTRY']):
                            header_line_idx = i
                            break
                    
                    # Try reading from the detected header line
                    self.dk_entries_df = pd.read_csv(file_path, skiprows=header_line_idx, on_bad_lines='skip')
                    read_success = True
                    logging.info(f"Successfully read CSV starting from line {header_line_idx}")
                except Exception as e3:
                    error_details.append(f"Header detection: {str(e3)}")
            
            # Method 4: Manual parsing with flexible field handling
            if not read_success:
                try:
                    import csv
                    data_rows = []
                    header_row = None
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Try to detect delimiter
                        sample = f.read(1024)
                        f.seek(0)
                        sniffer = csv.Sniffer()
                        try:
                            delimiter = sniffer.sniff(sample).delimiter
                        except:
                            delimiter = ','
                        
                        reader = csv.reader(f, delimiter=delimiter)
                        for i, row in enumerate(reader):
                            # Skip empty rows
                            if not any(cell.strip() for cell in row if cell):
                                continue
                            
                            # Look for header row
                            if header_row is None and any(pos in str(row).upper() for pos in ['P', 'C', '1B', '2B', 'SS', 'OF', 'ENTRY']):
                                header_row = [cell.strip() for cell in row if cell.strip()]
                                continue
                            
                            # Process data rows
                            if header_row and len(row) > 0:
                                # Trim or pad row to match header length
                                processed_row = []
                                for j in range(len(header_row)):
                                    if j < len(row):
                                        processed_row.append(row[j].strip())
                                    else:
                                        processed_row.append('')
                                data_rows.append(processed_row)
                    
                    if header_row and data_rows:
                        self.dk_entries_df = pd.DataFrame(data_rows, columns=header_row)
                        read_success = True
                        logging.info("Successfully parsed CSV manually")
                    else:
                        raise ValueError("Could not detect proper header or data rows")
                        
                except Exception as e4:
                    error_details.append(f"Manual parsing: {str(e4)}")
            
            if not read_success:
                raise ValueError(f"Could not read CSV file after trying multiple methods:\n" + "\n".join(error_details))
            
            # Clean up the DataFrame
            if self.dk_entries_df is not None:
                # Remove completely empty rows
                self.dk_entries_df = self.dk_entries_df.dropna(how='all')
                
                # Clean column names
                self.dk_entries_df.columns = [str(col).strip() for col in self.dk_entries_df.columns]
                
                # Remove unnamed/empty columns
                cols_to_drop = []
                for col in self.dk_entries_df.columns:
                    if 'Unnamed' in str(col) or str(col).strip() == '':
                        if self.dk_entries_df[col].isna().all() or (self.dk_entries_df[col] == '').all():
                            cols_to_drop.append(col)
                
                if cols_to_drop:
                    self.dk_entries_df = self.dk_entries_df.drop(columns=cols_to_drop)
                    logging.info(f"Removed {len(cols_to_drop)} empty columns")
            
            # Store the original file path for saving back
            self.dk_entries_file_path = file_path
            
            # Detect format
            self.dk_entries_format = self.detect_dk_format()
            
            # Show success message
            num_entries = len(self.dk_entries_df)
            file_columns = list(self.dk_entries_df.columns)
            
            success_msg = f"✅ DraftKings entries file loaded successfully!\n\n"
            success_msg += f"📁 File: {os.path.basename(file_path)}\n"
            success_msg += f"📊 Number of entries: {num_entries}\n"
            success_msg += f"📋 Columns ({len(file_columns)}): {', '.join(file_columns[:5])}{'...' if len(file_columns) > 5 else ''}\n"
            success_msg += f"🎯 Format detected: {self.dk_entries_format}"
            
            if num_entries == 0:
                success_msg += "\n\n⚠️ File appears to be empty - will create new entries when you fill it."
            
            QMessageBox.information(self, "DraftKings Entries Loaded", success_msg)
            self.status_label.setText(f'DK Entries loaded: {num_entries} entries from {os.path.basename(file_path)}')
            
        except Exception as e:
            error_msg = f"Error loading DraftKings entries file: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            
            # Show detailed error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Load Error")
            error_text = f"Failed to load DraftKings entries file:\n\n{str(e)}\n\n"
            error_text += "Common solutions:\n"
            error_text += "• Check that the file is a valid CSV format\n"
            error_text += "• Ensure all rows have consistent column counts\n"
            error_text += "• Try opening the file in Excel and re-saving as CSV\n"
            error_text += "• Remove any extra commas or special characters\n"
            error_text += "• Make sure the file contains proper DraftKings headers"
            error_dialog.setText(error_text)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()

    def detect_dk_format(self):
        """Detect the DraftKings file format"""
        file_columns = list(self.dk_entries_df.columns)
        expected_positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
        
        # Check for contest format (Entry ID, Contest Name, etc.)
        if 'Entry ID' in file_columns and 'Contest Name' in file_columns:
            return 'contest_format'
        
        # Check if columns match DK position format exactly (including duplicate positions)
        dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        # Count positions in file columns
        position_counts = {}
        legacy_position_counts = {}
        
        for col in file_columns:
            col_upper = str(col).upper().strip()
            # Standard format counting
            if col_upper in ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']:
                position_counts[col_upper] = position_counts.get(col_upper, 0) + 1
            # Legacy format counting (P.1, OF.1, OF.2)
            elif col_upper in ['P.1', 'P 1', 'P1']:
                legacy_position_counts['P'] = legacy_position_counts.get('P', 0) + 1
            elif col_upper in ['OF.1', 'OF.1', 'OF 1', 'OF1']:
                legacy_position_counts['OF'] = legacy_position_counts.get('OF', 0) + 1
            elif col_upper in ['OF.2', 'OF 2', 'OF2']:
                legacy_position_counts['OF'] = legacy_position_counts.get('OF', 0) + 1
        
        # Add base positions to legacy count if they exist
        for col in file_columns:
            col_upper = str(col).upper().strip()
            if col_upper == 'P' and 'P' not in legacy_position_counts:
                legacy_position_counts['P'] = 1
            elif col_upper == 'OF' and 'OF' not in legacy_position_counts:
                legacy_position_counts['OF'] = 1
            elif col_upper in ['C', '1B', '2B', '3B', 'SS']:
                legacy_position_counts[col_upper] = 1
        
        # Expected position counts for standard DK format
        expected_counts = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
        
        # Check if position counts match expected DK format
        if position_counts == expected_counts:
            return 'standard'
        elif legacy_position_counts.get('P', 0) >= 1 and legacy_position_counts.get('OF', 0) >= 1:
            return 'legacy_standard'
        
        # Check if it has position columns in any order
        elif all(pos in file_columns for pos in expected_positions):
            return 'flexible'
        
        # Check if it has at least some position-like columns
        elif any(pos in ' '.join(file_columns).upper() for pos in expected_positions):
            return 'custom'
        
        return 'unknown'

    def fill_dk_entries_dynamic(self):
        """Fill the loaded DraftKings entries file with optimized lineups"""
        # Check if we have optimized lineups
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            QMessageBox.warning(self, "No Lineups Available", "No optimized lineups available.\n\nPlease run the optimization first to generate lineups.")
            return
        
        # Check if we have a loaded entries file
        if not hasattr(self, 'dk_entries_df') or not hasattr(self, 'dk_entries_file_path'):
            QMessageBox.warning(self, "No Entries File Loaded", "No DraftKings entries file loaded.\n\nPlease load a DraftKings entries file first using 'Load DraftKings Entries File' button.")
            return
        
        try:
            # Ask user how many lineups to use
            num_available = len(self.optimized_lineups)
            
            # Check how many reserved entry IDs are available
            reserved_entries_available = 0
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None:
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip().isdigit():
                        reserved_entries_available += 1
            
            max_possible = min(reserved_entries_available, 1000)  # Reasonable limit
            default_request = min(num_available, reserved_entries_available, 150)
            
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Number of Lineups', 
                f'How many lineups to fill?\n\n'
                f'• Optimized lineups available: {num_available}\n'
                f'• Reserved entry IDs available: {reserved_entries_available}\n'
                f'• Recommended: {default_request}\n\n'
                f'Note: If you request more than {num_available} lineups,\n'
                f'the system will create variations of the optimized lineups.', 
                value=default_request,
                min=1, 
                max=max_possible
            )
            
            if not ok:
                return
            
            # CRITICAL FIX: Clear cached favorites data to force fresh generation
            if hasattr(self, '_favorites_with_player_data'):
                delattr(self, '_favorites_with_player_data')
                logging.info("🔄 Cleared cached favorites data to force fresh generation")
                print("🔄 CLEARED CACHED FAVORITES DATA - will use fresh optimized lineups")
            
            # Clear cached contest info to force fresh extraction
            if hasattr(self, '_contest_info_list'):
                delattr(self, '_contest_info_list')
                logging.info("🔄 Cleared cached contest info to force fresh extraction")
            
            # FORCE FRESH LINEUP GENERATION: Temporarily disable favorites file reading
            original_favorites_file_path = None
            backup_favorites_file_path = None
            
            try:
                # Temporarily rename the favorites file so it can't be found
                workspace_favorites_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
                onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
                
                if os.path.exists(workspace_favorites_path):
                    original_favorites_file_path = workspace_favorites_path
                    backup_favorites_file_path = workspace_favorites_path + ".backup_temp"
                    os.rename(workspace_favorites_path, backup_favorites_file_path)
                    logging.info(f"🔄 Temporarily renamed favorites file to force fresh generation")
                    print("🔄 TEMPORARILY RENAMED FAVORITES FILE - forcing fresh generation")
                elif os.path.exists(onedrive_favorites_path):
                    original_favorites_file_path = onedrive_favorites_path
                    backup_favorites_file_path = onedrive_favorites_path + ".backup_temp"
                    os.rename(onedrive_favorites_path, backup_favorites_file_path)
                    logging.info(f"🔄 Temporarily renamed OneDrive favorites file to force fresh generation")
                    print("🔄 TEMPORARILY RENAMED ONEDRIVE FAVORITES FILE - forcing fresh generation")
                
                # Create the filled entries DataFrame
                filled_entries = self.create_filled_entries_df(num_to_use)
                
            finally:
                # Restore the temporarily renamed favorites file
                if original_favorites_file_path and backup_favorites_file_path:
                    try:
                        if os.path.exists(backup_favorites_file_path):
                            os.rename(backup_favorites_file_path, original_favorites_file_path)
                            logging.info("🔄 Restored favorites file after filling entries")
                            print("🔄 RESTORED FAVORITES FILE")
                    except Exception as e:
                        logging.error(f"Error restoring favorites file: {e}")
            
            # Ask where to save
            save_path, _ = QFileDialog.getSaveFileName(
                self, 
                'Save Filled DraftKings Entries', 
                self.dk_entries_file_path.replace('.csv', '_filled.csv'),
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if not save_path:
                return
            
            # Save the filled entries
            filled_entries.to_csv(save_path, index=False)
            
            # Show success message with feedback if fewer than requested
            success_msg = f"🎉 DraftKings entries filled successfully!\n\n"
            success_msg += f"📊 Filled {num_to_use} entries\n"
            success_msg += f"💾 Saved to: {os.path.basename(save_path)}\n\n"
            success_msg += f"🚀 Ready to upload to DraftKings!"
            
            if num_to_use < num_available:
                success_msg += f"\n\n⚠️ Note: Used {num_to_use} lineups (available {num_available})"
            
            QMessageBox.information(self, "Entries Filled Successfully", success_msg)
            self.status_label.setText(f'Filled {num_to_use} entries and saved to {os.path.basename(save_path)}')
            
        except Exception as e:
            error_msg = f"Error filling entries: {str(e)}"
            self.status_label.setText(error_msg)
            logging.error(error_msg)
            QMessageBox.critical(self, "Fill Error", f"Failed to fill entries:\n\n{str(e)}")

    def create_filled_entries_df(self, num_to_use):
        """Create a filled entries DataFrame by combining the original entries structure with optimized lineups"""
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            raise ValueError("No optimized lineups available")
        
        # Extract contest information from favorites (this populates self._contest_info_list)
        self.extract_contest_info_from_favorites()
        
        # CRITICAL: Initialize player ID mapping early so it's available in all code paths
        player_name_to_id_map = {}
        if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None:
            player_name_to_id_map = self.extract_player_id_mapping_from_dk_file()
            logging.info(f"🎯 Extracted {len(player_name_to_id_map)} player mappings from DK entries file")
        
        # If no DK entries file or no ID mapping, create IDs from loaded player data
        if not player_name_to_id_map:
            player_name_to_id_map = self.create_player_id_mapping_from_loaded_data()
            logging.info(f"📋 Created {len(player_name_to_id_map)} player mappings from loaded data")
        
        # Define the correct DraftKings headers
        correct_headers = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        
        # CHECK FOR CACHE-CLEARING MODE: If we're in cache-clearing mode, use fresh optimized lineups
        workspace_backup_exists = os.path.exists(r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv.backup_temp")
        onedrive_backup_exists = os.path.exists(r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv.backup_temp")
        use_fresh_lineups = workspace_backup_exists or onedrive_backup_exists
        
        if use_fresh_lineups:
            logging.info("🔄 CACHE-CLEARING MODE: Using fresh optimized lineups instead of cached favorites")
            print(f"🔄 USING FRESH OPTIMIZED LINEUPS: {len(self.optimized_lineups)} fresh lineups available")
            
            # ENHANCED: Generate additional lineups if more are requested than available
            available_lineups = len(self.optimized_lineups)
            if num_to_use > available_lineups:
                logging.info(f"🔄 Requested {num_to_use} lineups, but only {available_lineups} available. Will create variations.")
                print(f"🔄 Creating variations: {num_to_use} requested vs {available_lineups} available")
                
                # Create additional lineup variations by cycling through existing ones
                extended_lineups = []
                for i in range(num_to_use):
                    base_lineup_index = i % available_lineups
                    base_lineup = self.optimized_lineups[base_lineup_index].copy()
                    extended_lineups.append(base_lineup)
                
                # Temporarily extend the optimized lineups list
                original_lineups = self.optimized_lineups
                self.optimized_lineups = extended_lineups
                lineups_to_use = num_to_use
                logging.info(f"✅ Extended lineups from {available_lineups} to {num_to_use} using variations")
            else:
                lineups_to_use = min(num_to_use, available_lineups)
            
            # Get contest info for template - try multiple sources
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                # Try to get contest info from favorites file even in cache-clearing mode (for contest template only)
                base_contest_info = None
                try:
                    # Check backup files first (they contain the real contest info)
                    workspace_fav = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
                    onedrive_fav = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
                    workspace_backup = workspace_fav + ".backup_temp"
                    onedrive_backup = onedrive_fav + ".backup_temp"
                    
                    # Try backup files first (they're the renamed originals with contest info)
                    for fav_path in [workspace_backup, onedrive_backup, workspace_fav, onedrive_fav]:
                        if os.path.exists(fav_path):
                            temp_df = pd.read_csv(fav_path)
                            if not temp_df.empty and len(temp_df.columns) >= 4:
                                base_contest_info = {
                                    'entry_id': str(temp_df.iloc[0, 0]),
                                    'contest_name': str(temp_df.iloc[0, 1]),
                                    'contest_id': str(temp_df.iloc[0, 2]),
                                    'entry_fee': str(temp_df.iloc[0, 3])
                                }
                                logging.info(f"Extracted contest template from {fav_path}: {base_contest_info['contest_name']}")
                                break
                except Exception as e:
                    logging.debug(f"Could not extract contest info from favorites: {e}")
                
                # Final fallback to default
                if not base_contest_info:
                    base_contest_info = {
                        'entry_id': '4763920000',
                        'contest_name': 'MLB Main Slate',
                        'contest_id': '999999999',
                        'entry_fee': '$1'
                    }
                    logging.info("Using default contest information for fresh lineups")
            
            # Create filled entries using FRESH optimized lineups
            filled_entries = []
            import random
            
            # Use proper base Entry ID from contest info
            if base_contest_info and base_contest_info['entry_id'].isdigit():
                base_entry_id = int(base_contest_info['entry_id'])
            else:
                base_entry_id = 4763920000  # Default base
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file")
                # Extract entry IDs from the first column of the loaded file
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        entry_id = str(row.iloc[0]).strip()
                        # Only include numeric entry IDs (skip header rows with text)
                        if entry_id.isdigit():
                            actual_entry_ids.append(entry_id)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} reserved entry IDs from DK file: {actual_entry_ids[:5]}...")
                    print(f"🎯 Using reserved entry IDs from DK file: {actual_entry_ids[:5]}... (total: {len(actual_entry_ids)})")
                else:
                    logging.warning("⚠️ No valid entry IDs found in loaded DK file")
            else:
                logging.warning("⚠️ No DK entries file loaded - cannot use reserved entry IDs")
            
            # Limit lineups to available entry IDs
            if actual_entry_ids:
                lineups_to_use = min(lineups_to_use, len(actual_entry_ids))
                logging.info(f"🎯 Will fill {lineups_to_use} reserved entries (out of {len(actual_entry_ids)} available)")
            else:
                logging.error("❌ No reserved entry IDs available - cannot export lineups")
                raise ValueError("No reserved entry IDs found in DK entries file. Please load a valid DKEntries.csv file first.")
            
            # CRITICAL FIX: Use actual reserved entry IDs from DK file (these are pre-existing entries to fill)
            for i in range(lineups_to_use):
                lineup = self.optimized_lineups[i]
                
                # Use the reserved entry ID from the loaded DK file
                unique_entry_id = actual_entry_ids[i]
                logging.debug(f"Filling reserved entry ID: {unique_entry_id} with lineup {i+1}")
                
                # Convert lineup DataFrame to player IDs with proper position mapping
                player_ids = []
                
                # Try multiple column names for player IDs
                id_columns = ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID', 'DK_ID']
                found_id_column = None
                
                for col in id_columns:
                    if col in lineup.columns:
                        found_id_column = col
                        break
                
                if found_id_column:
                    # Use the ID column directly but ensure all 10 positions are filled
                    lineup_ids = lineup[found_id_column].astype(str).tolist()
                    player_ids = []
                    
                    for pid in lineup_ids:
                        if pid and pid != 'nan' and pid.strip():
                            player_ids.append(pid.strip())
                    
                    # Ensure we have exactly 10 players
                    while len(player_ids) < 10:
                        player_ids.append("")
                    player_ids = player_ids[:10]
                    
                    logging.debug(f"Using player IDs from column '{found_id_column}': {len([p for p in player_ids if p])}/10 filled")
                else:
                    # Format positions properly using the position mapping function
                    formatted_positions = self.format_lineup_positions_only(lineup, player_name_to_id_map)
                    player_ids = formatted_positions
                    
                    # Validate that we have 10 positions and all are properly filled
                    valid_ids = [pid for pid in player_ids if pid and pid.strip() and pid != 'nan']
                    if len(valid_ids) < 8:  # Need at least 8 valid players for a reasonable lineup
                        logging.warning(f"Lineup {i+1} has only {len(valid_ids)} valid player IDs, attempting to fill gaps")
                        
                        # Try to get player IDs from player names as fallback
                        if 'Name' in lineup.columns:
                            player_names = lineup['Name'].tolist()
                            fallback_ids = []
                            
                            for name in player_names:
                                player_id_found = False
                                
                                # PRIORITY 1: Use the extracted DK entries player mapping
                                if player_name_to_id_map and name in player_name_to_id_map:
                                    fallback_ids.append(str(player_name_to_id_map[name]))
                                    player_id_found = True
                                else:
                                    # PRIORITY 2: Try to match with loaded player data if available
                                    if hasattr(self, 'df_players') and self.df_players is not None:
                                        matching_players = self.df_players[self.df_players['Name'] == name]
                                        if not matching_players.empty:
                                            # Try different ID column names in player data
                                            for id_col in ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID']:
                                                if id_col in matching_players.columns:
                                                    player_id = str(matching_players.iloc[0][id_col])
                                                    if player_id and player_id.strip() and player_id != 'nan':
                                                        fallback_ids.append(player_id.strip())
                                                        player_id_found = True
                                                        break
                                
                                # FALLBACK: Use generic ID if no mapping found
                                if not player_id_found:
                                    generic_id = str(39200000 + len(fallback_ids))
                                    fallback_ids.append(generic_id)
                            
                            # Ensure exactly 10 players
                            while len(fallback_ids) < 10:
                                fallback_ids.append("")
                            fallback_ids = fallback_ids[:10]
                            
                            player_ids = fallback_ids
                            logging.info(f"Used fallback player ID mapping for lineup {i+1}: {len([p for p in player_ids if p])}/10 filled")
                    
                    logging.debug(f"Position-mapped player IDs: {len([p for p in player_ids if p])}/10 filled")
                
                # CRITICAL VALIDATION: Ensure ALL 10 positions are filled with valid IDs
                final_player_ids = []
                for j, pid in enumerate(player_ids):
                    if pid and pid.strip() and pid != 'nan' and pid.strip() != '':
                        final_player_ids.append(pid.strip())
                    else:
                        # Generate fallback ID for empty positions
                        fallback_id = str(39200000 + (i * 10) + j)
                        final_player_ids.append(fallback_id)
                        logging.warning(f"Empty position {j+1} in lineup {i+1}, using fallback ID: {fallback_id}")
                
                # Ensure exactly 10 positions
                while len(final_player_ids) < 10:
                    fallback_id = str(39200000 + (i * 10) + len(final_player_ids))
                    final_player_ids.append(fallback_id)
                    logging.warning(f"Adding fallback ID for missing position: {fallback_id}")
                
                final_player_ids = final_player_ids[:10]  # Truncate to exactly 10
                
                # Final validation
                valid_count = len([p for p in final_player_ids if p and p.strip() and p != 'nan'])
                if valid_count < 10:
                    logging.error(f"Lineup {i+1} still has {10-valid_count} invalid positions after fallback!")
                else:
                    logging.debug(f"✅ Lineup {i+1} has all 10 positions filled with valid IDs")
                
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ] + final_player_ids
                
                filled_entries.append(row_data)
                logging.debug(f"Created fresh entry {i+1}/{lineups_to_use}: ID={unique_entry_id}, Players={len([p for p in player_ids if p])}/10")
            
            result_df = pd.DataFrame(filled_entries, columns=correct_headers)
            logging.info(f"Created {len(result_df)} entries using FRESH optimized lineups")
            return result_df
        
        # ORIGINAL LOGIC: Check if we have the favorites file with player data (only when NOT in cache-clearing mode)
        elif hasattr(self, '_favorites_with_player_data') and self._favorites_with_player_data is not None:
            logging.info("🔄 Using player data directly from favorites file")
            print(f"🔄 USING CACHED FAVORITES DATA: {len(self._favorites_with_player_data)} cached lineups")
            
            # Create filled entries DataFrame
            filled_entries = pd.DataFrame(columns=correct_headers)
            
            # Use the favorites data directly, limited to the requested number
            favorites_df = self._favorites_with_player_data
            lineups_to_use = min(num_to_use, len(favorites_df))
            
            # Get contest info - use the first one as template
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                base_contest_info = {
                    'entry_id': '1000000001',
                    'contest_name': 'Generated Lineups',
                    'contest_id': '999999999',
                    'entry_fee': '$1'
                }
                logging.warning("Using default contest information")
            
            # Generate unique Entry IDs for each lineup while preserving contest info
            import random
            base_entry_id = int(base_contest_info['entry_id']) if base_contest_info['entry_id'].isdigit() else 1000000001
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file (cached favorites path)")
                # Extract entry IDs from the first column of the loaded file
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        entry_id = str(row.iloc[0]).strip()
                        # Only include numeric entry IDs (skip header rows with text)
                        if entry_id.isdigit():
                            actual_entry_ids.append(entry_id)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} actual entry IDs for cached favorites")
            
            # CRITICAL FIX: Generate UNIQUE entry IDs to prevent duplicates
            used_entry_ids = set()
            
            for i in range(lineups_to_use):
                # Generate a unique entry ID that hasn't been used yet
                if i < len(actual_entry_ids):
                    # Try to use actual entry ID first, but ensure uniqueness
                    candidate_id = actual_entry_ids[i]
                    if candidate_id not in used_entry_ids:
                        unique_entry_id = candidate_id
                        logging.debug(f"Using actual entry ID (cached): {unique_entry_id}")
                    else:
                        # If actual ID already used, generate unique one
                        attempts = 0
                        while attempts < 1000:  # Prevent infinite loop
                            unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                            if unique_entry_id not in used_entry_ids:
                                break
                            attempts += 1
                        logging.warning(f"Actual entry ID {candidate_id} already used, generated unique ID: {unique_entry_id}")
                else:
                    # Generate a unique Entry ID for this lineup as fallback
                    attempts = 0
                    while attempts < 1000:  # Prevent infinite loop
                        unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                        if unique_entry_id not in used_entry_ids:
                            break
                        attempts += 1
                    logging.debug(f"Generated unique entry ID (cached): {unique_entry_id}")
                
                # Track this ID as used
                used_entry_ids.add(unique_entry_id)
                
                # Use the same contest info but with unique Entry ID
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ]
                
                # Extract player IDs directly from the favorites file
                favorites_row = favorites_df.iloc[i]
                player_positions = []
                
                # Extract the 10 position columns (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)
                position_start_col = 4  # After Entry ID, Contest Name, Contest ID, Entry Fee
                for pos_idx in range(10):
                    col_idx = position_start_col + pos_idx
                    if col_idx < len(favorites_row):
                        player_id = str(favorites_row.iloc[col_idx]).strip()
                        # Only include if it's a valid player ID (numeric and reasonable length)
                        if player_id and player_id != 'nan' and player_id.isdigit() and len(player_id) >= 6:
                            player_positions.append(player_id)
                        else:
                            player_positions.append("")
                    else:
                        player_positions.append("")
                
                # Add player positions to row
                row_data.extend(player_positions)
                
                # Add the row to the DataFrame
                filled_entries.loc[i] = row_data
                
                logging.debug(f"Created entry {i+1}/{lineups_to_use}: ID={unique_entry_id}, Players={len([p for p in player_positions if p])}/10")
            
            logging.info(f"Created {len(filled_entries)} entries using player data from favorites file")
            
            return filled_entries
        
        else:
            # Fallback to original logic if favorites file data is not available
            logging.warning("Favorites file with player data not available, using original logic")
            print(f"🔄 USING FRESH OPTIMIZED LINEUPS: {len(self.optimized_lineups)} current lineups")
            
            # Create filled entries DataFrame
            filled_entries = pd.DataFrame(columns=correct_headers)
            
            # Use lineups up to the requested number
            lineups_to_use = self.optimized_lineups[:num_to_use]
            
            # Get contest info - use the first one as template
            has_contest_info = hasattr(self, '_contest_info_list') and self._contest_info_list
            
            if has_contest_info:
                base_contest_info = self._contest_info_list[0]
                logging.info(f"Using contest template: {base_contest_info['contest_name']} (ID: {base_contest_info['contest_id']})")
            else:
                base_contest_info = {
                    'entry_id': '1000000001',
                    'contest_name': 'Generated Lineups',
                    'contest_id': '999999999',
                    'entry_fee': '$1'
                }
                logging.warning("Using default contest information")
            
            # Generate unique Entry IDs for each lineup while preserving contest info
            import random
            base_entry_id = int(base_contest_info['entry_id']) if base_contest_info['entry_id'].isdigit() else 1000000001
            
            # CRITICAL FIX: Extract actual entry IDs from loaded DK entries file
            actual_entry_ids = []
            if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
                logging.info("🎯 Extracting actual entry IDs from loaded DK entries file (fallback path)")
                # Extract entry IDs from the first column of the loaded file
                for _, row in self.dk_entries_df.iterrows():
                    if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip():
                        entry_id = str(row.iloc[0]).strip()
                        # Only include numeric entry IDs (skip header rows with text)
                        if entry_id.isdigit():
                            actual_entry_ids.append(entry_id)
                
                if actual_entry_ids:
                    logging.info(f"✅ Found {len(actual_entry_ids)} actual entry IDs for fallback")
            
            # CRITICAL FIX: Generate UNIQUE entry IDs to prevent duplicates (fallback path)
            used_entry_ids = set()
            
            for i, lineup in enumerate(lineups_to_use):
                # Generate a unique entry ID that hasn't been used yet
                if i < len(actual_entry_ids):
                    # Try to use actual entry ID first, but ensure uniqueness
                    candidate_id = actual_entry_ids[i]
                    if candidate_id not in used_entry_ids:
                        unique_entry_id = candidate_id
                        logging.debug(f"Using actual entry ID (fallback): {unique_entry_id}")
                    else:
                        # If actual ID already used, generate unique one
                        attempts = 0
                        while attempts < 1000:  # Prevent infinite loop
                            unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                            if unique_entry_id not in used_entry_ids:
                                break
                            attempts += 1
                        logging.warning(f"Actual entry ID {candidate_id} already used, generated unique ID: {unique_entry_id}")
                else:
                    # Generate a unique Entry ID for this lineup as fallback
                    attempts = 0
                    while attempts < 1000:  # Prevent infinite loop
                        unique_entry_id = str(base_entry_id + i + random.randint(1000, 9999))
                        if unique_entry_id not in used_entry_ids:
                            break
                        attempts += 1
                    logging.debug(f"Generated unique entry ID (fallback): {unique_entry_id}")
                
                # Track this ID as used
                used_entry_ids.add(unique_entry_id)
                
                # Use the same contest info but with unique Entry ID
                row_data = [
                    unique_entry_id,
                    base_contest_info['contest_name'],
                    base_contest_info['contest_id'],
                    base_contest_info['entry_fee']
                ]
                
                # Format lineup positions and add to row
                formatted_positions = self.format_lineup_positions_only(lineup, player_name_to_id_map)
                row_data.extend(formatted_positions)
                
                # Add the row to the DataFrame
                filled_entries.loc[i] = row_data
                
                logging.debug(f"Created entry {i+1}/{num_to_use}: ID={unique_entry_id}")
            
            logging.info(f"Created filled entries DataFrame with {len(filled_entries)} rows using lineup formatting")
            
            return filled_entries
    
    def extract_contest_info_from_favorites(self):
        """Extract contest information from loaded DK entries file or favorites data, handling multiple contests"""
        # Default values in case extraction fails
        default_info = {
            'entry_id': '1000000001',
            'contest_name': 'Generated Lineups',
            'contest_id': '999999999',
            'entry_fee': '$1'
        }
        
        # PRIORITY 1: Try to extract contest info from loaded DK entries file
        if hasattr(self, 'dk_entries_df') and self.dk_entries_df is not None and not self.dk_entries_df.empty:
            try:
                logging.info("🎯 PRIORITY: Extracting contest info from loaded DK entries file")
                print(f"🎯 Using loaded DK entries file: {getattr(self, 'dk_entries_file_path', 'Unknown')}")
                
                # Extract contest info from the loaded DK entries DataFrame
                contest_info_list = []
                for _, row in self.dk_entries_df.iterrows():
                    # DK entries format: Entry ID, Contest Name, Contest ID, Entry Fee, then player positions
                    if len(row) >= 4:
                        contest_info = {
                            'entry_id': str(row.iloc[0]) if pd.notna(row.iloc[0]) else '1000000001',
                            'contest_name': str(row.iloc[1]) if pd.notna(row.iloc[1]) else 'Generated Lineups',
                            'contest_id': str(row.iloc[2]) if pd.notna(row.iloc[2]) else '999999999',
                            'entry_fee': str(row.iloc[3]) if pd.notna(row.iloc[3]) else '$1'
                        }
                        contest_info_list.append(contest_info)
                
                if contest_info_list:
                    logging.info(f"✅ Extracted contest info for {len(contest_info_list)} entries from loaded DK file")
                    
                    # Log unique contests found in loaded file
                    unique_contests = {}
                    for info in contest_info_list:
                        contest_key = f"{info['contest_name']}|{info['contest_id']}"
                        if contest_key not in unique_contests:
                            unique_contests[contest_key] = info
                            logging.info(f"📋 Loaded contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                            print(f"📋 Using contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                    
                    # Store the contest info and the loaded DataFrame with player data
                    self._contest_info_list = contest_info_list
                    self._favorites_with_player_data = self.dk_entries_df
                    logging.info(f"✅ Stored loaded DK entries DataFrame with {len(self.dk_entries_df)} lineups and player data")
                    
                    return contest_info_list[0]  # Return first for backward compatibility
                
            except Exception as e:
                logging.warning(f"Could not extract contest info from loaded DK entries file: {e}")
                print(f"⚠️ Could not use loaded DK entries file: {e}")
        
        # PRIORITY 2: Try to extract ALL contest info from the favorites file with player data
        try:
            # First try the workspace location (which has actual player data)
            favorites_file_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\my_favorites_entries.csv"
            onedrive_favorites_path = r"c:\Users\smtes\OneDrive\Documents\my_favorites_entries.csv"
            
            # CHECK FOR CACHE-CLEARING MODE: If files are renamed (.backup_temp), read from backup files
            workspace_backup_exists = os.path.exists(favorites_file_path + ".backup_temp")
            onedrive_backup_exists = os.path.exists(onedrive_favorites_path + ".backup_temp")
            
            if workspace_backup_exists or onedrive_backup_exists:
                logging.info("🚫 CACHE-CLEARING MODE: Reading contest info from backup files only")
                print("🚫 CACHE-CLEARING MODE: Using backup files for contest info but not caching player data")
                
                # In cache-clearing mode, read contest info from backup files
                backup_files = []
                if workspace_backup_exists:
                    backup_files.append(favorites_file_path + ".backup_temp")
                if onedrive_backup_exists:
                    backup_files.append(onedrive_favorites_path + ".backup_temp")
                
                for backup_file in backup_files:
                    try:
                        favorites_df = pd.read_csv(backup_file)
                        logging.info(f"Reading contest info from backup: {backup_file}")
                        
                        if not favorites_df.empty and len(favorites_df.columns) >= 4:
                            # Extract contest info only (don't cache player data)
                            contest_info_list = []
                            for _, row in favorites_df.iterrows():
                                contest_info = {
                                    'entry_id': str(row.iloc[0]),
                                    'contest_name': str(row.iloc[1]),
                                    'contest_id': str(row.iloc[2]),
                                    'entry_fee': str(row.iloc[3])
                                }
                                contest_info_list.append(contest_info)
                            
                            logging.info(f"Extracted contest info for {len(contest_info_list)} entries from backup")
                            
                            # Log unique contests found
                            unique_contests = {}
                            for info in contest_info_list:
                                contest_key = f"{info['contest_name']}|{info['contest_id']}"
                                if contest_key not in unique_contests:
                                    unique_contests[contest_key] = info
                                    logging.info(f"Found contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                            
                            # Store contest info but NOT player data (that's the key difference)
                            self._contest_info_list = contest_info_list
                            # DO NOT store _favorites_with_player_data to force fresh lineup usage
                            
                            return contest_info_list[0]  # Return first for backward compatibility
                    except Exception as e:
                        logging.debug(f"Could not read backup file {backup_file}: {e}")
                
                # If we get here, no valid backup files found, continue with normal logic
                logging.warning("No valid backup files found, continuing with normal contest info extraction")
            
            if not os.path.exists(favorites_file_path):
                # Fallback to OneDrive location if workspace file doesn't exist
                favorites_file_path = onedrive_favorites_path
            
            if os.path.exists(favorites_file_path):
                favorites_df = pd.read_csv(favorites_file_path)
                logging.info(f"Reading favorites from: {favorites_file_path}")
                
                if not favorites_df.empty and len(favorites_df.columns) >= 4:
                    # Extract contest info from ALL rows, not just the first
                    contest_info_list = []
                    for _, row in favorites_df.iterrows():
                        contest_info = {
                            'entry_id': str(row.iloc[0]),
                            'contest_name': str(row.iloc[1]),
                            'contest_id': str(row.iloc[2]),
                            'entry_fee': str(row.iloc[3])
                        }
                        contest_info_list.append(contest_info)
                    
                    logging.info(f"Extracted contest info for {len(contest_info_list)} favorites from multiple contests")
                    
                    # Log unique contests found
                    unique_contests = {}
                    for info in contest_info_list:
                        contest_key = f"{info['contest_name']}|{info['contest_id']}"
                        if contest_key not in unique_contests:
                            unique_contests[contest_key] = info
                            logging.info(f"Found contest: {info['contest_name']} (ID: {info['contest_id']}, Fee: {info['entry_fee']})")
                    
                    # Store the list for use in create_filled_entries_df
                    self._contest_info_list = contest_info_list
                    
                    # IMPORTANT: Also store the favorites DataFrame with player data for lineup extraction
                    self._favorites_with_player_data = favorites_df
                    logging.info(f"Stored favorites DataFrame with {len(favorites_df)} lineups and player data")
                    
                    return contest_info_list[0]  # Return first for backward compatibility
        except Exception as e:
            logging.debug(f"Could not read contest info from favorites file: {e}")
        
        # Try to extract from favorites_lineups if available
        if hasattr(self, 'favorites_lineups') and self.favorites_lineups:
            try:
                contest_info_list = []
                for favorite in self.favorites_lineups:
                    if isinstance(favorite, dict):
                        contest_info = {
                            'entry_id': favorite.get('entry_id', default_info['entry_id']),
                            'contest_name': favorite.get('contest_name', default_info['contest_name']),
                            'contest_id': favorite.get('contest_id', default_info['contest_id']),
                            'entry_fee': favorite.get('entry_fee', default_info['entry_fee'])
                        }
                        contest_info_list.append(contest_info)
                
                if contest_info_list:
                    self._contest_info_list = contest_info_list
                    return contest_info_list[0]
            except Exception as e:
                logging.debug(f"Could not extract contest info from favorites: {e}")
        
        # Return defaults if all else fails
        logging.warning("Using default contest information - could not extract from favorites")
        return default_info

    def extract_player_id_mapping_from_dk_file(self):
        """Extract player name to ID mapping from the DraftKings entries file (numeric IDs only)"""
        player_map = {}
        
        if not hasattr(self, 'dk_entries_df') or self.dk_entries_df is None:
            return player_map
        
        if not hasattr(self, 'dk_entries_file_path') or not self.dk_entries_file_path:
            logging.warning("No DK entries file path available for raw parsing")
            return self.extract_player_id_mapping_from_dk_file_pandas()
        
        # Use raw file parsing for more reliable extraction
        # DK files have inconsistent CSV structure that confuses pandas
        logging.info(f"Extracting player IDs from DK entries file: {self.dk_entries_file_path}")
        
        try:
            with open(self.dk_entries_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Look for the characteristic "Name (ID)" pattern in each line
                    # Pattern: player name followed by (8-digit number)
                    matches = re.findall(r'([A-Za-z][A-Za-z\s\.\-\']+)\s*\((\d{6,})\)', line)
                    
                    for name_part, id_part in matches:
                        name_part = name_part.strip()
                        id_part = id_part.strip()
                        
                        # Additional validation for real player names
                        if (len(name_part) > 3 and 
                            len(name_part.split()) >= 2 and 
                            len(id_part) >= 6):
                            
                            # Don't overwrite existing mappings (first occurrence wins)
                            if name_part not in player_map:
                                player_map[name_part] = id_part
                                if len(player_map) <= 10:  # Log first 10 for debugging
                                    logging.info(f"✅ Found player mapping: {name_part} -> {id_part}")
        
        except Exception as e:
            logging.warning(f"Error reading DK entries file for player mappings: {e}")
            return self.extract_player_id_mapping_from_dk_file_pandas()
        
        logging.info(f"🎯 Successfully extracted {len(player_map)} player ID mappings from DK entries file")
        if len(player_map) > 0:
            sample_mappings = list(player_map.items())[:3]
            logging.info(f"Sample mappings: {sample_mappings}")
        
        return player_map
        return player_map
    
    def extract_player_id_mapping_from_dk_file_pandas(self):
        """Fallback pandas-based extraction method"""
        player_map = {}
        
        df_cols = list(self.dk_entries_df.columns)
        logging.debug(f"DK file has {len(df_cols)} columns: {df_cols}")
        logging.debug(f"Scanning {len(self.dk_entries_df)} rows for player data...")
        
        # Scan each row looking for player data patterns  
        for index, row in self.dk_entries_df.iterrows():
            try:
                # Convert row to list to handle different column counts
                row_data = row.tolist()
                
                # Look for the "Name + ID" pattern in any cell
                for col_idx, cell_value in enumerate(row_data):
                    if pd.isna(cell_value) or cell_value == '':
                        continue
                    
                    cell_str = str(cell_value).strip()
                    
                    # Check if this cell contains "Name + ID" format: "Player Name (ID_NUMBER)"
                    if '(' in cell_str and ')' in cell_str and cell_str.endswith(')'):
                        try:
                            name_part = cell_str.split('(')[0].strip()
                            id_part = cell_str.split('(')[1].replace(')', '').strip()
                            
                            # Validate that we have a real player name and numeric ID
                            if (name_part and len(name_part) > 1 and 
                                id_part.isdigit() and len(id_part) >= 6):
                                
                                # Additional validation: make sure this looks like a real player name
                                if any(char.isalpha() for char in name_part) and len(name_part.split()) >= 2:
                                    if name_part not in player_map:
                                        player_map[name_part] = id_part
                                        logging.debug(f"Found player mapping: {name_part} -> {id_part}")
                        except Exception as e:
                            continue
                
            except Exception as e:
                continue
        
        logging.info(f"Extracted {len(player_map)} player ID mappings using pandas fallback")
        return player_map

    def create_player_id_mapping_from_loaded_data(self):
        """Create player ID mapping from the loaded player data CSV (numeric IDs only)"""
        player_map = {}
        
        if not hasattr(self, 'df_players') or self.df_players is None or self.df_players.empty:
            return player_map
        
        # Look for ID columns in the loaded player data
        id_columns = []
        for col in self.df_players.columns:
            if any(id_term in str(col).lower() for id_term in ['id', 'player_id', 'dk_id', 'draftkings_id']):
                id_columns.append(col)
        
        if not id_columns:
            logging.debug("No ID columns found in loaded player data")
            return player_map
        
        # Create mappings using the first valid ID column
        for _, player in self.df_players.iterrows():
            name = str(player['Name']).strip()
            
            for id_col in id_columns:
                if pd.notna(player[id_col]):
                    player_id = str(player[id_col]).strip()
                    if player_id.isdigit() and len(player_id) >= 6:
                        player_map[name] = player_id  # Store just the numeric ID
                        logging.debug(f"Created player mapping from loaded data: {name} -> {player_id}")
                        break
        
        logging.info(f"Created {len(player_map)} player ID mappings from loaded data")
        return player_map

    def format_lineup_positions_only(self, lineup, player_name_to_id_map):
        """Format a lineup to return only the position assignments with player IDs in DK format (P, P, C, 1B, 2B, 3B, SS, OF, OF, OF)"""
        # Create position mapping from lineup
        position_players = {'P': [], 'C': [], '1B': [], '2B': [], '3B': [], 'SS': [], 'OF': []}
        
        # Group players by position with numeric IDs only
        for _, player in lineup.iterrows():
            pos = str(player['Pos']).upper()
            name = str(player['Name'])
            
            # Get the numeric ID for this player
            player_id = ""
            
            # PRIORITY 1: Use DK entries mapping
            if player_name_to_id_map and name in player_name_to_id_map:
                player_id = str(player_name_to_id_map[name])
            else:
                # PRIORITY 2: Try to get ID from player data columns
                for id_col in ['ID', 'player_id', 'Player_ID', 'PlayerID', 'DraftKingsID', 'DK_ID']:
                    if id_col in player and pd.notna(player[id_col]):
                        potential_id = str(player[id_col]).strip()
                        if potential_id.isdigit() and len(potential_id) >= 6:
                            player_id = potential_id
                            break
                
                # PRIORITY 3: Check if the name already contains an ID
                if not player_id and '(' in name and ')' in name and name.endswith(')'):
                    try:
                        id_part = name.split('(')[1].replace(')', '').strip()
                        if id_part.isdigit() and len(id_part) >= 6:
                            player_id = id_part
                    except:
                        pass
                
                # PRIORITY 4: Try any column that might contain an ID
                if not player_id:
                    for col_name, col_value in player.items():
                        if 'id' in str(col_name).lower() and pd.notna(col_value):
                            potential_id = str(col_value).strip()
                            if potential_id.isdigit() and len(potential_id) >= 6:
                                player_id = potential_id
                                break
            
            # Only add if we found a valid player ID
            if not player_id:
                # Generate a fallback ID as last resort
                player_id = str(39200000 + len([p for sublist in position_players.values() for p in sublist]))
                logging.warning(f"No valid ID found for {name}, using fallback ID: {player_id}")
            
            # Handle multi-position players and pitcher designations
            if 'P' in pos or 'SP' in pos or 'RP' in pos:
                position_players['P'].append(player_id)
            elif 'C' in pos:
                position_players['C'].append(player_id)
            elif '1B' in pos:
                position_players['1B'].append(player_id)
            elif '2B' in pos:
                position_players['2B'].append(player_id)
            elif '3B' in pos:
                position_players['3B'].append(player_id)
            elif 'SS' in pos:
                position_players['SS'].append(player_id)
            elif 'OF' in pos:
                position_players['OF'].append(player_id)
            else:
                # If position is unclear, try to guess based on common patterns
                if any(p_term in pos for p_term in ['P', 'PITCH']):
                    position_players['P'].append(player_id)
                elif any(of_term in pos for of_term in ['OF', 'LF', 'CF', 'RF', 'OUTFIELD']):
                    position_players['OF'].append(player_id)
                else:
                    # Default to OF if we can't determine position
                    position_players['OF'].append(player_id)
                    logging.warning(f"Unclear position '{pos}' for {name}, defaulting to OF")
        
        # Create the position assignments in DK format: [P, P, C, 1B, 2B, 3B, SS, OF, OF, OF]
        position_assignments = []
        
        # Add two pitchers (ensure we have at least 2, use same pitcher twice if needed)
        if len(position_players['P']) >= 2:
            position_assignments.append(position_players['P'][0])
            position_assignments.append(position_players['P'][1])
        elif len(position_players['P']) == 1:
            position_assignments.append(position_players['P'][0])
            position_assignments.append(position_players['P'][0])  # Use same pitcher twice
            logging.warning("Only one pitcher found, using same pitcher for both P slots")
        else:
            position_assignments.append("")  # Empty P slot
            position_assignments.append("")  # Empty P slot
            logging.error("No pitchers found in lineup!")
        
        # Add catcher
        position_assignments.append(position_players['C'][0] if len(position_players['C']) > 0 else "")
        
        # Add infielders
        position_assignments.append(position_players['1B'][0] if len(position_players['1B']) > 0 else "")
        position_assignments.append(position_players['2B'][0] if len(position_players['2B']) > 0 else "")
        position_assignments.append(position_players['3B'][0] if len(position_players['3B']) > 0 else "")
        position_assignments.append(position_players['SS'][0] if len(position_players['SS']) > 0 else "")
        
        # Add three outfielders (ensure we have at least 3, reuse if needed)
        if len(position_players['OF']) >= 3:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][1])
            position_assignments.append(position_players['OF'][2])
        elif len(position_players['OF']) == 2:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][1])
            position_assignments.append(position_players['OF'][0])  # Reuse first OF
            logging.warning("Only 2 outfielders found, reusing first OF for third slot")
        elif len(position_players['OF']) == 1:
            position_assignments.append(position_players['OF'][0])
            position_assignments.append(position_players['OF'][0])  # Reuse
            position_assignments.append(position_players['OF'][0])  # Reuse
            logging.warning("Only 1 outfielder found, using same OF for all three slots")
        else:
            position_assignments.append("")  # Empty OF slots
            position_assignments.append("")
            position_assignments.append("")
            logging.error("No outfielders found in lineup!")
        
        # Final validation - ensure exactly 10 positions
        while len(position_assignments) < 10:
            position_assignments.append("")
        position_assignments = position_assignments[:10]
        
        # CRITICAL FIX: Fill any empty positions with fallback IDs
        for i in range(10):
            if not position_assignments[i] or position_assignments[i].strip() == '':
                # Generate fallback ID for empty position
                fallback_id = str(39200000 + i)
                position_assignments[i] = fallback_id
                logging.warning(f"Position {i+1} was empty, filled with fallback ID: {fallback_id}")
        
        # Count how many positions are actually filled
        filled_count = len([p for p in position_assignments if p and p.strip()])
        logging.debug(f"Position assignment result: {filled_count}/10 positions filled")
        
        return position_assignments
    
    def get_bankroll_setting(self):
        """Get the bankroll setting from UI"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'bankroll_input'):
                bankroll_text = self.bankroll_input.text().strip()
                if bankroll_text:
                    bankroll = float(bankroll_text)
                    return max(100, bankroll)  # Minimum $100
            return 1000  # Default
        except (ValueError, AttributeError):
            return 1000

    def get_risk_tolerance_setting(self):
        """Get the risk tolerance setting from UI"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'risk_tolerance_combo'):
                return self.risk_tolerance_combo.currentText()
            return 'medium'
        except AttributeError:
            return 'medium'

    def get_risk_management_enabled(self):
        """Check if risk management is enabled"""
        try:
            if RISK_ENGINE_AVAILABLE and hasattr(self, 'enable_risk_mgmt_checkbox'):
                return self.enable_risk_mgmt_checkbox.isChecked()
            return False
        except AttributeError:
            return False

    def update_selection_status(self):
        """Update the selection status label with current player count"""
        try:
            if not hasattr(self, 'selection_status_label'):
                return
                
            # Get total selected count using the same method as get_included_players
            included_players = self.get_included_players()  
            total_selected = len(included_players)
            
            # Update the status label with simple, non-cluttering text
            if total_selected == 0:
                status_text = "No players selected"
                self.selection_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
            elif total_selected <= 50:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #4CAF50; font-size: 10px; padding: 3px;")
            elif total_selected <= 150:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #FF9800; font-size: 10px; padding: 3px;")
            else:
                status_text = f"{total_selected} players selected"
                self.selection_status_label.setStyleSheet("color: #f44336; font-size: 10px; padding: 3px;")
            
            self.selection_status_label.setText(status_text)
            
        except Exception as e:
            logging.error(f"Error updating selection status: {e}")
if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = FantasyBaseballApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())