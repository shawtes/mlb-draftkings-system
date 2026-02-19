# import sys
# import logging
# import traceback
# import psutil
# import pulp
# import pandas as pd
# import numpy as np
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
# from concurrent.futures import ProcessPoolExecutor
# import multiprocessing
# import concurrent.futures
# from itertools import combinations
# import csv
# from collections import defaultdict

# # ... existing imports ...

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# SALARY_CAP = 50000
# POSITION_LIMITS = {
#     'P': 2,
#     'C': 1,
#     '1B': 1,
#     '2B': 1,
#     '3B': 1,
#     'SS': 1,
#     'OF': 3
# }
# REQUIRED_TEAM_SIZE = 10

# def optimize_single_lineup(args):
#     df, stack_type, team_projected_runs, team_selections = args
#     logging.debug(f"optimize_single_lineup: Starting with stack type {stack_type}")
    
#     problem = pulp.LpProblem("Stack_Optimization", pulp.LpMaximize)
#     player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}

#     # Calculate risk-adjusted projections using probability data
#     df_copy = df.copy()
#       # Check if probability columns exist and calculate upside factor
#     if 'Prob_Over_25' in df_copy.columns:
#         # Convert percentage strings to floats (e.g., "15.5%" -> 0.155)
#         prob_25_clean = pd.to_numeric(df_copy['Prob_Over_25'].astype(str).str.replace('%', ''), errors='coerce') / 100
#         prob_30_clean = pd.to_numeric(df_copy.get('Prob_Over_30', pd.Series([0]*len(df_copy))).astype(str).str.replace('%', ''), errors='coerce') / 100
        
#         # Handle NaN values
#         prob_25_clean = prob_25_clean.fillna(0)
#         prob_30_clean = prob_30_clean.fillna(0)
        
#         # Create upside factor: weight high ceiling potential
#         upside_factor = prob_25_clean * 0.1 + prob_30_clean * 0.2  # 10% + 20% weighting
        
#         # Apply risk adjustment to projections
#         df_copy['Risk_Adjusted_Points'] = df_copy['Predicted_DK_Points'] * (1 + upside_factor)
        
#         logging.debug(f"Applied probability-based risk adjustment. Sample adjustments:")
#         for idx in df_copy.head(3).index:
#             original = df_copy.at[idx, 'Predicted_DK_Points']
#             adjusted = df_copy.at[idx, 'Risk_Adjusted_Points']
#             logging.debug(f"  {df_copy.at[idx, 'Name']}: {original:.2f} -> {adjusted:.2f}")
#     else:
#         # Fallback: use original projections if no probability data
#         df_copy['Risk_Adjusted_Points'] = df_copy['Predicted_DK_Points']
#         logging.debug("No probability data found - using original projections")

#     # Objective: Maximize risk-adjusted projected points
#     problem += pulp.lpSum([df_copy.at[idx, 'Risk_Adjusted_Points'] * player_vars[idx] for idx in df_copy.index])    # Basic constraints
#     problem += pulp.lpSum(player_vars.values()) == REQUIRED_TEAM_SIZE
#     problem += pulp.lpSum([df_copy.at[idx, 'Salary'] * player_vars[idx] for idx in df_copy.index]) <= SALARY_CAP
#     for position, limit in POSITION_LIMITS.items():
#         problem += pulp.lpSum([player_vars[idx] for idx in df_copy.index if position in df_copy.at[idx, 'Pos']]) == limit

#     # Implement stacking
#     stack_sizes = [int(size) for size in stack_type.split('|')]
#     total_stack_size = sum(stack_sizes)
#     non_stack_size = REQUIRED_TEAM_SIZE - total_stack_size

#     for i, size in enumerate(stack_sizes):
#         team_vars = {team: pulp.LpVariable(f"team_{team}_{i}", cat='Binary') for team in team_selections[size]}
#         problem += pulp.lpSum(team_vars.values()) == 1
        
#         for team in team_selections[size]:
#             team_players = df_copy[(df_copy['Team'] == team) & (~df_copy['Pos'].str.contains('P'))].index
#             problem += pulp.lpSum([player_vars[idx] for idx in team_players]) >= size * team_vars[team]

#     # Ensure the correct number of non-stack players
#     problem += pulp.lpSum([player_vars[idx] for idx in df_copy.index if 'P' in df_copy.at[idx, 'Pos']]) + \
#                pulp.lpSum([player_vars[idx] for idx in df_copy.index if 'P' not in df_copy.at[idx, 'Pos']]) - \
#                pulp.lpSum([size * pulp.lpSum(team_vars.values()) for size in stack_sizes]) == non_stack_size

#     # Solve the problem
#     solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
#     status = problem.solve(solver)

#     if pulp.LpStatus[status] == 'Optimal':
#         lineup = df_copy.loc[[idx for idx in df_copy.index if player_vars[idx].varValue > 0.5]]
#         logging.debug(f"optimize_single_lineup: Found optimal solution with {len(lineup)} players")
#         return lineup, stack_type
#     else:
#         logging.debug(f"optimize_single_lineup: No optimal solution found. Status: {pulp.LpStatus[status]}")
#         logging.debug(f"Constraints: {problem.constraints}")
#         return pd.DataFrame(), stack_type
# def simulate_iteration(df):
#     random_factors = np.random.normal(1, 0.1, size=len(df))
#     df = df.copy()
#     df['Predicted_DK_Points'] = df['Predicted_DK_Points'] * random_factors
#     df['Predicted_DK_Points'] = df['Predicted_DK_Points'].clip(lower=1)
#     return df

# class OptimizationWorker(QThread):
#     optimization_done = pyqtSignal(dict, dict, dict)

#     def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations,num_lineups):
#         super().__init__()
#         self.df_players = df_players
#         self.num_lineups = num_lineups
#         self.salary_cap = salary_cap
#         self.position_limits = position_limits
#         self.included_players = included_players
#         self.stack_settings = stack_settings
#         self.min_exposure = min_exposure
#         self.max_exposure = max_exposure
#         self.team_projected_runs = self.calculate_team_projected_runs(df_players)
        
#         self.max_workers = multiprocessing.cpu_count()  # Or set a specific number
#         self.min_points = min_points
#         self.monte_carlo_iterations = monte_carlo_iterations
#         self.team_selections = {}  # This will be populated in preprocess_data

#     def run(self):
#         logging.debug("OptimizationWorker: Starting optimization")
#         results, team_exposure, stack_exposure = self.optimize_lineups()
#         logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
#         self.optimization_done.emit(results, team_exposure, stack_exposure)

#     def optimize_lineups(self):
#         df_filtered = self.preprocess_data()
#         logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")

#         results = {}
#         team_exposure = defaultdict(int)
#         stack_exposure = defaultdict(int)
        
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = []
#             for stack_type in self.stack_settings:
#                 for _ in range(self.num_lineups):
#                     future = executor.submit(optimize_single_lineup, (df_filtered.copy(), stack_type, self.team_projected_runs, self.team_selections))
#                     futures.append(future)

#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     lineup, stack_type = future.result()
#                     if lineup.empty:
#                         logging.debug(f"optimize_lineups: Empty lineup returned for stack type {stack_type}")
#                     else:
#                         total_points = lineup['Predicted_DK_Points'].sum()
#                         results[len(results)] = {'total_points': total_points, 'lineup': lineup}
#                         for team in lineup['Team'].unique():
#                             team_exposure[team] += 1
#                         stack_exposure[stack_type] += 1
#                         logging.debug(f"optimize_lineups: Found valid lineup for stack type {stack_type}")
#                 except Exception as e:
#                     logging.error(f"Error in optimization: {str(e)}")

#         logging.debug(f"optimize_lineups: Completed. Found {len(results)} valid lineups")
#         logging.debug(f"Team exposure: {dict(team_exposure)}")
#         logging.debug(f"Stack exposure: {dict(stack_exposure)}")
        
#         return results, team_exposure, stack_exposure
#     def preprocess_data(self):
#         logging.debug("preprocess_data: Starting")
#         df_filtered = self.df_players[self.df_players['Predicted_DK_Points'] > 0]  # Filter out players with 0 or negative projections
#         df_filtered = df_filtered[df_filtered['Salary'] > 0]  # Filter out players with 0 or negative salary
        
#         if self.included_players:
#             df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
        
#         # Create team_selections based on available teams
#         available_teams = df_filtered['Team'].unique()
#         self.team_selections = {
#             stack_size: available_teams
#             for stack_size in set(int(size) for stack in self.stack_settings for size in stack.split('|'))
#         }
        
#         logging.debug(f"preprocess_data: Filtered data shape: {df_filtered.shape}")
#         logging.debug(f"preprocess_data: Available teams: {available_teams}")
#         logging.debug(f"preprocess_data: Team selections: {self.team_selections}")
#         return df_filtered
#     def calculate_team_projected_runs(self, df):
#         return {team: self.calculate_projected_runs(group) 
#                 for team, group in df.groupby('Team')}

#     def calculate_projected_runs(self, team_players):
#         if 'Saber Total' in team_players.columns:
#             return team_players['Saber Total'].mean()
#         elif 'Predicted_DK_Points' in team_players.columns:
#             return team_players['Predicted_DK_Points'].sum() * 0.5
#         else:
#             logging.warning(f"No projection data available for team {team_players['Team'].iloc[0]}")
#             return 0

# class FantasyBaseballApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Advanced MLB DFS Optimizer")
#         self.setGeometry(100, 100, 1600, 1000)
#         self.setup_ui()
#         self.included_players = []
#         self.stack_settings = {}
#         self.min_exposure = {}
#         self.max_exposure = {}
#         self.min_points = 1
#         self.monte_carlo_iterations = 100

#     def setup_ui(self):
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.main_layout = QVBoxLayout(self.central_widget)
#         self.splitter = QSplitter(Qt.Horizontal)
#         self.main_layout.addWidget(self.splitter)

#         self.tabs = QTabWidget()
#         self.splitter.addWidget(self.tabs)

#         self.df_players = None
#         self.df_entries = None
#         self.player_exposure = {}
#         self.optimized_lineups = []

#         self.create_players_tab()
#         self.create_team_stack_tab()
#         self.create_stack_exposure_tab()
#         self.create_control_panel()

#     def create_players_tab(self):
#         players_tab = QWidget()
#         self.tabs.addTab(players_tab, "Players")

#         players_layout = QVBoxLayout(players_tab)

#         position_tabs = QTabWidget()
#         players_layout.addWidget(position_tabs)

#         self.player_tables = {}

#         positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
#         for position in positions:
#             sub_tab = QWidget()
#             position_tabs.addTab(sub_tab, position)
#             layout = QVBoxLayout(sub_tab)

#             select_all_button = QPushButton("Select All")
#             deselect_all_button = QPushButton("Deselect All")
#             select_all_button.clicked.connect(lambda _, p=position: self.select_all(p))
#             deselect_all_button.clicked.connect(lambda _, p=position: self.deselect_all(p))
#             button_layout = QHBoxLayout()
#             button_layout.addWidget(select_all_button)
#             button_layout.addWidget(deselect_all_button)
#             layout.addLayout(button_layout)

#             table = QTableWidget(0, 11)
#             table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Pos", "Salary", "Predicted_DK_Points", "Own", "Min Exp", "Max Exp", "Actual Exp (%)", "Predicted_DK_Points"])
#             layout.addWidget(table)

#             self.player_tables[position] = table

#     def create_team_stack_tab(self):
#         team_stack_tab = QWidget()
#         self.tabs.addTab(team_stack_tab, "Team Stacks")

#         layout = QVBoxLayout(team_stack_tab)

#         stack_size_tabs = QTabWidget()
#         layout.addWidget(stack_size_tabs)

#         stack_sizes = ["All Stacks", "2 Stack", "3 Stack", "4 Stack", "5 Stack"]
#         self.team_stack_tables = {}

#         for stack_size in stack_sizes:
#             sub_tab = QWidget()
#             stack_size_tabs.addTab(sub_tab, stack_size)
#             sub_layout = QVBoxLayout(sub_tab)

#             table = QTableWidget(0, 8)
#             table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
#             sub_layout.addWidget(table)

#             self.team_stack_tables[stack_size] = table

#         self.team_stack_table = self.team_stack_tables["All Stacks"]

#         refresh_button = QPushButton("Refresh Team Stacks")
#         refresh_button.clicked.connect(self.refresh_team_stacks)
#         layout.addWidget(refresh_button)

#     def refresh_team_stacks(self):
#         self.populate_team_stack_table()

#     def create_stack_exposure_tab(self):
#         stack_exposure_tab = QWidget()
#         self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
    
#         layout = QVBoxLayout(stack_exposure_tab)
    
#         self.stack_exposure_table = QTableWidget(0, 7)
#         self.stack_exposure_table.setHorizontalHeaderLabels(["Select", "Stack Type", "Min Exp", "Max Exp", "Lineup Exp", "Pool Exp", "Entry Exp"])
#         layout.addWidget(self.stack_exposure_table)
    
#         stack_types = ["4|2|2", "4|2", "3|3|2", "3|2|2", "2|2|2", "5|3", "5|2", "No Stacks"]
#         for stack_type in stack_types:
#             row_position = self.stack_exposure_table.rowCount()
#             self.stack_exposure_table.insertRow(row_position)
    
#             checkbox = QCheckBox()
#             checkbox_widget = QWidget()
#             layout_checkbox = QHBoxLayout(checkbox_widget)
#             layout_checkbox.addWidget(checkbox)
#             layout_checkbox.setAlignment(Qt.AlignCenter)
#             layout_checkbox.setContentsMargins(0, 0, 0, 0)
#             self.stack_exposure_table.setCellWidget(row_position, 0, checkbox_widget)
    
#             self.stack_exposure_table.setItem(row_position, 1, QTableWidgetItem(stack_type))
#             min_exp_item = QTableWidgetItem("0")
#             min_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
#             self.stack_exposure_table.setItem(row_position, 2, min_exp_item)
    
#             max_exp_item = QTableWidgetItem("100")
#             max_exp_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
#             self.stack_exposure_table.setItem(row_position, 3, max_exp_item)
    
#             self.stack_exposure_table.setItem(row_position, 4, QTableWidgetItem("0.0%"))
#             self.stack_exposure_table.setItem(row_position, 5, QTableWidgetItem("0.0%"))
#             self.stack_exposure_table.setItem(row_position, 6, QTableWidgetItem("0.0%"))

#     def create_control_panel(self):
#         control_panel = QFrame()
#         control_panel.setFrameShape(QFrame.StyledPanel)
#         control_layout = QVBoxLayout(control_panel)

#         self.splitter.addWidget(control_panel)

#         load_button = QPushButton('Load CSV')
#         load_button.clicked.connect(self.load_file)
#         control_layout.addWidget(load_button)

#         load_dk_predictions_button = QPushButton('Load DraftKings Predictions')
#         load_dk_predictions_button.clicked.connect(self.load_dk_predictions)
#         control_layout.addWidget(load_dk_predictions_button)

#         load_entries_button = QPushButton('Load Entries CSV')
#         load_entries_button.clicked.connect(self.load_entries_csv)
#         control_layout.addWidget(load_entries_button)

#         self.min_unique_label = QLabel('Min Unique:')
#         self.min_unique_input = QLineEdit()
#         control_layout.addWidget(self.min_unique_label)
#         control_layout.addWidget(self.min_unique_input)

#         self.sorting_label = QLabel('Sorting Method:')
#         self.sorting_combo = QComboBox()
#         self.sorting_combo.addItems(["Points", "Value", "Salary"])
#         control_layout.addWidget(self.sorting_label)
#         control_layout.addWidget(self.sorting_combo)

#         run_button = QPushButton('Run Contest Sim')
#         run_button.clicked.connect(self.run_optimization)
#         control_layout.addWidget(run_button)

#         save_button = QPushButton('Save CSV for DraftKings')
#         save_button.clicked.connect(self.save_csv)
#         control_layout.addWidget(save_button)

#         self.results_table = QTableWidget(0, 9)
#         self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Pos", "Salary", "Predicted_DK_Points", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
#         control_layout.addWidget(self.results_table)

#         self.status_label = QLabel('')
#         control_layout.addWidget(self.status_label)

#     def load_file(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
#         if file_path:
#             self.df_players = self.load_players(file_path)
#             self.populate_player_tables()

#     def load_entries_csv(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, 'Open Entries CSV', '', 'CSV Files (*.csv)')
#         if file_path:
#             self.df_entries = self.load_and_standardize_csv(file_path)
#             if self.df_entries is not None:
#                 self.status_label.setText('Entries CSV loaded and standardized successfully.')
#             else:
#                 self.status_label.setText('Failed to standardize Entries CSV.')

#     def load_players(self, csv_path):
#         # Check if this is a probability predictions file, regular predictions file, or merged file
#         df = pd.read_csv(csv_path)
        
#         print(f"Loading CSV file with columns: {list(df.columns)}")
        
#         # If it's a merged projections file (contains 'My_Proj')
#         if 'My_Proj' in df.columns:
#             print("Loading merged projections file...")
#             # Rename columns to match expected format
#             df = df.rename(columns={
#                 'My_Proj': 'Predicted_DK_Points'
#             })
            
#             # Add required columns if missing
#             required_columns = ['Name', 'Team', 'Opp', 'Pos', 'Predicted_DK_Points', 'Salary']
#             for col in required_columns:
#                 if col not in df.columns:
#                     if col == 'Opp':
#                         df[col] = 'vs UNK'  # Default opponent
#                     else:
#                         df[col] = np.nan
                        
#         # If it's a probability predictions file (contains 'Predicted_DK_Points')
#         elif 'Predicted_DK_Points' in df.columns:
#             print("Loading DraftKings probability predictions file...")
#             # Rename columns to match expected format
#             df = df.rename(columns={
#                 'Predicted_DK_Points': 'Predicted_DK_Points'
#             })
            
#             # Add required columns if missing
#             required_columns = ['Name', 'Team', 'Opp', 'Pos', 'Predicted_DK_Points', 'Salary']
#             for col in required_columns:
#                 if col not in df.columns:
#                     if col == 'Team':
#                         df[col] = 'UNK'  # Default team
#                     elif col == 'Opp':
#                         df[col] = 'vs UNK'  # Default opponent
#                     elif col == 'Pos':
#                         # Try to infer position from name or set default
#                         df[col] = 'OF'  # Default position
#                     elif col == 'Salary':
#                         # Set default salary based on projection
#                         df[col] = (df['Predicted_DK_Points'] * 200).clip(lower=2000, upper=12000).astype(int)
#                     else:
#                         df[col] = np.nan
                        
#         # If it's a regular predictions file (contains 'predicted_dk_fpts')  
#         elif 'predicted_dk_fpts' in df.columns:
#             print("Loading regular DraftKings predictions file...")
#             # Rename columns to match expected format
#             df = df.rename(columns={
#                 'predicted_dk_fpts': 'Predicted_DK_Points'
#             })
            
#             # Add required columns if missing
#             required_columns = ['Name', 'Team', 'Opp', 'Pos', 'Predicted_DK_Points', 'Salary']
#             for col in required_columns:
#                 if col not in df.columns:
#                     if col == 'Team':
#                         df[col] = 'UNK'
#                     elif col == 'Opp':
#                         df[col] = 'vs UNK'
#                     elif col == 'Pos':
#                         df[col] = 'OF'
#                     elif col == 'Salary':
#                         df[col] = (df['Predicted_DK_Points'] * 200).clip(lower=2000, upper=12000).astype(int)
#                     else:
#                         df[col] = np.nan
#         else:
#             # Regular CSV format - ensure required columns exist
#             required_columns = ['Name', 'Team', 'Opp', 'Pos', 'Predicted_DK_Points', 'Salary']
#             for col in required_columns:
#                 if col not in df.columns:
#                     df[col] = np.nan
        
#         # Clean and process data
#         df['Predicted_DK_Points'] = pd.to_numeric(df['Predicted_DK_Points'], errors='coerce')
#         df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        
#         # Handle positions - ensure they're properly formatted
#         df['Pos'] = df['Pos'].fillna('OF')  # Default to OF if position is missing
#         df['Positions'] = df['Pos'].apply(lambda x: x.split('/') if pd.notna(x) else ['OF'])
        
#         # Filter out invalid data
#         df = df.dropna(subset=['Name', 'Predicted_DK_Points', 'Salary'])
#         df = df[df['Predicted_DK_Points'] > 0]  # Remove players with 0 or negative projections
#         df = df[df['Salary'] > 0]   # Remove players with 0 or negative salary
        
#         print(f"Loaded {len(df)} players with valid predictions")
#         print(f"Projection range: {df['Predicted_DK_Points'].min():.2f} to {df['Predicted_DK_Points'].max():.2f}")
#         print(f"Sample data:\n{df[['Name', 'Team', 'Pos', 'Predicted_DK_Points', 'Salary']].head()}")
        
#         return df

#     def load_and_standardize_csv(self, file_path):
#         try:
#             df = pd.read_csv(file_path, skiprows=6, on_bad_lines='skip')
#             df.columns = ['ID', 'Name', 'Other Columns...'] + df.columns[3:].tolist()
#             return df
#         except Exception as e:
#             logging.error(f"Error loading or processing file: {e}")
#             return None

#     def populate_player_tables(self):
#         positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        
#         for position in positions:
#             table = self.player_tables[position]
#             table.setRowCount(0)

#             if self.df_players is not None:
#                 if position == "P":
#                     df_filtered = self.df_players[self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
#                 elif position == "All Batters":
#                     df_filtered = self.df_players[~self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
#                 else:
#                     df_filtered = self.df_players[self.df_players['Positions'].apply(lambda x: position in x)]
            
#                 for _, row in df_filtered.iterrows():
#                     row_position = table.rowCount()
#                     table.insertRow(row_position)

#                     checkbox = QCheckBox()
#                     checkbox_widget = QWidget()
#                     layout_checkbox = QHBoxLayout(checkbox_widget)
#                     layout_checkbox.addWidget(checkbox)
#                     layout_checkbox.setAlignment(Qt.AlignCenter)
#                     layout_checkbox.setContentsMargins(0, 0, 0, 0)
#                     table.setCellWidget(row_position, 0, checkbox_widget)
        
#                     table.setItem(row_position, 1, QTableWidgetItem(str(row['Name'])))
#                     table.setItem(row_position, 2, QTableWidgetItem(str(row['Team'])))
#                     table.setItem(row_position, 3, QTableWidgetItem(str(row['Pos'])))
#                     table.setItem(row_position, 4, QTableWidgetItem(str(row['Salary'])))
#                     table.setItem(row_position, 5, QTableWidgetItem(str(row['Predicted_DK_Points'])))
        
#                     min_exp_spinbox = QSpinBox()
#                     min_exp_spinbox.setRange(0, 100)
#                     min_exp_spinbox.setValue(0)
#                     table.setCellWidget(row_position, 7, min_exp_spinbox)
        
#                     max_exp_spinbox = QSpinBox()
#                     max_exp_spinbox.setRange(0, 100)
#                     max_exp_spinbox.setValue(100)
#                     table.setCellWidget(row_position, 8, max_exp_spinbox)
        
#                     actual_exp_label = QLabel("")
#                     table.setCellWidget(row_position, 9, actual_exp_label)

#                     if row['Name'] not in self.player_exposure:
#                         self.player_exposure[row['Name']] = 0

#         self.populate_team_stack_table()
        
#     def populate_team_stack_table(self):
#         team_runs = self.calculate_team_projected_runs()
#         selected_teams = self.get_selected_teams()

#         for stack_size, table in self.team_stack_tables.items():
#             table.setRowCount(0)
#             for team in selected_teams:
#                 self.add_team_to_stack_table(table, team, team_runs.get(team, 0))

#     def get_selected_teams(self):
#         selected_teams = set()
#         for position in self.player_tables:
#             table = self.player_tables[position]
#             for row in range(table.rowCount()):
#                 checkbox_widget = table.cellWidget(row, 0)
#                 if checkbox_widget is not None:
#                     checkbox = checkbox_widget.findChild(QCheckBox)
#                     if checkbox is not None and checkbox.isChecked():
#                         selected_teams.add(table.item(row, 2).text())
#         return selected_teams
#     def calculate_team_projected_runs(self):
#         if self.df_players is None:
#             return {}
#         return {team: self.calculate_projected_runs(group) 
#                 for team, group in self.df_players.groupby('Team')}

#     def calculate_projected_runs(self, team_group):
#         if 'Saber Total' in team_group.columns:
#             return team_group['Saber Total'].mean()
#         elif 'Predicted_DK_Points' in team_group.columns:
#             return team_group['Predicted_DK_Points'].sum() * 0.5
#         else:
#             logging.warning(f"No projection data available for team {team_group['Team'].iloc[0]}")
#             return 0


#     def add_team_to_stack_table(self, table, team, proj_runs):
#         row_position = table.rowCount()
#         table.insertRow(row_position)

#         checkbox = QCheckBox()
#         checkbox_widget = QWidget()
#         layout_checkbox = QHBoxLayout(checkbox_widget)
#         layout_checkbox.addWidget(checkbox)
#         layout_checkbox.setAlignment(Qt.AlignCenter)
#         layout_checkbox.setContentsMargins(0, 0, 0, 0)
#         table.setCellWidget(row_position, 0, checkbox_widget)

#         table.setItem(row_position, 1, QTableWidgetItem(team))
#         table.setItem(row_position, 2, QTableWidgetItem("Playing"))
#         table.setItem(row_position, 3, QTableWidgetItem("7:00 PM"))
#         table.setItem(row_position, 4, QTableWidgetItem(f"{proj_runs:.2f}"))

#         min_exp_spinbox = QSpinBox()
#         min_exp_spinbox.setRange(0, 100)
#         min_exp_spinbox.setValue(0)
#         table.setCellWidget(row_position, 5, min_exp_spinbox)

#         max_exp_spinbox = QSpinBox()
#         max_exp_spinbox.setRange(0, 100)
#         max_exp_spinbox.setValue(100)
#         table.setCellWidget(row_position, 6, max_exp_spinbox)

#         actual_exp_label = QLabel("")
#         table.setCellWidget(row_position, 7, actual_exp_label)
#     def select_all(self, position):
#             table = self.player_tables[position]
#             for row in range(table.rowCount()):
#                 checkbox_widget = table.cellWidget(row, 0)
#                 if checkbox_widget is not None:
#                     checkbox = checkbox_widget.findChild(QCheckBox)
#                     if checkbox is not None:
#                         checkbox.setChecked(True)
#             self.populate_team_stack_table()

#     def deselect_all(self, position):
#             table = self.player_tables[position]
#             for row in range(table.rowCount()):
#                 checkbox_widget = table.cellWidget(row, 0)
#                 if checkbox_widget is not None:
#                     checkbox = checkbox_widget.findChild(QCheckBox)
#                     if checkbox is not None:
#                         checkbox.setChecked(False)
#             self.populate_team_stack_table()

#     def run_optimization(self):
#         logging.debug("Starting run_optimization method")
#         if self.df_players is None or self.df_players.empty:
#             self.status_label.setText("No player data loaded. Please load a CSV first.")
#             logging.debug("No player data loaded")
#             return
        
#         logging.debug(f"df_players shape: {self.df_players.shape}")
#         logging.debug(f"df_players columns: {self.df_players.columns}")
#         logging.debug(f"df_players sample:\n{self.df_players.head()}")
        
#         self.included_players = self.get_included_players()
#         self.stack_settings = self.collect_stack_settings()
#         self.min_exposure, self.max_exposure = self.collect_exposure_settings()
        
#         logging.debug(f"Included players: {len(self.included_players)}")
#         logging.debug(f"Stack settings: {self.stack_settings}")
        
#         self.optimization_thread = OptimizationWorker(
#             df_players=self.df_players,
#             salary_cap=SALARY_CAP,
#             position_limits=POSITION_LIMITS,
#             included_players=self.included_players,
#             stack_settings=self.stack_settings,
#             min_exposure=self.min_exposure,
#             max_exposure=self.max_exposure,
#             min_points=self.min_points,
#             monte_carlo_iterations=self.monte_carlo_iterations,
#             num_lineups=100 
#         )
#         self.optimization_thread.optimization_done.connect(self.display_results)
#         logging.debug("Starting optimization thread")
#         self.optimization_thread.start()
        
#         self.status_label.setText("Running optimization... Please wait.")

#     def display_results(self, results, team_exposure, stack_exposure):
#         logging.debug(f"display_results: Received {len(results)} results")
#         self.results_table.setRowCount(0)
#         total_lineups = len(results)

#         sorted_results = sorted(results.items(), key=lambda x: x[1]['total_points'], reverse=True)

#         self.optimized_lineups = []
#         for _, lineup_data in sorted_results:
#             self.add_lineup_to_results(lineup_data, total_lineups)
#             self.optimized_lineups.append(lineup_data['lineup'])

#         self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
#         self.refresh_team_stacks()
#         self.status_label.setText(f"Optimization complete. Generated {total_lineups} lineups.")

#     def add_lineup_to_results(self, lineup_data, total_lineups):
#         total_points = lineup_data['total_points']
#         lineup = lineup_data['lineup']
#         total_salary = lineup['Salary'].sum()

#         for _, player in lineup.iterrows():
#             row_position = self.results_table.rowCount()
#             self.results_table.insertRow(row_position)
#             self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
#             self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
#             self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Pos'])))
#             self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
#             self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
#             self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
#             self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

#             player_name = player['Name']
#             if player_name in self.player_exposure:
#                 self.player_exposure[player_name] += 1
#             else:
#                 self.player_exposure[player_name] = 1

#             exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
#             self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
#             self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))

#     def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
#         if total_lineups > 0:
#             for position in self.player_tables:
#                 table = self.player_tables[position]
#                 for row in range(table.rowCount()):
#                     player_name = table.item(row, 1).text()
#                     actual_exposure = min(self.player_exposure.get(player_name, 0) / total_lineups * 100, 100)
#                     actual_exposure_label = table.cellWidget(row, 9)
#                     if isinstance(actual_exposure_label, QLabel):
#                         actual_exposure_label.setText(f"{actual_exposure:.2f}%")

#             for stack_size, table in self.team_stack_tables.items():
#                 for row in range(table.rowCount()):
#                     team_name = table.item(row, 1).text()
#                     actual_exposure = min(team_exposure.get(team_name, 0) / total_lineups * 100, 100)
#                     table.setItem(row, 7, QTableWidgetItem(f"{actual_exposure:.2f}%"))

#             for row in range(self.stack_exposure_table.rowCount()):
#                 stack_type = self.stack_exposure_table.item(row, 1).text()
#                 actual_exposure = min(stack_exposure.get(stack_type, 0) / total_lineups * 100, 100)
#                 self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{actual_exposure:.2f}%"))

#     def save_csv(self):
#         if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
#             self.status_label.setText('No optimized lineups to save. Please run optimization first.')
#             return

#         output_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")

#         if not output_path:
#             self.status_label.setText('Save operation canceled.')
#             return

#         try:
#             with open(output_path, 'w', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'])
                
#                 for lineup in self.optimized_lineups:
#                     row = []
#                     for _, player in lineup.iterrows():
#                         row.append(player['Name'])
#                     writer.writerow(row)
            
#             self.status_label.setText(f'Optimized lineups saved successfully to {output_path}')
#         except Exception as e:
#             self.status_label.setText(f'Error saving CSV: {str(e)}')

#     def generate_output(self, entries_df, players_df, output_path):
#         optimized_output = players_df[["Name", "Team", "Pos", "Salary", "Predicted_DK_Points"]]
#         optimized_output.to_csv(output_path, index=False)

#     def get_included_players(self):
#         included_players = []
#         for position in self.player_tables:
#             table = self.player_tables[position]
#             for row in range(table.rowCount()):
#                 checkbox_widget = table.cellWidget(row, 0)
#                 if checkbox_widget is not None:
#                     checkbox = checkbox_widget.findChild(QCheckBox)
#                     if checkbox is not None and checkbox.isChecked():
#                         included_players.append(table.item(row, 1).text())
#         return included_players

#     def collect_stack_settings(self):
#         stack_settings = {}
#         for row in range(self.stack_exposure_table.rowCount()):
#             checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
#             if checkbox_widget is not None:
#                 checkbox = checkbox_widget.findChild(QCheckBox)
#                 if checkbox is not None and checkbox.isChecked():
#                     stack_type = self.stack_exposure_table.item(row, 1).text()
#                     stack_settings[stack_type] = True
#         return stack_settings

#     def collect_exposure_settings(self):
#         min_exposure = {}
#         self.max_exposure = {}
#         for position in self.player_tables:
#             table = self.player_tables[position]
#             for row in range(table.rowCount()):
#                 player_name = table.item(row, 1).text()
#                 min_exp_widget = table.cellWidget(row, 7)
#                 max_exp_widget = table.cellWidget(row, 8)
#                 if isinstance(min_exp_widget, QSpinBox) and isinstance(max_exp_widget, QSpinBox):
#                     min_exposure[player_name] = min_exp_widget.value() / 100
#                     self.max_exposure[player_name] = max_exp_widget.value() / 100
#         return min_exposure, self.max_exposure

#     def collect_team_selections(self):
#         team_selections = {}
#         for stack_size, table in self.team_stack_tables.items():
#             if stack_size != "All Stacks":
#                 team_selections[int(stack_size.split()[0])] = []
#                 for row in range(table.rowCount()):
#                     checkbox_widget = table.cellWidget(row, 0)
#                     if checkbox_widget is not None:
#                         checkbox = checkbox_widget.findChild(QCheckBox)
#                         if checkbox is not None and checkbox.isChecked():
#                             team_selections[int(stack_size.split()[0])].append(table.item(row, 1).text())
#         return team_selections

#     def load_dk_predictions(self):
#         """Load DraftKings predictions from the new prediction format"""
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, 
#             'Open DraftKings Predictions CSV', 
#             'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/', 
#             'CSV Files (*.csv)'
#         )
#         if file_path:
#             try:
#                 self.df_players = self.load_players(file_path)
#                 self.populate_player_tables()
#                 self.status_label.setText(f'DraftKings predictions loaded: {len(self.df_players)} players')
#             except Exception as e:
#                 self.status_label.setText(f'Error loading DraftKings predictions: {str(e)}')
#                 logging.error(f"Error loading DraftKings predictions: {e}")

# if __name__ == "__main__":
#     logging.debug(f"PuLP version: {pulp.__version__}")
    
#     app = QApplication(sys.argv)
#     window = FantasyBaseballApp()
#     window.show()
#     sys.exit(app.exec_())
# ?????????????????# This code is part of a larger application and is not meant to be run standalone.
# It is designed to be integrated into a PyQt5 application for optimizing MLB DFS lineups   
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

# ... existing imports ...

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
    
    def __init__(self, df_players, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, min_points, monte_carlo_iterations, num_lineups, team_selections, min_unique=0):
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

    def run(self):
        logging.debug("OptimizationWorker: Starting optimization")
        results, team_exposure, stack_exposure = self.optimize_lineups()
        logging.debug(f"OptimizationWorker: Optimization complete. Results: {len(results)}")
        self.optimization_done.emit(results, team_exposure, stack_exposure)

    def optimize_lineups(self):
        df_filtered = self.preprocess_data()
        logging.debug(f"optimize_lineups: Starting with {len(df_filtered)} players")

        results = {}
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)
        
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
        
        # Apply exposure constraints
        # For now, just return the data as-is since we don't have active exposure tracking in this context
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
        
        # Temporarily store the current optimized_lineups
        original_lineups = getattr(self, 'optimized_lineups', None)
        
        try:
            # Replace optimized_lineups with favorites for the duration of this function
            # Extract just the lineup DataFrames from favorites (favorites store {'lineup': df, ...})
            favorite_lineups = [fav['lineup'] for fav in self.favorites_lineups[:num_to_use]]
            self.optimized_lineups = favorite_lineups
            
            # Use the EXACT same logic as create_filled_entries_df
            filled_entries = self.create_filled_entries_df(num_to_use)
            
            # Save to CSV
            filled_entries.to_csv(output_path, index=False)
            
            logging.info(f"Successfully exported {len(filled_entries)} favorites using create_filled_entries_df logic")
            
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
            min_unique=min_unique  # Add min unique constraint
        )
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
        """Get the list of included players from the UI"""
        included_players = []
        
        if not hasattr(self, 'player_tables') or not self.player_tables:
            logging.debug("No player tables available")
            return included_players
        
        # Check the "All Batters" table for selected players
        if "All Batters" in self.player_tables:
            table = self.player_tables["All Batters"]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox and checkbox.isChecked():
                        name_item = table.item(row, 1)
                        if name_item:
                            included_players.append(name_item.text())
        
        logging.debug(f"Found {len(included_players)} included players from UI")
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
        """Display optimization results with unique constraint filtering"""
        logging.debug(f"display_results: Received {len(results)} results")
        self.results_table.setRowCount(0)
        
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
            self.add_lineup_to_results(lineup_data, total_lineups)
            self.optimized_lineups.append(lineup_data['lineup'])

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.refresh_team_stacks()
        
        # Create status message with feedback about generated vs requested lineups
        unique_msg = f" (Min unique: {min_unique})" if min_unique > 0 else ""
        
        if total_lineups < requested_lineups:
            # Show warning if fewer lineups generated than requested
            shortage_msg = f"⚠️ Generated {total_lineups} lineups (requested {requested_lineups})"
            self.status_label.setText(f"Optimization complete. {shortage_msg}{unique_msg}")
            
            # Show detailed feedback dialog
            QMessageBox.information(
                self,
                "Lineup Generation Notice",
                f"🎯 Optimization Results:\n\n"
                f"📊 Requested: {requested_lineups} lineups\n"
                f"✅ Generated: {total_lineups} lineups\n\n"
                f"{'🔄 Min unique constraint limited results' if min_unique > 0 else '⚡ Limited by available player combinations'}\n\n"
                f"💡 Tip: Try reducing min unique constraint or adjusting stack settings for more lineups."
            )
        else:
            self.status_label.setText(f"Optimization complete. Generated {total_lineups} lineups{unique_msg}.")

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

    def add_lineup_to_results(self, lineup_data, total_lineups):
        """Add a lineup to the results table"""
        total_points = lineup_data['total_points']
        lineup = lineup_data['lineup']
        total_salary = lineup['Salary'].sum()

        for _, player in lineup.iterrows():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)
            self.results_table.setItem(row_position, 0, QTableWidgetItem(str(player['Name'])))
            self.results_table.setItem(row_position, 1, QTableWidgetItem(str(player['Team'])))
            self.results_table.setItem(row_position, 2, QTableWidgetItem(str(player['Pos'])))
            self.results_table.setItem(row_position, 3, QTableWidgetItem(str(player['Salary'])))
            self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{player['Predicted_DK_Points']:.2f}"))
            self.results_table.setItem(row_position, 5, QTableWidgetItem(str(total_salary)))
            self.results_table.setItem(row_position, 6, QTableWidgetItem(f"{total_points:.2f}"))

            player_name = player['Name']
            if player_name in self.player_exposure:
                self.player_exposure[player_name] += 1
            else:
                self.player_exposure[player_name] = 1

            exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
            self.results_table.setItem(row_position, 7, QTableWidgetItem(f"{exposure:.2f}%"))
            self.results_table.setItem(row_position, 8, QTableWidgetItem(f"{self.max_exposure.get(player_name, 100):.2f}%"))

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

    def populate_position_table(self, table, df_pos):
        """Populate a specific position table with player data"""
        table.setRowCount(len(df_pos))
        
        for row_idx, (_, player) in enumerate(df_pos.iterrows()):
            # Checkbox for inclusion
            checkbox = QCheckBox()
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
        
        # Check if position counts match expected DK format
        expected_counts = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
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
            num_to_use, ok = QInputDialog.getInt(
                self, 
                'Number of Lineups', 
                f'How many lineups to use?\n\n(Available: {num_available} optimized lineups)', 
                value=min(num_available, 150),
                min=1, 
                max=num_available
            )
            
            if not ok:
                return
            
            # Create the filled entries DataFrame
            filled_entries = self.create_filled_entries_df(num_to_use)
            
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
        if not hasattr(self, 'dk_entries_df') or self.dk_entries_df is None:
            raise ValueError("No DraftKings entries file loaded")
        
        if not hasattr(self, 'optimized_lineups') or not self.optimized_lineups:
            raise ValueError("No optimized lineups available")
        
        # Extract player name to ID mapping from multiple sources
        player_name_to_id_map = self.extract_player_id_mapping_from_dk_file()
        
        # Also try to get mappings from loaded player data if DK file doesn't have them
        if len(player_name_to_id_map) == 0:
            logging.info("No player IDs found in DK file, trying loaded player data...")
            player_name_to_id_map = self.create_player_id_mapping_from_loaded_data()
        else:
            # Supplement with loaded data mappings for any missing players
            loaded_data_map = self.create_player_id_mapping_from_loaded_data()
            for name, player_id in loaded_data_map.items():
                if name not in player_name_to_id_map:
                    player_name_to_id_map[name] = player_id
        
        logging.debug(f"Found {len(player_name_to_id_map)} player ID mappings")
        
        # Check if we have any player ID mappings - warn if not
        if len(player_name_to_id_map) == 0:
            logging.warning("No player ID mappings found! This will result in empty lineups.")
            logging.warning("Make sure the DK file contains player data with 'Name + ID' format or separate Name/ID columns.")
        
        # Create the output DataFrame with correct DraftKings headers
        correct_headers = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
        
        # Find the first entry row (has an Entry ID value) from the original file
        entry_row_idx = None
        for idx, row in self.dk_entries_df.iterrows():
            if pd.notna(row.iloc[0]):
                # Check if this looks like an Entry ID (numeric)
                entry_id_value = row.iloc[0]
                if isinstance(entry_id_value, (int, float)) or (isinstance(entry_id_value, str) and entry_id_value.strip().replace('.', '').isdigit()):
                    entry_id_str = str(int(float(entry_id_value))).strip()
                    if len(entry_id_str) >= 8:  # DK entry IDs are typically 10 digits
                        entry_row_idx = idx
                        break
        
        if entry_row_idx is None:
            logging.warning("Could not find the entry row in the DK file - creating filled file without entry metadata")
            # Create new DataFrame without original entry metadata
            filled_entries = pd.DataFrame(columns=correct_headers)
        else:
            # Get the original entry metadata from the first valid entry row
            original_entry = self.dk_entries_df.iloc[entry_row_idx]
            
            # Handle the entry ID as float/int
            entry_id_raw = original_entry.iloc[0] if pd.notna(original_entry.iloc[0]) else ""
            if isinstance(entry_id_raw, (int, float)):
                entry_id = str(int(entry_id_raw))
            else:
                entry_id = str(entry_id_raw).strip()
                
            contest_name = str(original_entry.iloc[1]).strip() if len(original_entry) > 1 and pd.notna(original_entry.iloc[1]) else ""
            
            contest_id_raw = original_entry.iloc[2] if len(original_entry) > 2 and pd.notna(original_entry.iloc[2]) else ""
            if isinstance(contest_id_raw, (int, float)):
                contest_id = str(int(contest_id_raw))
            else:
                contest_id = str(contest_id_raw).strip()
                
            entry_fee = str(original_entry.iloc[3]).strip() if len(original_entry) > 3 and pd.notna(original_entry.iloc[3]) else ""
            
            # Create DataFrame with correct headers
            filled_entries = pd.DataFrame(columns=correct_headers)
            
            logging.info(f"Found original entry metadata: ID={entry_id}, Name={contest_name}, Contest ID={contest_id}, Fee={entry_fee}")
        
        # Fill entries with optimized lineups
        lineups_to_use = self.optimized_lineups[:num_to_use]
        
        for i, lineup in enumerate(lineups_to_use):
            # Create row data starting with entry metadata (if available)
            if entry_row_idx is not None:
                row_data = [entry_id, contest_name, contest_id, entry_fee]
            else:
                row_data = ["", "", "", ""]
            
            # Format lineup and add position assignments
            formatted_positions = self.format_lineup_positions_only(lineup, player_name_to_id_map)
            row_data.extend(formatted_positions)
            
            # Add the row to the DataFrame
            filled_entries.loc[i] = row_data
        
        logging.debug(f"Created filled entries DataFrame with {len(filled_entries)} rows")
        return filled_entries

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
        logging.debug(f"Using raw file parsing to extract player IDs from {self.dk_entries_file_path}")
        
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
                                    logging.debug(f"Found player mapping: {name_part} -> {id_part}")
        
        except Exception as e:
            logging.error(f"Error reading DK file for raw parsing: {e}")
            # Fallback to the pandas-based approach
            return self.extract_player_id_mapping_from_dk_file_pandas()
        
        logging.info(f"Extracted {len(player_map)} player ID mappings from DK file using raw parsing")
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
            player_id = player_name_to_id_map.get(name, "")
            
            # If no ID mapping found, try to create one from available player data
            if not player_id:
                # Check if player has an ID column
                if 'ID' in player and pd.notna(player['ID']):
                    potential_id = str(player['ID']).strip()
                    if potential_id.isdigit() and len(potential_id) >= 6:
                        player_id = potential_id
                # Check for Player ID column
                elif hasattr(player, 'get') and player.get('Player ID'):
                    potential_id = str(player['Player ID']).strip()
                    if potential_id.isdigit() and len(potential_id) >= 6:
                        player_id = potential_id
                # Check if the name itself already contains an ID
                elif '(' in name and ')' in name and name.endswith(')'):
                    try:
                        id_part = name.split('(')[1].replace(')', '').strip()
                        if id_part.isdigit() and len(id_part) >= 6:
                            player_id = id_part
                    except:
                        pass
                else:
                    # Try to find ID from any column that might contain it
                    for col_name, col_value in player.items():
                        if 'id' in str(col_name).lower() and pd.notna(col_value):
                            potential_id = str(col_value).strip()
                            if potential_id.isdigit() and len(potential_id) >= 6:
                                player_id = potential_id
                                break
            
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
        
        # Create the position assignments in DK format: [P, P, C, 1B, 2B, 3B, SS, OF, OF, OF]
        position_assignments = []
        
        # Add two pitchers
        position_assignments.append(position_players['P'][0] if len(position_players['P']) > 0 else "")
        position_assignments.append(position_players['P'][1] if len(position_players['P']) > 1 else "")
        
        # Add catcher
        position_assignments.append(position_players['C'][0] if len(position_players['C']) > 0 else "")
        
        # Add infielders
        position_assignments.append(position_players['1B'][0] if len(position_players['1B']) > 0 else "")
        position_assignments.append(position_players['2B'][0] if len(position_players['2B']) > 0 else "")
        position_assignments.append(position_players['3B'][0] if len(position_players['3B']) > 0 else "")
        position_assignments.append(position_players['SS'][0] if len(position_players['SS']) > 0 else "")
        
        # Add three outfielders
        position_assignments.append(position_players['OF'][0] if len(position_players['OF']) > 0 else "")
        position_assignments.append(position_players['OF'][1] if len(position_players['OF']) > 1 else "")
        position_assignments.append(position_players['OF'][2] if len(position_players['OF']) > 2 else "")
        
        return position_assignments
if __name__ == "__main__":
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = FantasyBaseballApp()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())