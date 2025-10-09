import sys
sys.stdout = sys.__stdout__

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from pulp import GLPK

# Replace the solver initialization with:

import traceback
import psutil
import pulp
import pandas as pd
import pulp
print(pulp.pulpTestAll())
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import concurrent.futures
from itertools import combinations
from scipy.stats import multivariate_normal
from collections import defaultdict

# Constants
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

class OptimizationWorker(QThread):
    optimization_done = pyqtSignal(list, dict, dict)

    def __init__(self, df, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, team_selections, min_points=1, monte_carlo_iterations=100):
        super().__init__()
        self.df = df
        self.salary_cap = salary_cap
        self.position_limits = position_limits
        self.included_players = included_players
        self.stack_settings = stack_settings
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.team_selections = team_selections
        self.min_points = min_points
        self.monte_carlo_iterations = monte_carlo_iterations

    def run(self):
        logging.debug("Starting optimization worker run method")
        try:
            results, team_exposure, stack_exposure = self.find_best_lineups(
                self.df,
                self.salary_cap,
                self.position_limits,
                self.included_players,
                self.stack_settings,
                self.min_exposure,
                self.max_exposure,
                self.team_selections,
                self.min_points,
                self.monte_carlo_iterations
            )
            logging.debug("Optimization completed, emitting results")
            self.optimization_done.emit(results, team_exposure, stack_exposure)
        except Exception as e:
            logging.error(f"Error in optimization thread: {e}")
            logging.error(traceback.format_exc())

    def find_best_lineups(self, df, salary_cap, position_limits, included_players, stack_settings, min_exposure, max_exposure, team_selections, min_points=1, monte_carlo_iterations=100):
        logging.debug("Starting find_best_lineups method")
        logging.debug(f"Stack settings: {stack_settings}")
        results = []
        team_exposure = defaultdict(int)
        stack_exposure = defaultdict(int)

        logging.debug("Starting Monte Carlo simulation")
        simulated_data = self.monte_carlo_simulation(df, monte_carlo_iterations)
        logging.debug(f"Completed Monte Carlo simulation with {len(simulated_data)} iterations")
        for i, sim_df in enumerate(simulated_data):
            logging.debug(f"Processing simulation {i+1}/{monte_carlo_iterations}")
            for stack_type in stack_settings:
                if stack_settings[stack_type]:
                    logging.debug(f"Optimizing for stack type: {stack_type}")
                    lineup = self.optimize_lineup(sim_df, salary_cap, position_limits, included_players, stack_type, min_exposure, max_exposure, min_points, team_selections)
                    if not lineup.empty:
                        total_points = lineup['My Proj'].sum()
                        results.append((total_points, lineup))
                        for team in team_selections:
                            if team in lineup['Team'].values:
                                team_exposure[team] += 1
                        stack_exposure[stack_type] += 1
                    else:
                        logging.warning(f"No valid lineup found for stack type: {stack_type}")

        logging.debug("Completed all simulations and optimizations")

        total_lineups = len(results)
        for team in team_exposure:
            team_exposure[team] = team_exposure[team] / total_lineups * 100 if total_lineups > 0 else 0
        for stack_type in stack_exposure:
            stack_exposure[stack_type] = stack_exposure[stack_type] / total_lineups * 100 if total_lineups > 0 else 0

        sorted_results = sorted(results, key=lambda x: -x[0])[:20]

        return sorted_results, team_exposure, stack_exposure

    def optimize_lineup(self, df, salary_cap, position_limits, included_players, stack_type, min_exposure, max_exposure, min_points, team_selections):
        try:
            logging.debug("Starting optimize_lineup method")
            df = self.validate_data(df)
            logging.debug(f"Validated dataframe shape: {df.shape}")

            # Log the number of included players
            logging.debug(f"Number of included players: {len(included_players)}")

            # Filter by included players only if the list is not empty
            if included_players:
                df = df[df['Name'].isin(included_players)]
            df = df[df['My Proj'] >= min_points]
            logging.debug(f"Filtered dataframe shape: {df.shape}")

            if df.empty:
                logging.warning("No players available after filtering")
                return pd.DataFrame()

            problem = pulp.LpProblem("MyProblem", pulp.LpMaximize)

            player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') for idx in df.index}
            # Objective function
            problem += pulp.lpSum([df.at[idx, 'My Proj'] * player_vars[idx] for idx in df.index])

            # Salary cap constraint
            problem += pulp.lpSum([df.at[idx, 'Salary'] * player_vars[idx] for idx in df.index]) <= salary_cap

            # Position constraints
            for position, limit in position_limits.items():
                problem += pulp.lpSum([player_vars[idx] for idx in df.index if position in df.at[idx, 'Pos'].split('/')]) == limit

            # Team size constraint
            problem += pulp.lpSum([player_vars[idx] for idx in df.index]) == sum(position_limits.values())

            # Add stacking constraints based on stack_type
            if stack_type != "No Stacks":
                stack_sizes = [int(x) for x in stack_type.split('|')]
                teams = df['Team'].unique()
                for team in teams:
                    team_players = [idx for idx in df.index if df.at[idx, 'Team'] == team]
                    problem += pulp.lpSum([player_vars[idx] for idx in team_players]) >= max(stack_sizes)

            # Log the problem details
            logging.debug(f"Number of variables: {len(problem.variables())}")
            logging.debug(f"Number of constraints: {len(problem.constraints)}")

            # Solve the problem
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
            status = problem.solve(solver)

            logging.debug(f"Optimization status: {pulp.LpStatus[status]}")

            if pulp.LpStatus[status] == "Optimal":
                selected_players = [idx for idx in df.index if player_vars[idx].value() > 0.5]
                return df.loc[selected_players]
            else:
                logging.warning(f"No optimal solution found. Status: {pulp.LpStatus[status]}")
                return pd.DataFrame()

        except Exception as e:
            logging.error(f"Error during optimization: {e}")
            logging.error(traceback.format_exc())
            return pd.DataFrame()

    def monte_carlo_simulation(self, df, iterations=100):
        projection_columns = ['My Proj']
        if df[projection_columns].dropna().empty:
            raise ValueError("Projection data is empty or invalid.")
        df.loc[:, projection_columns] = df[projection_columns].fillna(0)
        covariance_matrix = np.cov(df[projection_columns].values.T)
        if np.isnan(covariance_matrix).any() or np.isinf(covariance_matrix).any():
            raise ValueError("Covariance matrix contains NaNs or infinite values.")
        simulated_data = []
        mean_projections = df[projection_columns].mean().values
        for _ in range(iterations):
            simulated_projections = multivariate_normal.rvs(mean=mean_projections, cov=covariance_matrix)
            simulation = df.copy()
            simulation['My Proj'] = simulated_projections
            simulated_data.append(simulation)
        return simulated_data

    def validate_data(self, df):
        if df.isnull().values.any():
            df = df.fillna(0)
        numeric_cols = ['Salary', 'My Proj', 'Value', 'Min Exp', 'Max Exp']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        return df

class FantasyBaseballApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced MLB DFS Optimizer")
        self.setGeometry(100, 100, 1600, 1000)

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

        self.create_players_tab()
        self.create_team_stack_tab()
        self.create_stack_exposure_tab()
        self.create_control_panel()

    def create_players_tab(self):
        players_tab = QWidget()
        self.tabs.addTab(players_tab, "Players")

        players_layout = QVBoxLayout()
        players_tab.setLayout(players_layout)

        position_tabs = QTabWidget()
        players_layout.addWidget(position_tabs)

        self.player_tables = {}

        positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        for position in positions:
            sub_tab = QWidget()
            position_tabs.addTab(sub_tab, position)
            layout = QVBoxLayout()
            sub_tab.setLayout(layout)

            select_all_button = QPushButton("Select All")
            deselect_all_button = QPushButton("Deselect All")
            select_all_button.clicked.connect(lambda _, p=position: self.select_all(p))
            deselect_all_button.clicked.connect(lambda _, p=position: self.deselect_all(p))
            button_layout = QHBoxLayout()
            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            layout.addLayout(button_layout)

            table = QTableWidget(0, 11)
            table.setHorizontalHeaderLabels(["Select", "Name", "Team", "Pos", "Salary", "My Proj", "Own", "Min Exp", "Max Exp", "Actual Exp (%)", "My Proj"])
            layout.addWidget(table)

            self.player_tables[position] = table

    def create_team_stack_tab(self):
        team_stack_tab = QWidget()
        self.tabs.addTab(team_stack_tab, "Team Stacks")

        layout = QVBoxLayout()
        team_stack_tab.setLayout(layout)

        self.team_stack_table = QTableWidget(0, 8)
        self.team_stack_table.setHorizontalHeaderLabels(["Select", "Teams", "Status", "Time", "Proj Runs", "Min Exp", "Max Exp", "Actual Exp (%)"])
        layout.addWidget(self.team_stack_table)

    def create_stack_exposure_tab(self):
        stack_exposure_tab = QWidget()
        self.tabs.addTab(stack_exposure_tab, "Stack Exposure")
        layout = QVBoxLayout()
        stack_exposure_tab.setLayout(layout)
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

        load_entries_button = QPushButton('Load Entries CSV')
        load_entries_button.clicked.connect(self.load_entries_csv)
        control_layout.addWidget(load_entries_button)

        self.min_unique_label = QLabel('Min Unique:')
        self.min_unique_input = QLineEdit()
        control_layout.addWidget(self.min_unique_label)
        control_layout.addWidget(self.min_unique_input)

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

        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(["Player", "Team", "Pos", "Salary", "Proj Points", "Total Salary", "Total Points", "Exposure (%)", "Max Exp (%)"])
        control_layout.addWidget(self.results_table)

        self.status_label = QLabel('')
        control_layout.addWidget(self.status_label)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_players = self.load_players(file_path)
            self.populate_player_tables()

    def load_entries_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Entries CSV', '', 'CSV Files (*.csv)')
        if file_path:
            self.df_entries = self.load_and_standardize_csv(file_path)
            if self.df_entries is not None:
                self.status_label.setText('Entries CSV loaded and standardized successfully.')
            else:
                self.status_label.setText('Failed to standardize Entries CSV.')

    def load_players(self, csv_path):
        df = pd.read_csv(csv_path)
        required_columns = ['Name', 'Team', 'Opp', 'Pos', 'My Proj', 'Salary']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        df['MyProj'] = df['My Proj']
        df['Positions'] = df['Pos'].apply(lambda x: x.split('/') if pd.notna(x) else [])
        return df

    def load_and_standardize_csv(self, file_path):
        try:
            df = pd.read_csv(file_path, skiprows=6, on_bad_lines='skip')
            df.columns = ['ID', 'Name', 'Other Columns...'] + df.columns[3:].tolist()
            return df
        except Exception as e:
            logging.error(f"Error loading or processing file: {e}")
            return None

    def populate_player_tables(self):
        positions = ["All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"]
        for position in positions:
            table = self.player_tables[position]
            table.setRowCount(0)

            if self.df_players is not None:
                if position == "P":
                    df_filtered = self.df_players[self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
                elif position == "All Batters":
                    df_filtered = self.df_players[~self.df_players['Pos'].str.contains('SP|RP|P', na=False)]
                else:
                    df_filtered = self.df_players[self.df_players['Positions'].apply(lambda x: position in x)]
                
                for _, row in df_filtered.iterrows():
                    row_position = table.rowCount()
                    table.insertRow(row_position)

                    checkbox = QCheckBox()
                    checkbox_widget = QWidget()
                    layout_checkbox = QHBoxLayout(checkbox_widget)
                    layout_checkbox.addWidget(checkbox)
                    layout_checkbox.setAlignment(Qt.AlignCenter)
                    layout_checkbox.setContentsMargins(0, 0, 0, 0)
                    table.setCellWidget(row_position, 0, checkbox_widget)
                    table.setItem(row_position, 1, QTableWidgetItem(str(row['Name'])))
                    table.setItem(row_position, 2, QTableWidgetItem(str(row['Team'])))
                    table.setItem(row_position, 3, QTableWidgetItem(str(row['Pos'])))
                    table.setItem(row_position, 4, QTableWidgetItem(str(row['Salary'])))
                    table.setItem(row_position, 5, QTableWidgetItem(str(row['My Proj'])))
                    
                    min_exp_spinbox = QSpinBox()
                    min_exp_spinbox.setRange(0, 100)
                    min_exp_spinbox.setValue(0)
                    table.setCellWidget(row_position, 7, min_exp_spinbox)
                    
                    max_exp_spinbox = QSpinBox()
                    max_exp_spinbox.setRange(0, 100)
                    max_exp_spinbox.setValue(100)
                    table.setCellWidget(row_position, 8, max_exp_spinbox)
                    
                    actual_exp_label = QLabel("0%")
                    table.setCellWidget(row_position, 9, actual_exp_label)

                    if row['Name'] not in self.player_exposure:
                        self.player_exposure[row['Name']] = 0

        self.populate_team_stack_table()

    def populate_team_stack_table(self):
        self.team_stack_table.setRowCount(0)
        selected_teams = set()
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        selected_teams.add(table.item(row, 2).text())
        
        for team in selected_teams:
            row_position = self.team_stack_table.rowCount()
            self.team_stack_table.insertRow(row_position)
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            layout_checkbox = QHBoxLayout(checkbox_widget)
            layout_checkbox.addWidget(checkbox)
            layout_checkbox.setAlignment(Qt.AlignCenter)
            layout_checkbox.setContentsMargins(0, 0, 0, 0)
            self.team_stack_table.setCellWidget(row_position, 0, checkbox_widget)
            self.team_stack_table.setItem(row_position, 1, QTableWidgetItem(team))
            self.team_stack_table.setItem(row_position, 2, QTableWidgetItem("Playing"))
            self.team_stack_table.setItem(row_position, 3, QTableWidgetItem("7:00 PM"))
            self.team_stack_table.setItem(row_position, 4, QTableWidgetItem(str(np.random.uniform(3, 10))))
            
            min_exp_spinbox = QSpinBox()
            min_exp_spinbox.setRange(0, 100)
            min_exp_spinbox.setValue(0)
            self.team_stack_table.setCellWidget(row_position, 5, min_exp_spinbox)
            
            max_exp_spinbox = QSpinBox()
            max_exp_spinbox.setRange(0, 100)
            max_exp_spinbox.setValue(100)
            self.team_stack_table.setCellWidget(row_position, 6, max_exp_spinbox)

            actual_exp_label = QLabel("0%")
            self.team_stack_table.setCellWidget(row_position, 7, actual_exp_label)

    def select_all(self, position):
        table = self.player_tables[position]
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(True)
        self.populate_team_stack_table()

    def deselect_all(self, position):
        table = self.player_tables[position]
        for row in range(table.rowCount()):
            checkbox_widget = table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None:
                    checkbox.setChecked(False)
        self.populate_team_stack_table()

    def run_optimization(self):
        logging.debug("Starting run_optimization method")
        included_players = self.get_included_players()
        stack_settings = self.collect_stack_settings()
        min_exposure, max_exposure = self.collect_exposure_settings()
        team_selections = self.collect_team_selections()

        logging.debug(f"Included players: {len(included_players)}")
        logging.debug(f"Stack settings: {stack_settings}")
        logging.debug(f"Team selections: {team_selections}")
        logging.debug(f"Min exposure: {min_exposure}")
        logging.debug(f"Max exposure: {max_exposure}")

        if self.df_players is None or self.df_players.empty:
            self.status_label.setText("No player data loaded. Please load a CSV first.")
            return

        logging.debug(f"df_players shape: {self.df_players.shape}")

        # Log the first few rows of df_players
        logging.debug(f"First few rows of df_players:\n{self.df_players.head().to_string()}")

        self.optimization_thread = OptimizationWorker(
            self.df_players,
            SALARY_CAP,
            POSITION_LIMITS,
            included_players=included_players,
            stack_settings=stack_settings,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
            team_selections=team_selections,
            min_points=1,
            monte_carlo_iterations=100
        )
        self.optimization_thread.optimization_done.connect(self.display_results)
        logging.debug("Starting optimization thread")
        self.optimization_thread.start()
        self.status_label.setText("Running optimization... Please wait.")

    def display_results(self, results, team_exposure, stack_exposure):
        logging.debug("Starting display_results method")
        self.results_table.setRowCount(0)
        total_lineups = len(results)

        for i, (total_points, lineup) in enumerate(results):
            for _, row in lineup.iterrows():
                row_position = self.results_table.rowCount()
                self.results_table.insertRow(row_position)
                self.results_table.setItem(row_position, 0, QTableWidgetItem(str(row['Name'])))
                self.results_table.setItem(row_position, 1, QTableWidgetItem(str(row['Team'])))
                self.results_table.setItem(row_position, 2, QTableWidgetItem(str(row['Pos'])))
                self.results_table.setItem(row_position, 3, QTableWidgetItem(str(row['Salary'])))
                self.results_table.setItem(row_position, 4, QTableWidgetItem(str(row['My Proj'])))
                self.results_table.setItem(row_position, 5, QTableWidgetItem(str(lineup['Salary'].sum())))
                self.results_table.setItem(row_position, 6, QTableWidgetItem(str(total_points)))

                player_name = row['Name']
                if player_name in self.player_exposure:
                    self.player_exposure[player_name] += 1
                else:
                    self.player_exposure[player_name] = 1

        self.update_exposure_in_all_tabs(total_lineups, team_exposure, stack_exposure)
        self.status_label.setText(f"Optimization complete. Generated {total_lineups} lineups.")

    def update_exposure_in_all_tabs(self, total_lineups, team_exposure, stack_exposure):
        if total_lineups > 0:
            for position in self.player_tables:
                table = self.player_tables[position]
                for row in range(table.rowCount()):
                    player_name = table.item(row, 1).text()
                    actual_exposure = self.player_exposure.get(player_name, 0) / total_lineups * 100
                    actual_exposure_label = table.cellWidget(row, 9)
                    if isinstance(actual_exposure_label, QLabel):
                        actual_exposure_label.setText(f"{actual_exposure:.2f}%")
            
            for row in range(self.team_stack_table.rowCount()):
                team_name = self.team_stack_table.item(row, 1).text()
                actual_exposure = team_exposure.get(team_name, 0) / total_lineups * 100 if total_lineups > 0 else 0
                self.team_stack_table.setItem(row, 7, QTableWidgetItem(f"{actual_exposure:.2f}%"))
            
            for row in range(self.stack_exposure_table.rowCount()):
                stack_type = self.stack_exposure_table.item(row, 1).text()
                actual_exposure = stack_exposure.get(stack_type, 0) / total_lineups * 100 if total_lineups > 0 else 0
                self.stack_exposure_table.setItem(row, 4, QTableWidgetItem(f"{actual_exposure:.2f}%"))

    def save_csv(self):
        output_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")

        if output_path:
            if hasattr(self, 'df_entries') and self.df_entries is not None:
                try:
                    self.generate_output(self.df_entries, self.df_players, output_path)
                    self.status_label.setText(f'CSV saved successfully at {output_path}')
                except Exception as e:
                    self.status_label.setText(f'Error saving CSV: {e}')
            else:
                self.status_label.setText('Entries CSV not loaded. Please load the Entries CSV first.')
        else:
            self.status_label.setText('Save operation canceled.')

    def generate_output(self, entries_df, players_df, output_path):
        optimized_output = players_df[["Name", "Team", "Pos", "Salary", "My Proj"]]
        optimized_output.to_csv(output_path, index=False)

    def get_included_players(self):
        included_players = []
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                checkbox_widget = table.cellWidget(row, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QCheckBox)
                    if checkbox is not None and checkbox.isChecked():
                        included_players.append(table.item(row, 1).text())
        return included_players

    def collect_stack_settings(self):
        stack_settings = {}
        for row in range(self.stack_exposure_table.rowCount()):
            checkbox_widget = self.stack_exposure_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None and checkbox.isChecked():
                    stack_type = self.stack_exposure_table.item(row, 1).text()
                    stack_settings[stack_type] = True
        return stack_settings

    def collect_exposure_settings(self):
        min_exposure = {}
        max_exposure = {}
        for position in self.player_tables:
            table = self.player_tables[position]
            for row in range(table.rowCount()):
                player_name = table.item(row, 1).text()
                min_exp_widget = table.cellWidget(row, 7)
                max_exp_widget = table.cellWidget(row, 8)
                if isinstance(min_exp_widget, QSpinBox) and isinstance(max_exp_widget, QSpinBox):
                    min_exposure[player_name] = min_exp_widget.value() / 100
                    max_exposure[player_name] = max_exp_widget.value() / 100
        return min_exposure, max_exposure

    def collect_team_selections(self):
        selected_teams = set()
        for row in range(self.team_stack_table.rowCount()):
            checkbox_widget = self.team_stack_table.cellWidget(row, 0)
            if checkbox_widget is not None:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox is not None and checkbox.isChecked():
                    selected_teams.add(self.team_stack_table.item(row, 1).text())
        return list(selected_teams)

if __name__ == "__main__":
    logging.debug(f"PuLP version: {pulp.__version__}")
    logging.debug(f"PuLP test results: {pulp.pulpTestAll()}")
    logging.debug(f"Initial memory usage: {psutil.virtual_memory().percent}%")
    logging.debug(f"Initial CPU usage: {psutil.cpu_percent()}%")
    app = QApplication(sys.argv)
    window = FantasyBaseballApp()
    window.show()
    sys.exit(app.exec_())
