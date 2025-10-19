#!/usr/bin/env python3
"""
NFL Underdog Fantasy GUI - Display and Analyze NFL Predictions
Loads NFL prediction CSV files from sportsdata.io and displays results in an interactive interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
from datetime import datetime
import numpy as np
from underdog_api_client import UnderdogAPIClient
from nfl_probability_engine import NFLProbabilityEngine

class NFLUnderdogFantasyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NFL Underdog Fantasy Prediction Viewer - Real API Lines")
        self.root.geometry("1400x900")
        
        # Data storage
        self.predictions_df = None
        self.recommendations_df = None
        self.nfl_data_df = None
        self.prop_bets = []
        self.parlays = []
        self.underdog_props = []  # Real Underdog props from API
        
        # Team selection storage
        self.available_teams = []
        self.selected_teams = set()
        
        # Default directory for NFL data files
        self.default_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION"
        self.analysis_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/7_ANALYSIS"
        
        # Initialize Underdog API client and probability engine
        self.underdog_client = UnderdogAPIClient()
        self.prob_engine = NFLProbabilityEngine()
        
        self.setup_gui()
        self.load_nfl_data()
        self.load_underdog_props()  # Load real Underdog lines
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🏈 NFL Underdog Fantasy Prediction Viewer - SportsData.io", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load buttons
        ttk.Button(control_frame, text="Load NFL Data", 
                  command=self.load_nfl_data).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Generate Props", 
                  command=self.generate_nfl_props).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Build Parlays", 
                  command=self.build_parlays).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Refresh", 
                  command=self.refresh_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: NFL Data Overview
        self.nfl_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.nfl_data_frame, text="NFL Data Overview")
        self.setup_nfl_data_tab()
        
        # Tab 2: Prop Bets
        self.props_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.props_frame, text="Prop Bets")
        self.setup_props_tab()
        
        # Tab 3: Power Plays
        self.power_plays_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.power_plays_frame, text="Power Plays")
        self.setup_power_plays_tab()
        
        # Tab 4: Parlay Builder
        self.parlay_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.parlay_frame, text="Parlay Builder")
        self.setup_parlay_builder_tab()
        
        # Tab 5: Team Analysis
        self.team_analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.team_analysis_frame, text="Team Analysis")
        self.setup_team_analysis_tab()
    
    def setup_nfl_data_tab(self):
        """Setup NFL data overview tab"""
        # Data summary frame
        summary_frame = ttk.LabelFrame(self.nfl_data_frame, text="NFL Data Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Summary labels
        self.data_summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD)
        data_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.data_summary_text.yview)
        self.data_summary_text.configure(yscrollcommand=data_scrollbar.set)
        
        self.data_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Player data tree
        tree_frame = ttk.LabelFrame(self.nfl_data_frame, text="Player Data")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for player data
        columns = ('Name', 'Position', 'Team', 'Opponent', 'Projected Points', 'Salary')
        self.player_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.player_tree.heading(col, text=col)
            self.player_tree.column(col, width=120)
        
        # Scrollbars for tree
        tree_scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.player_tree.yview)
        tree_scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.player_tree.xview)
        self.player_tree.configure(yscrollcommand=tree_scrollbar_y.set, xscrollcommand=tree_scrollbar_x.set)
        
        self.player_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_props_tab(self):
        """Setup prop bets tab"""
        # Filter frame
        filter_frame = ttk.LabelFrame(self.props_frame, text="Filters")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Prop type filter
        ttk.Label(filter_frame, text="Prop Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.prop_type_var = tk.StringVar(value="All")
        prop_type_combo = ttk.Combobox(filter_frame, textvariable=self.prop_type_var, 
                                      values=["All", "Passing Yards", "Rushing Yards", "Receiving Yards", 
                                             "Passing TDs", "Rushing TDs", "Receiving TDs", "Receptions"])
        prop_type_combo.grid(row=0, column=1, padx=5, pady=5)
        
        # Team filter
        ttk.Label(filter_frame, text="Team:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.team_var = tk.StringVar(value="All")
        self.team_combo = ttk.Combobox(filter_frame, textvariable=self.team_var, values=["All"])
        self.team_combo.grid(row=0, column=3, padx=5, pady=5)
        
        # Apply filter button
        ttk.Button(filter_frame, text="Apply Filter", 
                  command=self.apply_prop_filter).grid(row=0, column=4, padx=5, pady=5)
        
        # Props tree
        tree_frame = ttk.LabelFrame(self.props_frame, text="Prop Bets")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Player', 'Team', 'Prop', 'Line', 'Projection', 'Probability', 'Multiplier')
        self.props_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.props_tree.heading(col, text=col)
            self.props_tree.column(col, width=120)
        
        # Scrollbars
        props_scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.props_tree.yview)
        props_scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.props_tree.xview)
        self.props_tree.configure(yscrollcommand=props_scrollbar_y.set, xscrollcommand=props_scrollbar_x.set)
        
        self.props_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        props_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        props_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_power_plays_tab(self):
        """Setup power plays tab"""
        # Power plays tree
        tree_frame = ttk.LabelFrame(self.power_plays_frame, text="Power Plays (High Probability Bets)")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('Player', 'Team', 'Prop', 'Line', 'Projection', 'Probability', 'Multiplier', 'Expected Return', 'Kelly Fraction')
        self.power_plays_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.power_plays_tree.heading(col, text=col)
            self.power_plays_tree.column(col, width=100)
        
        # Scrollbars
        power_scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.power_plays_tree.yview)
        power_scrollbar_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.power_plays_tree.xview)
        self.power_plays_tree.configure(yscrollcommand=power_scrollbar_y.set, xscrollcommand=power_scrollbar_x.set)
        
        self.power_plays_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        power_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        power_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_parlay_builder_tab(self):
        """Setup parlay builder tab"""
        # Parlay controls
        controls_frame = ttk.LabelFrame(self.parlay_frame, text="Parlay Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Max Legs:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_legs_var = tk.StringVar(value="6")
        ttk.Entry(controls_frame, textvariable=self.max_legs_var, width=5).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Min Probability:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.min_prob_var = tk.StringVar(value="0.20")
        ttk.Entry(controls_frame, textvariable=self.min_prob_var, width=5).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Build Parlays", 
                  command=self.build_parlays).grid(row=0, column=4, padx=5, pady=5)
        
        # Filter by leg count
        ttk.Label(controls_frame, text="Show:").grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        self.leg_filter_var = tk.StringVar(value="All")
        leg_filter_combo = ttk.Combobox(controls_frame, textvariable=self.leg_filter_var, 
                                       values=["All", "2-Leg", "3-Leg", "4-Leg", "5-Leg", "6-Leg"], width=8)
        leg_filter_combo.grid(row=0, column=6, padx=5, pady=5)
        leg_filter_combo.bind('<<ComboboxSelected>>', lambda e: self.filter_parlays_by_legs())
        
        # Summary frame for parlay counts
        summary_frame = ttk.LabelFrame(self.parlay_frame, text="Parlay Summary by Leg Count")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.parlay_summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD, font=("Courier", 10))
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.parlay_summary_text.yview)
        self.parlay_summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.parlay_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Create notebook for parlay categories
        self.parlay_notebook = ttk.Notebook(self.parlay_frame)
        self.parlay_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Store parlay trees by leg count
        self.parlay_trees = {}
        
        # Tab for All Parlays
        all_parlays_frame = ttk.Frame(self.parlay_notebook)
        self.parlay_notebook.add(all_parlays_frame, text="All Parlays")
        self.parlay_trees['all'] = self.setup_parlay_tree_dynamic(all_parlays_frame)
        
        # Create tabs for 2-6 leg parlays (will be populated dynamically)
        for leg_count in range(2, 7):  # 2, 3, 4, 5, 6 legs
            leg_frame = ttk.Frame(self.parlay_notebook)
            self.parlay_notebook.add(leg_frame, text=f"{leg_count}-Leg")
            self.parlay_trees[leg_count] = self.setup_parlay_tree_dynamic(leg_frame)
    
    def setup_parlay_tree_dynamic(self, parent_frame):
        """Setup a parlay tree - returns the tree widget"""
        columns = ('Parlay ID', 'Legs', 'Combined Prob', 'Payout', 'Expected Value', 'Leg Details')
        tree = ttk.Treeview(parent_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            tree.heading(col, text=col)
            if col == 'Leg Details':
                tree.column(col, width=400)
            else:
                tree.column(col, width=100)
        
        # Scrollbars
        scrollbar_y = ttk.Scrollbar(parent_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar_x = ttk.Scrollbar(parent_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        return tree
    
    def setup_team_analysis_tab(self):
        """Setup team analysis tab"""
        # Team analysis text
        analysis_frame = ttk.LabelFrame(self.team_analysis_frame, text="Team Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.team_analysis_text = tk.Text(analysis_frame, wrap=tk.WORD)
        team_scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, command=self.team_analysis_text.yview)
        self.team_analysis_text.configure(yscrollcommand=team_scrollbar.set)
        
        self.team_analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        team_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    def load_nfl_data(self):
        """Load NFL data from sportsdata.io"""
        try:
            # Look for SPORTSDATA files specifically (they have detailed projections)
            nfl_files = [f for f in os.listdir(self.default_dir) 
                        if 'SPORTSDATA' in f and f.endswith('.csv')]
            
            if not nfl_files:
                messagebox.showwarning("Warning", 
                    "No SportsData.io files found!\n\n"
                    "Please load a file with detailed projections like:\n"
                    "- nfl_week7_CASH_SPORTSDATA.csv\n"
                    "- nfl_week7_GPP_SPORTSDATA.csv")
                return
            
            # Prefer CASH file, fallback to first SPORTSDATA file
            if any('CASH' in f for f in nfl_files):
                latest_nfl_file = [f for f in nfl_files if 'CASH' in f][0]
            else:
                latest_nfl_file = nfl_files[0]
            
            nfl_path = os.path.join(self.default_dir, latest_nfl_file)
            self.nfl_data_df = pd.read_csv(nfl_path)
            
            # Verify required columns exist
            required_cols = ['PassingYards', 'RushingYards', 'ReceivingYards']
            missing_cols = [col for col in required_cols if col not in self.nfl_data_df.columns]
            
            if missing_cols:
                messagebox.showerror("Error", 
                    f"File missing required projection columns:\n{', '.join(missing_cols)}\n\n"
                    f"Please load a SportsData.io file with detailed projections.")
                self.nfl_data_df = None
                return
            
            self.status_label.config(text=f"Loaded NFL Data: {latest_nfl_file}", foreground="green")
            self.update_data_summary()
            self.populate_player_tree()
            self.update_team_list()
            
            print(f"✅ Successfully loaded {len(self.nfl_data_df)} players from {latest_nfl_file}")
            print(f"📋 Available columns: {self.nfl_data_df.columns.tolist()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NFL data: {str(e)}")
            self.status_label.config(text="Error loading data", foreground="red")
    
    def update_data_summary(self):
        """Update the data summary text"""
        if self.nfl_data_df is None:
            return
        
        summary = f"""NFL Data Summary:
        
Total Players: {len(self.nfl_data_df)}
Teams: {', '.join(sorted(self.nfl_data_df['Team'].unique()))}
Positions: {', '.join(sorted(self.nfl_data_df['Position'].unique()))}

Top 5 Players by Projected Points:
"""
        
        # Get top 5 players by projected points
        if 'Predicted_DK_Points' in self.nfl_data_df.columns:
            top_players = self.nfl_data_df.nlargest(5, 'Predicted_DK_Points')
            for _, player in top_players.iterrows():
                summary += f"• {player['Name']} ({player['Position']}, {player['Team']}) - {player['Predicted_DK_Points']:.1f} pts\n"
        
        self.data_summary_text.delete(1.0, tk.END)
        self.data_summary_text.insert(1.0, summary)
    
    def populate_player_tree(self):
        """Populate the player data tree"""
        if self.nfl_data_df is None:
            return
        
        # Clear existing items
        for item in self.player_tree.get_children():
            self.player_tree.delete(item)
        
        # Add players to tree
        for _, player in self.nfl_data_df.iterrows():
            values = (
                player['Name'],
                player['Position'],
                player['Team'],
                player.get('Opponent', 'N/A'),
                f"{player.get('Predicted_DK_Points', 0):.1f}",
                f"${player.get('Salary', 0):,}"
            )
            self.player_tree.insert('', 'end', values=values)
    
    def update_team_list(self):
        """Update the team list in filters"""
        if self.nfl_data_df is None:
            return
        
        teams = ['All'] + sorted(self.nfl_data_df['Team'].unique())
        self.team_combo['values'] = teams
        self.available_teams = teams[1:]  # Exclude 'All'
    
    def generate_nfl_props(self):
        """Generate NFL prop bets from the data"""
        if self.nfl_data_df is None:
            print("Warning: Please load NFL data first")
            return
        
        self.prop_bets = []
        print(f"Generating props from {len(self.nfl_data_df)} players...")
        
        for _, player in self.nfl_data_df.iterrows():
            name = player['Name']
            team = player['Team']
            position = player['Position']
            opponent = player.get('Opponent', 'N/A')
            
            # Skip if missing essential data
            if pd.isna(name) or pd.isna(team) or pd.isna(position):
                continue
            
            print(f"Processing {name} ({position}, {team})...")
            
            # QB Props
            if position == 'QB':
                if 'PassingYards' in player and not pd.isna(player['PassingYards']):
                    yards = player['PassingYards']
                    if yards > 0:
                        multiplier = self.get_multiplier('Passing Yards', yards)
                        prob = self.calculate_probability(
                            yards, round(yards), 
                            prop_type='Passing Yards',
                            position='QB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Passing Yards', 'line': round(yards),
                            'projection': yards, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Passing Yards prop: {yards}")
                
                if 'PassingTouchdowns' in player and not pd.isna(player['PassingTouchdowns']):
                    tds = player['PassingTouchdowns']
                    if tds > 0:
                        multiplier = self.get_multiplier('Passing TDs', tds)
                        prob = self.calculate_probability(
                            tds, round(tds, 1),
                            prop_type='Passing TDs',
                            position='QB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Passing TDs', 'line': round(tds, 1),
                            'projection': tds, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Passing TDs prop: {tds}")
                
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    rush_yards = player['RushingYards']
                    if rush_yards > 5:
                        multiplier = self.get_multiplier('Rushing Yards', rush_yards)
                        prob = self.calculate_probability(
                            rush_yards, round(rush_yards),
                            prop_type='Rushing Yards',
                            position='QB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Rushing Yards', 'line': round(rush_yards),
                            'projection': rush_yards, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Rushing Yards prop: {rush_yards}")
            
            # RB Props
            elif position == 'RB':
                if 'RushingYards' in player and not pd.isna(player['RushingYards']):
                    yards = player['RushingYards']
                    if yards > 5:  # Lowered threshold
                        multiplier = self.get_multiplier('Rushing Yards', yards)
                        prob = self.calculate_probability(
                            yards, round(yards),
                            prop_type='Rushing Yards',
                            position='RB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Rushing Yards', 'line': round(yards),
                            'projection': yards, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Rushing Yards prop: {yards}")
                
                if 'RushingTouchdowns' in player and not pd.isna(player['RushingTouchdowns']):
                    tds = player['RushingTouchdowns']
                    if tds > 0:
                        multiplier = self.get_multiplier('Rushing TDs', tds)
                        prob = self.calculate_probability(
                            tds, round(tds, 1),
                            prop_type='Rushing TDs',
                            position='RB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Rushing TDs', 'line': round(tds, 1),
                            'projection': tds, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Rushing TDs prop: {tds}")
                
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    rec_yards = player['ReceivingYards']
                    if rec_yards > 0:
                        multiplier = self.get_multiplier('Receiving Yards', rec_yards)
                        prob = self.calculate_probability(
                            rec_yards, round(rec_yards),
                            prop_type='Receiving Yards',
                            position='RB',
                            player_name=name,
                            multiplier=multiplier
                        )
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receiving Yards', 'line': round(rec_yards),
                            'projection': rec_yards, 'probability': prob,
                            'multiplier': multiplier
                        })
                        print(f"  Added Receiving Yards prop: {rec_yards}")
            
            # WR Props
            elif position == 'WR':
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    if yards > 5:  # Lowered threshold
                        prob = self.calculate_probability(yards, round(yards))
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receiving Yards', 'line': round(yards),
                            'projection': yards, 'probability': prob,
                            'multiplier': self.get_multiplier('Receiving Yards', yards)
                        })
                        print(f"  Added Receiving Yards prop: {yards}")
                
                if 'ReceivingTouchdowns' in player and not pd.isna(player['ReceivingTouchdowns']):
                    tds = player['ReceivingTouchdowns']
                    if tds > 0:
                        prob = self.calculate_probability(tds, round(tds, 1))
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receiving TDs', 'line': round(tds, 1),
                            'projection': tds, 'probability': prob,
                            'multiplier': self.get_multiplier('Receiving TDs', tds)
                        })
                        print(f"  Added Receiving TDs prop: {tds}")
                
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    if rec > 0:
                        prob = self.calculate_probability(rec, round(rec, 1))
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receptions', 'line': round(rec, 1),
                            'projection': rec, 'probability': prob,
                            'multiplier': self.get_multiplier('Receptions', rec)
                        })
                        print(f"  Added Receptions prop: {rec}")
            
            # TE Props
            elif position == 'TE':
                if 'ReceivingYards' in player and not pd.isna(player['ReceivingYards']):
                    yards = player['ReceivingYards']
                    if yards > 0:
                        prob = self.calculate_probability(yards, round(yards))
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receiving Yards', 'line': round(yards),
                            'projection': yards, 'probability': prob,
                            'multiplier': self.get_multiplier('Receiving Yards', yards)
                        })
                        print(f"  Added Receiving Yards prop: {yards}")
                
                if 'Receptions' in player and not pd.isna(player['Receptions']):
                    rec = player['Receptions']
                    if rec > 0:
                        prob = self.calculate_probability(rec, round(rec, 1))
                        self.prop_bets.append({
                            'player': name, 'team': team, 'opponent': opponent,
                            'prop': 'Receptions', 'line': round(rec, 1),
                            'projection': rec, 'probability': prob,
                            'multiplier': self.get_multiplier('Receptions', rec)
                        })
                        print(f"  Added Receptions prop: {rec}")
        
        print(f"Generated {len(self.prop_bets)} total prop bets")
        self.populate_props_tree()
        self.status_label.config(text=f"Generated {len(self.prop_bets)} prop bets", foreground="green")
    
    def load_underdog_props(self):
        """Load real Underdog Fantasy props"""
        try:
            print("🌐 Loading Underdog Fantasy props...")
            self.underdog_props = self.underdog_client.get_nfl_props()
            print(f"✅ Loaded {len(self.underdog_props)} Underdog props")
            
            # Load historical data for accurate probabilities
            self.load_historical_data()
            
        except Exception as e:
            print(f"⚠️  Error loading Underdog props: {e}")
            self.underdog_props = []
    
    def load_historical_data(self):
        """Load historical player data for accurate probability calculations"""
        import pandas as pd
        import json
        
        try:
            cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nfl_historical_cache"
            
            # Load config
            config_file = f"{cache_dir}/config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    print(f"✅ Historical data available: {config['games_analyzed']} games analyzed")
            
            # Load player database
            player_db_file = f"{cache_dir}/player_consistency_database.csv"
            if os.path.exists(player_db_file):
                self.player_database = pd.read_csv(player_db_file)
                print(f"✅ Loaded {len(self.player_database)} players with consistency data")
            else:
                self.player_database = None
                print("⚠️  No player database found. Run: python setup_historical_data.py")
            
            # Load hit rates
            hit_rates_file = f"{cache_dir}/hit_rates.json"
            if os.path.exists(hit_rates_file):
                with open(hit_rates_file, 'r') as f:
                    self.historical_hit_rates = json.load(f)
                    print(f"✅ Loaded historical hit rates")
            else:
                self.historical_hit_rates = {}
                
        except Exception as e:
            print(f"⚠️  Could not load historical data: {e}")
            self.player_database = None
            self.historical_hit_rates = {}
    
    def calculate_probability(self, projection, line, prop_type='Passing Yards', 
                             position='QB', player_name=None, multiplier=None):
        """
        Calculate probability using advanced engine with historical data.
        
        Falls back to simple calculation if historical data not available.
        """
        # Try to get player consistency if we have historical data
        player_consistency = None
        if self.player_database is not None and player_name:
            player_row = self.player_database[self.player_database['Player'] == player_name]
            if not player_row.empty:
                # Get consistency for this specific stat
                stat_col_map = {
                    'Passing Yards': 'PassingYards_consistency',
                    'Passing TDs': 'PassingTouchdowns_consistency',
                    'Rushing Yards': 'RushingYards_consistency',
                    'Rushing TDs': 'RushingTouchdowns_consistency',
                    'Receiving Yards': 'ReceivingYards_consistency',
                    'Receiving TDs': 'ReceivingTouchdowns_consistency',
                    'Receptions': 'Receptions_consistency'
                }
                
                consistency_col = stat_col_map.get(prop_type)
                if consistency_col and consistency_col in player_row.columns:
                    player_consistency = player_row[consistency_col].iloc[0]
        
        # Use advanced probability engine
        try:
            result = self.prob_engine.calculate_combined_probability(
                projection=projection,
                line=line,
                prop_type=prop_type,
                position=position,
                multiplier=multiplier,
                player_consistency=player_consistency
            )
            return result['probability']
            
        except Exception as e:
            print(f"⚠️  Probability engine error: {e}, using fallback")
            # Fallback to simple calculation
        if line == 0:
            return 0.5
        
        diff = projection - line
        if diff > 0:
            prob = min(0.85, 0.5 + (diff / abs(line)) * 0.3)
        else:
            prob = max(0.15, 0.5 - (abs(diff) / abs(line)) * 0.3)
        
        return prob
    
    def get_multiplier(self, prop_type, value):
        """Get appropriate multiplier for prop type"""
        if 'TDs' in prop_type:
            if value >= 2.0:
                return '20x'
            elif value >= 1.5:
                return '10x'
            elif value >= 1.0:
                return '6x'
            else:
                return '3x'
        elif 'Yards' in prop_type:
            if value >= 100:
                return '20x'
            elif value >= 75:
                return '10x'
            elif value >= 50:
                return '6x'
            else:
                return '3x'
        else:
            return '3x'
    
    def populate_props_tree(self):
        """Populate the props tree"""
        # Clear existing items
        for item in self.props_tree.get_children():
            self.props_tree.delete(item)
        
        # Add props to tree
        for prop in self.prop_bets:
            values = (
                prop['player'],
                prop['team'],
                prop['prop'],
                prop['line'],
                f"{prop['projection']:.1f}",
                f"{prop['probability']:.1%}",
                prop['multiplier']
            )
            self.props_tree.insert('', 'end', values=values)
    
    def apply_prop_filter(self):
        """Apply filters to prop bets"""
        # This would filter the props based on selected criteria
        # For now, just refresh the display
        self.populate_props_tree()
    
    def build_parlays(self):
        """Build parlay combinations"""
        if not self.prop_bets:
            messagebox.showwarning("Warning", "Please generate prop bets first")
            return
        
        try:
            max_legs = int(self.max_legs_var.get())
            min_prob = float(self.min_prob_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for max legs and min probability")
            return
        
        # Filter high-probability props and limit to top props
        eligible_props = [p for p in self.prop_bets if p['probability'] >= min_prob]
        
        # Sort by probability and take only top 30 props to avoid exponential explosion
        eligible_props.sort(key=lambda x: x['probability'], reverse=True)
        eligible_props = eligible_props[:30]  # Limit to top 30 props
        
        if len(eligible_props) < 2:
            messagebox.showwarning("Warning", "Not enough eligible props for parlays")
            return
        
        print(f"Building parlays from {len(eligible_props)} top props...")
        self.status_label.config(text=f"Building parlays from {len(eligible_props)} props...", foreground="orange")
        self.root.update()  # Force GUI update
        
        # Allocate parlays per leg count to ensure we get some of each type
        parlays_per_leg = 50  # Generate 50 best parlays per leg count
        max_combos_per_leg = 2000  # Check up to 2000 combinations per leg count
        
        # Generate parlays (simplified version)
        from itertools import combinations
        
        all_parlays_by_leg = {}  # Store separately by leg count
        
        for leg_count in range(2, min(max_legs + 1, len(eligible_props) + 1)):
            print(f"\n🎲 Building {leg_count}-leg parlays...")
            leg_parlays = []
            combo_count = 0
            
            for combo in combinations(eligible_props, leg_count):
                combo_count += 1
                
                # Break early if we've checked too many combinations for this leg count
                if combo_count > max_combos_per_leg:
                    print(f"  Checked {max_combos_per_leg} combinations for {leg_count}-leg parlays")
                    break
                
                combined_prob = 1.0
                for prop in combo:
                    combined_prob *= prop['probability']
                
                payout_multiplier = 1.0
                for prop in combo:
                    implied_odds = 1 / prop['probability'] if prop['probability'] > 0 else 1
                    payout_multiplier *= implied_odds
                
                expected_value = combined_prob * payout_multiplier
                
                if expected_value > 1.0:
                    leg_details = " + ".join([f"{p['player']} {p['prop']} {p['line']}" for p in combo])
                    leg_parlays.append({
                        'leg_count': leg_count,
                        'combined_probability': combined_prob,
                        'payout_multiplier': payout_multiplier,
                        'expected_value': expected_value,
                        'leg_details': leg_details
                    })
            
            # Sort by expected value and keep top N for this leg count
            leg_parlays.sort(key=lambda x: x['expected_value'], reverse=True)
            leg_parlays = leg_parlays[:parlays_per_leg]
            
            all_parlays_by_leg[leg_count] = leg_parlays
            print(f"  ✅ Generated {len(leg_parlays)} {leg_count}-leg parlays")
        
        # Combine all parlays
        self.parlays = []
        for leg_count, parlays in all_parlays_by_leg.items():
            self.parlays.extend(parlays)
        
        # Sort all parlays by expected value
        self.parlays.sort(key=lambda x: x['expected_value'], reverse=True)
        
        print(f"Generated {len(self.parlays)} parlays")
        self.populate_parlays_tree()
        self.status_label.config(text=f"Built {len(self.parlays)} parlays", foreground="green")
    
    def populate_parlays_tree(self):
        """Populate all parlay trees subdivided by leg count"""
        # Clear all existing items
        for tree in self.parlay_trees.values():
            for item in tree.get_children():
                tree.delete(item)
        
        # Organize parlays by leg count (up to 6 legs)
        parlays_by_legs = {i: [] for i in range(2, 7)}  # 2, 3, 4, 5, 6 legs
        
        for parlay in self.parlays:
            leg_count = parlay['leg_count']
            if leg_count in parlays_by_legs:
                parlays_by_legs[leg_count].append(parlay)
        
        # Update summary text
        self.update_parlay_summary(parlays_by_legs)
        
        # Populate All Parlays tab
        for i, parlay in enumerate(self.parlays, 1):
            values = (
                f"Parlay_{i}",
                f"{parlay['leg_count']}-Leg",
                f"{parlay['combined_probability']:.1%}",
                f"{parlay['payout_multiplier']:.1f}x",
                f"{parlay['expected_value']:.2f}",
                parlay['leg_details']
            )
            self.parlay_trees['all'].insert('', 'end', values=values)
        
        # Populate individual leg count tabs
        for leg_count in range(2, 7):  # 2-6 legs
            for i, parlay in enumerate(parlays_by_legs[leg_count], 1):
                values = (
                    f"{leg_count}Leg_{i}",
                    str(leg_count),
                    f"{parlay['combined_probability']:.1%}",
                    f"{parlay['payout_multiplier']:.1f}x",
                    f"{parlay['expected_value']:.2f}",
                    parlay['leg_details']
                )
                self.parlay_trees[leg_count].insert('', 'end', values=values)
        
        # Update status with counts
        count_parts = [f"All: {len(self.parlays)}"]
        for leg_count in range(2, 7):
            count = len(parlays_by_legs[leg_count])
            if count > 0:
                count_parts.append(f"{leg_count}-Leg: {count}")
        
        counts = " | ".join(count_parts)
        self.status_label.config(text=f"Built parlays - {counts}", foreground="green")
    
    def update_parlay_summary(self, parlays_by_legs):
        """Update the parlay summary text with counts and top parlays"""
        self.parlay_summary_text.configure(state='normal')
        self.parlay_summary_text.delete(1.0, tk.END)
        
        summary = f"{'='*80}\n"
        summary += f"PARLAY SUMMARY - Total: {len(self.parlays)} Parlays Generated\n"
        summary += f"{'='*80}\n\n"
        
        # Show summary for all leg counts (2-6)
        for leg_count in range(2, 7):
            parlays = parlays_by_legs.get(leg_count, [])
            count = len(parlays)
            
            if count > 0:
                # Get top parlay by EV
                top_parlay = max(parlays, key=lambda x: x['expected_value'])
                avg_prob = sum(p['combined_probability'] for p in parlays) / count
                avg_payout = sum(p['payout_multiplier'] for p in parlays) / count
                
                summary += f"📊 {leg_count}-LEG PARLAYS: {count} found\n"
                summary += f"   Avg Probability: {avg_prob:.1%}  |  Avg Payout: {avg_payout:.1f}x\n"
                summary += f"   Top Parlay: {top_parlay['combined_probability']:.1%} prob, "
                summary += f"{top_parlay['payout_multiplier']:.1f}x payout, EV: {top_parlay['expected_value']:.2f}\n"
                
                # Show leg details (truncate if too long)
                legs_preview = top_parlay['leg_details']
                if len(legs_preview) > 100:
                    legs_preview = legs_preview[:97] + "..."
                summary += f"   Example: {legs_preview}\n\n"
        
        summary += f"{'='*80}\n"
        summary += f"💡 TIP: Click tabs below to see all parlays organized by leg count\n"
        summary += f"💡 Set Max Legs to 5 or 6 to generate higher leg count parlays\n"
        
        self.parlay_summary_text.insert(1.0, summary)
        self.parlay_summary_text.configure(state='disabled')
    
    def filter_parlays_by_legs(self):
        """Filter parlays by selected leg count"""
        filter_value = self.leg_filter_var.get()
        
        # Switch to appropriate tab
        if filter_value == "All":
            self.parlay_notebook.select(0)
        elif filter_value == "2-Leg":
            self.parlay_notebook.select(1)
        elif filter_value == "3-Leg":
            self.parlay_notebook.select(2)
        elif filter_value == "4-Leg":
            self.parlay_notebook.select(3)
        elif filter_value == "5-Leg":
            self.parlay_notebook.select(4)
        elif filter_value == "6-Leg":
            self.parlay_notebook.select(5)
    
    def refresh_display(self):
        """Refresh all displays"""
        if self.nfl_data_df is not None:
            self.update_data_summary()
            self.populate_player_tree()
            self.update_team_list()
        
        if self.prop_bets:
            self.populate_props_tree()
        
        if self.parlays:
            self.populate_parlays_tree()
        
        self.status_label.config(text="Display refreshed", foreground="green")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = NFLUnderdogFantasyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
