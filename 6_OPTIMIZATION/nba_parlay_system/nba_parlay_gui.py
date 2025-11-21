#!/usr/bin/env python3
"""
NBA Parlay GUI
Interactive interface for generating NBA parlays using the 86.7% win rate model
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
import sys
import os.path

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from nba_markov_probabilities import apply_markov_adjustments
    _MARKOV_AVAILABLE = True
    print("✅ Markov probabilities available for NBA Parlay GUI")
except Exception as _e:
    _MARKOV_AVAILABLE = False
    print(f"⚠️ Markov probabilities not available in GUI: {_e}")

class NBAParlayGUI:
    """NBA Parlay Generator GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Parlay Generator - Points O/U Only (Top 35%)")
        self.root.geometry("1200x800")
        
        self.nba_data_df = None
        self.team_checkboxes = {}  # Store team checkboxes
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="🏀 NBA Parlay Generator", 
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = ttk.Label(header_frame, text="Points O/U Only - Top 35% Players", 
                                  font=("Arial", 10))
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status bar
        self.status_label = ttk.Label(header_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Load NBA Data", 
                  command=self.load_nba_data).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="Fetch Tonight's Data", 
                  command=self.fetch_tonight_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Data", 
                  command=self.generate_data).pack(side=tk.LEFT, padx=5)
        
        # Markov toggle
        self.use_markov_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Use Markov (3-yr history)", variable=self.use_markov_var).pack(side=tk.LEFT, padx=(15, 5))
        
        # Min average points filter
        ttk.Label(control_frame, text="Min Avg Pts:").pack(side=tk.LEFT, padx=(15, 5))
        self.min_avg_pts_var = tk.StringVar(value="10")
        min_avg_entry = ttk.Entry(control_frame, textvariable=self.min_avg_pts_var, width=5)
        min_avg_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Legs selection
        ttk.Label(control_frame, text="Legs:").pack(side=tk.LEFT, padx=(20, 5))
        self.legs_var = tk.StringVar(value="4")
        legs_combo = ttk.Combobox(control_frame, textvariable=self.legs_var, 
                                  values=["2", "3", "4"], width=5, state="readonly")
        legs_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Generate Parlays", 
                  command=self.generate_parlays).pack(side=tk.LEFT, padx=(10, 5))
        
        ttk.Button(control_frame, text="Clear", 
                  command=self.clear_parlays).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="List All Props", 
                  command=self.list_all_props).pack(side=tk.LEFT, padx=(10, 5))
        
        # Team selection frame
        self.team_frame = ttk.LabelFrame(main_frame, text="Select Teams to Use")
        self.team_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Data summary
        summary_frame = ttk.LabelFrame(main_frame, text="Data Summary")
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=4, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        self.summary_text.insert(1.0, "No data loaded. Click 'Load NBA Data' or 'Generate Data' to start.")
        self.summary_text.config(state=tk.DISABLED)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Generated Parlays")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for parlay display
        columns = ('Player', 'Team', 'Prop', 'Line', 'Bet Type', 'Hit Rate')
        self.parlay_tree = ttk.Treeview(results_frame, columns=columns, show='tree headings', height=25)
        
        # Configure columns
        self.parlay_tree.heading('#0', text='Parlay #', anchor='w')
        self.parlay_tree.column('#0', width=100, stretch=False)
        
        for col in columns:
            self.parlay_tree.heading(col, text=col)
            if col == 'Hit Rate':
                self.parlay_tree.column(col, width=80, stretch=False)
            elif col == 'Bet Type':
                self.parlay_tree.column(col, width=80, stretch=False)
            elif col == 'Line':
                self.parlay_tree.column(col, width=80, stretch=False)
            elif col == 'Team':
                self.parlay_tree.column(col, width=60, stretch=False)
            else:
                self.parlay_tree.column(col, width=120)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.parlay_tree.yview)
        self.parlay_tree.configure(yscrollcommand=scrollbar.set)
        
        self.parlay_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # All props frame
        props_frame = ttk.LabelFrame(main_frame, text="All Player Props (Best → Worst)")
        props_frame.pack(fill=tk.BOTH, expand=True)
        
        props_columns = ('Player', 'Team', 'Prop', 'Line', 'Bet Type', 'Hit Rate')
        self.props_tree = ttk.Treeview(props_frame, columns=props_columns, show='headings', height=18)
        for col in props_columns:
            self.props_tree.heading(col, text=col)
            if col == 'Hit Rate':
                self.props_tree.column(col, width=80, stretch=False)
            elif col == 'Bet Type':
                self.props_tree.column(col, width=80, stretch=False)
            elif col == 'Line':
                self.props_tree.column(col, width=80, stretch=False)
            elif col == 'Team':
                self.props_tree.column(col, width=60, stretch=False)
            else:
                self.props_tree.column(col, width=160)
        props_scrollbar = ttk.Scrollbar(props_frame, orient=tk.VERTICAL, command=self.props_tree.yview)
        self.props_tree.configure(yscrollcommand=props_scrollbar.set)
        self.props_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        props_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_nba_data(self):
        """Load NBA data from file"""
        file_path = filedialog.askopenfilename(
            title="Select NBA Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.nba_data_df = pd.read_csv(file_path)
            self._apply_markov_if_enabled()
            self._normalize_for_generator()
            self.update_summary()
            self.status_label.config(text=f"Loaded {len(self.nba_data_df)} players", foreground="green")
            messagebox.showinfo("Success", f"Loaded {len(self.nba_data_df)} NBA players")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_label.config(text="Error loading data", foreground="red")
    
    def fetch_tonight_data(self):
        """Fetch real NBA data for tonight's games"""
        try:
            import sys
            import os
            from datetime import datetime
            
            # Add path to NBA fetcher
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from nba_sportsdata_fetcher import NBADataFetcher
            
            # API key from test file
            API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
            
            self.status_label.config(text="Fetching NBA data...", foreground="blue")
            
            # Initialize fetcher
            fetcher = NBADataFetcher(API_KEY)
            
            # Get tonight's date (Oct 25, 2025)
            tonight = "2025-OCT-25"
            print(f"📡 Fetching NBA projections for {tonight}...")
            
            # Fetch projections
            projections = fetcher.get_daily_projections(tonight)
            
            # Debug: show what columns we got
            print(f"Columns in fetched data: {projections.columns.tolist()}")
            
            if projections.empty:
                messagebox.showwarning("No Games", f"No games scheduled for {tonight}\nUsing synthetic data instead.")
                self.generate_data()
                return
            
            # Rename columns to match expected format
            # The fetcher already renamed some columns to ProjectedPoints, ProjectedRebounds, etc.
            column_mapping = {
                'Name': 'player_name_proj',
                'Team': 'team_proj',
                'Position': 'position_proj',
                'ID': 'player_id',
                'PlayerID': 'player_id',
                'GlobalPlayerID': 'player_id',
                'ProjectedPoints': 'projected_points',
                'ProjectedRebounds': 'projected_rebounds',
                'ProjectedAssists': 'projected_assists',
                'ProjectedSteals': 'projected_steals',
                'ProjectedBlocks': 'projected_blocks',
                'ThreePointersMade': 'projected_three_pointers',
                'Predicted_DK_Points': 'projected_dk_points'
            }
            
            # Only rename columns that exist
            existing_cols = {k: v for k, v in column_mapping.items() if k in projections.columns}
            projections = projections.rename(columns=existing_cols)
            
            # Try to load historical variance from training data
            training_file = os.path.join(os.path.dirname(__file__), 'nba_training_data.csv')
            if os.path.exists(training_file):
                try:
                    training_df = pd.read_csv(training_file)
                    # Get average variance by player position from training data
                    if len(training_df) > 0:
                        variance_cols = ['points_accuracy_std', 'rebounds_accuracy_std', 'assists_accuracy_std',
                                        'steals_accuracy_std', 'blocks_accuracy_std', 'three_pointers_accuracy_std']
                        hit_cols = ['points_hit_mean', 'rebounds_hit_mean', 'assists_hit_mean',
                                   'steals_hit_mean', 'blocks_hit_mean', 'three_pointers_hit_mean']
                        
                        # Get average variance by position
                        pos_col = 'position_proj' if 'position_proj' in training_df.columns else 'Position'
                        if pos_col in training_df.columns:
                            for _, player in projections.iterrows():
                                pos = player.get('position_proj', 'SF')
                                pos_data = training_df[training_df[pos_col] == pos]
                                
                                if len(pos_data) > 0:
                                    for col in variance_cols:
                                        if col in pos_data.columns:
                                            avg_var = pos_data[col].mean()
                                            projections.loc[projections.index == _.index, col] = avg_var
                                    
                                    for col in hit_cols:
                                        if col in pos_data.columns:
                                            avg_hit = pos_data[col].mean()
                                            projections.loc[projections.index == _.index, col] = avg_hit
                        
                        print("✅ Loaded historical variance from training data")
                except Exception as e:
                    print(f"⚠️ Could not load training data: {e}")
            
            # Add default accuracy columns if they don't exist
            stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'three_pointers']
            accuracy_values = [0.25, 0.20, 0.20, 0.25, 0.30, 0.35]
            hit_mean_values = [0.70, 0.75, 0.75, 0.68, 0.65, 0.60]
            
            for stat, acc_std, hit_mean in zip(stats, accuracy_values, hit_mean_values):
                if f'{stat}_accuracy_std' not in projections.columns:
                    projections[f'{stat}_accuracy_std'] = acc_std
                if f'{stat}_hit_mean' not in projections.columns:
                    projections[f'{stat}_hit_mean'] = hit_mean
            
            # Fill missing values
            numeric_cols = [col for col in projections.columns if col.startswith('projected_')]
            for col in numeric_cols:
                if col in projections.columns:
                    projections[col] = projections[col].fillna(0)
            
            # Filter out players with 0 projections (injured/not playing)
            if 'projected_points' in projections.columns:
                projections = projections[projections['projected_points'] > 0].copy()
            
            print(f"After processing, columns: {projections.columns.tolist()}")
            print(f"Number of rows: {len(projections)}")
            
            # Show sample of actual projected values
            if 'projected_points' in projections.columns:
                print(f"\nSample of projected points:")
                sample = projections[['player_name_proj', 'team_proj', 'projected_points', 'projected_rebounds', 'projected_assists']].head(10)
                print(sample.to_string())
            
            print("✅ Successfully loaded REAL NBA data!")
            
            self.nba_data_df = projections
            self._apply_markov_if_enabled()
            self._normalize_for_generator()
            self.update_summary()
            
            self.status_label.config(text=f"Fetched {len(projections)} players", foreground="green")
            messagebox.showinfo("Success", f"Fetched {len(projections)} NBA players for tonight!")
            
        except Exception as e:
            print(f"❌ Exception in fetch_tonight_data: {e}")
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")
            self.status_label.config(text="Error fetching data", foreground="red")
    
    def generate_data(self):
        """Generate synthetic NBA data"""
        try:
            import numpy as np
            import random
            
            print("🏀 Generating synthetic NBA data...")
            
            nba_data = []
            # All NBA teams
            teams = ['LAL', 'GSW', 'BOS', 'MIA', 'MIL', 'PHI', 'BKN', 'NYK', 'DEN', 'POR', 'DAL', 'HOU', 'PHX', 'SAC', 'ATL']
            positions = ['PG', 'SG', 'SF', 'PF', 'C']
            
            # Generate ~20 players per team
            for player_id in range(300):
                proj_points = max(0, np.random.normal(15, 5))
                proj_rebounds = max(0, np.random.normal(6, 2))
                proj_assists = max(0, np.random.normal(5, 2))
                proj_steals = max(0, np.random.normal(1.2, 0.5))
                proj_blocks = max(0, np.random.normal(0.8, 0.4))
                proj_threes = max(0, np.random.normal(2.5, 1.0))
                proj_dk = max(0, np.random.normal(25, 8))
                
                # Create realistic player names (proper pairs)
                player_name_pairs = [
                    ('LeBron', 'James'), ('Stephen', 'Curry'), ('Kevin', 'Durant'), ('James', 'Harden'),
                    ('Russell', 'Westbrook'), ('Damian', 'Lillard'), ('Devin', 'Booker'), ('Luka', 'Doncic'),
                    ('Jayson', 'Tatum'), ('Joel', 'Embiid'), ('Giannis', 'Antetokounmpo'), ('Kawhi', 'Leonard'),
                    ('Paul', 'George'), ('Jimmy', 'Butler'), ('Trae', 'Young'), ('Jaylen', 'Brown'),
                    ('Bradley', 'Beal'), ('Zion', 'Williamson'), ('Ja', 'Morant'), ('Karl-Anthony', 'Towns'),
                    ('De\'Aaron', 'Fox'), ('Shai', 'Gilgeous-Alexander'), ('Deandre', 'Ayton'), ('Brandon', 'Ingram'),
                    ('Bam', 'Adebayo'), ('Kristaps', 'Porzingis'), ('Tobias', 'Harris'), ('Myles', 'Turner')
                ]
                
                first_name, last_name = random.choice(player_name_pairs)
                
                player_data = {
                    'player_id': f'player_{player_id}',
                    'player_name_proj': f'{first_name} {last_name}',
                    'team_proj': random.choice(teams),
                    'position_proj': random.choice(positions),
                    'week': random.randint(1, 4),
                    'year': random.choice([2023, 2024, 2025]),
                    'projected_points': proj_points,
                    'projected_rebounds': proj_rebounds,
                    'projected_assists': proj_assists,
                    'projected_steals': proj_steals,
                    'projected_blocks': proj_blocks,
                    'projected_three_pointers': proj_threes,
                    'projected_dk_points': proj_dk,
                    'actual_points': max(0, np.random.normal(proj_points, proj_points * 0.25)),
                    'actual_rebounds': max(0, np.random.normal(proj_rebounds, proj_rebounds * 0.20)),
                    'actual_assists': max(0, np.random.normal(proj_assists, proj_assists * 0.20)),
                    'actual_steals': max(0, np.random.normal(proj_steals, proj_steals * 0.25)),
                    'actual_blocks': max(0, np.random.normal(proj_blocks, proj_blocks * 0.30)),
                    'actual_three_pointers': max(0, np.random.normal(proj_threes, proj_threes * 0.35)),
                    'actual_dk_points': max(0, np.random.normal(proj_dk, proj_dk * 0.25)),
                    'points_accuracy_std': 0.25,
                    'rebounds_accuracy_std': 0.20,
                    'assists_accuracy_std': 0.20,
                    'steals_accuracy_std': 0.25,
                    'blocks_accuracy_std': 0.30,
                    'three_pointers_accuracy_std': 0.35,
                    'dk_points_accuracy_std': 0.25,
                    'points_hit_mean': 0.70,
                    'rebounds_hit_mean': 0.75,
                    'assists_hit_mean': 0.75,
                    'steals_hit_mean': 0.68,
                    'blocks_hit_mean': 0.65,
                    'three_pointers_hit_mean': 0.60,
                    'dk_points_hit_mean': 0.70
                }
                
                nba_data.append(player_data)
            
            self.nba_data_df = pd.DataFrame(nba_data)
            self._apply_markov_if_enabled()
            self._normalize_for_generator()
            self.update_summary()
            self.status_label.config(text=f"Generated {len(self.nba_data_df)} players", foreground="green")
            messagebox.showinfo("Success", f"Generated {len(self.nba_data_df)} NBA players")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
            self.status_label.config(text="Error generating data", foreground="red")
    
    def _apply_markov_if_enabled(self):
        """Apply Markov chain adjustments using 3-year historical cache if enabled."""
        try:
            if not self.use_markov_var.get():
                return
            if not _MARKOV_AVAILABLE:
                return
            if self.nba_data_df is None or len(self.nba_data_df) == 0:
                return
            
            df = self.nba_data_df.copy()
            
            # Ensure columns expected by the Markov module
            if 'Predicted_DK_Points' not in df.columns:
                if 'projected_dk_points' in df.columns:
                    df['Predicted_DK_Points'] = df['projected_dk_points']
                elif 'projected_points' in df.columns:
                    df['Predicted_DK_Points'] = df['projected_points']
            
            if 'Name' not in df.columns:
                if 'player_name_proj' in df.columns:
                    df['Name'] = df['player_name_proj']
                elif 'Name' in df.columns:
                    pass
                else:
                    # Without a name/ID to match, skip
                    self.nba_data_df = df
                    return
            
            cache_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_historical_cache"
            before_cols = set(df.columns)
            df = apply_markov_adjustments(
                df_players=df,
                history_df=None,
                cache_dir=cache_dir,
                blend_alpha=0.20,
                min_games=30,
                player_thresholds=(20.0, 25.0, 30.0),
            )
            added = sorted(list(set(df.columns) - before_cols))
            if any(c.startswith('MC_') for c in added):
                print(f"✅ GUI: Markov adjustments applied. Added: {added}")
            else:
                print("ℹ️ GUI: Markov attempted but cache not found; using base projections.")
            
            # Propagate blended projection back to GUI columns
            if 'Predicted_DK_Points' in df.columns:
                df['projected_points'] = df.get('projected_points', df['Predicted_DK_Points'])
                df['projected_points'] = df['Predicted_DK_Points']
                df['projected_dk_points'] = df['Predicted_DK_Points']
            
            self.nba_data_df = df
        except Exception as e:
            print(f"⚠️ GUI: Skipped Markov adjustments due to error: {e}")
    
    def _normalize_for_generator(self):
        """Ensure required columns exist for the parlay generator."""
        if self.nba_data_df is None or len(self.nba_data_df) == 0:
            return
        df = self.nba_data_df.copy()
        
        # Ensure player_id
        if 'player_id' not in df.columns:
            if 'DK_ID' in df.columns:
                df['player_id'] = df['DK_ID'].astype(str)
            elif 'ID' in df.columns:
                df['player_id'] = df['ID'].astype(str)
            else:
                name_col = 'player_name_proj' if 'player_name_proj' in df.columns else ('Name' if 'Name' in df.columns else None)
                team_col = 'team_proj' if 'team_proj' in df.columns else ('Team' if 'Team' in df.columns else None)
                if name_col and team_col:
                    df['player_id'] = df[name_col].astype(str).str.strip() + '_' + df[team_col].astype(str).str.strip()
                elif name_col:
                    df['player_id'] = df[name_col].astype(str).str.strip()
                else:
                    df['player_id'] = df.index.astype(str)
        
        # Ensure name/position/team columns expected by generator
        if 'player_name_proj' not in df.columns and 'Name' in df.columns:
            df['player_name_proj'] = df['Name']
        if 'team_proj' not in df.columns and 'Team' in df.columns:
            df['team_proj'] = df['Team']
        if 'position_proj' not in df.columns:
            if 'Position' in df.columns:
                df['position_proj'] = df['Position']
            elif 'Roster_Position' in df.columns:
                import re as _re
                def _first_pos(s):
                    s = str(s)
                    m = _re.search(r'(PG|SG|SF|PF|C)', s)
                    return m.group(1) if m else 'G'
                df['position_proj'] = df['Roster_Position'].apply(_first_pos)
        
        # Ensure projections for generator
        if 'projected_points' not in df.columns:
            if 'Predicted_DK_Points' in df.columns:
                df['projected_points'] = df['Predicted_DK_Points']
            elif 'projected_dk_points' in df.columns:
                df['projected_points'] = df['projected_dk_points']
        
        if 'projected_dk_points' not in df.columns and 'projected_points' in df.columns:
            df['projected_dk_points'] = df['projected_points']
        
        self.nba_data_df = df
    
    def create_team_checkboxes(self, teams):
        """Create checkboxes for team selection"""
        # Clear existing checkboxes
        for widget in self.team_frame.winfo_children():
            widget.destroy()
        self.team_checkboxes.clear()
        
        if not teams:
            return
        
        # Create checkboxes in a grid
        row = 0
        col = 0
        max_cols = 8
        
        for team in teams:
            var = tk.BooleanVar(value=True)  # All teams selected by default
            checkbox = ttk.Checkbutton(self.team_frame, text=team, variable=var)
            checkbox.grid(row=row, column=col, padx=5, pady=2, sticky='w')
            self.team_checkboxes[team] = var
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def get_selected_teams(self):
        """Get list of selected teams"""
        return [team for team, var in self.team_checkboxes.items() if var.get()]
    
    def update_summary(self):
        """Update the data summary"""
        if self.nba_data_df is None:
            return
        
        # Get teams and positions (handle both formats)
        team_col = 'team_proj' if 'team_proj' in self.nba_data_df.columns else 'Team'
        pos_col = 'position_proj' if 'position_proj' in self.nba_data_df.columns else 'Position'
        
        teams = sorted(self.nba_data_df[team_col].unique()) if team_col in self.nba_data_df.columns else []
        positions = sorted(self.nba_data_df[pos_col].unique()) if pos_col in self.nba_data_df.columns else []
        
        # Create team checkboxes
        self.create_team_checkboxes(teams)
        
        summary = f"""NBA Data Summary:
        
Total Players: {len(self.nba_data_df)}
Teams: {', '.join(teams)}{'...' if len(teams) == 10 else ''}
Positions: {', '.join(positions)}

Top 5 Players by Projected Points:
"""
        
        if 'projected_points' in self.nba_data_df.columns:
            top_players = self.nba_data_df.nlargest(5, 'projected_points')
            for _, player in top_players.iterrows():
                name = player.get('player_name_proj', player.get('Name', 'Unknown'))
                pos = player.get('position_proj', player.get('Position', 'N/A'))
                team = player.get('team_proj', player.get('Team', 'N/A'))
                pts = player['projected_points']
                summary += f"• {name} ({pos}, {team}) - {pts:.1f} pts\n"
        
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
        self.summary_text.config(state=tk.DISABLED)
    
    def generate_parlays(self):
        """Generate NBA parlays"""
        if self.nba_data_df is None:
            messagebox.showerror("Error", "Please load or generate NBA data first")
            return
        
        # Get selected teams
        selected_teams = self.get_selected_teams()
        if not selected_teams:
            messagebox.showerror("Error", "Please select at least one team")
            return
        
        # Filter data by selected teams
        team_col = 'team_proj' if 'team_proj' in self.nba_data_df.columns else 'Team'
        filtered_data = self.nba_data_df[self.nba_data_df[team_col].isin(selected_teams)].copy()
        
        # Apply minimum average points filter
        try:
            min_avg = float(self.min_avg_pts_var.get().strip()) if self.min_avg_pts_var.get() else 10.0
        except Exception:
            min_avg = 10.0
        avg_series = self._get_average_points_series(filtered_data)
        if avg_series is not None:
            filtered_data = filtered_data[avg_series >= min_avg].copy()
        else:
            # Fallback: use projected_points if averages not available
            if 'projected_points' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['projected_points'] >= min_avg].copy()
        
        if len(filtered_data) == 0:
            messagebox.showerror("Error", f"No players found for selected teams: {', '.join(selected_teams)}")
            return
        
        # Clear existing items
        for item in self.parlay_tree.get_children():
            self.parlay_tree.delete(item)
        
        try:
            # Import NBA generator
            from nba_parlay_generator import NBAAdvancedParlayGenerator
            
            # Get legs from UI
            max_legs = int(self.legs_var.get())
            
            # Create generator with filtered data
            generator = NBAAdvancedParlayGenerator(filtered_data)
            
            # Generate parlays
            num_parlays = 15
            parlays = []
            for i in range(num_parlays):
                parlay = generator.generate_parlay(max_legs=max_legs)
                if parlay.legs:
                    parlays.append(parlay)
            
            if not parlays:
                messagebox.showinfo("No Parlays", "No valid parlays generated")
                return
            
            # Populate treeview
            for idx, parlay in enumerate(parlays, 1):
                parlay_id = f"P#{idx}"
                
                # Add parlay as parent node
                values = (f"{len(parlay.legs)} legs", f"{parlay.combined_hit_rate:.1%}", 
                         f"+{parlay.estimated_odds:.0f}", "", "", "")
                parent = self.parlay_tree.insert('', 'end', text=f"Parlay #{idx}", values=values)
                
                # Add legs as children
                for leg in parlay.legs:
                    # Skip DK points
                    if leg.prop_type == 'dk_points':
                        continue
                    
                    prop_display = leg.prop_type.replace('_', ' ').title()
                    
                    # Format the line
                    if leg.line >= 1:
                        line_str = f"{leg.line:.0f}"
                    else:
                        line_str = f"{leg.line:.1f}"
                    
                    leg_values = (
                        leg.player_name,
                        leg.team,
                        prop_display,
                        line_str,
                        leg.bet_type,
                        f"{leg.hit_rate:.0%}"
                    )
                    self.parlay_tree.insert(parent, 'end', text='', values=leg_values)
            
            # Update status
            avg_hit_rate = sum(p.combined_hit_rate for p in parlays) / len(parlays)
            self.status_label.config(text=f"Generated {len(parlays)} parlays (Avg Hit Rate: {avg_hit_rate:.1%})", 
                                   foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate parlays: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.status_label.config(text="Error generating parlays", foreground="red")
    
    def _get_average_points_series(self, df):
        """Return a Series of average points per player if present or derivable, else None."""
        if df is None or len(df) == 0:
            return None
        # Common average points columns from various sources
        candidates = [
            'AvgPointsPerGame', 'PointsPerGame', 'PPG', 'AveragePoints', 'AvgPoints',
            'SeasonAveragePoints', 'PointsPerGameAverage'
        ]
        for c in candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                if s.notna().any():
                    return s
        # Derive from totals if present
        totals_candidates = [
            ('SeasonPoints', 'Games'),
            ('TotalPoints', 'Games'),
            ('Points', 'Games')  # if Points is season total and Games present
        ]
        for total_col, games_col in totals_candidates:
            if total_col in df.columns and games_col in df.columns:
                total = pd.to_numeric(df[total_col], errors='coerce')
                games = pd.to_numeric(df[games_col], errors='coerce').replace(0, pd.NA)
                avg = total.divide(games)
                if avg.notna().any():
                    return avg
        return None
    
    def clear_parlays(self):
        """Clear the parlay tree"""
        for item in self.parlay_tree.get_children():
            self.parlay_tree.delete(item)
        self.status_label.config(text="Ready", foreground="black")
    
    def list_all_props(self):
        """List best OVER/UNDER for each player, sorted by hit rate."""
        if self.nba_data_df is None:
            messagebox.showerror("Error", "Please load or generate NBA data first")
            return
        
        # Teams filter
        selected_teams = self.get_selected_teams()
        if not selected_teams:
            messagebox.showerror("Error", "Please select at least one team")
            return
        team_col = 'team_proj' if 'team_proj' in self.nba_data_df.columns else 'Team'
        df = self.nba_data_df[self.nba_data_df[team_col].isin(selected_teams)].copy()
        
        # Min avg points filter
        try:
            min_avg = float(self.min_avg_pts_var.get().strip()) if self.min_avg_pts_var.get() else 10.0
        except Exception:
            min_avg = 10.0
        avg_series = self._get_average_points_series(df)
        if avg_series is not None:
            df = df[avg_series >= min_avg].copy()
        else:
            if 'projected_points' in df.columns:
                df = df[df['projected_points'] >= min_avg].copy()
        
        if len(df) == 0:
            messagebox.showinfo("No Players", "No players match filters.")
            return
        
        # Clear table
        for item in self.props_tree.get_children():
            self.props_tree.delete(item)
        
        # Compute best prop (points only) for each player
        results = []
        for _, player in df.iterrows():
            proj = float(player.get('projected_points', player.get('Predicted_DK_Points', 0)) or 0)
            if proj <= 0:
                continue
            best = self._best_prop_for_player(player, projection=proj)
            if best is None:
                continue
            results.append(best)
        
        # Sort by hit rate desc
        results.sort(key=lambda x: x['hit_rate'], reverse=True)
        
        # Populate
        for item in results:
            self.props_tree.insert('', 'end', values=(item['player_name'], item['team'], 'Points', item['line_str'], item['bet_type'], f"{item['hit_rate']:.0%}"))
        
        self.status_label.config(text=f"Listed {len(results)} props (sorted)", foreground="green")
    
    def _best_prop_for_player(self, player, projection: float):
        """Return best prop dict for a player across candidate lines and bet types."""
        name = player.get('player_name_proj', player.get('Name', 'Unknown'))
        team = player.get('team_proj', player.get('Team', 'N/A'))
        position = player.get('position_proj', player.get('Position', 'SF'))
        
        # Candidate line multipliers (same as generator)
        candidates = [0.50, 0.55, 0.60, 0.65, 0.70]
        best = None
        for m in candidates:
            line = projection * m
            # Round to 0.5 increments
            line = round(line * 2) / 2
            over_hr = self._compute_over_hit_rate(player, line, projection, position)
            under_hr = 1 - over_hr if over_hr is not None else None
            for bet_type, hr in [('OVER', over_hr), ('UNDER', under_hr)]:
                if hr is None:
                    continue
                record = {
                    'player_name': name,
                    'team': team,
                    'line': line,
                    'line_str': f"{line:.1f}" if line < 1 else f"{line:.0f}",
                    'bet_type': bet_type,
                    'hit_rate': max(0.0, min(1.0, float(hr)))
                }
                if best is None or record['hit_rate'] > best['hit_rate']:
                    best = record
        return best
    
    def _compute_over_hit_rate(self, player, line: float, projection: float, position: str) -> float:
        """Compute OVER hit rate using Markov if available, otherwise normal model."""
        import numpy as _np
        import pandas as _pd
        from math import isfinite as _isfinite
        
        # Use Markov expected value if available
        mean = player.get('MC_Expected', projection)
        try:
            mean = float(mean)
        except Exception:
            mean = projection
        
        # Std via accuracy std or default CV
        std = None
        acc_col = 'points_accuracy_std'
        if acc_col in player.index and _pd.notna(player.get(acc_col, _np.nan)):
            try:
                std = abs(float(player.get(acc_col))) * 1.5 * projection
            except Exception:
                std = None
        if std is None or not _isfinite(std) or std <= 0:
            cv = self._get_default_cv_improved(position, 'points')
            std = max(1e-6, float(cv) * projection)
        
        # If Markov probability exists for close thresholds, prefer it
        try:
            mc_cols = [c for c in player.index if isinstance(c, str) and c.startswith('MC_Prob_Over_')]
            if mc_cols:
                thresholds = []
                for c in mc_cols:
                    try:
                        t = float(str(c).split('_')[-1])
                        thresholds.append((t, c))
                    except Exception:
                        continue
                if thresholds:
                    closest_t, closest_col = min(thresholds, key=lambda x: abs(x[0] - float(line)))
                    if abs(float(closest_t) - float(line)) <= 0.6:
                        val = player.get(closest_col, _np.nan)
                        if _pd.notna(val) and 0.0 <= float(val) <= 1.0:
                            return float(val)
        except Exception:
            pass
        
        # Normal approximation as fallback
        try:
            from math import erf, sqrt
            z = (line - mean) / std
            # CDF of standard normal using erf
            cdf = 0.5 * (1 + erf(z / sqrt(2)))
            over_hr = 1 - cdf
            return float(over_hr)
        except Exception:
            return 0.5

def main():
    """Main function"""
    root = tk.Tk()
    app = NBAParlayGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

