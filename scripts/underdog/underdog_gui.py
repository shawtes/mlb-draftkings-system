"""
NFL Underdog Fantasy GUI - Display and Analyze NFL Predictions
Loads NFL prediction CSV files from sportsdata.io and displays results in an interactive interface.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import os
from datetime import datetime

class NFLUnderdogFantasyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NFL Underdog Fantasy Prediction Viewer - SportsData.io")
        self.root.geometry("1400x900")
        
        # Data storage
        self.predictions_df = None
        self.recommendations_df = None
        self.nfl_data_df = None
        
        # Team selection storage
        self.available_teams = []
        self.selected_teams = set()
        
        # Default directory for NFL data files
        self.default_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION"
        self.analysis_dir = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/7_ANALYSIS"
        
        self.setup_gui()
        self.load_latest_predictions()
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 Underdog Fantasy Prediction Viewer", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load buttons
        ttk.Button(control_frame, text="Load Latest Predictions", 
                  command=self.load_latest_predictions).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Load Custom File", 
                  command=self.load_custom_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Refresh", 
                  command=self.refresh_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Top Predictions
        self.top_predictions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.top_predictions_frame, text="Top Predictions")
        self.setup_top_predictions_tab()
        
        # Tab 2: Player Search
        self.player_search_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.player_search_frame, text="Player Search")
        self.setup_player_search_tab()
        
        # Tab 3: Recommendations
        self.recommendations_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendations_frame, text="Recommendations")
        self.setup_recommendations_tab()
        
        # Tab 4: Parlay Builder
        self.parlay_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.parlay_frame, text="Parlay Builder")
        self.setup_parlay_builder_tab()
        
        # Tab 5: Lineup Optimizer
        self.lineup_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.lineup_frame, text="Lineup Optimizer")
        self.setup_lineup_optimizer_tab()
        
        # Tab 6: Summary Stats
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        self.setup_summary_tab()
    
    def setup_top_predictions_tab(self):
        """Setup the top predictions tab"""
        # Filter frame
        filter_frame = ttk.LabelFrame(self.top_predictions_frame, text="Filters")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Prop type filter
        ttk.Label(filter_frame, text="Prop Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.prop_var = tk.StringVar(value="All")
        prop_combo = ttk.Combobox(filter_frame, textvariable=self.prop_var, width=20)
        prop_combo['values'] = ["All", "hits_over_0.5", "hits_over_1.5", "runs_over_0.5", 
                               "rbis_over_0.5", "hrs_over_0.5", "sbs_over_0.5", 
                               "total_bases_over_1.5", "hits_runs_rbis_over_1.5"]
        prop_combo.grid(row=0, column=1, padx=5, pady=5)
        prop_combo.bind('<<ComboboxSelected>>', self.filter_predictions)
        
        # Minimum probability filter
        ttk.Label(filter_frame, text="Min Probability:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.min_prob_var = tk.DoubleVar(value=0.0)
        prob_spin = ttk.Spinbox(filter_frame, from_=0.0, to=1.0, increment=0.1, 
                               textvariable=self.min_prob_var, width=10)
        prob_spin.grid(row=0, column=3, padx=5, pady=5)
        prob_spin.bind('<KeyRelease>', self.filter_predictions)
        
        # Results tree
        columns = ("Player", "Prop", "Probability", "Expected Value", "Confidence")
        self.top_tree = ttk.Treeview(self.top_predictions_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            self.top_tree.heading(col, text=col)
            self.top_tree.column(col, width=150)
        
        # Scrollbar for top predictions
        top_scrollbar = ttk.Scrollbar(self.top_predictions_frame, orient=tk.VERTICAL, command=self.top_tree.yview)
        self.top_tree.configure(yscrollcommand=top_scrollbar.set)
        
        self.top_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        top_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    def setup_player_search_tab(self):
        """Setup the player search tab"""
        # Search frame
        search_frame = ttk.LabelFrame(self.player_search_frame, text="Player Search")
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Player Name:").pack(side=tk.LEFT, padx=5)
        self.player_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.player_search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind('<KeyRelease>', self.search_player)
        
        # Player results
        player_columns = ("Prop", "Probability", "Expected Value", "Confidence", "Prediction")
        self.player_tree = ttk.Treeview(self.player_search_frame, columns=player_columns, show="headings", height=15)
        
        for col in player_columns:
            self.player_tree.heading(col, text=col)
            self.player_tree.column(col, width=120)
        
        player_scrollbar = ttk.Scrollbar(self.player_search_frame, orient=tk.VERTICAL, command=self.player_tree.yview)
        self.player_tree.configure(yscrollcommand=player_scrollbar.set)
        
        self.player_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        player_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    def setup_recommendations_tab(self):
        """Setup the recommendations tab"""
        rec_label = ttk.Label(self.recommendations_frame, text="Top Recommendations", font=("Arial", 12, "bold"))
        rec_label.pack(pady=10)
        
        # Recommendations tree
        rec_columns = ("Player", "Prop", "Probability", "Expected Value", "Multiplier", "Risk Level")
        self.rec_tree = ttk.Treeview(self.recommendations_frame, columns=rec_columns, show="headings", height=20)
        
        for col in rec_columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=130)
        
        rec_scrollbar = ttk.Scrollbar(self.recommendations_frame, orient=tk.VERTICAL, command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scrollbar.set)
        
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        rec_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    def setup_summary_tab(self):
        """Setup the summary statistics tab"""
        # Summary text
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, font=("Consolas", 10))
        summary_scrollbar = ttk.Scrollbar(self.summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
    
    def load_latest_predictions(self):
        """Load the most recent prediction files"""
        try:
            if not os.path.exists(self.default_dir):
                messagebox.showerror("Error", f"Analysis directory not found: {self.default_dir}")
                return
            
            # Find latest prediction file
            pred_files = [f for f in os.listdir(self.default_dir) if f.startswith('underdog_predictions_') and f.endswith('.csv')]
            if not pred_files:
                messagebox.showwarning("Warning", "No prediction files found in analysis directory")
                return
            
            latest_pred_file = max(pred_files)
            pred_path = os.path.join(self.default_dir, latest_pred_file)
            
            # Load predictions
            self.predictions_df = pd.read_csv(pred_path)
            
            # Try to load recommendations file
            rec_file = latest_pred_file.replace('predictions', 'recommendations')
            rec_path = os.path.join(self.default_dir, rec_file)
            if os.path.exists(rec_path):
                self.recommendations_df = pd.read_csv(rec_path)
            
            self.status_label.config(text=f"Loaded: {latest_pred_file}", foreground="green")
            self.update_available_teams()  # Update team list when data is loaded
            self.refresh_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load predictions: {str(e)}")
            self.status_label.config(text="Error loading files", foreground="red")
    
    def load_custom_file(self):
        """Load a custom prediction file"""
        file_path = filedialog.askopenfilename(
            title="Select Prediction File",
            initialdir=self.default_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.predictions_df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                self.status_label.config(text=f"Loaded: {filename}", foreground="green")
                self.update_available_teams()  # Update team list when data is loaded
                self.refresh_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def refresh_display(self):
        """Refresh all display elements"""
        if self.predictions_df is not None:
            self.update_top_predictions()
            self.update_summary()
            if self.recommendations_df is not None:
                self.update_recommendations()
    
    def update_top_predictions(self):
        """Update the top predictions display"""
        # Clear existing items
        for item in self.top_tree.get_children():
            self.top_tree.delete(item)
        
        if self.predictions_df is None:
            return
        
        # Get all predictions in a single list
        predictions = []
        
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                if probability > 0:  # Only show non-zero predictions
                    ev_col = f"{prop_name}_expected_value"
                    conf_col = f"{prop_name}_confidence"
                    
                    expected_value = row.get(ev_col, 0)
                    confidence = row.get(conf_col, 'unknown')
                    
                    predictions.append({
                        'Player': player,
                        'Prop': prop_name.replace('_', ' ').title(),
                        'Probability': probability,
                        'Expected_Value': expected_value,
                        'Confidence': confidence
                    })
        
        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x['Probability'], reverse=True)
        
        # Apply filters
        prop_filter = self.prop_var.get()
        min_prob = self.min_prob_var.get()
        
        for pred in predictions:
            if prop_filter != "All" and prop_filter.replace('_', ' ').title() != pred['Prop']:
                continue
            if pred['Probability'] < min_prob:
                continue
            
            self.top_tree.insert("", "end", values=(
                pred['Player'],
                pred['Prop'],
                f"{pred['Probability']:.3f}",
                f"{pred['Expected_Value']:+.2f}",
                pred['Confidence']
            ))
    
    def filter_predictions(self, event=None):
        """Filter predictions based on current filter settings"""
        self.update_top_predictions()
    
    def search_player(self, event=None):
        """Search for a specific player"""
        search_term = self.player_search_var.get().lower()
        
        # Clear existing items
        for item in self.player_tree.get_children():
            self.player_tree.delete(item)
        
        if not search_term or self.predictions_df is None:
            return
        
        # Find matching players
        matching_players = self.predictions_df[
            self.predictions_df['Name'].str.lower().str.contains(search_term, na=False)
        ]
        
        for _, row in matching_players.iterrows():
            player = row['Name']
            
            prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                if probability > 0:
                    ev_col = f"{prop_name}_expected_value"
                    conf_col = f"{prop_name}_confidence"
                    pred_col = f"{prop_name}_prediction"
                    
                    expected_value = row.get(ev_col, 0)
                    confidence = row.get(conf_col, 'unknown')
                    prediction = row.get(pred_col, 0)
                    
                    self.player_tree.insert("", "end", values=(
                        prop_name.replace('_', ' ').title(),
                        f"{probability:.3f}",
                        f"{expected_value:+.2f}",
                        confidence,
                        prediction
                    ))
    
    def update_recommendations(self):
        """Update the recommendations display"""
        # Clear existing items
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        if self.recommendations_df is None:
            return
        
        for _, row in self.recommendations_df.iterrows():
            self.rec_tree.insert("", "end", values=(
                row.get('Player', ''),
                row.get('Prop', ''),
                f"{row.get('Probability', 0):.3f}",
                f"{row.get('Expected_Value', 0):+.2f}",
                f"{row.get('Multiplier', 0):.1f}",
                row.get('Risk_Level', '')
            ))
    
    def update_summary(self):
        """Update the summary statistics"""
        self.summary_text.delete(1.0, tk.END)
        
        if self.predictions_df is None:
            self.summary_text.insert(tk.END, "No data loaded")
            return
        
        # Calculate summary statistics
        total_players = len(self.predictions_df)
        
        # Count predictions by prop type
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        summary = f"UNDERDOG FANTASY PREDICTION SUMMARY\n"
        summary += f"=" * 50 + "\n\n"
        summary += f"Total Players: {total_players}\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        summary += "PREDICTIONS BY PROP TYPE:\n"
        summary += "-" * 30 + "\n"
        
        for prop_col in prop_columns:
            prop_name = prop_col.replace('_probability', '')
            non_zero_count = (self.predictions_df[prop_col] > 0).sum()
            avg_prob = self.predictions_df[prop_col].mean()
            max_prob = self.predictions_df[prop_col].max()
            
            summary += f"{prop_name.replace('_', ' ').title()}:\n"
            summary += f"  Non-zero predictions: {non_zero_count}\n"
            summary += f"  Average probability: {avg_prob:.3f}\n"
            summary += f"  Maximum probability: {max_prob:.3f}\n\n"
        
        if self.recommendations_df is not None:
            summary += f"\nRECOMMENDATIONS:\n"
            summary += f"-" * 20 + "\n"
            summary += f"Total recommendations: {len(self.recommendations_df)}\n"
            
            if len(self.recommendations_df) > 0:
                top_5 = self.recommendations_df.head(5)
                summary += f"\nTop 5 Recommendations:\n"
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    summary += f"{i}. {row.get('Player', '')} - {row.get('Prop', '')} "
                    summary += f"(Prob: {row.get('Probability', 0):.2f}, EV: {row.get('Expected_Value', 0):+.2f})\n"
        
        self.summary_text.insert(tk.END, summary)
    
    def setup_parlay_builder_tab(self):
        """Setup the parlay builder tab"""
        # Control frame
        control_frame = ttk.LabelFrame(self.parlay_frame, text="Parlay Settings")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Parlay settings
        ttk.Label(control_frame, text="Max Legs:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_legs_var = tk.IntVar(value=6)
        ttk.Spinbox(control_frame, from_=2, to=6, textvariable=self.max_legs_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Min Probability:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.parlay_min_prob_var = tk.DoubleVar(value=0.3)  # Lowered from 0.6
        ttk.Spinbox(control_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.parlay_min_prob_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Min Combined Prob:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.min_combined_prob_var = tk.DoubleVar(value=0.05)  # Lowered from 0.2
        ttk.Spinbox(control_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.min_combined_prob_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Max Players:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.max_players_var = tk.IntVar(value=6)  # Increased from 3
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.max_players_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Team selection
        ttk.Label(control_frame, text="Team Filter:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.team_filter_var = tk.StringVar(value="All Teams")
        self.team_combo = ttk.Combobox(control_frame, textvariable=self.team_filter_var, width=15)
        self.team_combo.grid(row=2, column=1, padx=5, pady=5)
        
        # Team selection buttons
        ttk.Button(control_frame, text="Select Teams", command=self.open_team_selector).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Reset Teams", command=self.reset_team_selection).grid(row=2, column=3, padx=5, pady=5)
        
        # Generate button
        ttk.Button(control_frame, text="Generate Parlays", command=self.generate_parlays).grid(row=0, column=4, padx=20, pady=5, rowspan=3, sticky=tk.NS)
        
        # Parlay results
        parlay_columns = ("Parlay ID", "Players", "Props", "Combined Prob", "Estimated Payout", "Expected Value", "Risk Score")
        self.parlay_tree = ttk.Treeview(self.parlay_frame, columns=parlay_columns, show="headings", height=15)
        
        for col in parlay_columns:
            self.parlay_tree.heading(col, text=col)
            if col == "Parlay ID":
                self.parlay_tree.column(col, width=80)
            elif col == "Players":
                self.parlay_tree.column(col, width=380)  # Much wider for 6+ players
            elif col == "Props":
                self.parlay_tree.column(col, width=400)  # Much wider for 6+ leg details
            else:
                self.parlay_tree.column(col, width=120)
        
        parlay_scrollbar = ttk.Scrollbar(self.parlay_frame, orient=tk.VERTICAL, command=self.parlay_tree.yview)
        self.parlay_tree.configure(yscrollcommand=parlay_scrollbar.set)
        
        self.parlay_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        parlay_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Bind double-click to show parlay details
        self.parlay_tree.bind("<Double-1>", self.show_parlay_details)
    
    def open_team_selector(self):
        """Open team selection dialog"""
        if self.predictions_df is None:
            messagebox.showwarning("Warning", "No prediction data loaded")
            return
        
        # Update available teams from current data
        self.update_available_teams()
        
        # Create team selection window
        team_window = tk.Toplevel(self.root)
        team_window.title("Select Teams for Parlays")
        team_window.geometry("500x600")
        team_window.transient(self.root)
        team_window.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(team_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions
        ttk.Label(main_frame, text="Select teams to include in parlay generation:", 
                 font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # Selection buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="Select All", 
                  command=lambda: self.select_all_teams(team_vars)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", 
                  command=lambda: self.clear_all_teams(team_vars)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="AL Only", 
                  command=lambda: self.select_league_teams(team_vars, "AL")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="NL Only", 
                  command=lambda: self.select_league_teams(team_vars, "NL")).pack(side=tk.LEFT)
        
        # Teams frame with scrollbar
        teams_frame = ttk.Frame(main_frame)
        teams_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(teams_frame)
        scrollbar = ttk.Scrollbar(teams_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Team checkboxes
        team_vars = {}
        for i, team in enumerate(sorted(self.available_teams)):
            var = tk.BooleanVar()
            var.set(team in self.selected_teams or len(self.selected_teams) == 0)
            team_vars[team] = var
            
            ttk.Checkbutton(scrollable_frame, text=team, variable=var).grid(
                row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2
            )
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        def apply_selection():
            self.selected_teams = {team for team, var in team_vars.items() if var.get()}
            self.update_team_filter_display()
            team_window.destroy()
        
        def cancel_selection():
            team_window.destroy()
        
        ttk.Button(bottom_frame, text="Apply", command=apply_selection).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(bottom_frame, text="Cancel", command=cancel_selection).pack(side=tk.RIGHT)
        
        # Show current selection count
        selected_count = len(self.selected_teams) if self.selected_teams else len(self.available_teams)
        ttk.Label(bottom_frame, text=f"Currently selected: {selected_count} teams").pack(side=tk.LEFT)
    
    def update_available_teams(self):
        """Update the list of available teams from prediction data"""
        if self.predictions_df is None:
            return
        
        # Get teams from player mapping
        teams = set()
        
        # Extract teams from all players in the dataset
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            # First try to get team from CSV Team column
            if 'Team' in self.predictions_df.columns:
                team = row.get('Team')
                if team and team != 'Unknown Team':
                    teams.add(team)
            else:
                # Fallback to player mapping
                player_team = self.get_team_for_player(player)
                if player_team and player_team != 'Unknown Team':
                    teams.add(player_team)
        
        # If no teams found from mapping, use standard MLB teams as fallback
        if not teams:
            teams = {
                'Arizona Diamondbacks', 'Atlanta Braves', 'Baltimore Orioles', 'Boston Red Sox',
                'Chicago Cubs', 'Chicago White Sox', 'Cincinnati Reds', 'Cleveland Guardians',
                'Colorado Rockies', 'Detroit Tigers', 'Houston Astros', 'Kansas City Royals',
                'Los Angeles Angels', 'Los Angeles Dodgers', 'Miami Marlins', 'Milwaukee Brewers',
                'Minnesota Twins', 'New York Mets', 'New York Yankees', 'Oakland Athletics',
                'Philadelphia Phillies', 'Pittsburgh Pirates', 'San Diego Padres', 'San Francisco Giants',
                'Seattle Mariners', 'St. Louis Cardinals', 'Tampa Bay Rays', 'Texas Rangers',
                'Toronto Blue Jays', 'Washington Nationals'
            }
        
        self.available_teams = sorted(list(teams))
        
        # Update combo box values
        team_options = ["All Teams", "Selected Teams Only"]
        if hasattr(self, 'team_combo'):
            self.team_combo['values'] = team_options
        
        print(f"Found {len(self.available_teams)} teams from {len(self.predictions_df)} players")  # Debug
    
    def select_all_teams(self, team_vars):
        """Select all teams"""
        for var in team_vars.values():
            var.set(True)
    
    def clear_all_teams(self, team_vars):
        """Clear all team selections"""
        for var in team_vars.values():
            var.set(False)
    
    def select_league_teams(self, team_vars, league):
        """Select teams from specific league"""
        al_teams = {
            'Baltimore Orioles', 'Boston Red Sox', 'New York Yankees', 'Tampa Bay Rays', 'Toronto Blue Jays',
            'Chicago White Sox', 'Cleveland Guardians', 'Detroit Tigers', 'Kansas City Royals', 'Minnesota Twins',
            'Houston Astros', 'Los Angeles Angels', 'Oakland Athletics', 'Seattle Mariners', 'Texas Rangers'
        }
        
        # Clear all first
        self.clear_all_teams(team_vars)
        
        # Select based on league
        for team, var in team_vars.items():
            if league == "AL" and team in al_teams:
                var.set(True)
            elif league == "NL" and team not in al_teams:
                var.set(True)
    
    def reset_team_selection(self):
        """Reset team selection to all teams"""
        self.selected_teams = set()
        self.update_team_filter_display()
    
    def update_team_filter_display(self):
        """Update the team filter display"""
        if not self.selected_teams:
            self.team_filter_var.set("All Teams")
        else:
            count = len(self.selected_teams)
            self.team_filter_var.set(f"{count} Teams Selected")
    
    def get_team_for_player(self, player_name):
        """Get team for a player using CSV data first, then fallback to mapping"""
        # First check if we have team data in the CSV
        if self.predictions_df is not None and 'Team' in self.predictions_df.columns:
            player_data = self.predictions_df[self.predictions_df['Name'] == player_name]
            if not player_data.empty:
                team = player_data['Team'].iloc[0]
                if team and team != 'Unknown Team':
                    return team
        
        # Fallback to hardcoded mapping if needed
        player_team_mapping = {
            # American League East
            'Aaron Judge': 'New York Yankees', 'Juan Soto': 'New York Yankees', 'Giancarlo Stanton': 'New York Yankees',
            'Gleyber Torres': 'New York Yankees', 'Anthony Rizzo': 'New York Yankees', 'Jose Trevino': 'New York Yankees',
            'DJ LeMahieu': 'New York Yankees', 'Anthony Volpe': 'New York Yankees', 'Alex Verdugo': 'New York Yankees',
            'Jasson Domínguez': 'New York Yankees', 'Austin Wells': 'New York Yankees', 'Ben Rice': 'New York Yankees',
            'Oswald Peraza': 'New York Yankees', 'Trey Sweeney': 'New York Yankees',
            
            'Rafael Devers': 'Boston Red Sox', 'Xander Bogaerts': 'Boston Red Sox', 'Trevor Story': 'Boston Red Sox',
            'Kike Hernandez': 'Boston Red Sox', 'Christian Arroyo': 'Boston Red Sox', 'Jarren Duran': 'Boston Red Sox',
            'Masataka Yoshida': 'Boston Red Sox', 'Connor Wong': 'Boston Red Sox', 'David Hamilton': 'Boston Red Sox',
            'Ceddanne Rafaela': 'Boston Red Sox', 'Nick Sogard': 'Boston Red Sox', 'Rob Refsnyder': 'Boston Red Sox',
            'Wilyer Abreu': 'Boston Red Sox', 'Chase Meidroth': 'Boston Red Sox', 'Nick Yorke': 'Boston Red Sox',
            
            'Vladimir Guerrero Jr.': 'Toronto Blue Jays', 'Bo Bichette': 'Toronto Blue Jays', 'George Springer': 'Toronto Blue Jays',
            'Matt Chapman': 'Toronto Blue Jays', 'Alejandro Kirk': 'Toronto Blue Jays', 'Daulton Varsho': 'Toronto Blue Jays',
            'Davis Schneider': 'Toronto Blue Jays', 'Spencer Horwitz': 'Toronto Blue Jays', 'Addison Barger': 'Toronto Blue Jays',
            'Otto Lopez': 'Toronto Blue Jays', 'Ernie Clement': 'Toronto Blue Jays', 'Danny Jansen': 'Toronto Blue Jays',
            
            'Adley Rutschman': 'Baltimore Orioles', 'Gunnar Henderson': 'Baltimore Orioles', 'Anthony Santander': 'Baltimore Orioles',
            'Ryan Mountcastle': 'Baltimore Orioles', 'Austin Hays': 'Baltimore Orioles', 'Adam Frazier': 'Baltimore Orioles',
            'Cedric Mullins': 'Baltimore Orioles', 'Jordan Westburg': 'Baltimore Orioles', 'Jackson Holliday': 'Baltimore Orioles',
            'Connor Norby': 'Baltimore Orioles', 'Coby Mayo': 'Baltimore Orioles', 'Colton Cowser': 'Baltimore Orioles',
            'Samuel Basallo': 'Baltimore Orioles', 'Ramón Urías': 'Baltimore Orioles',
            
            'Randy Arozarena': 'Tampa Bay Rays', 'Wander Franco': 'Tampa Bay Rays', 'Brandon Lowe': 'Tampa Bay Rays',
            'Isaac Paredes': 'Tampa Bay Rays', 'Harold Ramirez': 'Tampa Bay Rays', 'Christian Bethancourt': 'Tampa Bay Rays',
            'Josh Lowe': 'Tampa Bay Rays', 'Jonathan Aranda': 'Tampa Bay Rays', 'Curtis Mead': 'Tampa Bay Rays',
            'Junior Caminero': 'Tampa Bay Rays', 'Yandy Díaz': 'Tampa Bay Rays',
            
            # American League Central
            'Jose Altuve': 'Houston Astros', 'Alex Bregman': 'Houston Astros', 'Kyle Tucker': 'Houston Astros',
            'Yordan Alvarez': 'Houston Astros', 'Yainer Diaz': 'Houston Astros', 'Chas McCormick': 'Houston Astros',
            'Jeremy Peña': 'Houston Astros', 'Jose Abreu': 'Houston Astros', 'Jake Meyers': 'Houston Astros',
            
            'Corey Seager': 'Texas Rangers', 'Nathaniel Lowe': 'Texas Rangers', 'Marcus Semien': 'Texas Rangers',
            'Adolis García': 'Texas Rangers', 'Jonah Heim': 'Texas Rangers', 'Leody Taveras': 'Texas Rangers',
            'Ezequiel Duran': 'Texas Rangers', 'Robbie Grossman': 'Texas Rangers', 'Josh Jung': 'Texas Rangers',
            'Wyatt Langford': 'Texas Rangers',
            
            'Salvador Perez': 'Kansas City Royals', 'Bobby Witt Jr.': 'Kansas City Royals', 'Vinnie Pasquantino': 'Kansas City Royals',
            'MJ Melendez': 'Kansas City Royals', 'Maikel Garcia': 'Kansas City Royals', 'Hunter Dozier': 'Kansas City Royals',
            'Michael Massey': 'Kansas City Royals', 'Freddy Fermin': 'Kansas City Royals',
            
            'José Ramírez': 'Cleveland Guardians', 'Josh Naylor': 'Cleveland Guardians', 'Steven Kwan': 'Cleveland Guardians',
            'Andrés Giménez': 'Cleveland Guardians', 'Tyler Freeman': 'Cleveland Guardians', 'Will Brennan': 'Cleveland Guardians',
            'Brayan Rocchio': 'Cleveland Guardians', 'Daniel Schneemann': 'Cleveland Guardians', 'Bo Naylor': 'Cleveland Guardians',
            'Angel Martínez': 'Cleveland Guardians', 'Jhonkensy Noel': 'Cleveland Guardians',
            
            'Byron Buxton': 'Minnesota Twins', 'Carlos Correa': 'Minnesota Twins', 'Max Kepler': 'Minnesota Twins',
            'Jorge Polanco': 'Minnesota Twins', 'Royce Lewis': 'Minnesota Twins', 'Ryan Jeffers': 'Minnesota Twins',
            'Trevor Larnach': 'Minnesota Twins', 'Brooks Lee': 'Minnesota Twins',
            
            'Tim Anderson': 'Chicago White Sox', 'Eloy Jimenez': 'Chicago White Sox', 'Luis Robert Jr.': 'Chicago White Sox',
            'Andrew Vaughn': 'Chicago White Sox', 'Yoán Moncada': 'Chicago White Sox', 'Seby Zavala': 'Chicago White Sox',
            'Gavin Sheets': 'Chicago White Sox', 'Lenyn Sosa': 'Chicago White Sox', 'Colson Montgomery': 'Chicago White Sox',
            'Edgar Quero': 'Chicago White Sox',
            
            'Spencer Torkelson': 'Detroit Tigers', 'Riley Greene': 'Detroit Tigers', 'Javier Báez': 'Detroit Tigers',
            'Colt Keith': 'Detroit Tigers', 'Mark Canha': 'Detroit Tigers', 'Jake Rogers': 'Detroit Tigers',
            'Kerry Carpenter': 'Detroit Tigers', 'Parker Meadows': 'Detroit Tigers', 'Justyn-Henry Malloy': 'Detroit Tigers',
            'Dillon Dingler': 'Detroit Tigers',
            
            # American League West
            'Mike Trout': 'Los Angeles Angels', 'Shohei Ohtani': 'Los Angeles Angels', 'Anthony Rendon': 'Los Angeles Angels',
            'Hunter Renfroe': 'Los Angeles Angels', 'Taylor Ward': 'Los Angeles Angels', 'Logan O\'Hoppe': 'Los Angeles Angels',
            'Jo Adell': 'Los Angeles Angels', 'Mickey Moniak': 'Los Angeles Angels', 'Luis Rengifo': 'Los Angeles Angels',
            'Nolan Schanuel': 'Los Angeles Angels',
            
            'Julio Rodríguez': 'Seattle Mariners', 'Eugenio Suárez': 'Seattle Mariners', 'Cal Raleigh': 'Seattle Mariners',
            'J.P. Crawford': 'Seattle Mariners', 'Ty France': 'Seattle Mariners', 'Teoscar Hernández': 'Seattle Mariners',
            'Mitch Garver': 'Seattle Mariners', 'Harry Ford': 'Seattle Mariners', 'Dominic Canzone': 'Seattle Mariners',
            
            'Brent Rooker': 'Oakland Athletics', 'Seth Brown': 'Oakland Athletics', 'Ryan Noda': 'Oakland Athletics',
            'Shea Langeliers': 'Oakland Athletics', 'Tony Kemp': 'Oakland Athletics', 'Aledmys Diaz': 'Oakland Athletics',
            'Lawrence Butler': 'Oakland Athletics', 'Tyler Soderstrom': 'Oakland Athletics', 'Max Schuemann': 'Oakland Athletics',
            
            # National League East
            'Ronald Acuña Jr.': 'Atlanta Braves', 'Freddie Freeman': 'Atlanta Braves', 'Ozzie Albies': 'Atlanta Braves',
            'Austin Riley': 'Atlanta Braves', 'Matt Olson': 'Atlanta Braves', 'Sean Murphy': 'Atlanta Braves',
            'Michael Harris II': 'Atlanta Braves', 'Orlando Arcia': 'Atlanta Braves', 'Marcell Ozuna': 'Atlanta Braves',
            
            'Bryce Harper': 'Philadelphia Phillies', 'Trea Turner': 'Philadelphia Phillies', 'Kyle Schwarber': 'Philadelphia Phillies',
            'Nick Castellanos': 'Philadelphia Phillies', 'J.T. Realmuto': 'Philadelphia Phillies', 'Alec Bohm': 'Philadelphia Phillies',
            'Bryson Stott': 'Philadelphia Phillies', 'Brandon Marsh': 'Philadelphia Phillies',
            
            'Pete Alonso': 'New York Mets', 'Francisco Lindor': 'New York Mets', 'Starling Marte': 'New York Mets',
            'Eduardo Escobar': 'New York Mets', 'Jeff McNeil': 'New York Mets', 'Brandon Nimmo': 'New York Mets',
            'Francisco Alvarez': 'New York Mets', 'Mark Vientos': 'New York Mets', 'Brett Baty': 'New York Mets',
            'Luisangel Acuña': 'New York Mets', 'Ronny Mauricio': 'New York Mets',
            
            'CJ Abrams': 'Washington Nationals', 'Keibert Ruiz': 'Washington Nationals',
            'Joey Meneses': 'Washington Nationals', 'Luis García Jr.': 'Washington Nationals', 'Lane Thomas': 'Washington Nationals',
            'James Wood': 'Washington Nationals', 'Dylan Crews': 'Washington Nationals', 'Brady House': 'Washington Nationals',
            
            'Jazz Chisholm Jr.': 'Miami Marlins', 'Jorge Soler': 'Miami Marlins', 'Jesús Sánchez': 'Miami Marlins',
            'Jacob Stallings': 'Miami Marlins', 'Jon Berti': 'Miami Marlins', 'Nick Fortes': 'Miami Marlins',
            'JJ Bleday': 'Miami Marlins', 'Griffin Conine': 'Miami Marlins',
            
            # National League Central
            'Cody Bellinger': 'Chicago Cubs', 'Nico Hoerner': 'Chicago Cubs', 'Ian Happ': 'Chicago Cubs',
            'Dansby Swanson': 'Chicago Cubs', 'Seiya Suzuki': 'Chicago Cubs', 'Yan Gomes': 'Chicago Cubs',
            'Christopher Morel': 'Chicago Cubs', 'Mike Tauchman': 'Chicago Cubs', 'Pete Crow-Armstrong': 'Chicago Cubs',
            'Michael Busch': 'Chicago Cubs', 'Matt Shaw': 'Chicago Cubs',
            
            'Paul Goldschmidt': 'St. Louis Cardinals', 'Nolan Arenado': 'St. Louis Cardinals', 'Willson Contreras': 'St. Louis Cardinals',
            'Tyler O\'Neill': 'St. Louis Cardinals', 'Brendan Donovan': 'St. Louis Cardinals', 'Tommy Edman': 'St. Louis Cardinals',
            'Lars Nootbaar': 'St. Louis Cardinals', 'Dylan Carlson': 'St. Louis Cardinals', 'Nolan Gorman': 'St. Louis Cardinals',
            'Jordan Walker': 'St. Louis Cardinals', 'Thomas Saggese': 'St. Louis Cardinals', 'Victor Scott II': 'St. Louis Cardinals',
            'Iván Herrera': 'St. Louis Cardinals',
            
            'Christian Yelich': 'Milwaukee Brewers', 'Willy Adames': 'Milwaukee Brewers', 'Rowdy Tellez': 'Milwaukee Brewers',
            'William Contreras': 'Milwaukee Brewers', 'Jesse Winker': 'Milwaukee Brewers', 'Brice Turang': 'Milwaukee Brewers',
            'Jackson Chourio': 'Milwaukee Brewers', 'Sal Frelick': 'Milwaukee Brewers', 'Joey Ortiz': 'Milwaukee Brewers',
            
            'Elly De La Cruz': 'Cincinnati Reds', 'Jonathan India': 'Cincinnati Reds', 'Tyler Stephenson': 'Cincinnati Reds',
            'Spencer Steer': 'Cincinnati Reds', 'TJ Friedl': 'Cincinnati Reds', 'Matt McLain': 'Cincinnati Reds',
            'Noelvi Marte': 'Cincinnati Reds', 'José Fermín': 'Cincinnati Reds',
            
            'Ke\'Bryan Hayes': 'Pittsburgh Pirates', 'Bryan Reynolds': 'Pittsburgh Pirates', 'Andrew McCutchen': 'Pittsburgh Pirates',
            'Carlos Santana': 'Pittsburgh Pirates', 'Henry Davis': 'Pittsburgh Pirates', 'Termarr Johnson': 'Pittsburgh Pirates',
            'Jack Suwinski': 'Pittsburgh Pirates', 'Jared Triolo': 'Pittsburgh Pirates', 'Nick Gonzales': 'Pittsburgh Pirates',
            'Oneil Cruz': 'Pittsburgh Pirates',
            
            # National League West
            'Manny Machado': 'San Diego Padres', 'Fernando Tatis Jr.': 'San Diego Padres', 'Xander Bogaerts': 'San Diego Padres',
            'Jake Cronenworth': 'San Diego Padres', 'Ha-seong Kim': 'San Diego Padres', 'Luis Campusano': 'San Diego Padres',
            'Jackson Merrill': 'San Diego Padres', 'Jurickson Profar': 'San Diego Padres',
            
            'Mookie Betts': 'Los Angeles Dodgers', 'Freddie Freeman': 'Los Angeles Dodgers', 'Max Muncy': 'Los Angeles Dodgers',
            'Will Smith': 'Los Angeles Dodgers', 'Justin Turner': 'Los Angeles Dodgers', 'Chris Taylor': 'Los Angeles Dodgers',
            'Teoscar Hernández': 'Los Angeles Dodgers', 'Tommy Edman': 'Los Angeles Dodgers', 'Gavin Lux': 'Los Angeles Dodgers',
            'James Outman': 'Los Angeles Dodgers', 'Andy Pages': 'Los Angeles Dodgers', 'Miguel Rojas': 'Los Angeles Dodgers',
            'Miguel Vargas': 'Los Angeles Dodgers', 'Dalton Rushing': 'Los Angeles Dodgers',
            
            'Corbin Carroll': 'Arizona Diamondbacks', 'Christian Walker': 'Arizona Diamondbacks', 'Ketel Marte': 'Arizona Diamondbacks',
            'Lourdes Gurriel Jr.': 'Arizona Diamondbacks', 'Gabriel Moreno': 'Arizona Diamondbacks', 'Alek Thomas': 'Arizona Diamondbacks',
            'Geraldo Perdomo': 'Arizona Diamondbacks', 'Jordan Lawlar': 'Arizona Diamondbacks',
            
            'Thairo Estrada': 'San Francisco Giants', 'Joc Pederson': 'San Francisco Giants', 'Mike Yastrzemski': 'San Francisco Giants',
            'LaMonte Wade Jr.': 'San Francisco Giants', 'Patrick Bailey': 'San Francisco Giants', 'Tyler Fitzgerald': 'San Francisco Giants',
            'Heliot Ramos': 'San Francisco Giants', 'Grant McCray': 'San Francisco Giants', 'Jung Hoo Lee': 'San Francisco Giants',
            'Casey Schmitt': 'San Francisco Giants',
            
            'Charlie Blackmon': 'Colorado Rockies', 'C.J. Cron': 'Colorado Rockies', 'Ryan McMahon': 'Colorado Rockies',
            'Ezequiel Tovar': 'Colorado Rockies', 'Elias Diaz': 'Colorado Rockies', 'Kris Bryant': 'Colorado Rockies',
            'Brenton Doyle': 'Colorado Rockies', 'Hunter Goodman': 'Colorado Rockies', 'Jordan Beck': 'Colorado Rockies',
        }
        
        # Additional partial name matching for common names
        player_lower = player_name.lower()
        for mapped_player, team in player_team_mapping.items():
            if mapped_player.lower() in player_lower or player_lower in mapped_player.lower():
                return team
        
        # If no exact match found, return None to include the player
        return None
    
    def setup_lineup_optimizer_tab(self):
        """Setup the lineup optimizer tab"""
        # Optimizer settings
        settings_frame = ttk.LabelFrame(self.lineup_frame, text="Lineup Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Lineup Type:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.lineup_type_var = tk.StringVar(value="Conservative")
        lineup_combo = ttk.Combobox(settings_frame, textvariable=self.lineup_type_var, width=15)
        lineup_combo['values'] = ["Conservative", "Balanced", "Aggressive", "High Risk"]
        lineup_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Budget:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.budget_var = tk.IntVar(value=100)
        ttk.Spinbox(settings_frame, from_=10, to=1000, increment=10, textvariable=self.budget_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Max Picks:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_picks_var = tk.IntVar(value=8)
        ttk.Spinbox(settings_frame, from_=3, to=15, textvariable=self.max_picks_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="Correlation Factor:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.correlation_var = tk.DoubleVar(value=0.8)
        ttk.Spinbox(settings_frame, from_=0.0, to=1.0, increment=0.1, textvariable=self.correlation_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # Generate button
        ttk.Button(settings_frame, text="Generate Lineups", command=self.generate_lineups).grid(row=0, column=4, padx=20, pady=5, rowspan=2, sticky=tk.NS)
        
        # Results notebook for different lineup types
        self.lineup_notebook = ttk.Notebook(self.lineup_frame)
        self.lineup_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Cash game lineups
        self.cash_frame = ttk.Frame(self.lineup_notebook)
        self.lineup_notebook.add(self.cash_frame, text="Cash Game Lineups")
        self.setup_lineup_display(self.cash_frame, "cash")
        
        # Tournament lineups
        self.tournament_frame = ttk.Frame(self.lineup_notebook)
        self.lineup_notebook.add(self.tournament_frame, text="Tournament Lineups")
        self.setup_lineup_display(self.tournament_frame, "tournament")
        
        # Contrarian lineups
        self.contrarian_frame = ttk.Frame(self.lineup_notebook)
        self.lineup_notebook.add(self.contrarian_frame, text="Contrarian Lineups")
        self.setup_lineup_display(self.contrarian_frame, "contrarian")
    
    def setup_lineup_display(self, parent, lineup_type):
        """Setup lineup display for a specific type"""
        # Create tree for lineup display
        lineup_columns = ("Lineup", "Total EV", "Total Prob", "Risk Score", "Picks", "Budget Used")
        tree = ttk.Treeview(parent, columns=lineup_columns, show="headings", height=12)
        
        for col in lineup_columns:
            tree.heading(col, text=col)
            tree.column(col, width=90 if col == "Lineup" else 120)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Store tree reference
        setattr(self, f"{lineup_type}_tree", tree)
        
        # Bind double-click to show lineup details
        tree.bind("<Double-1>", lambda e: self.show_lineup_details(e, lineup_type))

    def generate_parlays(self):
        """Generate optimized parlays based on current predictions"""
        if self.predictions_df is None:
            messagebox.showwarning("Warning", "No prediction data loaded")
            return
        
        # Clear existing parlays
        for item in self.parlay_tree.get_children():
            self.parlay_tree.delete(item)
        
        # Get settings
        max_legs = self.max_legs_var.get()
        min_prob = self.parlay_min_prob_var.get()
        min_combined_prob = self.min_combined_prob_var.get()
        max_players = self.max_players_var.get()
        
        # Get high-probability picks
        eligible_picks = []
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        print(f"Found {len(prop_columns)} probability columns: {prop_columns}")  # Debug
        
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            # Apply team filter if teams are selected
            if self.selected_teams:
                player_team = self.get_team_for_player(player)
                if player_team is None:
                    # If we can't find the team, skip this player when filtering is active
                    print(f"Warning: Could not find team for player {player}, skipping...")
                    continue
                elif player_team not in self.selected_teams:
                    # Skip this player if their team is not selected
                    continue
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                # Check if probability is valid and above threshold
                if pd.notna(probability) and probability >= min_prob:
                    ev_col = f"{prop_name}_expected_value"
                    expected_value = row.get(ev_col, 0)
                    
                    # Use the full prop name for display and multiplier lookup
                    prop_display = prop_name.replace('_', ' ').title()
                    
                    eligible_picks.append({
                        'player': player,
                        'prop': prop_name,
                        'prop_display': prop_display,
                        'probability': probability,
                        'expected_value': expected_value,
                        'multiplier': self.get_prop_multiplier(prop_name)
                    })
        
        print(f"Found {len(eligible_picks)} eligible picks")  # Debug
        
        # Debug: Show team filtering info
        if self.selected_teams:
            print(f"Team filtering active - selected teams: {sorted(list(self.selected_teams))}")
            # Count players by team
            team_counts = {}
            for pick in eligible_picks:
                team = self.get_team_for_player(pick['player'])
                if team:
                    team_counts[team] = team_counts.get(team, 0) + 1
            print(f"Players by team in eligible picks: {team_counts}")
        else:
            print("No team filtering - using all teams")
        
        if len(eligible_picks) < 2:
            messagebox.showwarning("Warning", f"Not enough eligible picks found ({len(eligible_picks)}). Try lowering the minimum probability threshold.")
            return
        
        # Sort by probability * expected value
        eligible_picks.sort(key=lambda x: x['probability'] * abs(x['expected_value']), reverse=True)
        
        # Generate parlay combinations - Find BEST parlay for each leg count
        from itertools import combinations
        best_parlays_by_legs = {}  # Store best parlay for each leg count
        parlay_id = 1
        
        # Try different leg counts and find the BEST for each
        for leg_count in range(2, max_legs + 1):
            print(f"Finding best {leg_count}-leg parlay...")  # Debug
            best_parlay = None
            best_ev = float('-inf')
            
            combination_count = 0
            for parlay_combo in combinations(eligible_picks[:80], leg_count):  # Use top 80 picks
                parlay_legs = list(parlay_combo)
                
                # Ensure no duplicate players (each player can only appear once per parlay)
                unique_players = set([leg['player'] for leg in parlay_legs])
                if len(unique_players) != len(parlay_legs):
                    continue  # Skip if same player appears multiple times
                
                # Check player limit
                if len(unique_players) > max_players:
                    continue
                
                # Calculate combined probability
                combined_prob = 1.0
                for leg in parlay_legs:
                    combined_prob *= leg['probability']
                
                if combined_prob < min_combined_prob:
                    continue
                
                # Calculate payout
                total_multiplier = 1.0
                for leg in parlay_legs:
                    total_multiplier *= leg['multiplier']
                
                # Calculate expected value and risk
                expected_value = (combined_prob * total_multiplier) - 1
                risk_score = self.calculate_parlay_risk(parlay_legs, combined_prob)
                
                # Check if this is the best parlay for this leg count
                if expected_value > best_ev:
                    best_ev = expected_value
                    best_parlay = {
                        'id': parlay_id,
                        'legs': parlay_legs,
                        'players': list(unique_players),
                        'combined_prob': combined_prob,
                        'total_multiplier': total_multiplier,
                        'expected_value': expected_value,
                        'risk_score': risk_score,
                        'leg_count': leg_count
                    }
                
                combination_count += 1
                # Limit combinations per leg count to keep it fast
                if combination_count >= 5000:
                    break
            
            # Store the best parlay for this leg count
            if best_parlay:
                best_parlays_by_legs[leg_count] = best_parlay
                parlay_id += 1
                print(f"Best {leg_count}-leg parlay: EV={best_ev:.2f}, Prob={best_parlay['combined_prob']:.3f}")
        
        # Generate more variety - top parlays for each leg count
        all_parlays = []
        for leg_count in range(2, max_legs + 1):
            leg_parlays = []
            combination_count = 0
            
            for parlay_combo in combinations(eligible_picks[:100], leg_count):  # Use top 100 picks
                parlay_legs = list(parlay_combo)
                
                # Ensure no duplicate players (each player can only appear once per parlay)
                unique_players = set([leg['player'] for leg in parlay_legs])
                if len(unique_players) != len(parlay_legs):
                    continue  # Skip if same player appears multiple times
                
                # Check player limit
                if len(unique_players) > max_players:
                    continue
                
                # Calculate combined probability
                combined_prob = 1.0
                for leg in parlay_legs:
                    combined_prob *= leg['probability']
                
                if combined_prob < min_combined_prob:
                    continue
                
                # Calculate payout
                total_multiplier = 1.0
                for leg in parlay_legs:
                    total_multiplier *= leg['multiplier']
                
                # Calculate expected value and risk
                expected_value = (combined_prob * total_multiplier) - 1
                risk_score = self.calculate_parlay_risk(parlay_legs, combined_prob)
                
                leg_parlays.append({
                    'id': len(all_parlays) + 1,
                    'legs': parlay_legs,
                    'players': list(unique_players),
                    'combined_prob': combined_prob,
                    'total_multiplier': total_multiplier,
                    'expected_value': expected_value,
                    'risk_score': risk_score,
                    'leg_count': leg_count
                })
                
                combination_count += 1
                if combination_count >= 10000:  # More combinations per leg count
                    break
            
            # Sort and take top 15 for this leg count (increased from 5)
            leg_parlays.sort(key=lambda x: x['expected_value'], reverse=True)
            all_parlays.extend(leg_parlays[:15])
        
        # Combine best parlays and variety parlays
        parlays = list(best_parlays_by_legs.values()) + all_parlays
        
        print(f"Generated {len(parlays)} total parlays")  # Debug
        
        # Sort by leg count first, then by expected value
        parlays.sort(key=lambda x: (x['leg_count'], -x['expected_value']))
        
        # Display parlays organized by leg count - show multiple options per leg count
        displayed_count = 0
        current_leg_count = 0
        parlay_counter = {}  # Track parlay numbers per leg count
        
        for parlay in parlays[:100]:  # Show more parlays
            leg_count = parlay['leg_count']
            
            # Initialize counter for this leg count
            if leg_count not in parlay_counter:
                parlay_counter[leg_count] = 0
            
            # Limit to 10 parlays per leg count for variety
            if parlay_counter[leg_count] >= 10:
                continue
                
            parlay_counter[leg_count] += 1
            
            # Add separator for new leg count
            if leg_count != current_leg_count:
                if current_leg_count > 0:
                    # Add a visual separator
                    self.parlay_tree.insert("", "end", values=(
                        "---",
                        "---",
                        "---",
                        "---",
                        "---",
                        "---",
                        "---"
                    ))
                current_leg_count = leg_count
            
            # Show ALL players with full names (no truncation)
            all_players = list(set([leg['player'] for leg in parlay['legs']]))
            # Always show last names for space efficiency, but show ALL players
            players_str = ", ".join([player.split()[-1] for player in all_players])
            
            # Show ALL props with player names (no truncation for any leg count)
            if len(parlay['legs']) <= 4:
                # Show full detail for smaller parlays
                props_str = " + ".join([f"{leg['player'].split()[-1]} {leg['prop_display'][:15]}" for leg in parlay['legs']])
            else:
                # For 5+ leg parlays, show abbreviated but ALL legs
                props_str = " + ".join([f"{leg['player'].split()[-1]} {leg['prop_display'][:12]}" for leg in parlay['legs']])
            
            # Truncate props string if too long for display (increased limit)
            if len(props_str) > 180:
                props_str = props_str[:177] + "..."
            
            # Add leg count and parlay number for clarity
            parlay_id_display = f"{leg_count}L-#{parlay_counter[leg_count]}"
            
            self.parlay_tree.insert("", "end", values=(
                parlay_id_display,
                players_str,
                props_str,
                f"{parlay['combined_prob']:.3f}",
                f"{parlay['total_multiplier']:.1f}x",
                f"{parlay['expected_value']:+.2f}",
                f"{parlay['risk_score']:.2f}"
            ))
            displayed_count += 1
        
        # Store parlays for detail view
        self.generated_parlays = parlays
        
        # Create summary of parlay counts by leg count
        leg_summary = []
        for leg_count in range(2, max_legs + 1):
            leg_count_parlays = [p for p in parlays if p['leg_count'] == leg_count]
            if leg_count_parlays:
                count = min(10, len(leg_count_parlays))  # Show up to 10 per leg count
                best_ev = max(leg_count_parlays, key=lambda x: x['expected_value'])['expected_value']
                leg_summary.append(f"{leg_count}-leg: {count} parlays (Best EV: {best_ev:+.2f})")
        
        summary_text = "\n".join(leg_summary)
        
        # Add team filter info to message
        team_info = ""
        if self.selected_teams:
            team_count = len(self.selected_teams)
            team_info = f"\nFiltered to {team_count} selected teams"
        else:
            team_info = f"\nUsing all available teams ({len(self.available_teams)} teams)"
        
        messagebox.showinfo("Multiple Parlays Generated!", 
                          f"Generated multiple options for each leg count:\n\n{summary_text}{team_info}\n\nShowing {displayed_count} total parlays.\nEligible picks: {len(eligible_picks)}\n\nDouble-click any parlay for full details!")
    
    def generate_lineups(self):
        """Generate optimized lineups based on different strategies"""
        if self.predictions_df is None:
            messagebox.showwarning("Warning", "No prediction data loaded")
            return
        
        # Get settings
        lineup_type = self.lineup_type_var.get()
        budget = self.budget_var.get()
        max_picks = self.max_picks_var.get()
        correlation_factor = self.correlation_var.get()
        
        # Generate different lineup types
        cash_lineups = self.optimize_cash_lineups(budget, max_picks)
        tournament_lineups = self.optimize_tournament_lineups(budget, max_picks)
        contrarian_lineups = self.optimize_contrarian_lineups(budget, max_picks)
        
        # Display lineups
        self.display_lineups(cash_lineups, self.cash_tree, "Cash")
        self.display_lineups(tournament_lineups, self.tournament_tree, "Tournament")
        self.display_lineups(contrarian_lineups, self.contrarian_tree, "Contrarian")
        
        messagebox.showinfo("Success", "Generated optimized lineups for all strategies!")
    
    def optimize_cash_lineups(self, budget, max_picks):
        """Generate conservative cash game lineups"""
        lineups = []
        
        # Get high-probability, low-risk picks
        safe_picks = []
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                if probability >= 0.7:  # High probability threshold for cash games
                    ev_col = f"{prop_name}_expected_value"
                    expected_value = row.get(ev_col, 0)
                    
                    safe_picks.append({
                        'player': player,
                        'prop': prop_name.replace('_', ' ').title(),
                        'probability': probability,
                        'expected_value': expected_value,
                        'cost': self.get_pick_cost(prop_name),
                        'value_score': probability * expected_value
                    })
        
        # Sort by value score
        safe_picks.sort(key=lambda x: x['value_score'], reverse=True)
        
        # Generate lineups using greedy approach
        for i in range(10):  # Generate 10 cash lineups
            lineup = self.build_lineup(safe_picks, budget, max_picks, strategy='cash', variation=i)
            if lineup:
                lineups.append(lineup)
        
        return lineups
    
    def optimize_tournament_lineups(self, budget, max_picks):
        """Generate high-upside tournament lineups"""
        lineups = []
        
        # Get high-upside picks (higher variance)
        upside_picks = []
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                if 0.4 <= probability <= 0.8:  # Medium probability, high upside
                    ev_col = f"{prop_name}_expected_value"
                    expected_value = row.get(ev_col, 0)
                    
                    # Calculate upside score (higher multiplier for lower prob, higher EV)
                    multiplier = self.get_prop_multiplier(prop_name)
                    upside_score = expected_value * multiplier * (1 - probability)
                    
                    upside_picks.append({
                        'player': player,
                        'prop': prop_name.replace('_', ' ').title(),
                        'probability': probability,
                        'expected_value': expected_value,
                        'cost': self.get_pick_cost(prop_name),
                        'upside_score': upside_score
                    })
        
        # Sort by upside score
        upside_picks.sort(key=lambda x: x['upside_score'], reverse=True)
        
        # Generate lineups
        for i in range(15):  # Generate 15 tournament lineups
            lineup = self.build_lineup(upside_picks, budget, max_picks, strategy='tournament', variation=i)
            if lineup:
                lineups.append(lineup)
        
        return lineups
    
    def optimize_contrarian_lineups(self, budget, max_picks):
        """Generate contrarian lineups with unique combinations"""
        lineups = []
        
        # Get medium probability picks that might be overlooked
        contrarian_picks = []
        prop_columns = [col for col in self.predictions_df.columns if col.endswith('_probability')]
        
        for _, row in self.predictions_df.iterrows():
            player = row['Name']
            
            for prop_col in prop_columns:
                prop_name = prop_col.replace('_probability', '')
                probability = row[prop_col]
                
                if 0.3 <= probability <= 0.6:  # Medium probability range
                    ev_col = f"{prop_name}_expected_value"
                    expected_value = row.get(ev_col, 0)
                    
                    # Contrarian score favors less obvious picks
                    contrarian_score = expected_value / (probability + 0.1)  # Inverse relationship
                    
                    contrarian_picks.append({
                        'player': player,
                        'prop': prop_name.replace('_', ' ').title(),
                        'probability': probability,
                        'expected_value': expected_value,
                        'cost': self.get_pick_cost(prop_name),
                        'contrarian_score': contrarian_score
                    })
        
        # Sort by contrarian score
        contrarian_picks.sort(key=lambda x: x['contrarian_score'], reverse=True)
        
        # Generate lineups
        for i in range(12):  # Generate 12 contrarian lineups
            lineup = self.build_lineup(contrarian_picks, budget, max_picks, strategy='contrarian', variation=i)
            if lineup:
                lineups.append(lineup)
        
        return lineups
    
    def build_lineup(self, picks, budget, max_picks, strategy='cash', variation=0):
        """Build a single lineup using knapsack-style optimization"""
        import random
        
        # Add some randomization for variety
        random.seed(42 + variation)
        available_picks = picks.copy()
        
        if variation > 0:
            # Shuffle slightly for variation
            random.shuffle(available_picks)
            available_picks = sorted(available_picks, key=lambda x: x.get('value_score', x.get('upside_score', x.get('contrarian_score', 0))), reverse=True)
        
        selected_picks = []
        total_cost = 0
        total_ev = 0
        total_prob = 1.0
        used_players = set()
        
        for pick in available_picks:
            # Check constraints
            if len(selected_picks) >= max_picks:
                break
            if total_cost + pick['cost'] > budget:
                continue
            if pick['player'] in used_players and len(used_players) >= max_picks // 2:
                continue  # Limit player stacking
            
            # Add pick
            selected_picks.append(pick)
            total_cost += pick['cost']
            total_ev += pick['expected_value']
            total_prob *= pick['probability']
            used_players.add(pick['player'])
        
        if len(selected_picks) < 3:  # Minimum lineup size
            return None
        
        # Calculate risk score
        risk_score = 1 - total_prob + (len(set(used_players)) / len(selected_picks))  # Diversification bonus
        
        return {
            'picks': selected_picks,
            'total_cost': total_cost,
            'total_ev': total_ev,
            'total_prob': total_prob,
            'risk_score': risk_score,
            'unique_players': len(used_players)
        }

    def get_prop_multiplier(self, prop_name):
        """Get betting multiplier for different prop types"""
        multipliers = {
            'hits_over_0.5': 2.5,
            'hits_over_1.5': 3.5,
            'runs_over_0.5': 3.0,
            'rbis_over_0.5': 3.2,
            'hrs_over_0.5': 8.0,
            'sbs_over_0.5': 6.0,
            'total_bases_over_1.5': 2.8,
            'hits_runs_rbis_over_1.5': 3.5,
            'hits': 2.5,
            'runs': 3.0,
            'rbis': 3.2,
            'home_runs': 8.0,
            'stolen_bases': 6.0,
            'total_bases': 2.8,
            'walks': 2.2,
            'strikeouts': 2.0
        }
        return multipliers.get(prop_name.lower(), 2.5)
    
    def get_pick_cost(self, prop_name):
        """Get cost for lineup building (simplified pricing)"""
        costs = {
            'hits_over_0.5': 50,
            'hits_over_1.5': 70,
            'runs_over_0.5': 60,
            'rbis_over_0.5': 65,
            'hrs_over_0.5': 120,
            'sbs_over_0.5': 100,
            'total_bases_over_1.5': 55,
            'hits_runs_rbis_over_1.5': 75,
            'hits': 50,
            'runs': 60,
            'rbis': 65,
            'home_runs': 120,
            'stolen_bases': 100,
            'total_bases': 55,
            'walks': 40,
            'strikeouts': 35
        }
        return costs.get(prop_name.lower(), 50)
    
    def calculate_parlay_risk(self, parlay_legs, combined_prob):
        """Calculate risk score for a parlay"""
        # Base risk from probability
        prob_risk = 1 - combined_prob
        
        # Player correlation risk
        players = [leg['player'] for leg in parlay_legs]
        unique_players = len(set(players))
        correlation_risk = 1 - (unique_players / len(parlay_legs))
        
        # Prop type correlation risk
        prop_types = [leg['prop'] for leg in parlay_legs]
        unique_props = len(set(prop_types))
        prop_risk = 1 - (unique_props / len(parlay_legs))
        
        # Combined risk score (0-1, lower is better)
        total_risk = (prob_risk * 0.6) + (correlation_risk * 0.3) + (prop_risk * 0.1)
        return min(total_risk, 1.0)
    
    def display_lineups(self, lineups, tree_widget, strategy_name):
        """Display lineups in the specified tree widget"""
        # Clear existing items
        for item in tree_widget.get_children():
            tree_widget.delete(item)
        
        # Display lineups
        for i, lineup in enumerate(lineups[:20], 1):  # Show top 20
            picks_str = ", ".join([f"{pick['player'][:8]} {pick['prop'][:8]}" for pick in lineup['picks'][:3]])
            if len(lineup['picks']) > 3:
                picks_str += f" +{len(lineup['picks'])-3}"
            
            tree_widget.insert("", "end", values=(
                f"{strategy_name[0]}{i}",
                f"{lineup['total_ev']:+.2f}",
                f"{lineup['total_prob']:.3f}",
                f"{lineup['risk_score']:.2f}",
                picks_str,
                f"${lineup['total_cost']:.0f}"
            ))
        
        # Store lineups for detail view
        if not hasattr(self, 'generated_lineups'):
            self.generated_lineups = {}
        self.generated_lineups[strategy_name] = lineups
    
    def show_parlay_details(self, event):
        """Show detailed parlay information"""
        selection = self.parlay_tree.selection()
        if not selection:
            return
        
        item = self.parlay_tree.item(selection[0])
        parlay_id = item['values'][0]
        
        # Find the parlay
        parlay = None
        if hasattr(self, 'generated_parlays'):
            # Handle new format: "2L-#1", "3L-#2", etc or old formats
            if parlay_id.startswith("---"):
                return  # Skip separator rows
            
            # Extract leg count and parlay number
            if "L-#" in parlay_id:
                parts = parlay_id.split("L-#")
                leg_count = int(parts[0])
                parlay_number = int(parts[1])
                
                # Find the Nth parlay for this leg count
                leg_parlays = [p for p in self.generated_parlays if p['leg_count'] == leg_count]
                leg_parlays.sort(key=lambda x: x['expected_value'], reverse=True)
                if parlay_number <= len(leg_parlays):
                    parlay = leg_parlays[parlay_number - 1]
            elif "L-P" in parlay_id:
                actual_id = int(parlay_id.split("L-P")[1])
                for p in self.generated_parlays:
                    if p['id'] == actual_id:
                        parlay = p
                        break
            elif parlay_id.startswith("P"):
                actual_id = int(parlay_id[1:])
                for p in self.generated_parlays:
                    if p['id'] == actual_id:
                        parlay = p
                        break
        
        if not parlay:
            return
        
        # Create detail window
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Parlay Details - {parlay_id}")
        detail_window.geometry("600x500")
        
        # Main frame
        main_frame = ttk.Frame(detail_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary info
        summary_frame = ttk.LabelFrame(main_frame, text="Parlay Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(summary_frame, text=f"Parlay ID: {parlay_id}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Legs: {parlay['leg_count']}").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(summary_frame, text=f"Combined Probability: {parlay['combined_prob']:.3f}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Total Multiplier: {parlay['total_multiplier']:.1f}x").grid(row=1, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(summary_frame, text=f"Expected Value: {parlay['expected_value']:+.2f}").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Risk Score: {parlay['risk_score']:.2f}").grid(row=2, column=1, sticky=tk.W, padx=(20, 0))
        
        # Legs details
        legs_frame = ttk.LabelFrame(main_frame, text="Parlay Legs", padding=10)
        legs_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for legs
        legs_tree = ttk.Treeview(legs_frame, columns=("Player", "Prop", "Probability", "Expected Value", "Multiplier"), show="headings")
        legs_tree.heading("Player", text="Player")
        legs_tree.heading("Prop", text="Proposition")
        legs_tree.heading("Probability", text="Probability")
        legs_tree.heading("Expected Value", text="Expected Value")
        legs_tree.heading("Multiplier", text="Multiplier")
        
        # Configure column widths
        legs_tree.column("Player", width=120)
        legs_tree.column("Prop", width=150)
        legs_tree.column("Probability", width=100)
        legs_tree.column("Expected Value", width=120)
        legs_tree.column("Multiplier", width=100)
        
        # Add scrollbar
        legs_scrollbar = ttk.Scrollbar(legs_frame, orient=tk.VERTICAL, command=legs_tree.yview)
        legs_tree.configure(yscrollcommand=legs_scrollbar.set)
        
        # Pack treeview and scrollbar
        legs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        legs_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate legs
        for leg in parlay['legs']:
            legs_tree.insert("", "end", values=(
                leg['player'],
                leg['prop_display'],
                f"{leg['probability']:.3f}",
                f"{leg['expected_value']:+.2f}",
                f"{leg['multiplier']:.1f}x"
            ))
    
    def show_lineup_details(self, event, lineup_type=None):
        """Show detailed lineup information"""
        tree_widget = event.widget
        selection = tree_widget.selection()
        if not selection:
            return
        
        item = tree_widget.item(selection[0])
        lineup_id = item['values'][0]
        
        # Determine strategy and find lineup
        strategy = None
        lineup = None
        
        if hasattr(self, 'generated_lineups'):
            for strategy_name, lineups in self.generated_lineups.items():
                for i, l in enumerate(lineups, 1):
                    if f"{strategy_name[0]}{i}" == lineup_id:
                        strategy = strategy_name
                        lineup = l
                        break
                if lineup:
                    break
        
        if not lineup:
            return
        
        # Create detail window
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Lineup Details - {lineup_id} ({strategy})")
        detail_window.geometry("700x500")
        
        # Main frame
        main_frame = ttk.Frame(detail_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary info
        summary_frame = ttk.LabelFrame(main_frame, text="Lineup Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(summary_frame, text=f"Lineup ID: {lineup_id}").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Strategy: {strategy}").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(summary_frame, text=f"Total Cost: ${lineup['total_cost']:.0f}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Unique Players: {lineup['unique_players']}").grid(row=1, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(summary_frame, text=f"Total Probability: {lineup['total_prob']:.3f}").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(summary_frame, text=f"Expected Value: {lineup['total_ev']:+.2f}").grid(row=2, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(summary_frame, text=f"Risk Score: {lineup['risk_score']:.2f}").grid(row=3, column=0, sticky=tk.W)
        
        # Picks details
        picks_frame = ttk.LabelFrame(main_frame, text="Lineup Picks", padding=10)
        picks_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for picks
        picks_tree = ttk.Treeview(picks_frame, columns=("Player", "Prop", "Probability", "Expected Value", "Cost"), show="headings")
        picks_tree.heading("Player", text="Player")
        picks_tree.heading("Prop", text="Proposition")
        picks_tree.heading("Probability", text="Probability")
        picks_tree.heading("Expected Value", text="Expected Value")
        picks_tree.heading("Cost", text="Cost")
        
        # Configure column widths
        picks_tree.column("Player", width=120)
        picks_tree.column("Prop", width=150)
        picks_tree.column("Probability", width=100)
        picks_tree.column("Expected Value", width=120)
        picks_tree.column("Cost", width=80)
        
        # Add scrollbar
        picks_scrollbar = ttk.Scrollbar(picks_frame, orient=tk.VERTICAL, command=picks_tree.yview)
        picks_tree.configure(yscrollcommand=picks_scrollbar.set)
        
        # Pack treeview and scrollbar
        picks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        picks_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate picks
        for pick in lineup['picks']:
            picks_tree.insert("", "end", values=(
                pick['player'],
                pick['prop'],
                f"{pick['probability']:.3f}",
                f"{pick['expected_value']:+.2f}",
                f"${pick['cost']:.0f}"
            ))


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = NFLUnderdogFantasyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
