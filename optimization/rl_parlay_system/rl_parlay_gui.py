#!/usr/bin/env python3
"""
RL Parlay GUI
GUI interface for the trained RL parlay generation model
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from .rl_parlay_environment import ParlayEnvironment
from .rl_parlay_agent import PPOAgent

class RLParlayGUI:
    """GUI for RL Parlay Generation"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 RL Parlay Generator")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.current_data = None
        self.trained_agent = None
        self.current_environment = None
        
        # Create GUI
        self.create_widgets()
        
        # Load available models
        self.load_available_models()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data Loading Tab
        self.create_data_tab()
        
        # Model Management Tab
        self.create_model_tab()
        
        # Parlay Generation Tab
        self.create_parlay_tab()
        
        # Results Tab
        self.create_results_tab()
    
    def create_data_tab(self):
        """Create data loading tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 Data Loading")
        
        # Data file selection
        ttk.Label(data_frame, text="Load NFL Data for RL Parlay Generation", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # File selection
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(file_frame, text="Data File:").pack(side=tk.LEFT)
        self.data_file_var = tk.StringVar()
        self.data_file_entry = ttk.Entry(file_frame, textvariable=self.data_file_var, width=50)
        self.data_file_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_data_file).pack(side=tk.LEFT, padx=5)
        
        # Load button
        ttk.Button(data_frame, text="Load Data", 
                  command=self.load_data).pack(pady=10)
        
        # Data info
        self.data_info_text = tk.Text(data_frame, height=10, width=80)
        self.data_info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar for data info
        data_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_info_text.yview)
        self.data_info_text.configure(yscrollcommand=data_scrollbar.set)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_model_tab(self):
        """Create model management tab"""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="🤖 Model Management")
        
        # Model selection
        ttk.Label(model_frame, text="RL Model Management", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Available models
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(model_select_frame, text="Available Models:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_select_frame, textvariable=self.model_var, 
                                       state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_select_frame, text="Load Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(model_select_frame, text="Refresh", 
                  command=self.load_available_models).pack(side=tk.LEFT, padx=5)
        
        # Model info
        self.model_info_text = tk.Text(model_frame, height=15, width=80)
        self.model_info_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Model actions
        action_frame = ttk.Frame(model_frame)
        action_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(action_frame, text="Test Model", 
                  command=self.test_model).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(action_frame, text="View Training Progress", 
                  command=self.view_training_progress).pack(side=tk.LEFT, padx=5)
    
    def create_parlay_tab(self):
        """Create parlay generation tab"""
        parlay_frame = ttk.Frame(self.notebook)
        self.notebook.add(parlay_frame, text="🎯 Parlay Generation")
        
        # Generation controls
        ttk.Label(parlay_frame, text="RL Parlay Generation", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(parlay_frame, text="Generation Settings")
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Number of parlays
        num_frame = ttk.Frame(settings_frame)
        num_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(num_frame, text="Number of Parlays:").pack(side=tk.LEFT)
        self.num_parlays_var = tk.StringVar(value="5")
        ttk.Entry(num_frame, textvariable=self.num_parlays_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Max legs per parlay
        legs_frame = ttk.Frame(settings_frame)
        legs_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(legs_frame, text="Max Legs per Parlay:").pack(side=tk.LEFT)
        self.max_legs_var = tk.StringVar(value="4")
        ttk.Entry(legs_frame, textvariable=self.max_legs_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Generation button
        ttk.Button(parlay_frame, text="Generate Parlays", 
                  command=self.generate_parlays).pack(pady=10)
        
        # Results display
        self.parlay_results_text = tk.Text(parlay_frame, height=20, width=80)
        self.parlay_results_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar for results
        parlay_scrollbar = ttk.Scrollbar(parlay_frame, orient=tk.VERTICAL, 
                                        command=self.parlay_results_text.yview)
        self.parlay_results_text.configure(yscrollcommand=parlay_scrollbar.set)
        parlay_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_results_tab(self):
        """Create results analysis tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📈 Results Analysis")
        
        # Analysis controls
        ttk.Label(results_frame, text="Parlay Results Analysis", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        # Analysis buttons
        analysis_frame = ttk.Frame(results_frame)
        analysis_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Button(analysis_frame, text="Analyze Generated Parlays", 
                  command=self.analyze_parlays).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analysis_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analysis_frame, text="Clear Results", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Analysis results
        self.analysis_text = tk.Text(results_frame, height=20, width=80)
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar for analysis
        analysis_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                          command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def browse_data_file(self):
        """Browse for data file"""
        filename = filedialog.askopenfilename(
            title="Select NFL Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_file_var.set(filename)
    
    def load_data(self):
        """Load NFL data"""
        filename = self.data_file_var.get()
        if not filename:
            messagebox.showerror("Error", "Please select a data file")
            return
        
        try:
            # Load data
            self.current_data = pd.read_csv(filename)
            
            # Display data info
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, f"✅ Data loaded successfully!\n\n")
            self.data_info_text.insert(tk.END, f"Records: {len(self.current_data)}\n")
            self.data_info_text.insert(tk.END, f"Columns: {list(self.current_data.columns)}\n\n")
            
            # Show sample data
            self.data_info_text.insert(tk.END, "Sample data:\n")
            self.data_info_text.insert(tk.END, str(self.current_data.head()))
            
            # Create environment
            self.current_environment = ParlayEnvironment(self.current_data)
            
            messagebox.showinfo("Success", "Data loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def load_available_models(self):
        """Load available trained models"""
        models_dir = "rl_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        model_names = [f.replace('.pth', '') for f in model_files]
        
        self.model_combo['values'] = model_names
        if model_names:
            self.model_combo.set(model_names[0])
    
    def load_model(self):
        """Load selected model"""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        
        if self.current_data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            # Create agent
            state_dim = self.current_environment.observation_space.shape[0]
            action_dims = self.current_environment.action_space.nvec.tolist()
            self.trained_agent = PPOAgent(state_dim, action_dims)
            
            # Load model
            model_path = os.path.join("rl_models", f"{model_name}.pth")
            self.trained_agent.load_model(model_path)
            
            # Display model info
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, f"✅ Model loaded successfully!\n\n")
            self.model_info_text.insert(tk.END, f"Model: {model_name}\n")
            self.model_info_text.insert(tk.END, f"State dimension: {state_dim}\n")
            self.model_info_text.insert(tk.END, f"Action dimensions: {action_dims}\n")
            self.model_info_text.insert(tk.END, f"Device: {self.trained_agent.device}\n\n")
            
            # Test model
            test_parlay = self.trained_agent.generate_parlay(self.current_environment)
            self.model_info_text.insert(tk.END, f"Test parlay generated:\n")
            self.model_info_text.insert(tk.END, f"Legs: {test_parlay['num_legs']}\n")
            self.model_info_text.insert(tk.END, f"Hit Rate: {test_parlay['combined_hit_rate']:.2%}\n")
            self.model_info_text.insert(tk.END, f"Odds: +{test_parlay['estimated_odds']:.0f}\n")
            
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def test_model(self):
        """Test the loaded model"""
        if self.trained_agent is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if self.current_environment is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            # Generate test parlay
            parlay = self.trained_agent.generate_parlay(self.current_environment)
            
            # Display results
            self.model_info_text.insert(tk.END, f"\n--- Test Parlay ---\n")
            self.model_info_text.insert(tk.END, f"Legs: {parlay['num_legs']}\n")
            self.model_info_text.insert(tk.END, f"Hit Rate: {parlay['combined_hit_rate']:.2%}\n")
            self.model_info_text.insert(tk.END, f"Odds: +{parlay['estimated_odds']:.0f}\n")
            self.model_info_text.insert(tk.END, f"Expected Value: ${parlay['expected_value']:.2f}\n")
            
            for i, leg in enumerate(parlay['legs'], 1):
                self.model_info_text.insert(tk.END, 
                    f"  {i}. {leg['player']} ({leg['team']}) - {leg['prop']} O{leg['line']:.1f}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Test failed: {str(e)}")
    
    def view_training_progress(self):
        """View training progress"""
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Load model to get training data
            model_path = os.path.join("rl_models", f"{model_name}.pth")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            episode_rewards = checkpoint.get('episode_rewards', [])
            training_losses = checkpoint.get('training_losses', [])
            
            if episode_rewards or training_losses:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                if episode_rewards:
                    ax1.plot(episode_rewards)
                    ax1.set_title('Episode Rewards')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Reward')
                    ax1.grid(True)
                
                if training_losses:
                    ax2.plot(training_losses)
                    ax2.set_title('Training Loss')
                    ax2.set_xlabel('Update')
                    ax2.set_ylabel('Loss')
                    ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
            else:
                messagebox.showinfo("Info", "No training data available for this model")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view training progress: {str(e)}")
    
    def generate_parlays(self):
        """Generate parlays using the trained model"""
        if self.trained_agent is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if self.current_environment is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            num_parlays = int(self.num_parlays_var.get())
            max_legs = int(self.max_legs_var.get())
            
            # Update environment max legs
            self.current_environment.max_legs = max_legs
            
            # Generate parlays
            self.parlay_results_text.delete(1.0, tk.END)
            self.parlay_results_text.insert(tk.END, f"🤖 Generating {num_parlays} parlays...\n\n")
            
            generated_parlays = []
            
            for i in range(num_parlays):
                parlay = self.trained_agent.generate_parlay(self.current_environment)
                generated_parlays.append(parlay)
                
                # Display parlay
                self.parlay_results_text.insert(tk.END, f"--- Parlay {i+1} ---\n")
                self.parlay_results_text.insert(tk.END, f"Legs: {parlay['num_legs']}\n")
                self.parlay_results_text.insert(tk.END, f"Hit Rate: {parlay['combined_hit_rate']:.2%}\n")
                self.parlay_results_text.insert(tk.END, f"Odds: +{parlay['estimated_odds']:.0f}\n")
                self.parlay_results_text.insert(tk.END, f"Expected Value: ${parlay['expected_value']:.2f}\n")
                
                for j, leg in enumerate(parlay['legs'], 1):
                    self.parlay_results_text.insert(tk.END, 
                        f"  {j}. {leg['player']} ({leg['team']}) - {leg['prop']} O{leg['line']:.1f} ({leg['hit_rate']:.1%})\n")
                
                self.parlay_results_text.insert(tk.END, "\n")
            
            # Store generated parlays for analysis
            self.generated_parlays = generated_parlays
            
            messagebox.showinfo("Success", f"Generated {num_parlays} parlays successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate parlays: {str(e)}")
    
    def analyze_parlays(self):
        """Analyze generated parlays"""
        if not hasattr(self, 'generated_parlays'):
            messagebox.showerror("Error", "No parlays generated yet")
            return
        
        try:
            parlays = self.generated_parlays
            
            # Calculate statistics
            num_legs = [p['num_legs'] for p in parlays]
            hit_rates = [p['combined_hit_rate'] for p in parlays]
            odds = [p['estimated_odds'] for p in parlays]
            expected_values = [p['expected_value'] for p in parlays]
            
            # Display analysis
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, "📊 Parlay Analysis Results\n")
            self.analysis_text.insert(tk.END, "=" * 40 + "\n\n")
            
            self.analysis_text.insert(tk.END, f"Total Parlays: {len(parlays)}\n")
            self.analysis_text.insert(tk.END, f"Average Legs: {np.mean(num_legs):.1f}\n")
            self.analysis_text.insert(tk.END, f"Average Hit Rate: {np.mean(hit_rates):.2%}\n")
            self.analysis_text.insert(tk.END, f"Average Odds: +{np.mean(odds):.0f}\n")
            self.analysis_text.insert(tk.END, f"Average Expected Value: ${np.mean(expected_values):.2f}\n\n")
            
            # Best parlay
            best_idx = np.argmax(expected_values)
            best_parlay = parlays[best_idx]
            
            self.analysis_text.insert(tk.END, "🏆 Best Parlay (by Expected Value):\n")
            self.analysis_text.insert(tk.END, f"Legs: {best_parlay['num_legs']}\n")
            self.analysis_text.insert(tk.END, f"Hit Rate: {best_parlay['combined_hit_rate']:.2%}\n")
            self.analysis_text.insert(tk.END, f"Odds: +{best_parlay['estimated_odds']:.0f}\n")
            self.analysis_text.insert(tk.END, f"Expected Value: ${best_parlay['expected_value']:.2f}\n\n")
            
            # Player frequency analysis
            all_players = []
            for parlay in parlays:
                for leg in parlay['legs']:
                    all_players.append(leg['player'])
            
            from collections import Counter
            player_counts = Counter(all_players)
            
            self.analysis_text.insert(tk.END, "👥 Most Selected Players:\n")
            for player, count in player_counts.most_common(10):
                self.analysis_text.insert(tk.END, f"  {player}: {count} times\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def export_results(self):
        """Export results to file"""
        if not hasattr(self, 'generated_parlays'):
            messagebox.showerror("Error", "No parlays to export")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Parlay Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.generated_parlays, f, indent=2)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        self.parlay_results_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        if hasattr(self, 'generated_parlays'):
            delattr(self, 'generated_parlays')
        messagebox.showinfo("Info", "Results cleared")

def main():
    """Main function"""
    root = tk.Tk()
    app = RLParlayGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
