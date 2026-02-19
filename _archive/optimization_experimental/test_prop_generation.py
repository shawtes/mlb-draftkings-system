#!/usr/bin/env python3
"""
Test script to verify prop generation works with the new data
"""

import pandas as pd
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from nfl_underdog_gui import NFLUnderdogFantasyGUI
import tkinter as tk

def test_prop_generation():
    """Test prop generation with the new data"""
    
    print("🧪 Testing prop generation with new NFL data...")
    
    # Load the new data
    data_file = "nfl_week8_WITH_PROJECTIONS.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Create a minimal GUI instance for testing
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    gui = NFLUnderdogFantasyGUI(root)
    
    # Load the data
    print(f"📁 Loading data from {data_file}...")
    gui.nfl_data_df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(gui.nfl_data_df)} players")
    
    # Generate props
    print("🎯 Generating prop bets...")
    gui.generate_nfl_props()
    
    print(f"✅ Generated {len(gui.prop_bets)} prop bets")
    
    if gui.prop_bets:
        print("\n📊 Sample prop bets:")
        for i, prop in enumerate(gui.prop_bets[:10]):  # Show first 10
            print(f"  {i+1}. {prop['player']} ({prop['team']}) - {prop['prop']} O{prop['line']} (Proj: {prop['projection']:.1f})")
        
        # Show breakdown by prop type
        prop_types = {}
        for prop in gui.prop_bets:
            prop_type = prop['prop']
            if prop_type not in prop_types:
                prop_types[prop_type] = 0
            prop_types[prop_type] += 1
        
        print(f"\n📈 Prop breakdown:")
        for prop_type, count in sorted(prop_types.items()):
            print(f"  {prop_type}: {count}")
    
    root.destroy()

if __name__ == "__main__":
    test_prop_generation()
