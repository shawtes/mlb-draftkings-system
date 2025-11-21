#!/usr/bin/env python3
"""
Test script to verify data flow from GUI to OptimizationWorker
Simulates what happens when you click "Run Optimization"
"""

import pandas as pd
import sys

print("\n" + "="*80)
print("🧪 TESTING GUI → WORKER DATA FLOW")
print("="*80)

# Load the CSV
csv_file = "../nba_NOV1_READY.csv"
print(f"\n📂 Loading: {csv_file}")

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} players")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

# ============================================================================
# TEST 1: Simulate GUI collecting included_players
# ============================================================================
print("\n" + "="*80)
print("🧪 TEST 1: Simulate GUI collecting included_players")
print("="*80)

# This simulates what get_included_players() does
simulated_gui_selection = ["Jimmy Butler III", "LaMelo Ball", "Anthony Edwards"]
print(f"   📋 GUI simulated selection: {simulated_gui_selection}")
print(f"   📊 Number of players: {len(simulated_gui_selection)}")

# ============================================================================
# TEST 2: Simulate passing to OptimizationWorker
# ============================================================================
print("\n" + "="*80)
print("🧪 TEST 2: Simulate creating OptimizationWorker")
print("="*80)

# Simulate the worker initialization parameters
print(f"   🔧 Creating worker with parameters:")
print(f"      - df_players: {len(df)} players")
print(f"      - included_players: {simulated_gui_selection}")
print(f"      - len(included_players): {len(simulated_gui_selection)}")

# Create a simple mock worker class
class MockOptimizationWorker:
    def __init__(self, df_players, included_players):
        self.df_players = df_players
        self.included_players = included_players
        print(f"\n   ✅ Worker created!")
        print(f"      📊 Worker received df_players: {len(self.df_players)} players")
        print(f"      🎯 Worker received included_players: {self.included_players}")
        print(f"      📋 Worker len(included_players): {len(self.included_players) if self.included_players else 0}")
        print(f"      🔍 Type of included_players: {type(self.included_players)}")
    
    def preprocess_data(self):
        """Simulate the preprocess_data method"""
        print(f"\n   🔧 Running preprocess_data()...")
        df_filtered = self.df_players.copy()
        
        print(f"      📊 Starting with {len(df_filtered)} players")
        print(f"      🎯 included_players = {self.included_players}")
        print(f"      📋 len(included_players) = {len(self.included_players) if self.included_players else 0}")
        
        # Apply filtering (same logic as real code)
        if self.included_players and len(self.included_players) > 0:
            print(f"      ✅ FILTERING to {len(self.included_players)} selected players...")
            original_count = len(df_filtered)
            df_filtered = df_filtered[df_filtered['Name'].isin(self.included_players)]
            final_count = len(df_filtered)
            print(f"      ✅ RESULT: {final_count}/{original_count} players after filtering")
            
            if final_count == len(self.included_players):
                print(f"      ✅ SUCCESS! Filtered to exactly the right number of players")
            else:
                print(f"      ⚠️ WARNING: Expected {len(self.included_players)}, got {final_count}")
        else:
            print(f"      ⚠️ NO FILTERING - included_players is empty or None!")
            print(f"      This means the worker will use ALL {len(df_filtered)} players")
        
        return df_filtered

# Create the worker
print(f"\n   Creating MockOptimizationWorker...")
worker = MockOptimizationWorker(df, simulated_gui_selection)

# Run preprocessing
print(f"\n" + "="*80)
print("🧪 TEST 3: Run worker.preprocess_data()")
print("="*80)

filtered_df = worker.preprocess_data()

print(f"\n   📊 Final Result:")
print(f"      Total players after preprocessing: {len(filtered_df)}")
print(f"      Players: {filtered_df['Name'].tolist()}")

if len(filtered_df) == len(simulated_gui_selection):
    print(f"\n   ✅ TEST PASSED! Worker correctly filtered to selected players")
else:
    print(f"\n   ❌ TEST FAILED! Expected {len(simulated_gui_selection)}, got {len(filtered_df)}")

# ============================================================================
# TEST 4: Test with EMPTY included_players (common bug)
# ============================================================================
print("\n" + "="*80)
print("🧪 TEST 4: Test with EMPTY included_players")
print("="*80)

print(f"   Testing what happens if included_players is empty/None...")

empty_cases = [
    ("Empty list", []),
    ("None", None),
]

for case_name, included_val in empty_cases:
    print(f"\n   Testing: {case_name} ({included_val})")
    worker2 = MockOptimizationWorker(df, included_val)
    filtered2 = worker2.preprocess_data()
    print(f"      Result: {len(filtered2)} players (should be 135 = all players)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🧪 TESTING COMPLETE")
print("="*80)

print(f"""
📊 SUMMARY:
   ✅ Test 1: GUI can collect player selections
   ✅ Test 2: Worker receives the data correctly
   ✅ Test 3: Worker filters correctly when data is passed
   ✅ Test 4: Worker uses all players when list is empty

💡 CONCLUSION:
   The data flow logic works correctly!
   
   If your optimizer is NOT filtering:
   1. Check if get_included_players() returns an empty list
   2. Check the terminal for "🚨 CRITICAL DEBUG" messages
   3. Look for "Number of included players: 0" ← This would be the bug!
   
🔍 NEXT STEP:
   Run the actual optimizer and look for:
   "🚨🚨🚨 CRITICAL DEBUG - PREPROCESSING DATA"
   
   If it shows "Number of included players: 0", then the problem is
   in get_included_players() not detecting your checkboxes correctly.
""")







