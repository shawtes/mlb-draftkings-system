#!/usr/bin/env python3
"""
Test script to verify player filtering logic
"""

import pandas as pd
import sys

print("\n" + "="*80)
print("🧪 TESTING PLAYER FILTERING LOGIC")
print("="*80)

# Load the CSV
csv_file = "../nba_NOV1_READY.csv"
print(f"\n📂 Loading: {csv_file}")

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} players")
    print(f"   Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

# Show first few players
print(f"\n📊 First 5 players in CSV:")
for idx, row in df.head().iterrows():
    print(f"   {idx+1}. {row['Name']} ({row['Position']}, {row['Team']})")

# TEST 1: Simulate selecting just Jimmy Butler
print("\n" + "="*80)
print("🧪 TEST 1: Filter to ONLY Jimmy Butler III")
print("="*80)

included_players = ["Jimmy Butler III"]
print(f"   🎯 Included players: {included_players}")
print(f"   📊 Original dataframe size: {len(df)}")

# Apply the filtering logic
filtered_df = df[df['Name'].isin(included_players)]
print(f"   ✅ Filtered dataframe size: {len(filtered_df)}")

if len(filtered_df) == 1:
    print(f"   ✅ SUCCESS! Only Jimmy Butler III in filtered pool:")
    print(f"      {filtered_df['Name'].tolist()}")
else:
    print(f"   ❌ FAILED! Expected 1 player, got {len(filtered_df)}")
    print(f"      Players: {filtered_df['Name'].tolist()}")

# TEST 2: Simulate selecting multiple players
print("\n" + "="*80)
print("🧪 TEST 2: Filter to 3 specific players")
print("="*80)

included_players = ["Jimmy Butler III", "LaMelo Ball", "Anthony Edwards"]
print(f"   🎯 Included players: {included_players}")
print(f"   📊 Original dataframe size: {len(df)}")

# Apply the filtering logic
filtered_df = df[df['Name'].isin(included_players)]
print(f"   ✅ Filtered dataframe size: {len(filtered_df)}")

if len(filtered_df) == 3:
    print(f"   ✅ SUCCESS! Only selected players in filtered pool:")
    for name in filtered_df['Name'].tolist():
        print(f"      - {name}")
else:
    print(f"   ⚠️ WARNING! Expected 3 players, got {len(filtered_df)}")
    print(f"      Players found: {filtered_df['Name'].tolist()}")
    
    # Check which ones are missing
    found_names = set(filtered_df['Name'].tolist())
    missing = [p for p in included_players if p not in found_names]
    if missing:
        print(f"      ❌ Missing players: {missing}")
        print(f"      💡 These names might not match exactly in the CSV")

# TEST 3: Check if all names in CSV
print("\n" + "="*80)
print("🧪 TEST 3: Verify all player names in CSV")
print("="*80)

all_names = df['Name'].tolist()
print(f"   📋 Total names in CSV: {len(all_names)}")
print(f"\n   🔍 Searching for specific players:")

test_names = ["Jimmy Butler III", "Jimmy Butler", "LaMelo Ball", "Anthony Edwards"]
for name in test_names:
    if name in all_names:
        print(f"      ✅ FOUND: {name}")
    else:
        print(f"      ❌ NOT FOUND: {name}")
        # Try to find similar names
        similar = [n for n in all_names if name.split()[0] in n]
        if similar:
            print(f"         💡 Similar names: {similar[:3]}")

# TEST 4: Show all Guards
print("\n" + "="*80)
print("🧪 TEST 4: Show all Guards (to find Jimmy Butler)")
print("="*80)

if 'Position' in df.columns:
    guards = df[df['Position'].str.contains('G', na=False)]
    print(f"   📊 Found {len(guards)} guards:")
    for idx, row in guards.iterrows():
        print(f"      {row['Name']} ({row['Position']}, {row['Team']})")
else:
    print(f"   ⚠️ No 'Position' column found")
    print(f"   Available columns: {df.columns.tolist()}")

print("\n" + "="*80)
print("🧪 TESTING COMPLETE")
print("="*80)
print("\n💡 If TEST 1 shows SUCCESS, the filtering logic works correctly.")
print("   The problem is somewhere else in the optimizer code.\n")

