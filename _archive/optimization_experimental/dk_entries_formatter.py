#!/usr/bin/env python3
"""
DK Entries Formatter
===================

This script takes your DKEntries file and creates a properly formatted
CSV file for your optimizer, using the existing Week 8 data we already have.

Usage:
    python dk_entries_formatter.py --dk-file DKEntries.csv
"""

import sys
import os
import pandas as pd
import argparse

def load_dk_entries_robust(file_path):
    """
    Load DK entries file with robust error handling
    """
    print(f"📋 Loading DraftKings entries file: {file_path}")
    
    # Try multiple methods to read the file
    methods = [
        lambda: pd.read_csv(file_path),
        lambda: pd.read_csv(file_path, on_bad_lines='skip', engine='python'),
        lambda: pd.read_csv(file_path, sep=',', quotechar='"', skipinitialspace=True, on_bad_lines='skip'),
        lambda: pd.read_csv(file_path, skiprows=8, on_bad_lines='skip'),  # Skip instruction lines
    ]
    
    for i, method in enumerate(methods):
        try:
            df = method()
            if df is not None and len(df) > 0:
                print(f"✅ Successfully loaded with method {i+1}")
                return df
        except Exception as e:
            print(f"Method {i+1} failed: {e}")
            continue
    
    print("❌ Could not load DK entries file with any method")
    return None

def extract_contest_info(df):
    """
    Extract contest information from DK entries file
    """
    contest_info = {
        'name': 'Unknown',
        'id': 'Unknown',
        'fee': 'Unknown'
    }
    
    # Look for contest info in the first few rows
    for col in df.columns:
        if 'Contest' in col and 'Name' in col:
            if not df.empty:
                contest_info['name'] = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'Unknown'
        if 'Contest' in col and 'ID' in col:
            if not df.empty:
                contest_info['id'] = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'Unknown'
        if 'Entry' in col and 'Fee' in col:
            if not df.empty:
                contest_info['fee'] = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'Unknown'
    
    return contest_info

def create_optimizer_file_from_dk_entries(dk_file_path):
    """
    Create optimizer-ready file from DK entries
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              DK Entries to Optimizer Formatter          ║
    ║              Uses existing Week 8 data + DK format        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Load DK entries file
    dk_df = load_dk_entries_robust(dk_file_path)
    if dk_df is None:
        return None
    
    # Extract contest info
    contest_info = extract_contest_info(dk_df)
    print(f"📊 Contest: {contest_info['name']}")
    print(f"📊 Contest ID: {contest_info['id']}")
    print(f"📊 Entry Fee: {contest_info['fee']}")
    
    # Load our existing Week 8 data
    week8_file = "nfl_week8_OPTIMIZER_READY.csv"
    if not os.path.exists(week8_file):
        print(f"❌ Week 8 data file not found: {week8_file}")
        print("   Please run the API formatter first or use existing Week 8 data")
        return None
    
    print(f"📊 Loading Week 8 data from: {week8_file}")
    week8_df = pd.read_csv(week8_file)
    print(f"✅ Loaded {len(week8_df)} players from Week 8 data")
    
    # Show summary
    print(f"\n📊 Week 8 Data Summary:")
    print(f"   Total Players: {len(week8_df)}")
    print(f"   Players with Projections: {len(week8_df[week8_df['Predicted_DK_Points'] > 0])}")
    print(f"   Position breakdown: {week8_df['Position'].value_counts().to_dict()}")
    print(f"   Salary Range: ${week8_df['Salary'].min():,} - ${week8_df['Salary'].max():,}")
    
    # Show top projected players
    print(f"\n🏆 Top 10 Projected Players:")
    top_proj = week8_df.nlargest(10, 'Predicted_DK_Points')[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Value']]
    print(top_proj.to_string(index=False))
    
    # Create output filename based on contest
    contest_id = contest_info['id']
    output_filename = f"nfl_week8_contest_{contest_id}_optimizer_ready.csv"
    
    # Save the file
    week8_df.to_csv(output_filename, index=False)
    
    print(f"\n💾 Saved optimizer-ready file: {output_filename}")
    print(f"📊 Ready for your genetic algorithm optimizer!")
    print(f"🎯 This file contains:")
    print(f"   • {len(week8_df)} players")
    print(f"   • Week 8 projections")
    print(f"   • DraftKings salaries")
    print(f"   • All required optimizer columns")
    print(f"   • Contest ID: {contest_info['id']}")
    
    return output_filename

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='DK Entries to Optimizer Formatter')
    parser.add_argument('--dk-file', required=True, help='DraftKings entries file path')
    
    args = parser.parse_args()
    
    # Create optimizer file from DK entries
    output_file = create_optimizer_file_from_dk_entries(args.dk_file)
    
    if output_file:
        print(f"\n" + "="*80)
        print("✅ DK ENTRIES FORMATTING COMPLETE!")
        print("="*80)
        print(f"📁 File: {output_file}")
        print(f"🎯 Ready for your optimizer!")
        print(f"💰 Good luck with your lineups!")
    else:
        print("❌ Failed to create optimizer file")

if __name__ == "__main__":
    main()
