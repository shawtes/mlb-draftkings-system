#!/usr/bin/env python3
"""
Example: Export NFL DFS Players to CSV
=======================================

This script demonstrates how to use the NFL API client to fetch
DFS player data and export it to CSV format for use with optimizers.

Usage:
    python export_dfs_to_csv_example.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def main():
    """Example usage of export_dfs_players_to_csv"""
    
    # API Key (already configured in your system)
    API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        NFL DFS Players to CSV Export Example              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize API client
    api = SportsDataNFLAPI(API_KEY)
    
    # Example: Export Week 8 DFS players to CSV
    # You can modify these parameters:
    date = "2025-10-26"  # Date of the slate (YYYY-MM-DD)
    season = "2025REG"    # Season format: YYYYREG, YYYYPRE, YYYYPOST
    week = 8              # Week number
    
    print(f"\n📊 Exporting DFS players for:")
    print(f"   Date: {date}")
    print(f"   Season: {season}")
    print(f"   Week: {week}")
    print()
    
    # Export to CSV
    csv_file = api.export_dfs_players_to_csv(
        date=date,
        season=season,
        week=week,
        include_projections=True  # Set to False if you don't want projections
    )
    
    if csv_file:
        print(f"\n✅ Success! CSV file saved to: {csv_file}")
        print(f"\n💡 You can now use this CSV file with your optimizer!")
    else:
        print("\n❌ Failed to export CSV. Check the error messages above.")
    
    # Example 2: Export without projections (faster, just salaries)
    print("\n" + "="*70)
    print("Example 2: Export without projections (salaries only)")
    print("="*70)
    
    csv_file_no_proj = api.export_dfs_players_to_csv(
        date=date,
        season=season,
        week=week,
        filename="nfl_dfs_salaries_only.csv",
        include_projections=False
    )
    
    if csv_file_no_proj:
        print(f"\n✅ Salaries-only CSV saved to: {csv_file_no_proj}")

if __name__ == "__main__":
    main()


