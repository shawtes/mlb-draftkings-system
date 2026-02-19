#!/usr/bin/env python3
"""
Fetch Today's NFL DFS Data and Export to CSV
============================================

This script automatically fetches today's DraftKings DFS slate data
and exports it to a CSV file ready for your optimizer.

Usage:
    python fetch_todays_dfs.py
    python fetch_todays_dfs.py --date 2025-10-26 --week 8
    python fetch_todays_dfs.py --week 8  # Uses today's date
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

def get_todays_date():
    """Get today's date in the format needed for the API"""
    today = datetime.now()
    # Format: YYYY-MM-DD (e.g., "2025-10-26")
    return today.strftime("%Y-%m-%d")

def estimate_week_from_date(date_str):
    """
    Estimate NFL week from date
    This is a rough estimate - you may need to adjust based on actual NFL schedule
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # NFL regular season typically starts first week of September
        # Week 1 is usually around September 5-10
        season_start = datetime(date_obj.year, 9, 5)
        
        # Calculate days since season start
        days_diff = (date_obj - season_start).days
        
        # Rough estimate: each week is ~7 days, but account for byes
        # Week 1 = days 0-6, Week 2 = days 7-13, etc.
        estimated_week = max(1, min(18, (days_diff // 7) + 1))
        
        return estimated_week
    except:
        return 8  # Default fallback

def main():
    """Main function to fetch today's DFS data"""
    
    parser = argparse.ArgumentParser(
        description='Fetch today\'s NFL DFS data and export to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's data (auto-detects date and week)
  python fetch_todays_dfs.py
  
  # Fetch specific date and week
  python fetch_todays_dfs.py --date 2025-10-26 --week 8
  
  # Fetch today's data but specify week
  python fetch_todays_dfs.py --week 8
  
  # Fetch without projections (faster, just salaries)
  python fetch_todays_dfs.py --no-projections
        """
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--week',
        type=int,
        help='NFL week number (required if date not provided, or will estimate from date)'
    )
    
    parser.add_argument(
        '--season',
        type=str,
        default='2025REG',
        help='Season format (default: 2025REG)'
    )
    
    parser.add_argument(
        '--no-projections',
        action='store_true',
        help='Skip fetching projections (faster, just salaries)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV filename (default: auto-generated)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default='1dd5e646265649af87e0d9cdb80d1c8c',
        help='SportsData.io API key (default: uses configured key)'
    )
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        NFL DFS Data Fetcher - Today's Slate              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Determine date
    if args.date:
        target_date = args.date
        print(f"📅 Using specified date: {target_date}")
    else:
        target_date = get_todays_date()
        print(f"📅 Using today's date: {target_date}")
    
    # Determine week
    if args.week:
        week = args.week
        print(f"📊 Using specified week: {week}")
    else:
        # Try to estimate from date
        week = estimate_week_from_date(target_date)
        print(f"📊 Estimated week from date: {week}")
        print(f"⚠️  Note: Week estimation may be inaccurate. Use --week to specify exact week.")
    
    # Initialize API
    print(f"\n🔑 Initializing API client...")
    api = SportsDataNFLAPI(args.api_key)
    
    # Export to CSV
    print(f"\n🚀 Fetching DFS data...")
    csv_file = api.export_dfs_players_to_csv(
        date=target_date,
        season=args.season,
        week=week,
        filename=args.output,
        include_projections=not args.no_projections
    )
    
    if csv_file:
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS!")
        print(f"{'='*70}")
        print(f"📁 CSV file saved to: {csv_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Open {csv_file} in your optimizer")
        print(f"   2. Load the file and generate lineups")
        print(f"   3. Good luck! 🍀")
    else:
        print(f"\n{'='*70}")
        print(f"❌ FAILED TO EXPORT CSV")
        print(f"{'='*70}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check if there's a DFS slate available for {target_date}")
        print(f"   2. Verify your API key is valid")
        print(f"   3. Try specifying --week explicitly")
        print(f"   4. Check the error messages above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()



