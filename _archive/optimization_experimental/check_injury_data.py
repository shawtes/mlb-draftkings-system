#!/usr/bin/env python3
"""
NBA Injury Data Checker
========================
Verifies that injury status data is present in your player files
and shows current injury report from the data source
"""

import pandas as pd
import sys
import os
from pathlib import Path

def check_csv_injury_data(csv_path):
    """Check if CSV file contains injury status information"""
    print(f"\n{'='*80}")
    print(f"🏥 NBA INJURY DATA CHECK")
    print(f"{'='*80}")
    print(f"📁 File: {csv_path}")
    print()
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded {len(df)} players")
        print()
        
        # Check columns
        print(f"📋 Available columns ({len(df.columns)}):")
        for col in sorted(df.columns):
            print(f"   - {col}")
        print()
        
        # Look for injury-related columns
        injury_cols = [col for col in df.columns if 'injury' in col.lower() or 'status' in col.lower()]
        
        if not injury_cols:
            print(f"❌ NO INJURY STATUS COLUMN FOUND")
            print(f"   Your data file does not include injury information")
            print(f"   The optimizer will include all players (no filtering)")
            print()
            return False
        
        # Found injury column(s)
        injury_col = injury_cols[0]
        print(f"✅ INJURY STATUS COLUMN FOUND: '{injury_col}'")
        print()
        
        # Analyze injury statuses
        print(f"🏥 INJURY REPORT:")
        print(f"{'='*80}")
        
        # Count by status
        status_counts = df[injury_col].fillna('HEALTHY').value_counts()
        
        # Show counts
        print(f"\n📊 Status Distribution:")
        for status, count in status_counts.items():
            status_str = str(status).upper()
            emoji = "❌" if status_str in ['OUT', 'O', 'DOUBTFUL', 'D'] else "✅"
            print(f"   {emoji} {status:15s}: {count:3d} players")
        
        # Show injured players details
        injured_df = df[df[injury_col].fillna('HEALTHY').str.upper().isin(['OUT', 'DOUBTFUL', 'O', 'D'])]
        
        if len(injured_df) > 0:
            print(f"\n⚠️  INJURED PLAYERS (will be filtered out):")
            print(f"{'='*80}")
            for _, player in injured_df.iterrows():
                name = player.get('Name', 'Unknown')
                pos = player.get('Position', 'N/A')
                team = player.get('Team', 'N/A')
                status = player[injury_col]
                salary = player.get('Salary', 0)
                print(f"   {pos:4s} {name:30s} {team:5s} ${salary:5.0f}  - {status}")
        else:
            print(f"\n✅ No injured players found - all players healthy!")
        
        # Show questionable players (included but risky)
        questionable_df = df[df[injury_col].fillna('').str.upper().isin(['QUESTIONABLE', 'Q', 'PROBABLE', 'P'])]
        
        if len(questionable_df) > 0:
            print(f"\n⚠️  QUESTIONABLE/PROBABLE PLAYERS (included but game-time decisions):")
            print(f"{'='*80}")
            for _, player in questionable_df.iterrows():
                name = player.get('Name', 'Unknown')
                pos = player.get('Position', 'N/A')
                team = player.get('Team', 'N/A')
                status = player[injury_col]
                salary = player.get('Salary', 0)
                proj = player.get('Predicted_DK_Points', 0)
                print(f"   {pos:4s} {name:30s} {team:5s} ${salary:5.0f} ({proj:.1f} pts) - {status}")
        
        print(f"\n{'='*80}")
        print(f"📈 SUMMARY:")
        print(f"   Total Players: {len(df)}")
        print(f"   Injured (OUT/DOUBTFUL): {len(injured_df)}")
        print(f"   Questionable/Probable: {len(questionable_df)}")
        print(f"   Healthy: {len(df) - len(injured_df) - len(questionable_df)}")
        print(f"   ✅ Will be used in optimizer: {len(df) - len(injured_df)}")
        print(f"{'='*80}\n")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {csv_path}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_api_connection():
    """Check if NBA data fetcher can retrieve injury data"""
    print(f"\n{'='*80}")
    print(f"🔌 CHECKING NBA API CONNECTION FOR INJURY DATA")
    print(f"{'='*80}\n")
    
    try:
        from nba_sportsdata_fetcher import NBADataFetcher
        from datetime import datetime
        
        # Try to load API key
        api_key_file = Path(__file__).parent / '.api_key'
        if not api_key_file.exists():
            print(f"⚠️  No API key file found at: {api_key_file}")
            print(f"   Cannot check live API data")
            return False
        
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
        
        print(f"✅ API key loaded")
        print(f"🔄 Fetching today's NBA projections with injury data...")
        
        fetcher = NBADataFetcher(api_key)
        date_str = datetime.now().strftime('%Y-%b-%d').upper()
        
        df = fetcher.get_daily_projections(date_str)
        
        if df.empty:
            print(f"⚠️  No games today or API returned empty data")
            return False
        
        print(f"✅ Retrieved {len(df)} player projections")
        
        # Check for InjuryStatus column
        if 'InjuryStatus' in df.columns:
            print(f"✅ InjuryStatus column present in API data")
            
            # Show injury counts
            injury_counts = df['InjuryStatus'].fillna('HEALTHY').value_counts()
            print(f"\n📊 Injury Status from API:")
            for status, count in injury_counts.items():
                print(f"   {status}: {count}")
        else:
            print(f"⚠️  InjuryStatus column NOT in API response")
            print(f"   Available columns: {list(df.columns)}")
        
        return True
        
    except ImportError:
        print(f"⚠️  NBA data fetcher module not found")
        print(f"   Cannot check live API data")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    print(f"\n{'*'*80}")
    print(f"  NBA INJURY DATA VERIFICATION TOOL")
    print(f"{'*'*80}")
    
    # Check command line args
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        check_csv_injury_data(csv_path)
    else:
        # Look for CSV files in current directory
        csv_files = list(Path('.').glob('*.csv'))
        
        if csv_files:
            print(f"\n📁 Found {len(csv_files)} CSV file(s) in current directory:")
            for i, f in enumerate(csv_files, 1):
                print(f"   {i}. {f.name}")
            
            print(f"\nChecking first file: {csv_files[0].name}")
            check_csv_injury_data(csv_files[0])
        else:
            print(f"\n❌ No CSV files found in current directory")
            print(f"\nUsage: python check_injury_data.py <path_to_csv>")
            print(f"Example: python check_injury_data.py nba_players_2024-10-22.csv")
    
    # Also check API if possible
    check_api_connection()
    
    print(f"\n{'*'*80}")
    print(f"  INJURY FILTERING IN OPTIMIZER")
    print(f"{'*'*80}")
    print(f"""
✅ Injury filtering is AUTOMATIC in the NBA optimizer

When you load a CSV file, the system:
1. Scans for 'InjuryStatus' or similar columns
2. Removes OUT and DOUBTFUL players
3. Keeps QUESTIONABLE and PROBABLE players
4. Shows you a report of what was filtered

Filtered statuses (removed):
  ❌ OUT (O) - Player will not play
  ❌ DOUBTFUL (D) - < 25% chance to play

Included statuses (kept in pool):
  ✅ QUESTIONABLE (Q) - Game-time decision (~50% chance)
  ✅ PROBABLE (P) - Likely to play (>75% chance)
  ✅ HEALTHY or blank - No injury designation

You can manually exclude questionable players using the UI checkboxes
if you want to be more conservative.
""")


if __name__ == "__main__":
    main()

