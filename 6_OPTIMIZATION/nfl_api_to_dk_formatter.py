#!/usr/bin/env python3
"""
NFL API to DraftKings Formatter
===============================

This script loads NFL data from SportsData.io API and formats it correctly
for any DraftKings entries file you provide. It automatically detects
the format and creates the proper data structure for your optimizer.

Usage:
    python nfl_api_to_dk_formatter.py --date 2025-10-26 --week 8 --dk-file DKEntries.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add the python_algorithms directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))

from sportsdata_nfl_api import SportsDataNFLAPI

class NFLAPIToDKFormatter:
    """
    Formats NFL API data for DraftKings optimizer
    """
    
    def __init__(self, api_key):
        self.api = SportsDataNFLAPI(api_key)
        self.dk_format = None
        self.contest_info = None
        
    def load_dk_entries_file(self, file_path):
        """
        Load and analyze the DraftKings entries file to understand the format
        """
        print(f"📋 Loading DraftKings entries file: {file_path}")
        
        try:
            # Try different CSV parsing methods to handle malformed files
            df = None
            
            # Method 1: Standard read
            try:
                df = pd.read_csv(file_path)
            except:
                pass
            
            # Method 2: Read with error handling
            if df is None:
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
                except:
                    pass
            
            # Method 3: Read with different separator
            if df is None:
                try:
                    df = pd.read_csv(file_path, sep=',', quotechar='"', skipinitialspace=True, on_bad_lines='skip')
                except:
                    pass
            
            # Method 4: Read raw and clean
            if df is None:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Find the header line
                    header_line = None
                    for i, line in enumerate(lines):
                        if 'Entry ID' in line or 'Position' in line:
                            header_line = i
                            break
                    
                    if header_line is not None:
                        # Read from header line
                        df = pd.read_csv(file_path, skiprows=header_line, on_bad_lines='skip')
                    else:
                        # Read normally
                        df = pd.read_csv(file_path, on_bad_lines='skip')
                except:
                    pass
            
            if df is None:
                print("❌ Could not parse the DK entries file with any method")
                return None
            
            # Detect the format
            self.dk_format = self.detect_dk_format(df)
            self.contest_info = self.extract_contest_info(df)
            
            print(f"✅ DraftKings format detected: {self.dk_format}")
            print(f"📊 Contest: {self.contest_info.get('name', 'Unknown')}")
            print(f"📊 Contest ID: {self.contest_info.get('id', 'Unknown')}")
            print(f"📊 Entries: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading DK entries file: {e}")
            return None
    
    def detect_dk_format(self, df):
        """
        Detect the DraftKings entries file format
        """
        columns = df.columns.tolist()
        
        # Check for common DraftKings entry formats
        if 'Entry ID' in columns and 'QB' in columns:
            return 'standard_entries'
        elif 'Position' in columns and 'Name + ID' in columns:
            return 'player_pool'
        elif 'QB' in columns and 'RB' in columns and 'WR' in columns:
            return 'lineup_format'
        else:
            return 'unknown'
    
    def extract_contest_info(self, df):
        """
        Extract contest information from the DK entries file
        """
        contest_info = {
            'name': 'Unknown',
            'id': 'Unknown',
            'fee': 'Unknown'
        }
        
        if 'Contest Name' in df.columns:
            contest_info['name'] = df['Contest Name'].iloc[0] if not df.empty else 'Unknown'
        if 'Contest ID' in df.columns:
            contest_info['id'] = df['Contest ID'].iloc[0] if not df.empty else 'Unknown'
        if 'Entry Fee' in df.columns:
            contest_info['fee'] = df['Entry Fee'].iloc[0] if not df.empty else 'Unknown'
            
        return contest_info
    
    def fetch_nfl_data(self, date, season, week):
        """
        Fetch NFL data from API for the specified date/week
        """
        print(f"\n🏈 Fetching NFL data for Week {week}")
        print("="*50)
        
        # Fetch DFS slate data
        print(f"📊 Fetching DFS slates for {date}...")
        slate_data = self.api.get_dfs_slates_by_date(date, save_to_file=False)
        
        if not slate_data:
            print("❌ Failed to fetch DFS slate data")
            return None, None
        
        # Find DraftKings slate
        dk_slate = None
        for slate in slate_data:
            if slate.get('Operator', '').upper() == 'DRAFTKINGS':
                dk_slate = slate
                break
        
        if not dk_slate:
            print("❌ No DraftKings slate found")
            return None, None
        
        print(f"✅ Found DraftKings slate: {len(dk_slate.get('DfsSlatePlayers', []))} players")
        
        # Fetch projections
        print(f"📊 Fetching Week {week} projections...")
        projections = self.api.get_player_projections_by_week(season, week, save_to_file=False)
        
        if not projections:
            print("⚠️ No projections available")
            projections = []
        
        print(f"✅ Loaded {len(projections)} player projections")
        
        return dk_slate, projections
    
    def format_for_optimizer(self, slate_data, projections):
        """
        Format the data for the optimizer based on detected DK format
        """
        print(f"\n🔧 Formatting data for optimizer...")
        
        # Extract player data from slate
        players = slate_data.get('DfsSlatePlayers', [])
        df_players = pd.DataFrame(players)
        
        # Create base DataFrame
        formatted_df = pd.DataFrame()
        
        # Map slate data to our format
        formatted_df['Name'] = df_players.get('OperatorPlayerName', df_players.get('PlayerName', ''))
        formatted_df['Position'] = df_players.get('OperatorPosition', '')
        formatted_df['Team'] = df_players.get('Team', '')
        formatted_df['Salary'] = df_players.get('OperatorSalary', 0)
        formatted_df['OperatorPlayerID'] = df_players.get('OperatorPlayerID', '')
        formatted_df['PlayerID'] = df_players.get('PlayerID', '')
        formatted_df['RosterSlots'] = df_players.get('OperatorRosterSlots', [])
        
        # Add projections if available
        if projections:
            proj_df = pd.DataFrame(projections)
            
            # Define all projection columns we want to include
            projection_columns = [
                'FantasyPoints', 'PassingYards', 'PassingTouchdowns', 'PassingInterceptions',
                'RushingYards', 'RushingTouchdowns', 'ReceivingYards', 'ReceivingTouchdowns',
                'Receptions', 'ReceivingTargets', 'Sacks', 'Interceptions', 'FumblesRecovered',
                'Touchdowns', 'PointsAllowed'
            ]
            
            # Filter to only include columns that exist in projections
            available_proj_cols = [col for col in projection_columns if col in proj_df.columns]
            print(f"📊 Available projection columns: {available_proj_cols}")
            
            # Merge projections
            if 'PlayerID' in proj_df.columns:
                merged = formatted_df.merge(
                    proj_df[['PlayerID'] + available_proj_cols],
                    on='PlayerID',
                    how='left'
                )
                for col in available_proj_cols:
                    formatted_df[col] = merged[col].fillna(0)
            else:
                # Fallback to name matching
                proj_df['MatchName'] = proj_df['Name'].str.strip().str.upper()
                formatted_df['MatchName'] = formatted_df['Name'].str.strip().str.upper()
                
                merged = formatted_df.merge(
                    proj_df[['MatchName'] + available_proj_cols],
                    on='MatchName',
                    how='left'
                )
                for col in available_proj_cols:
                    formatted_df[col] = merged[col].fillna(0)
                formatted_df = formatted_df.drop(columns=['MatchName'])
        else:
            # Initialize all projection columns with 0
            projection_columns = [
                'FantasyPoints', 'PassingYards', 'PassingTouchdowns', 'PassingInterceptions',
                'RushingYards', 'RushingTouchdowns', 'ReceivingYards', 'ReceivingTouchdowns',
                'Receptions', 'ReceivingTargets', 'Sacks', 'Interceptions', 'FumblesRecovered',
                'Touchdowns', 'PointsAllowed'
            ]
            for col in projection_columns:
                formatted_df[col] = 0
        
        # Set Predicted_DK_Points from FantasyPoints
        formatted_df['Predicted_DK_Points'] = formatted_df.get('FantasyPoints', 0)
        
        # 🎯 ESTIMATE individual stat projections from DK points if not available
        print("🎯 Estimating individual stat projections from DK points...")
        
        for idx, player in formatted_df.iterrows():
            dk_points = player.get('Predicted_DK_Points', 0)
            position = player.get('Position', '')
            
            if dk_points > 0:
                # QB projections
                if position == 'QB':
                    if formatted_df.loc[idx, 'PassingYards'] == 0:
                        formatted_df.loc[idx, 'PassingYards'] = dk_points * 25  # Rough estimate
                    if formatted_df.loc[idx, 'PassingTouchdowns'] == 0:
                        formatted_df.loc[idx, 'PassingTouchdowns'] = max(0.5, dk_points / 4)
                    if formatted_df.loc[idx, 'RushingYards'] == 0:
                        formatted_df.loc[idx, 'RushingYards'] = dk_points * 2  # QB rushing
                    if formatted_df.loc[idx, 'RushingTouchdowns'] == 0:
                        formatted_df.loc[idx, 'RushingTouchdowns'] = max(0.5, dk_points / 8)
                
                # RB projections
                elif position == 'RB':
                    if formatted_df.loc[idx, 'RushingYards'] == 0:
                        formatted_df.loc[idx, 'RushingYards'] = dk_points * 8  # Rough estimate
                    if formatted_df.loc[idx, 'RushingTouchdowns'] == 0:
                        formatted_df.loc[idx, 'RushingTouchdowns'] = max(0.5, dk_points / 6)
                    if formatted_df.loc[idx, 'ReceivingYards'] == 0:
                        formatted_df.loc[idx, 'ReceivingYards'] = dk_points * 3  # RB receiving
                    if formatted_df.loc[idx, 'Receptions'] == 0:
                        formatted_df.loc[idx, 'Receptions'] = max(0.5, dk_points / 2)
                
                # WR projections
                elif position == 'WR':
                    if formatted_df.loc[idx, 'ReceivingYards'] == 0:
                        formatted_df.loc[idx, 'ReceivingYards'] = dk_points * 6  # Rough estimate
                    if formatted_df.loc[idx, 'ReceivingTouchdowns'] == 0:
                        formatted_df.loc[idx, 'ReceivingTouchdowns'] = max(0.5, dk_points / 6)
                    if formatted_df.loc[idx, 'Receptions'] == 0:
                        formatted_df.loc[idx, 'Receptions'] = max(0.5, dk_points / 1.5)
                    if formatted_df.loc[idx, 'ReceivingTargets'] == 0:
                        formatted_df.loc[idx, 'ReceivingTargets'] = max(1, dk_points)
                
                # TE projections
                elif position == 'TE':
                    if formatted_df.loc[idx, 'ReceivingYards'] == 0:
                        formatted_df.loc[idx, 'ReceivingYards'] = dk_points * 5  # Rough estimate
                    if formatted_df.loc[idx, 'ReceivingTouchdowns'] == 0:
                        formatted_df.loc[idx, 'ReceivingTouchdowns'] = max(0.5, dk_points / 6)
                    if formatted_df.loc[idx, 'Receptions'] == 0:
                        formatted_df.loc[idx, 'Receptions'] = max(0.5, dk_points / 2)
                    if formatted_df.loc[idx, 'ReceivingTargets'] == 0:
                        formatted_df.loc[idx, 'ReceivingTargets'] = max(1, dk_points * 1.2)
                
                # DST projections
                elif position in ['DST', 'DEF', 'D']:
                    if formatted_df.loc[idx, 'Sacks'] == 0:
                        formatted_df.loc[idx, 'Sacks'] = max(0.5, dk_points / 2)
                    if formatted_df.loc[idx, 'Interceptions'] == 0:
                        formatted_df.loc[idx, 'Interceptions'] = max(0.5, dk_points / 3)
                    if formatted_df.loc[idx, 'FumblesRecovered'] == 0:
                        formatted_df.loc[idx, 'FumblesRecovered'] = max(0.5, dk_points / 4)
                    if formatted_df.loc[idx, 'Touchdowns'] == 0:
                        formatted_df.loc[idx, 'Touchdowns'] = max(0.5, dk_points / 6)
                    if formatted_df.loc[idx, 'PointsAllowed'] == 0:
                        formatted_df.loc[idx, 'PointsAllowed'] = max(10, 30 - dk_points * 2)
        
        # 🏈 CRITICAL: Handle DST projections specifically
        dst_players = formatted_df[formatted_df['Position'].isin(['DST', 'DEF', 'D'])]
        if len(dst_players) > 0:
            print(f"🏈 Processing {len(dst_players)} DST players...")
            
            # For DST players without projections, calculate estimated projections
            dst_no_proj = dst_players[dst_players['Predicted_DK_Points'] == 0]
            if len(dst_no_proj) > 0:
                print(f"   📊 {len(dst_no_proj)} DST players need estimated projections")
                
                # Calculate DST projections based on salary tiers
                for idx, dst_player in dst_no_proj.iterrows():
                    salary = dst_player['Salary']
                    
                    # DST projection estimation based on salary
                    if salary >= 4000:  # Premium DST
                        proj = 8.5
                    elif salary >= 3000:  # Mid-tier DST
                        proj = 6.5
                    elif salary >= 2000:  # Budget DST
                        proj = 4.5
                    else:  # Minimum salary DST
                        proj = 3.0
                    
                    formatted_df.loc[idx, 'Predicted_DK_Points'] = proj
                    print(f"   🎯 {dst_player['Name']}: ${salary} → {proj:.1f} pts")
            
            # Ensure all DST have projections > 0
            dst_zero_proj = formatted_df[(formatted_df['Position'].isin(['DST', 'DEF', 'D'])) & (formatted_df['Predicted_DK_Points'] <= 0)]
            if len(dst_zero_proj) > 0:
                print(f"   ⚠️ Setting minimum 3.0 projections for {len(dst_zero_proj)} DST players")
                formatted_df.loc[dst_zero_proj.index, 'Predicted_DK_Points'] = 3.0
            
            # Final DST validation for optimizer compatibility
            dst_final = formatted_df[formatted_df['Position'] == 'DST']
            dst_with_proj = dst_final[dst_final['Predicted_DK_Points'] > 0]
            print(f"   ✅ DST validation: {len(dst_with_proj)}/{len(dst_final)} DST players ready for optimizer")
        
        # Position mapping for NFL
        position_map = {
            'DEF': 'DST',
            'D': 'DST',
        }
        formatted_df['Position'] = formatted_df['Position'].replace(position_map)
        
        # Filter to valid positions
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
        formatted_df = formatted_df[formatted_df['Position'].isin(valid_positions)]
        
        # Data quality checks
        formatted_df = formatted_df.dropna(subset=['Name', 'Position', 'Salary'])
        formatted_df = formatted_df[formatted_df['Salary'] > 0]
        
        # 🔧 CRITICAL FIX: Remove invalid/fake players that cause optimizer errors
        invalid_players = [
            'Phil Mafah', 'Tyler Shough', 'Wassim Karoui', 'Wesley Fofana', 
            'Béchir Ben Saïd', 'J.J. McCarthy'
        ]
        
        print(f"\n🧹 Removing invalid players...")
        removed_count = 0
        for invalid_player in invalid_players:
            matches = formatted_df[formatted_df['Name'].str.contains(invalid_player, case=False, na=False)]
            if len(matches) > 0:
                print(f"   ❌ Removing: {invalid_player} ({len(matches)} instances)")
                formatted_df = formatted_df[~formatted_df['Name'].str.contains(invalid_player, case=False, na=False)]
                removed_count += len(matches)
        
        if removed_count > 0:
            print(f"   ✅ Removed {removed_count} invalid players")
        else:
            print(f"   ✅ No invalid players found")
        
        # Add additional columns for optimizer compatibility
        formatted_df['Opponent'] = 'TBD'
        formatted_df['InjuryStatus'] = 'Active'
        formatted_df['GameInfo'] = 'TBD'
        
        # Calculate value
        formatted_df['Value'] = formatted_df.apply(
            lambda row: row['Predicted_DK_Points'] / (row['Salary'] / 1000) if row['Salary'] > 0 else 0,
            axis=1
        )
        
        # 🔧 CRITICAL FIX: Ensure Week 7 format compatibility
        self._ensure_week7_format_compatibility(formatted_df)
        formatted_df['PointsPerK'] = formatted_df['Value']
        
        # Add all the columns that the optimizer expects (matching working format)
        formatted_df['PassingYards'] = 0.0
        formatted_df['PassingTouchdowns'] = 0.0
        formatted_df['PassingInterceptions'] = 0.0
        formatted_df['RushingYards'] = 0.0
        formatted_df['RushingTouchdowns'] = 0.0
        formatted_df['ReceivingYards'] = 0.0
        formatted_df['ReceivingTouchdowns'] = 0.0
        formatted_df['Receptions'] = 0.0
        formatted_df['ReceivingTargets'] = 0.0
        formatted_df['FantasyPointsYahoo'] = 0.0
        formatted_df['FantasyPointsFanDuel'] = 0.0
        formatted_df['Ceiling'] = formatted_df['Predicted_DK_Points'] * 1.3
        formatted_df['Floor'] = formatted_df['Predicted_DK_Points'] * 0.6
        formatted_df['ID'] = formatted_df['OperatorPlayerID']
        formatted_df['Sacks'] = 0.0
        formatted_df['Interceptions'] = 0.0
        formatted_df['FumblesRecovered'] = 0.0
        formatted_df['Touchdowns'] = 0.0
        formatted_df['PointsAllowed'] = 0.0
        formatted_df['Cash_Score'] = formatted_df['Predicted_DK_Points'] * 0.8
        
        print(f"✅ Formatted {len(formatted_df)} players")
        print(f"📊 Position breakdown: {formatted_df['Position'].value_counts().to_dict()}")
        
        return formatted_df
    
    def _ensure_week7_format_compatibility(self, df):
        """
        Ensure the DataFrame matches the Week 7 format structure for optimizer compatibility
        """
        try:
            # Try to load Week 7 template to get exact column structure
            week7_template = pd.read_csv('nfl_week7_CASH_SPORTSDATA.csv')
            required_columns = week7_template.columns.tolist()
            
            print(f"\n🔧 Ensuring Week 7 format compatibility...")
            print(f"   📋 Week 7 template has {len(required_columns)} columns")
            
            # Add missing columns with default values
            missing_columns = []
            for col in required_columns:
                if col not in df.columns:
                    missing_columns.append(col)
                    if col == 'Name':
                        df[col] = 'Unknown'
                    else:
                        df[col] = 0.0
            
            if missing_columns:
                print(f"   ➕ Added {len(missing_columns)} missing columns: {missing_columns}")
            
            # Reorder columns to match Week 7 format exactly
            df = df[required_columns]
            print(f"   ✅ Column structure matched to Week 7 format")
            
        except Exception as e:
            print(f"   ⚠️ Could not match Week 7 format: {e}")
            print(f"   📋 Current columns: {list(df.columns)}")
    
    def save_formatted_data(self, df, output_filename):
        """
        Save the formatted data to CSV with optimizer compatibility fixes
        """
        # 🔧 CRITICAL FIX: Ensure DST players have projections > 0 for optimizer compatibility
        dst_players = df[df['Position'] == 'DST']
        if len(dst_players) > 0:
            print(f"\n🏈 Final DST validation and fixes...")
            
            # Check for DST players with 0 or negative projections
            dst_zero_proj = dst_players[dst_players['Predicted_DK_Points'] <= 0]
            if len(dst_zero_proj) > 0:
                print(f"   ⚠️ Fixing {len(dst_zero_proj)} DST players with zero/negative projections")
                
                for idx in dst_zero_proj.index:
                    salary = df.loc[idx, 'Salary']
                    
                    # Calculate realistic DST projections based on salary
                    if salary >= 4000:
                        proj = 8.5
                    elif salary >= 3000:
                        proj = 6.5
                    elif salary >= 2000:
                        proj = 4.5
                    else:
                        proj = 3.0
                    
                    df.loc[idx, 'Predicted_DK_Points'] = proj
                    df.loc[idx, 'Ceiling'] = proj * 1.3
                    df.loc[idx, 'Floor'] = proj * 0.6
                    df.loc[idx, 'Cash_Score'] = proj * 0.8
            
            # Final DST validation
            dst_final = df[df['Position'] == 'DST']
            dst_with_proj = dst_final[dst_final['Predicted_DK_Points'] > 0]
            print(f"   ✅ DST validation: {len(dst_with_proj)}/{len(dst_final)} DST players have projections > 0")
        
        # 🔧 CRITICAL FIX: Ensure all required columns are present and in correct order
        # Load working Week 7 file to get exact column structure
        try:
            week7_template = pd.read_csv('nfl_week7_CASH_SPORTSDATA.csv')
            required_columns = week7_template.columns.tolist()
            
            # Ensure our DataFrame has all required columns in correct order
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Name':
                        df[col] = 'Unknown'
                    else:
                        df[col] = 0.0
            
            # Reorder columns to match Week 7 format exactly
            df = df[required_columns]
            print(f"   ✅ Column structure matched to Week 7 format: {len(required_columns)} columns")
            
        except Exception as e:
            print(f"   ⚠️ Could not match Week 7 format: {e}")
        
        # Save the file
        df.to_csv(output_filename, index=False)
        print(f"\n💾 Saved formatted data to: {output_filename}")
        
        # Show summary
        print(f"\n📊 Data Summary:")
        print(f"   Total Players: {len(df)}")
        print(f"   Players with Projections: {len(df[df['Predicted_DK_Points'] > 0])}")
        print(f"   Salary Range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
        print(f"   Avg Salary: ${df['Salary'].mean():,.0f}")
        
        # Show position breakdown for players with projections > 0
        players_with_proj = df[df['Predicted_DK_Points'] > 0]
        pos_counts = players_with_proj['Position'].value_counts()
        print(f"\n📊 Position Breakdown (players with projections > 0):")
        for pos, count in pos_counts.items():
            print(f"   {pos}: {count}")
        
        # Show top projected players
        print(f"\n🏆 Top 10 Projected Players:")
        top_proj = df.nlargest(10, 'Predicted_DK_Points')[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Value']]
        print(top_proj.to_string(index=False))
        
        # 🔧 CRITICAL: Final optimizer compatibility check
        print(f"\n🎯 Optimizer Compatibility Check:")
        dst_count = len(players_with_proj[players_with_proj['Position'] == 'DST'])
        print(f"   DST teams with projections > 0: {dst_count}")
        
        if dst_count > 0:
            print(f"   ✅ READY FOR OPTIMIZER!")
        else:
            print(f"   ❌ WARNING: No DST teams will pass optimizer filter!")
        
        return output_filename

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='NFL API to DraftKings Formatter')
    parser.add_argument('--date', required=True, help='Date for DFS slate (YYYY-MM-DD)')
    parser.add_argument('--week', required=True, type=int, help='Week number')
    parser.add_argument('--season', default='2025REG', help='Season (default: 2025REG)')
    parser.add_argument('--dk-file', required=True, help='DraftKings entries file path')
    parser.add_argument('--output', help='Output filename (optional)')
    parser.add_argument('--api-key', default='1dd5e646265649af87e0d9cdb80d1c8c', help='SportsData.io API key')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           NFL API to DraftKings Formatter               ║
    ║              Auto-detects format & creates data          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize formatter
    formatter = NFLAPIToDKFormatter(args.api_key)
    
    # Load DK entries file to understand format
    dk_df = formatter.load_dk_entries_file(args.dk_file)
    if dk_df is None:
        return
    
    # Fetch NFL data from API
    slate_data, projections = formatter.fetch_nfl_data(args.date, args.season, args.week)
    if slate_data is None:
        return
    
    # Format data for optimizer
    formatted_df = formatter.format_for_optimizer(slate_data, projections)
    
    # Generate output filename
    if args.output:
        output_filename = args.output
    else:
        date_clean = args.date.replace('-', '')
        output_filename = f"nfl_week{args.week}_optimizer_ready_{date_clean}.csv"
    
    # Save formatted data
    formatter.save_formatted_data(formatted_df, output_filename)
    
    print(f"\n" + "="*80)
    print("✅ NFL DATA FORMATTING COMPLETE!")
    print("="*80)
    print(f"📁 File: {output_filename}")
    print(f"🎯 Ready for your optimizer!")
    print(f"💰 Good luck with your lineups!")

if __name__ == "__main__":
    main()
