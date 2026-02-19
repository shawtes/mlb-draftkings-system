#!/usr/bin/env python3
"""
NFL DFS Week Data Creator
Creates enhanced GPP and Cash game CSV files for any NFL week

Usage:
    python create_nfl_week_data.py --week 6
    python create_nfl_week_data.py --week 7 --date 2025-10-20
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python_algorithms'))
from sportsdata_nfl_api import SportsDataNFLAPI

# SportsData.io API Key
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"


def fetch_week_data(api, season, week, slate_date):
    """Fetch all data needed for a week"""
    
    print(f"\n🏈 Fetching NFL Week {week} Data")
    print("=" * 70)
    
    # Fetch projections
    print(f"\n1️⃣ Fetching Week {week} Projections...")
    projections = api.get_player_projections_by_week(
        season=season,
        week=week,
        save_to_file=False
    )
    
    if not projections:
        print(f"   ❌ Failed to fetch projections")
        return None, None, None, None
    
    print(f"   ✅ Loaded {len(projections)} projected players")
    
    # Fetch actuals
    print(f"\n2️⃣ Fetching Week {week} Actuals...")
    actuals = api.get_player_stats_by_week(
        season=season,
        week=week,
        save_to_file=False
    )
    
    if actuals:
        print(f"   ✅ Loaded {len(actuals)} actual player records")
    else:
        print(f"   ⚠️  No actuals available yet (games not played)")
        actuals = []
    
    # Fetch injury data
    print(f"\n3️⃣ Fetching Week {week} Injury Reports...")
    injuries = api.get_injuries_by_week(
        season=season,
        week=week,
        save_to_file=False
    )
    
    if injuries:
        print(f"   ✅ Loaded {len(injuries)} injury records")
    else:
        print(f"   ⚠️  No injury data available")
        injuries = []
    
    # Fetch DFS salaries
    print(f"\n4️⃣ Fetching DraftKings Salaries for {slate_date}...")
    slates = api.get_dfs_slates_by_date(
        date=slate_date,
        save_to_file=False
    )
    
    if not slates:
        print(f"   ❌ Failed to fetch DFS slates")
        return None, None, None, None
    
    # Find DraftKings main slate - prefer slates with more games and valid salaries
    dk_slates = [s for s in slates if s.get('Operator') == 'DraftKings']
    if not dk_slates:
        print(f"   ❌ No DraftKings slates found")
        return None, None, None, None
    
    # Score slates by: number of games * 1000 + number of players
    def score_slate(slate):
        num_games = len(slate.get('DfsSlateGames', []))
        num_players = len(slate.get('DfsSlatePlayers', []))
        # Check if players have salaries
        players_with_salary = sum(1 for p in slate.get('DfsSlatePlayers', []) if p.get('OperatorSalary', 0) > 0)
        return (num_games * 1000) + num_players if players_with_salary > 0 else 0
    
    main_slate = max(dk_slates, key=score_slate)
    print(f"   ✅ Using DraftKings Slate {main_slate['SlateID']} with {len(main_slate.get('DfsSlatePlayers', []))} players")
    
    return projections, actuals, main_slate, injuries


def create_salary_dataframe(slate):
    """Create DataFrame from DFS slate"""
    salary_data = []
    for player in slate.get('DfsSlatePlayers', []):
        salary_data.append({
            'Name': player.get('OperatorPlayerName'),
            'Position': player.get('OperatorPosition'),
            'Salary': player.get('OperatorSalary'),
            'OperatorPlayerID': player.get('OperatorPlayerID')
        })
    return pd.DataFrame(salary_data)


def filter_injured_players(df, injuries):
    """Remove injured players who are OUT or DOUBTFUL"""
    
    if not injuries or len(injuries) == 0:
        print("   ℹ️  No injury data to filter")
        return df
    
    print("\n5️⃣ Filtering injured players...")
    
    # Create injury lookup: {PlayerID: InjuryStatus}
    injury_status = {}
    for injury in injuries:
        player_id = injury.get('PlayerID')
        status = injury.get('Status', '').upper()
        injury_status[player_id] = status
    
    # Filter out players who are OUT or DOUBTFUL
    before_filter = len(df)
    injured_statuses = ['OUT', 'DOUBTFUL']
    
    # Create mask for injured players
    df['InjuryStatus'] = df['PlayerID'].map(injury_status).fillna('HEALTHY')
    
    # Filter out injured players
    injured_players = df[df['InjuryStatus'].isin(injured_statuses)]
    df = df[~df['InjuryStatus'].isin(injured_statuses)]
    
    removed_count = before_filter - len(df)
    
    if removed_count > 0:
        print(f"   ❌ Removed {removed_count} injured players (OUT/DOUBTFUL):")
        for _, player in injured_players.iterrows():
            print(f"      {player['Position']:3s} {player['Name']:25s} - {player['InjuryStatus']}")
    else:
        print(f"   ✅ No injured players to remove (all active)")
    
    print(f"   ✅ {len(df)} healthy players remaining")
    
    return df


def merge_data(df_salary, df_proj):
    """Merge salary and projection data"""
    
    print("\n4️⃣ Merging data...")
    
    # Clean names
    df_salary['Name'] = df_salary['Name'].str.strip()
    df_proj['Name'] = df_proj['Name'].str.strip()
    
    # Merge
    df = df_salary.merge(
        df_proj[['Name', 'PlayerID', 'Team', 'FantasyPoints', 'PassingYards', 'PassingTouchdowns',
                 'RushingYards', 'RushingTouchdowns', 'ReceivingYards', 'ReceivingTouchdowns']],
        on='Name',
        how='left'
    )
    
    # Fill missing projections
    df['FantasyPoints'] = df['FantasyPoints'].fillna(0)
    print(f"   ✅ Merged to {len(df)} players with salaries and projections")
    
    # Filter to valid positions - be more lenient
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'D', 'DEF']
    df = df[df['Position'].isin(valid_positions)]
    
    # Standardize DST position names
    df.loc[df['Position'].isin(['D', 'DEF']), 'Position'] = 'DST'
    
    # Filter by salary
    before_salary_filter = len(df)
    df = df[df['Salary'] > 0]
    print(f"   ✅ Filtered to {len(df)} valid NFL players (removed {before_salary_filter - len(df)} with $0 salary)")
    
    return df


def add_enhanced_metrics(df):
    """Add all enhanced metrics for optimizer"""
    
    # Rename for optimizer
    df = df.rename(columns={'FantasyPoints': 'Predicted_DK_Points'})
    
    # Calculate value
    df['Value'] = df['Predicted_DK_Points'] / (df['Salary'] / 1000)
    df['PointsPerK'] = df['Predicted_DK_Points'] / (df['Salary'] / 1000)
    
    # Add ceiling and floor (20% variance)
    df['ceiling'] = df['Predicted_DK_Points'] * 1.2
    df['floor'] = df['Predicted_DK_Points'] * 0.8
    
    # Add mock ownership (inverse of salary + randomness)
    max_salary = df['Salary'].max()
    df['ownership'] = 100 * (1 - (df['Salary'] / max_salary)) * 0.3 + np.random.uniform(5, 15, len(df))
    
    # Add game environment (mock)
    df['game_total'] = 45.0
    df['spread'] = 0.0
    df['implied_points'] = 22.5
    df['is_dome'] = False
    df['wind'] = 5
    df['precip'] = 0
    df['temperature'] = 65
    
    # Add opponent data (mock)
    df['opponent'] = ''
    df['opp_def_rank'] = 16
    df['opp_def_rank_pos'] = 16
    
    # Add trends (mock)
    df['recent_form'] = 0.0
    df['trend'] = 0.0
    
    # Add value metrics
    df['value_projection'] = df['Value']
    df['value_ceiling'] = df['ceiling'] / (df['Salary'] / 1000)
    df['value_floor'] = df['floor'] / (df['Salary'] / 1000)
    df['value_rating'] = df['Value']
    
    # Add ownership tier
    try:
        df['ownership_tier'] = pd.qcut(df['ownership'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    except:
        df['ownership_tier'] = 'Medium'
    
    # Fill any remaining NaN teams with 'UNK' (Unknown)
    df['Team'] = df['Team'].fillna('UNK')
    
    return df


def add_dst_projections(df, contest_type='gpp'):
    """Add realistic DST projections"""
    
    # DST team abbreviation mapping (name to abbreviation)
    dst_team_map = {
        'Bills': 'BUF', 'Dolphins': 'MIA', 'Patriots': 'NE', 'Jets': 'NYJ',
        'Ravens': 'BAL', 'Bengals': 'CIN', 'Browns': 'CLE', 'Steelers': 'PIT',
        'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Titans': 'TEN',
        'Broncos': 'DEN', 'Chiefs': 'KC', 'Raiders': 'LV', 'Chargers': 'LAC',
        'Cowboys': 'DAL', 'Giants': 'NYG', 'Eagles': 'PHI', 'Commanders': 'WAS',
        'Bears': 'CHI', 'Lions': 'DET', 'Packers': 'GB', 'Vikings': 'MIN',
        'Falcons': 'ATL', 'Panthers': 'CAR', 'Saints': 'NO', 'Buccaneers': 'TB',
        'Cardinals': 'ARI', '49ers': 'SF', 'Rams': 'LA', 'Seahawks': 'SEA'
    }
    
    dst_mask = df['Position'] == 'DST'
    
    for idx in df[dst_mask].index:
        salary = df.loc[idx, 'Salary']
        dst_name = str(df.loc[idx, 'Name'])
        
        # Set Team from DST name
        team_abbr = dst_team_map.get(dst_name, dst_name[:3].upper())
        df.loc[idx, 'Team'] = team_abbr
        
        # Project points based on salary (6-10 point range)
        if salary >= 3000:
            projection = 9.0
        elif salary >= 2800:
            projection = 7.5
        elif salary >= 2500:
            projection = 6.5
        else:
            projection = 6.0
        
        # Contest-specific variance
        if contest_type == 'gpp':
            ceiling_boost = 1.6  # 60% upside for GPP
            floor_factor = 0.4
        else:  # cash
            ceiling_boost = 1.4  # 40% upside
            floor_factor = 0.5
        
        df.loc[idx, 'Predicted_DK_Points'] = projection
        df.loc[idx, 'ceiling'] = projection * ceiling_boost
        df.loc[idx, 'floor'] = projection * floor_factor
        df.loc[idx, 'Value'] = projection / (salary / 1000)
        df.loc[idx, 'PointsPerK'] = projection / (salary / 1000)
        df.loc[idx, 'value_projection'] = projection / (salary / 1000)
        df.loc[idx, 'value_ceiling'] = (projection * ceiling_boost) / (salary / 1000)
        df.loc[idx, 'value_floor'] = (projection * floor_factor) / (salary / 1000)
    
    return df


def create_gpp_version(df):
    """Create GPP-optimized version"""
    
    print("\n6️⃣ Creating GPP-Optimized Version...")
    df_gpp = df.copy()
    
    # GPP: Boost ceiling and high-upside plays
    df_gpp['Predicted_DK_Points'] = df_gpp['Predicted_DK_Points'] * 0.7 + df_gpp['ceiling'] * 0.3
    df_gpp['Value'] = df_gpp['Predicted_DK_Points'] / (df_gpp['Salary'] / 1000)
    
    # Boost low-ownership plays for GPP
    low_own_mask = df_gpp['ownership'] < df_gpp['ownership'].quantile(0.4)
    df_gpp.loc[low_own_mask, 'Predicted_DK_Points'] *= 1.1
    
    # Add DST projections
    df_gpp = add_dst_projections(df_gpp, contest_type='gpp')
    
    return df_gpp


def create_cash_version(df):
    """Create Cash game-optimized version"""
    
    print("\n7️⃣ Creating Cash Game-Optimized Version...")
    df_cash = df.copy()
    
    # Cash: Emphasize floor and consistency
    df_cash['Predicted_DK_Points'] = df_cash['Predicted_DK_Points'] * 0.6 + df_cash['floor'] * 0.4
    df_cash['Value'] = df_cash['Predicted_DK_Points'] / (df_cash['Salary'] / 1000)
    
    # Boost high-ownership safe plays for Cash
    high_own_mask = df_cash['ownership'] > df_cash['ownership'].quantile(0.6)
    df_cash.loc[high_own_mask, 'Predicted_DK_Points'] *= 1.05
    
    # Add DST projections
    df_cash = add_dst_projections(df_cash, contest_type='cash')
    
    return df_cash


def print_summary(df_gpp, df_cash, week):
    """Print summary statistics"""
    
    print("\n8️⃣ Summary Statistics:")
    print("=" * 70)
    
    # GPP Top 5
    print("\n📊 GPP Version - Top 5 by Ceiling:")
    top_gpp = df_gpp.nlargest(5, 'ceiling')[['Name', 'Position', 'Salary', 'Predicted_DK_Points', 'ceiling', 'ownership']]
    for _, p in top_gpp.iterrows():
        print(f"   {p['Name']:25s} {p['Position']:3s} ${p['Salary']:5.0f} - {p['Predicted_DK_Points']:5.2f} pts (ceiling: {p['ceiling']:5.2f}, own: {p['ownership']:.1f}%)")
    
    # Cash Top 5
    print("\n📊 Cash Version - Top 5 by Floor/Consistency:")
    top_cash = df_cash.nlargest(5, 'floor')[['Name', 'Position', 'Salary', 'Predicted_DK_Points', 'floor', 'ownership']]
    for _, p in top_cash.iterrows():
        print(f"   {p['Name']:25s} {p['Position']:3s} ${p['Salary']:5.0f} - {p['Predicted_DK_Points']:5.2f} pts (floor: {p['floor']:5.2f}, own: {p['ownership']:.1f}%)")
    
    # DST Summary
    print("\n🛡️  Defense/Special Teams:")
    dst_gpp = df_gpp[df_gpp['Position'] == 'DST'][['Name', 'Salary', 'Predicted_DK_Points', 'ceiling', 'floor']].sort_values('Predicted_DK_Points', ascending=False)
    if len(dst_gpp) > 0:
        print("   GPP:")
        for _, d in dst_gpp.iterrows():
            print(f"      {d['Name']:15s} ${d['Salary']:5.0f} - {d['Predicted_DK_Points']:5.2f} pts (ceiling: {d['ceiling']:5.2f})")
    else:
        print("   ⚠️  No DST available in slate")


def main():
    parser = argparse.ArgumentParser(description='Create NFL DFS Week Data (GPP & Cash)')
    parser.add_argument('--week', type=int, required=True, help='NFL week number (1-18)')
    parser.add_argument('--season', type=str, default='2025REG', help='Season (default: 2025REG)')
    parser.add_argument('--date', type=str, help='DFS slate date (YYYY-MM-DD). Auto-calculated if not provided.')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory (default: current)')
    
    args = parser.parse_args()
    
    # Calculate slate date if not provided (approx Sunday of that week)
    if not args.date:
        # 2025 NFL Week 1 started Sep 4 (Thursday)
        # Week 6 would be ~5 weeks later = Oct 9 (Thursday) or Oct 13 (Sunday)
        base_date = datetime(2025, 9, 7)  # Week 1 Sunday
        weeks_offset = args.week - 1
        slate_date_obj = base_date + timedelta(weeks=weeks_offset)
        args.date = slate_date_obj.strftime('%Y-%m-%d')
        print(f"📅 Auto-calculated slate date: {args.date}")
    
    # Initialize API
    api = SportsDataNFLAPI(API_KEY)
    
    # Fetch data
    projections, actuals, slate, injuries = fetch_week_data(api, args.season, args.week, args.date)
    
    if not projections or not slate:
        print("\n❌ Failed to fetch required data. Exiting.")
        sys.exit(1)
    
    # Create DataFrames
    df_proj = pd.DataFrame(projections)
    df_salary = create_salary_dataframe(slate)
    
    # Merge data
    df = merge_data(df_salary, df_proj)
    
    if len(df) == 0:
        print("\n❌ No valid players after merge. Exiting.")
        sys.exit(1)
    
    # Filter out injured players
    df = filter_injured_players(df, injuries)
    
    if len(df) == 0:
        print("\n❌ No healthy players remaining after injury filter. Exiting.")
        sys.exit(1)
    
    # Add enhanced metrics
    df = add_enhanced_metrics(df)
    
    # Create contest-specific versions
    df_gpp = create_gpp_version(df)
    df_cash = create_cash_version(df)
    
    # Save files
    gpp_file = os.path.join(args.output_dir, f'nfl_week{args.week}_gpp_enhanced.csv')
    cash_file = os.path.join(args.output_dir, f'nfl_week{args.week}_cash_enhanced.csv')
    
    df_gpp.to_csv(gpp_file, index=False)
    df_cash.to_csv(cash_file, index=False)
    
    print(f"\n   ✅ Saved GPP version: {gpp_file} ({len(df_gpp)} players)")
    print(f"   ✅ Saved Cash version: {cash_file} ({len(df_cash)} players)")
    
    # Print summary
    print_summary(df_gpp, df_cash, args.week)
    
    # Final message
    print("\n" + "=" * 70)
    print("✅ Week Data Created Successfully!")
    print("\n📁 Files saved to:")
    print(f"   • {os.path.abspath(gpp_file)}")
    print(f"   • {os.path.abspath(cash_file)}")
    print("\n🎯 Ready to load in optimizer and generate lineups!")
    print("=" * 70)


if __name__ == '__main__':
    main()

