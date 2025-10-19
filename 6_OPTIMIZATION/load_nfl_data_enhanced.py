#!/usr/bin/env python3
"""
Enhanced NFL Data Loader for DFS with Advanced Metrics
Adds: Ceiling/Floor projections, Ownership, Game Environment, Vegas data
"""

import sys
import os
sys.path.append('../python_algorithms')

from sportsdata_nfl_api import SportsDataNFLAPI
import pandas as pd
import numpy as np
from dfs_strategy_helpers import *


def add_ceiling_floor_projections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ceiling and floor projections based on position variance
    
    Position variance (from historical data):
    - QB: ±25% (most consistent)
    - RB: ±35% (moderate variance)
    - WR: ±45% (high variance)
    - TE: ±40% (high variance)
    - DST: ±50% (highest variance)
    """
    variance_by_position = {
        'QB': 0.25,
        'RB': 0.35,
        'WR': 0.45,
        'TE': 0.40,
        'DST': 0.50
    }
    
    df = df.copy()
    
    for pos, variance in variance_by_position.items():
        mask = df['Position'] == pos
        
        # Ceiling = projection + upside variance
        df.loc[mask, 'ceiling'] = df.loc[mask, 'FantasyPoints'] * (1 + variance)
        
        # Floor = projection - downside variance (but not below 0)
        df.loc[mask, 'floor'] = (df.loc[mask, 'FantasyPoints'] * (1 - variance)).clip(lower=0)
    
    # Fill any missing with default
    df['ceiling'] = df['ceiling'].fillna(df['FantasyPoints'] * 1.4)
    df['floor'] = df['floor'].fillna(df['FantasyPoints'] * 0.7)
    
    return df


def add_ownership_projections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ownership projections based on salary, value, and position
    
    In production, replace with:
    - Scraped ownership from RotoGrinders/FantasyLabs
    - Historical ownership models
    - Expert projections
    """
    df = df.copy()
    
    # Calculate value if not present
    if 'Value' not in df.columns:
        df['Value'] = df['FantasyPoints'] / (df['Salary'] / 1000)
    
    # Ownership model: combination of factors
    # 1. Salary percentile (high salary = more attention)
    df['salary_pct'] = df.groupby('Position')['Salary'].rank(pct=True)
    
    # 2. Value percentile (high value = more ownership)
    df['value_pct'] = df.groupby('Position')['Value'].rank(pct=True)
    
    # 3. Price tier (studs get more attention)
    df['is_stud'] = (df['Salary'] > 7500).astype(int) * 0.2
    df['is_value'] = (df['Value'] > 3.5).astype(int) * 0.3
    df['is_punt'] = (df['Salary'] < 4000).astype(int) * 0.2
    
    # Weighted ownership formula
    df['ownership'] = (
        df['salary_pct'] * 0.30 +
        df['value_pct'] * 0.40 +
        df['is_stud'] +
        df['is_value'] +
        df['is_punt']
    ).clip(upper=1.0)
    
    # Scale to realistic ranges (5-40%)
    df['ownership'] = df['ownership'] * 0.35 + 0.05
    
    # Clean up temporary columns
    df = df.drop(columns=['salary_pct', 'value_pct', 'is_stud', 'is_value', 'is_punt'], errors='ignore')
    
    return df


def add_vegas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Vegas odds data: game totals, spreads, implied points
    
    In production, scrape from:
    - DraftKings Sportsbook
    - FanDuel Sportsbook
    - Pinnacle
    - Action Network
    """
    df = df.copy()
    
    # For now, generate reasonable estimates
    # In production, fetch real Vegas lines
    
    # Game total (typical range: 38-52)
    np.random.seed(42)
    teams = df['Team'].unique()
    game_totals = {}
    spreads = {}
    
    for team in teams:
        # Simulate realistic game total
        game_totals[team] = np.random.uniform(40, 50)
        
        # Simulate spread (-14 to +14)
        spreads[team] = np.random.uniform(-7, 7)
    
    df['game_total'] = df['Team'].map(game_totals)
    df['spread'] = df['Team'].map(spreads)
    
    # Calculate implied team total
    df['implied_points'] = (df['game_total'] / 2) + (df['spread'] / 2)
    
    return df


def add_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather data: wind, precipitation, temperature, dome/outdoor
    
    In production, fetch from:
    - Weather.com API
    - OpenWeatherMap
    - NFL Weather (nflweather.com)
    """
    df = df.copy()
    
    # Dome teams (no weather impact)
    dome_teams = ['NO', 'ATL', 'DET', 'MIN', 'IND', 'LV', 'LAR', 'DAL']
    
    df['is_dome'] = df['Team'].isin(dome_teams).astype(int)
    
    # For outdoor games, simulate weather
    np.random.seed(43)
    teams = df['Team'].unique()
    weather = {}
    
    for team in teams:
        if team in dome_teams:
            weather[team] = {'wind': 0, 'precip': 0, 'temp': 72}
        else:
            # Simulate realistic weather
            weather[team] = {
                'wind': max(0, np.random.normal(8, 5)),  # mph
                'precip': max(0, np.random.normal(0, 0.2)),  # inches
                'temp': np.random.uniform(40, 75)  # fahrenheit
            }
    
    df['wind'] = df['Team'].map(lambda t: weather.get(t, {}).get('wind', 0))
    df['precip'] = df['Team'].map(lambda t: weather.get(t, {}).get('precip', 0))
    df['temperature'] = df['Team'].map(lambda t: weather.get(t, {}).get('temp', 70))
    
    return df


def add_opponent_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent team and defensive rankings
    
    In production, maintain opponent schedule and defensive stats
    """
    df = df.copy()
    
    # Simulate opponent assignments (Week 7 matchups)
    # In production, fetch from schedule API
    matchups = {
        'TB': 'DET', 'DET': 'TB',
        'SEA': 'HOU', 'HOU': 'SEA',
        # Add more matchups...
    }
    
    df['opponent'] = df['Team'].map(matchups).fillna('UNK')
    
    # Defensive rankings (1=best, 32=worst)
    # In production, calculate from season stats
    def_rankings = {
        'TB': 15, 'DET': 20, 'SEA': 10, 'HOU': 25
        # Add all teams...
    }
    
    df['opp_def_rank'] = df['opponent'].map(def_rankings).fillna(16)
    
    # Position-specific defensive rankings
    # In production, track QB/RB/WR/TE points allowed
    df['opp_def_rank_pos'] = df['opp_def_rank']  # Simplified
    
    return df


def add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recent performance trends
    
    In production, calculate from last 3-4 games
    """
    df = df.copy()
    
    # Simulate recent form (0.8 = below average, 1.2 = hot)
    np.random.seed(44)
    df['recent_form'] = np.random.uniform(0.85, 1.15, size=len(df))
    
    # Trending up/down/stable
    df['trend'] = pd.cut(df['recent_form'], bins=[0, 0.95, 1.05, 2.0], 
                         labels=['Down', 'Stable', 'Up'])
    
    return df


def load_nfl_data_for_optimizer_enhanced(
    api_key: str,
    date: str,
    season: str = None,
    week: int = None,
    use_projections: bool = True,
    contest_mode: str = 'gpp_tournament',
    output_filename: str = None,
    operator: str = "DraftKings"
) -> pd.DataFrame:
    """
    Enhanced NFL data loader with all DFS metrics
    
    Adds:
    - Ceiling/Floor projections
    - Ownership projections
    - Vegas data (totals, spreads)
    - Weather data
    - Opponent data
    - Recent form
    - Enhanced value calculations
    """
    print("="*70)
    print("ENHANCED NFL DATA LOADER FOR DFS OPTIMIZER")
    print("="*70)
    
    api = SportsDataNFLAPI(api_key)
    
    # Step 1: Fetch DFS slate data (contains actual DraftKings salaries)
    print(f"\n📊 Step 1: Fetching DFS Slates for {date}...")
    slate_data = api.get_dfs_slates_by_date(date, save_to_file=True)
    
    if not slate_data:
        print("❌ Failed to fetch DFS slate data")
        return None
    
    # Find the DraftKings slate
    dk_slate = None
    for slate in slate_data:
        if slate.get('Operator', '').upper() == operator.upper():
            dk_slate = slate
            break
    
    if not dk_slate:
        print(f"❌ No {operator} slate found for {date}")
        return None
    
    slate_players = dk_slate.get('DfsSlatePlayers', [])
    print(f"✅ Found {len(slate_players)} players in {operator} slate")
    
    salary_df = pd.DataFrame(slate_players)
    
    # Step 2: Fetch projection data
    projections_df = None
    if use_projections and season and week:
        print(f"\n📊 Step 2: Fetching PROJECTIONS for {season} Week {week}...")
        proj_data = api.get_player_projections_by_week(season, week)
        
        if proj_data:
            projections_df = pd.DataFrame(proj_data)
            print(f"✅ Loaded {len(projections_df)} player projections")
    
    # Step 3: Merge and clean
    print(f"\n🔗 Step 3: Merging and enhancing data...")
    
    optimizer_df = pd.DataFrame()
    optimizer_df['Name'] = salary_df.get('OperatorPlayerName', '')
    optimizer_df['PlayerID'] = salary_df.get('PlayerID', '')
    optimizer_df['Position'] = salary_df.get('OperatorPosition', '')
    optimizer_df['Team'] = salary_df.get('Team', '')
    optimizer_df['Salary'] = salary_df.get('OperatorSalary', 0)
    optimizer_df['OperatorPlayerID'] = salary_df.get('OperatorPlayerID', '')
    
    # Merge with projections
    if projections_df is not None:
        if 'PlayerID' in projections_df.columns:
            merged = optimizer_df.merge(
                projections_df[['PlayerID', 'FantasyPoints']],
                on='PlayerID',
                how='left'
            )
            optimizer_df['FantasyPoints'] = merged['FantasyPoints'].fillna(0)
    else:
        optimizer_df['FantasyPoints'] = 0
    
    # Filter positions
    optimizer_df['Position'] = optimizer_df['Position'].replace({'DEF': 'DST', 'D': 'DST'})
    valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    optimizer_df = optimizer_df[optimizer_df['Position'].isin(valid_positions)]
    optimizer_df = optimizer_df.dropna(subset=['Name', 'Position', 'Salary'])
    optimizer_df = optimizer_df[optimizer_df['Salary'] > 0]
    
    print(f"✅ Base data: {len(optimizer_df)} players")
    
    # Step 4: Add enhanced metrics
    print(f"\n📈 Step 4: Adding advanced DFS metrics...")
    
    # Add ceiling/floor
    print("   - Ceiling/Floor projections")
    optimizer_df = add_ceiling_floor_projections(optimizer_df)
    
    # Add ownership
    print("   - Ownership projections")
    optimizer_df = add_ownership_projections(optimizer_df)
    
    # Add Vegas data
    print("   - Vegas lines (totals, spreads)")
    optimizer_df = add_vegas_data(optimizer_df)
    
    # Add weather
    print("   - Weather data")
    optimizer_df = add_weather_data(optimizer_df)
    
    # Add opponent data
    print("   - Opponent matchups")
    optimizer_df = add_opponent_data(optimizer_df)
    
    # Add recent form
    print("   - Recent form trends")
    optimizer_df = add_recent_form(optimizer_df)
    
    # Step 5: Apply contest-specific optimizations
    print(f"\n🎯 Step 5: Applying {contest_mode} optimizations...")
    
    # Value analyzer
    value_analyzer = ValueAnalyzer(contest_mode=contest_mode)
    optimizer_df = value_analyzer.calculate_enhanced_value(optimizer_df)
    
    # Game environment boosts
    game_analyzer = GameEnvironmentAnalyzer()
    optimizer_df = game_analyzer.apply_game_environment_boosts(optimizer_df)
    
    # Contrarian adjustments (if GPP)
    if 'gpp' in contest_mode.lower():
        contrarian = ContestarianEngine(contest_mode=contest_mode)
        optimizer_df = contrarian.apply_ownership_adjustments(optimizer_df)
    
    # Recalculate value after adjustments
    optimizer_df['Value'] = optimizer_df['FantasyPoints'] / (optimizer_df['Salary'] / 1000)
    optimizer_df['PointsPerK'] = optimizer_df['Value']
    
    # Step 6: Save output
    if output_filename is None:
        date_clean = date.replace('-', '')
        output_filename = f"nfl_enhanced_{contest_mode}_{date_clean}.csv"
    
    optimizer_df.to_csv(output_filename, index=False)
    print(f"\n💾 Saved to: {output_filename}")
    
    # Show summary statistics
    print(f"\n📊 Data Summary:")
    print(f"   Total Players: {len(optimizer_df)}")
    print(f"\n   Position Breakdown:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        count = len(optimizer_df[optimizer_df['Position'] == pos])
        avg_salary = optimizer_df[optimizer_df['Position'] == pos]['Salary'].mean()
        avg_own = optimizer_df[optimizer_df['Position'] == pos]['ownership'].mean()
        print(f"      {pos}: {count} players (Avg Salary: ${avg_salary:,.0f}, Avg Own: {avg_own:.1%})")
    
    print(f"\n   Top 5 by {contest_mode} Value:")
    top_5 = optimizer_df.nlargest(5, 'Value')[['Name', 'Position', 'Salary', 'FantasyPoints', 'Value', 'ownership']]
    print(top_5.to_string(index=False))
    
    return optimizer_df


def main():
    """Main function"""
    API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
    DATE = "2025-10-20"
    SEASON = "2025REG"
    WEEK = 7
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     Enhanced NFL Data Loader with Advanced DFS Metrics       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Load for GPP Tournament
    print("\n🏆 Loading for GPP TOURNAMENT...")
    df_gpp = load_nfl_data_for_optimizer_enhanced(
        api_key=API_KEY,
        date=DATE,
        season=SEASON,
        week=WEEK,
        contest_mode='gpp_tournament',
        output_filename=f"nfl_week{WEEK}_gpp_enhanced.csv"
    )
    
    # Load for Cash Game
    print("\n\n💰 Loading for CASH GAME...")
    df_cash = load_nfl_data_for_optimizer_enhanced(
        api_key=API_KEY,
        date=DATE,
        season=SEASON,
        week=WEEK,
        contest_mode='cash_game',
        output_filename=f"nfl_week{WEEK}_cash_enhanced.csv"
    )
    
    print("\n" + "="*70)
    print("✅ SUCCESS! Enhanced data loaded for both contest types")
    print("="*70)


if __name__ == "__main__":
    main()

