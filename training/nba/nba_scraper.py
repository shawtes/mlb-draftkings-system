"""
nba_scraper.py — Download NBA player game logs from nba_api.

Fetches per-player per-game box scores from 1996-97 to present using the
PlayerGameLogs endpoint (returns ALL players for an entire season in one call).

Usage:
    python training/nba/nba_scraper.py --output ./data/nba_game_logs.csv
    python training/nba/nba_scraper.py --start-year 2004 --end-year 2025
    python training/nba/nba_scraper.py --playoffs  # include playoff games
"""

import os
import sys
import time
import argparse
import pandas as pd

try:
    from nba_api.stats.endpoints import playergamelogs
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("ERROR: nba_api not installed. Run: pip install nba_api")


# DraftKings NBA scoring formula
def compute_dk_fpts(df):
    """Compute DraftKings fantasy points from raw box score stats."""
    return (
        df['PTS'] * 1.0
        + df['FG3M'] * 0.5
        + df['REB'] * 1.25
        + df['AST'] * 1.5
        + df['STL'] * 2.0
        + df['BLK'] * 2.0
        + df['TOV'] * -0.5
        + df['DD2'] * 1.5
        + df['TD3'] * 3.0
    )


def fetch_season(season_str, season_type='Regular Season', retries=3):
    """Fetch all player game logs for one season + season type."""
    for attempt in range(retries):
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season_str,
                season_type_nullable=season_type,
            )
            df = logs.get_data_frames()[0]
            return df
        except Exception as e:
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"    Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    FAILED after {retries} retries: {e}")
                return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='NBA Game Log Scraper')
    parser.add_argument('--output', type=str, default='./data/nba_game_logs.csv',
                        help='Output CSV path')
    parser.add_argument('--start-year', type=int, default=2004,
                        help='Start season year (e.g., 2004 = 2004-05 season)')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End season year (e.g., 2025 = 2025-26 season)')
    parser.add_argument('--playoffs', action='store_true',
                        help='Include playoff games')
    parser.add_argument('--all-time', action='store_true',
                        help='Fetch from 1996-97 (earliest available)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds')
    args = parser.parse_args()

    if not NBA_API_AVAILABLE:
        sys.exit(1)

    start_year = 1996 if args.all_time else args.start_year
    end_year = args.end_year

    season_types = ['Regular Season']
    if args.playoffs:
        season_types.append('Playoffs')

    print(f"\n{'='*60}")
    print(f"NBA GAME LOG SCRAPER")
    print(f"{'='*60}")
    print(f"  Seasons: {start_year}-{str(start_year+1)[-2:]} to {end_year}-{str(end_year+1)[-2:]}")
    print(f"  Season types: {season_types}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    all_seasons = []
    total_rows = 0
    t0 = time.time()

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"

        for st in season_types:
            label = f"{season_str} {st}"
            print(f"  Fetching {label}...", end=' ', flush=True)

            df = fetch_season(season_str, st)

            if len(df) > 0:
                df['SEASON_TYPE'] = st
                all_seasons.append(df)
                total_rows += len(df)
                print(f"{len(df)} rows")
            else:
                print("0 rows (no data or future season)")

            time.sleep(args.delay)

    if not all_seasons:
        print("ERROR: No data fetched.")
        sys.exit(1)

    # Combine all seasons
    print(f"\nCombining {len(all_seasons)} season chunks...")
    df = pd.concat(all_seasons, ignore_index=True)

    # Parse date
    df['date'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')

    # Standardize column names for pipeline compatibility
    df['Name'] = df['PLAYER_NAME']
    df['Team'] = df['TEAM_ABBREVIATION']

    # Compute DraftKings fantasy points
    df['calculated_dk_fpts'] = compute_dk_fpts(df)

    # Drop rank columns to save space (they're ~30 columns of ranking data)
    rank_cols = [c for c in df.columns if c.endswith('_RANK')]
    df.drop(columns=rank_cols, inplace=True, errors='ignore')

    # Sort by player and date
    df.sort_values(['Name', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save
    df.to_csv(args.output, index=False)

    # Write to database if available
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, os.path.join(repo_root, 'data'))
        from db import get_connection, init_db, import_nba_csv
        db_path = os.path.join(repo_root, 'data', 'dfs.db')
        conn = get_connection(db_path)
        init_db(conn)
        import_nba_csv(conn, args.output)
        conn.close()
    except Exception as e_db:
        print(f"  DB write skipped: {e_db}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SCRAPE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total rows:    {len(df):,}")
    print(f"  Players:       {df['Name'].nunique():,}")
    print(f"  Seasons:       {df['SEASON_YEAR'].nunique()}")
    print(f"  Date range:    {df['date'].min()} to {df['date'].max()}")
    print(f"  Avg DK points: {df['calculated_dk_fpts'].mean():.1f}")
    print(f"  Columns:       {len(df.columns)}")
    print(f"  File size:     {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")
    print(f"  Time:          {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:        {args.output}")
    print(f"{'='*60}")

    # Quick data quality check
    print(f"\nData quality check:")
    print(f"  DK pts mean:   {df['calculated_dk_fpts'].mean():.2f}")
    print(f"  DK pts median: {df['calculated_dk_fpts'].median():.2f}")
    print(f"  DK pts std:    {df['calculated_dk_fpts'].std():.2f}")
    print(f"  DK pts max:    {df['calculated_dk_fpts'].max():.2f}")
    print(f"  NaN in PTS:    {df['PTS'].isna().sum()}")
    print(f"  NaN in MIN:    {df['MIN'].isna().sum()}")
    print(f"  Zero-minute games: {(df['MIN'] == 0).sum()}")


if __name__ == '__main__':
    main()
