#!/usr/bin/env python3
"""
db.py — Unified SQLite database layer for the DFS optimization system.

Provides schemas, CRUD operations, and CSV import/export for all three sports
(MLB, NBA, NFL). The DB file lives at data/dfs.db (gitignored).

Usage:
    python data/db.py init                              # Create tables
    python data/db.py import-mlb <csv_path>             # Import MLB batter logs
    python data/db.py import-nba <csv_path>             # Import NBA game logs
    python data/db.py import-mlb-pitchers <csv_path>    # Import MLB pitcher logs
    python data/db.py stats                             # Show row counts
"""

import os
import sys
import sqlite3
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_THIS_DIR, 'dfs.db')


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_MLB_GAME_LOGS = """
CREATE TABLE IF NOT EXISTS mlb_game_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team TEXT,
    date TEXT NOT NULL,
    pa INTEGER, ab INTEGER, h INTEGER,
    "1b" INTEGER, "2b" INTEGER, "3b" INTEGER, hr INTEGER,
    r INTEGER, rbi INTEGER, bb INTEGER, so INTEGER,
    hbp INTEGER, sb INTEGER, cs INTEGER, gdp INTEGER,
    avg REAL, obp REAL, slg REAL, ops REAL,
    iso REAL, babip REAL, woba REAL, wrc_plus REAL,
    wrc REAL, wraa REAL, off REAL, def_ REAL, war REAL,
    gb_pct REAL, fb_pct REAL, ld_pct REAL,
    hr_fb REAL, ifh_pct REAL,
    o_swing_pct REAL, z_swing_pct REAL,
    o_contact_pct REAL, z_contact_pct REAL,
    swing_pct REAL, contact_pct REAL,
    re24 REAL, rar REAL, wpa REAL, wpa_li REAL,
    calculated_dk_fpts REAL,
    download_date TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name, date)
);
CREATE INDEX IF NOT EXISTS idx_mlb_name_date ON mlb_game_logs(name, date);
CREATE INDEX IF NOT EXISTS idx_mlb_date ON mlb_game_logs(date);
CREATE INDEX IF NOT EXISTS idx_mlb_team ON mlb_game_logs(team);
"""

SCHEMA_MLB_PITCHER_LOGS = """
CREATE TABLE IF NOT EXISTS mlb_pitcher_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team TEXT,
    date TEXT NOT NULL,
    ip REAL, w INTEGER, l INTEGER, sv INTEGER,
    h_allowed INTEGER, r_allowed INTEGER, er INTEGER,
    bb_allowed INTEGER, so_pitcher INTEGER, hr_allowed INTEGER,
    era REAL, whip REAL, fip REAL, xfip REAL, siera REAL,
    k_9 REAL, bb_9 REAL, k_bb REAL,
    calculated_dk_fpts REAL,
    download_date TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name, date)
);
CREATE INDEX IF NOT EXISTS idx_mlb_pitcher_name_date ON mlb_pitcher_logs(name, date);
"""

SCHEMA_NBA_GAME_LOGS = """
CREATE TABLE IF NOT EXISTS nba_game_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year TEXT,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    team_abbreviation TEXT,
    team_name TEXT,
    game_id TEXT,
    game_date TEXT,
    matchup TEXT,
    wl TEXT,
    min REAL,
    fgm INTEGER, fga INTEGER, fg_pct REAL,
    fg3m INTEGER, fg3a INTEGER, fg3_pct REAL,
    ftm INTEGER, fta INTEGER, ft_pct REAL,
    oreb INTEGER, dreb INTEGER, reb INTEGER,
    ast INTEGER, tov INTEGER, stl INTEGER, blk INTEGER,
    blka INTEGER, pf INTEGER, pfd INTEGER,
    pts INTEGER, plus_minus INTEGER,
    nba_fantasy_pts REAL,
    dd2 INTEGER, td3 INTEGER,
    calculated_dk_fpts REAL,
    season_type TEXT,
    date TEXT NOT NULL,
    name TEXT NOT NULL,
    team TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(player_id, game_id)
);
CREATE INDEX IF NOT EXISTS idx_nba_name_date ON nba_game_logs(name, date);
CREATE INDEX IF NOT EXISTS idx_nba_player_id ON nba_game_logs(player_id);
CREATE INDEX IF NOT EXISTS idx_nba_date ON nba_game_logs(date);
CREATE INDEX IF NOT EXISTS idx_nba_team ON nba_game_logs(team);
"""

SCHEMA_NFL_GAME_LOGS = """
CREATE TABLE IF NOT EXISTS nfl_game_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    date TEXT NOT NULL,
    season INTEGER,
    week INTEGER,
    opponent TEXT,
    passing_yards REAL, passing_tds REAL,
    passing_attempts INTEGER, passing_completions INTEGER,
    interceptions REAL,
    rushing_yards REAL, rushing_tds REAL,
    rushing_attempts INTEGER,
    receiving_yards REAL, receiving_tds REAL,
    receptions REAL, targets INTEGER,
    fumbles_lost REAL,
    two_point_conversions INTEGER,
    salary INTEGER,
    calculated_dk_fpts REAL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(name, date)
);
CREATE INDEX IF NOT EXISTS idx_nfl_name_date ON nfl_game_logs(name, date);
CREATE INDEX IF NOT EXISTS idx_nfl_date ON nfl_game_logs(date);
"""

SCHEMA_DAILY_SLATES = """
CREATE TABLE IF NOT EXISTS daily_slates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    date TEXT NOT NULL,
    name TEXT NOT NULL,
    dk_id TEXT,
    position TEXT,
    team TEXT,
    opponent TEXT,
    salary INTEGER,
    projected_dk_pts REAL,
    ceiling REAL,
    floor_ REAL,
    ownership REAL,
    game_info TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(sport, date, name)
);
CREATE INDEX IF NOT EXISTS idx_slates_sport_date ON daily_slates(sport, date);
"""

SCHEMA_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    predicted_dk_fpts REAL,
    actual_dk_fpts REAL,
    q10 REAL, q25 REAL, q50 REAL, q75 REAL, q90 REAL,
    prob_over_5 REAL, prob_over_10 REAL, prob_over_15 REAL,
    prob_over_20 REAL, prob_over_25 REAL, prob_over_30 REAL,
    model_version TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(sport, name, date, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_sport_date ON predictions(sport, date);
"""

ALL_SCHEMAS = [
    SCHEMA_MLB_GAME_LOGS,
    SCHEMA_MLB_PITCHER_LOGS,
    SCHEMA_NBA_GAME_LOGS,
    SCHEMA_NFL_GAME_LOGS,
    SCHEMA_DAILY_SLATES,
    SCHEMA_PREDICTIONS,
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(db_path=None):
    """Return a sqlite3 connection. Default: data/dfs.db."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn):
    """Create all tables and indexes."""
    cur = conn.cursor()
    for schema in ALL_SCHEMAS:
        cur.executescript(schema)
    conn.commit()
    print("Database initialized with 6 tables.")


# ---------------------------------------------------------------------------
# Column name normalization helpers
# ---------------------------------------------------------------------------

# FanGraphs CSV column -> DB column mapping for MLB batters
_MLB_COL_MAP = {
    'Name': 'name', 'Team': 'team', 'date': 'date',
    'PA': 'pa', 'AB': 'ab', 'H': 'h',
    '1B': '1b', '2B': '2b', '3B': '3b', 'HR': 'hr',
    'R': 'r', 'RBI': 'rbi', 'BB': 'bb', 'SO': 'so',
    'HBP': 'hbp', 'SB': 'sb', 'CS': 'cs', 'GDP': 'gdp',
    'AVG': 'avg', 'OBP': 'obp', 'SLG': 'slg', 'OPS': 'ops',
    'ISO': 'iso', 'BABIP': 'babip', 'wOBA': 'woba', 'wRC+': 'wrc_plus',
    'wRC': 'wrc', 'wRAA': 'wraa', 'Off': 'off', 'Def': 'def_', 'WAR': 'war',
    'GB%': 'gb_pct', 'FB%': 'fb_pct', 'LD%': 'ld_pct',
    'HR/FB': 'hr_fb', 'IFH%': 'ifh_pct',
    'O-Swing%': 'o_swing_pct', 'Z-Swing%': 'z_swing_pct',
    'O-Contact%': 'o_contact_pct', 'Z-Contact%': 'z_contact_pct',
    'Swing%': 'swing_pct', 'Contact%': 'contact_pct',
    'RE24': 're24', 'RAR': 'rar', 'WPA': 'wpa', 'WPA/LI': 'wpa_li',
    'calculated_dk_fpts': 'calculated_dk_fpts',
    'download_date': 'download_date',
}

# nba_api / scraper column -> DB column mapping
_NBA_COL_MAP = {
    'SEASON_YEAR': 'season_year', 'PLAYER_ID': 'player_id',
    'PLAYER_NAME': 'player_name', 'TEAM_ABBREVIATION': 'team_abbreviation',
    'TEAM_NAME': 'team_name', 'GAME_ID': 'game_id',
    'GAME_DATE': 'game_date', 'MATCHUP': 'matchup', 'WL': 'wl',
    'MIN': 'min', 'FGM': 'fgm', 'FGA': 'fga', 'FG_PCT': 'fg_pct',
    'FG3M': 'fg3m', 'FG3A': 'fg3a', 'FG3_PCT': 'fg3_pct',
    'FTM': 'ftm', 'FTA': 'fta', 'FT_PCT': 'ft_pct',
    'OREB': 'oreb', 'DREB': 'dreb', 'REB': 'reb',
    'AST': 'ast', 'TOV': 'tov', 'STL': 'stl', 'BLK': 'blk',
    'BLKA': 'blka', 'PF': 'pf', 'PFD': 'pfd',
    'PTS': 'pts', 'PLUS_MINUS': 'plus_minus',
    'NBA_FANTASY_PTS': 'nba_fantasy_pts',
    'DD2': 'dd2', 'TD3': 'td3',
    'calculated_dk_fpts': 'calculated_dk_fpts',
    'SEASON_TYPE': 'season_type',
    'date': 'date', 'Name': 'name', 'Team': 'team',
}


def _safe_val(val):
    """Convert numpy/pandas types to Python native for sqlite3."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return str(val)
    return val


# ---------------------------------------------------------------------------
# MLB imports
# ---------------------------------------------------------------------------

def import_mlb_csv(conn, csv_path):
    """Bulk import MLB batter game logs from merged_fangraphs_data.csv."""
    print(f"Importing MLB batter data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    count = _upsert_mlb_batch(conn, df)
    print(f"  Imported {count} MLB batter rows.")
    return count


def _upsert_mlb_batch(conn, df):
    """Insert/replace MLB batter rows from a DataFrame."""
    # Map CSV columns to DB columns
    db_cols = []
    csv_cols = []
    for csv_col, db_col in _MLB_COL_MAP.items():
        if csv_col in df.columns:
            db_cols.append(db_col)
            csv_cols.append(csv_col)

    if 'name' not in db_cols or 'date' not in db_cols:
        print("  ERROR: CSV missing required Name or date column.")
        return 0

    placeholders = ', '.join(['?'] * len(db_cols))
    col_names = ', '.join(f'"{c}"' for c in db_cols)
    sql = f'INSERT OR REPLACE INTO mlb_game_logs ({col_names}) VALUES ({placeholders})'

    rows = []
    for _, row in df.iterrows():
        vals = tuple(_safe_val(row.get(c)) for c in csv_cols)
        rows.append(vals)

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_mlb_game_log(conn, row_dict):
    """Insert/update a single MLB batter game log row."""
    mapped = {}
    for csv_col, db_col in _MLB_COL_MAP.items():
        if csv_col in row_dict:
            mapped[db_col] = _safe_val(row_dict[csv_col])
    if 'name' not in mapped or 'date' not in mapped:
        return False
    cols = ', '.join(f'"{c}"' for c in mapped.keys())
    placeholders = ', '.join(['?'] * len(mapped))
    sql = f'INSERT OR REPLACE INTO mlb_game_logs ({cols}) VALUES ({placeholders})'
    conn.execute(sql, tuple(mapped.values()))
    conn.commit()
    return True


def import_mlb_pitchers_csv(conn, csv_path):
    """Bulk import MLB pitcher game logs."""
    print(f"Importing MLB pitcher data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)

    pitcher_col_map = {
        'Name': 'name', 'Team': 'team', 'date': 'date',
        'IP': 'ip', 'W': 'w', 'L': 'l', 'SV': 'sv',
        'H': 'h_allowed', 'R': 'r_allowed', 'ER': 'er',
        'BB': 'bb_allowed', 'SO': 'so_pitcher', 'HR': 'hr_allowed',
        'ERA': 'era', 'WHIP': 'whip', 'FIP': 'fip',
        'xFIP': 'xfip', 'SIERA': 'siera',
        'K/9': 'k_9', 'BB/9': 'bb_9', 'K/BB': 'k_bb',
        'calculated_dk_fpts': 'calculated_dk_fpts',
        'download_date': 'download_date',
    }

    db_cols = []
    csv_cols = []
    for csv_col, db_col in pitcher_col_map.items():
        if csv_col in df.columns:
            db_cols.append(db_col)
            csv_cols.append(csv_col)

    if 'name' not in db_cols or 'date' not in db_cols:
        print("  ERROR: CSV missing required Name or date column.")
        return 0

    placeholders = ', '.join(['?'] * len(db_cols))
    col_names = ', '.join(f'"{c}"' for c in db_cols)
    sql = f'INSERT OR REPLACE INTO mlb_pitcher_logs ({col_names}) VALUES ({placeholders})'

    rows = []
    for _, row in df.iterrows():
        vals = tuple(_safe_val(row.get(c)) for c in csv_cols)
        rows.append(vals)

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    print(f"  Imported {len(rows)} MLB pitcher rows.")
    return len(rows)


# ---------------------------------------------------------------------------
# NBA imports
# ---------------------------------------------------------------------------

def import_nba_csv(conn, csv_path):
    """Bulk import NBA game logs from nba_game_logs.csv."""
    print(f"Importing NBA data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    count = _upsert_nba_batch(conn, df)
    print(f"  Imported {count} NBA rows.")
    return count


def _upsert_nba_batch(conn, df):
    """Insert/replace NBA game log rows from a DataFrame."""
    db_cols = []
    csv_cols = []
    for csv_col, db_col in _NBA_COL_MAP.items():
        if csv_col in df.columns:
            db_cols.append(db_col)
            csv_cols.append(csv_col)

    if 'name' not in db_cols or 'date' not in db_cols:
        # Try lowercase fallback
        if 'player_name' in db_cols:
            pass  # at least we have player_name
        else:
            print("  ERROR: CSV missing required name/date columns.")
            return 0

    # Ensure unique constraint columns exist
    if 'player_id' not in db_cols or 'game_id' not in db_cols:
        # Fall back to name+date uniqueness via INSERT OR IGNORE
        pass

    placeholders = ', '.join(['?'] * len(db_cols))
    col_names = ', '.join(f'"{c}"' for c in db_cols)
    sql = f'INSERT OR REPLACE INTO nba_game_logs ({col_names}) VALUES ({placeholders})'

    rows = []
    for _, row in df.iterrows():
        vals = tuple(_safe_val(row.get(c)) for c in csv_cols)
        rows.append(vals)

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


def upsert_nba_game_log(conn, row_dict):
    """Insert/update a single NBA game log row."""
    mapped = {}
    for csv_col, db_col in _NBA_COL_MAP.items():
        if csv_col in row_dict:
            mapped[db_col] = _safe_val(row_dict[csv_col])
    cols = ', '.join(f'"{c}"' for c in mapped.keys())
    placeholders = ', '.join(['?'] * len(mapped))
    sql = f'INSERT OR REPLACE INTO nba_game_logs ({cols}) VALUES ({placeholders})'
    conn.execute(sql, tuple(mapped.values()))
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# NFL imports (prepared for future scraper)
# ---------------------------------------------------------------------------

def upsert_nfl_game_log(conn, row_dict):
    """Insert/update a single NFL game log row."""
    # Direct column mapping — NFL has no scraper yet, so keys match DB columns
    allowed = {
        'name', 'position', 'team', 'date', 'season', 'week', 'opponent',
        'passing_yards', 'passing_tds', 'passing_attempts', 'passing_completions',
        'interceptions', 'rushing_yards', 'rushing_tds', 'rushing_attempts',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'fumbles_lost', 'two_point_conversions', 'salary', 'calculated_dk_fpts',
    }
    mapped = {k: _safe_val(v) for k, v in row_dict.items() if k in allowed}
    if 'name' not in mapped or 'date' not in mapped:
        return False
    cols = ', '.join(f'"{c}"' for c in mapped.keys())
    placeholders = ', '.join(['?'] * len(mapped))
    sql = f'INSERT OR REPLACE INTO nfl_game_logs ({cols}) VALUES ({placeholders})'
    conn.execute(sql, tuple(mapped.values()))
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# Daily slates
# ---------------------------------------------------------------------------

def upsert_daily_slate(conn, sport, date, players_df):
    """Import a DK slate into the daily_slates table."""
    col_map = {
        'Name': 'name', 'name': 'name',
        'DK_ID': 'dk_id', 'dk_id': 'dk_id',
        'Position': 'position', 'position': 'position',
        'Team': 'team', 'team': 'team',
        'Opponent': 'opponent', 'opponent': 'opponent',
        'Salary': 'salary', 'salary': 'salary',
        'Proj_DK': 'projected_dk_pts', 'projected_dk_pts': 'projected_dk_pts',
        'Ceiling': 'ceiling', 'ceiling': 'ceiling',
        'Floor': 'floor_', 'floor_': 'floor_',
        'Ownership': 'ownership', 'ownership': 'ownership',
        'Game_Info': 'game_info', 'game_info': 'game_info',
    }
    count = 0
    for _, row in players_df.iterrows():
        mapped = {'sport': sport, 'date': date}
        for csv_col, db_col in col_map.items():
            if csv_col in row.index:
                mapped[db_col] = _safe_val(row[csv_col])
        if 'name' not in mapped:
            continue
        cols = ', '.join(f'"{c}"' for c in mapped.keys())
        placeholders = ', '.join(['?'] * len(mapped))
        sql = f'INSERT OR REPLACE INTO daily_slates ({cols}) VALUES ({placeholders})'
        conn.execute(sql, tuple(mapped.values()))
        count += 1
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def upsert_predictions(conn, sport, predictions_df, model_version=None):
    """Save model predictions to the predictions table."""
    if model_version is None:
        model_version = datetime.now().strftime('%Y%m%d')

    col_map = {
        'Name': 'name', 'name': 'name',
        'Date': 'date', 'date': 'date',
        'Predicted': 'predicted_dk_fpts', 'predicted_dk_fpts': 'predicted_dk_fpts',
        'Actual': 'actual_dk_fpts', 'actual_dk_fpts': 'actual_dk_fpts',
        'q10': 'q10', 'q25': 'q25', 'q50': 'q50', 'q75': 'q75', 'q90': 'q90',
        'prob_over_5': 'prob_over_5', 'prob_over_10': 'prob_over_10',
        'prob_over_15': 'prob_over_15', 'prob_over_20': 'prob_over_20',
        'prob_over_25': 'prob_over_25', 'prob_over_30': 'prob_over_30',
    }
    count = 0
    for _, row in predictions_df.iterrows():
        mapped = {'sport': sport, 'model_version': model_version}
        for csv_col, db_col in col_map.items():
            if csv_col in row.index:
                mapped[db_col] = _safe_val(row[csv_col])
        if 'name' not in mapped or 'date' not in mapped:
            continue
        cols = ', '.join(f'"{c}"' for c in mapped.keys())
        placeholders = ', '.join(['?'] * len(mapped))
        sql = f'INSERT OR REPLACE INTO predictions ({cols}) VALUES ({placeholders})'
        conn.execute(sql, tuple(mapped.values()))
        count += 1
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def query_mlb_training_data(conn, start_date=None, end_date=None):
    """Load MLB batter game logs as a DataFrame for the training pipeline."""
    sql = "SELECT * FROM mlb_game_logs"
    conditions = []
    params = []
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY name, date"
    df = pd.read_sql_query(sql, conn, params=params)
    # Restore original FanGraphs-style column names for pipeline compatibility
    reverse_map = {v: k for k, v in _MLB_COL_MAP.items()}
    df.rename(columns=reverse_map, inplace=True)
    return df


def query_nba_training_data(conn, start_date=None, end_date=None):
    """Load NBA game logs as a DataFrame for the training pipeline."""
    sql = "SELECT * FROM nba_game_logs"
    conditions = []
    params = []
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY name, date"
    df = pd.read_sql_query(sql, conn, params=params)
    # Restore original nba_api-style column names for pipeline compatibility
    reverse_map = {v: k for k, v in _NBA_COL_MAP.items()}
    df.rename(columns=reverse_map, inplace=True)
    return df


def query_nfl_training_data(conn, start_date=None, end_date=None):
    """Load NFL game logs as a DataFrame. Returns empty until populated."""
    sql = "SELECT * FROM nfl_game_logs"
    conditions = []
    params = []
    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY name, date"
    return pd.read_sql_query(sql, conn, params=params)


def query_daily_slate(conn, sport, date):
    """Load a daily slate as a DataFrame."""
    sql = "SELECT * FROM daily_slates WHERE sport = ? AND date = ?"
    return pd.read_sql_query(sql, conn, params=[sport, date])


def query_predictions(conn, sport, date):
    """Load predictions for a sport+date as a DataFrame."""
    sql = "SELECT * FROM predictions WHERE sport = ? AND date = ?"
    return pd.read_sql_query(sql, conn, params=[sport, date])


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_db_stats(conn):
    """Return row counts per table."""
    tables = [
        'mlb_game_logs', 'mlb_pitcher_logs', 'nba_game_logs',
        'nfl_game_logs', 'daily_slates', 'predictions',
    ]
    stats = {}
    cur = conn.cursor()
    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()[0]
        except sqlite3.OperationalError:
            stats[table] = -1  # table doesn't exist
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DFS Database Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init                          Create all tables
  import-mlb <csv>              Import MLB batter game logs
  import-mlb-pitchers <csv>     Import MLB pitcher game logs
  import-nba <csv>              Import NBA game logs
  stats                         Show row counts per table
        """,
    )
    parser.add_argument('command', choices=['init', 'import-mlb', 'import-mlb-pitchers',
                                            'import-nba', 'stats'])
    parser.add_argument('csv_path', nargs='?', default=None,
                        help='CSV file path (for import commands)')
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH,
                        help=f'Database path (default: {DEFAULT_DB_PATH})')
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    # Always ensure tables exist
    init_db(conn)

    if args.command == 'init':
        print(f"Database ready at {args.db_path}")

    elif args.command == 'import-mlb':
        if not args.csv_path:
            print("ERROR: CSV path required. Usage: python data/db.py import-mlb <csv_path>")
            sys.exit(1)
        if not os.path.exists(args.csv_path):
            print(f"ERROR: File not found: {args.csv_path}")
            sys.exit(1)
        import_mlb_csv(conn, args.csv_path)

    elif args.command == 'import-mlb-pitchers':
        if not args.csv_path:
            print("ERROR: CSV path required.")
            sys.exit(1)
        if not os.path.exists(args.csv_path):
            print(f"ERROR: File not found: {args.csv_path}")
            sys.exit(1)
        import_mlb_pitchers_csv(conn, args.csv_path)

    elif args.command == 'import-nba':
        if not args.csv_path:
            print("ERROR: CSV path required. Usage: python data/db.py import-nba <csv_path>")
            sys.exit(1)
        if not os.path.exists(args.csv_path):
            print(f"ERROR: File not found: {args.csv_path}")
            sys.exit(1)
        import_nba_csv(conn, args.csv_path)

    elif args.command == 'stats':
        stats = get_db_stats(conn)
        print(f"\nDatabase: {args.db_path}")
        print(f"{'='*40}")
        for table, count in stats.items():
            status = f"{count:>10,} rows" if count >= 0 else "  (missing)"
            print(f"  {table:<25} {status}")
        total = sum(c for c in stats.values() if c > 0)
        print(f"{'='*40}")
        print(f"  {'TOTAL':<25} {total:>10,} rows")

    conn.close()


if __name__ == '__main__':
    main()
