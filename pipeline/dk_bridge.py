#!/usr/bin/env python3
"""
dk_to_optimizer.py — Bridge: DraftKings CSV + Training Predictions → Optimizer-Ready CSV

Automates the manual step between training and optimization:
  1. Scans data/dk_drop/ for the latest DraftKings CSV export
  2. Parses the DK format (DKSalaries or DKEntries with embedded player pool)
  3. Loads training predictions (final_predictions_with_probabilities.csv)
  4. Merges on player name (fuzzy matching for name mismatches)
  5. Outputs a single CSV ready for the web optimizer upload

Usage:
  python pipeline/dk_bridge.py
  python pipeline/dk_bridge.py --dk-file data/dk_drop/DKSalaries.csv
  python pipeline/dk_bridge.py --predictions-dir output/
  python pipeline/dk_bridge.py --sport MLB

Output:  data/optimizer_ready/YYYY-MM-DD_MLB_optimizer_ready.csv
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DK_DROP_DIR = os.path.join(REPO_ROOT, 'data', 'dk_drop')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'data', 'optimizer_ready')
DEFAULT_PREDICTIONS_DIR = os.path.join(REPO_ROOT, 'training', 'mlb', 'output')


# ---------------------------------------------------------------------------
# DK CSV Parsing — handles both DKSalaries and DKEntries formats
# ---------------------------------------------------------------------------

# Known header patterns in DK exports
DK_HEADERS = [
    ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
     'Game Info', 'TeamAbbrev', 'AvgPointsPerGame'],
    ['Position', 'Name+ID', 'Name', 'ID', 'Roster Position', 'Salary',
     'Game Info', 'TeamAbbrev', 'AvgPointsPerGame'],
    ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
     'Game Info', 'TeamAbbrev', 'Fantasy Points'],
    ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
     'Game Info', 'TeamAbbrev', 'FPTS'],
]


def parse_dk_csv(file_path):
    """
    Parse a DraftKings CSV export into a clean DataFrame.

    Handles:
      - DKSalaries.csv (clean CSV with Position, Name+ID, Name, ID, ...)
      - DKEntries-*.csv (entries file with embedded player pool block)

    Returns DataFrame with columns:
      Name, DK_ID, Position, Roster_Position, Salary, Team, Game_Info, AvgPointsPerGame
    """
    print(f"Parsing DK file: {os.path.basename(file_path)}")

    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"  Failed to read: {e}")
        return pd.DataFrame()

    # --- Strategy 1: Find embedded player pool header ---
    start_idx = None
    start_col = None
    header_len = 0
    matched_header = None

    for i, row in enumerate(rows):
        clean = [c.strip() for c in row]
        for header in DK_HEADERS:
            for c in range(max(0, len(clean) - len(header)) + 1):
                if clean[c:c + len(header)] == header:
                    start_idx = i
                    start_col = c
                    header_len = len(header)
                    matched_header = header
                    break
            if start_idx is not None:
                break
        if start_idx is not None:
            break

    if start_idx is not None:
        print(f"  Found player pool at row {start_idx + 1} (col offset {start_col})")
        records = []
        for j in range(start_idx + 1, len(rows)):
            r = rows[j]
            if len(r) < start_col + header_len:
                continue
            position = (r[start_col + 0] or '').strip()
            name = (r[start_col + 2] or '').strip()
            dk_id = (r[start_col + 3] or '').strip()
            roster_pos = (r[start_col + 4] or '').strip()
            salary_raw = (r[start_col + 5] or '').strip().replace(',', '').replace('$', '')
            game_info = (r[start_col + 6] or '').strip()
            team = (r[start_col + 7] or '').strip().upper()
            avg_pts_raw = (r[start_col + 8] or '').strip() if start_col + 8 < len(r) else ''

            if not dk_id.isdigit():
                continue

            try:
                salary = int(float(salary_raw)) if salary_raw else 0
            except ValueError:
                salary = 0

            try:
                avg_pts = float(avg_pts_raw) if avg_pts_raw else 0.0
            except ValueError:
                avg_pts = 0.0

            # Extract opponent from game info (e.g. "TEX@HOU 06/15/2025 07:05PM ET")
            opponent = ''
            game_match = re.match(r'([A-Z]+)@([A-Z]+)', game_info)
            if game_match:
                t1, t2 = game_match.groups()
                opponent = t2 if team == t1 else t1

            records.append({
                'Name': name,
                'DK_ID': dk_id,
                'Position': position,
                'Roster_Position': roster_pos,
                'Salary': salary,
                'Team': team,
                'Opponent': opponent,
                'Game_Info': game_info,
                'AvgPointsPerGame': avg_pts,
            })

        if records:
            df = pd.DataFrame(records).drop_duplicates('DK_ID')
            print(f"  Parsed {len(df)} players from embedded pool")
            return df

    # --- Strategy 2: Try plain CSV read (DKSalaries.csv format) ---
    print("  No embedded pool found, trying plain CSV parse...")
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"  Plain CSV read failed: {e}")
        return pd.DataFrame()

    # Map columns to our standard names
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ('name', 'player', 'playername'):
            col_map[col] = 'Name'
        elif lower in ('id', 'dk_id', 'player_id'):
            col_map[col] = 'DK_ID'
        elif lower in ('position', 'pos'):
            col_map[col] = 'Position'
        elif lower in ('roster position', 'roster_position', 'rosterposition'):
            col_map[col] = 'Roster_Position'
        elif lower in ('salary', 'cost', 'dk_salary'):
            col_map[col] = 'Salary'
        elif lower in ('teamabbrev', 'team', 'tm'):
            col_map[col] = 'Team'
        elif lower in ('game info', 'gameinfo', 'game_info'):
            col_map[col] = 'Game_Info'
        elif lower in ('avgpointspergame', 'fpts', 'fantasy points', 'fantasypoints'):
            col_map[col] = 'AvgPointsPerGame'
        elif lower in ('opponent', 'opp'):
            col_map[col] = 'Opponent'

    df = df.rename(columns=col_map)
    required = ['Name', 'Salary']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  Missing required columns after mapping: {missing}")
        print(f"  Available: {list(df.columns)}")
        return pd.DataFrame()

    # Clean salary
    if df['Salary'].dtype == object:
        df['Salary'] = df['Salary'].str.replace(',', '').str.replace('$', '').astype(float).astype(int)

    print(f"  Parsed {len(df)} players from plain CSV")
    return df


# ---------------------------------------------------------------------------
# Name normalization for matching
# ---------------------------------------------------------------------------

def normalize_name(name):
    """Normalize player name for fuzzy matching."""
    if not isinstance(name, str):
        return ''
    # Lowercase, strip whitespace, remove periods/hyphens, Jr/Sr/III etc
    n = name.strip().lower()
    n = re.sub(r'[.\'-]', '', n)
    n = re.sub(r'\s+(jr|sr|ii|iii|iv|v)$', '', n)
    n = re.sub(r'\s+', ' ', n)
    return n


def build_name_lookup(names):
    """Build a dict mapping normalized names to original names."""
    lookup = {}
    for name in names:
        key = normalize_name(name)
        if key:
            lookup[key] = name
    return lookup


# ---------------------------------------------------------------------------
# Load training predictions
# ---------------------------------------------------------------------------

QUANT_PROFILE_COLS = [
    'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
    'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
    'entropy', 'hurst_exponent', 'rolling_sharpe',
    'avg_player_correlation', 'correlation_volatility',
    'evt_return_level', 'exceedance_prob',
]


def load_quant_profile(predictions_dir):
    """
    Load player_quant_profile.csv from the predictions directory if it exists.
    Returns a DataFrame or None.
    """
    path = os.path.join(predictions_dir, 'player_quant_profile.csv')
    if os.path.exists(path):
        qdf = pd.read_csv(path)
        print(f"Loaded quant profile: {len(qdf)} players, columns: {[c for c in qdf.columns if c != 'Name']}")
        return qdf
    else:
        print(f"No quant profile found at {path} (optional)")
        return None


def load_predictions(predictions_dir):
    """
    Load the best available predictions file from the training output.
    Priority: final_predictions_with_probabilities.csv > probability_summary.csv > final_predictions.csv
    """
    candidates = [
        'final_predictions_with_probabilities.csv',
        'probability_summary.csv',
        'final_predictions.csv',
    ]

    for fname in candidates:
        path = os.path.join(predictions_dir, fname)
        if os.path.exists(path):
            print(f"Loading predictions: {fname}")
            df = pd.read_csv(path)
            print(f"  {len(df)} rows, columns: {list(df.columns)}")
            return df, fname

    print(f"No prediction files found in {predictions_dir}")
    print(f"  Looked for: {candidates}")
    return None, None


# ---------------------------------------------------------------------------
# Detect sport from DK data
# ---------------------------------------------------------------------------

def detect_sport(dk_df):
    """Auto-detect sport from position columns in DK data."""
    if 'Position' not in dk_df.columns and 'Roster_Position' not in dk_df.columns:
        return 'MLB'  # default

    pos_col = 'Roster_Position' if 'Roster_Position' in dk_df.columns else 'Position'
    positions = dk_df[pos_col].str.upper().str.split('/').explode().unique()
    positions = set(p.strip() for p in positions if isinstance(p, str))

    # MLB positions
    mlb_pos = {'P', 'C', '1B', '2B', '3B', 'SS', 'OF'}
    # NFL positions
    nfl_pos = {'QB', 'RB', 'WR', 'TE', 'DST', 'K', 'FLEX'}
    # NBA positions
    nba_pos = {'PG', 'SG', 'SF', 'PF', 'UTIL', 'G', 'F'}

    mlb_overlap = len(positions & mlb_pos)
    nfl_overlap = len(positions & nfl_pos)
    nba_overlap = len(positions & nba_pos)

    if mlb_overlap >= nfl_overlap and mlb_overlap >= nba_overlap:
        return 'MLB'
    elif nfl_overlap >= nba_overlap:
        return 'NFL'
    else:
        return 'NBA'


# ---------------------------------------------------------------------------
# Merge DK + Predictions
# ---------------------------------------------------------------------------

def merge_dk_with_predictions(dk_df, pred_df, sport, quant_df=None):
    """
    Merge DraftKings player pool with training predictions on player name.

    For players without predictions, uses AvgPointsPerGame from DK as fallback.
    If quant_df is provided, merges the 14 quant profile columns onto the result.
    """
    # Get the most recent prediction per player (last date)
    if 'Date' in pred_df.columns:
        pred_df['Date'] = pd.to_datetime(pred_df['Date'], errors='coerce')
        # Use most recent prediction per player
        latest_pred = (pred_df
                       .sort_values('Date', ascending=False)
                       .drop_duplicates('Name', keep='first')
                       .copy())
    else:
        latest_pred = pred_df.drop_duplicates('Name', keep='last').copy()

    # Build name lookups
    dk_lookup = build_name_lookup(dk_df['Name'].unique())
    pred_lookup = build_name_lookup(latest_pred['Name'].unique())

    # Add normalized name columns
    dk_df = dk_df.copy()
    dk_df['_norm_name'] = dk_df['Name'].apply(normalize_name)
    latest_pred = latest_pred.copy()
    latest_pred['_norm_name'] = latest_pred['Name'].apply(normalize_name)

    # Merge on normalized name
    merged = dk_df.merge(
        latest_pred[['_norm_name', 'Predicted'] +
                     [c for c in latest_pred.columns
                      if c.startswith('prediction_') or c.startswith('prob_') or c.startswith('q')]],
        on='_norm_name',
        how='left',
    )

    matched = merged['Predicted'].notna().sum()
    total = len(merged)
    print(f"\nName matching: {matched}/{total} players matched ({matched/total*100:.0f}%)")

    if matched < total:
        unmatched = merged[merged['Predicted'].isna()]['Name'].tolist()
        print(f"  Unmatched ({len(unmatched)}): {', '.join(unmatched[:15])}")
        if len(unmatched) > 15:
            print(f"    ... and {len(unmatched) - 15} more")

    # Fallback: use AvgPointsPerGame from DK for unmatched players
    if 'AvgPointsPerGame' in merged.columns:
        fallback_mask = merged['Predicted'].isna() & (merged['AvgPointsPerGame'] > 0)
        n_fallback = fallback_mask.sum()
        if n_fallback > 0:
            merged.loc[fallback_mask, 'Predicted'] = merged.loc[fallback_mask, 'AvgPointsPerGame']
            print(f"  Used DK AvgPointsPerGame as fallback for {n_fallback} players")

    # Still missing? Use salary-based estimate
    still_missing = merged['Predicted'].isna()
    if still_missing.sum() > 0:
        # MLB: roughly 2.5-3.5 pts per $1K salary
        tier_ratios = {
            'MLB': [(8000, 3.5), (6000, 3.2), (4000, 3.0), (0, 2.7)],
            'NBA': [(9000, 4.8), (7000, 4.5), (5000, 4.2), (0, 3.8)],
            'NFL': [(8000, 3.0), (6000, 2.8), (4000, 2.5), (0, 2.2)],
        }
        ratios = tier_ratios.get(sport, tier_ratios['MLB'])
        for idx in merged[still_missing].index:
            sal = merged.loc[idx, 'Salary']
            ratio = next((r for min_sal, r in ratios if sal >= min_sal), ratios[-1][1])
            merged.loc[idx, 'Predicted'] = round(sal / 1000 * ratio, 2)
        print(f"  Used salary-based estimate for {still_missing.sum()} remaining players")

    # Merge quant profile columns if available
    if quant_df is not None:
        quant_df = quant_df.copy()
        quant_df['_norm_name'] = quant_df['Name'].apply(normalize_name)
        quant_cols = [c for c in quant_df.columns if c in QUANT_PROFILE_COLS]
        if quant_cols:
            merged = merged.merge(
                quant_df[['_norm_name'] + quant_cols],
                on='_norm_name',
                how='left',
            )
            quant_matched = merged[quant_cols[0]].notna().sum()
            print(f"  Quant profile: {quant_matched}/{len(merged)} players matched ({len(quant_cols)} columns)")

    return merged


# ---------------------------------------------------------------------------
# Format output for optimizer
# ---------------------------------------------------------------------------

def format_for_optimizer(merged_df, sport):
    """
    Format the merged DataFrame into the columns the web optimizer expects.

    Output columns: Name, Pos, Team, Salary, Predicted_DK_Points, Opponent,
                    Ceiling, Floor, StdDev (if available from quantile models)
    """
    out = pd.DataFrame()
    out['Name'] = merged_df['Name']

    # Position: use Roster_Position (multi-eligible) if available, else Position
    if 'Roster_Position' in merged_df.columns:
        out['Pos'] = merged_df['Roster_Position']
    elif 'Position' in merged_df.columns:
        out['Pos'] = merged_df['Position']
    else:
        out['Pos'] = 'UTIL'

    out['Team'] = merged_df.get('Team', '')
    out['Salary'] = merged_df['Salary']
    out['Predicted_DK_Points'] = merged_df['Predicted'].round(2)

    if 'Opponent' in merged_df.columns:
        out['Opponent'] = merged_df['Opponent']

    if 'DK_ID' in merged_df.columns:
        out['DK_ID'] = merged_df['DK_ID']

    # Quantile-derived metrics (from training pipeline CQR)
    if 'prediction_upper_80' in merged_df.columns:
        out['Ceiling'] = merged_df['prediction_upper_80'].round(2)
    if 'prediction_lower_80' in merged_df.columns:
        out['Floor'] = merged_df['prediction_lower_80'].round(2)
    if 'prediction_std' in merged_df.columns:
        out['StdDev'] = merged_df['prediction_std'].round(2)

    # Probability columns (prob_over_X from quantile models)
    for col in merged_df.columns:
        if col.startswith('prob_over_'):
            out[col] = merged_df[col].round(4)

    # Quant profile columns (from training pipeline)
    for col in QUANT_PROFILE_COLS:
        if col in merged_df.columns and merged_df[col].notna().any():
            out[col] = merged_df[col].round(6)

    # Drop players with zero salary (likely invalid)
    out = out[out['Salary'] > 0].copy()

    # Sort by salary descending (convention)
    out = out.sort_values('Salary', ascending=False).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Find latest DK file in drop folder
# ---------------------------------------------------------------------------

def find_latest_dk_file(drop_dir):
    """Find the most recently modified CSV in the drop folder."""
    patterns = ['*.csv', '*.CSV']
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(drop_dir, pat)))

    if not files:
        return None

    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Bridge: DraftKings CSV + Training Predictions → Optimizer-Ready CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect: scan dk_drop folder, use default predictions
  python pipeline/dk_bridge.py

  # Specify DK file explicitly
  python pipeline/dk_bridge.py --dk-file ~/Downloads/DKSalaries.csv

  # Specify predictions directory
  python pipeline/dk_bridge.py --predictions-dir training/mlb/output

  # Force sport detection
  python pipeline/dk_bridge.py --sport MLB
        """,
    )
    parser.add_argument('--dk-file', type=str, default=None,
                        help='Path to DraftKings CSV (default: latest file in data/dk_drop/)')
    parser.add_argument('--predictions-dir', type=str, default=DEFAULT_PREDICTIONS_DIR,
                        help='Directory containing training prediction CSVs')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for optimizer-ready CSV')
    parser.add_argument('--sport', type=str, choices=['MLB', 'NBA', 'NFL'], default=None,
                        help='Force sport (default: auto-detect from positions)')
    parser.add_argument('--no-predictions', action='store_true',
                        help='Skip merging predictions; use DK AvgPointsPerGame only')
    args = parser.parse_args()

    print("=" * 60)
    print("DK → OPTIMIZER BRIDGE")
    print("=" * 60)

    # 1. Find DK file
    if args.dk_file:
        dk_path = args.dk_file
    else:
        dk_path = find_latest_dk_file(DK_DROP_DIR)

    if not dk_path or not os.path.exists(dk_path):
        print(f"\nNo DraftKings CSV found.")
        print(f"  Drop your DK export into: {DK_DROP_DIR}/")
        print(f"  Or specify: --dk-file /path/to/DKSalaries.csv")
        sys.exit(1)

    print(f"\nDK file: {dk_path}")

    # 2. Parse DK CSV
    dk_df = parse_dk_csv(dk_path)
    if dk_df.empty:
        print("Failed to parse DK file. Check format.")
        sys.exit(1)

    # 3. Detect sport
    sport = args.sport or detect_sport(dk_df)
    print(f"Sport: {sport}")

    # 4. Load predictions (unless --no-predictions)
    if not args.no_predictions:
        pred_df, pred_file = load_predictions(args.predictions_dir)
    else:
        pred_df = None
        pred_file = None

    # 4b. Load quant profile (optional — from training pipeline)
    quant_df = load_quant_profile(args.predictions_dir)

    # 5. Merge
    if pred_df is not None:
        merged = merge_dk_with_predictions(dk_df, pred_df, sport, quant_df=quant_df)
    else:
        print("\nNo predictions loaded — using DK AvgPointsPerGame as projection")
        merged = dk_df.copy()
        if 'AvgPointsPerGame' in merged.columns:
            merged['Predicted'] = merged['AvgPointsPerGame']
        else:
            # Salary-based fallback
            merged['Predicted'] = (merged['Salary'] / 1000 * 3.0).round(2)

    # 6. Format for optimizer
    output_df = format_for_optimizer(merged, sport)

    # 7. Save
    os.makedirs(args.output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    out_filename = f"{today}_{sport}_optimizer_ready.csv"
    out_path = os.path.join(args.output_dir, out_filename)
    output_df.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"OUTPUT: {out_path}")
    print(f"{'=' * 60}")
    print(f"  Players:    {len(output_df)}")
    print(f"  Sport:      {sport}")
    print(f"  Salary:     ${output_df['Salary'].min():,} - ${output_df['Salary'].max():,}")
    print(f"  Projection: {output_df['Predicted_DK_Points'].min():.1f} - {output_df['Predicted_DK_Points'].max():.1f}")
    has_ceiling = 'Ceiling' in output_df.columns and output_df['Ceiling'].notna().any()
    has_floor = 'Floor' in output_df.columns and output_df['Floor'].notna().any()
    has_std = 'StdDev' in output_df.columns and output_df['StdDev'].notna().any()
    print(f"  Ceiling:    {'Yes' if has_ceiling else 'No (run training with quantile models)'}")
    print(f"  Floor:      {'Yes' if has_floor else 'No'}")
    print(f"  StdDev:     {'Yes' if has_std else 'No'}")
    if pred_file:
        print(f"  Predictions: {pred_file}")
    else:
        print(f"  Predictions: DK AvgPointsPerGame / salary-based fallback")
    print(f"\nUpload this CSV to the web optimizer or pass to the Python optimizer.")


if __name__ == '__main__':
    main()
