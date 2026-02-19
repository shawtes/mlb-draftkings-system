"""
matchup_engine.py — Pitcher-Batter Matchup Feature Engine

Adds opposing pitcher context to batter DK fantasy point predictions.
Uses pybaseball to build a date+team → opposing_starter lookup, then
merges the starter's lagged rolling profile onto each batter-game row.

Usage (standalone test):
    python matchup_engine.py --batter-csv /path/to/merged_fangraphs_data.csv \
                             --pitcher-csv /path/to/merged_fangraphs_data_pitchers.csv \
                             --cache-dir ./data/matchup_cache \
                             --start-date 2025-03-27 --end-date 2025-04-27

Integration (called from training.py):
    from matchup_engine import prepare_matchup_dataset
    df = prepare_matchup_dataset(df, pitcher_csv=..., cache_dir=...,
                                 start_date=..., end_date=...)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pitcher stats to compute lagged rolling averages for (Tier 1)
PITCHER_ROLLING_STATS = [
    'K/9', 'BB/9', 'HR/9', 'ERA', 'FIP', 'xFIP', 'WHIP',
    'K%', 'BB%', 'GB%', 'FB%', 'SwStr%', 'Contact%',
    'O-Swing%', 'Z-Contact%', 'SIERA', 'LOB%',
    'ERA-', 'FIP-', 'HR/FB', 'LD%', 'BABIP',
]

# Pitcher arsenal stats (Tier 4 — velocity and pitch mix)
PITCHER_ARSENAL_STATS = [
    'FBv', 'FB%.1', 'SL%', 'SLv', 'CH%', 'CHv', 'CB%', 'CBv',
]

ROLLING_WINDOWS = [7, 14, 28]

# Minimum fill rate to include a pitcher stat (skip cols that are mostly NaN)
MIN_FILL_RATE = 0.30

# pybaseball uses full city names; FanGraphs uses abbreviations.
# Map full→abbrev for team matching.
TEAM_NAME_TO_ABBREV = {
    'Arizona': 'ARI', 'Atlanta': 'ATL', 'Baltimore': 'BAL',
    'Boston': 'BOS', 'Chi Cubs': 'CHC', 'Chi White Sox': 'CHW',
    'Chicago': 'CHC',  # ambiguous — context-dependent
    'Cincinnati': 'CIN', 'Cleveland': 'CLE', 'Colorado': 'COL',
    'Detroit': 'DET', 'Houston': 'HOU', 'Kansas City': 'KCR',
    'LA Angels': 'LAA', 'LA Dodgers': 'LAD', 'Los Angeles': 'LAD',
    'Miami': 'MIA', 'Milwaukee': 'MIL', 'Minnesota': 'MIN',
    'NY Mets': 'NYM', 'NY Yankees': 'NYY', 'New York': 'NYY',
    'Oakland': 'ATH', 'Athletics': 'ATH', 'Philadelphia': 'PHI',
    'Pittsburgh': 'PIT', 'San Diego': 'SDP', 'San Francisco': 'SFG',
    'Seattle': 'SEA', 'St. Louis': 'STL', 'Tampa Bay': 'TBR',
    'Texas': 'TEX', 'Toronto': 'TOR', 'Washington': 'WSN',
    # Alternate spellings
    'CHC': 'CHC', 'CHW': 'CHW', 'CWS': 'CHW',
    'ARI': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
    'CIN': 'CIN', 'CLE': 'CLE', 'COL': 'COL', 'DET': 'DET',
    'HOU': 'HOU', 'KCR': 'KCR', 'KC': 'KCR', 'LAA': 'LAA',
    'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL', 'MIN': 'MIN',
    'NYM': 'NYM', 'NYY': 'NYY', 'ATH': 'ATH', 'OAK': 'ATH',
    'PHI': 'PHI', 'PIT': 'PIT', 'SDP': 'SDP', 'SD': 'SDP',
    'SFG': 'SFG', 'SF': 'SFG', 'SEA': 'SEA', 'STL': 'STL',
    'TBR': 'TBR', 'TB': 'TBR', 'TEX': 'TEX', 'TOR': 'TOR',
    'WSN': 'WSN', 'WSH': 'WSN', 'WAS': 'WSN',
}


def _normalize_team(name):
    """Convert a team name (full or abbreviation) to standard FanGraphs abbreviation."""
    if pd.isna(name):
        return None
    name = str(name).strip()
    return TEAM_NAME_TO_ABBREV.get(name, name)


# ---------------------------------------------------------------------------
# 1. Opponent Lookup: date + team → opposing starter
# ---------------------------------------------------------------------------

def build_opponent_lookup_from_pybaseball(start_date, end_date, cache_dir=None):
    """Use pybaseball pitching_stats_range (day-by-day) to find each date's
    starting pitcher and which team they faced.

    NOTE: pybaseball only returns Date/Opp columns on single-day queries.
    Multi-day ranges aggregate stats and drop per-game info. So we fetch
    one day at a time and cache each result.

    Returns DataFrame:
        date | pitcher_name | pitcher_mlbID | pitcher_team | opponent_team
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Check full-range cache first
    full_cache_path = None
    if cache_dir:
        full_cache_path = os.path.join(cache_dir, f'opponent_lookup_{start_date}_{end_date}.csv')
        if os.path.exists(full_cache_path):
            print(f"  Loading cached opponent lookup from {full_cache_path}")
            return pd.read_csv(full_cache_path)

    try:
        from pybaseball import pitching_stats_range
    except ImportError:
        print("WARNING: pybaseball not installed. Cannot build opponent lookup.")
        return pd.DataFrame()

    # Generate list of dates to fetch
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    print(f"  Fetching pitching game logs from pybaseball ({len(date_range)} days)...")
    t0 = time.time()

    all_rows = []
    cached_count = 0
    fetched_count = 0

    for dt in date_range:
        date_str = dt.strftime('%Y-%m-%d')
        day_cache = os.path.join(cache_dir, f'starters_{date_str}.csv') if cache_dir else None

        if day_cache and os.path.exists(day_cache):
            day_df = pd.read_csv(day_cache)
            cached_count += 1
        else:
            try:
                pitch_logs = pitching_stats_range(date_str, date_str)
                fetched_count += 1
            except Exception as e:
                # No games on this date or network error — skip
                continue

            if len(pitch_logs) == 0 or 'GS' not in pitch_logs.columns:
                continue

            starters = pitch_logs[pitch_logs['GS'] == 1].copy()
            if len(starters) == 0:
                continue

            n = len(starters)
            day_df = pd.DataFrame({
                'date': [date_str] * n,
                'pitcher_name': starters['Name'].values,
                'pitcher_mlbID': starters['mlbID'].values if 'mlbID' in starters.columns else [None] * n,
                'pitcher_team': starters['Tm'].apply(_normalize_team).values,
                'opponent_team': (
                    starters['Opp'].str.replace(r'^@?\s*', '', regex=True)
                    .apply(_normalize_team).values
                    if 'Opp' in starters.columns else [None] * n
                ),
            })

            if day_cache:
                day_df.to_csv(day_cache, index=False)

        if len(day_df) > 0:
            all_rows.append(day_df)

    if not all_rows:
        print("  WARNING: No starters found in pybaseball data.")
        return pd.DataFrame()

    lookup = pd.concat(all_rows, ignore_index=True)

    # Dedupe: keep first starter per team per date (handles doubleheaders)
    lookup = lookup.drop_duplicates(subset=['date', 'pitcher_team'], keep='first')
    # Also dedupe by (date, opponent_team) — ensures 1-to-1 merge with batters
    lookup = lookup.drop_duplicates(subset=['date', 'opponent_team'], keep='first')

    elapsed = time.time() - t0
    print(f"  Built opponent lookup: {len(lookup)} starter-game rows across "
          f"{lookup['date'].nunique()} dates ({cached_count} cached, {fetched_count} fetched, "
          f"{elapsed:.1f}s)")

    # Save full-range cache
    if full_cache_path:
        lookup.to_csv(full_cache_path, index=False)
        print(f"  Cached to {full_cache_path}")

    return lookup


def build_opponent_lookup_from_pitcher_csv(pitcher_df):
    """Build opponent lookup from our existing FanGraphs pitcher CSV.

    Since our pitcher CSV doesn't have an Opp column, we infer opponents:
    on each date, teams come in pairs (they played each other).
    For dates with an even number of teams, we pair by matching teams.
    For full MLB slates (30 teams = 15 games), each team has exactly 1 starter.
    """
    starters = pitcher_df[pitcher_df['GS'] > 0].copy()
    starters['date'] = pd.to_datetime(starters['date']).dt.strftime('%Y-%m-%d')

    rows = []
    for date, group in starters.groupby('date'):
        # Keep only 1 starter per team per date (handles edge cases)
        team_starters = group.drop_duplicates(subset='Team', keep='first')
        teams = list(team_starters['Team'].unique())
        n_teams = len(teams)

        if n_teams == 2:
            # Simple case: 2 teams = they played each other
            for _, row in team_starters.iterrows():
                opp = [t for t in teams if t != row['Team']][0]
                rows.append({
                    'date': date,
                    'pitcher_name': row['Name'],
                    'pitcher_mlbID': row.get('MLBAMID', None),
                    'pitcher_team': _normalize_team(row['Team']),
                    'opponent_team': _normalize_team(opp),
                })
        elif n_teams > 2 and n_teams % 2 == 0:
            # Full or partial slate — pair teams by schedule knowledge
            # We can't know the pairings without a schedule, but if we have
            # exactly 2 starters for each "game", the pairing is implicit.
            # For now, skip multi-team dates in CSV fallback — pybaseball handles these.
            pass

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Pitcher Rolling Profiles
# ---------------------------------------------------------------------------

def compute_pitcher_rolling_profiles(pitcher_df, windows=None):
    """Compute lagged rolling averages of pitcher stats, grouped by pitcher.

    All features are shifted by 1 game (uses only data BEFORE current game).

    Args:
        pitcher_df: DataFrame with pitcher game logs (one row per pitcher-game).
                    Must have columns: Name (or NameASCII), date, and stat columns.
        windows: List of rolling window sizes (default: [7, 14, 28])

    Returns:
        DataFrame keyed by (pitcher_name, date) with opp_* columns.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = pitcher_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Name', 'date'])

    # Determine which stats are actually available and have decent fill
    available_stats = []
    for stat in PITCHER_ROLLING_STATS + PITCHER_ARSENAL_STATS:
        if stat in df.columns:
            fill_rate = df[stat].notna().mean()
            if fill_rate >= MIN_FILL_RATE:
                available_stats.append(stat)

    print(f"  Computing pitcher rolling profiles: {len(available_stats)} stats x "
          f"{len(windows)} windows")

    profile_cols = {}
    for stat in available_stats:
        # Convert to numeric (some FanGraphs columns may have % signs)
        series = pd.to_numeric(df[stat], errors='coerce')
        for w in windows:
            col_name = f'opp_{stat}_{w}g'
            # Rolling mean of past games, shifted by 1 (excludes current game)
            profile_cols[col_name] = series.groupby(df['Name']).transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )

    profiles = pd.DataFrame(profile_cols, index=df.index)
    profiles['pitcher_name'] = df['Name'].values
    profiles['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Also add the pitcher's career/season-to-date averages as fallback features
    for stat in available_stats[:10]:  # Top 10 most important stats
        series = pd.to_numeric(df[stat], errors='coerce')
        col_name = f'opp_{stat}_season'
        profiles[col_name] = series.groupby(df['Name']).transform(
            lambda x: x.expanding().mean().shift(1)
        )

    # Dedupe: keep last entry per pitcher per date
    profiles = profiles.drop_duplicates(subset=['pitcher_name', 'date'], keep='last')

    n_features = len([c for c in profiles.columns if c.startswith('opp_')])
    print(f"  Created {n_features} pitcher profile features for "
          f"{profiles['pitcher_name'].nunique()} pitchers")

    return profiles


# ---------------------------------------------------------------------------
# 3. Matchup Interaction Features
# ---------------------------------------------------------------------------

def compute_interaction_features(df):
    """Compute Tier 2 matchup interaction features from batter stats +
    opposing pitcher profile.

    Expects df to already have batter columns (K%, ISO, Contact%, BB%, GB%, wOBA)
    and opp_* pitcher columns.

    Returns df with new matchup_* columns.
    """
    n_before = len([c for c in df.columns if c.startswith('matchup_')])

    # Use 14-game window as primary (good balance of stability and recency)
    opp_k = _get_best_opp_col(df, 'K%')
    opp_bb = _get_best_opp_col(df, 'BB%')
    opp_gb = _get_best_opp_col(df, 'GB%')
    opp_fb = _get_best_opp_col(df, 'FB%')
    opp_fip = _get_best_opp_col(df, 'FIP')
    opp_xfip = _get_best_opp_col(df, 'xFIP')
    opp_contact = _get_best_opp_col(df, 'Contact%')
    opp_swstr = _get_best_opp_col(df, 'SwStr%')

    # Batter stats — use pgm_ (pregame lagged) versions if available, else raw
    bat_k = _get_batter_col(df, 'K%')
    bat_bb = _get_batter_col(df, 'BB%')
    bat_iso = _get_batter_col(df, 'ISO')
    bat_contact = _get_batter_col(df, 'Contact%')
    bat_gb = _get_batter_col(df, 'GB%')
    bat_woba = _get_batter_col(df, 'wOBA')

    # --- Interaction features ---

    # FanGraphs log5 expected K rate
    if opp_k is not None and bat_k is not None:
        b = df[bat_k].clip(0.01, 0.99)
        p = df[opp_k].clip(0.01, 0.99)
        df['matchup_K_rate'] = (b * p) / (0.84 * b * p + 0.16)

    # Power hitter vs fly-ball pitcher
    if bat_iso is not None and opp_fb is not None:
        df['matchup_power_vs_flyball'] = df[bat_iso] * df[opp_fb]

    # Contact advantage: batter makes more contact than pitcher allows
    if bat_contact is not None and opp_contact is not None:
        df['matchup_contact_advantage'] = df[bat_contact] - df[opp_contact]

    # Discipline edge: batter walks more than pitcher walks people
    if bat_bb is not None and opp_bb is not None:
        df['matchup_discipline_edge'] = df[bat_bb] - df[opp_bb]

    # K differential: pitcher K% - batter K% (positive = suppression)
    if opp_k is not None and bat_k is not None:
        df['matchup_K_differential'] = df[opp_k] - df[bat_k]

    # Ground ball risk: both ground-ball-prone = low ceiling
    if bat_gb is not None and opp_gb is not None:
        df['matchup_ground_ball_risk'] = df[bat_gb] * df[opp_gb]

    # Quality score: good hitter vs bad pitcher
    if bat_woba is not None and opp_xfip is not None:
        # Higher xFIP = worse pitcher → invert
        safe_xfip = df[opp_xfip].clip(lower=1.0)
        df['matchup_quality_score'] = df[bat_woba] / safe_xfip

    # SwStr suppression: high whiff pitcher vs any batter
    if opp_swstr is not None and bat_k is not None:
        df['matchup_swstr_suppression'] = df[opp_swstr] * df[bat_k]

    n_after = len([c for c in df.columns if c.startswith('matchup_')])
    print(f"  Computed {n_after - n_before} interaction features")

    return df


def _get_best_opp_col(df, stat_name):
    """Find the best available opp_ column for a stat, preferring 14g window."""
    candidates = [
        f'opp_{stat_name}_14g',
        f'opp_{stat_name}_7g',
        f'opp_{stat_name}_28g',
        f'opp_{stat_name}_season',
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_batter_col(df, stat_name):
    """Find the best available batter column, preferring pgm_ lagged versions."""
    candidates = [
        f'pgm_{stat_name}_14',
        f'pgm_{stat_name}_7',
        f'pgm_{stat_name}_28',
        stat_name,
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------

def prepare_matchup_dataset(batter_df, pitcher_csv=None, cache_dir=None,
                            start_date=None, end_date=None,
                            use_pybaseball=True):
    """Full matchup feature pipeline.

    1. Build opponent lookup (who faced whom on each date)
    2. Compute pitcher rolling profiles (lagged stats)
    3. Merge opposing pitcher profile onto each batter row
    4. Compute interaction features

    Args:
        batter_df: DataFrame of batter game logs (from merged_fangraphs_data.csv).
                   Must have 'Name', 'Team', 'date' columns.
        pitcher_csv: Path to pitcher CSV (merged_fangraphs_data_pitchers.csv).
        cache_dir: Directory to cache pybaseball scrapes.
        start_date: Earliest date to fetch opponent data for (YYYY-MM-DD).
        end_date: Latest date to fetch opponent data for (YYYY-MM-DD).
        use_pybaseball: If True, use pybaseball for opponent lookup.

    Returns:
        batter_df with opp_* and matchup_* columns added.
    """
    print(f"\n{'='*60}")
    print("MATCHUP ENGINE: Adding opposing pitcher features")
    print(f"{'='*60}")
    t0 = time.time()

    batter_df = batter_df.copy()
    batter_df['date'] = pd.to_datetime(batter_df['date']).dt.strftime('%Y-%m-%d')

    # Determine date range from batter data if not specified
    if start_date is None:
        start_date = batter_df['date'].min()
    if end_date is None:
        end_date = batter_df['date'].max()

    # --- Step 1: Load pitcher data ---
    pitcher_df = None
    if pitcher_csv and os.path.exists(pitcher_csv):
        print(f"\n  Loading pitcher data from {pitcher_csv}")
        pitcher_df = pd.read_csv(pitcher_csv)
        pitcher_df['date'] = pd.to_datetime(pitcher_df['date']).dt.strftime('%Y-%m-%d')
        print(f"  Pitcher data: {len(pitcher_df)} rows, {pitcher_df['Name'].nunique()} pitchers, "
              f"{pitcher_df['date'].nunique()} dates")

    # --- Step 2: Build opponent lookup ---
    print("\n  Building opponent lookup (date + team → opposing starter)...")
    opponent_lookup = pd.DataFrame()

    if use_pybaseball:
        opponent_lookup = build_opponent_lookup_from_pybaseball(
            start_date, end_date, cache_dir=cache_dir
        )

    # Fallback: try to build from pitcher CSV for dates not covered
    if pitcher_df is not None and len(opponent_lookup) == 0:
        print("  Falling back to pitcher CSV for opponent lookup...")
        opponent_lookup = build_opponent_lookup_from_pitcher_csv(pitcher_df)

    if len(opponent_lookup) == 0:
        print("  WARNING: No opponent lookup data. Skipping matchup features.")
        return batter_df

    # --- Step 3: Merge opposing starter onto batter rows ---
    print(f"\n  Merging opposing starter onto batter rows...")

    # For each batter row: batter is on Team X → find starter who pitched AGAINST Team X
    # In the lookup, opponent_team is the team the pitcher FACED
    # So we join: batter.Team == lookup.opponent_team AND same date
    # This gives us the pitcher who faced the batter's team

    # Normalize batter team names to match lookup abbreviations
    batter_team_normalized = batter_df['Team'].apply(_normalize_team)

    merge_key_batter = pd.DataFrame({
        'date': batter_df['date'].values,
        'opponent_team': batter_team_normalized.values,
    })

    merged = merge_key_batter.merge(
        opponent_lookup[['date', 'opponent_team', 'pitcher_name', 'pitcher_mlbID']],
        on=['date', 'opponent_team'],
        how='left'
    )

    batter_df['opposing_pitcher'] = merged['pitcher_name'].values
    batter_df['opposing_pitcher_mlbID'] = merged.get('pitcher_mlbID', pd.Series([None] * len(merged))).values

    n_matched = batter_df['opposing_pitcher'].notna().sum()
    n_total = len(batter_df)
    print(f"  Matched {n_matched}/{n_total} batter rows to opposing starter "
          f"({n_matched/n_total*100:.1f}%)")

    # --- Step 4: Compute pitcher rolling profiles ---
    if pitcher_df is not None and len(pitcher_df) > 0:
        print(f"\n  Computing pitcher rolling profiles from FanGraphs data...")
        pitcher_profiles = compute_pitcher_rolling_profiles(pitcher_df)
    else:
        # Build profiles from pybaseball data if no FanGraphs pitcher CSV
        print(f"\n  No pitcher CSV; building profiles from pybaseball game logs...")
        pitcher_profiles = _build_profiles_from_pybaseball(start_date, end_date, cache_dir)

    if pitcher_profiles is None or len(pitcher_profiles) == 0:
        print("  WARNING: No pitcher profiles computed. Skipping opp_* features.")
        return batter_df

    # --- Step 5: Merge pitcher profiles onto batter rows ---
    print(f"\n  Merging pitcher profiles onto batter rows...")

    opp_cols = [c for c in pitcher_profiles.columns if c.startswith('opp_')]
    merge_df = pitcher_profiles[['pitcher_name', 'date'] + opp_cols].copy()

    batter_df = batter_df.merge(
        merge_df,
        left_on=['opposing_pitcher', 'date'],
        right_on=['pitcher_name', 'date'],
        how='left',
        suffixes=('', '_pitcher_profile')
    )

    # Drop the redundant pitcher_name column from the merge
    if 'pitcher_name' in batter_df.columns:
        batter_df.drop(columns=['pitcher_name'], inplace=True, errors='ignore')

    n_opp_filled = batter_df[opp_cols[0]].notna().sum() if opp_cols else 0
    print(f"  {n_opp_filled}/{n_total} batter rows have pitcher profile data "
          f"({n_opp_filled/n_total*100:.1f}%)")

    # --- Step 6: Compute interaction features ---
    print(f"\n  Computing matchup interaction features...")
    batter_df = compute_interaction_features(batter_df)

    # --- Summary ---
    opp_features = [c for c in batter_df.columns if c.startswith('opp_')]
    matchup_features = [c for c in batter_df.columns if c.startswith('matchup_')]
    elapsed = time.time() - t0

    print(f"\n  Matchup engine complete in {elapsed:.1f}s")
    print(f"  Added {len(opp_features)} opp_* features + {len(matchup_features)} matchup_* features")
    print(f"  Total new columns: {len(opp_features) + len(matchup_features)}")

    return batter_df


def _build_profiles_from_pybaseball(start_date, end_date, cache_dir):
    """Build pitcher profiles from pybaseball pitching_stats_range data.

    This is the fallback when no FanGraphs pitcher CSV is available.
    Baseball Reference game logs have fewer stats but still useful:
    IP, SO, ERA, WHIP, SO9, SO/W, GB/FB, BAbip, etc.
    """
    try:
        from pybaseball import pitching_stats_range
    except ImportError:
        return None

    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'pitcher_bref_logs_{start_date}_{end_date}.csv')
        if os.path.exists(cache_path):
            pitch_logs = pd.read_csv(cache_path)
        else:
            pitch_logs = pitching_stats_range(start_date, end_date)
            pitch_logs.to_csv(cache_path, index=False)
    else:
        pitch_logs = pitching_stats_range(start_date, end_date)

    if len(pitch_logs) == 0:
        return None

    # Map Baseball Reference columns to our standard names
    bref_stat_map = {
        'SO9': 'K/9',
        'ERA': 'ERA',
        'WHIP': 'WHIP',
        'BAbip': 'BABIP',
        'GB/FB': 'GB/FB',
    }

    df = pitch_logs.copy()
    df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # Compute K% and BB% from raw counts if available
    if 'SO' in df.columns and 'BF' in df.columns:
        bf = pd.to_numeric(df['BF'], errors='coerce').clip(lower=1)
        df['K%'] = pd.to_numeric(df['SO'], errors='coerce') / bf
        df['BB%'] = pd.to_numeric(df['BB'], errors='coerce') / bf

    # Rename mapped columns
    for bref_col, our_col in bref_stat_map.items():
        if bref_col in df.columns and our_col not in df.columns:
            df[our_col] = pd.to_numeric(df[bref_col], errors='coerce')

    return compute_pitcher_rolling_profiles(df)


# ---------------------------------------------------------------------------
# 5. Feature Name Lists (for config.py integration)
# ---------------------------------------------------------------------------

def get_matchup_feature_names(pitcher_df=None):
    """Return list of all possible matchup feature names.
    Used by config.py to extend PREGAME_NUMERIC_FEATURES.
    """
    features = []

    stats = PITCHER_ROLLING_STATS + PITCHER_ARSENAL_STATS
    for stat in stats:
        for w in ROLLING_WINDOWS:
            features.append(f'opp_{stat}_{w}g')
        features.append(f'opp_{stat}_season')

    # Interaction features
    features.extend([
        'matchup_K_rate',
        'matchup_power_vs_flyball',
        'matchup_contact_advantage',
        'matchup_discipline_edge',
        'matchup_K_differential',
        'matchup_ground_ball_risk',
        'matchup_quality_score',
        'matchup_swstr_suppression',
    ])

    return features


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Matchup Engine Test')
    parser.add_argument('--batter-csv', type=str, required=True)
    parser.add_argument('--pitcher-csv', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='./data/matchup_cache')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--sample', type=int, default=None,
                        help='Only process first N rows (for testing)')
    args = parser.parse_args()

    print("Loading batter data...")
    batter_df = pd.read_csv(args.batter_csv)
    if args.sample:
        batter_df = batter_df.head(args.sample)
    print(f"Batter data: {len(batter_df)} rows")

    result = prepare_matchup_dataset(
        batter_df,
        pitcher_csv=args.pitcher_csv,
        cache_dir=args.cache_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Print summary
    opp_cols = [c for c in result.columns if c.startswith('opp_')]
    matchup_cols = [c for c in result.columns if c.startswith('matchup_')]
    print(f"\nFinal shape: {result.shape}")
    print(f"opp_* columns: {len(opp_cols)}")
    print(f"matchup_* columns: {len(matchup_cols)}")

    # Show fill rates for matchup features
    print("\nTop matchup feature fill rates:")
    for c in sorted(opp_cols + matchup_cols):
        fill = result[c].notna().mean() * 100
        if fill > 0:
            print(f"  {c:40s} {fill:.1f}%")
