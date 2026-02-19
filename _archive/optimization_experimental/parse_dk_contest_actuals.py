#!/usr/bin/env python3
"""
Parse DraftKings contest export to extract actual DK points per player (yesterday's results),
separate from projections. Supports multiple common DK formats:
- Columns with player rows: Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, Fantasy Points, FPTS, Points
- Standings/Entries files where lineups may not include per-player points (skipped)

Output: CSV with columns: DK_ID, Name, Actual_DK_Points, Roster_Position (if available), Team (if available)
"""

import argparse
import csv
import os
import sys
import pandas as pd


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def load_dk_entries_player_map(entries_path: str) -> pd.DataFrame:
    """Extract player name -> DK_ID mapping from a DraftKings entries CSV."""
    if not entries_path or not os.path.exists(entries_path):
        print(f"⚠️ DK entries file not found: {entries_path}")
        return pd.DataFrame()

    try:
        with open(entries_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"⚠️ Failed to read DK entries file '{entries_path}': {e}")
        return pd.DataFrame()

    header_variants = [
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'AvgPointsPerGame'
        ],
        [
            'Position', 'Name+ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'AvgPointsPerGame'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'Fantasy Points'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'FPTS'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'Points'
        ],
    ]

    start_idx = None
    start_col = None
    header_len = 0

    for i, row in enumerate(rows):
        clean = [c.strip() for c in row]
        for header in header_variants:
            for c in range(0, max(0, len(clean) - len(header)) + 1):
                if clean[c:c + len(header)] == header:
                    start_idx = i
                    start_col = c
                    header_len = len(header)
                    break
            if start_idx is not None:
                break
        if start_idx is not None:
            break

    if start_idx is None:
        print(f"⚠️ Could not locate player table in DK entries file: {entries_path}")
        return pd.DataFrame()

    records = []
    name_idx = 2
    id_idx = 3
    for j in range(start_idx + 1, len(rows)):
        r = rows[j]
        if len(r) < start_col + header_len:
            continue
        name = (r[start_col + name_idx] or '').strip()
        dk_id = (r[start_col + id_idx] or '').strip()
        if name and dk_id.isdigit():
            records.append({'Name_lower': name.lower(), 'DK_ID': dk_id})

    if not records:
        print(f"⚠️ No player entries extracted from DK entries file: {entries_path}")
        return pd.DataFrame()

    df = pd.DataFrame(records).drop_duplicates('Name_lower')
    print(f"✅ Loaded {len(df)} player mappings from DK entries file")
    return df


def parse_contest_csv(file_path: str) -> pd.DataFrame:
    """Robustly parse a DK contest CSV for per-player actual points.

    Tries two strategies:
    1) Scan for an embedded player table (columns include Name/ID and a points column like FPTS/Fantasy Points/Points)
    2) Fallback: try Pandas for row-based player tables
    """
    print_header(f"📥 LOADING CONTEST CSV: {os.path.basename(file_path)}")

    # First pass: raw CSV scan to locate an embedded player list header row
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return pd.DataFrame()

    # Candidate header variants observed in DK exports
    header_variants = [
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'AvgPointsPerGame', 'Fantasy Points'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'Fantasy Points'
        ],
        [
            'Position', 'Name+ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'Fantasy Points'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'FPTS'
        ],
        [
            'Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary',
            'Game Info', 'TeamAbbrev', 'Points'
        ],
    ]

    start_idx = None
    start_col = None
    header_len = 0
    for i, row in enumerate(rows):
        clean = [c.strip() for c in row]
        for header in header_variants:
            for c in range(0, max(0, len(clean) - len(header)) + 1):
                if clean[c:c + len(header)] == header:
                    start_idx = i
                    start_col = c
                    header_len = len(header)
                    break
            if start_idx is not None:
                break
        if start_idx is not None:
            break

    points_col_offsets = {
        'Fantasy Points': None,
        'FPTS': None,
        'Points': None,
    }

    def build_from_rows() -> pd.DataFrame:
        if start_idx is None:
            return pd.DataFrame()
        # Determine which points column exists by matching header slice
        header_row = [c.strip() for c in rows[start_idx][start_col:start_col + header_len]]
        for label in points_col_offsets.keys():
            if label in header_row:
                points_col_offsets[label] = header_row.index(label)
                break
        # Required columns relative to start_col
        name_idx = 2
        id_idx = 3
        roster_idx = 4
        team_idx = 7
        points_rel_idx = None
        for label, rel in points_col_offsets.items():
            if rel is not None:
                points_rel_idx = rel
                break

        records = []
        for j in range(start_idx + 1, len(rows)):
            r = rows[j]
            if len(r) < start_col + header_len:
                continue
            name = (r[start_col + name_idx] or '').strip()
            dk_id = (r[start_col + id_idx] or '').strip()
            roster_pos = (r[start_col + roster_idx] or '').strip().upper()
            team = (r[start_col + team_idx] or '').strip().upper()

            if not dk_id.isdigit():
                continue

            actual_points = None
            if points_rel_idx is not None:
                pts_raw = (r[start_col + points_rel_idx] or '').strip()
                try:
                    actual_points = float(pts_raw)
                except Exception:
                    actual_points = None

            if actual_points is None:
                continue  # we only want rows with actual slate points

            records.append({
                'DK_ID': dk_id,
                'Name': name,
                'Actual_DK_Points': actual_points,
                'Roster_Position': roster_pos,
                'Team': team,
            })

        return pd.DataFrame(records)

    df = build_from_rows()
    if not df.empty:
        print(f"✅ Extracted {len(df)} player actuals from embedded table")
        return df.drop_duplicates('DK_ID')

    # Fallback: try pandas-based parsing if a clean player table is present
    try:
        pdf = pd.read_csv(file_path)
    except Exception:
        pdf = pd.DataFrame()

    if not pdf.empty:
        cols_lower = {c.lower(): c for c in pdf.columns}
        id_col = cols_lower.get('id') or cols_lower.get('player id') or cols_lower.get('dk_id')
        name_col = cols_lower.get('name')
        # search for a points column
        pts_col = None
        for c in ['Fantasy Points', 'FPTS', 'Points']:
            if c.lower() in cols_lower:
                pts_col = cols_lower[c.lower()]
                break
        if id_col and name_col and pts_col:
            out = pd.DataFrame()
            out['DK_ID'] = pdf[id_col].astype(str).str.extract(r'(\d+)')[0]
            out['Name'] = pdf[name_col].astype(str).str.strip()
            out['Actual_DK_Points'] = pd.to_numeric(pdf[pts_col], errors='coerce')
            if 'roster position' in cols_lower:
                out['Roster_Position'] = pdf[cols_lower['roster position']].astype(str).str.upper()
            if 'teamabbrev' in cols_lower:
                out['Team'] = pdf[cols_lower['teamabbrev']].astype(str).str.upper()
            out = out.dropna(subset=['DK_ID', 'Actual_DK_Points'])
            if not out.empty:
                print(f"✅ Extracted {len(out)} player actuals from flat table")
                return out.drop_duplicates('DK_ID')
        else:
            # Handle standings export format with columns like Player / FPTS but no DK IDs
            player_col = cols_lower.get('player')
            if player_col and pts_col:
                out = pd.DataFrame()
                out['Name'] = pdf[player_col].astype(str).str.strip()
                out['Actual_DK_Points'] = pd.to_numeric(pdf[pts_col], errors='coerce')
                if 'roster position' in cols_lower:
                    out['Roster_Position'] = pdf[cols_lower['roster position']].astype(str).str.upper()
                if 'teamabbrev' in cols_lower:
                    out['Team'] = pdf[cols_lower['teamabbrev']].astype(str).str.upper()
                
                out = out.dropna(subset=['Name', 'Actual_DK_Points'])
                out = out[out['Name'] != '']
                if out.empty:
                    return pd.DataFrame()
                
                # Deduplicate by player name while preserving actual points
                agg_dict = {'Actual_DK_Points': 'first'}
                if 'Roster_Position' in out.columns:
                    agg_dict['Roster_Position'] = 'first'
                if 'Team' in out.columns:
                    agg_dict['Team'] = 'first'
                out = out.groupby('Name', as_index=False).agg(agg_dict)
                out.insert(0, 'DK_ID', '')
                print("⚠️ DK IDs not present in contest file. Output will omit DK_ID values.")
                return out

    print("⚠️ Could not find per-player actual points in this contest file")
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse DK contest CSV for actual per-player DK points")
    # Hard-coded defaults so it runs without arguments
    default_file = "/Users/sineshawmesfintesfaye/Downloads/contest-standings-184084881.csv"
    default_out = "dk_actuals.csv"
    default_entries = "/Users/sineshawmesfintesfaye/Downloads/DKEntries-7.csv"
    parser.add_argument('--file', default=default_file, help='Path to DK contest CSV (results/export)')
    parser.add_argument('--out', default=default_out, help='Output CSV filename')
    parser.add_argument('--dk-entries', default=default_entries, help='DK entries CSV for name -> DK_ID mapping')
    args = parser.parse_args()

    df = parse_contest_csv(args.file)
    if df.empty:
        print("❌ No actuals extracted.")
        sys.exit(1)

    # Attempt to backfill missing DK_ID values using the entries CSV
    if 'Name' in df.columns:
        if 'DK_ID' not in df.columns:
            df['DK_ID'] = ''
        missing_mask = df['DK_ID'].fillna('').str.strip() == ''
        if missing_mask.any():
            mapping_df = load_dk_entries_player_map(args.dk_entries)
            if not mapping_df.empty:
                df['Name_lower'] = df['Name'].astype(str).str.strip().str.lower()
                df = df.merge(mapping_df, on='Name_lower', how='left')
                # Combine original DK_ID (if any) with mapped DK_ID
                if 'DK_ID_x' in df.columns:
                    df['DK_ID'] = df['DK_ID_x'].fillna('').astype(str)
                fill_mask = df['DK_ID'].fillna('').str.strip() == ''
                if 'DK_ID_y' in df.columns:
                    df.loc[fill_mask, 'DK_ID'] = df.loc[fill_mask, 'DK_ID_y'].fillna('')
                df = df.drop(columns=['Name_lower', 'DK_ID_x', 'DK_ID_y'], errors='ignore')
                still_missing = (df['DK_ID'].fillna('').str.strip() == '').sum()
                if still_missing > 0:
                    print(f"⚠️ Unable to find DK_ID for {still_missing} player(s); they will remain blank.")
            else:
                print("⚠️ DK entries mapping unavailable; DK_ID values will remain blank.")

    df.to_csv(args.out, index=False)
    print_header("💾 SAVED")
    print(f"✅ Wrote {len(df)} rows to {args.out}")


if __name__ == '__main__':
    main()
