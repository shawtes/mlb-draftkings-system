#!/usr/bin/env python3
"""
ALL-IN-ONE NBA DATA FETCHER
Run this daily to get fresh NBA data with DraftKings player IDs and positions
"""

import requests
import pandas as pd
from datetime import datetime
import sys
import os
from glob import glob
import argparse
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"
# Fantasy API base (per your plan/docs)
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

# Historical cache location for Markov probabilities
HIST_CACHE_DIR = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_historical_cache"

# Output file
OUTPUT_FILE = "nba_NOV15_READY_2.csv"

# Date - change this daily or use datetime.now()
GAME_DATE = "2025-NOV-29"  # Format: YYYY-MMM-DD

# Hard-coded DK Entries CSV path (dynamic player extraction)
# NOTE: If DK entries don't match the date, set to "" to skip filtering
DK_ENTRIES_DEFAULT = "/Users/sineshawmesfintesfaye/Downloads/DKEntries-19.csv"  # Set to "" to get ALL projections without DK filtering
# DK_ENTRIES_DEFAULT = "/Users/sineshawmesfintesfaye/Downloads/DKEntries-8.csv"  # Oct 29 only

# ============================================================================
# STEP 1: PASTE YOUR DK PLAYER POOL HERE
# ============================================================================
# Get this from DraftKings website -> Copy all players (Name, ID, Position, Salary)
# Format: Name \t ID \t Position \t Salary \t Game Info

DK_PLAYER_POOL_DATA = """Nikola Jokic	40556309	C/UTIL	11500	DEN@MIN 10/27/2025 09:30PM ET
Victor Wembanyama	40556311	C/UTIL	11000	TOR@SAS 10/27/2025 08:00PM ET
Shai Gilgeous-Alexander	40556313	PG/G/UTIL	10500	OKC@DAL 10/27/2025 08:30PM ET
Anthony Davis	40556316	PF/C/F/UTIL	10000	OKC@DAL 10/27/2025 08:30PM ET
Cade Cunningham	40556320	PG/G/UTIL	9800	CLE@DET 10/27/2025 07:00PM ET
Zion Williamson	40556323	PF/F/UTIL	9600	BOS@NOP 10/27/2025 08:00PM ET
Alperen Sengun	40556326	PF/C/F/UTIL	9500	BKN@HOU 10/27/2025 08:00PM ET
Donovan Mitchell	40556330	PG/SG/G/UTIL	9400	CLE@DET 10/27/2025 07:00PM ET
Tyrese Maxey	40556334	PG/G/UTIL	9300	ORL@PHI 10/27/2025 07:00PM ET
Trae Young	40556337	PG/G/UTIL	9200	ATL@CHI 10/27/2025 08:00PM ET
Anthony Edwards	40556340	SG/G/UTIL	9100	DEN@MIN 10/27/2025 09:30PM ET
Kevin Durant	40556343	SG/SF/F/G/UTIL	9000	BKN@HOU 10/27/2025 08:00PM ET
Devin Booker	40556348	PG/G/UTIL	8900	PHX@UTA 10/27/2025 09:00PM ET
Josh Giddey	40556351	PG/SG/G/UTIL	8700	ATL@CHI 10/27/2025 08:00PM ET
Evan Mobley	40556355	PF/C/F/UTIL	8600	CLE@DET 10/27/2025 07:00PM ET
Jaylen Brown	40556359	SG/SF/F/G/UTIL	8500	BOS@NOP 10/27/2025 08:00PM ET
Jalen Johnson	40556364	PF/F/UTIL	8400	ATL@CHI 10/27/2025 08:00PM ET
Paolo Banchero	40556367	PF/F/UTIL	8300	ORL@PHI 10/27/2025 07:00PM ET
Scottie Barnes	40556370	PG/SF/F/G/UTIL	8200	TOR@SAS 10/27/2025 08:00PM ET
Nikola Vucevic	40556375	C/UTIL	8100	ATL@CHI 10/27/2025 08:00PM ET
Jalen Williams	40556380	SF/PF/F/UTIL	8000	OKC@DAL 10/27/2025 08:30PM ET
Derrick White	40556377	PG/G/UTIL	8000	BOS@NOP 10/27/2025 08:00PM ET
Darius Garland	40556384	PG/G/UTIL	7900	CLE@DET 10/27/2025 07:00PM ET
Chet Holmgren	40556387	PF/C/F/UTIL	7900	OKC@DAL 10/27/2025 08:30PM ET
Franz Wagner	40556391	SF/PF/F/UTIL	7800	ORL@PHI 10/27/2025 07:00PM ET
Joel Embiid	40556395	C/UTIL	7700	ORL@PHI 10/27/2025 07:00PM ET
Coby White	40556397	PG/SG/G/UTIL	7600	ATL@CHI 10/27/2025 08:00PM ET
Julius Randle	40556401	PF/F/UTIL	7500	DEN@MIN 10/27/2025 09:30PM ET
Jamal Murray	40556404	PG/G/UTIL	7500	DEN@MIN 10/27/2025 09:30PM ET
Mark Williams	40556410	C/UTIL	7500	PHX@UTA 10/27/2025 09:00PM ET
RJ Barrett	40556407	SG/G/UTIL	7500	TOR@SAS 10/27/2025 08:00PM ET
Amen Thompson	40556412	PG/G/UTIL	7400	BKN@HOU 10/27/2025 08:00PM ET
Paul George	40556415	SF/PF/F/UTIL	7400	ORL@PHI 10/27/2025 07:00PM ET
Cam Thomas	40556419	PG/SG/G/UTIL	7300	BKN@HOU 10/27/2025 08:00PM ET
Jalen Duren	40556438	C/UTIL	7200	CLE@DET 10/27/2025 07:00PM ET
Kristaps Porzingis	40556431	PF/C/F/UTIL	7200	ATL@CHI 10/27/2025 08:00PM ET
Brandon Ingram	40556423	SF/F/UTIL	7200	TOR@SAS 10/27/2025 08:00PM ET
De'Aaron Fox	40556435	PG/G/UTIL	7200	TOR@SAS 10/27/2025 08:00PM ET
Trey Murphy III	40556426	SG/SF/F/G/UTIL	7200	BOS@NOP 10/27/2025 08:00PM ET
Stephon Castle	40556440	PG/SG/G/UTIL	7100	TOR@SAS 10/27/2025 08:00PM ET
Aaron Gordon	40556453	PF/F/UTIL	7000	DEN@MIN 10/27/2025 09:30PM ET
Onyeka Okongwu	40556451	C/UTIL	7000	ATL@CHI 10/27/2025 08:00PM ET
Lauri Markkanen	40556447	SF/PF/F/UTIL	7000	PHX@UTA 10/27/2025 09:00PM ET
Immanuel Quickley	40556444	PG/G/UTIL	7000	TOR@SAS 10/27/2025 08:00PM ET
Desmond Bane	40556456	SG/SF/F/G/UTIL	7000	ORL@PHI 10/27/2025 07:00PM ET
Cooper Flagg	40556461	PG/G/UTIL	6900	OKC@DAL 10/27/2025 08:30PM ET
Michael Porter Jr.	40556468	PF/F/UTIL	6800	BKN@HOU 10/27/2025 08:00PM ET
VJ Edgecombe	40556471	SG/SF/F/G/UTIL	6800	ORL@PHI 10/27/2025 07:00PM ET
Jordan Poole	40556464	PG/SG/G/UTIL	6800	BOS@NOP 10/27/2025 08:00PM ET"""

# ============================================================================
# FUNCTIONS
# ============================================================================

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def parse_dk_player_pool(text_data):
    """Parse DK player pool from tab-separated text"""
    
    print_header("📋 PARSING DK PLAYER POOL")
    
    players = []
    lines = text_data.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        parts = line.split('\t')
        if len(parts) >= 4:
            try:
                name = parts[0].strip()
                dk_id = parts[1].strip()
                roster_position = parts[2].strip()
                salary_str = parts[3].strip().replace(',', '').replace('$', '')
                team_val = ''
                if len(parts) >= 5:
                    game_info = parts[4].strip()
                    # e.g. TOR@SAS ... -> take first team as player's team
                    if '@' in game_info:
                        team_val = game_info.split('@')[0].split()[-1].strip().upper()
                
                # Parse salary
                try:
                    salary = int(salary_str)
                except ValueError:
                    salary = int(float(salary_str))
                
                players.append({
                    'Name': name,
                    'DK_ID': dk_id,
                    'Roster_Position': roster_position,
                    'Salary': salary,
                    'Team': team_val
                })
            except Exception as e:
                print(f"⚠️ Skipped line: {line[:50]}... (Error: {e})")
                continue
    
    df = pd.DataFrame(players)
    print(f"✅ Parsed {len(df)} players from DK pool")
    
    if len(df) == 0:
        print("❌ No players parsed! Check your DK_PLAYER_POOL_DATA format")
        sys.exit(1)
    
    return df


def load_dk_entries_csv(file_path: str) -> pd.DataFrame:
    """Load a DraftKings Entries CSV (or Favorites export) and extract player IDs and names.
    Supports cells like 'Scottie Barnes (40556370)' or pure numeric IDs.
    Returns DataFrame with at least: DK_ID and optional Name.
    """
    print_header(f"📥 LOADING DK ENTRIES CSV: {os.path.basename(file_path)}")
    # 1) Robust parse of embedded player list block using csv.reader (since it's inside data rows)
    import csv
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
    except Exception as e:
        print(f"❌ Failed to read DK entries CSV: {e}")
        rows = []

    target_headers = [
        ['Position', 'Name + ID', 'Name', 'ID', 'Roster Position', 'Salary', 'Game Info', 'TeamAbbrev', 'AvgPointsPerGame'],
        ['Position', 'Name+ID', 'Name', 'ID', 'Roster Position', 'Salary', 'Game Info', 'TeamAbbrev', 'AvgPointsPerGame']
    ]

    start_idx = None
    start_col = None
    matched_header = None
    header_len = 0
    for i, row in enumerate(rows):
        clean = [c.strip() for c in row]
        for header in target_headers:
            for c in range(0, max(0, len(clean) - len(header)) + 1):
                if clean[c:c+len(header)] == header:
                    start_idx = i
                    start_col = c
                    matched_header = header
                    header_len = len(header)
                    break
            if start_idx is not None:
                break
        if start_idx is not None:
            break

    if start_idx is not None:
        pool_records = []
        for j in range(start_idx + 1, len(rows)):
            r = rows[j]
            if len(r) < start_col + header_len:
                continue
            name = (r[start_col + 2] or '').strip()
            dk_id = (r[start_col + 3] or '').strip()
            roster_pos = (r[start_col + 4] or '').strip().upper()
            salary_raw = (r[start_col + 5] or '').strip().replace(',', '').replace('$', '')
            team = (r[start_col + 7] or '').strip().upper()
            if not dk_id.isdigit():
                continue
            try:
                salary = int(float(salary_raw)) if salary_raw else None
            except Exception:
                salary = None
            pool_records.append({'Name': name, 'DK_ID': dk_id, 'Roster_Position': roster_pos, 'Salary': salary, 'Team': team})

        if pool_records:
            pool_df = pd.DataFrame(pool_records).drop_duplicates('DK_ID')
            print(f"✅ Parsed embedded player list: {len(pool_df)} players")
            return pool_df

    # Case B: Entries rows with PG/SG/SF/PF/C/G/F/UTIL columns containing either "Name (ID)" or pure numeric IDs
    try:
        df = pd.read_csv(file_path, engine='python', dtype=str, on_bad_lines='skip')
    except Exception:
        df = pd.DataFrame()
    slot_names = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    slot_cols = [c for c in df.columns if c in slot_names] if not df.empty else []
    if not slot_cols:
        print("⚠️ Not an Entries CSV (no PG/SG/SF/PF/C/G/F/UTIL columns found)")
        return pd.DataFrame()

    id_to_name = {}
    id_set = set()
    for col in slot_cols:
        for val in df[col].astype(str).dropna():
            s = val.strip()
            if not s:
                continue
            # Try Name (ID)
            if '(' in s and ')' in s and s.endswith(')'):
                name_part = s.rsplit('(', 1)[0].strip()
                id_part = s.rsplit('(', 1)[1].replace(')', '').strip()
                if id_part.isdigit():
                    id_set.add(id_part)
                    if name_part:
                        id_to_name[id_part] = name_part
                continue
            # Pure numeric ID
            if s.isdigit():
                id_set.add(s)

    out = pd.DataFrame({'DK_ID': list(id_set)})
    if id_to_name:
        out['Name'] = out['DK_ID'].map(id_to_name).fillna('')
    print(f"✅ Parsed entries CSV: {len(out)} unique players")
    return out


def load_dk_player_pool_csv(file_path: str) -> pd.DataFrame:
    """Load a DraftKings Player List CSV (with Name, ID, Roster Position, Salary).
    Returns DataFrame with: DK_ID, Name, Roster_Position, Salary, Team (if present).
    """
    print_header(f"📥 LOADING DK PLAYER POOL CSV: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Failed to read DK player pool CSV: {e}")
        return pd.DataFrame()

    cols_lower = {c.lower(): c for c in df.columns}
    name_col = cols_lower.get('name') or cols_lower.get('player name')
    id_col = cols_lower.get('id') or cols_lower.get('player id') or cols_lower.get('dk_id')
    pos_col = cols_lower.get('roster position') or cols_lower.get('position')
    salary_col = cols_lower.get('salary')
    team_col = cols_lower.get('team')

    if not id_col:
        print("⚠️ No ID column found in DK player pool CSV")
        return pd.DataFrame()

    out = pd.DataFrame()
    out['DK_ID'] = df[id_col].astype(str).str.extract(r'(\d+)')[0]
    if name_col:
        out['Name'] = df[name_col].astype(str).str.strip()
    if pos_col:
        out['Roster_Position'] = df[pos_col].astype(str).str.upper()
    if salary_col:
        out['Salary'] = pd.to_numeric(df[salary_col], errors='coerce')
    if team_col:
        out['Team'] = df[team_col].astype(str).str.upper()

    out = out.dropna(subset=['DK_ID']).drop_duplicates('DK_ID')
    print(f"✅ Parsed DK player pool CSV: {len(out)} players")
    return out

def fetch_nba_projections(date):
    """Fetch NBA projections from API with retries and SSL fallback"""
    print_header("🏀 FETCHING NBA PROJECTIONS FROM API")

    endpoint = f"/PlayerGameProjectionStatsByDate/{date}"
    url_header = f"{BASE_URL}{endpoint}"
    url_query = f"{BASE_URL}{endpoint}?key={API_KEY}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}

    print(f"📥 Date: {date}")
    print(f"📥 Endpoint: {endpoint}")

    # Try up to 3 attempts: standard, then retry, then SSL-verify disabled as last resort
    attempts = [
        {"verify": True, "label": "standard"},
        {"verify": True, "label": "retry"},
        {"verify": False, "label": "ssl_fallback"},
    ]

    last_error = None
    for attempt in attempts:
        try:
            # Try header auth first
            resp = requests.get(url_header, headers=headers, timeout=30, verify=attempt["verify"]) 
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ Retrieved {len(data)} player projections from API ({attempt['label']})")
                
                # Save the original raw API response (convert to DataFrame)
                if len(data) > 0:
                    raw_filename = f"nba_raw_api_{GAME_DATE.replace('-', '').lower()}.csv"
                    raw_df = pd.DataFrame(data)
                    raw_df.to_csv(raw_filename, index=False)
                    print(f"💾 Saved raw API data to: {raw_filename} ({len(raw_df)} rows, {len(raw_df.columns)} columns)")
                    return raw_df
                else:
                    print("⚠️ API returned empty data")
                    return pd.DataFrame()
            else:
                # Fallback: try query-string key
                resp2 = requests.get(url_query, timeout=30, verify=attempt["verify"]) 
                if resp2.status_code == 200:
                    data = resp2.json()
                    print(f"✅ Retrieved {len(data)} player projections from API via query key ({attempt['label']})")
                    
                    # Save the original raw API response (convert to DataFrame)
                    if len(data) > 0:
                        raw_filename = f"nba_raw_api_{GAME_DATE.replace('-', '').lower()}.csv"
                        raw_df = pd.DataFrame(data)
                        raw_df.to_csv(raw_filename, index=False)
                        print(f"💾 Saved raw API data to: {raw_filename} ({len(raw_df)} rows, {len(raw_df.columns)} columns)")
                        return raw_df
                    else:
                        print("⚠️ API returned empty data")
                        return pd.DataFrame()
                print(f"❌ API Error ({attempt['label']}): Status {resp.status_code} / {resp2.status_code}")
                print((resp.text or '')[:200] or (resp2.text or '')[:200])
        except Exception as e:
            last_error = e
            print(f"❌ Request failed ({attempt['label']}): {e}")

    print("❌ All attempts to fetch projections failed.")
    if last_error:
        print(f"Last error: {last_error}")
    return None


def process_api_projections(api_df):
    """Process API data to extract relevant fields"""
    
    print_header("📊 PROCESSING API PROJECTIONS")
    
    # Check what columns we actually have
    print(f"API columns available: {list(api_df.columns)[:15]}")
    
    processed = pd.DataFrame()
    
    # Basic info - handle different column names
    if 'Name' in api_df.columns:
        processed['Name'] = api_df['Name']
    elif 'PlayerName' in api_df.columns:
        processed['Name'] = api_df['PlayerName']
    else:
        print("⚠️ No Name/PlayerName column found in API data")
        processed['Name'] = 'Unknown'
    
    if 'Team' in api_df.columns:
        processed['Team'] = api_df['Team']
    elif 'TeamAbbr' in api_df.columns:
        processed['Team'] = api_df['TeamAbbr']
    
    # DraftKings fantasy points
    if 'FantasyPointsDraftKings' in api_df.columns:
        processed['Predicted_DK_Points'] = api_df['FantasyPointsDraftKings']
    else:
        # Calculate a projection proxy if DK FP not present
        proxy = pd.Series(0, index=api_df.index, dtype='float64')
        if 'Points' in api_df.columns:
            proxy = proxy.add(api_df['Points'].fillna(0))
        if 'Rebounds' in api_df.columns:
            proxy = proxy.add(api_df['Rebounds'].fillna(0) * 1.25)
        if 'Assists' in api_df.columns:
            proxy = proxy.add(api_df['Assists'].fillna(0) * 1.5)
        if 'Steals' in api_df.columns:
            proxy = proxy.add(api_df['Steals'].fillna(0) * 2)
        if 'BlockedShots' in api_df.columns:
            proxy = proxy.add(api_df['BlockedShots'].fillna(0) * 2)
        if 'Turnovers' in api_df.columns:
            proxy = proxy.sub(api_df['Turnovers'].fillna(0) * 0.5)
        processed['Predicted_DK_Points'] = proxy.round(2)
    
    # Opponent
    if 'Opponent' in api_df.columns:
        processed['Opponent'] = api_df['Opponent']
    elif 'OpponentTeam' in api_df.columns:
        processed['Opponent'] = api_df['OpponentTeam']
    else:
        processed['Opponent'] = ''
    
    # Carry-through additional useful API fields when present
    extra_fields = [
        'Minutes','Started','IsInactive','InjuryStatus','InjuryBodyPart','InjuryNotes','InjuryStartDate',
        'GameID','GlobalGameID','HomeOrAway','DateTime','Day','StadiumID','TeamID','OpponentID',
        'Points','Rebounds','Assists','Steals','BlockedShots','Turnovers','ThreePointersMade',
        'DoubleDouble','TripleDouble','FantasyPoints','FantasyPointsDraftKings'
    ]
    for col in extra_fields:
        if col in api_df.columns and col not in processed.columns:
            processed[col] = api_df[col]
    
    print(f"✅ Processed {len(processed)} projections with {len([c for c in processed.columns if c not in ['Name','Predicted_DK_Points','Team','Opponent']])} extra API columns")
    return processed


def _compute_dk_points_from_stats(df: pd.DataFrame) -> pd.Series:
    """Vectorized DraftKings NBA scoring from box score columns."""
    pts = pd.Series(0.0, index=df.index)
    if 'Points' in df.columns:
        pts = pts.add(df['Points'].fillna(0) * 1.0)
    if 'ThreePointersMade' in df.columns:
        pts = pts.add(df['ThreePointersMade'].fillna(0) * 0.5)
    if 'Rebounds' in df.columns:
        pts = pts.add(df['Rebounds'].fillna(0) * 1.25)
    if 'Assists' in df.columns:
        pts = pts.add(df['Assists'].fillna(0) * 1.5)
    if 'Steals' in df.columns:
        pts = pts.add(df['Steals'].fillna(0) * 2.0)
    if 'BlockedShots' in df.columns:
        pts = pts.add(df['BlockedShots'].fillna(0) * 2.0)
    if 'Turnovers' in df.columns:
        pts = pts.sub(df['Turnovers'].fillna(0) * 0.5)
    # Double-double and triple-double bonuses if present
    if 'DoubleDouble' in df.columns:
        pts = pts.add((df['DoubleDouble'].fillna(0) > 0).astype(float) * 1.5)
    if 'TripleDouble' in df.columns:
        pts = pts.add((df['TripleDouble'].fillna(0) > 0).astype(float) * 3.0)
    return pts.round(2)


def fetch_player_game_stats_by_date(date: str) -> pd.DataFrame:
    """Fetch player game stats by date from SportsData.io."""
    endpoint = f"/PlayerGameStatsByDate/{date}"
    url_header = f"{BASE_URL}{endpoint}"
    url_query = f"{BASE_URL}{endpoint}?key={API_KEY}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    attempts = [
        {"verify": True, "label": "standard"},
        {"verify": True, "label": "retry"},
        {"verify": False, "label": "ssl_fallback"},
    ]
    last_error = None
    for attempt in attempts:
        try:
            resp = requests.get(url_header, headers=headers, timeout=30, verify=attempt["verify"]) 
            if resp.status_code == 200:
                return pd.DataFrame(resp.json())
            resp2 = requests.get(url_query, timeout=30, verify=attempt["verify"]) 
            if resp2.status_code == 200:
                return pd.DataFrame(resp2.json())
        except Exception as e:
            last_error = e
            continue
    if last_error:
        print(f"❌ Historical request failed for {date}: {last_error}")
    return pd.DataFrame()


def _normalize_history_day(df_day: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Normalize one day of history to required columns for Markov cache."""
    if df_day is None or df_day.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    # Name
    if 'Name' in df_day.columns:
        out['Name'] = df_day['Name']
    elif 'PlayerName' in df_day.columns:
        out['Name'] = df_day['PlayerName']
    else:
        out['Name'] = ''
    # Team
    if 'Team' in df_day.columns:
        out['Team'] = df_day['Team']
    elif 'TeamAbbr' in df_day.columns:
        out['Team'] = df_day['TeamAbbr']
    else:
        out['Team'] = ''
    # ID
    for c in ['ID', 'PlayerID', 'DK_ID', 'GlobalPlayerID']:
        if c in df_day.columns:
            out[c] = df_day[c]
            break
    # Date
    out['Date'] = date_str
    # Fantasy points
    if 'FantasyPointsDraftKings' in df_day.columns:
        out['FantasyPointsDraftKings'] = pd.to_numeric(df_day['FantasyPointsDraftKings'], errors='coerce')
    elif 'FantasyPoints' in df_day.columns:
        out['FantasyPointsDraftKings'] = pd.to_numeric(df_day['FantasyPoints'], errors='coerce')
    else:
        out['FantasyPointsDraftKings'] = _compute_dk_points_from_stats(df_day)
    return out


def ensure_historical_cache(days: int = 1095, end_date: str = None) -> Optional[str]:
    """Build a 3-year (default) historical cache if missing. Returns cache path."""
    os.makedirs(HIST_CACHE_DIR, exist_ok=True)
    pkl_path = os.path.join(HIST_CACHE_DIR, 'historical_3years.pkl')
    csv_path = os.path.join(HIST_CACHE_DIR, 'historical_3years.csv')
    if os.path.exists(pkl_path) or os.path.exists(csv_path):
        print(f"🗂  Historical cache already exists at {HIST_CACHE_DIR}")
        return pkl_path if os.path.exists(pkl_path) else csv_path

    # Determine date range
    if end_date is None:
        end_dt = datetime.now()
    else:
        try:
            end_dt = datetime.strptime(end_date.replace('-', ' '), '%Y %b %d')
        except Exception:
            end_dt = datetime.now()
    dates = [(end_dt - pd.Timedelta(days=i)).strftime('%Y-%b-%d').upper() for i in range(days)]

    print_header("📚 BUILDING 3-YEAR HISTORICAL CACHE (DK Fantasy Points)")
    print(f"Total dates to fetch: {len(dates)} (this may take a while, API-rate limited)")

    all_days = []
    fetched = 0
    for i, d in enumerate(dates):
        try:
            day_df = fetch_player_game_stats_by_date(d)
            if not day_df.empty:
                norm = _normalize_history_day(day_df, d)
                if not norm.empty:
                    all_days.append(norm)
            fetched += 1
            if fetched % 25 == 0:
                print(f"   Progress: {fetched}/{len(dates)} days...")
        except Exception as e:
            print(f"⚠️  Error fetching {d}: {e}")
        # Be kind to API
        try:
            import time
            time.sleep(0.12)
        except Exception:
            pass

    if not all_days:
        print("❌ No historical data fetched. Cache not created.")
        return None

    hist_df = pd.concat(all_days, ignore_index=True)
    try:
        hist_df.to_pickle(pkl_path)
        hist_df.to_csv(csv_path, index=False)
        # Save basic metadata
        meta = {
            'games_analyzed': int(len(hist_df)),
            'players': int(hist_df['Name'].nunique() if 'Name' in hist_df.columns else 0),
            'dates': int(hist_df['Date'].nunique() if 'Date' in hist_df.columns else 0),
            'generated_at': datetime.now().isoformat(timespec='seconds')
        }
        with open(os.path.join(HIST_CACHE_DIR, 'config.json'), 'w') as f:
            import json as _json
            _json.dump(meta, f, indent=2)
        print(f"✅ Historical cache saved: {pkl_path} and {csv_path}")
        print(f"   Summary: {meta}")
        return pkl_path
    except Exception as e:
        print(f"⚠️  Failed to save historical cache: {e}")
        return None


def fetch_injuries() -> pd.DataFrame:
    """Fetch current NBA injuries from SportsDataIO and return as DataFrame."""
    print_header("🩺 FETCHING INJURIES FROM API")
    url = "https://api.sportsdata.io/v3/nba/scores/json/Injuries"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    attempts = [
        {"verify": True, "label": "standard"},
        {"verify": True, "label": "retry"},
        {"verify": False, "label": "ssl_fallback"},
    ]
    last_error = None
    for attempt in attempts:
        try:
            resp = requests.get(url, headers=headers, timeout=30, verify=attempt["verify"])
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                print(f"✅ Retrieved {len(df)} injury records ({attempt['label']})")
                return df
            else:
                print(f"❌ Injury API Error ({attempt['label']}): Status {resp.status_code}")
                print(resp.text[:200])
        except Exception as e:
            last_error = e
            print(f"❌ Injury request failed ({attempt['label']}): {e}")
    print("⚠️ Injury API unavailable; proceeding without injuries")
    if last_error:
        print(f"Last injury error: {last_error}")
    return pd.DataFrame()


def merge_injuries(players_df: pd.DataFrame, injuries_df: pd.DataFrame) -> pd.DataFrame:
    """Merge injury info into players DataFrame by Name + Team when available."""
    if players_df.empty or injuries_df.empty:
        return players_df
    df = players_df.copy()
    inj = injuries_df.copy()
    df['Name_lower'] = df['Name'].astype(str).str.strip().str.lower()
    df['Team_upper'] = df.get('Team', '').astype(str).str.upper()
    inj['Name_lower'] = inj.get('Name', inj.get('Player', '')).astype(str).str.strip().str.lower()
    inj['Team_upper'] = inj.get('Team', '').astype(str).str.upper()
    keep_cols = [c for c in ['InjuryStatus', 'InjuryBodyPart', 'InjuryNotes', 'IsInactive', 'InjuryStartDate'] if c in inj.columns]
    inj_small = inj[['Name_lower', 'Team_upper'] + keep_cols].drop_duplicates(['Name_lower', 'Team_upper'])
    merged = df.merge(inj_small, on=['Name_lower', 'Team_upper'], how='left')
    merged = merged.drop(columns=['Name_lower', 'Team_upper'], errors='ignore')
    if keep_cols:
        print(f"🩺 Injuries merged: {merged[keep_cols].notna().any(axis=1).sum()} players with injury info")
    return merged

def normalize_player_name(name):
    """Normalize player names for better matching - handles suffixes like Jr., III, IV"""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    # Remove common suffixes
    name = re.sub(r'\s+(jr\.?|sr\.?|iii|iv|ii)$', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def merge_data(api_df, dk_df):
    """Merge API projections with DK player data"""
    
    print_header("🔗 MERGING API + DK DATA")
    
    # Prefer name-based match; if names missing, require DK_ID-to-Name mapping from pool
    if 'Name' in dk_df.columns and dk_df['Name'].fillna('').str.strip().ne('').sum() > 0:
        # Normalize names for better matching (handles Jr., III, IV, etc.)
        api_df['Name_normalized'] = api_df['Name'].apply(normalize_player_name)
        dk_df['Name_normalized'] = dk_df['Name'].apply(normalize_player_name)
        
        merged = api_df.merge(
            dk_df[['Name_normalized', 'DK_ID', 'Roster_Position', 'Salary', 'Name']],
            on='Name_normalized',
            how='inner',
            suffixes=('', '_DK')
        )
        
        # Drop helper columns
        merged = merged.drop(columns=['Name_normalized'], errors='ignore')
        
        # Show matching info
        if len(merged) < len(dk_df):
            missing = len(dk_df) - len(merged)
            print(f"⚠️  {missing} DK players not found in API (may not be playing today)")
    else:
        print("⚠️ DK entries lack names; cannot match projections without a DK player pool mapping.")
        return pd.DataFrame()
    
    # Extract simple position from Roster_Position for backward compatibility
    merged['Position'] = merged['Roster_Position'].str.extract(r'(PG|SG|SF|PF|C)')[0]
    
    print(f"✅ Matched {len(merged)} players")
    print(f"📊 Teams included: {sorted(merged['Team'].unique().tolist())}")
    
    if len(merged) == 0:
        print("\n❌ WARNING: No players matched!")
        print("   Check that player names in API match DK pool exactly")
        sys.exit(1)
    
    return merged


def save_output(df, filename):
    """Save final dataset"""
    
    print_header("💾 SAVING OUTPUT")
    
    # Reorder columns: core first, then keep any extra API fields
    core = ['Name', 'DK_ID', 'Position', 'Roster_Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Opponent']
    front = [c for c in core if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    
    # Add columns required by parlay generator
    if 'player_id' not in df.columns and 'Name' in df.columns and 'Team' in df.columns:
        df['player_id'] = df['Name'].astype(str) + '_' + df['Team'].astype(str)
    
    if 'player_name_proj' not in df.columns and 'Name' in df.columns:
        df['player_name_proj'] = df['Name']
    
    if 'team_proj' not in df.columns and 'Team' in df.columns:
        df['team_proj'] = df['Team']
    
    if 'position_proj' not in df.columns:
        if 'Position' in df.columns:
            df['position_proj'] = df['Position']
        elif 'Roster_Position' in df.columns:
            df['position_proj'] = df['Roster_Position']
        else:
            df['position_proj'] = 'G'
    
    if 'projected_points' not in df.columns:
        df['projected_points'] = df['Predicted_DK_Points']
    
    if 'projected_dk_points' not in df.columns:
        df['projected_dk_points'] = df['Predicted_DK_Points']
    
    # Save
    df.to_csv(filename, index=False)
    
    print(f"✅ Saved to: {filename} (parlay-compatible)")
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total players: {len(df)}")
    if 'Team' in df.columns:
        print(f"   Teams: {len(df['Team'].unique())}")
    if 'Salary' in df.columns:
        print(f"   Salary range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
        print(f"   Avg salary: ${df['Salary'].mean():,.0f}")
    if 'Predicted_DK_Points' in df.columns:
        print(f"   Avg projected points: {df['Predicted_DK_Points'].mean():.2f}")
    
    if 'Predicted_DK_Points' in df.columns:
        print(f"\n📋 Top 10 projected scorers:")
        cols = ['Name', 'Team', 'Predicted_DK_Points']
        if 'Salary' in df.columns:
            cols = ['Name', 'Team', 'Salary', 'Predicted_DK_Points']
        top10 = df.nlargest(10, 'Predicted_DK_Points')[[c for c in cols if c in df.columns]]
        print(top10.to_string(index=False))


def load_cached_api_projections():
    """Try to load the latest cached API CSV in the current directory"""
    print_header("🗂  LOOKING FOR CACHED API DATA")
    candidates = sorted(glob('nba_all_players*.csv') + glob('nba_all_players_*/*.csv'), reverse=True)
    for path in candidates:
        try:
            df = pd.read_csv(path)
            if not df.empty and 'Name' in df.columns:
                print(f"✅ Loaded cached API data: {path} ({len(df)} rows)")
                return df
        except Exception:
            continue
    print("⚠️ No cached API data found")
    return None


def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(description="Fetch NBA projections and merge with DK entries/player pool")
    parser.add_argument('--date', default=GAME_DATE, help='Date like 2025-OCT-27')
    parser.add_argument('--out', default=OUTPUT_FILE, help='Output CSV file')
    parser.add_argument('--dk-entries', default=DK_ENTRIES_DEFAULT, help='Path to DK Entries CSV to parse players from')
    parser.add_argument('--dk-player-pool', help='Path to DK Player Pool CSV for Name/ID/Position/Salary')
    parser.add_argument('--build-history', action='store_true', default=False, help='Force build/update historical 3-year cache now')
    parser.add_argument('--history-days', type=int, default=1095, help='How many days back to fetch when building history')
    parser.add_argument('--history-end-date', default=None, help='End date for history (defaults to today)')
    args = parser.parse_args()

    print(f"\n🏀 DAILY NBA DATA FETCHER")
    print(f"{'='*70}")
    print(f"📅 Date: {args.date}")
    print(f"💾 Output: {args.out}")
    print(f"{'='*70}")
    
    # Step 1: Build DK player set dynamically
    dk_df = pd.DataFrame()
    use_dk_filtering = False
    
    if args.dk_entries and os.path.exists(args.dk_entries):
        entries_df = load_dk_entries_csv(args.dk_entries)
        dk_df = entries_df
        use_dk_filtering = True
    elif args.dk_entries:
        print(f"⚠️ DK entries file not found: {args.dk_entries}")
        print(f"   Fetching ALL projections without DK filtering\n")
    else:
        print(f"ℹ️  No DK entries specified - fetching ALL projections\n")

    # If we have a DK player pool CSV, enrich entries with names/positions/salary
    if args.dk_player_pool and os.path.exists(args.dk_player_pool):
        pool_df = load_dk_player_pool_csv(args.dk_player_pool)
        if not pool_df.empty:
            dk_df = dk_df.merge(pool_df, on='DK_ID', how='left', suffixes=('', '_pool'))
            # Prefer pool values where missing
            if 'Name_pool' in dk_df.columns:
                dk_df['Name'] = dk_df['Name'].where(dk_df['Name'].astype(str).str.strip().ne(''), dk_df['Name_pool'])
            if 'Roster_Position_pool' in dk_df.columns and 'Roster_Position' not in dk_df.columns:
                dk_df['Roster_Position'] = dk_df['Roster_Position_pool']
            if 'Salary_pool' in dk_df.columns and 'Salary' not in dk_df.columns:
                dk_df['Salary'] = dk_df['Salary_pool']
            if 'Team_pool' in dk_df.columns and 'Team' not in dk_df.columns:
                dk_df['Team'] = dk_df['Team_pool']
            # Drop helper cols
            dk_df = dk_df[[c for c in dk_df.columns if not c.endswith('_pool')]]
    
    # Step 2: Fetch API data (with fallback to cached or DK-only)
    api_df = fetch_nba_projections(args.date)
    if api_df is None:
        # Try cached
        api_df = load_cached_api_projections()
    
    if api_df is None:
        print("\n⚠️ Using DK pool only (no API projections). Estimating projections from salary")
        # Build a minimal API dataframe from DK pool
        api_df = pd.DataFrame({
            'Name': dk_df.get('Name', pd.Series(dtype=str)),
            'Team': dk_df.get('Team', ''),
            'FantasyPointsDraftKings': (dk_df.get('Salary', pd.Series(0)) / 200.0).round(2),
            'Opponent': ''
        })
    
    # Step 3: Process API data
    processed_df = process_api_projections(api_df)
    
    # Step 4: Save FULL PARLAY version (all projections, no DK filtering)
    parlay_file = args.out.replace('_1READY.csv', '_PARLAY_READY.csv')
    if use_dk_filtering:
        print_header("💾 SAVING PARLAY VERSION (ALL PROJECTIONS)")
        parlay_df = processed_df.copy()
        if 'Predicted_DK_Points' in parlay_df.columns:
            parlay_df = parlay_df[parlay_df['Predicted_DK_Points'] > 0]
        save_output(parlay_df, parlay_file)
        print(f"✅ Parlay file: {parlay_file} ({len(parlay_df)} players)")
    
    # Step 5: Merge for DFS OPTIMIZER (filtered by DK entries)
    final_df = merge_data(processed_df, dk_df)
    
    # Step 6: Injuries merge and Save
    if final_df.empty:
        print("⚠️ No players matched DK entries. Skipping DFS optimizer file.")
        if not use_dk_filtering:
            # If no DK filtering, save all projections as both files
            save_output(processed_df, args.out)
    else:
        injuries_df = fetch_injuries()
        final_df = merge_injuries(final_df, injuries_df)
        save_output(final_df, args.out)
        print(f"✅ DFS optimizer file: {args.out} ({len(final_df)} players)")

    # Step 7: Ensure 3-year historical cache for Markov probabilities
    try:
        need_build = args.build_history
        os.makedirs(HIST_CACHE_DIR, exist_ok=True)
        pkl_path = os.path.join(HIST_CACHE_DIR, 'historical_3years.pkl')
        csv_path = os.path.join(HIST_CACHE_DIR, 'historical_3years.csv')
        if not os.path.exists(pkl_path) and not os.path.exists(csv_path):
            need_build = True
        if need_build:
            ensure_historical_cache(days=max(1, int(args.history_days)), end_date=args.history_end_date or None)
        else:
            print(f"🗂  Historical cache already present; skipping build")
    except Exception as e:
        print(f"⚠️  Skipped building historical cache: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS! Data ready for optimization")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
