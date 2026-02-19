#!/usr/bin/env python3
"""
NBA Optimizer Backtesting System
=================================
Fetches projection + actual data from SportsData.io for ~15 game days,
runs the PuLP ILP optimizer with many different setting combinations,
scores generated lineups against actual results, computes hindsight-optimal
lineups for comparison, and ranks configurations by actual performance.

Usage:
    python3 backtest_optimizer.py                    # Full run (all dates, all configs)
    python3 backtest_optimizer.py --dates 1          # Quick test: 1 date only
    python3 backtest_optimizer.py --dates 3 --configs 4  # 3 dates, 4 configs
    python3 backtest_optimizer.py --no-cache          # Force re-fetch from API

References:
    - Hunter, Vielma, Zaman (2016) "Picking Winners in DFS Using IP" (arXiv:1604.01455)
    - Haugh & Singal (2021) "How to Play Fantasy Sports Strategically" (Management Science)
    - Mlcoch & Hubacek (2024) "Competing in DFS Using Generative Models" (ITOR 31(3))
"""

import os
import sys
import json
import time
import argparse
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from itertools import product

import requests
import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:
    print("ERROR: PuLP required. Install with: pip install pulp")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.environ.get("SPORTSDATA_API_KEY", "d62d0ae315504e53a232ff7d1c3bea33")
BASE_URL = "https://api.sportsdata.io/api/nba/fantasy/json"

PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"
RESULTS_DIR = PROJECT_ROOT / "data"
ADAPTER_PATH = PROJECT_ROOT / "web_optimizer" / "server" / "makrov_cli_adapter.py"

# DK NBA scoring constants
DK_PTS = 1.0
DK_3PM = 0.5
DK_REB = 1.25
DK_AST = 1.5
DK_STL = 2.0
DK_BLK = 2.0
DK_TOV = -0.5
DK_DD = 1.5
DK_TD = 3.0

# 15 backtest dates (diverse slate sizes, Oct 2025 - Feb 2026)
BACKTEST_DATES = [
    "2025-NOV-04", "2025-NOV-08", "2025-NOV-15", "2025-NOV-22",
    "2025-NOV-29", "2025-DEC-06", "2025-DEC-20", "2025-DEC-25",
    "2026-JAN-03", "2026-JAN-10", "2026-JAN-17", "2026-JAN-24",
    "2026-JAN-31", "2026-FEB-07", "2026-FEB-12",
]

# DK NBA roster: 8 players
ROSTER_SIZE = 8
SALARY_CAP = 50000
GUARD_POSITIONS = ["PG", "SG"]
FORWARD_POSITIONS = ["SF", "PF"]
ALL_POSITIONS = ["PG", "SG", "SF", "PF", "C"]

# Multi-position eligibility for DK NBA
POSITION_ELIGIBILITY = {
    "PG": ["PG", "G", "UTIL"],
    "SG": ["SG", "G", "UTIL"],
    "SF": ["SF", "F", "UTIL"],
    "PF": ["PF", "F", "UTIL"],
    "C":  ["C", "UTIL"],
}

# Cash line threshold (conservative — most 50/50s pay ~240-260 DK)
CASH_LINE = 240


# ============================================================================
# PARAMETER GRID
# ============================================================================

PARAM_GRID = {
    "num_lineups":   [20],
    "max_exposure":  [30, 50, 70, 100],
    "stack_enabled": [False, True],
    "stack_sizes":   [[2], [3], [2, 3]],   # only used when stack_enabled=True
    "min_unique":    [2, 3],
    "min_salary":    [49000],
}


def build_configs():
    """Generate all valid parameter configurations."""
    configs = []
    for nl, me, se, ss, mu, ms in product(
        PARAM_GRID["num_lineups"],
        PARAM_GRID["max_exposure"],
        PARAM_GRID["stack_enabled"],
        PARAM_GRID["stack_sizes"],
        PARAM_GRID["min_unique"],
        PARAM_GRID["min_salary"],
    ):
        if not se and ss != [2]:
            # Skip stack_sizes variations when stacking is disabled (avoid duplicates)
            continue
        config = {
            "num_lineups": nl,
            "max_exposure": me,
            "stack_enabled": se,
            "stack_sizes": ss if se else [],
            "min_unique": mu,
            "min_salary": ms,
        }
        config["label"] = (
            f"L{nl}_exp{me}"
            f"{'_stack' + ''.join(map(str, ss)) if se else '_nostack'}"
            f"_uniq{mu}_sal{ms}"
        )
        configs.append(config)
    return configs


# ============================================================================
# DATA FETCHING + CACHING
# ============================================================================

def fetch_json(endpoint, date_str, cache_name, use_cache=True):
    """Fetch JSON from SportsData.io with file-based caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{date_str}_{cache_name}.json"

    if use_cache and cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)

    url = f"{BASE_URL}/{endpoint}/{date_str}?key={API_KEY}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  API ERROR ({endpoint}): {e}")
        return None

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data


def compute_dk_points(row):
    """Compute DraftKings NBA fantasy points from stat components."""
    def _val(key):
        v = row.get(key, 0)
        return float(v) if v is not None else 0.0

    pts = (
        _val("Points") * DK_PTS
        + _val("ThreePointersMade") * DK_3PM
        + _val("Rebounds") * DK_REB
        + _val("Assists") * DK_AST
        + _val("Steals") * DK_STL
        + _val("BlockedShots") * DK_BLK
        + _val("Turnovers") * DK_TOV
    )
    # Double-double / triple-double bonuses
    # API uses "DoubleDoubles" and "TripleDoubles" (plural)
    dd = _val("DoubleDoubles") if "DoubleDoubles" in row else _val("DoubleDouble")
    td = _val("TripleDoubles") if "TripleDoubles" in row else _val("TripleDouble")
    if dd > 0:
        pts += DK_DD
    if td > 0:
        pts += DK_TD
    # Fallback: compute from stat categories if API fields missing
    if dd == 0 and td == 0:
        stats_10 = sum(1 for s in [
            _val("Points"), _val("Rebounds"), _val("Assists"),
            _val("Steals"), _val("BlockedShots")
        ] if s >= 10)
        if stats_10 >= 2:
            pts += DK_DD
        if stats_10 >= 3:
            pts += DK_TD
    return round(pts, 2)


def fetch_slate(date_str, use_cache=True):
    """
    Fetch projections + actuals + DK salaries for a given date.
    Uses 3 endpoints:
      1. PlayerGameProjectionStatsByDate -> projected stats
      2. PlayerGameStatsByDate -> actual stats
      3. DfsSlatesByDate -> DK salary, multi-position eligibility
    Returns DataFrame with: Name, Position, Team, Salary, Proj_DK, Actual_DK
    """
    proj_data = fetch_json("PlayerGameProjectionStatsByDate", date_str, "proj", use_cache)
    act_data = fetch_json("PlayerGameStatsByDate", date_str, "act", use_cache)
    slate_data = fetch_json("DfsSlatesByDate", date_str, "slates", use_cache)

    if not proj_data or not act_data:
        return None

    # --- Step 1: Build DK salary + position lookup from slates ---
    dk_salary = {}    # PlayerID -> salary
    dk_position = {}  # PlayerID -> base position (PG/SG/SF/PF/C)
    dk_name = {}      # PlayerID -> operator name

    if slate_data:
        # Find the DraftKings "Main" Classic slate (largest Classic slate)
        dk_slates = [
            s for s in slate_data
            if s.get("Operator") == "DraftKings"
            and s.get("OperatorGameType") == "Classic"
        ]
        # Prefer "Main" slate, fall back to largest
        main_slates = [s for s in dk_slates if s.get("OperatorName") == "Main"]
        target_slate = None
        if main_slates:
            target_slate = max(main_slates, key=lambda s: len(s.get("DfsSlatePlayers", [])))
        elif dk_slates:
            target_slate = max(dk_slates, key=lambda s: len(s.get("DfsSlatePlayers", [])))

        if target_slate:
            for sp in target_slate.get("DfsSlatePlayers", []):
                pid = sp.get("PlayerID")
                sal = sp.get("OperatorSalary", 0)
                pos = sp.get("OperatorPosition", "")
                name = sp.get("OperatorPlayerName", "")
                if pid and sal and sal > 0:
                    dk_salary[pid] = int(sal)
                    dk_position[pid] = pos
                    dk_name[pid] = name
            print(f"  DK Main slate: {len(dk_salary)} players with salaries")

    if not dk_salary:
        print(f"  No DK salary data found for {date_str}")
        return None

    # --- Step 2: Build projections for DK-eligible players ---
    # Index projections by PlayerID
    proj_by_id = {}
    for p in proj_data:
        pid = p.get("PlayerID")
        if pid:
            proj_by_id[pid] = p

    # --- Step 3: Build actuals lookup ---
    act_by_id = {}
    act_by_name = {}
    for a in act_data:
        actual_dk = compute_dk_points(a)
        pid = a.get("PlayerID")
        name = a.get("Name", "")
        if pid:
            act_by_id[pid] = actual_dk
        if name:
            act_by_name[name] = actual_dk

    # --- Step 4: Merge into final rows ---
    rows = []
    for pid, salary in dk_salary.items():
        proj = proj_by_id.get(pid)
        if not proj:
            continue

        pos = dk_position.get(pid, proj.get("Position", ""))
        # Normalize position to base DK position
        if "/" in pos:
            pos = pos.split("/")[0]
        if pos not in ALL_POSITIONS:
            continue

        name = dk_name.get(pid, proj.get("Name", ""))
        if not name:
            continue

        proj_dk = compute_dk_points(proj)
        if proj_dk <= 0:
            continue

        actual_dk = act_by_id.get(pid, act_by_name.get(name, 0.0))

        rows.append({
            "Name": name,
            "Position": pos,
            "Team": proj.get("Team", "UNK"),
            "Salary": salary,
            "Proj_DK": proj_dk,
            "Actual_DK": actual_dk,
            "PlayerID": pid,
        })

    df = pd.DataFrame(rows)
    if len(df) < ROSTER_SIZE:
        print(f"  Only {len(df)} valid players — skipping date")
        return None

    # Deduplicate by name (keep higher salary entry)
    df = df.sort_values("Salary", ascending=False).drop_duplicates(subset="Name", keep="first")

    # Count unique games via teams
    teams = df["Team"].nunique()
    games_approx = max(1, teams // 2)

    # Validate: are projections real (not actuals)?
    avg_val = (df["Proj_DK"] / df["Salary"] * 1000).mean()
    actual_val = (df["Actual_DK"] / df["Salary"] * 1000).mean()

    print(f"  {len(df)} players, ~{games_approx} games, "
          f"sal range ${df['Salary'].min()}-${df['Salary'].max()}, "
          f"proj range {df['Proj_DK'].min():.1f}-{df['Proj_DK'].max():.1f}, "
          f"proj_val={avg_val:.2f}/K, actual_val={actual_val:.2f}/K")

    return df


# ============================================================================
# OPTIMIZER CALL (via makrov_cli_adapter.py subprocess)
# ============================================================================

def build_team_selections(df, config):
    """Build teamSelections dict for the optimizer."""
    all_teams = sorted(df["Team"].unique().tolist())
    selections = {"all": all_teams}
    if config.get("stack_enabled"):
        for sz in config.get("stack_sizes", []):
            # Only include teams with enough players for the stack size
            valid = [t for t in all_teams if len(df[df["Team"] == t]) >= sz]
            selections[str(sz)] = valid
    return selections


def run_optimizer(df, config):
    """Call makrov_cli_adapter.py via subprocess with given config."""
    players = []
    for _, row in df.iterrows():
        players.append({
            "name": row["Name"],
            "position": row["Position"],
            "salary": int(row["Salary"]),
            "projection": float(row["Proj_DK"]),
            "team": row["Team"],
        })

    input_data = {
        "players": players,
        "numLineups": config["num_lineups"],
        "minSalary": config.get("min_salary", 49000),
        "maxSalary": SALARY_CAP,
        "maxExposure": config["max_exposure"],
        "stackSettings": {
            "enabled": config.get("stack_enabled", False),
            "requestedStackSizes": config.get("stack_sizes", []),
            "minPlayersPerTeam": min(config.get("stack_sizes", [2])) if config.get("stack_sizes") else 0,
            "maxPlayersPerTeam": max(config.get("stack_sizes", [2])) if config.get("stack_sizes") else 0,
        },
        "teamSelections": build_team_selections(df, config),
        "teamExposures": {},
        "uniquePlayers": config.get("min_unique", 3),
    }

    try:
        result = subprocess.run(
            [sys.executable, str(ADAPTER_PATH), json.dumps(input_data)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "lineups": []}

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[-200:]
        return {"error": f"exit {result.returncode}: {stderr_snippet}", "lineups": []}

    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid JSON output", "lineups": []}

    return output


# ============================================================================
# SCORING
# ============================================================================

def score_lineups(lineups, actuals_lookup):
    """Score each lineup using actual DK points."""
    results = []
    for lineup in lineups:
        player_names = [p.get("name", p.get("Name", "")) for p in lineup.get("players", [])]
        actual_total = sum(actuals_lookup.get(n, 0.0) for n in player_names)
        projected_total = lineup.get("totalProjection", 0.0)
        results.append({
            "projected": projected_total,
            "actual": actual_total,
            "diff": actual_total - projected_total,
            "players": player_names,
            "salary": lineup.get("totalSalary", 0),
        })
    return results


# ============================================================================
# HINDSIGHT OPTIMAL (PuLP ILP with actual points as objective)
# ============================================================================

def compute_optimal(df):
    """
    Solve ILP with actual DK points as objective -> theoretical maximum.
    Subject to: salary <= 50000, 8 players, DK NBA position constraints.
    """
    df_opt = df[df["Actual_DK"] > 0].copy().reset_index(drop=True)
    if len(df_opt) < ROSTER_SIZE:
        return 0.0, []

    prob = pulp.LpProblem("DFS_Optimal", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(len(df_opt))}

    # Objective: maximize actual DK points
    prob += pulp.lpSum(df_opt.iloc[i]["Actual_DK"] * x[i] for i in range(len(df_opt)))

    # Salary cap
    prob += pulp.lpSum(df_opt.iloc[i]["Salary"] * x[i] for i in range(len(df_opt))) <= SALARY_CAP

    # Exactly 8 players
    prob += pulp.lpSum(x[i] for i in range(len(df_opt))) == ROSTER_SIZE

    # Position constraints (DK NBA: PG, SG, SF, PF, C, G, F, UTIL)
    for pos in ALL_POSITIONS:
        eligible = [i for i in range(len(df_opt)) if df_opt.iloc[i]["Position"] == pos]
        prob += pulp.lpSum(x[i] for i in eligible) >= 1

    # G slot: at least 3 guards total (PG + SG + G)
    guard_eligible = [i for i in range(len(df_opt)) if df_opt.iloc[i]["Position"] in GUARD_POSITIONS]
    prob += pulp.lpSum(x[i] for i in guard_eligible) >= 3

    # F slot: at least 3 forwards total (SF + PF + F)
    forward_eligible = [i for i in range(len(df_opt)) if df_opt.iloc[i]["Position"] in FORWARD_POSITIONS]
    prob += pulp.lpSum(x[i] for i in forward_eligible) >= 3

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != pulp.constants.LpStatusOptimal:
        return 0.0, []

    selected = [i for i in range(len(df_opt)) if x[i].varValue and x[i].varValue > 0.5]
    optimal_score = sum(df_opt.iloc[i]["Actual_DK"] for i in selected)
    optimal_names = [df_opt.iloc[i]["Name"] for i in selected]

    return round(optimal_score, 2), optimal_names


# ============================================================================
# MAIN BACKTESTING LOOP
# ============================================================================

def run_backtest(args):
    """Main backtesting entry point."""
    print("=" * 80)
    print("NBA OPTIMIZER BACKTESTING SYSTEM")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Build config grid
    all_configs = build_configs()
    if args.configs and args.configs < len(all_configs):
        all_configs = all_configs[:args.configs]
    print(f"\nConfigurations: {len(all_configs)}")

    # Select dates
    dates = BACKTEST_DATES[:args.dates] if args.dates else BACKTEST_DATES
    print(f"Dates: {len(dates)}")
    print(f"Total optimizer runs: {len(all_configs) * len(dates)}")
    print()

    use_cache = not args.no_cache
    all_results = []
    date_optionals = {}

    for date_idx, date_str in enumerate(dates):
        print(f"[{date_idx + 1}/{len(dates)}] {date_str}")

        # Fetch data
        df = fetch_slate(date_str, use_cache=use_cache)
        if df is None:
            print(f"  SKIPPED (no data)")
            continue

        # Build actuals lookup
        actuals_lookup = dict(zip(df["Name"], df["Actual_DK"]))

        # Compute hindsight optimal
        optimal_score, optimal_names = compute_optimal(df)
        date_optionals[date_str] = optimal_score
        print(f"  Hindsight optimal: {optimal_score:.2f} DK pts")
        if optimal_names:
            print(f"    Players: {', '.join(optimal_names[:4])}...")

        # Run each config
        for cfg_idx, config in enumerate(all_configs):
            t0 = time.time()
            output = run_optimizer(df, config)
            elapsed = time.time() - t0

            lineups = output.get("lineups", [])
            error = output.get("error")

            if error or not lineups:
                all_results.append({
                    "date": date_str,
                    "config": config["label"],
                    "error": error or "no lineups",
                    "num_lineups": 0,
                    "best_actual": 0, "avg_actual": 0, "worst_actual": 0,
                    "pct_optimal": 0, "cash_hit_rate": 0,
                    "gpp_ceiling": 0, "consistency": 0, "proj_accuracy": 0,
                    "optimal_score": optimal_score,
                    "elapsed_sec": round(elapsed, 1),
                    **config,
                })
                continue

            # Score lineups against actuals
            scored = score_lineups(lineups, actuals_lookup)
            actuals = [s["actual"] for s in scored]
            projs = [s["projected"] for s in scored]

            best_actual = max(actuals)
            avg_actual = np.mean(actuals)
            worst_actual = min(actuals)
            pct_optimal = (best_actual / optimal_score * 100) if optimal_score > 0 else 0
            cash_hit_rate = sum(1 for a in actuals if a >= CASH_LINE) / len(actuals) * 100
            gpp_ceiling = np.percentile(actuals, 90)
            consistency = np.std(actuals)

            # Projection accuracy (correlation between projected and actual totals)
            if len(actuals) > 1 and np.std(projs) > 0 and np.std(actuals) > 0:
                proj_accuracy = float(np.corrcoef(projs, actuals)[0, 1])
            else:
                proj_accuracy = 0.0

            result = {
                "date": date_str,
                "config": config["label"],
                "error": None,
                "num_lineups": len(lineups),
                "best_actual": round(best_actual, 2),
                "avg_actual": round(avg_actual, 2),
                "worst_actual": round(worst_actual, 2),
                "pct_optimal": round(pct_optimal, 1),
                "cash_hit_rate": round(cash_hit_rate, 1),
                "gpp_ceiling": round(gpp_ceiling, 2),
                "consistency": round(consistency, 2),
                "proj_accuracy": round(proj_accuracy, 3),
                "optimal_score": optimal_score,
                "elapsed_sec": round(elapsed, 1),
                **config,
            }
            all_results.append(result)

            # Progress indicator
            if (cfg_idx + 1) % 6 == 0 or cfg_idx == len(all_configs) - 1:
                print(f"    Config {cfg_idx + 1}/{len(all_configs)}: "
                      f"{config['label']} -> best={best_actual:.1f} ({pct_optimal:.1f}% opt) "
                      f"avg={avg_actual:.1f} [{elapsed:.1f}s]")

        print()

    if not all_results:
        print("No results generated. Check API key and network connectivity.")
        return

    # ========================================================================
    # AGGREGATION + RANKING
    # ========================================================================

    results_df = pd.DataFrame(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw results
    csv_path = RESULTS_DIR / "backtest_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Raw results saved: {csv_path} ({len(results_df)} rows)")

    # Filter to successful runs only
    success = results_df[results_df["error"].isna()].copy()
    if success.empty:
        print("All optimizer runs failed. Check errors in backtest_results.csv")
        return

    # Aggregate by config
    agg = success.groupby("config").agg(
        avg_pct_optimal=("pct_optimal", "mean"),
        avg_actual_dk=("avg_actual", "mean"),
        avg_gpp_ceiling=("gpp_ceiling", "mean"),
        avg_cash_hit_rate=("cash_hit_rate", "mean"),
        avg_consistency=("consistency", "mean"),
        avg_best_actual=("best_actual", "mean"),
        avg_proj_accuracy=("proj_accuracy", "mean"),
        total_dates=("date", "nunique"),
        avg_elapsed=("elapsed_sec", "mean"),
    ).reset_index()

    # GPP ranking: avg_pct_optimal (how close to max possible)
    agg_gpp = agg.sort_values("avg_pct_optimal", ascending=False)

    # Cash ranking: cash_hit_rate * (1 / consistency) — high hit rate, low variance
    agg["cash_score"] = agg["avg_cash_hit_rate"] * (1.0 / agg["avg_consistency"].clip(lower=1.0))
    agg_cash = agg.sort_values("cash_score", ascending=False)

    # ========================================================================
    # PRINT SUMMARY
    # ========================================================================

    print("\n" + "=" * 100)
    print("TOP 10 CONFIGS FOR GPP (ranked by % of hindsight optimal)")
    print("=" * 100)
    print(f"{'RANK':<5} {'CONFIG':<40} {'%OPT':>7} {'AVG_DK':>8} {'GPP_90':>8} "
          f"{'CASH%':>7} {'PROJ_r':>7} {'DATES':>6}")
    print("-" * 100)
    for i, row in agg_gpp.head(10).iterrows():
        print(f"{agg_gpp.index.get_loc(i) + 1:<5} {row['config']:<40} "
              f"{row['avg_pct_optimal']:>6.1f}% {row['avg_actual_dk']:>7.1f} "
              f"{row['avg_gpp_ceiling']:>7.1f} {row['avg_cash_hit_rate']:>6.1f}% "
              f"{row['avg_proj_accuracy']:>6.3f} {row['total_dates']:>5}")

    print(f"\n{'=' * 100}")
    print("TOP 10 CONFIGS FOR CASH (ranked by hit_rate / consistency)")
    print("=" * 100)
    print(f"{'RANK':<5} {'CONFIG':<40} {'CASH%':>7} {'STDEV':>7} {'AVG_DK':>8} "
          f"{'%OPT':>7} {'PROJ_r':>7} {'DATES':>6}")
    print("-" * 100)
    for i, row in agg_cash.head(10).iterrows():
        print(f"{agg_cash.index.get_loc(i) + 1:<5} {row['config']:<40} "
              f"{row['avg_cash_hit_rate']:>6.1f}% {row['avg_consistency']:>6.1f} "
              f"{row['avg_actual_dk']:>7.1f} {row['avg_pct_optimal']:>6.1f}% "
              f"{row['avg_proj_accuracy']:>6.3f} {row['total_dates']:>5}")

    # ========================================================================
    # SAVE SUMMARY
    # ========================================================================

    summary_path = RESULTS_DIR / "backtest_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NBA OPTIMIZER BACKTESTING RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dates tested: {len(dates)} | Configs tested: {len(all_configs)}\n")
        f.write(f"Total optimizer runs: {len(success)} successful / {len(results_df)} total\n")
        f.write("=" * 80 + "\n\n")

        # Hindsight optimal per date
        f.write("HINDSIGHT OPTIMAL SCORES PER DATE\n")
        f.write("-" * 50 + "\n")
        for date_str, score in sorted(date_optionals.items()):
            f.write(f"  {date_str}: {score:.2f} DK pts\n")
        if date_optionals:
            f.write(f"  Average: {np.mean(list(date_optionals.values())):.2f} DK pts\n")
        f.write("\n")

        # GPP Rankings
        f.write("TOP 10 GPP CONFIGURATIONS (by % of hindsight optimal)\n")
        f.write("-" * 80 + "\n")
        for rank, (_, row) in enumerate(agg_gpp.head(10).iterrows(), 1):
            f.write(f"  #{rank}: {row['config']}\n")
            f.write(f"      Avg % Optimal: {row['avg_pct_optimal']:.1f}%\n")
            f.write(f"      Avg Actual DK: {row['avg_actual_dk']:.1f}\n")
            f.write(f"      GPP Ceiling (90th): {row['avg_gpp_ceiling']:.1f}\n")
            f.write(f"      Cash Hit Rate: {row['avg_cash_hit_rate']:.1f}%\n")
            f.write(f"      Proj Accuracy (r): {row['avg_proj_accuracy']:.3f}\n")
            f.write(f"      Avg Solve Time: {row['avg_elapsed']:.1f}s\n")
            f.write("\n")

        # Cash Rankings
        f.write("TOP 10 CASH CONFIGURATIONS (by hit_rate / consistency)\n")
        f.write("-" * 80 + "\n")
        for rank, (_, row) in enumerate(agg_cash.head(10).iterrows(), 1):
            f.write(f"  #{rank}: {row['config']}\n")
            f.write(f"      Cash Hit Rate: {row['avg_cash_hit_rate']:.1f}%\n")
            f.write(f"      Consistency (stddev): {row['avg_consistency']:.1f}\n")
            f.write(f"      Avg Actual DK: {row['avg_actual_dk']:.1f}\n")
            f.write(f"      Avg % Optimal: {row['avg_pct_optimal']:.1f}%\n")
            f.write("\n")

        # Best config details
        if not agg_gpp.empty:
            best_gpp = agg_gpp.iloc[0]
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDED SETTINGS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best GPP Config: {best_gpp['config']}\n")
            f.write(f"  -> {best_gpp['avg_pct_optimal']:.1f}% of hindsight optimal on average\n\n")

        if not agg_cash.empty:
            best_cash = agg_cash.iloc[0]
            f.write(f"Best Cash Config: {best_cash['config']}\n")
            f.write(f"  -> {best_cash['avg_cash_hit_rate']:.1f}% cash hit rate with "
                    f"{best_cash['avg_consistency']:.1f} stddev\n\n")

        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-" * 50 + "\n")

        # Does stacking help?
        stack_yes = success[success["stack_enabled"] == True]
        stack_no = success[success["stack_enabled"] == False]
        if not stack_yes.empty and not stack_no.empty:
            f.write(f"  Stacking ON avg best actual:  {stack_yes['best_actual'].mean():.1f}\n")
            f.write(f"  Stacking OFF avg best actual: {stack_no['best_actual'].mean():.1f}\n")
            diff = stack_yes['best_actual'].mean() - stack_no['best_actual'].mean()
            f.write(f"  -> Stacking {'helps' if diff > 0 else 'hurts'} by {abs(diff):.1f} DK pts on average\n\n")

        # What exposure is best?
        for exp_val in sorted(PARAM_GRID["max_exposure"]):
            subset = success[success["max_exposure"] == exp_val]
            if not subset.empty:
                f.write(f"  Exposure {exp_val}%: avg_best={subset['best_actual'].mean():.1f}, "
                        f"cash_rate={subset['cash_hit_rate'].mean():.1f}%\n")
        f.write("\n")

        # Lineup count
        for nl in PARAM_GRID["num_lineups"]:
            subset = success[success["num_lineups"] == nl]
            if not subset.empty:
                f.write(f"  {nl} lineups: avg_best={subset['best_actual'].mean():.1f}, "
                        f"pct_opt={subset['pct_optimal'].mean():.1f}%\n")

    print(f"\nSummary saved: {summary_path}")
    print(f"\nDone! Total time: check backtest_summary.txt for details")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NBA Optimizer Backtesting System")
    parser.add_argument("--dates", type=int, default=None,
                        help="Number of dates to test (default: all 15)")
    parser.add_argument("--configs", type=int, default=None,
                        help="Max number of configs to test (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-fetch from API (ignore cache)")
    args = parser.parse_args()

    run_backtest(args)


if __name__ == "__main__":
    main()
