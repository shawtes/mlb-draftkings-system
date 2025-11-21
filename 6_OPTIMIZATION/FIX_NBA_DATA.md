# Fix NBA Data Issues

## Problem Summary

The NBA data files have these issues:

1. **Missing Real Salaries**: The `nba_tonight_6pm_2025OCT26.csv` file has **projection data** but NO salary information
2. **Wrong File Format**: The `nba_draftkings_tonight_2025OCT26.csv` has **Showdown/Captain mode** data (not regular DFS)
3. **Only 7 Games**: Your current slate has 7 games, but you need 9 games for a full slate

## Current Data Files

| File | Games | Players | Has Salaries? | Type |
|------|-------|---------|---------------|------|
| `nba_tonight_6pm_2025OCT26.csv` | 7 | 244 | ❌ No | Projections only |
| `nba_draftkings_tonight_2025OCT26.csv` | ? | 70 | ✅ Yes | Showdown (Captain) format |

## Issues

1. **nba_tonight_6pm_2025OCT26.csv** (your projection data):
   - ✅ Has Name, Position, Team, Projected points
   - ❌ NO salary column
   - ❌ Only 7 games (need 9)
   - This is projection data from API

2. **nba_draftkings_tonight_2025OCT26.csv** (DraftKings salary data):
   - ✅ Has real DraftKings salaries (OperatorSalary)
   - ❌ Showdown/Captain mode (not regular DFS)
   - ❌ Different players (Victor Wembanyama, etc.)
   - Different format/API endpoint

## Why Salaries Don't Match

The two files are **completely different**:
- Projection file: All players from 7 games (MIL, CLE, MIA, NY, etc.)
- Salary file: Showdown slate with only a few players (SA, BKN, etc.)

They represent **different slate types**:
- **Main Slate**: 7-game main slate (Oct 26 6pm EST)
- **Showdown**: Single game with Captain mode (different players)

## Solutions

### Option 1: Get Real Main Slate Data from API

You need to fetch BOTH:
1. **DFS Slate with Salaries** - DraftKings main slate for Oct 26
2. **Projections** - For those same players

The API should provide:
- `/DfsSlatesByDate/{date}` - Get main slate with salaries
- `/PlayerGameProjectionStatsByDate/{date}` - Get projections

### Option 2: Use What You Have

If you want to proceed with the 7-game slate:

1. The projections have fantasy points
2. You can estimate salaries OR get them manually
3. Use the optimizer with fewer games

### Option 3: Find Correct Date with 9 Games

Try different dates to find one with 9 games.

## Files Created

1. ✅ `filter_nba_with_salaries.py` - Attempts to merge salary data
2. ✅ `analyze_nba_slate.py` - Analyzes game counts
3. ✅ `fetch_nba_9game_slate.py` - Fetches from API (need to fix auth)

## Next Steps

1. **Check API access** - Fix the 401 error to get real data
2. **Find correct slate date** - Look for a date with 9 games
3. **Manually add salaries** - If you have DraftKings, scrape salaries

## How to Check Files

```bash
# Check projection data
python3 analyze_nba_slate.py nba_tonight_6pm_2025OCT26.csv

# Check salary data  
head nba_draftkings_tonight_2025OCT26.csv

# See what games are available
awk -F',' 'NR>1 {print $11}' nba_tonight_6pm_2025OCT26.csv | sort -u
```

## Bottom Line

**Your data is projection-only.** You need **REAL DraftKings salaries** from the main slate API endpoint to get proper salary data.

The genetic optimizer requires:
- ✅ Name, Position, Team
- ✅ **REAL Salary** (not fake $5000)
- ✅ Predicted_DK_Points

You currently only have projections without salaries.









