# How to Get October 27th NBA Data

## Problem
The API subscription appears to be expired or invalid.

## Options

### Option 1: Use Existing Data
You can test the optimizer with Oct 26 data (already fixed):
```bash
python3 "nba_sportsdata.io_gentic algo.py"
# Then load: nba_tonight_FIXED.csv
```

### Option 2: Download Oct 27 Data Manually
1. Go to SportsData.io website
2. Download the PlayerGameProjectionStats for Oct 27
3. Save as: `nba_tonight_6pm_2025OCT27.csv`
4. Then run: `python3 fix_nba_data_for_optimizer.py`

### Option 3: Use DraftKings Export
1. Export player pool from DraftKings
2. Save as CSV with columns: Name, Position, Team, Salary
3. Add projections column

### Option 4: Renew API Subscription
Contact SportsData.io to renew your subscription.

## Quick Test
Try running the genetic optimizer with Oct 26 data first to make sure everything works:
```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION
python3 "nba_sportsdata.io_gentic algo.py"
# Load: nba_tonight_FIXED.csv
```

