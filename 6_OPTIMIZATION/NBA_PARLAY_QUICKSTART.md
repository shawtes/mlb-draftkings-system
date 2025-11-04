# NBA Parlay Generator - Quick Start Guide

## ✅ MODIFICATIONS COMPLETE

The NBA Parlay Generator has been updated to:
1. **Only generate Points O/U props** (no rebounds, assists, etc.)
2. **Only use top 35% of players** by projected points
3. **Save raw API data** for reference

## 🚀 HOW TO USE

### Step 1: Fetch Today's NBA Data

```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION

# Edit the date in daily_nba_data_fetch.py to today's date
# Example: GAME_DATE = "2025-OCT-30"

# Run the fetcher
python3 daily_nba_data_fetch.py
```

**Output files created:**
- `nba_rawapi_2025oct30.csv` - Original complete API response
- `nba_oct30_READY.csv` - Filtered and formatted for optimizer/parlay generator

### Step 2: Launch Parlay GUI

```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_parlay_system

python3 nba_parlay_gui.py
```

### Step 3: Load Data in GUI

1. Click **"Load NBA Data"**
2. Navigate to `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/`
3. Select `nba_oct29_READY.csv` (or today's date)
4. GUI will show: "✅ Filtered to top 35%: X players"

### Step 4: Generate Parlays

1. Select teams (or leave all selected)
2. Choose number of legs (2-4 recommended)
3. Click **"Generate Parlays"**
4. Results show **Points O/U only** from **top 35% players**

## 📊 EXAMPLE OUTPUT

```
Parlay #1 (3 legs) - 56.3% combined
  - LeBron James (LAL): points OVER 26.5 (72% hit rate)
  - Steph Curry (GSW): points OVER 28.0 (68% hit rate)
  - Luka Doncic (DAL): points OVER 31.5 (78% hit rate)
```

## 🔧 TROUBLESHOOTING

### Problem: "No data for this date"
**Solution:** Oct 30 might not have games. Check NBA schedule and use a different date.

### Problem: "KeyError: player_id" or similar
**Solution:** Use the `daily_nba_data_fetch.py` script - it adds all required columns automatically.

### Problem: "Only X players after filtering"
**Solution:** Normal! If you start with 67 players, top 35% = ~24 players. This is working correctly.

### Problem: GUI won't load the CSV
**Solution:** Make sure the CSV has these columns:
- `player_id`
- `player_name_proj`
- `team_proj`
- `position_proj`
- `projected_points`
- `projected_dk_points`

## 📁 FILE LOCATIONS

- **Fetcher script:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/daily_nba_data_fetch.py`
- **Parlay GUI:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_parlay_system/nba_parlay_gui.py`
- **Generator logic:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_parlay_system/nba_parlay_generator.py`
- **Output data:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_octXX_READY.csv`
- **Raw API data:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_raw_api_2025octXX.csv`

## 🔄 WORKFLOW SUMMARY

```
1. daily_nba_data_fetch.py 
   ↓
   (fetches API data, filters by DK entries, adds parlay columns)
   ↓
2. nba_octXX_READY.csv 
   ↓
   (load into GUI)
   ↓
3. nba_parlay_gui.py
   ↓
   (generates parlays: points O/U only, top 35% players)
   ↓
4. View/export parlays
```

## ✨ KEY FEATURES

- ✅ Only Points Over/Under (most consistent prop)
- ✅ Top 35% players only (elite scorers with higher floor)
- ✅ Raw API data saved for reference
- ✅ Automatic column mapping for compatibility
- ✅ Player name normalization (handles special characters)
- ✅ DraftKings salary integration

## 📝 NOTES

- The Oct 29 data has **67 players**, filtered to **24 top players** (35%)
- Minimum projected points for top 35%: **~19.0 DK points**
- All parlays will only show **"points OVER"** or **"points UNDER"**
- Hit rates shown are calculated from historical accuracy models





