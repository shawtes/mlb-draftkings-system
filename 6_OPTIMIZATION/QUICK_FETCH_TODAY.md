# Quick Guide: Fetch Today's DFS Data

## Simple Usage

### Option 1: Fetch Today's Data (Easiest)
```bash
python fetch_todays_dfs.py
```
This will:
- Use today's date automatically
- Estimate the NFL week
- Fetch all player data with projections
- Save to CSV file

### Option 2: Specify Week
```bash
python fetch_todays_dfs.py --week 8
```
This uses today's date but you specify the week number.

### Option 3: Specify Date and Week
```bash
python fetch_todays_dfs.py --date 2025-10-26 --week 8
```

### Option 4: Fast Mode (No Projections)
```bash
python fetch_todays_dfs.py --no-projections
```
This fetches only salaries and player info (faster, no projections).

## Output

The script creates a CSV file like:
- `nfl_dfs_players_2025REG_week8_20251026.csv`

This CSV contains:
- ✅ Player names
- ✅ Positions (QB, RB, WR, TE, DST)
- ✅ DraftKings salaries
- ✅ Player projections (if enabled)
- ✅ Team and opponent info
- ✅ All data formatted for your optimizer

## How It Works

1. **Connects to SportsData.io API** - Uses your API key to fetch real-time data
2. **Fetches DFS Slate** - Gets today's DraftKings slate with all players
3. **Gets Projections** - Fetches player projections for the week (optional)
4. **Formats Data** - Converts everything to CSV format your optimizer expects
5. **Saves File** - Creates a CSV file ready to use

## Troubleshooting

**No slate found?**
- Check if there are games scheduled for that date
- NFL games are typically Sunday, Monday, Thursday
- Try a different date with `--date YYYY-MM-DD`

**Wrong week?**
- Always specify `--week` if you know the exact week
- Week estimation from date is approximate

**API errors?**
- Check your API key is valid
- Make sure you have internet connection
- API may have rate limits

## Example Workflow

```bash
# Step 1: Fetch today's data
python fetch_todays_dfs.py --week 8

# Step 2: Use the CSV in your optimizer
# The CSV file will be named something like:
# nfl_dfs_players_2025REG_week8_20251026.csv

# Step 3: Load it in your optimizer and generate lineups!
```

## Advanced Options

```bash
# Custom output filename
python fetch_todays_dfs.py --week 8 --output my_lineup_data.csv

# Different season
python fetch_todays_dfs.py --date 2025-10-26 --week 8 --season 2025REG

# Custom API key
python fetch_todays_dfs.py --week 8 --api-key YOUR_API_KEY_HERE
```


