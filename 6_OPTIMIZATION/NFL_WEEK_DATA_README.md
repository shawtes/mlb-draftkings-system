# NFL Week Data Creator - Quick Reference

## **One Command to Rule Them All** 🏈

Instead of asking me to create Week X data each time, **just run this script!**

## Usage

### Basic (Auto-calculates date):
```bash
python3 create_nfl_week_data.py --week 6
```

### With specific date:
```bash
python3 create_nfl_week_data.py --week 7 --date 2025-10-20
```

### Full options:
```bash
python3 create_nfl_week_data.py --week 8 --season 2025REG --date 2025-10-27 --output-dir .
```

## What It Does

1. ✅ Fetches projections from SportsData.io API
2. ✅ Fetches actuals (if games completed)
3. ✅ Fetches DraftKings salaries
4. ✅ Merges all data intelligently
5. ✅ Creates **2 optimized versions:**
   - `nfl_weekX_gpp_enhanced.csv` (Tournament)
   - `nfl_weekX_cash_enhanced.csv` (Cash Games)
6. ✅ Adds realistic DST projections automatically
7. ✅ Shows summary of top players

## Output Files

### GPP Version:
- Emphasizes **ceiling** (high upside)
- Boosts **low-ownership** plays
- Best for tournaments

### Cash Version:
- Emphasizes **floor** (consistency)
- Boosts **high-ownership** safe plays  
- Best for 50/50s, head-to-heads

## Examples

### Create Week 7 data:
```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION
python3 create_nfl_week_data.py --week 7 --date 2025-10-20
```

**Output:**
```
✅ Week Data Created Successfully!

📁 Files saved to:
   • nfl_week7_gpp_enhanced.csv
   • nfl_week7_cash_enhanced.csv
```

### Create Week 8 data:
```bash
python3 create_nfl_week_data.py --week 8 --date 2025-10-27
```

### Create Week 9 data:
```bash
python3 create_nfl_week_data.py --week 9
```
*(Date auto-calculated)*

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--week` | ✅ Yes | - | NFL week number (1-18) |
| `--season` | No | `2025REG` | Season (format: YYYYRRR) |
| `--date` | No | Auto | DFS slate date (YYYY-MM-DD) |
| `--output-dir` | No | `.` | Where to save files |

## What's Included in Each File

- **98+ players** with DraftKings salaries
- **All positions:** QB, RB, WR, TE, DST
- **Projections:** Fantasy points, ceiling, floor
- **Value metrics:** Points per $1K
- **Ownership:** Estimated % (for strategy)
- **Game environment:** Total, spread, weather
- **Enhanced metrics:** Everything the optimizer needs

## Tips

1. **Run before each slate** to get latest data
2. **Use GPP file for tournaments** (20-150 lineups)
3. **Use Cash file for 50/50s** (1-5 lineups)
4. **Check DST projections** in output summary
5. **Dates matter** - use Sunday of that week

## Troubleshooting

### "Failed to fetch projections"
→ Week data not available yet (too far in future)

### "No DraftKings slates found"
→ Try a different date (usually Sunday of that week)

### "No valid players after merge"
→ All players had $0 salary (wrong slate), script auto-fixes now

## Quick Dates Reference (2025 Season)

| Week | Approx Date | Command |
|------|-------------|---------|
| 6 | Oct 13 | `--week 6 --date 2025-10-13` |
| 7 | Oct 20 | `--week 7 --date 2025-10-20` |
| 8 | Oct 27 | `--week 8 --date 2025-10-27` |
| 9 | Nov 3 | `--week 9 --date 2025-11-03` |
| 10 | Nov 10 | `--week 10 --date 2025-11-10` |

## After Running

1. Open `genetic_algo_nfl_optimizer.py`
2. Load the generated CSV file
3. Generate lineups
4. Export to DraftKings format
5. Upload and win! 💰

## Script Location

```
/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/create_nfl_week_data.py
```

---

**🎯 No more asking me! Just run the script for any week you need!** 🚀

