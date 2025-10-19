# NFL Injury Filtering - Automatic Player Health Checks

## Overview

The optimizer now **automatically filters out injured players** to ensure you only use healthy, active players in your lineups.

## What It Does

✅ **Fetches real injury data** from SportsData.io API  
✅ **Removes OUT players** - Players ruled out for the game  
✅ **Removes DOUBTFUL players** - Players unlikely to play (<25% chance)  
✅ **Keeps QUESTIONABLE players** - Game-time decisions  
✅ **Keeps PROBABLE players** - Likely to play  
✅ **Keeps HEALTHY players** - No injury designation  

## How It Works

### 1. Automatic Injury Check (Built Into Script)

When you run `create_nfl_week_data.py`, it automatically:
```bash
python3 create_nfl_week_data.py --week 7 --date 2025-10-20
```

**What happens:**
```
1️⃣ Fetching Week 7 Projections... ✅
2️⃣ Fetching Week 7 Actuals... ✅
3️⃣ Fetching Week 7 Injury Reports... ✅
4️⃣ Fetching DraftKings Salaries... ✅
5️⃣ Filtering injured players...
   ❌ Removed 3 injured players (OUT/DOUBTFUL):
      RB  Christian McCaffrey     - OUT
      WR  Justin Jefferson        - DOUBTFUL
      QB  Trevor Lawrence         - OUT
   ✅ 95 healthy players remaining
6️⃣ Creating GPP version...
7️⃣ Creating Cash version...
8️⃣ Summary Statistics...
```

### 2. Injury Statuses

| Status | Included? | Description |
|--------|-----------|-------------|
| **OUT** | ❌ No | Player ruled out, will not play |
| **DOUBTFUL** | ❌ No | Very unlikely to play (<25% chance) |
| **QUESTIONABLE** | ✅ Yes | Game-time decision (50% chance) |
| **PROBABLE** | ✅ Yes | Likely to play (>75% chance) |
| **HEALTHY** | ✅ Yes | No injury designation |

### 3. Why QUESTIONABLE Players Included

**QUESTIONABLE players are kept in the pool because:**
- They often do play (50%+ chance)
- You can manually exclude them if you want
- Late-breaking news can change status
- Gives you flexibility for lineup decisions

**If you want to be more conservative:**
1. Generate the CSV
2. Manually remove QUESTIONABLE players
3. Reload into optimizer

## Example Output

### With Injuries (Before Filter):
```
Total: 98 players
- 3 OUT players
- 2 DOUBTFUL players  
- 93 HEALTHY/QUESTIONABLE players
```

### After Automatic Filtering:
```
❌ Removed 5 injured players:
   RB  Christian McCaffrey     - OUT
   WR  Justin Jefferson        - DOUBTFUL
   QB  Trevor Lawrence         - OUT
   TE  Mark Andrews            - DOUBTFUL
   RB  Nick Chubb              - OUT

✅ 93 healthy players remaining (for your optimizer)
```

## What If Injury Data Unavailable?

If the API doesn't have injury data for that week:
```
3️⃣ Fetching Week 7 Injury Reports...
❌ Error: Data not found (404)
   No injury data available for 2025REG Week 7
   ⚠️  No injury data available

5️⃣ Filtering injured players...
   ℹ️  No injury data to filter
   ✅ All players included (no injury filter applied)
```

**What this means:**
- The script continues without injury filtering
- All players in the slate are included
- You should manually check injury reports
- Safer for weeks where injury data isn't available yet

## Manual Override

**If you want to manually exclude a player:**

1. Open the generated CSV (e.g., `nfl_week7_gpp_enhanced.csv`)
2. Delete the row(s) for players you don't want
3. Save the file
4. Load it in the optimizer

**Or filter in the optimizer GUI:**
- Use the player table to exclude specific players
- Set custom exposure limits to 0%

## API Endpoint Used

```
Endpoint: https://api.sportsdata.io/api/nfl/fantasy/json/Injuries/{season}/{week}

Example: https://api.sportsdata.io/api/nfl/fantasy/json/Injuries/2025REG/7
```

## Files Modified

1. **`sportsdata_nfl_api.py`**
   - Added `get_injuries_by_week()` method
   - Fetches injury reports from SportsData.io

2. **`create_nfl_week_data.py`**
   - Added `filter_injured_players()` function
   - Automatically removes OUT/DOUBTFUL players
   - Shows which players were filtered

## Benefits

✅ **Avoid dead weight** - No players who won't play  
✅ **Save time** - Automatic instead of manual checks  
✅ **Stay updated** - Uses latest injury reports  
✅ **Transparency** - Shows exactly who was removed  
✅ **Flexibility** - Can still manually adjust if needed  

## Conservative vs Aggressive Strategy

### Conservative (Safest):
```python
# Manually edit the filter to remove QUESTIONABLE too
injured_statuses = ['OUT', 'DOUBTFUL', 'QUESTIONABLE']
```

### Balanced (Default):
```python
# Current setting - only removes OUT/DOUBTFUL
injured_statuses = ['OUT', 'DOUBTFUL']
```

### Aggressive (Riskiest):
```python
# Only remove OUT players
injured_statuses = ['OUT']
```

## Troubleshooting

### "No injury data available"
→ API doesn't have data for that week yet (too early or too late)  
→ Script continues without filtering - manually check injury reports

### "Removed X injured players" but player still seems healthy
→ Injury status might be outdated (check latest news)  
→ Re-run script closer to game time for latest data

### Want to keep a DOUBTFUL player (gut feeling they'll play)
→ Manually add them back to the CSV after generation  
→ Or change the filter in the script

## Quick Reference

### Run with injury filtering (automatic):
```bash
python3 create_nfl_week_data.py --week 7
```

### Check what was filtered:
Look for this section in the output:
```
5️⃣ Filtering injured players...
   ❌ Removed X injured players (OUT/DOUBTFUL):
      [List of removed players]
```

### Files created:
- `nfl_weekX_gpp_enhanced.csv` - **Only healthy players**
- `nfl_weekX_cash_enhanced.csv` - **Only healthy players**

## Summary

🏥 **Injury filtering is now automatic!**  
✅ OUT and DOUBTFUL players are automatically removed  
⚠️ QUESTIONABLE players are kept (you decide)  
🔧 Can be overridden manually if needed  

**You're now protected from using injured players in your lineups!** 🛡️

---

Last Updated: October 18, 2025

