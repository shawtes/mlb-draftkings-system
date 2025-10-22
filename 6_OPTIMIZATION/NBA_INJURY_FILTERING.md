# NBA Injury Filtering - Automatic Player Health Checks

## Overview

The NBA optimizer now **automatically filters out injured players** to ensure you only use healthy, active players in your lineups.

## What It Does

✅ **Automatically detects injury columns** - Looks for `InjuryStatus`, `Status`, or any injury-related column  
✅ **Removes OUT players** - Players ruled out for the game  
✅ **Removes DOUBTFUL players** - Players unlikely to play (<25% chance)  
✅ **Keeps QUESTIONABLE players** - Game-time decisions (50% chance to play)  
✅ **Keeps PROBABLE players** - Likely to play (>75% chance)  
✅ **Keeps HEALTHY players** - No injury designation  

## How It Works

### Automatic Filtering

When you load a player CSV file, the optimizer automatically:

1. **Scans for injury columns** - Looks for columns containing "injury", "status", etc.
2. **Filters OUT/DOUBTFUL** - Removes players who won't play
3. **Shows removed players** - Displays who was filtered and why
4. **Proceeds with healthy players** - Optimization uses only available players

### Console Output Example

```
🏥 INJURY REPORT FILTERING
======================================================================
   📋 Using injury column: 'InjuryStatus'
   ❌ Removed 4 injured players (OUT/DOUBTFUL):
      PG   Ja Morant                      - OUT
      SF   Kawhi Leonard                  - OUT
      C    Joel Embiid                    - DOUBTFUL
      SG   Bradley Beal                   - D
   ✅ 416 healthy players remaining
======================================================================
```

## Injury Statuses

| Status | Included? | Description |
|--------|-----------|-------------|
| **OUT** or **O** | ❌ No | Player ruled out, will not play |
| **DOUBTFUL** or **D** | ❌ No | Very unlikely to play (<25% chance) |
| **QUESTIONABLE** or **Q** | ✅ Yes | Game-time decision (50% chance) |
| **PROBABLE** or **P** | ✅ Yes | Likely to play (>75% chance) |
| **HEALTHY** or blank | ✅ Yes | No injury designation |

## CSV Format Requirements

Your player CSV should include an injury status column. Supported column names:
- `InjuryStatus`
- `Status`
- `Injury_Status`
- Any column with "injury" or "status" in the name

### Example CSV Format

```csv
Name,Position,Team,Salary,Predicted_DK_Points,InjuryStatus
LeBron James,SF,LAL,9500,48.5,HEALTHY
Stephen Curry,PG,GSW,9200,46.2,Q
Joel Embiid,C,PHI,10000,52.1,OUT
Giannis Antetokounmpo,PF,MIL,11000,55.8,
```

## Why QUESTIONABLE Players Are Included

**QUESTIONABLE (Q) players are kept in the pool because:**
- They often do play (50%+ chance)
- You can manually exclude them via the UI if you want
- Late-breaking news can change status before game time
- Gives you flexibility for lineup decisions

## Manual Exclusion

Even with healthy players, you can still:
1. Use the **player checkboxes** to exclude specific players
2. Monitor **late injury updates** before games start
3. Remove **game-time decisions** if you want to be conservative

## Notes

- If no injury column is found, **all players are included** (backward compatible)
- Injury filtering happens **before optimization** starts
- Filtering is **case-insensitive** (OUT, out, O all work)
- The original CSV file is **not modified** - filtering only affects the loaded data

## Best Practices

1. **Get fresh data** - Update your CSV with latest injury reports
2. **Check game times** - Remove QUESTIONABLE players for early games
3. **Monitor news** - Check Twitter/Rotoworld for late scratches
4. **Use multiple lineups** - Spread risk across different player combinations

---

✅ **Injury filtering is automatic** - Just load your CSV and the optimizer handles the rest!

