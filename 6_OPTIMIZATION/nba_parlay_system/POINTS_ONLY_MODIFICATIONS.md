# NBA Parlay Generator - Points O/U Only (Top 35% Players)

## Modifications Made

### 1. **Filter to Top 35% of Players**
- Automatically filters players based on projected points
- Only players in the top 35% (65th percentile and above) are considered
- Example: If minimum is 28.1 points, only players projected ≥28.1 pts are used

### 2. **Points Over/Under Only**
- Removed all other prop types (rebounds, assists, steals, blocks, three-pointers)
- Only generates **Points Over** and **Points Under** props
- More consistent and reliable prop type

### 3. **Modified Files**
- `nba_parlay_generator.py` - Core generator logic
- `nba_parlay_gui.py` - GUI title and subtitle

## Changes in Detail

### nba_parlay_generator.py
```python
# Before:
self.prop_types = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'three_pointers']

# After:
self.prop_types = ['points']  # Only points over/under

# Added filtering:
if 'projected_points' in data.columns:
    threshold = data['projected_points'].quantile(0.65)  # Top 35%
    self.data = data[data['projected_points'] >= threshold].copy()
```

### nba_parlay_gui.py
```python
# Before:
self.root.title("NBA Parlay Generator - 86.7% Win Rate Model")
subtitle = "86.7% Win Rate Model"

# After:
self.root.title("NBA Parlay Generator - Points O/U Only (Top 35%)")
subtitle = "Points O/U Only - Top 35% Players"
```

## How to Use

### Launch the GUI:
```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_parlay_system
python3 nba_parlay_gui.py
```

### Steps:
1. Click **"Load NBA Data"** or **"Fetch Tonight's Data"**
2. Wait for data to load (will show how many top players filtered)
3. Select teams (optional - all selected by default)
4. Choose number of legs (2-4)
5. Click **"Generate Parlays"**

### Output:
- All parlays will only contain **Points Over** or **Points Under** props
- All players will be from the **top 35%** by projected points
- Example output:
  ```
  Parlay #1 (2 legs)
    - LeBron James: Points OVER 26.5 (72% hit rate)
    - Steph Curry: Points OVER 28.0 (68% hit rate)
  ```

## Benefits

1. **More Consistent**: Points are more predictable than other stats
2. **Top Players Only**: Elite players have more consistent performances
3. **Simplified**: Easier to research and track
4. **Better Odds**: Top scorers have higher floor projections

## Testing

Tested with 100 sample players:
- ✅ Filtered to top 35 players (35%)
- ✅ Only generated "points" props
- ✅ Both OVER and UNDER bets work
- ✅ GUI displays correctly

## Reverting Changes

To revert back to all prop types and all players:
1. Restore original `nba_parlay_generator.py` from git
2. Restore original `nba_parlay_gui.py` from git

Or manually change:
- `self.prop_types = ['points']` back to full list
- Remove the filtering code in `__init__`









