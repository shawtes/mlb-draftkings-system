# My Entries Tab Design
## Multi-Session Lineup Management and Favorites

---

## Purpose

The **My Entries Tab** (also called "Favorites") enables users to curate lineups across multiple optimization sessions. Instead of exporting immediately after each optimization, users can:
- Save favorite lineups from multiple runs
- Build a portfolio of 150-500 lineups over time
- Mix lineups from different strategies
- Review and refine before final export

**Use Case:** "I want to run optimization 5 times with different settings, keep the best 30 lineups from each run, then export my top 150 for the contest"

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│ 💾 My Entries Tab                                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Build your final contest lineup from multiple optimization   │
│  runs. Add lineups to favorites, then export when ready.      │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ ACTIONS                                                  │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ [➕ Add Current Pool]  [🗑️ Clear All]  [💾 Export]      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 📊 Total Favorites: 150 lineups from 5 runs              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ FAVORITES TABLE                                          │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ ☐│Run│Player      │Team│Pos│Sal  │Pts │Total│Tot │Date ││ │
│  ├──┼───┼────────────┼────┼───┼─────┼────┼─────┼───┼─────┤│ │
│  │☑│R1 │Shohei O.   │LAA │P  │11000│25.3│49800│125│10/17││ │
│  │☑│R1 │Aaron Judge │NYY │OF │6500 │12.8│49800│125│10/17││ │
│  │☑│R1 │Mookie B.   │LAD │OF │6200 │11.7│49800│125│10/17││ │
│  │☑│R1 │...         │... │.. │...  │... │     │   │     ││ │
│  │☑│R2 │Sandy A.    │SD  │P  │10800│24.1│49500│123│10/17││ │
│  │☑│R2 │Rafael D.   │BOS │3B │5800 │11.2│49500│123│10/17││ │
│  │☑│R2 │...         │... │.. │...  │... │     │   │     ││ │
│  │☑│R3 │Corbin B.   │ATL │P  │10500│23.5│49700│122│10/17││ │
│  │☑│R3 │...         │... │.. │...  │... │     │   │     ││ │
│  │ ...                                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Showing 1-100 of 150 lineups  [< Prev] [Next >]             │
└────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Action Toolbar

#### Add Current Pool to Favorites Button
```
┌────────────────────────────┐
│ ➕ Add Current Pool to     │
│    Favorites               │
└────────────────────────────┘
```

**Function:**
- Adds lineups from most recent optimization to favorites
- Opens dialog to specify how many to add
- Tags with run number for tracking

**Dialog:**
```
┌─────────────────────────────────────────┐
│ Add to Favorites                        │
├─────────────────────────────────────────┤
│                                         │
│ How many lineups to add?                │
│                                         │
│ Available: 100 optimized lineups       │
│                                         │
│ Add: [30_____] lineups                  │
│      (top scoring will be selected)     │
│                                         │
│ Run #: 5 (auto-assigned)                │
│ Date: 2025-10-17 20:45                  │
│                                         │
│ Current favorites: 120 lineups          │
│ After add: 150 lineups total            │
│                                         │
│     [Cancel]  [Add to Favorites]        │
└─────────────────────────────────────────┘
```

**Behavior:**
- Defaults to all lineups or 50 (whichever is smaller)
- Sorts by projected points (highest first)
- Adds top N to favorites collection
- Auto-increments run number
- Updates display immediately

#### Clear All Button
```
┌──────────────────┐
│ 🗑️ Clear All     │
└──────────────────┘
```

**Function:**
- Removes ALL favorite lineups
- Shows confirmation dialog
- Cannot be undone

**Confirmation:**
```
┌─────────────────────────────────────────┐
│ Clear Favorites                         │
├─────────────────────────────────────────┤
│                                         │
│ ⚠ Delete all 150 favorite lineups?     │
│                                         │
│ This action cannot be undone.           │
│                                         │
│     [Cancel]  [Yes, Clear All]          │
└─────────────────────────────────────────┘
```

#### Export Button
```
┌────────────────────────────┐
│ 💾 Export Favorites as     │
│    New Lineups             │
└────────────────────────────┘
```

**Function:**
- Exports favorites to DraftKings CSV format
- Can select how many to export
- Uses player IDs for proper DK format

**Dialog:**
```
┌─────────────────────────────────────────┐
│ Export Favorites                        │
├─────────────────────────────────────────┤
│                                         │
│ Export how many lineups?                │
│                                         │
│ Available: 150 favorite lineups         │
│ From: 5 optimization runs               │
│                                         │
│ Export: [150____] lineups               │
│                                         │
│ Save as: [my_favorites.csv___________]  │
│                                         │
│ Format: DraftKings contest entry format │
│ • Entry ID, Contest, ID, Fee            │
│ • P, P, C, 1B, 2B, 3B, SS, OF, OF, OF   │
│ • Uses player IDs (not names)           │
│                                         │
│ ✓ Ready to upload to DraftKings         │
│                                         │
│     [Cancel]  [Export]                  │
└─────────────────────────────────────────┘
```

### 2. Statistics Display

```
┌──────────────────────────────────────────────────┐
│ 📊 Total Favorites: 150 lineups from 5 runs     │
│                                                  │
│ By Run:                                          │
│ • Run 1: 30 lineups (10/17 14:23)               │
│ • Run 2: 35 lineups (10/17 15:45)               │
│ • Run 3: 25 lineups (10/17 17:12)               │
│ • Run 4: 30 lineups (10/17 18:30)               │
│ • Run 5: 30 lineups (10/17 20:15)               │
│                                                  │
│ Point Range: 118.5 - 128.3                      │
│ Salary Range: $48,900 - $50,000                 │
└──────────────────────────────────────────────────┘
```

### 3. Favorites Table

**Display Format:**
Each lineup is shown as multiple rows (one per player)

**Columns:**

#### Select Checkbox
- **Width:** 30px
- **Purpose:** Mark lineup for actions (export, delete)
- **Default:** Checked
- **Behavior:** Check/uncheck entire lineup

#### Run Number
- **Width:** 50px
- **Content:** "R1", "R2", "R3", etc.
- **Purpose:** Track which optimization session
- **Grouping:** Lineups from same run have same number
- **Visual:** Color-coded by run

#### Player
- **Width:** 150px
- **Content:** Player name
- **Format:** "First Last" or "First L."
- **Sorting:** Alphabetical

#### Team
- **Width:** 50px
- **Content:** Team abbreviation
- **Example:** "NYY", "LAD"

#### Position
- **Width:** 50px
- **Content:** DraftKings position
- **Example:** "P", "OF", "SS"

#### Salary
- **Width:** 70px
- **Content:** Player salary
- **Format:** "$X,XXX"

#### Points
- **Width:** 60px
- **Content:** Projected DK points
- **Format:** "XX.X"

#### Total Salary
- **Width:** 80px
- **Content:** Lineup total salary
- **Format:** "$XX,XXX"
- **Repeats:** Same for all 10 players in lineup

#### Total Points
- **Width:** 70px
- **Content:** Lineup total points
- **Format:** "XXX.X"
- **Repeats:** Same for all 10 players in lineup

#### Date Added
- **Width:** 90px
- **Content:** When added to favorites
- **Format:** "MM/DD HH:MM"

#### Actions
- **Width:** 80px
- **Content:** Delete button for this lineup
- **Button:** "🗑️" delete icon
- **Confirmation:** "Delete this lineup?"

---

## Data Persistence

### Storage Format

**File:** `favorites_lineups.json` (in app directory)

**Structure:**
```json
[
  {
    "lineup_data": [
      {
        "Name": "Shohei Ohtani",
        "Team": "LAA",
        "Position": "P",
        "Salary": 11000,
        "Predicted_DK_Points": 25.3
      },
      // ... 9 more players
    ],
    "total_points": 125.3,
    "total_salary": 49800,
    "run_number": 1,
    "date_added": "2025-10-17 14:23:15"
  },
  // ... more lineups
]
```

### Auto-Save

**Triggers:**
- After adding lineups
- After deleting lineups
- After clearing all
- On app close (final save)

**Benefits:**
- Persist across sessions
- Survive app crashes
- Build portfolio over days

---

## User Workflows

### Workflow 1: Multi-Run Portfolio Building

**Scenario:** Building 150 lineups over 5 optimization runs

```
Session 1 (2:00 PM):
├─ Settings: 4-Stack, NYY/LAD/ATL
├─ Generate: 100 lineups
├─ Review: Sort by points
├─ Action: Add top 30 to favorites
└─ Status: 30 favorites (Run #1)

Session 2 (3:45 PM):
├─ Settings: 5-Stack, NYY/LAD only
├─ Generate: 100 lineups
├─ Review: Different lineup construction
├─ Action: Add top 35 to favorites
└─ Status: 65 favorites (Run #1-2)

Session 3 (5:12 PM):
├─ Settings: 3|3|2 multi-stack
├─ Generate: 100 lineups
├─ Review: More diverse lineups
├─ Action: Add top 25 to favorites
└─ Status: 90 favorites (Run #1-3)

Session 4 (6:30 PM):
├─ Settings: No stacks (contrarian)
├─ Generate: 100 lineups
├─ Review: Unique construction
├─ Action: Add top 30 to favorites
└─ Status: 120 favorites (Run #1-4)

Session 5 (8:15 PM - Lock approaching):
├─ Settings: Final tweaks, 4|2 stacks
├─ Generate: 100 lineups
├─ Review: Last-minute adjustments
├─ Action: Add top 30 to favorites
└─ Status: 150 favorites (Run #1-5)

Final Export:
├─ Review: All 150 favorites
├─ Remove: 0 (keep all)
├─ Export: my_final_150_lineups.csv
└─ Upload: To DraftKings ✓
```

### Workflow 2: Strategy Testing

**Scenario:** Test multiple strategies, keep only best

```
Test 1: Conservative (4-Stack)
├─ Generate 50 lineups
├─ Add all 50 to favorites (Run #1)
└─ Tag: "Conservative approach"

Test 2: Balanced (4|2 Multi-Stack)
├─ Generate 50 lineups
├─ Add all 50 to favorites (Run #2)
└─ Tag: "Balanced approach"

Test 3: Aggressive (5-Stack)
├─ Generate 50 lineups
├─ Add all 50 to favorites (Run #3)
└─ Tag: "Aggressive approach"

Review Phase:
├─ Compare average scores by run
├─ Compare lineup diversity
├─ Identify best performing strategy
└─ Decision: Keep Run #2 (balanced), delete others

Final Export:
├─ Deselect Run #1 and #3 lineups
├─ Keep only Run #2 (50 lineups)
├─ Export to DraftKings
```

### Workflow 3: Incremental Refinement

**Scenario:** Start small, expand to target

```
Day Before Contest:
├─ Run 1: Generate 20 core lineups (Run #1)
├─ Review: Sleep on it
└─ Save: 20 favorites

Morning of Contest:
├─ Load favorites from yesterday
├─ Run 2: Generate 30 variations (Run #2)
├─ Add: Best 30 from new run
└─ Status: 50 favorites

Mid-Day Updates:
├─ News: Scratches/weather/lineups announced
├─ Filter: Remove affected lineups
├─ Run 3: Replace with 20 new (Run #3)
└─ Status: 50 favorites (refreshed)

Pre-Lock (1 hour before):
├─ Run 4: Generate 50 more (Run #4)
├─ Add: Top 25 from final run
├─ Review: All 75 lineups
└─ Export: Final 75 lineups
```

---

## Integration Features

### Player ID Mapping

**Challenge:** Favorites need DraftKings player IDs for export

**Solution:**
```python
# When adding to favorites, capture player data including IDs
def add_lineup_to_favorites(lineup_df):
    lineup_data = []
    for _, player in lineup_df.iterrows():
        player_data = {
            'Name': player['Name'],
            'Team': player['Team'],
            'Position': player['Position'],
            'Salary': player['Salary'],
            'Predicted_DK_Points': player['Predicted_DK_Points'],
            'ID': player.get('ID', None)  # Capture ID if available
        }
        lineup_data.append(player_data)
    
    favorites.append({
        'lineup_data': lineup_data,
        'total_points': lineup_df['Predicted_DK_Points'].sum(),
        'total_salary': lineup_df['Salary'].sum(),
        'run_number': current_run,
        'date_added': datetime.now().isoformat()
    })
```

### Contest Information Preservation

When loaded DK entries file exists:
```python
# Preserve contest metadata
contest_info = {
    'entry_id': '4766459650',
    'contest_name': 'MLB $3 Double Up',
    'contest_id': '162584429',
    'entry_fee': '$3'
}

# Apply to all favorites on export
for lineup in favorites:
    lineup['contest_info'] = contest_info
```

---

## Visual Enhancements

### Run Color Coding

```
Run #1: Blue background
Run #2: Green background
Run #3: Yellow background
Run #4: Orange background
Run #5: Purple background
...
```

### Lineup Grouping

```
┌────────────────────────────────────┐
│ ═══ Run 1 - 30 lineups ═══        │
│ Lineup 1 (125.3 pts, $49,800)     │
│   [Player rows...]                 │
│ Lineup 2 (124.8 pts, $49,500)     │
│   [Player rows...]                 │
│ ...                                │
│                                    │
│ ═══ Run 2 - 35 lineups ═══        │
│ Lineup 31 (126.1 pts, $49,900)    │
│   [Player rows...]                 │
│ ...                                │
└────────────────────────────────────┘
```

### Sort/Filter Options

```
┌──────────────────────────────────────┐
│ Sort by: [Points (High) ▼]          │
│   • Points (High to Low)             │
│   • Points (Low to High)             │
│   • Salary (High to Low)             │
│   • Salary (Low to High)             │
│   • Run Number                       │
│   • Date Added                       │
│                                      │
│ Filter: [All Runs ▼]                 │
│   • All Runs                         │
│   • Run 1 only                       │
│   • Run 2 only                       │
│   • ...                              │
└──────────────────────────────────────┘
```

---

## Best Practices

### For Cash Games
```
Strategy:
• Run optimization 2-3 times with different settings
• Add 15-20 lineups from each run
• Total target: 30-60 lineups
• Export all for multi-entry
```

### For GPP Tournaments
```
Strategy:
• Run optimization 5-10 times
• Vary stack sizes and team selections
• Add 20-30 from each run
• Total target: 100-300 lineups
• Review for diversity before export
```

### For Large-Field GPPs
```
Strategy:
• Run 10+ optimization sessions
• Use different algorithms each time
• Add 50-100 from each run
• Total target: 500+ lineups
• Export in batches to different contests
```

---

## Error Handling

### No Lineups in Current Pool
```
⚠ Warning: No lineups available

No optimized lineups in current pool.

Run optimization first, then add lineups
to favorites.

[Go to Run Optimization]
```

### Exceeds Contest Limit
```
⚠ Warning: Too many lineups

Attempting to export 500 lineups
Most contests limit to 150 entries

Options:
• Export in batches
• Select top 150 only
• Create multiple contest entries

[Export Top 150] [Export All] [Cancel]
```

### Missing Player IDs
```
⚠ Warning: Missing player IDs

Some lineups don't have DraftKings player IDs

This may occur if original CSV didn't include IDs

Export will use player names instead
May require manual ID entry in DraftKings

[Continue Anyway] [Cancel]
```

---

## Technical Implementation

### In-Memory Structure

```python
class FavoritesManager:
    def __init__(self):
        self.lineups = []
        self.current_run = 1
        self.favorites_file = "favorites_lineups.json"
    
    def add_lineups(self, lineup_list, count):
        """Add top N lineups from list"""
        sorted_lineups = sorted(
            lineup_list,
            key=lambda x: x['Predicted_DK_Points'].sum(),
            reverse=True
        )
        
        for lineup in sorted_lineups[:count]:
            self.lineups.append({
                'lineup': lineup,
                'run_number': self.current_run,
                'date_added': datetime.now()
            })
        
        self.current_run += 1
        self.save()
    
    def export_to_dk_format(self, output_path, count):
        """Export top N favorites to DK CSV"""
        # Implementation details...
    
    def save(self):
        """Persist to JSON"""
        # Implementation details...
    
    def load(self):
        """Load from JSON"""
        # Implementation details...
```

---

Next: [Control Panel](08_CONTROL_PANEL.md)

