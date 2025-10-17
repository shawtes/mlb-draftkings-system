# Players Tab Design
## Player Selection and Management Interface

---

## Purpose

The **Players Tab** is the primary interface for viewing, filtering, and selecting players for lineup optimization. Users can see all available players, their projections, and mark which players should be considered for lineups.

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│ Players Tab                                                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Position Sub-Tabs:                                      │ │
│  │  [All Batters] [C] [1B] [2B] [3B] [SS] [OF] [P]        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  [✓ Select All]  [✗ Deselect All]  Sorting: [Points ▼] │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Select │ Name        │ Team │ Pos │ Salary │ Proj │ Val │ │
│  ├────────┼─────────────┼──────┼─────┼────────┼──────┼─────┤ │
│  │  [✓]   │ Shohei      │ LAA  │ P   │ 11000  │ 25.3 │ 2.3││ │
│  │  [✓]   │ Aaron Judge │ NYY  │ OF  │  6500  │ 12.8 │ 2.0││ │
│  │  [ ]   │ Mike Trout  │ LAA  │ OF  │  5800  │ 10.5 │ 1.8││ │
│  │  [✓]   │ Mookie      │ LAD  │ OF  │  6200  │ 11.7 │ 1.9││ │
│  │  ...                                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Status: 45/187 players selected                              │
└────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Position Sub-Tabs

**Purpose:** Filter players by position for easier navigation

**Sub-Tabs:**
1. **All Batters** - Shows C, 1B, 2B, 3B, SS, OF (excludes pitchers)
2. **C** - Catchers only
3. **1B** - First basemen only
4. **2B** - Second basemen only
5. **3B** - Third basemen only
6. **SS** - Shortstops only
7. **OF** - Outfielders only
8. **P** - Pitchers only (SP and RP)

**Behavior:**
- Clicking a tab instantly shows players for that position
- Multi-position players (e.g., "SS/2B") appear in both relevant tabs
- Player selection persists across tab switches
- Tab badge shows count: `OF (42)` - 42 outfielders available

### 2. Action Toolbar

**Components:**

#### Select All Button
```
┌──────────────┐
│ ✓ Select All │
└──────────────┘
```
- **Function:** Checks all checkboxes in current position tab
- **Keyboard:** `Ctrl+A` (when table focused)
- **Visual Feedback:** All checkboxes become checked
- **Status Update:** "Selected all X players in [Position]"

#### Deselect All Button
```
┌────────────────┐
│ ✗ Deselect All │
└────────────────┘
```
- **Function:** Unchecks all checkboxes in current position tab
- **Keyboard:** `Ctrl+Shift+A`
- **Visual Feedback:** All checkboxes become unchecked
- **Status Update:** "Deselected all players in [Position]"

#### Sorting Dropdown
```
┌─────────────────┐
│ Sort by: [▼]    │
│  • Points (desc)│
│  • Value (desc) │
│  • Salary (desc)│
│  • Name (asc)   │
└─────────────────┘
```
- **Default:** Points (descending) - highest projected first
- **Persistence:** Sorting preference saved per position
- **Behavior:** Instant re-sort on selection

### 3. Player Table

**Columns:**

#### Column 1: Select Checkbox
- **Width:** 50px fixed
- **Purpose:** Mark player for inclusion in optimization
- **Interaction:** Click to toggle
- **Default State:** Unchecked
- **Visual:** Standard checkbox, centered

#### Column 2: Name
- **Width:** Auto (flexible, min 150px)
- **Content:** Full player name
- **Sorting:** Alphabetical A-Z or Z-A
- **Example:** "Shohei Ohtani", "Aaron Judge"
- **Tooltip:** Shows full name if truncated

#### Column 3: Team  
- **Width:** 60px fixed
- **Content:** Team abbreviation (3 letters)
- **Sorting:** Alphabetical
- **Example:** "NYY", "LAD", "SF"
- **Color Coding:** Optional team colors background

#### Column 4: Position
- **Width:** 60px fixed
- **Content:** DraftKings position(s)
- **Sorting:** Alphabetical
- **Example:** "OF", "SS/2B", "P"
- **Multi-Position:** Shows all eligible positions

#### Column 5: Salary
- **Width:** 80px fixed
- **Content:** DraftKings salary
- **Sorting:** Numeric high-to-low or low-to-high
- **Format:** "$X,XXX" (e.g., "$6,500")
- **Alignment:** Right-aligned

#### Column 6: Predicted_DK_Points
- **Width:** 80px fixed
- **Content:** Projected fantasy points
- **Sorting:** Numeric high-to-low (default)
- **Format:** "X.XX" (e.g., "12.75")
- **Alignment:** Right-aligned
- **Color Coding:**
  - Green: Top 25% projections
  - Yellow: Middle 50%
  - Default: Bottom 25%

#### Column 7: Value
- **Width:** 70px fixed
- **Content:** Points per $1000 salary
- **Calculation:** `Predicted_DK_Points / (Salary / 1000)`
- **Format:** "X.XX" (e.g., "2.35")
- **Sorting:** Numeric high-to-low
- **Alignment:** Right-aligned
- **Color Coding:**
  - Green: Value > 2.5
  - Yellow: Value 2.0-2.5
  - Default: Value < 2.0

#### Column 8: Min Exp (Spinbox)
- **Width:** 80px fixed
- **Content:** Minimum exposure percentage (0-100)
- **Default:** 0
- **Interaction:** Click to edit, type number, or use spinbox arrows
- **Validation:** Integer 0-100
- **Purpose:** Force player in at least X% of lineups

#### Column 9: Max Exp (Spinbox)
- **Width:** 80px fixed
- **Content:** Maximum exposure percentage (0-100)
- **Default:** 100
- **Interaction:** Same as Min Exp
- **Validation:** Integer 0-100, must be ≥ Min Exp
- **Purpose:** Cap player in at most X% of lineups

#### Column 10: Actual Exp (%)
- **Width:** 90px fixed
- **Content:** Actual exposure after optimization
- **Format:** "X.XX%" (e.g., "45.23%")
- **Calculation:** `(times_used / total_lineups) * 100`
- **State:** Read-only, populated after optimization
- **Color Coding:**
  - Red: Exceeds Max Exp
  - Green: Within Min/Max range
  - Yellow: Below Min Exp

---

## Interactions

### Row Selection
- **Click Checkbox:** Toggle player selection
- **Click Row:** Select row (highlight), does NOT toggle checkbox
- **Double-Click Name:** Opens player details dialog (future)
- **Right-Click Row:** Context menu (future)
  - "Lock in all lineups"
  - "Exclude completely"
  - "View player history"

### Keyboard Navigation
- **Tab:** Move between cells
- **Space:** Toggle checkbox when focused
- **Enter:** Edit spinbox value
- **Arrow Keys:** Navigate cells
- **Ctrl+Click:** Multi-row selection

### Sorting
- **Click Column Header:** Toggle sort direction
- **Shift+Click:** Secondary sort
- **Visual Indicator:** Arrow (▲ ascending, ▼ descending)

---

## Data Validation

### Input Validation

#### Min Exposure
```python
if min_exp < 0:
    min_exp = 0
    show_warning("Minimum exposure cannot be negative")
elif min_exp > 100:
    min_exp = 100
    show_warning("Minimum exposure cannot exceed 100%")
elif min_exp > max_exp:
    show_warning("Minimum exposure cannot exceed maximum exposure")
    # Auto-adjust max_exp or reject change
```

#### Max Exposure
```python
if max_exp < 0:
    max_exp = 0
elif max_exp > 100:
    max_exp = 100
elif max_exp < min_exp:
    show_warning("Maximum exposure cannot be less than minimum exposure")
```

### Data Integrity Checks

**On CSV Load:**
```python
# Required columns check
required = ['Name', 'Team', 'Position', 'Salary', 'Predicted_DK_Points']
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Data type validation
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df['Predicted_DK_Points'] = pd.to_numeric(df['Predicted_DK_Points'], errors='coerce')

# Remove invalid rows
df = df.dropna(subset=['Name', 'Salary', 'Predicted_DK_Points'])
df = df[df['Salary'] > 0]
df = df[df['Predicted_DK_Points'] > 0]
```

---

## Special Features

### 1. Probability Columns Detection

If CSV contains probability columns (`Prob_Over_5`, `Prob_Over_10`, etc.), display additional columns:

#### Column: Expected Utility
- **Width:** 90px
- **Content:** Calculated expected utility value
- **Formula:** Based on probability distribution
- **Format:** "X.XXX"
- **Purpose:** Advanced optimization metric

#### Column: Risk-Adjusted Points
- **Width:** 100px
- **Content:** Points adjusted for variance
- **Formula:** `points - (volatility_penalty * variance)`
- **Format:** "X.XX"

**Indicator:**
```
┌──────────────────────────────────────┐
│ ✓ Probability data detected         │
│ Enhanced metrics enabled             │
└──────────────────────────────────────┘
```

### 2. Multi-Position Players

Players eligible for multiple positions:
- **Display:** Show all positions (e.g., "SS/2B")
- **Availability:** Appear in multiple position tabs
- **Selection:** Selecting in one tab selects in all
- **Optimization:** Optimizer can use in any eligible position

### 3. Player Search (Future Enhancement)

```
┌─────────────────────────────────┐
│ 🔍 Search: [Judge________]      │
└─────────────────────────────────┘
```
- **Function:** Filter table to matching players
- **Matching:** Name, team, or position
- **Real-time:** Updates as you type
- **Clear:** X button to reset search

---

## Status Display

### Selection Counter
```
Status: 45/187 players selected (24%)
```

**Information Shown:**
- **Selected Count:** Number of checked players
- **Total Count:** Total players in current view
- **Percentage:** Selection rate
- **Warning:** If < 30 players selected, show warning

### Warnings

#### Insufficient Players
```
⚠ Warning: Only 15 players selected across all positions
   At least 30 recommended for diverse lineups
```

#### No Pitchers Selected
```
⚠ Error: No pitchers selected
   Must select at least 2 pitchers
```

#### Position Gaps
```
⚠ Warning: Only 2 catchers selected
   Recommended: At least 3 for backup options
```

---

## Performance Optimization

### Large Player Pools (200+ players)

1. **Virtual Scrolling:** Only render visible rows
2. **Lazy Loading:** Load position tabs on demand
3. **Debounced Sorting:** Wait 300ms after sort before rendering
4. **Checkbox State Cache:** Store in hash map for O(1) lookup

```python
# Efficient checkbox tracking
checkbox_states = {
    'Shohei Ohtani': True,
    'Aaron Judge': True,
    'Mike Trout': False,
    # ... O(1) lookup
}
```

---

## Mobile/Responsive Considerations (Future)

### Tablet View (1024px width)
- **Hide columns:** Min Exp, Max Exp (move to detail view)
- **Narrower columns:** Reduce padding
- **Touch targets:** Increase checkbox hit area to 44x44px

### Mobile View (< 768px)
- **Card Layout:** Switch from table to card list
- **Essential Info Only:** Name, Position, Salary, Points
- **Tap to Expand:** Full details on tap

---

## Accessibility

### Screen Reader Support
```html
<checkbox aria-label="Select Aaron Judge, Outfielder, New York Yankees, $6500, 12.8 projected points">
```

### Color Blind Mode
- **Alternative:** Patterns instead of colors only
- **High Contrast:** WCAG AA compliant
- **Focus Indicators:** 2px blue outline on focus

---

## Error States

### No Data Loaded
```
┌─────────────────────────────────┐
│                                 │
│   No player data loaded         │
│                                 │
│   Load a CSV file to begin  →  │
│         [Load CSV Button]       │
│                                 │
└─────────────────────────────────┘
```

### Load Error
```
┌─────────────────────────────────┐
│ ⚠ Error loading player data     │
│                                 │
│ Missing required columns:       │
│ • Predicted_DK_Points           │
│                                 │
│ Please check your CSV format    │
│         [Try Again]             │
└─────────────────────────────────┘
```

### Invalid Data
```
┌─────────────────────────────────┐
│ ⚠ Warning: Data issues found    │
│                                 │
│ Removed 5 players:              │
│ • 3 with invalid salaries       │
│ • 2 with missing projections    │
│                                 │
│ Loaded: 182 valid players       │
│          [OK]                   │
└─────────────────────────────────┘
```

---

## Best Practices for Users

### Player Selection Strategy

**For Cash Games (50/50, Double-Up):**
```
✓ Select 30-50 players
✓ Focus on high-floor projections
✓ Avoid very cheap or very expensive outliers
✓ Use Max Exposure to cap variance
```

**For GPPs (Tournaments):**
```
✓ Select 50-100+ players
✓ Include high-upside chalky plays
✓ Include low-owned contrarians
✓ Use Min Exposure on key stack pieces
```

### Exposure Management

**Example Setup:**
```
| Player      | Position | Salary | Proj | Min | Max | Strategy |
|-------------|----------|--------|------|-----|-----|----------|
| Ace Pitcher | P        | 11000  | 25   | 30  | 50  | Core play|
| Chalk Batter| OF       | 6500   | 13   | 0   | 40  | Cap chalk|
| Value Play  | 2B       | 4200   | 8    | 10  | 100 | Ensure   |
| Contrarian  | SS       | 5800   | 11   | 0   | 15  | Sprinkle |
```

---

## Integration with Other Tabs

### Data Flow Out
```
Players Tab Selection
        ↓
[Optimization Worker]
        ↓
    Results Display
```

### Synchronized State
- **Team Stacks Tab:** Only shows teams with selected players
- **Stack Exposure:** Uses only positions with selected players
- **Control Panel:** Updates "X players selected" indicator

---

## Future Enhancements

1. **Advanced Filters:**
   - Filter by salary range
   - Filter by projection range
   - Filter by opponent
   - Filter by home/away

2. **Batch Operations:**
   - Select all above $6000
   - Select top 20 by value
   - Randomize exposures

3. **Player Details Modal:**
   - Recent game log
   - Opponent stats
   - Weather conditions
   - Vegas betting lines

4. **Import/Export Selections:**
   - Save selection presets
   - Share configurations
   - Load from previous sessions

---

## Technical Implementation Notes

### Data Structure
```python
class PlayerTableModel:
    def __init__(self, df_players):
        self.df = df_players
        self.checkbox_states = {}
        self.exposure_settings = {}
        
    def get_selected_players(self):
        return [name for name, checked in self.checkbox_states.items() if checked]
    
    def get_player_count_by_position(self, position):
        return len(self.df[self.df['Position'].str.contains(position)])
```

### Performance Targets
- **Initial Load:** < 1 second for 200 players
- **Tab Switch:** < 100ms
- **Sort Operation:** < 200ms
- **Checkbox Toggle:** < 16ms (60fps)
- **Memory:** < 50MB for table data

---

Next: [Team Stacks Tab](03_TEAM_STACKS_TAB.md)

