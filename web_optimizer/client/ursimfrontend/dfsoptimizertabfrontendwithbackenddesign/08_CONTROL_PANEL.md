# Control Panel Design
## Right-Side Optimization Control Center

---

## Purpose

The **Control Panel** is the primary action center of the application, located on the right side of the interface (30% of screen width). It contains:
- File operations (load/save)
- Optimization settings
- Risk management controls
- Action buttons
- Results display

**Always Visible:** Unlike tabs that switch content, the control panel remains constant and accessible.

---

## Layout Structure

```
┌────────────────────────────────┐
│ CONTROL PANEL                  │  ← 30% of window width
├────────────────────────────────┤
│                                │
│ FILE OPERATIONS                │
│ ┌────────────────────────────┐ │
│ │ [Load CSV]                 │ │
│ │ [Load DK Predictions]      │ │
│ │ [Load DK Entries]          │ │
│ └────────────────────────────┘ │
│                                │
│ OPTIMIZATION SETTINGS          │
│ ┌────────────────────────────┐ │
│ │ Min Unique: [3________]    │ │
│ │ # Lineups:  [100______]    │ │
│ │ □ Disable Kelly Sizing     │ │
│ └────────────────────────────┘ │
│                                │
│ SALARY CONSTRAINTS             │
│ ┌────────────────────────────┐ │
│ │ Min Salary: [$45000___]    │ │
│ │ Max Salary: [$50000___]    │ │
│ └────────────────────────────┘ │
│                                │
│ SORTING                        │
│ ┌────────────────────────────┐ │
│ │ Sort: [Points ▼]           │ │
│ └────────────────────────────┘ │
│                                │
│ PROBABILITY METRICS            │
│ ┌────────────────────────────┐ │
│ │ ○ No prob data             │ │
│ └────────────────────────────┘ │
│                                │
│ RISK MANAGEMENT                │
│ ┌────────────────────────────┐ │
│ │ Bankroll: [$1000______]    │ │
│ │ Risk: [Medium ▼]           │ │
│ │ ☑ Enable Risk Mgmt         │ │
│ └────────────────────────────┘ │
│                                │
│ ACTIONS                        │
│ ┌────────────────────────────┐ │
│ │ [Run Contest Sim]          │ │
│ │ [Save CSV for DK]          │ │
│ │ [Load DK Entries File]     │ │
│ │ [Fill Entries w/Lineups]   │ │
│ └────────────────────────────┘ │
│                                │
│ FAVORITES                      │
│ ┌────────────────────────────┐ │
│ │ [➕ Add to Favorites]       │ │
│ │ [💾 Export Favorites]       │ │
│ └────────────────────────────┘ │
│                                │
│ RESULTS TABLE                  │
│ ┌────────────────────────────┐ │
│ │Player  │Pos│Sal│Pts│Exp%  ││ │
│ ├────────┼───┼───┼───┼──────┤│ │
│ │Shohei  │P  │11K│25 │45%   ││ │
│ │Judge   │OF │6.5│13 │52%   ││ │
│ │Mookie  │OF │6.2│12 │38%   ││ │
│ │...                          │ │
│ │                             │ │
│ │ [Scrollable]                │ │
│ └────────────────────────────┘ │
│                                │
│ STATUS BAR                     │
│ ┌────────────────────────────┐ │
│ │ Ready | Players: 187 |     │ │
│ │ Lineups: 0                 │ │
│ └────────────────────────────┘ │
└────────────────────────────────┘
```

---

## Section 1: File Operations

### Load CSV Button
```
┌──────────────────┐
│ Load CSV         │
└──────────────────┘
```

**Function:** Load player projections from CSV file

**Behavior:**
1. Opens file dialog
2. User selects CSV file
3. Validates required columns
4. Populates player tables
5. Updates status

**Required Columns:**
- `Name` - Player full name
- `Team` - Team abbreviation
- `Position` - DK position
- `Salary` - Integer salary
- `Predicted_DK_Points` or `My_Proj` - Projections

**Optional Columns:**
- `ID` or `player_id` - DraftKings player ID
- `Prob_Over_X` - Probability columns
- `Value` - Pre-calculated value
- Custom metric columns

**Success:**
```
Status: "Players loaded: 187 players"
Players Tab: Populated with data
Team Stacks: Auto-populated with teams
```

**Error Handling:**
```
Missing columns: "Missing required column: Predicted_DK_Points"
Invalid data: "Removed 5 players with invalid data"
Empty file: "No valid players found in CSV"
```

### Load DK Predictions Button
```
┌──────────────────┐
│ Load DK          │
│ Predictions      │
└──────────────────┘
```

**Function:** Same as Load CSV, but specifically for DraftKings format

**Special Handling:**
- Recognizes DK-specific column names
- Auto-maps DK position format
- Extracts player IDs from "Name (ID)" format

### Load DK Entries Button
```
┌──────────────────┐
│ Load Entries CSV │
└──────────────────┘
```

**Function:** Load DraftKings entries file for filling

**File Format:**
```
Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
4766459650, MLB Main Slate, 162584429, $1, , , , , , , , , , 
```

**Purpose:**
- Reserve entry slots
- Preserve contest information
- Fill with optimized lineups later

**Success:**
```
Status: "DK Entries loaded: 150 entries from DKEntries_162584429.csv"
Dialog: Shows entry count, contest name, entry fee
```

---

## Section 2: Optimization Settings

### Min Unique Input
```
┌──────────────────────────────┐
│ Min Unique:                  │
│ [3_________]  (0-10)         │
└──────────────────────────────┘
```

**Purpose:** Minimum number of different players between any two lineups

**Values:**
- **0:** No constraint (maximum speed)
- **1-3:** Light diversity (recommended)
- **4-6:** Moderate diversity
- **7-10:** High diversity (may limit results)

**Impact:**
```
Min Unique = 3:
  Lineup 1: [A,B,C,D,E,F,G,H,I,J]
  Lineup 2: Must differ by ≥3 players
           [A,B,C,D,E,F,G,X,Y,Z] ✓ (3 different)
           [A,B,C,D,E,F,G,H,I,Z] ✗ (only 1 different)
```

**Tooltip:**
```
Higher values create more diverse lineups but may:
• Reduce total lineups generated
• Increase optimization time
• Exclude optimal combinations
```

### Number of Lineups Input
```
┌──────────────────────────────┐
│ Number of Lineups:           │
│ [100_______]  (1-500)        │
└──────────────────────────────┘
```

**Purpose:** How many lineups to generate

**Ranges:**
- **1-20:** Single entry or small contests
- **20-50:** Cash games, small GPPs
- **50-150:** Standard GPP multi-entry
- **150-500:** Large-field GPPs, mass multi-entry

**Performance:**
- 1-50 lineups: < 30 seconds
- 100 lineups: 30-60 seconds
- 500 lineups: 2-5 minutes

**Tooltip:**
```
Recommended:
• Cash games: 20-50
• GPPs: 100-200
• Large field: 150-500
```

### Disable Kelly Sizing Checkbox
```
┌──────────────────────────────┐
│ □ Disable Kelly Sizing       │
│   (Generate All Requested)   │
└──────────────────────────────┘
```

**Purpose:** Override Kelly Criterion risk management

**When Checked:**
- Generates exactly the requested number of lineups
- Ignores bankroll management recommendations
- Useful for multi-entry contests with required lineup counts

**When Unchecked:**
- Kelly Criterion may reduce lineup count
- Optimizes for bankroll safety
- May generate fewer lineups than requested

**Use Cases:**
```
Check this when:
✓ Contest requires exact lineup count
✓ You know you want 150 lineups
✓ You're confident in your projections
✓ You understand the risks

Uncheck when:
✓ You want optimal bankroll management
✓ You're uncertain about projections
✓ You prefer conservative approach
✓ You're new to advanced optimization
```

---

## Section 3: Salary Constraints

### Minimum Salary Input
```
┌──────────────────────────────┐
│ 💰 Minimum Salary:           │
│ [$45000____]  ($0-$50000)    │
└──────────────────────────────┘
```

**Purpose:** Prevent "too cheap" lineups

**Default:** $45,000 (90% of cap)

**Rationale:**
```
Minimum salary ensures:
• No lineup is too cheap
• Utilizes available budget
• Prevents value trap lineups
• Forces use of quality players
```

**Common Settings:**
- **$43,000-$45,000:** Cash games (use budget)
- **$40,000-$43,000:** GPPs (some punts okay)
- **$35,000-$40,000:** Ultra-contrarian (risky)
- **$0:** No minimum (not recommended)

### Maximum Salary Input
```
┌──────────────────────────────┐
│ Maximum Salary:              │
│ [$50000____]  (Fixed)        │
└──────────────────────────────┘
```

**Purpose:** Salary cap (DraftKings MLB = $50,000)

**Fixed Value:** Always $50,000 for MLB
**Not Editable:** Enforced by DraftKings rules

**Display:** Grayed out, shows only for completeness

---

## Section 4: Sorting

### Sorting Dropdown
```
┌──────────────────────────────┐
│ Sorting Method:              │
│ [Points ▼]                   │
│  • Points (desc)             │
│  • Value (desc)              │
│  • Salary (desc)             │
└──────────────────────────────┘
```

**Purpose:** Sort players in tables

**Options:**

1. **Points (Descending)** - Default
   - Highest projected points first
   - Best for cash games
   
2. **Value (Descending)**
   - Best points per $1000 first
   - Good for budget optimization
   
3. **Salary (Descending)**
   - Most expensive first
   - Useful for lineup building strategy

**Scope:** Applies to all position tables in Players Tab

---

## Section 5: Probability Metrics

```
┌──────────────────────────────┐
│ 🎲 Probability Metrics:      │
│                              │
│ Status: ○ No data loaded     │
│                              │
│ [Summary area when loaded]   │
└──────────────────────────────┘
```

**Display When No Data:**
```
○ No probability data loaded

To use probability-based optimization,
load CSV with columns like:
• Prob_Over_5
• Prob_Over_10
• Prob_Over_15
```

**Display When Data Loaded:**
```
✓ Probability data detected

Enhanced metrics enabled:
• Expected_Utility: 187 players
• Risk_Adjusted_Points: 187 players
• Kelly_Fraction: 187 players

Contest Strategy: GPP
```

**Interactive:** Click to see detailed probability summary

---

## Section 6: Risk Management

**Available When:** Risk engine libraries installed

### Bankroll Input
```
┌──────────────────────────────┐
│ Bankroll ($):                │
│ [$1000_____]  ($100-$100K)   │
└──────────────────────────────┘
```

**Purpose:** Total bankroll for Kelly sizing and position limits

**Recommended:**
- Enter your actual DFS bankroll
- Used for Kelly Criterion calculations
- Determines recommended lineup counts
- Affects position sizing

**Example:**
```
Bankroll: $1,000
Kelly suggests: 150 lineups
@ $1 each = $150 total (15% of bankroll)
```

### Risk Profile Dropdown
```
┌──────────────────────────────┐
│ Risk Profile:                │
│ [Medium ▼]                   │
│  • Conservative              │
│  • Medium                    │
│  • Aggressive                │
└──────────────────────────────┘
```

**Profiles:**

**Conservative:**
- Focus on Sharpe ratio
- Minimize volatility
- Prefer cash games
- Lower exposure to variance

**Medium (Default):**
- Balanced approach
- Moderate risk/reward
- Good for most contests
- Standard optimization

**Aggressive:**
- Higher risk tolerance
- Accept more volatility
- GPP focused
- Max leverage plays

### Enable Risk Management Checkbox
```
┌──────────────────────────────┐
│ ☑ Enable Advanced Risk       │
│   Management                 │
└──────────────────────────────┘
```

**When Checked:**
- Uses Kelly Criterion
- Applies GARCH volatility models
- Portfolio theory optimization
- Risk metrics displayed

**When Unchecked:**
- Standard optimization only
- Faster performance
- No advanced risk calculations

---

## Section 7: Actions

### Run Contest Sim Button
```
┌──────────────────────────────┐
│ [Run Contest Sim]            │
└──────────────────────────────┘
```

**Primary Action:** Starts optimization

**Pre-Flight Checks:**
1. Player data loaded? ✓
2. At least one stack type selected? ✓
3. Sufficient players for positions? ✓
4. Valid settings? ✓

**States:**

**Ready State:**
```
[Run Contest Sim]
```

**Running State:**
```
[⏳ Optimizing... Cancel]
```

**Complete State:**
```
[Run Contest Sim]
Status: "✅ Generated 100 lineups"
```

**Click Behavior:**
1. Collects all settings
2. Validates configuration
3. Starts OptimizationWorker thread
4. Shows progress
5. Displays results when complete

### Save CSV for DK Button
```
┌──────────────────────────────┐
│ [Save CSV for DraftKings]   │
└──────────────────────────────┘
```

**Function:** Export optimized lineups

**Requirements:**
- Optimization must be complete
- At least 1 lineup generated

**Output Format:**
```
P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
Shohei Ohtani, Sandy Alcantara, Will Smith, ...
Aaron Judge, Corbin Burnes, Sean Murphy, ...
...
```

**Save Dialog:**
```
┌──────────────────────────────────┐
│ Save Optimized Lineups           │
├──────────────────────────────────┤
│ Save as: [optimized_lineups.csv] │
│                                  │
│ Format: Simple (Player names)    │
│                                  │
│ [Cancel]  [Save]                 │
└──────────────────────────────────┘
```

### Load DK Entries File Button
```
┌──────────────────────────────┐
│ [Load DK Entries File]       │
└──────────────────────────────┘
```

**Function:** Load empty DK contest entries

**Purpose:**
- Reserve entry IDs
- Preserve contest metadata
- Ready for filling with lineups

**Result:**
```
Status: "DK Entries loaded: 150 entries"
Enable: "Fill Entries" button
```

### Fill Entries with Lineups Button
```
┌──────────────────────────────┐
│ [Fill Entries w/ Lineups]    │
└──────────────────────────────┘
```

**Function:** Fill loaded entries with optimized lineups

**Requirements:**
- DK entries file loaded ✓
- Optimized lineups available ✓
- Player IDs available (preferred)

**Dialog:**
```
┌──────────────────────────────────┐
│ Fill DK Entries                  │
├──────────────────────────────────┤
│ Available lineups: 100           │
│ Reserved entries: 150            │
│                                  │
│ Fill how many? [100_____]        │
│                                  │
│ [Cancel]  [Fill Entries]         │
└──────────────────────────────────┘
```

**Output:**
```
Entry ID, Contest, ID, Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
4766459650, MLB Main, 162584429, $1, 12345678, 23456789, ...
4766459651, MLB Main, 162584429, $1, 12345678, 23456780, ...
...
```

---

## Section 8: Favorites Management

### Add to Favorites Button
```
┌──────────────────────────────┐
│ [➕ Add Current to           │
│     Favorites]               │
└──────────────────────────────┘
```

**Function:** Add current lineups to favorites collection

**Behavior:**
1. Opens dialog to select how many
2. Adds top N lineups by points
3. Tags with run number
4. Saves to persistent storage

**Result:** Lineups added to My Entries tab

### Export Favorites Button
```
┌──────────────────────────────┐
│ [💾 Export Favorites as      │
│     New Lineups]             │
└──────────────────────────────┘
```

**Function:** Export favorites in DK format

**Behavior:**
1. Opens export dialog
2. Select how many to export
3. Exports to DK contest format
4. Ready for upload

---

## Section 9: Results Table

```
┌────────────────────────────────┐
│ RESULTS TABLE                  │
├────┬────┬────┬────┬────┬──────┤
│Plr │Pos │Sal │Pts │Tot │Exp % │
├────┼────┼────┼────┼────┼──────┤
│Sho │P   │11K │25.3│125 │45.2% │
│Jud │OF  │6.5K│12.8│125 │52.1% │
│Moo │OF  │6.2K│11.7│125 │38.4% │
│...                              │
│ [Scrollable Area]              │
└────────────────────────────────┘
```

**Purpose:** Quick view of generated lineups

**Columns:**
- **Player:** Abbreviated name
- **Pos:** Position
- **Sal:** Salary (abbreviated: "11K" = $11,000)
- **Pts:** Projected points
- **Tot:** Lineup total points (repeats for all 10 players)
- **Exp %:** Player exposure across all lineups

**Features:**
- Scrollable
- Color-coded by lineup
- Click row to see full lineup
- Right-click for actions

**Condensed Display:**
Shows only essential info due to limited width

---

## Section 10: Status Bar

```
┌────────────────────────────────┐
│ Status: Ready                  │
│ Players: 187 | Lineups: 0     │
│ Last Run: Never                │
└────────────────────────────────┘
```

**Information:**
- **Status:** Current application state
- **Players:** Count of loaded players
- **Lineups:** Count of generated lineups
- **Last Run:** Time of last optimization

**Status States:**
- "Ready" - Waiting for action
- "Loading..." - Reading CSV
- "Optimizing..." - Running algorithm
- "Complete" - Optimization finished
- "Error: [message]" - Something went wrong

---

## Responsive Behavior

### Window Resize
- **Splitter:** Adjustable ratio (default 70/30)
- **Min Width:** 300px (panel collapses below this)
- **Scroll:** Vertical scroll if content exceeds height

### Collapsed State (< 400px width)
```
┌──────┐
│ ≡    │ ← Hamburger menu
├──────┤
│ [Fn] │ ← Collapsed buttons
│ [Fn] │
│ [Fn] │
└──────┘
```

---

## Keyboard Shortcuts

```
Ctrl+O    - Load CSV
Ctrl+R    - Run Optimization
Ctrl+S    - Save Results
Ctrl+E    - Export Favorites
F5        - Refresh All
Esc       - Cancel Optimization
```

---

## Best Practices Display

**Tooltips show recommendations:**

```
Min Unique = 3 (hovering)
────────────────────────────
Recommended: 2-4 for most uses

Higher values:
✓ More diverse lineups
✓ Better for large fields
✗ May limit total count
✗ Slower optimization
────────────────────────────
```

---

Next: [Data Flow & Interactions](09_DATA_FLOW.md)

