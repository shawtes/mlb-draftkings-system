# Team Stacks Tab Design
## Team Selection and Stacking Strategy Interface

---

## Purpose

The **Team Stacks Tab** allows users to select which teams should be used for stacking strategies (multiple players from the same team). This is crucial for correlating player performances in DFS contests.

---

## Layout Structure

```
┌──────────────────────────────────────────────────────────────────┐
│ Team Stacks Tab                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Stack Size Sub-Tabs:                                      │ │
│  │  [All Stacks] [2 Stack] [3 Stack] [4 Stack] [5 Stack]    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  [✓ Select All]  [✗ Deselect All]  [🔍 Test Detection]   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Select│Teams│Status │Time  │Proj Runs│Min Exp│Max Exp│Act%││
│  ├───────┼─────┼───────┼──────┼─────────┼───────┼───────┼────┤│
│  │  [✓]  │ NYY │Active │7:05PM│  5.2    │  0    │ 100   │ 0  ││
│  │  [✓]  │ LAD │Active │7:10PM│  4.8    │  0    │ 100   │ 0  ││
│  │  [ ]  │ BAL │Active │7:05PM│  4.5    │  0    │ 100   │ 0  ││
│  │  [✓]  │ ATL │Active │7:20PM│  5.0    │  0    │ 100   │ 0  ││
│  │  ...                                                        ││
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [Refresh Team Stacks]           Selected: NYY, LAD, ATL (3/30)│
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Stack Size Sub-Tabs

**Purpose:** Configure team selections for different stack sizes

**Sub-Tabs:**

#### All Stacks Tab
- Shows all available teams
- Selections apply to ALL stack sizes
- Use for "I want these teams in any stack size"
- **Warning indicator** if this conflicts with specific tabs

#### 2 Stack Tab
- Configure teams eligible for 2-player stacks
- Typically used for pitcher + catcher combos
- Lower correlation, lower ownership

#### 3 Stack Tab  
- Configure teams for 3-player stacks
- Standard balanced approach
- Good for cash games

#### 4 Stack Tab
- Configure teams for 4-player stacks
- High correlation strategy
- Popular for GPP tournaments

#### 5 Stack Tab
- Configure teams for 5-player stacks (maximum)
- Highest correlation, highest risk
- GPP leverage plays

**Tab Badges:**
```
[All Stacks (8)] [2 Stack (15)] [3 Stack (12)] [4 Stack (6)] [5 Stack (3)]
     ↑                ↑               ↑              ↑             ↑
  Selected       Selected        Selected       Selected      Selected
   in all         for 2s          for 3s         for 4s        for 5s
```

### 2. Action Toolbar

#### Select All Button
```
┌──────────────┐
│ ✓ Select All │  
└──────────────┘
```
- **Function:** Select all teams in current stack size tab
- **Behavior:** Checks all checkboxes
- **Use Case:** "I'll stack any team at this size"

#### Deselect All Button
```
┌────────────────┐
│ ✗ Deselect All │
└────────────────┘
```
- **Function:** Deselect all teams in current tab
- **Behavior:** Unchecks all checkboxes
- **Use Case:** Start fresh with selections

#### Test Detection Button
```
┌──────────────────────┐
│ 🔍 Test Detection    │
└──────────────────────┘
```
- **Function:** Debug team selection collection
- **Output:** Console log showing exactly what was detected
- **Purpose:** Troubleshooting checkbox issues
- **Display:**
```
===== TEAM SELECTION DEBUG =====
✓ Found team selections:
  All Stacks: ['NYY', 'LAD', 'ATL', 'SF', 'CHC', 'BOS', 'HOU', 'TB']
  4-Stack: ['NYY', 'LAD']
  3-Stack: ['ATL', 'SF', 'CHC']
================================
```

### 3. Team Stack Table

**Columns:**

#### Column 1: Select Checkbox
- **Width:** 50px
- **Purpose:** Mark team for this stack size
- **Behavior:** Click to toggle
- **State Management:** Independent per stack size tab

#### Column 2: Teams
- **Width:** 80px
- **Content:** Team abbreviation (3-4 letters)
- **Sorting:** Alphabetical
- **Example:** "NYY", "LAD", "ATL"
- **Visual:** Bold team name

#### Column 3: Status
- **Width:** 80px
- **Content:** Team game status
- **Values:**
  - "Active" - Normal game
  - "Postponed" - Game delayed/cancelled
  - "Final" - Game completed (shouldn't appear in active slate)
- **Color:**
  - Green: Active
  - Red: Postponed
  - Gray: Final

#### Column 4: Time
- **Width:** 70px
- **Content:** Game start time (local)
- **Format:** "H:MM AM/PM" (e.g., "7:05 PM")
- **Purpose:** Identify early/late games
- **Sort:** Chronological

#### Column 5: Proj Runs
- **Width:** 90px
- **Content:** Team's projected run total
- **Calculation:** Sum of team's projected DK points / 10
- **Format:** "X.X" (e.g., "5.2")
- **Purpose:** Identify high-scoring team games
- **Color Coding:**
  - Green: > 5.0 runs (high)
  - Yellow: 4.0-5.0 runs (medium)
  - Default: < 4.0 runs (low)

#### Column 6: Min Exp (Spinbox)
- **Width:** 80px
- **Content:** Minimum exposure for this team (0-100%)
- **Default:** 0
- **Purpose:** Ensure team appears in at least X% of lineups
- **Validation:** Integer 0-100

#### Column 7: Max Exp (Spinbox)
- **Width:** 80px
- **Content:** Maximum exposure for this team (0-100%)
- **Default:** 100
- **Purpose:** Cap team at most X% of lineups
- **Validation:** Integer 0-100, ≥ Min Exp

#### Column 8: Actual Exp (%)
- **Width:** 80px
- **Content:** Actual team exposure after optimization
- **Format:** "XX.X%" (e.g., "45.2%")
- **State:** Read-only, populated after optimization
- **Calculation:** `(lineups_with_team / total_lineups) * 100`

---

## Stack Size Logic

### How Stack Selections Work

**Example Configuration:**
```
All Stacks Tab:    [✓] NYY, LAD, ATL, SF, CHC
4 Stack Tab:       [✓] NYY, LAD
3 Stack Tab:       [✓] ATL, SF
2 Stack Tab:       [ ] (none selected)
```

**Resulting Behavior:**
- **5-Stack constraint:** Use any of {NYY, LAD, ATL, SF, CHC}
- **4-Stack constraint:** Use ONLY {NYY, LAD}
- **3-Stack constraint:** Use ONLY {ATL, SF}  
- **2-Stack constraint:** Use any of {NYY, LAD, ATL, SF, CHC} (falls back to All Stacks)

### Priority System
```
Specific Tab Selection > All Stacks Selection > No Constraint
```

**Logic Flow:**
```python
def get_teams_for_stack_size(stack_size):
    # Priority 1: Specific stack size tab
    if stack_size in specific_selections:
        return specific_selections[stack_size]
    
    # Priority 2: All Stacks tab
    elif "all" in all_stacks_selection:
        return all_stacks_selection["all"]
    
    # Priority 3: No restriction (use all teams)
    else:
        return all_available_teams
```

---

## Team Data Population

### Automatic Population

**When player CSV is loaded:**
```python
1. Extract unique teams from df_players['Team']
2. Calculate team metrics:
   - Player count by team
   - Average projected points
   - Projected team total
3. Populate all stack size tabs with team rows
4. Default: No teams selected (all unchecked)
```

**Derived Metrics:**
```python
for team in teams:
    team_players = df_players[df_players['Team'] == team]
    
    metrics = {
        'player_count': len(team_players),
        'avg_salary': team_players['Salary'].mean(),
        'total_proj': team_players['Predicted_DK_Points'].sum(),
        'proj_runs': total_proj / 10,  # Rough estimation
        'batters_only': len(team_players[team_players['Position'] != 'P'])
    }
```

### Validation

**Sufficient Batters Check:**
```python
def validate_team_for_stack(team, stack_size):
    """Check if team has enough non-pitchers for stack"""
    batters = df_players[
        (df_players['Team'] == team) & 
        (~df_players['Position'].str.contains('P'))
    ]
    
    if len(batters) < stack_size:
        return False, f"{team} has only {len(batters)} batters (need {stack_size})"
    
    return True, "OK"
```

**Warning Display:**
```
⚠ Warning: CHC selected for 5-Stack
   Only 4 batters available
   This team will be skipped
```

---

## Interactions

### Team Selection Flow

**User Story:** "I want to create 4-stacks with NYY and LAD only"

```
1. User clicks "4 Stack" tab
2. User clicks "Deselect All" (clear any defaults)
3. User checks [✓] NYY
4. User checks [✓] LAD
5. User clicks "Test Detection" to verify
6. Console shows: "4-Stack: ['NYY', 'LAD']"
7. User proceeds to run optimization
```

### Multi-Stack Strategy

**User Story:** "I want 4-stacks from top teams, 3-stacks from medium teams"

```
Configuration:
  4 Stack Tab: [✓] NYY, LAD
  3 Stack Tab: [✓] ATL, SF, CHC
  
  Stack Exposure Tab:
  [✓] 4-Stack (50% of lineups)
  [✓] 3-Stack (50% of lineups)
  
Result: Optimizer will create:
  - 50 lineups with 4 NYY or LAD players
  - 50 lineups with 3 ATL, SF, or CHC players
```

---

## Refresh Mechanism

### Refresh Team Stacks Button

```
┌───────────────────────┐
│ Refresh Team Stacks   │
└───────────────────────┘
```

**Function:** Reload team data from current df_players

**Use Cases:**
1. After loading new CSV
2. After filtering players
3. After manual data updates
4. Troubleshooting sync issues

**Behavior:**
```python
def refresh_team_stacks():
    # Preserve existing selections
    current_selections = save_current_selections()
    
    # Rebuild tables
    teams = df_players['Team'].unique()
    clear_all_tables()
    populate_tables_with_teams(teams)
    
    # Restore selections where teams still exist
    restore_selections(current_selections)
    
    # Recalculate metrics
    update_team_metrics()
```

---

## Visual Indicators

### Selection Status Display

**Bottom Status Bar:**
```
┌────────────────────────────────────────────────────────────┐
│ Selected: NYY, LAD, ATL (3/30 teams) | Est. stack %..: 45%│
└────────────────────────────────────────────────────────────┘
```

**Components:**
- **Team List:** Shows selected team abbreviations
- **Count:** X selected / Y total teams
- **Estimated Coverage:** Rough estimate of lineup coverage

### Conflict Warnings

**When All Stacks conflicts with specific tabs:**
```
┌──────────────────────────────────────────────┐
│ ⚠ Notice: Specific tab selections override  │
│   All Stacks selections                      │
│                                              │
│   4-Stack uses: NYY, LAD (specific)         │
│   5-Stack uses: All 8 teams (from All)      │
└──────────────────────────────────────────────┘
```

---

## Advanced Features

### 1. Game Stacks (Future)

Instead of team-based, stack by game:
```
┌──────────────────────────────────────┐
│ Game: NYY @ BOS                      │
│ O/U: 9.5 | Proj: NYY 5.2, BOS 4.3   │
│ [✓] Stack this game                  │
└──────────────────────────────────────┘
```

### 2. Bring-Back Stacks (Future)

Correlate opposing team players:
```
Main Stack: NYY (4 players)
Bring-Back: BOS (1-2 players)
  
Logic: If NYY scores, BOS likely responds
```

### 3. Vegas Integration (Future)

Auto-select teams based on Vegas lines:
```
[✓] Auto-select teams with O/U > 9.0
[✓] Auto-select teams with team total > 5.0
[✓] Auto-select favorites of 150+ games
```

---

## Data Storage

### Selection Data Structure

```python
team_selections = {
    "all": ["NYY", "LAD", "ATL", "SF", "CHC", "BOS", "HOU", "TB"],
    4: ["NYY", "LAD"],
    3: ["ATL", "SF", "CHC"],
    2: ["BOS", "HOU", "TB"],
    5: []  # Empty = use "all"
}
```

### Exposure Settings

```python
team_exposure = {
    "NYY": {"min": 0, "max": 100, "actual": 0},
    "LAD": {"min": 10, "max": 60, "actual": 0},
    "ATL": {"min": 0, "max": 100, "actual": 0},
    # ...
}
```

---

## Integration with Optimization

### Constraint Generation

```python
# Example: 4-Stack with NYY or LAD
for team in team_selections[4]:  # ["NYY", "LAD"]
    team_batters = get_team_batters(team)
    
    # Create binary variable for this team
    use_team[team] = pulp.LpVariable(f"use_team_{team}", cat='Binary')
    
    # If this team is used, enforce 4+ players
    problem += pulp.lpSum([
        player_vars[p] for p in team_batters
    ]) >= 4 * use_team[team]

# At least one team must be used for the stack
problem += pulp.lpSum(use_team.values()) >= 1
```

### Validation Before Optimization

```python
def validate_stack_configuration():
    errors = []
    
    for stack_size, teams in team_selections.items():
        for team in teams:
            valid, msg = validate_team_for_stack(team, stack_size)
            if not valid:
                errors.append(msg)
    
    if errors:
        show_error_dialog("\n".join(errors))
        return False
    
    return True
```

---

## Error Handling

### No Teams Selected
```
⚠ Warning: No teams selected for stacking
   
   Optimizer will use "No Stacks" mode
   
   To enable stacking:
   • Select teams in Team Stacks tab
   • Enable stack types in Stack Exposure tab
```

### Insufficient Teams
```
⚠ Error: Only 1 team selected for 4-Stack
   
   Need at least 1 team with 4+ batters
   
   Selected: CHC (only 3 batters available)
   
   Please select additional teams or reduce stack size
```

### All Teams Selected
```
⚠ Notice: All 30 teams selected
   
   This is essentially "No Stacks" mode
   
   For focused stacking:
   • Deselect weaker offenses
   • Focus on high-scoring games
   • Use 3-6 teams per stack size
```

---

## Best Practices

### Cash Game Strategy
```
Stack Selection:
✓ Select 4-6 teams
✓ Focus on high team totals (5.0+ runs)
✓ Prefer favorites
✓ Use 4-Stack for correlation

Example: NYY, LAD, ATL, HOU (all 5.0+ run projection)
```

### GPP Tournament Strategy
```
Stack Selection:
✓ Select 10-15 teams
✓ Include high-owned AND contrarian teams
✓ Mix favorites and underdogs
✓ Use 5-Stack for max leverage

Example: Chalk (NYY, LAD) + Contrarian (MIA, OAK, DET)
```

### Multi-Stack Strategy
```
4-Stack: Top 2 teams (NYY, LAD)
3-Stack: Next tier teams (ATL, SF, CHC, HOU)
2-Stack: Value teams (Any with good matchup)

Result: Diverse lineup construction
```

---

## Performance Considerations

### Checkbox State Management

**Problem:** With 30 teams × 5 tabs = 150 checkboxes, need efficient tracking

**Solution:**
```python
class CheckboxManager:
    def __init__(self):
        self.states = {}  # {(stack_size, team): boolean}
    
    def set_state(self, stack_size, team, checked):
        self.states[(stack_size, team)] = checked
    
    def get_state(self, stack_size, team):
        return self.states.get((stack_size, team), False)
    
    def get_selected_teams(self, stack_size):
        return [
            team for (size, team), checked in self.states.items()
            if size == stack_size and checked
        ]
```

---

## Testing Scenarios

### Scenario 1: Basic Stack
```
Input:
  - 4 Stack Tab: [✓] NYY
  - Stack Exposure: [✓] 4-Stack (100%)

Expected: All lineups have 4+ NYY batters
```

### Scenario 2: Multiple Teams
```
Input:
  - 4 Stack Tab: [✓] NYY, LAD, ATL
  - Stack Exposure: [✓] 4-Stack (100%)

Expected: Each lineup has 4+ batters from ONE of {NYY, LAD, ATL}
```

### Scenario 3: Mixed Stacks
```
Input:
  - 4 Stack Tab: [✓] NYY, LAD
  - 3 Stack Tab: [✓] ATL, SF
  - Stack Exposure: [✓] 4-Stack (50%), 3-Stack (50%)

Expected:
  - 50% of lineups: 4+ from NYY or LAD
  - 50% of lineups: 3+ from ATL or SF
```

---

Next: [Stack Exposure Tab](04_STACK_EXPOSURE_TAB.md)

