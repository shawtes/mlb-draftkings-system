# Stack Exposure Tab Design
## Stack Type Configuration and Exposure Management

---

## Purpose

The **Stack Exposure Tab** defines which stacking strategies should be used and in what proportion. This controls the mix of lineup construction strategies.

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│ Stack Exposure Tab                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Configure which stack types to use and their proportions     │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Select│Stack Type│Min Exp│Max Exp│Lineup│Pool│Entry Exp││ │
│  ├───────┼──────────┼───────┼───────┼──────┼────┼─────────┤│ │
│  │  [✓]  │    5     │   0   │  100  │ 15%  │15% │  15%    ││ │
│  │  [✓]  │    4     │   0   │  100  │ 45%  │45% │  45%    ││ │
│  │  [✓]  │    3     │   0   │  100  │ 30%  │30% │  30%    ││ │
│  │  [ ]  │No Stacks │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │   4|2|2  │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │    4|2   │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │   3|3|2  │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │   3|2|2  │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │   2|2|2  │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │    5|3   │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  │  [ ]  │    5|2   │   0   │  100  │  0%  │ 0% │   0%    ││ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ✓ Tip: Selected stack types will be distributed across      │
│         generated lineups. Exposure percentages calculated    │
│         after optimization.                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## Stack Type Definitions

### Simple Stacks

#### 5 Stack
- **Definition:** 5 players from the same team
- **Composition:** Typically P + 4 batters OR 5 batters
- **Use Case:** Maximum correlation, GPP leverage
- **Risk:** High - very boom/bust
- **Example:** NYY pitcher + 4 NYY batters

#### 4 Stack  
- **Definition:** 4 players from the same team
- **Composition:** Typically 4 batters OR P + 3 batters
- **Use Case:** Standard GPP strategy, balanced correlation
- **Risk:** Medium-High
- **Example:** 4 LAD batters

#### 3 Stack
- **Definition:** 3 players from the same team
- **Composition:** Usually 3 batters
- **Use Case:** Cash games, moderate correlation
- **Risk:** Medium
- **Example:** Top 3 of NYY lineup

#### 2 Stack
- **Definition:** 2 players from the same team
- **Composition:** Often P + C combo or 1-2 hitters
- **Use Case:** Light correlation, flexibility
- **Risk:** Low
- **Example:** Pitcher + his catcher

#### No Stacks
- **Definition:** No team has more than 2 players
- **Composition:** Fully diversified across teams
- **Use Case:** Maximum roster flexibility
- **Risk:** Lowest correlation
- **Example:** 1 player from 8-10 different teams

### Complex Multi-Stacks

#### 4|2|2 Stack
- **Definition:** 4 from Team A + 2 from Team B + 2 from Team C
- **Composition:** Primary stack + two mini-stacks
- **Use Case:** Balanced multi-game correlation
- **Example:** 4 NYY + 2 LAD + 2 ATL

#### 4|2 Stack
- **Definition:** 4 from Team A + 2 from Team B
- **Composition:** Primary stack + secondary
- **Use Case:** Two-game correlation
- **Example:** 4 NYY + 2 LAD

#### 3|3|2 Stack
- **Definition:** 3 from Team A + 3 from Team B + 2 from Team C
- **Composition:** Two equal stacks + mini-stack
- **Use Case:** Three-game balanced exposure
- **Example:** 3 LAD + 3 SF + 2 ATL

#### 3|2|2 Stack
- **Definition:** 3 from Team A + 2 from Team B + 2 from Team C
- **Composition:** Main stack + two secondaries
- **Use Case:** Diversified correlation
- **Example:** 3 NYY + 2 BOS + 2 TB

#### 2|2|2 Stack
- **Definition:** 2 from Team A + 2 from Team B + 2 from Team C
- **Composition:** Three balanced mini-stacks
- **Use Case:** Maximum game diversification
- **Example:** 2 NYY + 2 LAD + 2 ATL

#### 5|3 Stack
- **Definition:** 5 from Team A + 3 from Team B
- **Composition:** Heavy primary + secondary
- **Use Case:** Extreme two-game correlation
- **Risk:** Very High
- **Example:** 5 NYY + 3 BOS (same game)

#### 5|2 Stack
- **Definition:** 5 from Team A + 2 from Team B
- **Composition:** Max stack + mini-stack
- **Use Case:** GPP lever play with safety
- **Example:** 5 LAD + 2 SF

---

## Table Columns

### Column 1: Select Checkbox
- **Width:** 50px
- **Purpose:** Enable this stack type for optimization
- **Default:** Unchecked
- **Behavior:** At least one must be selected to optimize

### Column 2: Stack Type
- **Width:** 100px
- **Content:** Stack pattern identifier
- **Format:**
  - Single number: "5", "4", "3", "2"
  - Multiple: "4|2|2" (pipe-separated)
  - Special: "No Stacks"
- **Sorting:** Descending by stack size

### Column 3: Min Exp (Spinbox)
- **Width:** 80px
- **Content:** Minimum lineup percentage (0-100)
- **Default:** 0
- **Purpose:** Ensure at least X% of lineups use this stack
- **Example:** Min 20 → At least 20% of lineups will be 4-stacks

### Column 4: Max Exp (Spinbox)
- **Width:** 80px
- **Content:** Maximum lineup percentage (0-100)
- **Default:** 100
- **Purpose:** Cap at most X% of lineups with this stack
- **Validation:** Must be ≥ Min Exp

### Column 5: Lineup Exp (%)
- **Width:** 80px
- **Content:** Actual percentage of lineups with this stack
- **Format:** "XX.X%"
- **State:** Read-only, calculated after optimization
- **Calculation:** `(count_of_type / total_lineups) * 100`

### Column 6: Pool Exp (%)
- **Width:** 80px
- **Content:** Percentage across entire lineup pool
- **Purpose:** Track distribution in multi-session workflow
- **Future:** Aggregate across multiple optimization runs

### Column 7: Entry Exp (%)
- **Width:** 80px
- **Content:** Percentage in contest entries
- **Purpose:** Track what's actually entered in contests
- **Future:** Integration with entry tracking

---

## Stack Distribution Logic

### Automatic Distribution

**When multiple stacks selected (no min/max constraints):**

```python
# Example: User selects 5, 4, and 3 stacks
selected_stacks = ["5", "4", "3"]
num_lineups = 100

# Equal distribution
lineups_per_stack = num_lineups / len(selected_stacks)
# Result: ~33 five-stacks, ~33 four-stacks, ~34 three-stacks
```

### Weighted Distribution

**With min/max constraints:**

```python
# Example:
# 5 Stack: Min 10%, Max 30%
# 4 Stack: Min 40%, Max 60%
# 3 Stack: Min 20%, Max 40%

# Optimizer distributes within constraints
# Possible result:
#   15% five-stacks (within 10-30%)
#   55% four-stacks (within 40-60%)
#   30% three-stacks (within 20-40%)
```

### Priority System

**If constraints conflict:**

```python
priority_order = [
    "5",      # Highest complexity first
    "4|2|2",
    "5|3",
    "4|2",
    "4",
    "3|3|2",
    "3|2|2",
    "3",
    "2|2|2",
    "2",
    "No Stacks"  # Fallback
]
```

---

## User Workflows

### Workflow 1: Simple Cash Game
```
Goal: Reliable 3-stacks from top teams

Configuration:
  [✓] 3 Stack only
  Min: 0, Max: 100
  Team Stacks: Select 4-6 top teams

Result: All 100 lineups are 3-stacks from selected teams
```

### Workflow 2: Balanced GPP
```
Goal: Mix of stack sizes for leverage

Configuration:
  [✓] 5 Stack (Min 10%, Max 20%)
  [✓] 4 Stack (Min 60%, Max 70%)
  [✓] 3 Stack (Min 10%, Max 30%)

Result:
  ~15% five-stacks (high leverage)
  ~65% four-stacks (core strategy)
  ~20% three-stacks (safe plays)
```

### Workflow 3: Multi-Stack Strategy
```
Goal: Complex game correlations

Configuration:
  [✓] 4|2|2 (50% of lineups)
  [✓] 3|3|2 (50% of lineups)
  Team Stacks:
    4/3 Stack: NYY, LAD, ATL
    2 Stack: SF, CHC, HOU

Result: Every lineup has multi-game exposure
```

---

## Validation Rules

### Rule 1: At Least One Selected
```python
if not any_stack_selected():
    show_error("Must select at least one stack type")
    return False
```

### Rule 2: Min ≤ Max
```python
for stack_type in selected_stacks:
    if stack_type.min_exp > stack_type.max_exp:
        show_error(f"{stack_type}: Min cannot exceed Max")
        return False
```

### Rule 3: Total Min ≤ 100%
```python
total_min = sum(stack.min_exp for stack in selected_stacks)
if total_min > 100:
    show_warning(f"Total minimum exposure is {total_min}% (> 100%)")
    show_warning("Constraints may conflict")
```

### Rule 4: Sufficient Teams
```python
for stack_type in multi_stacks:  # e.g., "4|2|2"
    required_teams = len(stack_type.split('|'))
    
    if len(selected_teams) < required_teams:
        show_error(f"{stack_type} requires {required_teams} teams")
        show_error(f"Only {len(selected_teams)} teams selected")
        return False
```

---

## Visual Feedback

### Active Stack Indicator
```
┌────────────────────────────────┐
│ Active Stack Types: 3          │
│ • 5 Stack (15% target)         │
│ • 4 Stack (60% target)         │
│ • 3 Stack (25% target)         │
└────────────────────────────────┘
```

### Distribution Preview
```
┌────────────────────────────────────────────┐
│ Projected Distribution (100 lineups):     │
│                                            │
│ 5 Stack:  ██████           15 lineups     │
│ 4 Stack:  ████████████████████  60 lineups│
│ 3 Stack:  ██████████          25 lineups  │
└────────────────────────────────────────────┘
```

### After Optimization
```
┌────────────────────────────────────────────┐
│ Actual Distribution:                       │
│                                            │
│ 5 Stack:  14% (14/100) ✓ Within range     │
│ 4 Stack:  61% (61/100) ✓ Within range     │
│ 3 Stack:  25% (25/100) ✓ Within range     │
└────────────────────────────────────────────┘
```

---

## Advanced Features

### 1. Preset Strategies
```
┌────────────────────────────────┐
│ Load Preset:                   │
│ • Cash Game (3-stacks)         │
│ • GPP Balanced (4/5 mix)       │
│ • Contrarian (5-stacks heavy)  │
│ • Multi-Game (4|2|2, 3|3|2)    │
└────────────────────────────────┘
```

### 2. Exposure Balancing
```
[✓] Auto-balance exposures
    Automatically adjust min/max to total 100%
    
Example:
  Input: 5 Stack (20%), 4 Stack (60%), 3 Stack (leave empty)
  Auto: 3 Stack set to 20% (= 100% total)
```

### 3. Save/Load Configurations
```
[Save Current Config]  [Load Config]

Saved configs stored as JSON:
{
  "name": "GPP Balanced",
  "stacks": [
    {"type": "5", "min": 10, "max": 20},
    {"type": "4", "min": 60, "max": 70},
    {"type": "3", "min": 10, "max": 30}
  ]
}
```

---

## Integration with Optimizer

### Constraint Generation

```python
def generate_stack_constraints(problem, selected_stacks, num_lineups):
    """
    Create optimization constraints for stack distribution
    """
    for stack_type in selected_stacks:
        # Calculate target lineup count
        min_count = int(num_lineups * stack_type.min_exp / 100)
        max_count = int(num_lineups * stack_type.max_exp / 100)
        
        # Generate lineups meeting stack requirements
        for i in range(min_count, max_count):
            lineup = optimize_with_stack_constraint(stack_type)
            if lineup.valid:
                add_to_results(lineup, stack_type)
```

### Distribution Tracking

```python
class StackDistributionTracker:
    def __init__(self):
        self.stack_counts = defaultdict(int)
    
    def add_lineup(self, lineup):
        stack_type = self.detect_stack_type(lineup)
        self.stack_counts[stack_type] += 1
    
    def get_exposure(self, stack_type, total_lineups):
        return (self.stack_counts[stack_type] / total_lineups) * 100
    
    def detect_stack_type(self, lineup):
        """Identify what stack pattern this lineup uses"""
        team_counts = lineup['Team'].value_counts()
        
        # Sort by count descending
        sorted_counts = sorted(team_counts.values(), reverse=True)
        
        # Map to stack type
        if sorted_counts[0] >= 5:
            if len(sorted_counts) > 1 and sorted_counts[1] >= 3:
                return "5|3"
            elif len(sorted_counts) > 1 and sorted_counts[1] >= 2:
                return "5|2"
            else:
                return "5"
        elif sorted_counts[0] >= 4:
            if len(sorted_counts) > 1 and sorted_counts[1] >= 2:
                if len(sorted_counts) > 2 and sorted_counts[2] >= 2:
                    return "4|2|2"
                else:
                    return "4|2"
            else:
                return "4"
        # ... and so on
```

---

## Best Practices

### Cash Games
```
Recommended:
[✓] 3 Stack only
  Min: 0, Max: 100
  
Why: Consistent scoring, reduced variance
```

### GPP Tournaments
```
Recommended:
[✓] 5 Stack (10-20%)
[✓] 4 Stack (60-70%)
[✓] 3 Stack (10-30%)

Why: Mix of leverage and stability
```

### Contrarian Strategy
```
Recommended:
[✓] 5 Stack (40-50%)
[✓] 5|3 Stack (20-30%)
[✓] 4 Stack (20-30%)

Why: Max differentiation from field
```

---

## Error Handling

### No Stack Types Selected
```
⚠ Error: No stack types selected

You must enable at least one stack type:
• Simple: 5, 4, 3, 2, or No Stacks
• Complex: 4|2|2, 3|3|2, etc.

Click a checkbox in the 'Select' column to enable.
```

### Conflicting Constraints
```
⚠ Warning: Total minimum exposure exceeds 100%

Current minimums:
• 5 Stack: 30% min
• 4 Stack: 50% min
• 3 Stack: 40% min
Total: 120% (impossible!)

Adjust minimums to total ≤ 100%
```

### Insufficient Diversity
```
⚠ Warning: Only simple stacks selected

For 100+ lineups, consider:
• Using multiple stack types
• Enabling multi-stacks (4|2|2, etc.)
• Reducing min_unique constraint

This will improve lineup diversity.
```

---

Next: [Team Combinations Tab](05_TEAM_COMBINATIONS_TAB.md)

