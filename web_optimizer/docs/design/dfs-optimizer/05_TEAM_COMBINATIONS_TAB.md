# Team Combinations Tab Design
## Automated Multi-Team Stack Generation

---

## Purpose

The **Team Combinations Tab** automates the process of generating ALL possible team combinations for a given stack pattern. Instead of manually configuring each combination, users select teams and stack size, and the system generates every permutation.

---

## Use Case Example

**Scenario:** "I want to test every possible 4|2 combination using Yankees, Dodgers, Braves, and Giants"

**Manual Approach:** Would require configuring:
- NYY(4) + LAD(2)
- NYY(4) + ATL(2)
- NYY(4) + SF(2)
- LAD(4) + NYY(2)
- LAD(4) + ATL(2)
- LAD(4) + SF(2)
- ATL(4) + NYY(2)
- ...16 more combinations

**Automated Approach:** Select 4 teams, choose "4|2" pattern, generate all 12 combinations automatically

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│ Team Combinations Tab                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────┬─────────────────────────────────────┐ │
│  │ TEAM SELECTION     │ CONFIGURATION                       │ │
│  ├────────────────────┼─────────────────────────────────────┤ │
│  │                    │                                     │ │
│  │ Select Teams:      │ Stack Type:                         │ │
│  │ [✓] NYY            │ [4|2 ▼]                             │ │
│  │ [✓] LAD            │   • 5                               │ │
│  │ [✓] ATL            │   • 4                               │ │
│  │ [✓] SF             │   • 3                               │ │
│  │ [ ] CHC            │   • No Stacks                       │ │
│  │ [ ] BOS            │   • 5|2                             │ │
│  │ [ ] HOU            │   • 4|2         ← Selected          │ │
│  │ ...                │   • 4|2|2                           │ │
│  │                    │   • 3|3|2                           │ │
│  │ [✓ Select All]     │   • 3|2|2                           │ │
│  │ [✗ Deselect All]   │   • 2|2|2                           │ │
│  │                    │   • 5|3                             │ │
│  │ Selected: 4/30     │                                     │ │
│  │                    │ Default Lineups/Combo: [5____]      │ │
│  │                    │                                     │ │
│  │                    │ [Generate Combinations]             │ │
│  └────────────────────┴─────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Generated Combinations:                                  │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Select│Team Combination      │Lineups/Combo│Actions     ││ │
│  ├───────┼──────────────────────┼─────────────┼────────────┤│ │
│  │  [✓]  │ NYY(4) + LAD(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ NYY(4) + ATL(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ NYY(4) + SF(2)      │     5       │   Ready    ││ │
│  │  [✓]  │ LAD(4) + NYY(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ LAD(4) + ATL(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ LAD(4) + SF(2)      │     5       │   Ready    ││ │
│  │  [✓]  │ ATL(4) + NYY(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ ATL(4) + LAD(2)     │     5       │   Ready    ││ │
│  │  [✓]  │ ATL(4) + SF(2)      │     5       │   Ready    ││ │
│  │  [✓]  │ SF(4) + NYY(2)      │     5       │   Ready    ││ │
│  │  [✓]  │ SF(4) + LAD(2)      │     5       │   Ready    ││ │
│  │  [✓]  │ SF(4) + ATL(2)      │     5       │   Ready    ││ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Total Lineups: 60 (12 combos × 5 lineups)                   │
│  [Generate All Combination Lineups]                           │
└────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Team Selection Panel (Left)

#### Team Checkboxes
- **Display:** Scrollable list of all available teams
- **Source:** Populated from loaded player CSV
- **Interaction:** Click to toggle selection
- **Visual:** Standard checkbox with team abbreviation

#### Select All / Deselect All Buttons
```
┌──────────────┐  ┌────────────────┐
│ ✓ Select All │  │ ✗ Deselect All │
└──────────────┘  └────────────────┘
```

#### Selection Counter
```
Selected: 4/30 teams
```
- Shows how many teams are selected
- Updates in real-time

### 2. Configuration Panel (Right)

#### Stack Type Dropdown
```
┌─────────────────┐
│ Stack: [4|2 ▼]  │
│  • 5            │
│  • 4            │
│  • 3            │
│  • No Stacks    │
│  • 5|2          │
│  • 4|2          │ ← Selected
│  • 4|2|2        │
│  • 3|3|2        │
│  • 3|2|2        │
│  • 2|2|2        │
│  • 5|3          │
└─────────────────┘
```

**Options:**
- All single stack sizes
- All multi-stack patterns
- Default: "4" (most popular)

#### Default Lineups per Combo Input
```
┌──────────────────────────────────┐
│ Lineups per Combo: [5_____]      │
└──────────────────────────────────┘
```
- **Purpose:** How many lineups to generate for each combination
- **Range:** 1-100
- **Default:** 5
- **Calculation:** Total lineups = combos × lineups_per_combo

#### Generate Combinations Button
```
┌──────────────────────────┐
│ Generate Combinations    │
└──────────────────────────┘
```
- **Function:** Create all possible team combinations
- **Action:** Populate combinations table below
- **Validation:** Requires 2+ teams selected

### 3. Combinations Table

**Purpose:** Display all generated combinations for review

#### Columns:

**Select Checkbox**
- **Width:** 50px
- **Purpose:** Mark this combination for lineup generation
- **Default:** Checked (all selected)
- **Behavior:** Can uncheck to skip unwanted combos

**Team Combination**
- **Width:** 250px
- **Format:** "TEAM1(stack_size) + TEAM2(stack_size) + ..."
- **Example:** "NYY(4) + LAD(2)" or "ATL(3) + SF(3) + CHC(2)"
- **Sorting:** Alphabetical by first team

**Lineups per Combo**
- **Width:** 120px
- **Content:** Number input
- **Editable:** Yes, users can adjust per combination
- **Default:** Inherits from "Default Lineups/Combo"
- **Range:** 1-100

**Actions**
- **Width:** 100px
- **Content:** Status indicator
- **Values:**
  - "Ready" - Not yet generated
  - "Generating..." - In progress
  - "Complete (X lineups)" - Done
  - "Error" - Failed

### 4. Total Display
```
┌──────────────────────────────────────────────────────────────┐
│ Total Lineups: 60                                            │
│ (12 combinations × 5 lineups each)                           │
└──────────────────────────────────────────────────────────────┘
```

**Calculation:**
```python
total = sum(combo.lineups_per_combo for combo in selected_combos)
```

**Auto-Update:** Recalculates when:
- Combos are checked/unchecked
- Lineups per combo values change
- New combinations generated

### 5. Generate Button
```
┌──────────────────────────────────┐
│ 🚀 Generate All Combination      │
│    Lineups                       │
└──────────────────────────────────┘
```
- **Function:** Run optimization for all selected combinations
- **Behavior:** Creates lineups for each combination
- **Status:** Shows progress during generation
- **Result:** Displays in results panel

---

## Combination Generation Logic

### Algorithm

```python
def generate_combinations(selected_teams, stack_pattern):
    """
    Generate all possible team combinations for a stack pattern
    
    Args:
        selected_teams: List of team abbreviations
        stack_pattern: String like "4|2" or "3|3|2"
    
    Returns:
        List of team combination dictionaries
    """
    stack_sizes = [int(x) for x in stack_pattern.split('|')]
    teams_needed = len(stack_sizes)
    
    if len(selected_teams) < teams_needed:
        raise ValueError(f"Need {teams_needed} teams, only {len(selected_teams)} selected")
    
    combinations_list = []
    
    # Generate all team combinations
    from itertools import combinations, permutations
    
    for team_combo in combinations(selected_teams, teams_needed):
        # Generate all permutations to assign different stack sizes
        for team_perm in permutations(team_combo):
            combo_info = {
                'teams': team_perm,
                'stack_sizes': stack_sizes,
                'display': format_combination(team_perm, stack_sizes),
                'lineups': default_lineups_per_combo
            }
            combinations_list.append(combo_info)
    
    return combinations_list

def format_combination(teams, stack_sizes):
    """Format as 'TEAM1(size1) + TEAM2(size2) + ...'"""
    parts = [f"{team}({size})" for team, size in zip(teams, stack_sizes)]
    return " + ".join(parts)
```

### Example Calculations

**Input: 3 teams (NYY, LAD, ATL), Pattern: 4|2**

```python
teams_needed = 2  # Two stack sizes in "4|2"
combinations = choose(3, 2) = 3
permutations_each = 2! = 2

total = 3 × 2 = 6 combinations:
  1. NYY(4) + LAD(2)
  2. NYY(4) + ATL(2)
  3. LAD(4) + NYY(2)
  4. LAD(4) + ATL(2)
  5. ATL(4) + NYY(2)
  6. ATL(4) + LAD(2)
```

**Input: 4 teams (NYY, LAD, ATL, SF), Pattern: 3|3|2**

```python
teams_needed = 3
combinations = choose(4, 3) = 4
permutations_each = 3! = 6

total = 4 × 6 = 24 combinations
```

**Mathematical Formula:**
```
Total Combinations = C(n, k) × k!

Where:
  n = number of selected teams
  k = number of stack sizes in pattern
  C(n, k) = n! / (k! × (n-k)!)
```

---

## User Workflows

### Workflow 1: Basic Combination Testing
```
Goal: Test all 4|2 combinations with top 4 teams

Steps:
1. Select teams: NYY, LAD, ATL, SF
2. Choose stack pattern: "4|2"
3. Set lineups per combo: 10
4. Click "Generate Combinations"
   → Creates 12 combinations
5. Review combinations in table
6. Click "Generate All Combination Lineups"
7. Wait for optimization (12 × 10 = 120 lineups)
8. Review results

Result: 120 diverse lineups covering all team pairings
```

### Workflow 2: Selective Generation
```
Goal: Only want certain combinations, not all

Steps:
1. Generate all combinations as above
2. Review combinations table
3. Uncheck unwanted combinations:
   [ ] ATL(4) + SF(2)
   [ ] SF(4) + ATL(2)
4. Adjust lineups for specific combos:
   NYY(4) + LAD(2): 20 lineups (prioritize)
   Others: 5 lineups each
5. Generate selected only

Result: Customized combination set
```

### Workflow 3: Multi-Stack Exploration
```
Goal: Compare 3|3|2 vs 4|2|2 strategies

Steps:
1. Select 6 teams
2. Generate "3|3|2" combinations
   → 120 combinations!
3. Set lineups per combo: 2
4. Generate → 240 lineups
5. Review performance
6. Repeat with "4|2|2" pattern
7. Compare results
```

---

## Performance Considerations

### Combination Explosion

**Warning Thresholds:**

```python
if total_combinations > 50:
    show_warning(f"This will create {total_combinations} combinations!")
    show_warning("Consider:")
    show_warning("• Selecting fewer teams")
    show_warning("• Using simpler stack pattern")
    show_warning("• Reducing lineups per combo")

if total_lineups > 500:
    show_warning(f"Total lineups: {total_lineups}")
    show_warning("This may take several minutes")
    show_confirm("Continue?")
```

**Example Explosions:**

| Teams | Pattern | Combos | × Lineups | Total |
|-------|---------|--------|-----------|-------|
| 3     | 4\|2    | 6      | × 5       | 30    |
| 4     | 4\|2    | 12     | × 5       | 60    |
| 5     | 4\|2    | 20     | × 5       | 100   |
| 6     | 4\|2    | 30     | × 10      | 300   |
| 5     | 3\|3\|2 | 60     | × 5       | 300   |
| 6     | 3\|3\|2 | 120    | × 5       | 600   |

### Optimization Strategy

**For Large Combination Sets:**

```python
# Generate combinations in batches
batch_size = 10
for i in range(0, len(combinations), batch_size):
    batch = combinations[i:i+batch_size]
    
    # Run optimization for batch
    results = optimize_batch(batch)
    
    # Update progress
    progress = (i + len(batch)) / len(combinations) * 100
    update_progress_bar(progress)
    
    # Allow UI to remain responsive
    QApplication.processEvents()
```

---

## Validation

### Pre-Generation Validation

```python
def validate_combination_request():
    errors = []
    
    # Check team count
    if len(selected_teams) < 2:
        errors.append("Select at least 2 teams")
    
    # Check against stack pattern
    stack_sizes = parse_stack_pattern(stack_pattern)
    teams_needed = len(stack_sizes)
    
    if len(selected_teams) < teams_needed:
        errors.append(f"{stack_pattern} requires {teams_needed} teams")
        errors.append(f"Only {len(selected_teams)} selected")
    
    # Check sufficient players per team
    for team in selected_teams:
        for stack_size in stack_sizes:
            if not has_sufficient_batters(team, stack_size):
                errors.append(f"{team} lacks batters for {stack_size}-stack")
    
    # Warn about large combinations
    total_combos = calculate_combination_count()
    if total_combos > 100:
        warnings.append(f"Will create {total_combos} combinations")
        warnings.append("This is a lot! Consider reducing teams")
    
    return errors, warnings
```

---

## Visual Feedback

### Generation Progress

```
┌──────────────────────────────────────────────────────┐
│ Generating Combinations...                           │
│                                                       │
│ ████████████████████░░░░░░░░░░░░░░░░░  60%          │
│                                                       │
│ Processing: LAD(4) + ATL(2)                          │
│ Complete: 7/12 combinations                          │
│ Lineups created: 35/60                               │
│                                                       │
│ [Cancel]                                             │
└──────────────────────────────────────────────────────┘
```

### Completion Summary

```
┌──────────────────────────────────────────────────────┐
│ ✓ Combination Lineups Generated                      │
│                                                       │
│ • Total combinations: 12                             │
│ • Total lineups: 60                                  │
│ • Generation time: 45 seconds                        │
│ • Avg lineups/sec: 1.3                               │
│                                                       │
│ Results displayed in control panel →                │
│                                                       │
│ [OK]                                                 │
└──────────────────────────────────────────────────────┘
```

---

## Error Handling

### Insufficient Teams
```
⚠ Error: Not enough teams selected

Pattern "4|2|2" requires 3 teams
Currently selected: 2 teams (NYY, LAD)

Select at least 1 more team to proceed.
```

### Team Lacks Players
```
⚠ Warning: MIA has insufficient players

MIA selected for 4-stack combinations
Only 3 batters available (need 4)

Options:
• Deselect MIA
• Reduce stack size
• Select more MIA players in Players tab
```

### Too Many Combinations
```
⚠ Warning: Large combination count

Configuration will create 360 combinations:
• 6 teams selected
• Pattern: 3|3|2
• Calc: C(6,3) × 3! = 20 × 6 × 3 = 360

At 5 lineups each = 1,800 total lineups!

This will take ~5-10 minutes.

[Reduce Teams] [Reduce Lineups] [Continue Anyway]
```

---

## Best Practices

### Cash Games
```
Recommendation:
• Select 3-4 teams (top offenses)
• Use 4|2 pattern
• 5-10 lineups per combo
• Total: 18-48 combinations

Why: Manageable size, solid correlation
```

### GPP Tournaments
```
Recommendation:
• Select 5-6 teams (mix chalk + contrarian)
• Use 4|2|2 or 3|3|2 patterns
• 3-5 lineups per combo
• Total: 60-360 combinations

Why: Maximum differentiation from field
```

### Mass Multi-Entry
```
Recommendation:
• Select 4-5 teams
• Use multiple patterns separately:
  - Generate 4|2 (20 combos × 5 = 100)
  - Generate 3|3|2 (60 combos × 3 = 180)
• Total: 280 unique lineups

Why: Comprehensive game coverage
```

---

## Technical Implementation

### Data Structure

```python
class CombinationConfig:
    def __init__(self, teams, stack_pattern):
        self.teams = teams  # ['NYY', 'LAD', 'ATL']
        self.stack_pattern = stack_pattern  # "4|2"
        self.stack_sizes = [int(x) for x in stack_pattern.split('|')]
        self.combinations = []
    
    def generate(self):
        for team_combo in combinations(self.teams, len(self.stack_sizes)):
            for team_perm in permutations(team_combo):
                combo = TeamCombination(
                    teams=team_perm,
                    stack_sizes=self.stack_sizes
                )
                self.combinations.append(combo)
        
        return self.combinations

class TeamCombination:
    def __init__(self, teams, stack_sizes):
        self.teams = teams
        self.stack_sizes = stack_sizes
        self.lineups_requested = 5
        self.lineups_generated = []
        self.status = "Ready"
    
    def display_name(self):
        parts = [f"{t}({s})" for t, s in zip(self.teams, self.stack_sizes)]
        return " + ".join(parts)
    
    def to_team_selections(self):
        """Convert to format optimizer expects"""
        return {
            size: [team] for team, size in zip(self.teams, self.stack_sizes)
        }
```

---

Next: [Advanced Quant Tab](06_ADVANCED_QUANT_TAB.md)

