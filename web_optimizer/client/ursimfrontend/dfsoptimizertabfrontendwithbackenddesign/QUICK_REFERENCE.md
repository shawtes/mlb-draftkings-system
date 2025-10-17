# Quick Reference Guide
## Fast Lookup for Frontend-Backend Integration

---

## Backend Classes You'll Use

```python
# Main worker - runs optimization
from optimizer.genetic.algo import OptimizationWorker

# Main app - has utility methods
from optimizer.genetic.algo import FantasyBaseballApp

# Genetic diversity - for unique lineups
from optimizer.genetic.algo import GeneticDiversityEngine

# Core optimization function
from optimizer.genetic.algo import optimize_single_lineup
```

---

## Essential Data Structures

### Input: Player Data
```python
df_players = pd.DataFrame({
    'Name': str,              # Required
    'Team': str,              # Required
    'Position': str,          # Required (P, C, 1B, 2B, 3B, SS, OF)
    'Salary': int,            # Required (e.g., 11000)
    'Predicted_DK_Points': float,  # Required (e.g., 25.3)
    'ID': str                 # Optional (DK player ID)
})
```

### Input: Settings
```python
settings = {
    'included_players': ['Player A', 'Player B', ...],
    'team_selections': {
        4: ['NYY', 'LAD'],    # Teams for 4-stacks
        3: ['ATL', 'SF']      # Teams for 3-stacks
    },
    'stack_settings': ['4', '3'],
    'num_lineups': 100,
    'min_unique': 3,
    'min_salary': 45000,
    'disable_kelly': False,
    'bankroll': 1000,
    'risk_tolerance': 'medium'
}
```

### Output: Results
```python
results = {
    0: {
        'lineup': DataFrame,      # 10 players
        'total_points': 125.3,
        'total_salary': 49800,
        'stack_type': '4'
    },
    # ... more lineups
}

team_exposure = {'NYY': 45, 'LAD': 38, ...}
stack_exposure = {'4': 60, '3': 40}
```

---

## Backend Function Calls

### Load Players
```python
app = FantasyBaseballApp()
df = app.load_players('players.csv')
```

### Run Optimization
```python
worker = OptimizationWorker(
    df_players=df,
    salary_cap=50000,
    position_limits={'P':2, 'C':1, '1B':1, '2B':1, '3B':1, 'SS':1, 'OF':3},
    included_players=selected_players,
    stack_settings=['4', '3'],
    min_exposure={},
    max_exposure={},
    min_points=1,
    monte_carlo_iterations=100,
    num_lineups=100,
    team_selections={4: ['NYY', 'LAD']},
    min_unique=3,
    bankroll=1000,
    risk_tolerance='medium',
    disable_kelly=False,
    min_salary=45000,
    use_advanced_quant=False,
    advanced_quant_params={}
)

results, team_exp, stack_exp = worker.optimize_lineups()
```

### Export to CSV
```python
# Simple format
dk_positions = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
with open('output.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(dk_positions)
    for lineup in lineups:
        row = app.format_lineup_for_dk(lineup, dk_positions)
        writer.writerow(row)
```

---

## Constraints Reference

### DraftKings MLB Rules
```python
SALARY_CAP = 50000
REQUIRED_TEAM_SIZE = 10
POSITION_LIMITS = {
    'P': 2,    # 2 pitchers
    'C': 1,    # 1 catcher
    '1B': 1,   # 1 first baseman
    '2B': 1,   # 1 second baseman
    '3B': 1,   # 1 third baseman
    'SS': 1,   # 1 shortstop
    'OF': 3    # 3 outfielders
}
```

### Application Constraints
```python
MIN_SALARY_DEFAULT = 45000  # Adjustable
MAX_LINEUPS = 500
MIN_UNIQUE_RANGE = (0, 10)
EXPOSURE_RANGE = (0, 100)  # Percentage
```

---

## Stack Type Formats

### Simple Stacks
```python
stack_settings = ['5']      # 5 from one team
stack_settings = ['4']      # 4 from one team
stack_settings = ['3']      # 3 from one team
stack_settings = ['No Stacks']  # No stacking
```

### Multi-Stacks
```python
stack_settings = ['4|2']      # 4 + 2 from different teams
stack_settings = ['4|2|2']    # 4 + 2 + 2 from different teams
stack_settings = ['3|3|2']    # 3 + 3 + 2 from different teams
stack_settings = ['5|3']      # 5 + 3 from different teams
```

---

## Team Selection Format

```python
# Simple: All stack sizes use same teams
team_selections = {
    'all': ['NYY', 'LAD', 'ATL']
}

# Advanced: Different teams per stack size
team_selections = {
    5: ['NYY', 'LAD'],        # Only NYY/LAD for 5-stacks
    4: ['NYY', 'LAD', 'ATL'], # These 3 for 4-stacks
    3: ['ATL', 'SF', 'CHC']   # These 3 for 3-stacks
}

# Priority: Specific size > 'all' > no constraint
```

---

## Error Codes

### Common Errors
```python
# No data loaded
if df_players is None:
    return {'error': 'NO_DATA', 'message': 'Load CSV first'}

# Insufficient players
if len(included_players) < 30:
    return {'error': 'INSUFFICIENT_PLAYERS', 'message': f'Only {len(included_players)} selected'}

# No stacks selected
if not stack_settings:
    return {'error': 'NO_STACKS', 'message': 'Select at least one stack type'}

# Infeasible constraints
if optimization_status != 'Optimal':
    return {'error': 'NO_SOLUTION', 'message': 'Cannot satisfy constraints'}
```

---

## Performance Targets

```python
# Optimization speed (approximate)
lineups = 20:    10-15 seconds
lineups = 50:    20-30 seconds
lineups = 100:   30-60 seconds
lineups = 200:   60-120 seconds
lineups = 500:   150-300 seconds

# Factors affecting speed:
# • Player pool size (more = slower)
# • Stack complexity (multi-stacks slower)
# • Min unique constraint (higher = slower)
# • Advanced quant (adds 30-60 sec)
```

---

## UI State Management

```python
# Application states
STATE = {
    'INITIAL': 'No data loaded',
    'DATA_LOADED': 'CSV loaded, ready to configure',
    'OPTIMIZING': 'Running optimization',
    'RESULTS_READY': 'Lineups generated',
    'ERROR': 'Error occurred'
}

# UI element enable/disable
if state == 'INITIAL':
    enable = ['Load CSV']
    disable = ['Run Optimization', 'Export']
elif state == 'DATA_LOADED':
    enable = ['Load CSV', 'Run Optimization']
    disable = ['Export']
elif state == 'OPTIMIZING':
    enable = ['Cancel']
    disable = ['Load CSV', 'Run Optimization', 'Export']
elif state == 'RESULTS_READY':
    enable = ['Load CSV', 'Run Optimization', 'Export']
```

---

## Validation Checklist

```python
def validate_before_optimize():
    checks = []
    
    # Data loaded?
    if df_players is None:
        return False, "No player data"
    
    # Enough players?
    if len(included_players) == 0:
        included_players = df_players['Name'].tolist()
    
    # Position coverage?
    for pos in ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']:
        count = count_position_players(pos, included_players)
        if count < POSITION_LIMITS[pos]:
            return False, f"Need {POSITION_LIMITS[pos]} {pos}, have {count}"
    
    # Stack types?
    if not stack_settings or len(stack_settings) == 0:
        return False, "No stack types selected"
    
    # Valid settings?
    if num_lineups < 1 or num_lineups > 500:
        return False, "Lineups must be 1-500"
    
    return True, "OK"
```

---

## Export Format

### DraftKings CSV Structure
```
Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
4766459650, MLB Main, 162584429, $1, 12345678, 23456789, 34567890, ...
```

**Column Order (CRITICAL):**
1. Entry ID (10-digit number)
2. Contest Name (string)
3. Contest ID (9-digit number)
4. Entry Fee (e.g., "$1")
5. P (pitcher 1 ID)
6. P (pitcher 2 ID)
7. C (catcher ID)
8. 1B (first base ID)
9. 2B (second base ID)
10. 3B (third base ID)
11. SS (shortstop ID)
12. OF (outfielder 1 ID)
13. OF (outfielder 2 ID)
14. OF (outfielder 3 ID)

**MUST USE PLAYER IDs (not names)**

---

## Common Pitfalls

### ❌ Don't Do This
```python
# Modifying df_players during optimization
worker.df_players['Predicted_DK_Points'] *= 2  # BAD!

# Accessing worker from wrong thread
worker.results_table.update()  # BAD!

# Wrong team selection format
team_selections = ['NYY', 'LAD']  # BAD! Should be dict
```

### ✅ Do This
```python
# Pass copy of data
worker = OptimizationWorker(df_players=df.copy(), ...)

# Emit signal from worker, update in main thread
worker.optimization_done.emit(results)

# Correct team selection format
team_selections = {4: ['NYY', 'LAD']}
```

---

## Quick Start Template

```python
# File: my_frontend_integration.py

from optimizer.genetic.algo import OptimizationWorker
import pandas as pd

# 1. Load your data
df_players = pd.read_csv('your_data.csv')

# 2. Get settings from YOUR UI
settings = your_ui.get_all_settings()

# 3. Create worker with existing backend
worker = OptimizationWorker(
    df_players=df_players,
    salary_cap=50000,
    position_limits={'P':2, 'C':1, '1B':1, '2B':1, '3B':1, 'SS':1, 'OF':3},
    included_players=settings['selected_players'],
    stack_settings=settings['stack_types'],
    min_exposure={},
    max_exposure={},
    min_points=1,
    monte_carlo_iterations=100,
    num_lineups=settings['num_lineups'],
    team_selections=settings['team_selections'],
    min_unique=settings['min_unique'],
    bankroll=settings['bankroll'],
    risk_tolerance=settings['risk_tolerance'],
    disable_kelly=settings['disable_kelly'],
    min_salary=settings['min_salary'],
    use_advanced_quant=False,
    advanced_quant_params={}
)

# 4. Run optimization
results, team_exp, stack_exp = worker.optimize_lineups()

# 5. Display in YOUR UI
your_ui.display_results(results)
```

---

## Design Docs Index

**Read in this order:**

1. **[00_OVERVIEW.md](00_OVERVIEW.md)** - Start here
2. **[01_ARCHITECTURE.md](01_ARCHITECTURE.md)** - Understand structure
3. **[09_DATA_FLOW.md](09_DATA_FLOW.md)** - See how it all connects
4. **[10_FRONTEND_BACKEND_INTEGRATION.md](10_FRONTEND_BACKEND_INTEGRATION.md)** - Integration patterns
5. Individual tab docs (02-08) - Implement each UI component

---

**Ready to integrate your frontend! 🎯**

