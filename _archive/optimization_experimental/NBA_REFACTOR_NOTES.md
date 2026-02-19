# NBA Genetic Algorithm Optimizer - Refactoring Guide

## Status: IN PROGRESS
File: `nba_sportsdata.io_gentic algo.py` (7,160 lines)

## ✅ Changes Made So Far:

### 1. Core Settings Updated (Lines 98-151)
- ✅ Changed from NFL to NBA DraftKings positions
- ✅ Updated `SALARY_CAP = 50000`
- ✅ Updated `MIN_SALARY_DEFAULT = 49000`
- ✅ Changed `REQUIRED_TEAM_SIZE = 8` (was 9 for NFL)

### 2. Position Limits Updated
```python
POSITION_LIMITS = {
    'PG': 1,   # Point Guard
    'SG': 1,   # Shooting Guard
    'SF': 1,   # Small Forward
    'PF': 1,   # Power Forward
    'C': 1,    # Center
    'G': 1,    # Guard (PG/SG)
    'F': 1,    # Forward (SF/PF)
    'UTIL': 1  # Utility (any position)
}
```

### 3. Stack Types Updated
```python
stack_mapping = {
    "PG + C": "pg_c_stack",           # Point Guard + Center correlation
    "PG + Wing": "pg_wing_stack",     # Point Guard + Wing (SF/SG)
    "Stars + Value": "stars_value",   # 2-3 stars + value plays
    "Game Stack": "game_stack",        # 4+ players from one high-scoring game
    "Balanced": "balanced",            # No specific stacking
    "No Stack": "No Stacks"
}
```

### 4. Import Updates
- ✅ Changed `NFL_Stack_Engine` → `NBA_Stack_Engine`
- ✅ Changed `map_nfl_stack_to_backend` → `map_nba_stack_to_backend`

---

## 🚧 TODO: Major Sections Needing NBA-Specific Logic

### Critical Updates Needed:

1. **Position Eligibility Logic** (Lines ~340-400)
   - Remove: `FLEX_POSITIONS = ['RB', 'WR', 'TE']`
   - Add: Guard/Forward/UTIL eligibility checks
   - NBA players can be multi-positional (e.g., `PG/SG`, `SF/PF/C`)

2. **Genetic Algorithm Position Mutations** (Lines ~370-390)
   - Update position swapping for NBA positions
   - Handle multi-position eligibility

3. **PuLP Optimizer Constraints** (Lines ~580-620)
   - Remove FLEX constraint logic
   - Add G (PG/SG), F (SF/PF), UTIL (any) constraints
   - Handle position eligibility from DK roster positions

4. **Stack Enforcement Logic** (Lines ~620-850)
   - Remove: QB-based stacks, Game stacks (QB vs opposing D)
   - Add: PG-C correlation, Backcourt stacks, Game stacks (high O/U games)
   - Update team correlation logic

5. **Team Selection Logic** (Lines ~690-780)
   - Remove: DST/Offense correlation
   - Add: Game totals-based stacking (high O/U games)
   - Add: Pace-adjusted team selection

6. **Lineup Validation** (Lines ~470-490)
   - Update for 8 players (not 9)
   - Validate position eligibility for G/F/UTIL spots

7. **GUI Integration** (Lines ~3000-5000)
   - Update position dropdowns (PG, SG, SF, PF, C, G, F, UTIL)
   - Update stack type options
   - Update team selection (remove game/defense logic)

8. **Export Format** (Lines ~2500-2700)
   - Update DraftKings CSV column headers for NBA
   - Position order: PG, SG, SF, PF, C, G, F, UTIL

---

## 📊 NBA-Specific Features to Add:

### 1. Position Eligibility Parser
```python
def get_player_eligible_positions(roster_position):
    """
    Parse DK roster position string (e.g., 'PG/SG/G/UTIL') 
    to determine which lineup spots a player can fill.
    """
    positions = roster_position.split('/')
    eligible = []
    
    if 'PG' in positions: eligible.append('PG')
    if 'SG' in positions: eligible.append('SG')
    if 'SF' in positions: eligible.append('SF')
    if 'PF' in positions: eligible.append('PF')
    if 'C' in positions: eligible.append('C')
    
    # G spot: eligible if PG or SG
    if 'PG' in positions or 'SG' in positions or 'G' in positions:
        eligible.append('G')
    
    # F spot: eligible if SF or PF
    if 'SF' in positions or 'PF' in positions or 'F' in positions:
        eligible.append('F')
    
    # UTIL: everyone is eligible
    eligible.append('UTIL')
    
    return list(set(eligible))
```

### 2. NBA Stack Logic
```python
def enforce_nba_stacks(problem, df, player_vars, stack_type, team_selections):
    """
    Enforce NBA-specific stacking constraints
    """
    if stack_type == "pg_c_stack":
        # Ensure PG and C from same team
        for team in team_selections:
            team_pgs = df[(df['Team'] == team) & (df['Position'] == 'PG')].index
            team_cs = df[(df['Team'] == team) & (df['Position'] == 'C')].index
            
            if team_pgs and team_cs:
                # If we select a PG from this team, must also select a C
                problem += (
                    pulp.lpSum([player_vars[idx] for idx in team_pgs]) >= 
                    pulp.lpSum([player_vars[idx] for idx in team_cs])
                )
    
    elif stack_type == "game_stack":
        # 4+ players from games with O/U > 225
        # This would require game totals data
        pass
    
    elif stack_type == "stars_value":
        # 2-3 players over $9K, rest under $5K
        stars = df[df['Salary'] >= 9000].index
        value = df[df['Salary'] <= 5000].index
        
        problem += pulp.lpSum([player_vars[idx] for idx in stars]) >= 2
        problem += pulp.lpSum([player_vars[idx] for idx in stars]) <= 3
        problem += pulp.lpSum([player_vars[idx] for idx in value]) >= 3
```

### 3. Minutes/Usage Correlation
```python
def add_minutes_constraints(problem, df, player_vars):
    """
    NBA-specific: Ensure lineup has enough projected minutes
    Target: 250+ total minutes (8 players * 31+ min avg)
    """
    if 'Minutes' in df.columns:
        problem += pulp.lpSum([
            df.at[idx, 'Minutes'] * player_vars[idx] 
            for idx in df.index
        ]) >= 250
```

---

## 🎯 Recommended Approach:

Given the file's massive size (7,160 lines), I recommend:

### Option A: Full Refactor (Time: 2-3 hours)
- Systematically update all 7,160 lines
- Keep all advanced features (genetic algorithm, risk engine, etc.)
- Fully test and debug

### Option B: Create New Streamlined NBA Version (Time: 30-45 min) ✅ **RECOMMENDED**
- Keep core genetic algorithm engine
- Remove NFL-specific complexity
- Focus on essential NBA features
- 2,000-3,000 lines instead of 7,160

### Option C: Hybrid Approach
- Update existing file's core sections (Done ✅)
- Create separate NBA stack engine module
- Keep most genetic algorithm logic as-is

---

## 📝 Next Steps:

1. **Decide on approach** (A, B, or C)
2. **Test with your nba_slate_optimized CSV**
3. **Add NBA-specific features** (PG-C correlation, pace adjustments)
4. **Integrate with your API data fetcher**

---

## 💡 Quick Test Command:

```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION
python3 "nba_sportsdata.io_gentic algo.py"
```

---

**Current Status**: Core settings updated, needs position eligibility logic and stack enforcement updated for NBA.

