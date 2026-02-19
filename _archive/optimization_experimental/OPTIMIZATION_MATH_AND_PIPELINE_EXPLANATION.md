# Optimization Math & Python Backend Pipeline Explanation

## Overview

This document explains the mathematical foundations and Python backend pipeline for the DFS (Daily Fantasy Sports) optimizer in the `6_OPTIMIZATION` folder. The system uses a **hybrid approach** combining **Linear Programming (PuLP)** for optimal lineup generation and **Genetic Algorithms** for diversity.

---

## 1. Core Optimization Concept

### Problem Statement

The optimizer solves a **constrained optimization problem**:

**Goal:** Select a lineup of players that maximizes projected fantasy points while satisfying:
- Salary cap constraint (e.g., $50,000 for NFL, $50,000 for NBA)
- Position requirements (e.g., 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DST for NFL)
- Stacking constraints (optional: multiple players from same team)
- Player inclusion/exclusion rules
- Exposure limits (min/max ownership percentages)

### Mathematical Formulation

This is a **Binary Integer Linear Programming (BILP)** problem:

```
Maximize: Σ (player_points[i] × x[i])
Subject to:
  Σ (player_salary[i] × x[i]) ≤ SALARY_CAP
  Σ x[i] = LINEUP_SIZE
  Position constraints (e.g., Σ x[i] for QB = 1)
  Stacking constraints (if applicable)
  x[i] ∈ {0, 1}  (binary: player selected or not)
```

Where:
- `x[i]` = binary variable (1 if player i is selected, 0 otherwise)
- `player_points[i]` = projected fantasy points for player i
- `player_salary[i]` = salary cost for player i

---

## 2. PuLP Linear Programming (Core Solver)

### What is PuLP?

**PuLP** (Python Linear Programming) is a library that solves linear programming problems using the **Simplex Method** or **Branch-and-Bound** algorithms. It finds the mathematically optimal solution.

### Implementation Flow

```python
# Step 1: Create optimization problem
problem = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)

# Step 2: Create binary variables for each player
player_vars = {}
for idx in df.index:
    player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')

# Step 3: Define objective function (maximize points)
problem += pulp.lpSum([
    df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] 
    for idx in df.index
])

# Step 4: Add constraints
# Constraint 1: Total players = 9 (NFL) or 8 (NBA)
problem += pulp.lpSum([player_vars[idx] for idx in df.index]) == 9

# Constraint 2: Salary cap
problem += pulp.lpSum([
    df.at[idx, 'Salary'] * player_vars[idx] 
    for idx in df.index
]) <= 50000

# Constraint 3: Position requirements
problem += pulp.lpSum([
    player_vars[idx] for idx in df.index 
    if df.at[idx, 'Position'] == 'QB'
]) == 1  # Exactly 1 QB

# Constraint 4: Stacking (if applicable)
if stack_type == 'qb_2wr_te':
    # Ensure QB + 2 WRs + TE from same team
    team_players = df[df['Team'] == selected_team].index
    problem += pulp.lpSum([player_vars[idx] for idx in team_players]) >= 4

# Step 5: Solve
problem.solve(pulp.PULP_CBC_CMD(msg=0))

# Step 6: Extract solution
selected_players = [idx for idx in df.index if player_vars[idx].varValue == 1]
lineup = df.loc[selected_players]
```

### Why PuLP?

**Advantages:**
- ✅ Finds **mathematically optimal** solution
- ✅ Fast (solves in <1 second per lineup)
- ✅ Handles complex constraints easily
- ✅ Deterministic (same inputs = same output)

**Disadvantages:**
- ❌ Always returns the same lineup (no diversity)
- ❌ Can't handle soft constraints (preferences)
- ❌ No randomness for multi-lineup generation

**Solution:** Combine with Genetic Algorithm for diversity!

---

## 3. Genetic Algorithm (Diversity Engine)

### The Diversity Problem

**Problem:** PuLP alone generates THE optimal lineup, but we need 20-100 diverse lineups for multi-entry contests.

**Solution:** Genetic Algorithm adds controlled randomness while maintaining quality.

### How It Works

#### Phase 1: Initial Population Creation

Generate 3x more lineups than requested using controlled randomness:

```python
# Add 35-70% random noise to projections
diversity_factor = random.uniform(0.35, 0.70)
noise = np.random.lognormal(0, diversity_factor, len(df))

# Apply noise to projections
df['Fantasy_Points_Adjusted'] = df['Fantasy_Points'] * noise

# Randomly boost 3-7 players
num_boosts = random.randint(3, 7)
boost_indices = np.random.choice(df.index, num_boosts, replace=False)
for idx in boost_indices:
    df.at[idx, 'Fantasy_Points_Adjusted'] *= random.uniform(1.15, 1.40)

# Randomly penalize 2-4 players
num_penalties = random.randint(2, 4)
penalty_indices = np.random.choice(df.index, num_penalties, replace=False)
for idx in penalty_indices:
    df.at[idx, 'Fantasy_Points_Adjusted'] *= random.uniform(0.60, 0.85)

# Solve with adjusted projections → different lineup
lineup = optimize_with_pulp(df_with_adjusted_points)
```

**Result:** Each optimization sees slightly different player values → different lineups

#### Phase 2: Evolution (Genetic Operators)

Run 3 generations of evolution:

```python
def evolve_population(population, generations=3):
    for gen in range(generations):
        # 1. Tournament Selection: Keep best 50%
        elite = select_top_50_percent(population)
        
        # 2. Crossover: Mix players from two parent lineups
        children = []
        for i in range(len(elite) // 2):
            parent1 = elite[i]
            parent2 = elite[i + len(elite) // 2]
            child = crossover_lineups(parent1, parent2)  # Mix players
            children.append(child)
        
        # 3. Mutation: Randomly change 1-2 positions
        for child in children:
            if random.random() < 0.3:  # 30% mutation rate
                child = mutate_lineup(child)  # Replace random player
        
        # 4. Diversity Enforcement: Remove too-similar lineups
        population = ensure_diversity(elite + children, min_similarity=0.3)
    
    return population
```

**Genetic Operators:**

1. **Selection:** Tournament selection keeps best-performing lineups
2. **Crossover:** Combines players from two parent lineups
   ```python
   def crossover_lineups(parent1, parent2):
       # Take 60% from parent1, 40% from parent2
       child = parent1.copy()
       positions_to_replace = random.sample(parent2.columns, k=len(parent2) * 0.4)
       for pos in positions_to_replace:
           child[pos] = parent2[pos]
       return child
   ```

3. **Mutation:** Randomly replaces 1-2 players
   ```python
   def mutate_lineup(lineup):
       # Replace random player with alternative
       position_to_mutate = random.choice(['RB', 'WR', 'TE'])
       current_player = lineup[lineup['Position'] == position_to_mutate].iloc[0]
       alternatives = df[(df['Position'] == position_to_mutate) & 
                         (df['Name'] != current_player['Name'])]
       replacement = alternatives.sample(1).iloc[0]
       lineup.replace(current_player, replacement)
       return lineup
   ```

#### Phase 3: Diverse Subset Selection

Select most diverse lineups using maximal diversity algorithm:

```python
def select_diverse_subset(population, num_lineups):
    selected = [population[0]]  # Start with first lineup
    
    while len(selected) < num_lineups:
        best_candidate = None
        max_min_distance = -1
        
        for candidate in population:
            if candidate in selected:
                continue
            
            # Calculate minimum distance to any selected lineup
            min_distance = min([
                lineup_distance(candidate, existing) 
                for existing in selected
            ])
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_candidate = candidate
        
        selected.append(best_candidate)
    
    return selected
```

**Distance Metric:**
```python
def lineup_distance(lineup1, lineup2):
    # Hamming distance: number of different players
    players1 = set(lineup1['Name'].values)
    players2 = set(lineup2['Name'].values)
    return len(players1.symmetric_difference(players2))  # Different players
```

---

## 4. Python Backend Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Data Collection                                     │
│     ├─ SportsData.io API                               │
│     ├─ Player projections                               │
│     ├─ DraftKings salaries                              │
│     └─ Injury reports                                   │
│                                                         │
│  2. Data Preprocessing                                  │
│     ├─ Merge projections + salaries                     │
│     ├─ Filter injured players (OUT, DOUBTFUL)          │
│     ├─ Calculate value metrics (points/$)               │
│     └─ Add contest-specific enhancements                │
│                                                         │
│  3. Optimization Engine                                 │
│     ├─ PuLP Linear Programming (core solver)              │
│     ├─ Genetic Algorithm (diversity)                    │
│     └─ Advanced Quantitative Methods (optional)          │
│                                                         │
│  4. Post-Processing                                     │
│     ├─ Deduplication                                    │
│     ├─ Exposure tracking                                 │
│     ├─ Position assignment (FLEX handling)               │
│     └─ Export to DraftKings CSV format                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Detailed Pipeline Steps

#### Step 1: Data Collection

**File:** `create_nfl_week_data.py` or `load_nfl_data_enhanced.py`

```python
# Fetch from SportsData.io API
api = SportsDataNFLAPI()

# Get projections
projections = api.get_player_projections_by_week(
    season='2025REG',
    week=7
)

# Get DraftKings salaries
slates = api.get_dfs_slates_by_date(
    date='2025-10-20'
)

# Get injuries
injuries = api.get_injuries_by_week(
    season='2025REG',
    week=7
)
```

**Output:** Raw CSV files with player data

#### Step 2: Data Preprocessing

**File:** `load_nfl_data_enhanced.py`

```python
def preprocess_data(df_players):
    # 1. Merge projections with salaries
    df = merge_projections_and_salaries(projections, slates)
    
    # 2. Filter injured players
    df = filter_injured_players(df, injuries)
    # Removes: OUT, DOUBTFUL status
    
    # 3. Calculate value metrics
    df['Value'] = df['Fantasy_Points'] / (df['Salary'] / 1000)
    df['PointsPerK'] = df['Fantasy_Points'] / (df['Salary'] / 1000)
    
    # 4. Contest-specific enhancements
    if contest_mode == 'gpp':
        # GPP: Use ceiling projections
        df['Fantasy_Points'] = df['Ceiling']  # 80th percentile
        # Boost low-owned players
        low_owned = df['Ownership'] < 15
        df.loc[low_owned, 'Fantasy_Points'] *= 1.1
    elif contest_mode == 'cash':
        # Cash: Use floor projections
        df['Fantasy_Points'] = df['Floor']  # 20th percentile
        # Boost high-owned safe plays
        high_owned = df['Ownership'] > 30
        df.loc[high_owned, 'Fantasy_Points'] *= 1.05
    
    return df
```

#### Step 3: Optimization Engine

**File:** `optimizer.py` or `genetic_algo_nfl_optimizer.py`

**Main Entry Point:**
```python
class OptimizationWorker(QThread):
    def optimize_lineups(self):
        # 1. Preprocess data
        df_filtered = self.preprocess_data()
        
        # 2. Generate candidate lineups
        all_candidates = []
        
        # Generate 20x more candidates than requested
        total_candidates = self.num_lineups * 20
        
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            
            for stack_type in self.stack_settings:
                for i in range(candidates_per_stack):
                    # Apply diversity noise
                    df_variant = apply_diversity_noise(df_filtered.copy())
                    
                    # Optimize with PuLP
                    future = executor.submit(
                        optimize_single_lineup,
                        (df_variant, stack_type, ...)
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                lineup = future.result()
                if not lineup.empty:
                    all_candidates.append(lineup)
        
        # 3. Apply genetic algorithm for diversity
        if self.num_lineups >= 5:
            # Use genetic algorithm
            genetic_engine = GeneticDiversityEngine(...)
            diverse_lineups = genetic_engine.create_diverse_lineups(
                num_lineups=self.num_lineups,
                stack_type=stack_type
            )
        else:
            # Simple deduplication
            diverse_lineups = deduplicate_lineups(all_candidates)
        
        # 4. Select final lineups
        final_lineups = select_best_unique(
            diverse_lineups, 
            num_lineups=self.num_lineups
        )
        
        return final_lineups
```

**Core Optimization Function:**
```python
def optimize_single_lineup(args):
    df, stack_type, team_selections = args
    
    # Create PuLP problem
    problem = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)
    player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') 
                   for idx in df.index}
    
    # Objective: Maximize points
    problem += pulp.lpSum([
        df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] 
        for idx in df.index
    ])
    
    # Constraints
    problem += pulp.lpSum(player_vars.values()) == 9  # 9 players
    problem += pulp.lpSum([
        df.at[idx, 'Salary'] * player_vars[idx] 
        for idx in df.index
    ]) <= 50000  # Salary cap
    
    # Position constraints
    problem += pulp.lpSum([
        player_vars[idx] for idx in df.index 
        if df.at[idx, 'Position'] == 'QB'
    ]) == 1  # 1 QB
    
    # ... more position constraints ...
    
    # Stacking constraints
    if stack_type == 'qb_2wr_te':
        # Ensure QB + 2 WRs + TE from same team
        selected_team = team_selections[stack_type]
        team_players = df[df['Team'] == selected_team].index
        
        qb_players = df[(df['Team'] == selected_team) & 
                       (df['Position'] == 'QB')].index
        wr_players = df[(df['Team'] == selected_team) & 
                       (df['Position'] == 'WR')].index
        te_players = df[(df['Team'] == selected_team) & 
                       (df['Position'] == 'TE')].index
        
        problem += pulp.lpSum([player_vars[idx] for idx in qb_players]) == 1
        problem += pulp.lpSum([player_vars[idx] for idx in wr_players]) >= 2
        problem += pulp.lpSum([player_vars[idx] for idx in te_players]) >= 1
    
    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    if pulp.LpStatus[problem.status] == 'Optimal':
        selected = [idx for idx in df.index 
                   if player_vars[idx].varValue == 1]
        return df.loc[selected]
    else:
        return pd.DataFrame()  # No solution
```

#### Step 4: Post-Processing

**File:** `optimizer.py` (export functions)

```python
def post_process_lineups(lineups):
    # 1. Fix position ordering (critical for FLEX)
    for lineup in lineups:
        lineup = fix_lineup_position_order(lineup)
        # Sort by projection before assigning positions
        # Ensures best players in primary slots, not FLEX
    
    # 2. Deduplication
    unique_lineups = []
    lineup_hashes = set()
    for lineup in lineups:
        lineup_hash = hash_lineup(lineup)
        if lineup_hash not in lineup_hashes:
            lineup_hashes.add(lineup_hash)
            unique_lineups.append(lineup)
    
    # 3. Exposure tracking
    team_exposure = defaultdict(int)
    stack_exposure = defaultdict(int)
    for lineup in unique_lineups:
        for team in lineup['Team'].unique():
            team_exposure[team] += 1
        stack_exposure[lineup['stack_type']] += 1
    
    # 4. Format for DraftKings CSV
    dk_lineups = []
    for lineup in unique_lineups:
        dk_lineup = format_for_draftkings(lineup)
        dk_lineups.append(dk_lineup)
    
    return dk_lineups, team_exposure, stack_exposure
```

---

## 5. Advanced Quantitative Features

### Overview

The system includes advanced quantitative finance techniques (optional):

**File:** `advanced_quant_optimizer.py` and `portfolio_optimization.py`

### Key Features

#### 1. GARCH Volatility Estimation

Models time-varying volatility of player performance:

```python
from arch import arch_model

def estimate_volatility(historical_points):
    # Fit GARCH(1,1) model
    model = arch_model(historical_points, vol='Garch', p=1, q=1)
    fitted = model.fit()
    return fitted.conditional_volatility[-1]  # Latest volatility
```

#### 2. Risk-Adjusted Optimization

Instead of maximizing raw points, maximize risk-adjusted returns:

```python
# Sharpe Ratio optimization
def calculate_sharpe_ratio(points, volatility, risk_free_rate=0):
    return (points - risk_free_rate) / volatility

# Risk-adjusted points
df['risk_adjusted_points'] = df['Predicted_DK_Points'] / (1 + df['volatility'])

# Optimize using risk-adjusted points
problem += pulp.lpSum([
    df.at[idx, 'risk_adjusted_points'] * player_vars[idx] 
    for idx in df.index
])
```

#### 3. Monte Carlo Simulation

Simulates 10,000+ scenarios for robust risk assessment:

```python
def monte_carlo_simulation(df, num_simulations=10000):
    results = []
    
    for _ in range(num_simulations):
        # Sample random outcomes
        simulated_points = np.random.normal(
            df['Predicted_DK_Points'],
            df['volatility']
        )
        
        # Optimize with simulated points
        lineup = optimize_with_pulp(df, simulated_points)
        results.append(lineup['total_points'])
    
    # Calculate statistics
    mean_points = np.mean(results)
    var_95 = np.percentile(results, 5)  # 95% VaR
    cvar_95 = np.mean(results[results <= var_95])  # CVaR
    
    return {
        'expected_points': mean_points,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'distribution': results
    }
```

#### 4. Value at Risk (VaR) & Conditional VaR

Measures downside risk:

```python
def calculate_var(portfolio_returns, confidence=0.95):
    # VaR: Maximum loss at confidence level
    return np.percentile(portfolio_returns, (1 - confidence) * 100)

def calculate_cvar(portfolio_returns, confidence=0.95):
    # CVaR: Expected loss given VaR threshold exceeded
    var = calculate_var(portfolio_returns, confidence)
    return np.mean(portfolio_returns[portfolio_returns <= var])
```

#### 5. Kelly Criterion Position Sizing

Optimal bet sizing based on expected win rate:

```python
def kelly_criterion(win_probability, win_amount, loss_amount):
    # Kelly fraction = (p × b - q) / b
    # where p = win prob, q = 1-p, b = win/loss ratio
    b = win_amount / loss_amount
    q = 1 - win_probability
    kelly = (win_probability * b - q) / b
    return max(0, min(kelly, 0.25))  # Cap at 25% of bankroll
```

#### 6. Copula-Based Dependency Modeling

Models complex player correlations:

```python
from copulas.multivariate import GaussianMultivariate

def model_player_correlations(df):
    # Fit Gaussian copula to player performance
    copula = GaussianMultivariate()
    copula.fit(df[['Player1_Points', 'Player2_Points', ...]])
    
    # Generate correlated samples
    correlated_samples = copula.sample(1000)
    
    return correlated_samples
```

---

## 6. Key Mathematical Concepts Summary

### Linear Programming

**Form:** `Ax ≤ b, x ≥ 0, maximize c^T x`

- **A** = constraint matrix (salary, positions, etc.)
- **b** = constraint bounds (salary cap, position limits)
- **c** = objective coefficients (player points)
- **x** = decision variables (player selections)

**Solver:** PuLP uses CBC (COIN-OR Branch and Cut) algorithm

### Genetic Algorithm

**Evolutionary Process:**
1. **Initialization:** Random population
2. **Selection:** Keep best performers (tournament selection)
3. **Crossover:** Combine parent solutions
4. **Mutation:** Random changes
5. **Evaluation:** Fitness function (projected points)
6. **Termination:** After N generations or convergence

**Fitness Function:**
```
Fitness(lineup) = Σ(player_points) - penalty(over_salary) - penalty(duplicate)
```

### Constraint Satisfaction

**Hard Constraints (must satisfy):**
- Salary cap
- Position requirements
- Player count

**Soft Constraints (preferences):**
- Stacking (can be relaxed)
- Exposure limits (can be violated slightly)
- Min salary usage (preference, not requirement)

---

## 7. Performance Characteristics

### Time Complexity

- **PuLP Optimization:** O(n × m) where n = players, m = constraints
  - Typically <1 second per lineup
- **Genetic Algorithm:** O(g × p × n) where g = generations, p = population, n = players
  - Typically 5-30 seconds for 20 lineups

### Space Complexity

- **Player Data:** O(n) where n = number of players (~200-500)
- **Lineup Storage:** O(l × n) where l = number of lineups
- **Genetic Population:** O(p × n) where p = population size (3x requested)

### Scalability

- **Single Lineup:** <1 second
- **20 Lineups:** 5-30 seconds
- **100 Lineups:** 1-5 minutes
- **Parallel Processing:** Uses ThreadPoolExecutor for 4-8x speedup

---

## 8. Integration Points

### Data Flow

```
SportsData.io API
    ↓
CSV Files (projections, salaries, injuries)
    ↓
Preprocessing (merge, filter, enhance)
    ↓
Optimization Engine (PuLP + Genetic)
    ↓
Post-Processing (deduplication, formatting)
    ↓
DraftKings CSV Export
```

### Key Files

1. **`optimizer.py`** - Main optimizer with PuLP
2. **`genetic_algo_nfl_optimizer.py`** - Genetic algorithm implementation
3. **`advanced_quant_optimizer.py`** - Advanced quantitative methods
4. **`portfolio_optimization.py`** - Portfolio theory integration
5. **`load_nfl_data_enhanced.py`** - Data preprocessing
6. **`create_nfl_week_data.py`** - Data collection

---

## 9. Example Workflow

### Complete Pipeline Example

```python
# Step 1: Load data
df = load_nfl_data_enhanced.load_data('nfl_week7_gpp_enhanced.csv')

# Step 2: Configure optimization
worker = OptimizationWorker(
    df_players=df,
    salary_cap=50000,
    position_limits={'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1},
    num_lineups=20,
    stack_settings=['qb_2wr_te', 'qb_wr'],
    min_exposure=0.0,
    max_exposure=1.0
)

# Step 3: Run optimization
results, team_exposure, stack_exposure = worker.optimize_lineups()

# Step 4: Export
export_to_draftkings_csv(results, 'dk_upload_week7.csv')
```

### Output

- **20 unique lineups** with:
  - Maximum projected points (PuLP optimal)
  - High diversity (Genetic algorithm)
  - Proper position assignment
  - Stacking constraints satisfied
  - Exposure limits respected

---

## 10. Conclusion

The optimization system combines:

1. **Mathematical Optimization (PuLP):** Finds optimal solutions
2. **Evolutionary Algorithms (Genetic):** Ensures diversity
3. **Quantitative Finance (Advanced):** Risk-aware optimization
4. **Parallel Processing:** Fast execution
5. **Robust Pipeline:** Handles data collection → optimization → export

This hybrid approach provides both **optimality** (from PuLP) and **diversity** (from Genetic Algorithm), making it ideal for multi-entry DFS contests.

---

**Last Updated:** 2025
**Version:** 2.0
**Status:** Production Ready


