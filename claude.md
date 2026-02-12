# CLAUDE.md — UrSim DFS Optimization System

## Project Overview

UrSim is an NFL/NBA/MLB DraftKings optimization platform that combines mathematical optimization, machine learning, and quantitative finance methods to generate optimal daily fantasy sports lineups. The system's **primary differentiator** is its institutional-grade quantitative engine — the same mathematical frameworks used by hedge funds and quant trading desks (Monte Carlo simulation, Kelly Criterion, VaR/CVaR, Mean-Variance optimization), applied to DFS contest optimization.

**Positioning**: "You're not just building lineups — you're managing a portfolio." While competitors offer knapsack solvers with projection inputs, UrSim treats every lineup as a financial asset and every contest entry as a portfolio allocation decision.

---

## Build & Run

```bash
# Web Application (primary development mode)
cd web_optimizer && npm run dev          # Starts both servers concurrently
# Backend: http://localhost:5001   Frontend: http://localhost:3000

# Individual servers
cd web_optimizer && node server/index.js                    # Backend only (port 5001)
cd web_optimizer/client && npm run dev                       # Frontend only (port 3000)

# Production build
cd web_optimizer/client && npx vite build                    # Output: client/build/

# Python Optimization Pipeline (standalone)
cd 6_OPTIMIZATION && python3 pulp_lineup_optimizer.py        # PuLP BILP solver
cd 6_OPTIMIZATION && python3 makrovchain_optimizer.py        # Full Markov chain optimizer
```

**Tech Stack**: React 18 + TypeScript + Vite 6.4 + Radix UI (frontend) | Node.js + Express 4.17 + WebSocket (backend) | Python 3.8+ + PuLP + Pandas + NumPy + SciPy (optimization) | StackingRegressor + XGBoost (ML pipeline)

---

## Repository Structure

```
├── 6_OPTIMIZATION/                        # Core optimization engines
│   ├── makrovchain_optimizer.py           # Full Markov chain optimizer (8900+ lines) — PRODUCTION
│   ├── pulp_lineup_optimizer.py           # PuLP BILP solver — PRODUCTION
│   ├── advanced_quant_optimizer.py        # Copula, regime detection, MC — NEEDS INTEGRATION
│   ├── dfs_risk_engine.py                 # Kelly, GARCH, VaR, Sharpe, correlation — PARTIAL
│   └── probability_enhanced_optimizer.py  # Ownership probability mapping — PARTIAL
├── web_optimizer/
│   ├── server/
│   │   ├── index.js                       # Express API routes + Python subprocess spawning
│   │   ├── quant-engine.js                # JS Quant Engine (MC, Kelly, VaR, Sharpe, portfolio) — NBA PRODUCTION
│   │   ├── nba-optimizer.js               # NBA optimizer — QUANT INTEGRATED
│   │   ├── nfl-optimizer.js               # NFL optimizer — QUANT INTEGRATED
│   │   ├── optimizer.js                   # MLB optimizer — QUANT INTEGRATED
│   │   └── makrov_cli_adapter.py          # Python CLI adapter for web (multi-stack, team exposures)
│   └── client/src/components/
│       └── optimizer/
│           ├── DFSOptimizer.tsx            # Main orchestrator (multi-build, state management)
│           ├── BuildControlBar.tsx         # Merged build tabs + settings + CTA
│           ├── GameSlate.tsx               # Game context horizontal cards
│           ├── PlayerTable.tsx             # Dense data table with color-coded stats
│           ├── TeamStacksTab.tsx           # Multi-stack team selection + per-team exposure
│           ├── AdvancedQuantTab.tsx         # Quant settings UI (strategy, MC, Kelly, GARCH, copula)
│           ├── Sidebar.tsx                 # Compact lineup review panel
│           ├── hooks/useOptimizer.ts       # Optimization hook (request construction)
│           ├── hooks/useBuildManager.ts    # Multi-build state management
│           ├── hooks/useFileUpload.ts      # CSV upload + sport detection
│           └── types.ts                    # TypeScript interfaces + defaults
```

---

## Competitive Landscape & Strategic Positioning

### Market Analysis (2025-2026)

| Feature | SaberSim | Stokastic | LineupHQ | FantasyLabs | The Solver | **UrSim** |
|---------|----------|-----------|----------|-------------|------------|-----------|
| Monte Carlo Simulation | Play-by-play sims | Contest sims | No | SimLabs | 20K contest sims | **2K-10K per lineup** |
| Correlation (Implicit) | Via simulation | Partial | No | Partial | Partial | **Via stacking + MC** |
| Correlation (Explicit Stacking) | Basic | Basic | Best-in-class | Good | Good | **Multi-stack (2,3,4,5)** |
| Ownership Leverage | Yes | Best (Boom/Bust) | Yes | Yes (real-time) | Yes (ETR sync) | **GPP leverage scoring** |
| Portfolio Optimization | Diversifier tool | No | Exposure controls | No | No | **Cross-lineup Sharpe, Herfindahl** |
| Kelly Criterion / Bankroll | No | No | No | No | Tracker only | **Kelly exposure limits** |
| VaR/CVaR Risk Metrics | No | No | No | No | No | **Per-lineup + portfolio** |
| Sharpe Ratio per Lineup | No | No | No | No | No | **MC-derived per lineup** |
| Mean-Variance Strategy | No | No | No | No | No | **5 optimization strategies** |
| GARCH Volatility | No | No | No | No | No | **Python engine (needs wiring)** |
| Late Swap | No | No | No | No | DK support | Not yet |
| Multi-Sport | 15+ | 8+ | 15 | NFL/NBA/MLB | NFL/NBA/PGA | **NFL/NBA/MLB** |

### UrSim's Competitive Moat

**What no competitor offers that we do:**
1. **Per-lineup risk metrics** — VaR, CVaR, Sharpe ratio computed via Monte Carlo for every generated lineup
2. **Kelly-optimal exposure sizing** — Player exposure limits derived from edge/variance ratio, not arbitrary caps
3. **5 named optimization strategies** — Combined, Kelly, Mean-Variance, Risk Parity, Equal Weight — each with distinct mathematical objectives
4. **Portfolio-level analysis** — Cross-lineup Sharpe ratio, exposure Herfindahl concentration index, average pairwise uniqueness
5. **Strategy-aware contest modes** — Cash games use floor/consistency/Sharpe; GPPs use ceiling/leverage/ownership fade

**Key whitespace we should exploit:**
1. **Integrated bankroll management** — No tool automates contest selection + entry sizing based on Kelly + edge estimation
2. **Transparent game-theory metrics** — Only Stokastic provides leverage scores; most hide behind black-box optimization
3. **Correlation visualization** — Interactive heat maps, game-script probability curves
4. **Cross-contest portfolio optimization** — Optimizing across cash + GPP entries simultaneously

### How Competitors Present Quant Features (UX Patterns)

| Pattern | Who Does It | How to Apply |
|---------|-------------|-------------|
| **One-number summaries** | Stokastic (Leverage Score) | Show single "Quant Score" per player combining all signals |
| **Boom/Bust probability bars** | Stokastic | Horizontal bars: green (boom%) + red (bust%) per player |
| **Ownership vs Optimal scatter** | SaberSim | Plot ownership (x) vs optimal lineup % (y), above diagonal = leverage |
| **Distribution curves** | SaberSim | Show player projection as a curve, not a single number |
| **Color-coded signals** | Universal | Green/yellow/red for Sharpe, leverage, value thresholds |
| **Progressive disclosure** | SaberSim | Basic: "Build" button; Advanced: full quant parameter panel |
| **Pre-built contest profiles** | SaberSim, FantasyLabs | "GPP Mode" / "Cash Mode" buttons that set all params at once |
| **Portfolio exposure dashboards** | SaberSim | Heat map of player exposure across all entries |

---

## Quantitative Engine — Architecture & Implementation

### The Quant Engine Is the Product

The system's value proposition is **not** "another lineup optimizer" — it's a **quantitative risk management platform for DFS**. Every feature reinforces this positioning. The quant engine drives three core workflows:

1. **Player Selection** — Quantitative signals (volatility, correlations, Kelly sizing) to select players, not just "highest projected points"
2. **Lineup Construction** — Portfolio theory (mean-variance, risk parity) rather than greedy heuristics
3. **Multi-Entry Portfolio Optimization** — Treat lineups like assets, optimize for maximum risk-adjusted return across all entries

### Implementation Status

| Feature | Frontend UI | Python Backend | JS Backend | Status |
|---------|-------------|---------------|------------|--------|
| **PuLP BILP Solver** | — | ✅ Full | — | **Production** |
| **Genetic Algorithm** | — | ✅ Full | — | **Production** |
| **Exposure Management** | ✅ Full | ✅ Full | ✅ Full | **Production** |
| **Team Stacking (Multi-Stack)** | ✅ Full (2,3,4,5) | ✅ Full | ✅ Full | **Production** |
| **Per-Team Exposures** | ✅ Full | ✅ Full | — | **Production (Python path)** |
| **Kelly Criterion** | ✅ UI Controls | ⚠️ Partial | ✅ NBA | **NBA Production** |
| **Monte Carlo Simulation** | ✅ UI Controls | ⚠️ Exists, not called | ✅ NBA (2K-10K sims) | **NBA Production** |
| **Mean-Variance (Markowitz)** | ✅ UI Dropdown | ❌ Research | ✅ NBA (salary-aware) | **NBA Production** |
| **Risk Parity** | ✅ UI Dropdown | ❌ Not implemented | ✅ NBA (vol-normalized) | **NBA Production** |
| **VaR / CVaR** | ✅ UI Controls | ⚠️ Partial | ✅ NBA (MC-based) | **NBA Production** |
| **Ownership Leverage** | — | — | ✅ NBA (GPP scoring) | **NBA Production** |
| **Sharpe Ratio (per lineup)** | — | — | ✅ NBA (MC-derived) | **NBA Production** |
| **Portfolio Analysis** | — | — | ✅ NBA (Sharpe, Herfindahl) | **NBA Production** |
| **GARCH Volatility** | ✅ UI Controls | ⚠️ Optional (`arch`) | ❌ | **Needs Integration** |
| **Copula Dependency** | ✅ UI Controls | ⚠️ Optional (`copulas`) | ❌ | **Needs Implementation** |
| **Regime Detection** | ❌ No UI | ⚠️ KMeans code | ❌ | **Needs Integration** |

### Data Flow Architecture

```
User (AdvancedQuantTab.tsx)
  → AdvancedQuantSettings interface (types.ts)
    → useOptimizer hook merges with DEFAULT_ADVANCED_QUANT_SETTINGS
      → POST /api/optimize { advancedQuantSettings, contestMode, ... }
        → Express handler (index.js:782)

NBA path (ALL modes — PuLP ILP primary):
  → Python makrov_cli_adapter.py via subprocess
    → PuLP ILP solver (immutable projections as objective coefficients)
    → Iterative solving with exclusion constraints for diversity
    → Monte Carlo evaluation per lineup (2K sims → VaR, Sharpe, percentiles)
    → Multi-stack + team exposures via ILP constraints
    → Fallback: JS NBAOptimizer (salary-aware greedy heuristic)

NBA path (quant enabled — post-processing layer):
  → PuLP ILP generates lineups (same as above)
  → JS QuantEngine layers on:
    → monteCarloLineup() — 2K-10K sims per lineup (higher fidelity than Python MC)
    → analyzePortfolio() — cross-lineup Sharpe, uniqueness, exposure concentration
    → Sort by Sharpe ratio
  → Response: lineups[] with quantMetrics + summary.portfolioMetrics

NFL/MLB path:
  → Sport-specific JS optimizer (quant NOT yet wired)
```

### quant-engine.js — Module Reference

| Method | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `scorePlayersQuant(players, contestMode)` | Pre-compute quant score per player | Player[], 'gpp'/'cash' | Map<id, {quantScore, sharpe, leverage, kellyExposure, ceilingProb}> |
| `selectPlayerQuant(available, scores, strategy)` | Quant-weighted player selection | Player[], Map, strategy | Player |
| `kellyExposureLimits(players, defaultMax)` | Kelly-optimal exposure caps | Player[], number | Map<id, maxExposure%> |
| `monteCarloLineup(players, numSims?)` | Simulate lineup outcomes | Player[], number? | {mean, stdDev, sharpeRatio, valueAtRisk, conditionalVaR, ceilingProbability, percentiles} |
| `analyzePortfolio(lineups)` | Cross-lineup portfolio metrics | Lineup[] | {sharpeRatio, avgUniqueness, maxExposure, exposureConcentration} |

**Strategy Scoring Formulas:**

| Strategy | Cash Mode | GPP Mode |
|----------|-----------|----------|
| `combined` | 0.4×Sharpe + 0.3×(floor/salary) + 0.2×(proj/salary) - 0.1×(stdDev×λ) | 0.35×leverage + 0.25×Sharpe + 0.2×(ceilingProb×100) + 0.2×(ceiling/salary) |
| `kelly` | 0.4×Sharpe + 3×kellyExposure + 0.3×(proj/salary) | Same |
| `mean_variance` | (proj/salary) - λ×stdDev×0.5 + 0.5×Sharpe | Same |
| `risk_parity` | proj / stdDev | Same |
| `equal_weight` | proj / salary | Same |

Where: `leverage = ceiling × (1 - ownership%) / salary`, `Sharpe = (proj/salary×1000) / stdDev`, `Kelly = edge / variance`

---

## How to Maximize Quant Features as the Selling Point

### 1. Player Selection — Quantitative Player Scoring

**Current (quant enabled)**: Every player gets a composite `quantScore` from `scorePlayersQuant()` that blends Sharpe ratio, ownership leverage, Kelly exposure, and ceiling probability. Selection uses weighted-random from top-scored pool.

**Next Steps**:
- **Correlation Contribution**: Add marginal Sharpe improvement when adding player to partial lineup (requires covariance matrix)
- **Boom/Bust Probabilities**: Compute from MC simulation per player — `boomProb = P(points > 1.5x projection)`, `bustProb = P(points < 0.5x projection)`
- **Display in PlayerTable**: Add columns for quantScore, boomProb, leverage with color-coded cells (green/yellow/red thresholds)
- **Pre-built filters**: "Show only leverage plays" (quantScore > threshold AND ownership < 10%)

### 2. Lineup Construction — Portfolio Theory Applied

**Current (quant enabled)**: Position-by-position construction using quant-scored selection, then post-hoc MC analysis.

**Next Steps**:
- **Full Markowitz**: Replace greedy construction with `maximize μᵀx - λ√(xᵀΣx)` using player covariance matrix
- **Contest-Aware Objectives**: Cash → minimize variance for target return; GPP → maximize `ceiling × (1-ownership)` subject to constraints
- **Cholesky Correlation** (SaberSim's approach): Decompose player covariance matrix → generate correlated simulation vectors → lineups that naturally capture game-script correlations without explicit stacking rules
- **Risk Tolerance Mapping**: `riskTolerance` slider (0.1-2.0) maps to λ in mean-variance; higher λ = more conservative

### 3. Multi-Entry Portfolio — The Key Differentiator

**Current**: `analyzePortfolio()` computes cross-lineup Sharpe, uniqueness, exposure concentration post-hoc.

**Next Steps**:
- **Active Portfolio Construction**: Don't just analyze — *optimize* the portfolio. After generating candidate pool (3x requested), select subset that maximizes portfolio Sharpe while maintaining diversity
- **Correlation Management**: Compute pairwise lineup outcome correlations from MC sims; penalize highly-correlated lineup pairs
- **Risk Parity**: Weight entries so each lineup contributes equal variance to the portfolio
- **Kelly Entry Sizing**: Given bankroll + contest structure, compute optimal number of entries using Kelly criterion

### 4. GARCH Volatility — Dynamic Risk

**Current**: Player stdDev is static (derived from ceiling-floor range or default).

**Next Steps**:
- `dfs_risk_engine.py` has GARCH(1,1) via `arch` library with standard deviation fallback
- Call as preprocessing step: `python3 -c "from dfs_risk_engine import ...; print(json.dumps(garch_results))"`
- Replace static stdDev with GARCH-estimated volatility in all quant calculations
- Hot-streak players (low recent vol) → better for cash; spiking vol → better for GPPs

### 5. Copula Dependencies — Beyond Stacking Rules

**Current**: Correlation is implicit via team stacking constraints.

**Next Steps**:
- `advanced_quant_optimizer.py` has copula fitting code (needs `copulas` library)
- Use copula-generated correlated samples in MC simulation instead of independent normal samples
- This captures: teammate correlations (QB+WR), game-script effects (opposing players), weather impacts
- `copulaFamily` setting selects model; `dependencyThreshold` filters weak dependencies

### 6. Regime Detection — Context-Aware Optimization

**Next Steps**:
- Classify slates using KMeans on: avg ownership concentration, projection spread, total implied points, injury rate
- Auto-adjust: high-variance slate → more diversification; chalk slate → more contrarian; correlation slate → heavier stacking

---

## DFS Optimization Mathematics & Best Practices

### Core Problem: Integer Linear Programming (ILP)

DFS optimization is a **constrained combinatorial optimization** problem — a variant of the **0-1 Multi-Dimensional Knapsack Problem** (Dantzig, 1957; Kellerer et al., 2004).

**Mathematical Formulation:**
```
Maximize Z = SUM(i=1..n) [ p_i * x_i ]          (total projected points)

Subject to:
  SUM(i=1..n) [ s_i * x_i ] <= S                  (salary cap)
  SUM(i=1..n) [ x_i ] = R                          (roster size)
  SUM(i in P_j) [ x_i ] >= r_j  for each pos j    (position requirements)
  x_i in {0, 1}                                     (binary selection)
```

Where `p_i` = RAW projection (NEVER modified), `s_i` = salary, `x_i` = selection variable.

**CARDINAL RULE: The optimizer SELECTS players (`x_i`), it NEVER MODIFIES projections (`p_i`).**

This is proven by **LP/ILP sensitivity analysis** (Bertsimas & Tsitsiklis, 1997, Ch. 5): each objective coefficient `c_j` has a **range of optimality** within which the current basis remains optimal. Artificially modifying `p_i` to `p_i'` changes the optimization problem itself — the solver produces the optimal solution for a DIFFERENT problem, which is by definition suboptimal for the original (Cook et al., 1998).

### Academic Research Foundation

**Foundational Optimization Theory:**

| Reference | Key Contribution |
|-----------|-----------------|
| Dantzig (1957) "Discrete-Variable Extremum Problems" | LP relaxation of knapsack; greedy by profit-to-weight ratio |
| Kellerer, Pferschy, Pisinger (2004) "Knapsack Problems" (Springer) | Comprehensive reference: 0-1, bounded, multi-dimensional knapsack |
| Bertsimas & Tsitsiklis (1997) "Introduction to Linear Optimization" (Athena Scientific) | LP/ILP theory, sensitivity analysis, duality — proves projection immutability |
| Cook, Cunningham, Pulleyblank, Schrijver (1998) "Combinatorial Optimization" (Wiley) | Perturbation of objective coefficients → bounded-away-from-optimal solutions |

**DFS-Specific Academic Papers:**

| Paper | Key Finding |
|-------|-------------|
| Hunter, Vielma, Zaman (2016) "Picking Winners in DFS Using Integer Programming" (MIT/BU, arXiv:1604.01455) | DFS portfolio as submodular optimization with jointly Gaussian projections; won DK contests; **projections are immutable ILP inputs** |
| Haugh & Singal (2021) "How to Play Fantasy Sports Strategically" (Management Science 67(1), Columbia) | Portfolio optimization with opponent modeling via Dirichlet-multinomial; **350% ROI over 17-week NFL season**; portfolio > single-lineup optimization |
| Newell & Easton (2017) "Optimizing DFS via Stochastic IP" (Kansas State) | First stochastic IP for DFS expected payout; player projection DISTRIBUTIONS > point estimates |
| Mlcoch & Hubacek (2024) "Competing in DFS Using Generative Models" (ITOR 31(3)) | Mixed-integer quadratic program optimizing expected value AND variance; **34% ROI** |
| Bonomo et al. (2023) "Optimal Lineup Creation Using ML and LP" (arXiv:2309.15253) | Neural network projection + ILP optimization — proven two-stage pipeline |

**Portfolio Theory:**

| Reference | Application to DFS |
|-----------|-------------------|
| Markowitz (1952) "Portfolio Selection" (Journal of Finance, Nobel Prize) | Multiple lineups = portfolio; efficient frontier identifies optimal lineup sets maximizing E[return] for given risk |
| Michaud (1998) "Efficient Asset Management" (Oxford) | Resampled Efficient Frontier via Monte Carlo; classical MV is an "error maximizer" — MC resampling produces more robust, diversified lineup sets |

### Proper Optimization Pipeline (Academically Backed)

```
Step 1: IMPORT raw projections from CSV (immutable inputs — Bertsimas & Tsitsiklis)
Step 2: DEFINE constraints (salary, positions, stacks, exposure)
Step 3: SOLVE ILP → maximize SUM(projection * x_i) subject to constraints (Hunter et al.)
Step 4: DIVERSIFY → constraint-based diversity (NOT projection perturbation) or
         portfolio optimization selecting optimal subset (Haugh & Singal)
Step 5: EVALUATE → Monte Carlo simulation with ORIGINAL projections for
         VaR/CVaR/Sharpe scoring (Markowitz framework)
```

**Diversity must use CONSTRAINTS, not projection corruption:**
- Add exclusion constraints: `x_i + x_j <= 1` to prevent duplicate lineups
- Add overlap limits: `SUM(shared_players) <= K` between lineup pairs
- Or use portfolio optimization: generate candidate pool → select subset maximizing portfolio Sharpe (Michaud)

### Python Implementation (PuLP)
```python
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
prob = LpProblem("DFS", LpMaximize)
x = {p['id']: LpVariable(f"x_{p['id']}", cat=LpBinary) for p in players}
prob += lpSum(p['projection'] * x[p['id']] for p in players)  # RAW projections
prob += lpSum(p['salary'] * x[p['id']] for p in players) <= salary_cap
prob += lpSum(x[p['id']] for p in players) == roster_size
```

### Monte Carlo Simulation
- Purpose: Evaluate lineup outcome distributions AFTER optimization (Markowitz + Michaud)
- Each player: `Actual ~ Normal(projection, stdDev)` where projection is the RAW input
- Correlation: Same-team players have correlated outcomes (Cholesky decomposition)
- Simulation does NOT replace optimization — it supplements it for ranking/scoring
- **Resampled Efficient Frontier** (Michaud): MC resample projections → solve ILP each time → intersection of solutions = robust lineup set

### Common Pitfalls (AVOID THESE)
1. **Inflating projections** with "quant adjustments" — player projected for 8.0 becomes 10.6 after stack/vegas/streak bonuses → optimizer maximizes fictional numbers. Bertsimas sensitivity analysis proves this produces suboptimal lineups for the true problem.
2. **Stacking as projection modifier** — "add 2 pts for stacked players" is WRONG; stacking is a CONSTRAINT (require 4+ from same team), not a projection boost
3. **Double-counting** — adjusting projections before optimization AND scoring with adjusted values after
4. **Using actual game results as projections** — `FantasyPointsDraftKings` from SportsData.io may contain ACTUAL historical scores (Zion=67.7), not forward-looking projections
5. **Single-lineup optimization** — Haugh & Singal (2021) proved portfolio optimization (across multiple lineups) is the correct framework, especially for top-heavy GPP payout structures

### Data Pipeline Rules & Validation
- `Predicted_DK_Points` must be FORWARD-LOOKING projections, never actual game results
- Always compute from individual stat projections using DK scoring formula
- **Value-based detection**: Compute avg pts per $1K salary across all players. True projections have avg value ~3.5-5.0; actual results have avg value >5.5+ (DFS salaries approximate expected fantasy points — if data systematically exceeds this, it's actuals)
- **Salary-tier normalization**: When actuals detected, normalize to realistic projections using salary-tier target ratios (NBA: $9K+=4.8x, $7-9K=4.5x, $5-7K=4.2x, <$5K=3.8x) while preserving relative player ordering via rank-based noise
- **Correlation with salary**: True projections correlate r>0.7 with salary; actuals correlate r~0.3-0.5 due to game variance
- The `FantasyPointsDraftKings` API field may contain actuals for past dates — DO NOT copy directly

### Libraries
- **PuLP**: ILP solver for lineup optimization (recommended — used by Bonomo et al.)
- **OR-Tools** (Google): Alternative ILP solver, supports constraint programming
- **NumPy**: Monte Carlo simulation
- **SciPy**: Statistical distributions (log-normal for DFS outcomes, Cholesky for correlations)
- **cvxpy**: Convex optimization for Markowitz mean-variance portfolio construction

---

## Sport-Specific Stacking Strategy Guide

### Why Stacking Matters
Stacking exploits player-to-player correlations — when one player does well, correlated teammates benefit too. Without stacking, optimizers generate lineups with mathematically optimal individual projections but miss the covariance structure that drives GPP-winning ceilings.

### NFL Stacking (Highest Correlation Sport)

| Stack Type | Correlation | GPP Win Rate Impact | Description |
|-----------|------------|-------------------|-------------|
| **QB + WR1** | +0.54 | Baseline | Core stack — QB passing TD = WR receiving TD |
| **QB + WR1 + WR2** | +0.54/+0.41 | +15% ceiling | Double stack — captures multi-TD game scripts |
| **QB + TE** | +0.38 | Moderate | Red-zone correlation — TDs cluster in short yardage |
| **QB + WR + Opp WR (Bring-Back)** | +0.37 (cross-game) | +22% ceiling | Game-script hedge — shootout benefits both sides |
| **QB + WR + RB** | +0.54/+0.12 | Lower | RB correlation is weak — only in blowout scripts |
| **Game Stack (QB+WR+OppWR)** | +0.54/+0.37 | Best GPP | Full game environment capture |
| **Run-Back Stack** | +0.37 | GPP only | Opposing team's top pass catcher paired with your QB stack |
| **Naked QB** | N/A | -18% ceiling | Avoid — misses correlation upside entirely |

**NFL Key Insights:**
- **64% of GPP winners** use QB + at least 1 WR from same team
- Optimal NFL lineup structure: **QB + 2 WR (same team) + Bring-Back WR + value RB + DST opposing weak offense**
- DST negatively correlates with opposing QB (-0.31) — never stack DST with opposing QB
- Bring-back correlation (+0.37) is almost as strong as primary WR (+0.54) — always include in GPPs

### NBA Stacking (Game Environment Driven)

| Stack Type | Correlation | When to Use | Description |
|-----------|------------|------------|-------------|
| **Game Stack (2-3 from high total)** | +0.25 to +0.40 | High O/U games (230+) | Players from games with highest implied totals |
| **Pace Stack (2+ from fast teams)** | +0.20 to +0.35 | Pace > 100 | More possessions = more fantasy opportunities |
| **Blowout Fade** | -0.15 | Spread > 10 | Avoid starters from heavy favorites (bench risk) |
| **Mini-Correlation (PG + SG/SF)** | +0.10 to +0.20 | Always | Weak but real — assist-to-score correlation |
| **Negative Correlation (Teammates)** | -0.05 to -0.15 | Volume caps | Same-team players compete for shots/rebounds |
| **Stars + Value** | N/A | Salary construction | 2-3 studs ($8K+) + value plays ($3.5-5K) |
| **Showdown Captain** | 1.5x weight | Single-game | Captain slot gets 1.5x points — always your highest-ceiling play |

**NBA Key Insights:**
- **Game environment > individual stacking** — NBA correlation is weaker than NFL because players rotate independently
- Teammate negative correlation (-0.05 to -0.15) means **avoid heavy same-team stacking** in NBA (unlike NFL)
- Optimal NBA structure: **2-3 players from highest O/U game + 1-2 from second-highest + value filler**
- Pace and game total are the strongest predictors of ceiling — always check Vegas lines
- Late swap is critical in NBA — injuries/rest decisions come 30min before lock

### MLB Stacking (Batting Order Driven)

| Stack Type | Correlation | GPP Win Rate Impact | Description |
|-----------|------------|-------------------|-------------|
| **4-Man Batter Stack** | +0.30 to +0.45 | Baseline GPP | 4 batters from same team, consecutive in order |
| **5-Man Batter Stack** | +0.35 to +0.50 | +25% ceiling | 5 consecutive batters — captures big innings |
| **5+3 Structure** | N/A | Best GPP | 5-man primary stack + 3-man secondary stack |
| **Pitcher + Opposing Stack** | -0.40 to -0.55 | Avoid | Pitcher anti-correlates with opposing batters |
| **Consecutive Order Stack** | +0.40 to +0.50 | Critical | Batters adjacent in lineup share innings — RBI chains |
| **Wrap-Around Stack (8-9-1-2-3)** | +0.35 | Moderate | Top of order bats around with bottom — more ABs |
| **Secondary Stack (3-man)** | +0.25 to +0.35 | Complementary | Diversifies away from single-game dependency |

**MLB Key Insights:**
- **64.5% of GPP-winning MLB lineups** have a 5+ man stack from one team
- Consecutive batting order is crucial — batters 3-4-5 share the most plate appearances in high-scoring innings
- **5+3 structure** (5 from Team A + 3 from Team B + pitcher) is the dominant winning format
- **Never pair your pitcher with batters from the same team he's facing** — strong negative correlation (-0.40 to -0.55)
- Target teams facing bad pitchers (ERA 4.5+, high WHIP) in hitter-friendly parks
- Stacking against left-handed pitchers with right-handed heavy lineups adds +0.08 correlation

### Stack Types Configuration (Per Sport)

The "Stack Types" tab controls what structural constraints are applied during lineup generation. Each stack type can be enabled/disabled with min/max exposure targets:

**NFL Stack Types:**
1. `QB + WR` — Primary correlation stack (r=+0.54)
2. `QB + 2WR` — Double stack for ceiling (r=+0.54/+0.41)
3. `QB + WR + TE` — Red zone stack (r=+0.54/+0.38)
4. `Game Stack` — QB+WR (Team A) + WR/TE (Team B) — full game environment
5. `Bring-Back` — Opposing pass catcher paired with QB stack (r=+0.37)
6. `Run-Back` — Opposing RB/WR with QB stack
7. `Mini Stack (WR+WR)` — Same-team WR pair without QB
8. `No Stack` — Uncorrelated lineup

**NBA Stack Types:**
1. `Game Environment (2+)` — 2+ players from same high-total game
2. `Game Environment (3+)` — 3+ from same game (aggressive)
3. `Pace Stack` — Players from top-pace teams
4. `Stars + Value` — Salary structure: 2-3 studs + values
5. `Mini Stack (PG+Wing)` — Backcourt assist correlation
6. `Blowout Fade` — Avoid heavy favorites' starters
7. `Balanced` — No structural constraint, projection-driven
8. `No Stack` — Pure individual optimization

**MLB Stack Types:**
1. `4-Man Batter Stack` — 4 consecutive batters from same team
2. `5-Man Batter Stack` — 5 consecutive batters (r=+0.35-0.50)
3. `5+3 Structure` — 5-man primary + 3-man secondary stack
4. `Wrap-Around Stack` — Bottom + top of order (8-9-1-2-3)
5. `Pitcher vs. Weak Lineup` — Pitcher facing bottom-5 offense
6. `Secondary Stack (3-man)` — Complementary 3-man from different team
7. `Batter Stack (Generic)` — Any same-team batter grouping
8. `No Stacks` — Uncorrelated

---

## UX/UI Architecture (Phase 3 — SaberSim-Parity)

### Layout (Completed)
- **BuildControlBar** (40px): Build tabs + sport pills | Inline settings + status counters | Gear dropdown + CSV + BUILD LINEUPS CTA
- **GameSlate**: Horizontal scrolling game cards with implied totals (click to filter)
- **Main Content**: Tabbed — Players | Team Stacks | Exposure | Advanced Quant | My Entries
- **Sidebar** (288px): Compact lineup cards with [POS] Name (TM) $salary projection

### Phase 3 Completed Items
- Chrome reduced from 164px → 72px overhead
- PlayerTable: dense 28px rows, Opp column, color-coded ownership/value
- Sidebar: w-96 → w-72, compact lineup cards with team/salary
- BuildControlBar: merged BuildBar + ControlStrip, prominent cyan BUILD LINEUPS CTA
- GameSlate: game context cards with implied totals
- Multi-build system with per-build state isolation

### Phase 4 Completed Items (Multi-Stack Exposure)
- TeamStacksTab: multi-stack tabs (All, 2, 3, 4, 5), per-team min/max exposure inputs
- `teamExposures` flows: TeamStacksTab → BuildState → useOptimizer → API → server → Python
- `requestedStackSizes` computed from teamSelections keys, sent as explicit contract
- Python adapter: exposure-aware lineup distribution, feasibility validation, multi-stack simultaneous

---

## Known Issues & Technical Debt

1. **Data Pipeline: Actuals vs Projections** (FIXED) — `daily_nba_data_fetch.py` was copying `FantasyPointsDraftKings` (actual game scores) into `Predicted_DK_Points`. Fix: value-based detection (avg pts/$1K > 5.5 = actuals) + salary-tier normalization.
2. **Optimizer: PuLP ILP Primary** (FIXED) — NBA optimization now ALWAYS uses Python PuLP ILP solver first (`makrov_cli_adapter.py`). JS NBAOptimizer is fallback only. Lineup diversity via exclusion constraints (Hunter et al., 2016), not projection noise.
3. **Projection Noise Removed** (FIXED) — `makrov_cli_adapter.py` was multiplying projections by `lognormal(0, 0.10-0.15)` noise. Removed per Bertsimas & Tsitsiklis sensitivity analysis — projections are immutable ILP inputs.
4. **Quant Engine: Post-Optimization Only** (FIXED) — JS QuantEngine (MC, Sharpe, VaR, portfolio) now runs as POST-OPTIMIZATION evaluation on PuLP results, never modifies the optimization objective.

---

## Coding Patterns

- **Python optimizers**: Invoked from Node.js via `child_process.spawn('python3', [script, JSON.stringify(data)])`
- **Quant settings flow**: `AdvancedQuantSettings` interface → POST body → optimizer constructor → calculation methods
- **Multi-build state**: Each build has isolated `BuildState` (sport, players, selections, exposures, results, teamExposures)
- **Component lazy loading**: `React.lazy()` for Dashboard, GamesHub, PropBettingCenter, HowToUse
- **CSS variables**: `--dfs-bg-primary`, `--dfs-bg-secondary`, `--dfs-accent` (cyan), `--dfs-border`, `--dfs-text-*`
- **All quant defaults**: `DEFAULT_ADVANCED_QUANT_SETTINGS` in `types.ts` (disabled by default, strategy='combined', 10K sims, 0.95 VaR, 0.25 Kelly fraction)

### When Making Changes

- **Adding quant feature**: types.ts interface → AdvancedQuantTab.tsx UI → useOptimizer.ts passthrough → quant-engine.js math → nba-optimizer.js integration
- **Connecting Python quant code**: `child_process.spawn()` in Node → JSON stdin/stdout → parse results
- **Multi-stack changes**: TeamStacksTab.tsx → DFSOptimizer.tsx → useOptimizer.ts → dfs-api.ts → server/index.js → makrov_cli_adapter.py
- **Performance**: MC 10K+ sims in JS is ~50ms; GARCH fitting needs Python; cache where possible

---

## Priority Roadmap

### Phase 1: JS Quant Engine (NBA) — ✅ COMPLETE
1. ✅ `quant-engine.js` — Monte Carlo (2K-10K sims), Kelly, VaR/CVaR, Sharpe, portfolio analysis
2. ✅ `nba-optimizer.js` — Quant-integrated player selection, per-lineup MC, portfolio metrics
3. ✅ `index.js` routing — Quant enabled → JS optimizer; disabled → Python Markov (fallback JS)
4. ✅ 5 optimization strategies wired to distinct mathematical objectives

### Phase 2: Multi-Stack + Exposure Pipeline — ✅ COMPLETE
5. ✅ TeamStacksTab multi-stack (2,3,4,5) with per-team min/max exposure
6. ✅ requestedStackSizes explicit contract through full pipeline
7. ✅ Python adapter: exposure-aware distribution, feasibility validation
8. ✅ Frontend ↔ backend teamExposures wiring

### Phase 3: SaberSim-Parity UX — ✅ COMPLETE
9. ✅ Compact chrome (BuildControlBar, GameSlate, dense PlayerTable)
10. ✅ Compact sidebar (288px, inline lineup cards)
11. ✅ Multi-build system with state isolation

### Phase 5: Extend Quant to NFL/MLB + Surface Metrics in UI — ✅ COMPLETE
12. ✅ Wire `quant-engine.js` into `nfl-optimizer.js` and `optimizer.js` (same pattern as NBA)
13. ✅ Display quant metrics in lineup cards — VaR badge (blue), Sharpe color bar (green/yellow/red), ceiling probability % (purple)
14. ✅ Add Boom% and Leverage columns to PlayerTable with color-coded values
15. ✅ Add "Leverage Plays" filter pill in PlayerTable (purple — low ownership + high ceiling)
16. ✅ Portfolio-level dashboard after optimization — portfolio Sharpe, uniqueness, max exposure, concentration

### Phase 5.5: Stack Types Refactor + Data Organization
17. Rename "Exposure" tab → "Stack Types" with sport-specific correlation-backed stack types
18. Update `sport-config.ts` with industry-standard stack names per sport (NFL: QB+WR, Game Stack, Bring-Back; NBA: Game Environment, Pace Stack; MLB: 4-Man, 5-Man, 5+3 Structure)
19. Add correlation badges and sport-specific descriptions to each stack type row
20. Improve player lock/exclude visibility in PlayerTable
21. Add team exposure summary view showing per-team actual vs target exposure

### Phase 6: Deep Quant Differentiation (Competition Moat)
17. Connect GARCH volatility from Python to replace static stdDev
18. Implement Cholesky-decomposed correlated MC simulation (replaces independent sampling)
19. Copula-based dependency modeling for realistic tail scenarios
20. Regime detection → auto-tune riskTolerance and stacking aggressiveness per slate
21. Active portfolio construction — select optimal lineup subset from candidate pool using portfolio Sharpe maximization

### Phase 7: Product Polish (Monetization-Ready)
22. Contest simulation — simulate lineups against 10K realistic opponent fields
23. Backtesting framework — show quant strategies vs naive optimization on historical data
24. Bankroll management dashboard — Kelly-optimal contest selection + entry sizing
25. Export risk reports alongside lineup CSVs
26. "Cash Mode" / "GPP Mode" one-click presets that auto-configure all quant parameters
