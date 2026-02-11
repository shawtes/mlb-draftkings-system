# CLAUDE.md — Quantitative DFS Optimization System Guide

## Project Overview

This is an NFL/NBA/MLB DraftKings & FanDuel optimization platform that combines mathematical optimization, machine learning, and quantitative finance methods to generate optimal daily fantasy sports lineups. The system's **primary differentiator and selling point** is its institutional-grade quantitative engine — the same mathematical frameworks used by hedge funds and quant trading desks, applied to DFS contest optimization.

---

## Repository Structure

```
├── 1_CORE_TRAINING/          # ML model training pipeline (StackingRegressor, 500+ features)
├── 2_PREDICTIONS/            # Player projection generation
├── 5_DRAFTKINGS_ENTRIES/     # DraftKings entry management
├── 6_OPTIMIZATION/           # Core optimization engines (PuLP, Genetic, Quant)
│   ├── pulp_lineup_optimizer.py          # BILP solver (PuLP) — production
│   ├── optimizer.genetic.algo.py         # Genetic algorithm diversity engine — production
│   ├── advanced_quant_optimizer.py       # Advanced quant methods — needs integration
│   ├── dfs_risk_engine.py                # Risk management (Kelly, GARCH, VaR) — partial
│   ├── probability_enhanced_optimizer.py # Probability feature mapping — partial
│   └── markov_optimizer_wrapper.py       # NBA Markov wrapper
├── 7_ANALYSIS/               # Post-optimization analysis
├── 8_DOCUMENTATION/          # Research docs, math guides, implementation notes
├── web_optimizer/            # Full-stack web application
│   ├── server/               # Node.js Express backend
│   │   ├── index.js          # API routes, WebSocket, Python subprocess spawning
│   │   ├── optimizer.js      # MLB optimizer (JS) with quant settings handling
│   │   ├── nfl-optimizer.js  # NFL optimizer
│   │   ├── nba-optimizer.js  # NBA optimizer — QUANT INTEGRATED
│   │   └── quant-engine.js   # Shared quantitative engine (Monte Carlo, Kelly, VaR, Sharpe)
│   └── client/               # React 18 + TypeScript frontend
│       └── src/components/optimizer/
│           ├── AdvancedQuantTab.tsx       # Quant settings UI
│           ├── TeamStacksTab.tsx          # Team stacking UI
│           ├── StackExposureTab.tsx       # Exposure management
│           ├── hooks/useOptimizer.ts      # Optimization hook
│           └── types.ts                  # TypeScript interfaces
├── python_algorithms/        # Utility algorithms
├── README.md                 # System documentation
└── RUN_ALL.sh               # Pipeline execution script
```

---

## Quantitative Features — Current State & Strategy

### The Quant Engine Is the Product

The system's value proposition is **not** "another lineup optimizer" — it's a **quantitative risk management platform for DFS**. Every feature should reinforce this positioning. The quant engine should drive three core workflows:

1. **Player Selection** — Use quantitative signals (volatility, correlations, Kelly sizing) to select which players to include, not just "highest projected points"
2. **Lineup Construction** — Build lineups using portfolio theory (mean-variance, risk parity) rather than greedy heuristics
3. **Multi-Entry Portfolio Optimization** — Treat a portfolio of lineups like a portfolio of assets, optimizing for maximum risk-adjusted return across all entries

### Implementation Status

| Feature | Frontend UI | Python Backend | JS Backend | Status |
|---------|-------------|---------------|------------|--------|
| **PuLP BILP Solver** | — | ✅ Full | — | **Production** |
| **Genetic Algorithm** | — | ✅ Full | — | **Production** |
| **Exposure Management** | ✅ Full | ✅ Full | ✅ Full | **Production** |
| **Team Stacking** | ✅ Full | ✅ Full | ✅ Full | **Production** |
| **Kelly Criterion** | ✅ UI Controls | ⚠️ Partial (`dfs_risk_engine.py`) | ✅ NBA (`quant-engine.js`) | **NBA Production** |
| **GARCH Volatility** | ✅ UI Controls | ⚠️ Optional (`arch` lib) | ❌ Not implemented | **Needs Integration** |
| **Monte Carlo Simulation** | ✅ UI Controls | ⚠️ Declared, not called | ✅ NBA (`quant-engine.js`) | **NBA Production** |
| **Mean-Variance (Markowitz)** | ✅ UI Dropdown | ❌ Research only | ✅ NBA (salary-aware) | **NBA Production** |
| **Risk Parity** | ✅ UI Dropdown | ❌ Not implemented | ✅ NBA (volatility-normalized) | **NBA Production** |
| **Copula Dependency** | ✅ UI Controls | ⚠️ Optional (`copulas` lib) | ❌ Not implemented | **Needs Implementation** |
| **VaR / CVaR** | ✅ UI Controls | ⚠️ Partial (`dfs_risk_engine.py`) | ✅ NBA (Monte Carlo-based) | **NBA Production** |
| **Ownership Leverage** | — | — | ✅ NBA (GPP scoring) | **NBA Production** |
| **Sharpe Ratio (per lineup)** | — | — | ✅ NBA (Monte Carlo-derived) | **NBA Production** |
| **Portfolio Analysis** | — | — | ✅ NBA (cross-lineup metrics) | **NBA Production** |
| **Regime Detection** | ❌ No UI | ⚠️ KMeans code, disconnected | ❌ Not implemented | **Needs Integration** |

### Architecture: Data Flow for Quant Settings

```
User (AdvancedQuantTab.tsx)
  → AdvancedQuantSettings interface (types.ts)
  → useOptimizer hook merges with defaults
  → POST /api/optimize (dfs-api.ts)
  → Express handler (index.js:782)
  → NBA: if quant enabled → JS NBAOptimizer + QuantEngine
         if quant disabled → Python Markov (fallback: JS NBAOptimizer)
  → NFL/MLB: Sport-specific optimizer (quant not yet wired)
  → Results returned with quantMetrics per lineup + portfolioMetrics in summary
```

**NBA Quant Flow (IMPLEMENTED)**:
1. `QuantEngine` initialized with user's `advancedQuantSettings`
2. `scorePlayersQuant()` pre-computes quant scores for all players (Sharpe, leverage, Kelly, ceiling probability)
3. `kellyExposureLimits()` computes per-player exposure bounds based on Kelly fraction
4. During lineup generation, `selectPlayerQuant()` uses quant scores for player selection
5. After lineup is built, `monteCarloLineup()` runs 2K-10K simulations → VaR, CVaR, Sharpe, percentiles
6. After all lineups generated, `analyzePortfolio()` computes cross-lineup metrics (uniqueness, concentration)
7. Response includes `quantMetrics` per lineup and `portfolioMetrics` in summary

**Key Gap**: NFL and MLB optimizers still receive quant settings but don't apply real quantitative math. The `quant-engine.js` module is ready to be integrated into those optimizers using the same pattern.

---

## How to Maximize Quant Features as the Primary Selling Point

### 1. Player Selection — Quantitative Player Scoring

**Current**: Players are selected by greedy heuristics (highest projection, best value ratio, or random from top tier).

**Target**: Every player should have a **quantitative score** that incorporates:

- **Expected Value**: Projection weighted by confidence interval width
- **Volatility-Adjusted Value**: `projection / stdDev` (Sharpe-like ratio per player)
- **Ownership Leverage**: `(ceiling × (1 - ownership%)) / salary` — high-ceiling, low-owned players score higher in GPPs
- **Correlation Contribution**: How much a player improves the lineup's overall Sharpe ratio when added (requires covariance matrix)
- **Kelly-Optimal Sizing**: Translate Kelly fraction into exposure % — players with higher edge get more exposure across lineups

**Implementation Path**:
- The `Player` interface in `types.ts` already has `ceiling`, `floor`, `stdDev`, `ownership`, and `leverageScore` fields
- The JS optimizer's `selectAdvancedPlayer()` method should use these fields with the quant strategy selected
- Python's `dfs_risk_engine.py` has Kelly and VaR code that should be called via subprocess for heavy calculations

### 2. Lineup Construction — Portfolio Theory Applied

**Current**: Lineups are built position-by-position using greedy selection or random sampling.

**Target**: Each lineup should be constructed as a **portfolio** optimizing a risk-adjusted objective:

- **Cash Games (Mean-Variance)**: Minimize variance for a given expected return — `maximize μᵀx - λ√(xᵀΣx)` — produces consistent, safe lineups
- **Tournaments (Max Sharpe)**: Maximize the Sharpe ratio — `maximize (μᵀx - rf) / √(xᵀΣx)` — produces high-ceiling lineups with controlled risk
- **GPP Contrarian (Leverage-Weighted)**: Maximize `Σ(ceiling_i × (1 - ownership_i) × x_i)` subject to constraints — produces unique, high-upside lineups

**Implementation Path**:
- The `strategy` dropdown in `AdvancedQuantTab.tsx` already maps to `combined`, `kelly`, `risk_parity`, `mean_variance`, `equal_weight`
- Route each strategy to a distinct optimization objective in the backend
- For mean-variance: build a covariance matrix from player correlations (teammate pairs, game environment, position), then use `scipy.optimize.minimize` or PuLP with quadratic extension
- The `riskTolerance` slider (0.1–2.0) maps directly to the λ parameter in mean-variance optimization

### 3. Multi-Entry Portfolio Optimization — The Key Differentiator

**Current**: Multiple lineups are generated independently with diversity enforced by exposure limits and uniqueness checks.

**Target**: Treat the entire set of lineups as a **portfolio of assets**:

- **Correlation Management**: Ensure lineups are not just different players but different *exposures* — low correlation between lineup outcomes
- **Risk Parity Across Entries**: Each lineup contributes equal risk to the overall portfolio — no single lineup dominates variance
- **Monte Carlo Portfolio VaR**: Simulate 10,000+ scenarios for each lineup, then compute portfolio-level VaR/CVaR across all entries
- **Kelly-Optimal Entry Sizing**: If entering multiple contests, Kelly criterion determines how many entries to allocate per contest type

**Implementation Path**:
- Monte Carlo: For each lineup, sample player points from their distributions (using `stdDev`, `ceiling`, `floor`), sum to get lineup score, repeat 10K times → distribution of outcomes per lineup
- Portfolio VaR: Compute correlation between lineup outcome distributions, then calculate portfolio VaR at the configured confidence level
- The `monteCarloSims` parameter (1K–50K) controls simulation count
- The `varConfidence` parameter (90–99%) sets the VaR threshold

### 4. GARCH Volatility — Dynamic Risk Estimation

**Current**: Player volatility is static (`stdDev` field) or not used at all.

**Target**: Use GARCH(p,q) to model **time-varying volatility** of player performance:

- Players on hot streaks have lower recent volatility → more predictable → better for cash games
- Players with spiking volatility are high-risk/high-reward → better for GPPs
- Volatility clustering (a GARCH property) means recent variance predicts near-future variance

**Implementation Path**:
- `dfs_risk_engine.py` already has GARCH(1,1) code using the `arch` library (with fallback to standard deviation)
- Connect the `garchP`, `garchQ`, and `lookbackPeriod` settings from the frontend to the Python engine
- Use GARCH-estimated volatility instead of static `stdDev` in all calculations above
- The web optimizer should call the Python GARCH engine as a preprocessing step before lineup generation

### 5. Copula-Based Dependency Modeling — Correlation Done Right

**Current**: Player correlations are implicit (team stacking rules) but not mathematically modeled.

**Target**: Use copulas to model **non-linear dependencies** between players:

- Teammates on the same team have positive correlation (QB-WR stacks)
- Opposing players in the same game have complex dependencies (game script)
- Weather and venue create correlated performance shifts across all players in a game
- Copulas capture tail dependencies that simple correlation coefficients miss

**Implementation Path**:
- `advanced_quant_optimizer.py` references copula modeling but needs the `copulas` library installed
- The `copulaFamily` setting (gaussian, t, clayton, frank, gumbel) should select the copula type
- The `dependencyThreshold` (0.1–0.9) filters which player pairs are modeled as dependent
- Use copula-generated correlated samples in the Monte Carlo simulation for more realistic scenario generation

### 6. Regime Detection — Context-Aware Optimization

**Current**: Optimization uses the same parameters regardless of game context.

**Target**: Automatically detect and adapt to **market regimes**:

- **High-Variance Slates**: Many questionable players, weather concerns → increase diversification
- **Chalk Slates**: Clear optimal plays dominate ownership → contrarian strategy more valuable
- **Correlation Slates**: Multiple high-total games → stacking strategies more important
- **Low-Total Slates**: Pitcher-dominated MLB slates → floor-focused approach

**Implementation Path**:
- The KMeans clustering code in `advanced_quant_optimizer.py` can classify slates into regimes
- Based on detected regime, auto-adjust `riskTolerance`, stacking aggressiveness, and ownership fading
- Display detected regime in the UI so users understand why the optimizer is making certain choices

---

## Development Guidelines

### Build & Run

```bash
# Web Application
cd web_optimizer/server && npm install && node index.js    # Backend (port 3001)
cd web_optimizer/client && npm install && npm run dev       # Frontend (port 5173)

# Python Optimization Pipeline
cd 6_OPTIMIZATION && python pulp_lineup_optimizer.py        # Core optimizer
cd 6_OPTIMIZATION && python optimizer.genetic.algo.py       # Genetic engine

# Full Pipeline
bash RUN_ALL.sh                                             # End-to-end execution
```

### Key Technical Details

- **Frontend**: React 18 + TypeScript 5.9 + Vite 6.3 + Material-UI + Radix UI
- **Backend**: Node.js + Express 4.17 + WebSocket (ws 8.18)
- **Optimization**: Python 3.8+ + PuLP 2.7+ + Pandas + NumPy + SciPy
- **ML Pipeline**: StackingRegressor (sklearn) + XGBoost + Ridge + Random Forest
- **Optional Quant Libs**: `arch` (GARCH), `copulas` (dependency modeling)

### Coding Patterns

- Python optimizers are invoked from Node.js via `child_process.spawn()`
- Quant settings flow: `AdvancedQuantSettings` interface → POST body → optimizer constructor → calculation methods
- Player data: CSV upload → parsed in Node.js → passed to optimizers as structured data
- Results: Optimizer returns lineup arrays → formatted as JSON → sent via WebSocket for real-time updates
- All quant parameters have sensible defaults in `DEFAULT_ADVANCED_QUANT_SETTINGS` (types.ts)

### When Making Changes

- **Adding a new quant feature**: Add the parameter to `AdvancedQuantSettings` interface in `types.ts`, add UI control in `AdvancedQuantTab.tsx`, pass through `useOptimizer.ts`, handle in the sport-specific optimizer, implement the math in Python if computationally intensive
- **Connecting Python quant code to web app**: Use `child_process.spawn()` in the Node.js optimizer to call Python scripts, pass settings as JSON command-line arguments or stdin, return results as JSON stdout
- **Testing quant features**: Use the existing CSV test data files in the repository root, verify that optimizer output changes when quant settings are toggled
- **Performance consideration**: Monte Carlo with 10K+ sims and GARCH fitting can be slow — use Python multiprocessing for heavy compute, cache results where possible

---

## Priority Roadmap for Quant Integration

### Phase 1: Connect Existing Code (Highest Impact, Lowest Effort) — ✅ COMPLETE (NBA)
1. ✅ Built `quant-engine.js` with real Kelly Criterion (edge/variance formula) for player exposure sizing
2. ✅ Implemented Monte Carlo simulation (2K-10K sims) for lineup outcome distributions
3. ✅ Wired `riskTolerance` and `strategy` to quantitative player scoring objectives
4. ✅ Added VaR/CVaR and Sharpe ratio calculations to every generated lineup
5. ✅ Implemented ownership leverage scoring for GPP tournaments
6. ✅ Built portfolio-level analysis (cross-lineup Sharpe, uniqueness, exposure concentration)
7. ✅ Updated `index.js` NBA path: quant mode → JS optimizer, non-quant → Python Markov (fallback JS)

### Phase 2: Extend to NFL/MLB + Core Engine Enhancements
8. Wire `quant-engine.js` into `nfl-optimizer.js` (same pattern as NBA)
9. Wire `quant-engine.js` into `optimizer.js` (MLB)
10. Connect GARCH volatility from `dfs_risk_engine.py` to replace static `stdDev` values
11. Build mean-variance optimization using player covariance matrix (full Markowitz)

### Phase 3: Advanced Differentiation (Competition Moat)
12. Implement copula-based correlation modeling for realistic scenario generation
13. Add regime detection to auto-tune optimization parameters per slate
14. Build risk parity optimization for multi-entry portfolio construction
15. Add real-time Sharpe ratio / Sortino ratio metrics to generated lineups

### Phase 4: Product Polish (User-Facing Value)
16. Display quant metrics in lineup cards (VaR, Sharpe, expected Sharpe, ceiling probability)
17. Add backtesting framework to show quant strategies vs naive optimization historically
18. Build slate analysis dashboard showing detected regime, correlation heatmap, ownership vs projection scatter
19. Export risk reports alongside lineup CSVs for bankroll management
