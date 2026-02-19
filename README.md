# UrSim — Quantitative DFS Optimization Platform

A multi-sport (MLB/NBA/NFL) DraftKings optimization platform that treats every lineup as a financial asset and every contest entry as a portfolio allocation decision. The system combines binary integer linear programming, ensemble machine learning, genetic algorithms, and institutional-grade quantitative finance methods — Monte Carlo simulation, Kelly Criterion, VaR/CVaR, Mean-Variance optimization — to generate risk-adjusted optimal lineups.

## Overview

UrSim goes beyond simple knapsack solvers by applying the same mathematical frameworks used by hedge funds and quant trading desks to daily fantasy sports:

- **Binary Integer Linear Programming (BILP)** — Mathematically optimal lineup construction via PuLP (CBC solver)
- **Ensemble Machine Learning** — StackingRegressor (Ridge + Lasso + LightGBM + CatBoost → XGBRegressor) with Optuna HPO and SHAP feature selection
- **Genetic Algorithms** — Population-based evolution for diverse multi-lineup pools
- **Quantitative Risk Engine** — Monte Carlo (2K–10K sims), Kelly Criterion, Mean-Variance (Markowitz), GARCH volatility, VaR/CVaR, Sharpe ratio per lineup, portfolio-level analysis
- **Reinforcement Learning** — RL-based hyperparameter optimization and parlay selection
- **Full-Stack Web Application** — React + Express with real-time WebSocket updates, Firebase auth, CSV upload, and interactive lineup management

The platform generates 1–500+ unique lineups optimized for different contest types (cash games vs. GPP tournaments) with configurable stacking, exposure controls, and team correlation constraints.

---

## Quick Start

```bash
# Install dependencies
cd web_optimizer && npm run install-all

# Start the web app (frontend + backend)
make dev
# Frontend: http://localhost:3000   Backend: http://localhost:5001
```

### All Available Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start web app (frontend + backend concurrently) |
| `make server` | Start backend only (Express on port 5001) |
| `make client` | Start frontend only (Vite on port 3000) |
| `make build` | Production build of the frontend |
| `make train` | Run MLB training pipeline (with HPO) |
| `make train-fast` | Run MLB training without hyperparameter optimization |
| `make backtest` | Run NBA backtesting framework |
| `make pipeline` | Run full fetch → train → bridge pipeline |
| `make clean` | Remove `__pycache__`, `.pyc` files, and build artifacts |

---

## Repository Structure

```
mlb-draftkings-system/
├── web_optimizer/                       # Full-stack web application
│   ├── client/                          #   React 18 + TypeScript + Vite 6 + Radix UI + Tailwind
│   │   └── src/
│   │       ├── components/optimizer/    #     Core optimizer UI (PlayerTable, TeamStacks, AdvancedQuant, etc.)
│   │       ├── components/dfs/          #     DFS-specific components (ControlPanel, Favorites, StatusBar)
│   │       ├── components/ui/           #     50+ Radix UI component wrappers
│   │       ├── services/               #     API clients, WebSocket, sport config
│   │       ├── firebase/               #     Firebase auth configuration
│   │       └── contexts/               #     React context providers (Auth)
│   ├── server/                          #   Express + WebSocket + Python subprocess bridge
│   │   ├── index.js                     #     Main API server (CSV upload, optimize, favorites, DK entries)
│   │   ├── optimizer.js                 #     MLB optimization logic
│   │   ├── nba-optimizer.js             #     NBA optimization logic (quant integrated)
│   │   ├── nfl-optimizer.js             #     NFL optimization logic
│   │   ├── quant-engine.js              #     Quant engine (Monte Carlo, Kelly, VaR, Sharpe, portfolio)
│   │   ├── makrov_cli_adapter.py        #     Python CLI adapter for Markov chain optimizer
│   │   └── nba_optimizer_cli.py         #     Python CLI adapter for NBA optimizer
│   ├── launchers/                       #     Platform-specific launcher scripts (Windows, Python)
│   └── docs/                            #     Web app documentation (design specs, guides, fixes)
│
├── optimization/                        # Core optimization engines
│   ├── pulp_lineup_optimizer.py         #   PuLP BILP solver (salary cap, position, stacking constraints)
│   ├── makrovchain_optimizer.py         #   Full Markov chain optimizer with PyQt5 GUI (8900+ lines)
│   ├── genetic_algo_mlb_optimizer.py    #   Genetic algorithm — MLB multi-lineup diversity
│   ├── genetic_algo_nfl_optimizer.py    #   Genetic algorithm — NFL variant
│   ├── dfs_risk_engine.py              #   Kelly, GARCH, VaR/CVaR, Sharpe, correlation matrix
│   ├── portfolio_optimization.py        #   Mean-Variance, Kelly weighting, risk parity, equal weight
│   ├── probability_modeling.py          #   Bayesian priors (Beta-Binomial, Gamma-Poisson), HMM, Monte Carlo
│   ├── nfl_stack_engine.py              #   NFL stacking engine
│   ├── nba_stack_engine.py              #   NBA stacking engine
│   ├── daily_nba_data_fetch.py          #   Daily NBA data fetcher
│   ├── nba_sportsdata_fetcher.py        #   SportsData.io API client
│   ├── rl_hyperopt/                     #   RL hyperparameter optimization (agent, environment, reward shaping)
│   └── rl_parlay_system/               #   RL parlay system (agent, trainer, GUI, data collector)
│
├── training/                            # ML training pipelines
│   ├── mlb/                             #   MLB: feature engineering (copula PCA, sabermetrics, Marcel shrinkage),
│   │                                    #         ensemble model builder (Optuna HPO, SHAP selection),
│   │                                    #         walk-forward temporal CV, pregame prediction, matchup engine
│   └── nba/                             #   NBA: feature engineering, game log scraping, training
│
├── pipeline/                            # Orchestration & evaluation
│   ├── run_pipeline.py                  #   Master pipeline: fetch → train → bridge → optimize
│   ├── backtest_optimizer.py            #   NBA backtesting (15+ game days, 4 config combos, hindsight comparison)
│   └── dk_bridge.py                     #   Merges training predictions with DraftKings CSV
│
├── scripts/                             # Utility & scraping scripts
│   ├── fangraphs_batters.py             #   Selenium scraper — FanGraphs batter data (2005–2025)
│   ├── fangraphs_pitchers.py            #   Selenium scraper — FanGraphs pitcher data
│   ├── calculate_lineup_scores.py       #   Score generated lineups against actual results
│   └── underdog/                        #   Underdog parlay tools (API client, prediction, NFL builders, GUI)
│
├── docs/                                # Project documentation & research
│   ├── USE_CASES.md                     #   Application use cases
│   ├── matchup_engine_*.md              #   Matchup engine plan & status
│   ├── rl_optimization_research.md      #   RL research & theory
│   └── archive/                         #   Historical docs (training logs, parameter results, guides)
│
├── Makefile                             # Common commands (dev, train, backtest, pipeline, clean)
├── pyproject.toml                       # Python dependencies & tool config
├── _archive/                            # Deprecated experimental code
└── data/                                # Data files (gitignored)
```

---

## Web Application

The web app provides an interactive interface for uploading player projections, configuring optimization parameters, and generating lineups in real time.

### Key Features

- **CSV Upload** — Upload DraftKings player pools; auto-detects sport and column format
- **Multi-Sport Support** — MLB, NBA, and NFL with sport-specific position constraints and salary caps
- **Player Table** — Sortable, filterable player list with per-player exposure controls (min/max lock)
- **Team Stacking** — Multi-stack configuration (2, 3, 4, 5-player stacks) with per-team exposure limits
- **Advanced Quant Settings** — Configure Monte Carlo sims, Kelly fraction, VaR confidence, optimization strategy
- **Contest Modes** — Cash (floor/consistency/Sharpe) and GPP (ceiling/leverage/ownership fade)
- **Build Manager** — Generate and manage multiple builds with different settings
- **Favorites** — Save and reload favorite lineups (persisted server-side)
- **DK Entries Import** — Parse existing DraftKings entries for review
- **Real-time Updates** — WebSocket-powered progress and status notifications
- **Authentication** — Firebase-based login and registration
- **Dashboard** — Overview, how-to-use guide, and game analysis views

### Architecture

```
React Client (port 3000)
  → Uploads CSV, sets optimization parameters
    → POST /api/optimize → Express Server (port 5001)
      → Spawns Python subprocess (PuLP ILP solver)
        → Returns JSON lineups with quant metrics
      → JS Quant Engine layers on MC simulation, portfolio analysis
    → WebSocket pushes real-time progress
  → Renders lineup cards, exposure charts, portfolio metrics
```

---

## ML Training Pipeline

The training pipeline builds ensemble models to predict player DraftKings fantasy points from historical data.

### MLB Pipeline

1. **Data Collection** — FanGraphs batter/pitcher stats (2005–2025) via Selenium scraper
2. **Feature Engineering** — Copula modeling with PCA reduction (84→8 dims), sabermetrics (wOBA, HR/FB), Marcel shrinkage, DraftKings scoring features
3. **Model Building** — StackingRegressor: Ridge + Lasso + LightGBM + CatBoost → XGBRegressor meta-learner, with Optuna HPO and SHAP-based feature selection
4. **Validation** — Walk-forward temporal cross-validation (time-split, not random), per-player evaluation, Conformalized Quantile Regression (CQR)
5. **Bridge** — Merges model predictions with DraftKings CSV for optimizer consumption

### NBA Pipeline

Follows the same architecture with NBA-specific feature engineering (game logs, opponent metrics, advanced stats) and an integrated scraper for current-season data.

---

## Mathematical Foundations

### Core Optimization: Binary Integer Linear Programming

```
Maximize: Σ (projection[i] × x[i])

Subject to:
  Σ (salary[i] × x[i]) ≤ SALARY_CAP
  Σ x[i] = LINEUP_SIZE
  Position constraints per sport
  Stacking constraints (team correlation groups)
  Exclusion constraints (lineup diversity via iterative solving)
  Locked-player constraints (min-exposure enforcement)
  x[i] ∈ {0, 1}
```

Solved via PuLP (CBC backend). Projections are **immutable** ILP objective coefficients — never modified by quant adjustments.

### Quantitative Engine

| Method | Purpose |
|--------|---------|
| **Monte Carlo Simulation** (2K–10K sims) | Per-lineup VaR, CVaR, Sharpe ratio, ceiling/floor probability |
| **Kelly Criterion** | Optimal player exposure limits derived from edge/variance ratio |
| **Mean-Variance (Markowitz)** | 5 strategies: Combined, Kelly, Mean-Variance, Risk Parity, Equal Weight |
| **GARCH Volatility** | Dynamic risk estimates from historical performance variance |
| **VaR / CVaR** | Downside risk quantification at configurable confidence levels |
| **Portfolio Analysis** | Cross-lineup Sharpe ratio, exposure Herfindahl index, average pairwise uniqueness |

### Genetic Algorithm

Generates diverse multi-lineup pools via population-based evolution with tournament selection, crossover, mutation, and minimum Hamming distance enforcement between lineups.

### Probabilistic Modeling

Bayesian player modeling using conjugate priors (Beta-Binomial for rates, Gamma-Poisson for counts), Hidden Markov Models for regime detection, and Monte Carlo expected value optimization.

---

## RL Systems

### Hyperparameter Optimization (`optimization/rl_hyperopt/`)

Reinforcement learning agent that learns to tune optimizer parameters (variance targets, salary allocation, stacking aggressiveness) by framing parameter selection as a Markov Decision Process with reward shaping based on lineup quality and uniqueness.

### Parlay System (`optimization/rl_parlay_system/`)

End-to-end RL pipeline for parlay/prop bet selection: data collection, environment simulation, agent training, and a PyQt5 GUI for interactive parlay building. Includes Underdog-specific integrations via `scripts/underdog/`.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript 5, Vite 6, Radix UI (30+ primitives), Tailwind CSS, Framer Motion, Recharts, Lucide icons |
| **Backend** | Node.js, Express, WebSocket (`ws`), Multer (file upload), CSV parser |
| **Auth & Storage** | Firebase (authentication, database) |
| **Optimization** | Python 3.8+, PuLP (CBC solver), NumPy, SciPy |
| **ML Pipeline** | scikit-learn (StackingRegressor), XGBoost, LightGBM, CatBoost, Optuna, SHAP |
| **Quant Engine** | GARCH (`arch`), copulas, Monte Carlo, Kelly, VaR/CVaR |
| **Data** | Pandas, SportsData.io API, FanGraphs (Selenium scraper) |
| **Desktop GUI** | PyQt5 (Markov optimizer, parlay system) |

### Python Dependencies

Install core dependencies:

```bash
pip install -e .                  # Core: pandas, numpy, scipy, pulp, scikit-learn, xgboost, lightgbm
pip install -e ".[train]"         # + Optuna, SHAP, CatBoost
pip install -e ".[quant]"         # + arch (GARCH), copulas
pip install -e ".[scrape]"        # + Selenium, BeautifulSoup, requests
pip install -e ".[dev]"           # + pytest, ruff
```

---

## License

MIT License — Copyright (c) 2025 Sineshaw Tesfaye. See [LICENSE](LICENSE) for details.
