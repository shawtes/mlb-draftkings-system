# UrSim — Quantitative DFS Optimization Platform

Daily fantasy sports optimization platform combining mathematical optimization, machine learning, and quantitative finance techniques to generate optimal NFL/NBA/MLB DraftKings lineups.

## Overview

This system solves the complex constrained optimization problem of selecting optimal fantasy sports lineups by integrating:

- **Binary Integer Linear Programming (BILP)** for mathematically optimal solutions
- **Genetic Algorithms** for diverse multi-lineup generation
- **Ensemble Machine Learning** for player performance prediction
- **Quantitative Finance Methods** (Monte Carlo, Kelly Criterion, VaR/CVaR, Mean-Variance) for risk-adjusted optimization
- **Real-time Web Application** for interactive lineup generation

The platform processes player projections, applies sophisticated constraints (salary caps, position requirements, stacking strategies), and generates 1-500+ unique lineups optimized for different contest types (cash games vs. tournaments).

---

## Quick Start

```bash
# Web Application
cd web_optimizer && npm run dev          # Frontend: http://localhost:3000  Backend: http://localhost:5001

# Or use Makefile
make dev                                 # Same as above
make train                               # Run MLB training pipeline
make backtest                            # Run backtesting framework
```

---

## Repository Structure

```
mlb-draftkings-system/
├── training/                    # ML training pipelines
│   ├── mlb/                     #   MLB: config, feature_engine, model_builder, validator, training
│   └── nba/                     #   NBA: nba_config, nba_training, nba_feature_engine, nba_scraper
│
├── optimization/                # Core optimization engines
│   ├── makrovchain_optimizer.py #   Full Markov chain optimizer (8900+ lines)
│   ├── pulp_lineup_optimizer.py #   PuLP BILP solver
│   ├── genetic_algo_*.py        #   Genetic algorithm solvers (NFL, MLB)
│   ├── dfs_risk_engine.py       #   Kelly, GARCH, VaR, Sharpe
│   ├── rl_hyperopt/             #   RL hyperparameter optimization
│   └── rl_parlay_system/        #   RL parlay system
│
├── pipeline/                    # Orchestration
│   ├── run_pipeline.py          #   Master fetch → train → bridge pipeline
│   ├── backtest_optimizer.py    #   Backtesting framework
│   └── dk_bridge.py             #   DK CSV converter
│
├── web_optimizer/               # Web application
│   ├── client/                  #   React 18 + TypeScript + Vite + Radix UI
│   └── server/                  #   Express + WebSocket + Python subprocess bridge
│
├── scripts/                     # Utility scripts (FanGraphs scraper, etc.)
├── docs/                        # Documentation
├── tests/                       # Tests
├── data/                        # Data files (gitignored)
└── _archive/                    # Deprecated experimental code
```

---

## Mathematical Foundations

### Core Optimization: Binary Integer Linear Programming

```
Maximize: Σ (projection[i] × x[i])

Subject to:
  Σ (salary[i] × x[i]) ≤ SALARY_CAP
  Σ x[i] = LINEUP_SIZE
  Position constraints per sport
  Stacking constraints (if applicable)
  x[i] ∈ {0, 1}
```

Solved via PuLP (CBC backend). Projections are **immutable** ILP inputs — never modified by quant adjustments (Bertsimas & Tsitsiklis, 1997).

### Quantitative Engine

| Method | Purpose |
|--------|---------|
| **Monte Carlo Simulation** (2K-10K sims) | Per-lineup VaR, CVaR, Sharpe ratio, ceiling probability |
| **Kelly Criterion** | Optimal player exposure limits from edge/variance ratio |
| **Mean-Variance (Markowitz)** | 5 optimization strategies: Combined, Kelly, Mean-Variance, Risk Parity, Equal Weight |
| **GARCH Volatility** | Dynamic risk estimates from historical performance |
| **Portfolio Analysis** | Cross-lineup Sharpe, uniqueness, exposure concentration (Herfindahl) |

### Genetic Algorithm Diversity

Generates diverse multi-lineup pools via population-based evolution with tournament selection, crossover, mutation, and minimum Hamming distance enforcement.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, TypeScript, Vite 6.4, Radix UI |
| **Backend** | Node.js, Express 4.17, WebSocket |
| **Optimization** | Python 3.8+, PuLP, NumPy, SciPy |
| **ML Pipeline** | StackingRegressor, XGBoost, LightGBM, Optuna |
| **Data** | Pandas, SportsData.io API, FanGraphs |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
