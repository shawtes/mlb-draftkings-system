# MLB DraftKings Optimization System

A production-grade daily fantasy sports optimization platform combining advanced mathematical optimization, machine learning, and quantitative finance techniques to generate optimal MLB DraftKings lineups.

## Overview

This system solves the complex constrained optimization problem of selecting optimal fantasy sports lineups by integrating:

- **Binary Integer Linear Programming (BILP)** for mathematically optimal solutions
- **Genetic Algorithms** for diverse multi-lineup generation
- **Ensemble Machine Learning** for player performance prediction
- **Quantitative Finance Methods** for risk-adjusted optimization
- **Real-time Web Application** for interactive lineup generation

The platform processes player projections, applies sophisticated constraints (salary caps, position requirements, stacking strategies), and generates 1-500+ unique lineups optimized for different contest types (cash games vs. tournaments).

---

## Mathematical Foundations

### Core Optimization Problem

The system solves a **Binary Integer Linear Programming (BILP)** problem:

**Objective Function:**
```
Maximize: Σ (player_points[i] × x[i])
```

**Constraints:**
```
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

### PuLP Linear Programming Solver

**PuLP** (Python Linear Programming) uses the **Simplex Method** and **Branch-and-Bound** algorithms to find mathematically optimal solutions. The solver handles:

- Binary decision variables for each player
- Linear objective function (maximize projected points)
- Linear constraints (salary cap, position requirements, stacking rules)
- Fast convergence (<1 second per lineup)

**Implementation:**
```python
problem = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)
player_vars = {idx: pulp.LpVariable(f"player_{idx}", cat='Binary') 
               for idx in df.index}

# Objective: Maximize points
problem += pulp.lpSum([
    df.at[idx, 'Predicted_DK_Points'] * player_vars[idx] 
    for idx in df.index
])

# Constraints
problem += pulp.lpSum(player_vars.values()) == LINEUP_SIZE
problem += pulp.lpSum([
    df.at[idx, 'Salary'] * player_vars[idx] 
    for idx in df.index
]) <= SALARY_CAP
```

### Genetic Algorithm Diversity Engine

To generate multiple diverse lineups (required for multi-entry contests), the system employs a **Genetic Algorithm** with:

**Phase 1: Initial Population**
- Generate 3x candidate lineups using controlled randomness
- Apply 35-70% lognormal noise to player projections
- Randomly boost 3-7 players and penalize 2-4 players
- Each variant produces a different optimal lineup

**Phase 2: Evolution (3 Generations)**
- **Tournament Selection**: Keep top 50% of lineups by projected points
- **Crossover**: Combine players from two parent lineups (60% parent1, 40% parent2)
- **Mutation**: Randomly replace 1-2 players (30% mutation rate)
- **Diversity Enforcement**: Remove lineups with >70% player overlap

**Phase 3: Diverse Subset Selection**
- Use maximal diversity algorithm to select final lineups
- Maximize minimum Hamming distance between selected lineups
- Ensures each lineup differs by at least 3-4 players

### Advanced Quantitative Methods

#### Mean-Variance Optimization (Cash Games)
For double-up and 50/50 contests, the system uses **Sharpe Ratio optimization**:

```
Objective: maximize μ^T x - λ * √(x^T Σ x)
```

Where:
- μ = expected points vector
- Σ = covariance matrix of player performance
- λ = risk aversion parameter
- x = binary selection vector

#### Monte Carlo Simulation
Simulates 10,000+ scenarios for robust risk assessment:
- Samples from player point distributions
- Calculates Value at Risk (VaR) at 95% confidence
- Computes Conditional VaR (CVaR) for tail risk
- Provides probability distributions of lineup outcomes

#### GARCH Volatility Estimation
Models time-varying volatility of player performance:
- Fits GARCH(1,1) models to historical performance
- Provides dynamic risk estimates
- Adjusts projections based on recent volatility

#### Kelly Criterion Position Sizing
Optimal bet sizing based on expected win rate:
```
Kelly Fraction = (p × b - q) / b
```
Where p = win probability, q = 1-p, b = win/loss ratio

#### Copula-Based Dependency Modeling
Models complex player correlations using Gaussian copulas:
- Captures non-linear dependencies between players
- Generates correlated performance scenarios
- Improves stacking strategy effectiveness

#### Bayesian Probability Modeling
- Posterior parameter estimation for player performance
- Confidence intervals for projections
- Correlation modeling between teammates
- Regime-aware adjustments (home/away, weather, matchups)

---

## Technology Stack & Architecture

### Backend Optimization Engine

**Core Technologies:**
- **Python 3.8+** - Primary optimization and ML pipeline
- **PuLP 2.7.0+** - Linear programming solver (CBC backend)
- **Pandas 2.0+** - Data manipulation and analysis
- **NumPy 1.24+** - Numerical computations
- **PyQt5 5.15+** - Desktop GUI application

**Machine Learning Pipeline:**
- **StackingRegressor** - Multi-level ensemble model
  - Base models: Linear Regression, Ridge, Random Forest, XGBoost
  - Meta-learner: Ridge regression with cross-validation
- **Feature Engineering**: 500+ engineered features
  - Rolling statistics (5-game, 7-game, 49-game averages)
  - Lag features (previous game performance)
  - Opponent-adjusted metrics
  - Time series features
- **Cross-Validation**: 3-fold CV with TimeSeriesSplit
- **Hyperparameter Optimization**: Grid search with 15 iterations

**Parallel Processing:**
- **ThreadPoolExecutor** - Multi-threaded lineup generation
- **CPU Workers**: 4-8 workers (adaptive based on system)
- **Performance**: 4-8x speedup for multi-lineup generation

### Web Application

**Frontend:**
- **React 18.3** - Component-based UI framework
- **TypeScript 5.9** - Type-safe development
- **Material-UI 5.14** - Design system and components
- **Radix UI** - Accessible component primitives
- **Vite 6.3** - Build tool and dev server
- **WebSocket** - Real-time optimization progress

**Backend:**
- **Node.js 16+** - Runtime environment
- **Express 4.17** - Web framework
- **WebSocket (ws 8.18)** - Real-time communication
- **Multer** - File upload handling
- **CSV Parser** - Data processing

**Architecture:**
- RESTful API for data operations
- WebSocket for real-time updates
- Stateless server design
- CORS-enabled for cross-origin requests

### Data Sources

- **SportsData.io API** - Player projections, salaries, injuries, game data
- **DraftKings** - Salary and contest data
- **Historical Performance Data** - 3+ years of MLB statistics

---

## System Architecture

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                      │
├─────────────────────────────────────────────────────────┤
│  SportsData.io API → Player Projections                 │
│  DraftKings API → Salaries & Contest Data              │
│  Historical Database → Performance Metrics              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  MACHINE LEARNING TRAINING              │
├─────────────────────────────────────────────────────────┤
│  Feature Engineering (500+ features)                     │
│  StackingRegressor Ensemble Training                   │
│  Cross-Validation & Hyperparameter Tuning              │
│  Model Persistence & Versioning                        │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    PREDICTION GENERATION                 │
├─────────────────────────────────────────────────────────┤
│  Load Trained Model                                     │
│  Generate Player Projections                            │
│  Calculate Ceiling/Floor/Median                         │
│  Apply Contest-Specific Adjustments                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  OPTIMIZATION ENGINE                     │
├─────────────────────────────────────────────────────────┤
│  PuLP Linear Programming (Core Solver)                   │
│  Genetic Algorithm (Diversity Engine)                    │
│  Advanced Quantitative Methods (Optional)                │
│  Constraint Satisfaction (Salary, Position, Stacking)    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                      │
├─────────────────────────────────────────────────────────┤
│  Deduplication                                          │
│  Exposure Tracking                                      │
│  Position Assignment (FLEX handling)                    │
│  DraftKings CSV Format Export                           │
└─────────────────────────────────────────────────────────┘
```

### Component Architecture

**1. Training Module** (`1_CORE_TRAINING/`)
- Model training pipeline
- Feature engineering
- Hyperparameter optimization
- Model evaluation and validation

**2. Prediction Module** (`2_PREDICTIONS/`)
- Daily prediction generation
- Probability modeling
- Ceiling/floor calculations
- Contest-specific adjustments

**3. Optimization Module** (`6_OPTIMIZATION/`)
- Core optimization engine (PuLP + Genetic)
- Stacking strategy implementation
- Exposure management
- Lineup generation and export

**4. Analysis Module** (`7_ANALYSIS/`)
- Model performance evaluation
- Prediction accuracy metrics
- Lineup scoring and analysis
- Performance tracking

**5. Web Application** (`web_optimizer/`)
- React frontend for interactive optimization
- Node.js backend API
- Real-time WebSocket updates
- File upload and CSV export

---

## Key Features

### Multi-Algorithm Optimization
- **PuLP Linear Programming**: Mathematically optimal single lineup
- **Genetic Algorithm**: Diverse multi-lineup generation (1-500+ lineups)
- **Hybrid Approach**: Combines optimality with diversity

### Advanced Stacking Strategies
- **QB + WR Stacks**: Correlation-based stacking
- **Game Stacks**: Multiple players from high-scoring games
- **Bring-Back Plays**: Opposing team players in game stacks
- **Multi-Stack Combinations**: Complex 4|2|2 and 3|3|2 patterns

### Contest-Specific Optimization
- **Cash Games (50/50, Double-Up)**: Floor-focused, high-ownership plays
- **Tournaments (GPP)**: Ceiling-focused, contrarian plays
- **Ownership-Based Adjustments**: Fade chalk, target low-owned elite plays

### Exposure Management
- **Player Exposure Limits**: Min/max percentage across lineups
- **Team Exposure Controls**: Limit team representation
- **Stack Exposure Tracking**: Monitor stacking frequency

### Real-Time Generation
- **Interactive Web Interface**: Upload data, configure settings, generate lineups
- **Progress Tracking**: Real-time WebSocket updates during optimization
- **Desktop GUI**: PyQt5 application for offline use

### DraftKings Integration
- **CSV Export**: DraftKings-ready format
- **Position Validation**: Automatic FLEX assignment
- **Salary Cap Management**: $45,000 - $50,000 range
- **Injury Filtering**: Automatic exclusion of injured players

---

## Performance Metrics

### Optimization Speed

| Lineups | Players | Stacks | Time (Mac M1) | Time (Windows 4-core) |
|---------|---------|--------|---------------|----------------------|
| 1       | 100     | 1      | <1 sec        | <1 sec               |
| 20      | 100     | 2      | 5-15 sec      | 15-30 sec            |
| 50      | 150     | 3      | 30-60 sec      | 60-120 sec           |
| 100     | 200     | 5      | 1-2 min        | 2-4 min              |

**Parallel Processing**: 4-8x speedup with multi-threading

### Contest Performance

**Production Results (NFL Contest - 47,562 entries):**

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Best Score** | 149.98 | 217.94 | **+45.3%** |
| **Average Score** | 113.84 | 130.50 | **+14.6%** |
| **Cash Rate** | 10% | 40% | **+300%** |
| **Tournament Rank** | N/A | Top 0.01% | Winner |

**Key Achievement**: Generated lineup scored 217.94 points, beating contest winner (213.34 points) by 4.60 points.

### Model Performance

**Machine Learning Metrics:**
- **Mean Absolute Error**: 3.907 fantasy points
- **R² Score**: 0.157
- **Prediction Range**: 0.0 to 100.0 points
- **Training Time**: ~5.5 hours (171,479 samples, 500 features)
- **GPU Acceleration**: 2x faster with CUDA-enabled XGBoost

**Training Efficiency:**
- **Data Preprocessing**: 19.1 seconds (171k rows, 258 columns)
- **Feature Selection**: 500 features from 258 original columns
- **Cross-Validation**: 3-fold CV with 15 hyperparameter combinations
- **Memory Usage**: <16GB RAM with optimized chunking

---

## System Configuration

### Environment Requirements

**Python Environment:**
- Python 3.8 or higher
- Required packages:
  - `pulp>=2.7.0` - Linear programming solver
  - `pandas>=2.0.0` - Data manipulation
  - `numpy>=1.24.0` - Numerical computations
  - `PyQt5>=5.15.0` - Desktop GUI (optional)
  - `scikit-learn` - Machine learning models
  - `xgboost` - Gradient boosting (optional, for GPU acceleration)

**Node.js Environment:**
- Node.js 16 or higher
- npm or yarn package manager
- Required for web application only

**Hardware Recommendations:**
- **CPU**: 4+ cores (8+ recommended for parallel processing)
- **RAM**: 16GB minimum (32GB recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional, for XGBoost acceleration)
- **Storage**: SSD recommended for faster I/O operations

### API Configuration

**SportsData.io API:**
- API key required for player data, projections, and salaries
- Configuration in environment variables or config files
- Rate limiting handled automatically

**DraftKings Integration:**
- CSV file upload for player data
- Automatic salary and position parsing
- Export to DraftKings CSV format

### Optimization Parameters

**Default Settings:**
- **Salary Cap**: $50,000 (NFL/NBA), $45,000-$50,000 (MLB)
- **Lineup Size**: 9 players (NFL), 8 players (NBA), 9 players (MLB)
- **Position Limits**: Sport-specific (e.g., NFL: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DST)
- **Minimum Salary Usage**: Configurable (default: $48,000+)
- **Uniqueness Requirement**: Minimum 3-4 different players between lineups

**Stacking Configuration:**
- Stack types: QB+WR, QB+WR+TE, Game Stack, Bring-Back
- Minimum players per stack: 2-5 players
- Stack exposure limits: 0-100% per stack type

### ML Model Configuration

**Feature Engineering:**
- **Total Features**: 500+ engineered features
- **Feature Selection**: SelectKBest with f_regression (top 30-150 features)
- **Scaling**: StandardScaler for numeric features
- **Categorical Encoding**: OneHotEncoder for player/team/opponent

**Model Hyperparameters:**
- **StackingRegressor**:
  - Base models: Linear Regression, Ridge (α=1.0), Random Forest (100 trees), XGBoost (100 trees)
  - Meta-learner: Ridge (α=0.1)
  - Cross-validation: 3-fold
- **XGBoost** (when used):
  - n_estimators: 100
  - max_depth: 6-8
  - learning_rate: 0.1
  - GPU acceleration: Enabled if available

**Training Configuration:**
- **Dataset Size**: Auto-limited to 100K rows for efficiency
- **Chunk Size**: 15,000 rows (optimized for 16GB RAM)
- **Hyperparameter Iterations**: 15 combinations
- **Cross-Validation Folds**: 3-fold (balanced speed/accuracy)

---

## File Structure

```
mlb-draftkings-system/
├── 1_CORE_TRAINING/              # Machine learning model training
│   ├── training.py               # Main training pipeline
│   ├── stacking_ml_engine.py    # StackingRegressor implementation
│   ├── advanced_ml_models.py    # Advanced ML models
│   ├── ensemble_models.py        # Ensemble methods
│   ├── feature_engineering/      # Feature creation utilities
│   └── model_cache/              # Trained model storage
│
├── 2_PREDICTIONS/                # Daily prediction generation
│   ├── predction01.py            # Main prediction script
│   ├── add_probability_predictions.py  # Probability modeling
│   └── [prediction outputs]/     # Generated CSV files
│
├── 5_DRAFTKINGS_ENTRIES/         # Entry formatting utilities
│   ├── dk_entries_salary_generator.py
│   └── dk_file_handler.py
│
├── 6_OPTIMIZATION/               # Core optimization engine
│   ├── genetic_algo_nfl_optimizer.py    # Main optimizer (NFL)
│   ├── genetic_algo_mlb_optimizer.py    # MLB optimizer
│   ├── optimizer.genetic.algo.py        # Genetic algorithm core
│   ├── pulp_lineup_optimizer.py         # PuLP linear programming
│   ├── advanced_quant_optimizer.py      # Quantitative methods
│   ├── probability_modeling_engine.py  # Probability modeling
│   ├── portfolio_optimization.py        # Portfolio theory
│   ├── OPTIMIZATION_MATH_AND_PIPELINE_EXPLANATION.md  # Math docs
│   └── DOCUMENTATION/            # Implementation guides
│
├── 7_ANALYSIS/                   # Model evaluation and analysis
│   ├── evaluate_models.py        # Model performance evaluation
│   ├── ensemble_model_evaluation.py    # Ensemble analysis
│   ├── model_metrics_evaluation.py      # Metrics calculation
│   └── calculate_lineup_scores.py     # Contest result analysis
│
├── 8_DOCUMENTATION/              # Technical documentation
│   ├── TRAINING_INSTRUCTIONS.md
│   ├── OPTIMIZATION_SUMMARY.md
│   └── [additional docs]/
│
├── 9_BACKUP/                     # Backup files and archives
│
├── web_optimizer/                 # Web application
│   ├── client/                   # React frontend
│   │   ├── src/
│   │   │   ├── components/       # React components
│   │   │   ├── services/         # API and WebSocket services
│   │   │   └── App.tsx           # Main application
│   │   ├── package.json
│   │   └── vite.config.ts
│   ├── server/                   # Node.js backend
│   │   ├── index.js              # Express server
│   │   ├── optimizer.js         # Optimization engine
│   │   └── package.json
│   └── PYTHON_SETUP_GUIDE.md
│
├── python_algorithms/             # Core algorithm implementations
│   ├── sportsdata_nfl_api.py    # API client
│   └── [algorithm utilities]/
│
├── nfl_historical_cache/          # Cached historical data
├── nba_historical_cache/          # Cached historical data
└── README.md                      # This file
```

---

## Maintenance Tasks

### Model Retraining Schedule

**Daily:**
- Generate new predictions for upcoming games
- Update player projections with latest data
- Refresh injury reports and lineup confirmations

**Weekly:**
- Full model retraining with latest week's data
- Feature importance analysis and updates
- Performance evaluation against actual results
- Update bust/elite player lists based on performance

**Monthly:**
- Comprehensive model evaluation
- Hyperparameter re-optimization
- Feature engineering improvements
- System architecture review

### API Key Management

**SportsData.io API:**
- Monitor API usage and rate limits
- Rotate API keys quarterly for security
- Update API endpoints if service changes
- Cache responses to minimize API calls

### Performance Monitoring

**Optimization Performance:**
- Track lineup generation time
- Monitor memory usage during optimization
- Log optimization failures and errors
- Analyze lineup diversity metrics

**Model Performance:**
- Track prediction accuracy (MAE, R²)
- Compare predicted vs. actual player scores
- Monitor model drift over time
- Retrain if accuracy degrades significantly

**Contest Results:**
- Analyze lineup performance post-contest
- Calculate cash rates and ROI
- Identify successful strategies
- Update player exclusion/inclusion lists

### Dependency Updates

**Python Dependencies:**
- Update PuLP, Pandas, NumPy quarterly
- Test compatibility before production deployment
- Monitor security advisories
- Maintain requirements.txt with version pins

**Node.js Dependencies:**
- Update React, Express, WebSocket libraries
- Test web application after updates
- Monitor for breaking changes
- Maintain package.json with version constraints

### Data Pipeline Maintenance

**Data Quality:**
- Validate API responses for completeness
- Check for missing player data
- Verify salary and position accuracy
- Handle API failures gracefully

**Storage Management:**
- Archive old prediction files
- Clean up temporary optimization outputs
- Manage historical cache size
- Backup trained models regularly

**Database Maintenance:**
- Optimize historical data queries
- Index frequently accessed columns
- Archive old contest results
- Maintain data integrity

### System Health Checks

**Weekly:**
- Verify all API connections
- Test optimization engine with sample data
- Validate CSV export formats
- Check web application functionality

**Monthly:**
- Review system logs for errors
- Analyze performance trends
- Update documentation
- Test disaster recovery procedures

---

## License

This project is provided as-is for educational and research purposes.

