"""
nba_config.py — CLI arguments, constants, feature lists for NBA DFS Training Pipeline.

Mirrors 1_CORE_TRAINING/config.py but with NBA-specific stats, thresholds,
and feature definitions. Reuses detect_hardware() pattern.
"""

import argparse
import os
import multiprocessing


def detect_hardware():
    """Auto-detect CPU cores, RAM, and GPU availability."""
    cpu_count = multiprocessing.cpu_count()
    ram_gb = 0
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemTotal' in line:
                        ram_gb = int(line.split()[1]) / (1024 ** 2)
                        break
        except Exception:
            ram_gb = 0

    gpu_available = False
    gpu_name = None
    gpu_mem_gb = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except ImportError:
        pass

    return {
        'cpu_count': cpu_count,
        'ram_gb': ram_gb,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'gpu_mem_gb': gpu_mem_gb,
    }


def parse_args():
    """Parse command-line arguments for the NBA training pipeline."""
    hw = detect_hardware()

    parser = argparse.ArgumentParser(description='NBA DFS Training Pipeline')
    parser.add_argument('--data-path', type=str,
                        default=os.environ.get('NBA_DATA_PATH'),
                        help='Path to NBA game logs CSV (from nba_scraper.py)')
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('NBA_OUTPUT_DIR'),
                        help='Directory for all output artifacts')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of walk-forward CV folds')
    parser.add_argument('--gap-days', type=int, default=7,
                        help='Embargo gap (days) between train and test')
    parser.add_argument('--optuna-trials', type=int, default=50,
                        help='Number of Optuna HPO trials')
    parser.add_argument('--skip-hpo', action='store_true',
                        help='Skip Optuna HPO; use HARDCODED_OPTIMAL_PARAMS')
    parser.add_argument('--n-features', type=int, default=100,
                        help='Number of features to select via SHAP')
    # Hardware / parallelism flags
    parser.add_argument('--n-jobs', type=int, default=hw['cpu_count'],
                        help=f'Parallel workers (default: {hw["cpu_count"]} = all cores)')
    parser.add_argument('--use-gpu', action='store_true', default=hw['gpu_available'],
                        help='Use GPU for LightGBM/XGBoost/CatBoost (auto-detected)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU-only even if GPU is available')
    parser.add_argument('--feature-jobs', type=int,
                        default=min(hw['cpu_count'], 12),
                        help='Workers for feature engineering groupby loops')
    # Season filtering
    parser.add_argument('--min-season', type=str, default=None,
                        help='Minimum season to include (e.g., "2014-15")')
    parser.add_argument('--min-games', type=int, default=10,
                        help='Minimum games per player to include in training')
    args = parser.parse_known_args()[0]

    if args.no_gpu:
        args.use_gpu = False

    args.hardware = hw
    return args


# ---------------------------------------------------------------------------
# Hard-coded optimal XGBoost parameters (pre-tuned, used when --skip-hpo)
# ---------------------------------------------------------------------------
HARDCODED_OPTIMAL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

# ---------------------------------------------------------------------------
# NBA League Averages (approximate, 2020-2025)
# Used for advanced stat normalization and Marcel shrinkage
# ---------------------------------------------------------------------------
NBA_LEAGUE_AVG = {
    'FG_PCT': 0.471,
    'FG3_PCT': 0.362,
    'FT_PCT': 0.781,
    'TS_PCT': 0.572,     # True Shooting %
    'EFG_PCT': 0.539,    # Effective FG %
    'PTS_PER_MIN': 0.52,
    'REB_PER_MIN': 0.21,
    'AST_PER_MIN': 0.12,
}

# ---------------------------------------------------------------------------
# NBA stat columns available from nba_api PlayerGameLogs
# ---------------------------------------------------------------------------
NBA_BOX_SCORE_STATS = [
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG_PCT',
    'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT',
    'OREB', 'DREB',
    'PF', 'PLUS_MINUS',
    'MIN',
]

# Core DK-relevant counting stats for rolling feature engines
NBA_COUNTING_STATS = [
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'PF', 'PLUS_MINUS',
]

# Copula key metrics (highest fantasy relevance)
NBA_COPULA_KEY_METRICS = [
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'calculated_dk_fpts',
]

# Marcel shrinkage columns (rate stats that regress to mean for small samples)
NBA_MARCEL_SHRINK_COLS = [
    'FG_PCT', 'FG3_PCT', 'FT_PCT',
    'ts_pct', 'efg_pct',
]

# Copula feature prefixes (for PCA reduction)
COPULA_PREFIXES = [
    'gaussian_copula_', 'clayton_copula_',
    'upper_tail_dep_', 'lower_tail_dep_',
]

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ['Name', 'Team']

# Numeric features used in non-pregame (leaky) mode — for comparison only
NUMERIC_FEATURES = [
    # Core box score
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
    'PF', 'PLUS_MINUS', 'MIN',
    # Calendar
    'year', 'month', 'day',
    # Derived advanced stats
    'ts_pct', 'efg_pct', 'ast_tov_ratio', 'stl_blk_sum',
    'pts_per_min', 'reb_per_min', 'ast_per_min',
    'usage_proxy', 'fantasy_per_min',
    # Rolling stats
    'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
    'rolling_mean_fpts_49',
    # Probabilistic features
    'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
    'skewness_7d', 'skewness_14d', 'skewness_30d',
    'kurtosis_7d', 'kurtosis_14d', 'kurtosis_30d',
    'var_95_7d', 'var_95_14d', 'var_95_30d',
    'var_99_7d', 'var_99_14d', 'var_99_30d',
    'expected_shortfall_7d', 'expected_shortfall_14d', 'expected_shortfall_30d',
    'tail_ratio',
    'prob_exceed_15', 'prob_exceed_20', 'prob_exceed_25',
    'prob_exceed_30', 'prob_exceed_35', 'prob_exceed_40',
    'prob_exceed_45', 'prob_exceed_50', 'prob_exceed_60',
    'avg_player_correlation', 'correlation_volatility',
    'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
    'entropy', 'hurst_exponent', 'max_drawdown', 'current_drawdown',
    'drawdown_duration', 'rolling_sharpe',
    # Extreme value theory
    'evt_location', 'evt_scale', 'evt_shape', 'evt_return_level',
    'exceedance_prob', 'extreme_value_index', 'pot_threshold',
    'pot_excess_mean', 'pot_excess_std',
    # Network features
    'network_centrality', 'network_clustering', 'network_volatility', 'network_efficiency',
    # Spectral features
    'dominant_frequency', 'spectral_entropy', 'spectral_centroid',
    'spectral_rolloff', 'rolling_spectral_entropy',
]

# ---------------------------------------------------------------------------
# Pre-game feature configuration (leakage-free)
# ---------------------------------------------------------------------------

# Per-game stats to convert into lagged rolling averages.
PREGAME_LAG_STATS = [
    # Counting stats
    'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
    'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
    'OREB', 'DREB', 'PF', 'PLUS_MINUS', 'MIN',
    # Rate stats
    'FG_PCT', 'FG3_PCT', 'FT_PCT',
    # Derived advanced stats
    'ts_pct', 'efg_pct', 'ast_tov_ratio', 'usage_proxy',
    'pts_per_min', 'reb_per_min', 'ast_per_min', 'fantasy_per_min',
    # DK points
    'calculated_dk_fpts',
]

PREGAME_LAG_WINDOWS = [7, 14, 28]

# Probabilistic/rolling features to shift by 1 game
PREGAME_SHIFT_FEATURES = [
    'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
    'skewness_7d', 'skewness_14d', 'skewness_30d',
    'kurtosis_7d', 'kurtosis_14d', 'kurtosis_30d',
    'var_95_7d', 'var_99_7d',
    'expected_shortfall_7d',
    'tail_ratio',
    'prob_exceed_15', 'prob_exceed_20', 'prob_exceed_25',
    'prob_exceed_30', 'prob_exceed_35', 'prob_exceed_40',
    'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
    'entropy', 'hurst_exponent', 'rolling_sharpe',
    'avg_player_correlation', 'correlation_volatility',
    'network_centrality', 'network_clustering', 'network_volatility', 'network_efficiency',
]

# Final pre-game numeric feature list
PREGAME_NUMERIC_FEATURES = (
    # Calendar (always safe)
    ['year', 'month', 'day']
    # Already-lagged DK point features (.shift(1) in engineer_features)
    + ['lag_mean_fpts_3', 'lag_mean_fpts_7', 'lag_mean_fpts_14', 'lag_mean_fpts_28',
       'lag_max_fpts_3', 'lag_max_fpts_7', 'lag_max_fpts_14', 'lag_max_fpts_28',
       'lag_min_fpts_3', 'lag_min_fpts_7', 'lag_min_fpts_14', 'lag_min_fpts_28']
    # Lagged rolling averages of per-game stats (computed by prepare_pregame_features)
    + [f'pgm_{stat}_{w}' for stat in PREGAME_LAG_STATS for w in PREGAME_LAG_WINDOWS]
    # Shifted probabilistic features
    + [f'pgm_{f}' for f in PREGAME_SHIFT_FEATURES]
    # Shifted copula PCA
    + [f'pgm_copula_pc_{i}' for i in range(1, 9)]
    # Player-level historical features (stable across games)
    + ['evt_return_level', 'evt_location', 'evt_scale',
       'pot_threshold', 'pot_excess_mean',
       'spectral_entropy', 'dominant_frequency']
)

# NBA DK fantasy point probability thresholds (higher than MLB due to score range)
NBA_PROB_THRESHOLDS = [15, 20, 25, 30, 35, 40, 45, 50, 60]
