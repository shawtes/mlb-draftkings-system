"""
config.py — CLI arguments, constants, feature lists, and league averages.

Part of the MLB DFS Training Pipeline v2.
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
        # Fallback: read from OS
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
    """Parse command-line arguments for the training pipeline."""
    hw = detect_hardware()

    parser = argparse.ArgumentParser(description='MLB DFS Training Pipeline v2')
    parser.add_argument('--data-path', type=str,
                        default=os.environ.get('MLB_DATA_PATH'),
                        help='Path to merged_fangraphs_data.csv')
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('MLB_OUTPUT_DIR'),
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
    parser.add_argument('--pregame', action='store_true', default=True,
                        help='Use only pre-game features (default: True)')
    parser.add_argument('--no-pregame', dest='pregame', action='store_false',
                        help='Use all features including same-game (leaky, for comparison)')
    # Matchup engine flags
    parser.add_argument('--matchup', action='store_true', default=False,
                        help='Enable pitcher-batter matchup features')
    parser.add_argument('--pitcher-csv', type=str, default=None,
                        help='Path to FanGraphs pitcher game log CSV')
    parser.add_argument('--matchup-cache-dir', type=str, default=None,
                        help='Directory to cache pybaseball scrapes for matchup engine')
    # Database integration
    parser.add_argument('--from-db', action='store_true', default=False,
                        help='Load training data from SQLite DB instead of CSV')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to SQLite database (default: data/dfs.db)')
    args = parser.parse_known_args()[0]

    # --no-gpu overrides --use-gpu
    if args.no_gpu:
        args.use_gpu = False

    # Attach hardware info for downstream logging
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
# League averages (2020-2024)
# ---------------------------------------------------------------------------
LEAGUE_AVG_WOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315,
}

LEAGUE_AVG_HR_FLYBALL = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141,
}

WOBA_WEIGHTS = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04},
}

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
SELECTED_FEATURES = [
    'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+',
    'flyBalls', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
    'singles', 'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',
    'RE24', 'REW', 'SLG', 'WPA/LI', 'AB',
    # Engineered Statcast features
    'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 'WPA/LI_Statcast',
    'Name_encoded', 'team_encoded',
]

NUMERIC_FEATURES = [
    # Core sabermetric
    'wOBA', 'BABIP', 'ISO', 'wRAA', 'wRC', 'wRC+', 'flyBalls',
    'year', 'month', 'day',
    # Rolling stats
    'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
    'rolling_mean_fpts_49',
    # Statcast
    'wOBA_Statcast', 'SLG_Statcast', 'RAR_Statcast', 'Offense_Statcast',
    'Dollars_Statcast', 'WPA/LI_Statcast',
    'Off', 'WAR', 'Dol', 'RAR', 'RE24', 'REW', 'SLG', 'WPA/LI', 'AB',
    # Probabilistic features
    'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
    'skewness_7d', 'skewness_14d', 'skewness_30d',
    'kurtosis_7d', 'kurtosis_14d', 'kurtosis_30d',
    'var_95_7d', 'var_95_14d', 'var_95_30d',
    'var_99_7d', 'var_99_14d', 'var_99_30d',
    'expected_shortfall_7d', 'expected_shortfall_14d', 'expected_shortfall_30d',
    'tail_ratio', 'prob_exceed_5', 'prob_exceed_10', 'prob_exceed_15', 'prob_exceed_20',
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

CATEGORICAL_FEATURES = ['Name', 'Team']

# Copula feature prefixes (used by reduce_copula_features PCA)
COPULA_PREFIXES = [
    'gaussian_copula_', 'clayton_copula_',
    'upper_tail_dep_', 'lower_tail_dep_',
]

# ---------------------------------------------------------------------------
# Pre-game feature configuration (leakage-free)
# ---------------------------------------------------------------------------

# Per-game stats to convert into lagged rolling averages.
# These are SAME-GAME values in FanGraphs game logs — cannot use raw.
PREGAME_LAG_STATS = [
    # Counting stats
    'HR', 'RBI', 'R', 'H', 'BB', 'SB', 'AB', '1B', '2B', '3B', 'HBP', 'SO',
    # Rate stats (per-game from FanGraphs)
    'AVG', 'OBP', 'SLG', 'ISO', 'BABIP', 'wOBA',
    # Advanced per-game metrics
    'Off', 'WAR', 'wRC', 'wRC+', 'wRAA', 'RE24', 'WPA/LI', 'BsR',
    # DK points (for rolling performance features)
    'calculated_dk_fpts',
]

PREGAME_LAG_WINDOWS = [7, 14, 28]

# Probabilistic/rolling features to shift by 1 game (they include current row)
PREGAME_SHIFT_FEATURES = [
    'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
    'skewness_7d', 'skewness_14d', 'skewness_30d',
    'kurtosis_7d', 'kurtosis_14d', 'kurtosis_30d',
    'var_95_7d', 'var_99_7d',
    'expected_shortfall_7d',
    'tail_ratio', 'prob_exceed_5', 'prob_exceed_10', 'prob_exceed_15', 'prob_exceed_20',
    'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
    'entropy', 'hurst_exponent', 'rolling_sharpe',
    'avg_player_correlation', 'correlation_volatility',
    'network_centrality', 'network_clustering', 'network_volatility', 'network_efficiency',
]

# Final pre-game numeric feature list (only features knowable before first pitch)
PREGAME_NUMERIC_FEATURES = (
    # Calendar (always safe)
    ['year', 'month', 'day']
    # Already-lagged DK point features (.shift(1) in engineer_features)
    + ['lag_mean_fpts_3', 'lag_mean_fpts_7', 'lag_mean_fpts_14', 'lag_mean_fpts_28',
       'lag_max_fpts_3', 'lag_max_fpts_7', 'lag_max_fpts_14', 'lag_max_fpts_28',
       'lag_min_fpts_3', 'lag_min_fpts_7', 'lag_min_fpts_14', 'lag_min_fpts_28']
    # Lagged rolling averages of per-game stats (computed by prepare_pregame_features)
    # Format: pgm_{stat}_{window}  e.g. pgm_HR_7, pgm_SLG_14, pgm_Off_28
    + [f'pgm_{stat}_{w}' for stat in PREGAME_LAG_STATS for w in PREGAME_LAG_WINDOWS]
    # Shifted probabilistic features (pgm_ prefix)
    + [f'pgm_{f}' for f in PREGAME_SHIFT_FEATURES]
    # Shifted copula PCA (pgm_ prefix)
    + [f'pgm_copula_pc_{i}' for i in range(1, 9)]
    # Player-level historical features (stable across games, minimal current-game bias)
    + ['evt_return_level', 'evt_location', 'evt_scale',
       'pot_threshold', 'pot_excess_mean',
       'spectral_entropy', 'dominant_frequency']
)

# ---------------------------------------------------------------------------
# Matchup engine feature configuration (opposing pitcher context)
# ---------------------------------------------------------------------------

MATCHUP_INTERACTION_FEATURES = [
    'matchup_K_rate',
    'matchup_power_vs_flyball',
    'matchup_contact_advantage',
    'matchup_discipline_edge',
    'matchup_K_differential',
    'matchup_ground_ball_risk',
    'matchup_quality_score',
    'matchup_swstr_suppression',
]

# Pitcher stats whose rolling averages become opp_* features
MATCHUP_PITCHER_STATS = [
    'K/9', 'BB/9', 'HR/9', 'ERA', 'FIP', 'xFIP', 'WHIP',
    'K%', 'BB%', 'GB%', 'FB%', 'SwStr%', 'Contact%',
    'O-Swing%', 'Z-Contact%', 'SIERA', 'LOB%',
    'ERA-', 'FIP-', 'HR/FB', 'LD%', 'BABIP',
]

MATCHUP_PITCHER_ARSENAL_STATS = [
    'FBv', 'FB%.1', 'SL%', 'SLv', 'CH%', 'CHv', 'CB%', 'CBv',
]

MATCHUP_ROLLING_WINDOWS = [7, 14, 28]

# All possible opp_* feature names (generated from stats x windows + season)
MATCHUP_OPP_FEATURES = (
    [f'opp_{stat}_{w}g'
     for stat in MATCHUP_PITCHER_STATS + MATCHUP_PITCHER_ARSENAL_STATS
     for w in MATCHUP_ROLLING_WINDOWS]
    + [f'opp_{stat}_season'
       for stat in MATCHUP_PITCHER_STATS + MATCHUP_PITCHER_ARSENAL_STATS]
)

# Combined: all matchup feature names (for extending PREGAME_NUMERIC_FEATURES)
MATCHUP_ALL_FEATURES = MATCHUP_OPP_FEATURES + MATCHUP_INTERACTION_FEATURES
