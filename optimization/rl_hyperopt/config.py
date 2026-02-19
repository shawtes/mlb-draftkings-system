"""
RL Hyperparameter Optimizer — Configuration

Defines the discrete action space (32 hyperparameter combos), reward weights
for GPP/Cash modes, and training hyperparameters for the DQN agent.
"""

import os
from itertools import product

# ---------------------------------------------------------------------------
# Data path — resolves to <project_root>/data/backtest_results.csv
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data',
    'backtest_results.csv',
)

# ---------------------------------------------------------------------------
# Hyperparameter grid — every combination the optimizer tested
# ---------------------------------------------------------------------------
_MAX_EXPOSURES = [30, 50, 70, 100]
_STACK_OPTIONS = [
    {"stack_enabled": False, "stack_sizes": []},
    {"stack_enabled": True,  "stack_sizes": [2]},
    {"stack_enabled": True,  "stack_sizes": [3]},
    {"stack_enabled": True,  "stack_sizes": [2, 3]},
]
_MIN_UNIQUES = [2, 3]
_MIN_SALARY = 49000

# Build the 32-element action space: 4 exposures x 4 stack options x 2 uniques
ACTION_SPACE = []
for exp, stack_opt, uniq in product(_MAX_EXPOSURES, _STACK_OPTIONS, _MIN_UNIQUES):
    ACTION_SPACE.append({
        "max_exposure": exp,
        "stack_enabled": stack_opt["stack_enabled"],
        "stack_sizes": list(stack_opt["stack_sizes"]),
        "min_unique": uniq,
        "min_salary": _MIN_SALARY,
    })

NUM_ACTIONS = len(ACTION_SPACE)  # 32

# Build a config-label lookup so we can map action index -> CSV config label
def _build_config_label(action):
    """Reproduce the label format used in backtest_results.csv."""
    exp = action["max_exposure"]
    stack = action["stack_sizes"]
    uniq = action["min_unique"]
    sal = action["min_salary"]
    if not action["stack_enabled"]:
        stack_str = "nostack"
    elif stack == [2]:
        stack_str = "stack2"
    elif stack == [3]:
        stack_str = "stack3"
    elif stack == [2, 3]:
        stack_str = "stack23"
    else:
        stack_str = "nostack"
    return f"L20_exp{exp}_{stack_str}_uniq{uniq}_sal{sal}"


ACTION_LABELS = [_build_config_label(a) for a in ACTION_SPACE]

# Reverse mapping: config label -> action index
LABEL_TO_ACTION = {label: idx for idx, label in enumerate(ACTION_LABELS)}

# ---------------------------------------------------------------------------
# Reward weights — per contest mode
# ---------------------------------------------------------------------------
REWARD_WEIGHTS = {
    "gpp": {
        "gpp_ceiling": 0.4,
        "best_actual": 0.3,
        "avg_actual": 0.2,
        "consistency": 0.1,
    },
    "cash": {
        "cash_hit_rate": 0.4,
        "avg_actual": 0.3,
        "consistency": 0.2,
        "pct_optimal": 0.1,
    },
}

# Metrics used by each mode (ordered to match weight dicts)
REWARD_METRICS = {
    "gpp": ["gpp_ceiling", "best_actual", "avg_actual", "consistency"],
    "cash": ["cash_hit_rate", "avg_actual", "consistency", "pct_optimal"],
}

# ---------------------------------------------------------------------------
# Metric ranges (from backtest analysis) — used for normalization
# ---------------------------------------------------------------------------
METRIC_RANGES = {
    "best_actual":   {"min": 0.0, "max": 348.8},
    "avg_actual":    {"min": 0.0, "max": 317.6},
    "gpp_ceiling":   {"min": 0.0, "max": 344.2},
    "cash_hit_rate": {"min": 0.0, "max": 100.0},
    "pct_optimal":   {"min": 0.0, "max": 94.6},
    "consistency":   {"min": 0.0, "max": 33.2},
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
GAMMA = 0.99
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
EPISODES = 500

# State dimensions: 5 continuous features + one-hot of last action (32)
STATE_DIM = 5 + NUM_ACTIONS  # 37
