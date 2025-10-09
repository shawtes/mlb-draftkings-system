"""
Configuration file for MLB RL Team Selector

This file contains all the configuration parameters for the reinforcement learning
team selection system.
"""

# Data paths
DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
MODEL_SAVE_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/mlb_rl_model.pth'
RESULTS_PATH = '4_DATA/rl_results.csv'
PLOTS_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/rl_plots.png'

# DraftKings constraints
SALARY_CAP = 50000
LINEUP_SIZE = 8
POSITION_CONSTRAINTS = {
    'C': 1,     # Catcher
    '1B': 1,    # First Base
    '2B': 1,    # Second Base
    '3B': 1,    # Third Base
    'SS': 1,    # Shortstop
    'OF': 3,    # Outfield
    'UTIL': 1   # Utility (any position)
}

# RL Training parameters
TRAINING_EPISODES = 2000
SAVE_FREQUENCY = 200
BATCH_SIZE = 32
MEMORY_SIZE = 100000

# DQN parameters
LEARNING_RATE = 0.001
GAMMA = 0.99          # Discount factor
EPSILON = 1.0         # Initial exploration rate
EPSILON_MIN = 0.01    # Minimum exploration rate
EPSILON_DECAY = 0.995 # Exploration decay rate
HIDDEN_DIM = 512      # Neural network hidden layer size
UPDATE_TARGET_FREQ = 100  # How often to update target network

# Walk-forward validation parameters
INITIAL_TRAIN_DAYS = 365    # Initial training period (days)
VALIDATION_WINDOW = 1       # Days to validate on each step
RETRAIN_FREQUENCY = 7       # How often to retrain (days)
MAX_VALIDATIONS = 50        # Maximum validation steps

# Feature engineering parameters
ROLLING_WINDOWS = [3, 7, 14, 28, 45]  # Rolling window sizes for features
STAT_COLUMNS = [
    'HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts',
    'AVG', 'OBP', 'SLG', 'wOBA', 'wRC+', 'BABIP', 'ISO', 'SO', 'PA', 'AB'
]

# DraftKings scoring system
DK_SCORING = {
    '1B': 3,     # Single
    '2B': 5,     # Double
    '3B': 8,     # Triple
    'HR': 10,    # Home Run
    'RBI': 2,    # RBI
    'R': 2,      # Run
    'BB': 2,     # Walk
    'HBP': 2,    # Hit by Pitch
    'SB': 5      # Stolen Base
}

# Environment parameters
REWARD_SCALING = {
    'PLAYER_POINTS': 0.1,       # Scale factor for player fantasy points
    'SALARY_EFFICIENCY': 0.05,  # Reward for salary efficiency
    'LINEUP_COMPLETION': 0.5,   # Bonus for completing lineup
    'INVALID_ACTION': -1.0,     # Penalty for invalid actions
    'INCOMPLETE_LINEUP': -5.0   # Penalty per missing player
}

# Evaluation parameters
EVALUATION_EPISODES = 100
BASELINE_METHODS = ['random', 'top_salary', 'top_avg_points']

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Device configuration
USE_CUDA = True  # Use GPU if available
DEVICE = 'cuda' if USE_CUDA else 'cpu'

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation date ranges
VALIDATION_START_DATE = '2024-01-01'
VALIDATION_END_DATE = '2024-12-31'

# Model checkpointing
CHECKPOINT_FREQUENCY = 500  # Save checkpoint every N episodes
KEEP_BEST_MODEL = True     # Keep track of best performing model

# Performance thresholds
MIN_IMPROVEMENT_THRESHOLD = 0.5  # Minimum improvement to consider model better
CONVERGENCE_PATIENCE = 100       # Episodes to wait before considering convergence

# Plotting parameters
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (15, 12)
DPI = 300

# Output formatting
DECIMAL_PLACES = 2
PERCENTAGE_FORMAT = '.1%'

# Data preprocessing
FILL_MISSING_WITH_ZERO = True
SCALE_FEATURES = True
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 3  # Standard deviations

# Advanced features
USE_FINANCIAL_FEATURES = True
USE_MOMENTUM_INDICATORS = True
USE_VOLATILITY_BANDS = True
USE_TEMPORAL_FEATURES = True
