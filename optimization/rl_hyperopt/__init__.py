"""
rl_hyperopt — RL-based DFS Hyperparameter Optimizer

Uses reinforcement learning to find optimal DraftKings lineup optimizer
configurations (max exposure, stacking, min unique, etc.) from backtest data.

Agents learn which of 32 hyperparameter configurations produce the best
lineups across different game dates, optimizing for GPP ceiling or Cash
consistency depending on contest mode.

Usage:
    from rl_hyperopt import find_optimal_config
    result = find_optimal_config(mode='gpp', episodes=500)
    print(result['best_config'])

CLI:
    python -m rl_hyperopt.run --mode gpp --compare
"""

__version__ = "1.0.0"

# Config
from .config import ACTION_SPACE, NUM_ACTIONS, REWARD_WEIGHTS, REWARD_METRICS

# Environment
from .environment import DFSHyperOptEnv

# Agents
from .agent import (
    BaseAgent,
    EpsilonGreedyQAgent,
    UCBAgent,
    ThompsonSamplingAgent,
    DQNAgent,
    create_agent,
)

# Training
from .train import (
    train_agent,
    evaluate_agent,
    compare_agents,
    find_optimal_config,
)

# Reward functions
from .reward import composite_reward, shaped_reward

# Meta-actions
from .meta_actions import MetaActionManager, MetaAction

__all__ = [
    # Config
    "ACTION_SPACE",
    "NUM_ACTIONS",
    "REWARD_WEIGHTS",
    "REWARD_METRICS",
    # Environment
    "DFSHyperOptEnv",
    # Agents
    "BaseAgent",
    "EpsilonGreedyQAgent",
    "UCBAgent",
    "ThompsonSamplingAgent",
    "DQNAgent",
    "create_agent",
    # Training
    "train_agent",
    "evaluate_agent",
    "compare_agents",
    "find_optimal_config",
    # Reward
    "composite_reward",
    "shaped_reward",
    # Meta-actions
    "MetaActionManager",
    "MetaAction",
]
