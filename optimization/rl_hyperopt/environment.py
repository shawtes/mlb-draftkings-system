"""
RL Hyperparameter Optimizer — Gym-Compatible Environment

DFSHyperOptEnv walks through historical NBA game dates, letting the agent
choose a hyperparameter configuration (action) for each date. The reward
is derived from backtest performance metrics for that (date, config) pair.

State vector (37 dims):
    [0] last_reward_normalized
    [1] best_reward_so_far_normalized
    [2] steps_remaining / max_steps
    [3] running_avg_reward_normalized
    [4] running_std_reward_normalized
    [5..36] one-hot encoding of last action (32 dims)

Episode = walk through all 15 dates, selecting one config per date.
"""

import csv
import random
from collections import defaultdict

import numpy as np

from .config import (
    ACTION_LABELS,
    DATA_PATH,
    NUM_ACTIONS,
    STATE_DIM,
)
from .reward import shaped_reward

# ---------------------------------------------------------------------------
# Try gymnasium first, then gym, then define minimal standalone types
# ---------------------------------------------------------------------------
_GYM_BACKEND = None
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_BACKEND = "gymnasium"
except ImportError:
    try:
        import gym
        from gym import spaces
        _GYM_BACKEND = "gym"
    except ImportError:
        gym = None
        spaces = None
        _GYM_BACKEND = "standalone"


def _load_lookup_table(csv_path):
    """Load backtest results into a (date, config_label) -> row dict.

    Returns:
        lookup: dict mapping (date_str, config_label) to dict of metrics.
        dates: sorted list of unique date strings.
    """
    lookup = {}
    dates_set = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"]
            label = row["config"]
            dates_set.add(date)
            # Convert numeric fields to float, keep strings for non-numeric
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            lookup[(date, label)] = parsed
    dates = sorted(dates_set)
    return lookup, dates


class DFSHyperOptEnv:
    """Gym-compatible RL environment for DFS hyperparameter optimization.

    Each episode walks through the 15 historical game dates. At each step
    the agent selects one of 32 hyperparameter configurations. The environment
    returns the shaped reward from the backtest lookup table.
    """

    metadata = {"render_modes": []}

    def __init__(self, mode="gpp", csv_path=None):
        """
        Args:
            mode: 'gpp' or 'cash' — determines reward weighting.
            csv_path: path to backtest_results.csv. Defaults to DATA_PATH.
        """
        self.mode = mode
        csv_path = csv_path or DATA_PATH
        self.lookup, self.dates = _load_lookup_table(csv_path)
        self.max_steps = len(self.dates)  # 15

        # Spaces
        if spaces is not None:
            self.action_space = spaces.Discrete(NUM_ACTIONS)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32
            )
        else:
            self.action_space = _DiscreteSpace(NUM_ACTIONS)
            self.observation_space = _BoxSpace(STATE_DIM)

        # Episode state — initialized in reset()
        self._date_order = list(self.dates)
        self._step_idx = 0
        self._last_action = -1
        self._last_reward = 0.0
        self._best_reward = 0.0
        self._reward_history = []
        self._done = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset episode. Shuffles date order for exploration diversity."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._date_order = list(self.dates)
        random.shuffle(self._date_order)
        self._step_idx = 0
        self._last_action = -1
        self._last_reward = 0.0
        self._best_reward = 0.0
        self._reward_history = []
        self._done = False

        obs = self._get_obs()
        info = {"date": self._date_order[0] if self._date_order else ""}

        if _GYM_BACKEND == "gymnasium":
            return obs, info
        return obs  # classic gym returns obs only

    def step(self, action):
        """Execute one step: apply config ``action`` to the current date.

        Args:
            action: int in [0, NUM_ACTIONS).

        Returns:
            Gymnasium-style: (obs, reward, terminated, truncated, info)
            Classic gym:     (obs, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        action = int(action)
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be in [0, {NUM_ACTIONS}).")

        current_date = self._date_order[self._step_idx]
        config_label = ACTION_LABELS[action]

        # Look up metrics for (date, config)
        metrics_row = self.lookup.get((current_date, config_label))

        if metrics_row is None:
            # Missing row — treat as error with penalty
            reward = -0.5
            new_best = self._best_reward
        else:
            reward, new_best = shaped_reward(
                metrics_row, mode=self.mode, best_so_far=self._best_reward
            )

        # Update episode state
        self._last_action = action
        self._last_reward = reward
        self._best_reward = new_best
        self._reward_history.append(reward)
        self._step_idx += 1

        terminated = self._step_idx >= self.max_steps
        self._done = terminated

        obs = self._get_obs()
        info = {
            "date": current_date,
            "config": config_label,
            "reward_raw": reward,
            "best_reward": self._best_reward,
            "step": self._step_idx,
        }

        if _GYM_BACKEND == "gymnasium":
            return obs, reward, terminated, False, info
        return obs, reward, terminated, info

    def render(self):
        """No-op render for compatibility."""
        pass

    def close(self):
        """No-op close for compatibility."""
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build the 37-dimensional state vector."""
        steps_remaining = (self.max_steps - self._step_idx) / self.max_steps

        if len(self._reward_history) > 0:
            running_avg = np.mean(self._reward_history)
            running_std = np.std(self._reward_history) if len(self._reward_history) > 1 else 0.0
        else:
            running_avg = 0.0
            running_std = 0.0

        # Continuous features [5]
        continuous = np.array([
            self._last_reward,
            self._best_reward,
            steps_remaining,
            running_avg,
            running_std,
        ], dtype=np.float32)

        # One-hot of last action [32]
        one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if self._last_action >= 0:
            one_hot[self._last_action] = 1.0

        return np.concatenate([continuous, one_hot])


# ---------------------------------------------------------------------------
# Standalone space stubs (used only when neither gymnasium nor gym installed)
# ---------------------------------------------------------------------------

class _DiscreteSpace:
    """Minimal Discrete space replacement."""
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)

    def contains(self, x):
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


class _BoxSpace:
    """Minimal Box space replacement."""
    def __init__(self, dim):
        self.shape = (dim,)
        self.low = -np.inf
        self.high = np.inf
        self.dtype = np.float32

    def sample(self):
        return np.random.randn(self.shape[0]).astype(np.float32)

    def contains(self, x):
        return hasattr(x, "shape") and x.shape == self.shape


# ---------------------------------------------------------------------------
# Convenience: register with gymnasium if available
# ---------------------------------------------------------------------------
if _GYM_BACKEND == "gymnasium":
    # Wrap as a proper gymnasium.Env subclass
    _BaseEnv = gym.Env
elif _GYM_BACKEND == "gym":
    _BaseEnv = gym.Env
else:
    _BaseEnv = object

# Dynamically make DFSHyperOptEnv inherit from the correct base
DFSHyperOptEnv.__bases__ = (_BaseEnv,) if _BaseEnv is not object else (object,)
