"""
RL Hyperparameter Optimizer — Meta-Actions

Higher-level search strategies that augment the base RL agent's action selection.
Each meta-action returns a list of action indices to evaluate, with an associated
budget cost that the training loop can use for scheduling.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ACTION_SPACE, NUM_ACTIONS


class MetaAction(Enum):
    """Available meta-action strategies."""
    DIRECT_SELECT = auto()
    GRID_SEARCH = auto()
    RANDOM_SEARCH = auto()
    LOCAL_SEARCH = auto()
    EXPLOIT_TOP_K = auto()


# Budget cost per meta-action (number of configs evaluated)
META_ACTION_COSTS: Dict[MetaAction, int] = {
    MetaAction.DIRECT_SELECT: 1,
    MetaAction.GRID_SEARCH: 4,
    MetaAction.RANDOM_SEARCH: 5,
    MetaAction.LOCAL_SEARCH: 4,
    MetaAction.EXPLOIT_TOP_K: 1,
}

# Parameter names and their possible values for grid/local search
_PARAM_VALUES = {
    "max_exposure": [30, 50, 70, 100],
    "stack_enabled": [False, True],
    "stack_sizes": [[], [2], [3], [2, 3]],
    "min_unique": [2, 3],
}

# Searchable params (stack_enabled is implicitly tied to stack_sizes)
SEARCHABLE_PARAMS = ["max_exposure", "stack_sizes", "min_unique"]


def _config_matches(action: dict, **overrides) -> bool:
    """Check if an action config matches a set of param overrides."""
    for key, val in overrides.items():
        if key == "stack_sizes":
            if action["stack_sizes"] != val:
                return False
            expected_enabled = len(val) > 0
            if action["stack_enabled"] != expected_enabled:
                return False
        elif action[key] != val:
            return False
    return True


def _find_action_index(**overrides) -> Optional[int]:
    """Find the action index matching exact param overrides, or None."""
    for idx, action in enumerate(ACTION_SPACE):
        if _config_matches(action, **overrides):
            return idx
    return None


class MetaActionManager:
    """Manages meta-action strategies for hyperparameter exploration.

    Tracks history of (action_index, reward) pairs and provides structured
    search methods that return lists of action indices to evaluate.
    """

    def __init__(self, action_space: Optional[list] = None,
                 history: Optional[List[Tuple[int, float]]] = None):
        """Initialize the meta-action manager.

        Args:
            action_space: List of config dicts. Defaults to ACTION_SPACE.
            history: Optional pre-existing (action_index, reward) pairs.
        """
        self.action_space = action_space or ACTION_SPACE
        self.n_actions = len(self.action_space)
        self.history: List[Tuple[int, float]] = list(history) if history else []
        # Running best-reward tracker per action
        self._best_rewards: Dict[int, float] = {}
        if self.history:
            for action_idx, reward in self.history:
                if action_idx not in self._best_rewards or reward > self._best_rewards[action_idx]:
                    self._best_rewards[action_idx] = reward

    def record(self, action_idx: int, reward: float) -> None:
        """Record an observation."""
        self.history.append((action_idx, reward))
        if action_idx not in self._best_rewards or reward > self._best_rewards[action_idx]:
            self._best_rewards[action_idx] = reward

    def grid_search(self, param_name: str, current_config_idx: int) -> List[int]:
        """Try all values of one parameter while holding others fixed.

        Args:
            param_name: One of 'max_exposure', 'stack_sizes', 'min_unique'.
            current_config_idx: Index of the current config to vary from.

        Returns:
            List of action indices covering all values of param_name.
        """
        if param_name not in _PARAM_VALUES:
            return [current_config_idx]

        current = self.action_space[current_config_idx]
        results = []
        for val in _PARAM_VALUES[param_name]:
            overrides = {
                "max_exposure": current["max_exposure"],
                "stack_sizes": list(current["stack_sizes"]),
                "min_unique": current["min_unique"],
            }
            if param_name == "stack_sizes":
                overrides["stack_sizes"] = val
            else:
                overrides[param_name] = val
            idx = _find_action_index(**overrides)
            if idx is not None and idx not in results:
                results.append(idx)
        return results

    def random_search(self, n: int = 5) -> List[int]:
        """Sample n random configs from the action space.

        Args:
            n: Number of random configs to sample.

        Returns:
            List of n unique random action indices.
        """
        n = min(n, self.n_actions)
        return list(np.random.choice(self.n_actions, size=n, replace=False))

    def local_search(self, current_best_idx: int, n: int = 4) -> List[int]:
        """Find configs that differ from the current best by exactly one parameter.

        Args:
            current_best_idx: Index of the current best config.
            n: Maximum number of neighbors to return.

        Returns:
            List of action indices that are 1-param neighbors of current_best_idx.
        """
        current = self.action_space[current_best_idx]
        neighbors = []

        for param_name in SEARCHABLE_PARAMS:
            for val in _PARAM_VALUES[param_name]:
                # Skip the current value
                if param_name == "stack_sizes":
                    if val == current["stack_sizes"]:
                        continue
                elif val == current[param_name]:
                    continue

                overrides = {
                    "max_exposure": current["max_exposure"],
                    "stack_sizes": list(current["stack_sizes"]),
                    "min_unique": current["min_unique"],
                }
                if param_name == "stack_sizes":
                    overrides["stack_sizes"] = val
                else:
                    overrides[param_name] = val

                idx = _find_action_index(**overrides)
                if idx is not None and idx != current_best_idx and idx not in neighbors:
                    neighbors.append(idx)

        # Return at most n neighbors, prioritizing unexplored ones
        unexplored = [i for i in neighbors if i not in self._best_rewards]
        explored = [i for i in neighbors if i in self._best_rewards]
        # Sort explored by descending reward
        explored.sort(key=lambda i: self._best_rewards.get(i, 0.0), reverse=True)
        ordered = unexplored + explored
        return ordered[:n]

    def exploit_top_k(self, k: int = 5) -> List[int]:
        """Return indices of the top-k configs by observed reward.

        Args:
            k: Number of top configs to return.

        Returns:
            List of action indices with highest observed rewards.
        """
        if not self._best_rewards:
            return list(range(min(k, self.n_actions)))
        sorted_actions = sorted(self._best_rewards.items(),
                                key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_actions[:k]]

    def execute(self, meta_action: MetaAction, **kwargs) -> List[int]:
        """Execute a meta-action and return action indices to evaluate.

        Args:
            meta_action: The meta-action strategy to execute.
            **kwargs: Strategy-specific arguments:
                - DIRECT_SELECT: action_idx (int)
                - GRID_SEARCH: param_name (str), current_config_idx (int)
                - RANDOM_SEARCH: n (int, default=5)
                - LOCAL_SEARCH: current_best_idx (int), n (int, default=4)
                - EXPLOIT_TOP_K: k (int, default=5)

        Returns:
            List of action indices to try.
        """
        if meta_action == MetaAction.DIRECT_SELECT:
            action_idx = kwargs.get("action_idx", 0)
            return [action_idx]

        elif meta_action == MetaAction.GRID_SEARCH:
            param_name = kwargs.get("param_name", "max_exposure")
            current_config_idx = kwargs.get("current_config_idx", 0)
            return self.grid_search(param_name, current_config_idx)

        elif meta_action == MetaAction.RANDOM_SEARCH:
            n = kwargs.get("n", 5)
            return self.random_search(n)

        elif meta_action == MetaAction.LOCAL_SEARCH:
            current_best_idx = kwargs.get("current_best_idx", 0)
            n = kwargs.get("n", 4)
            return self.local_search(current_best_idx, n)

        elif meta_action == MetaAction.EXPLOIT_TOP_K:
            k = kwargs.get("k", 5)
            return self.exploit_top_k(k)

        else:
            return [0]

    @staticmethod
    def get_cost(meta_action: MetaAction) -> int:
        """Return the budget cost of a meta-action."""
        return META_ACTION_COSTS.get(meta_action, 1)
