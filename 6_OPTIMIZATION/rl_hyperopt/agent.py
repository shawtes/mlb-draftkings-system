"""
RL agents for DFS hyperparameter optimization.

Agents select among 32 discrete hyperparameter configurations (actions)
given a 37-dimensional state vector from DFSHyperOptEnv:
  [last_reward_norm, best_reward_norm, steps_remaining_frac,
   running_avg_norm, running_std_norm] + one-hot(32) of last action.

Implements:
  - EpsilonGreedyQAgent: Tabular Q-learning with state discretization
  - UCBAgent: Upper Confidence Bound bandit (UCB1)
  - ThompsonSamplingAgent: Bayesian bandit with Beta priors
  - DQNAgent: Deep Q-Network (requires PyTorch)
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import math
import pickle
import random

import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        ...

    @abstractmethod
    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        """Choose an action given the current state.

        Args:
            state: 37-dimensional observation from the environment.
            available_actions: Optional subset of valid action indices.
                If None, all actions are available.

        Returns:
            Action index in [0, n_actions).
        """
        ...

    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Learn from a single transition.

        Args:
            state: State before the action.
            action: Action taken.
            reward: Reward received.
            next_state: State after the action.
            done: Whether the episode ended.
        """
        ...

    @abstractmethod
    def get_best_action(self) -> int:
        """Return the greedy/exploit action (no exploration)."""
        ...

    def reset(self) -> None:
        """Called at the start of each new episode. Override if needed."""
        pass

    def save(self, path: str) -> None:
        """Persist the agent to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseAgent":
        """Load an agent from disk."""
        with open(path, "rb") as f:
            agent = pickle.load(f)
        if not isinstance(agent, BaseAgent):
            raise TypeError(f"Loaded object is {type(agent)}, not a BaseAgent")
        return agent


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------

class EpsilonGreedyQAgent(BaseAgent):
    """Tabular Q-learning with epsilon-greedy exploration.

    The 5 continuous state features are discretized into bins so the
    Q-table can be indexed by a hashable tuple key.  The remaining
    32 one-hot features (last action) are collapsed to a single int.
    """

    def __init__(
        self,
        n_actions: int = 32,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        n_bins: int = 10,
    ) -> None:
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins

        # Bin edges for the 5 continuous features (indices 0-4).
        # Values are roughly in [0, 1] after normalization.
        self._bin_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # 9 edges -> 10 bins

        # Q-table stored as dict: discretized_state_tuple -> np.array(n_actions)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

        # Track action counts across all states for get_best_action fallback
        self._global_action_rewards: np.ndarray = np.zeros(n_actions)
        self._global_action_counts: np.ndarray = np.zeros(n_actions)

    @property
    def name(self) -> str:
        return "EpsilonGreedyQ"

    def _discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous state to a hashable tuple of bin indices."""
        continuous = state[:5]
        bins = tuple(int(np.digitize(v, self._bin_edges)) for v in continuous)
        # Collapse one-hot last-action (indices 5:37) to a single int
        one_hot = state[5:]
        last_action = int(np.argmax(one_hot)) if one_hot.sum() > 0 else 0
        return bins + (last_action,)

    def _get_q(self, key: Tuple[int, ...]) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        q = self._get_q(self._discretize(state))
        # Mask unavailable actions to -inf
        masked = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            masked[a] = q[a]
        return int(np.argmax(masked))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        key = self._discretize(state)
        q = self._get_q(key)

        if done:
            target = reward
        else:
            next_q = self._get_q(self._discretize(next_state))
            target = reward + self.gamma * np.max(next_q)

        q[action] += self.lr * (target - q[action])

        # Track global action performance for get_best_action
        self._global_action_counts[action] += 1
        self._global_action_rewards[action] += reward

    def get_best_action(self) -> int:
        # Aggregate across all visited states
        if len(self.q_table) == 0:
            return 0
        avg_q = np.zeros(self.n_actions)
        counts = np.zeros(self.n_actions)
        for q in self.q_table.values():
            for a in range(self.n_actions):
                if q[a] != 0.0:
                    avg_q[a] += q[a]
                    counts[a] += 1
        safe_counts = np.where(counts > 0, counts, 1)
        avg_q = avg_q / safe_counts
        return int(np.argmax(avg_q))

    def reset(self) -> None:
        # Decay epsilon at the start of each new episode
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Upper Confidence Bound (UCB1) bandit
# ---------------------------------------------------------------------------

class UCBAgent(BaseAgent):
    """UCB1 multi-armed bandit.

    Treats each of the 32 configs as an independent arm.  State is ignored
    since this is a pure bandit approach.
    """

    def __init__(self, n_actions: int = 32, c: float = 2.0) -> None:
        self.n_actions = n_actions
        self.c = c
        self.q_values = np.zeros(n_actions)   # mean reward per action
        self.counts = np.zeros(n_actions)      # pull count per action
        self.total_steps = 0

    @property
    def name(self) -> str:
        return "UCB1"

    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        # Pull each arm at least once
        for a in available_actions:
            if self.counts[a] == 0:
                return a

        ucb = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            bonus = self.c * math.sqrt(math.log(self.total_steps) / self.counts[a])
            ucb[a] = self.q_values[a] + bonus
        return int(np.argmax(ucb))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        self.total_steps += 1
        self.counts[action] += 1
        # Incremental mean update
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]

    def get_best_action(self) -> int:
        return int(np.argmax(self.q_values))

    def reset(self) -> None:
        pass  # UCB does not reset between episodes — it accumulates


# ---------------------------------------------------------------------------
# Thompson Sampling bandit
# ---------------------------------------------------------------------------

class ThompsonSamplingAgent(BaseAgent):
    """Thompson Sampling with Beta-Bernoulli model.

    Rewards are normalized to [0, 1] and used as soft success probability
    updates to per-action Beta(alpha, beta) distributions.
    """

    def __init__(self, n_actions: int = 32, reward_scale: float = 1.0) -> None:
        self.n_actions = n_actions
        self.reward_scale = reward_scale
        # Beta priors: uniform (1, 1)
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)
        # Track raw rewards for normalization
        self._reward_min = float("inf")
        self._reward_max = float("-inf")

    @property
    def name(self) -> str:
        return "ThompsonSampling"

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward to [0, 1] using observed min/max."""
        # Update range
        self._reward_min = min(self._reward_min, reward)
        self._reward_max = max(self._reward_max, reward)

        rng = self._reward_max - self._reward_min
        if rng < 1e-8:
            return 0.5  # no spread yet
        return np.clip((reward - self._reward_min) / rng, 0.0, 1.0)

    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        samples = np.array([
            np.random.beta(self.alpha[a], self.beta[a]) for a in available_actions
        ])
        return available_actions[int(np.argmax(samples))]

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        p = self._normalize_reward(reward * self.reward_scale)
        self.alpha[action] += p
        self.beta[action] += (1.0 - p)

    def get_best_action(self) -> int:
        # Expected value of Beta = alpha / (alpha + beta)
        expected = self.alpha / (self.alpha + self.beta)
        return int(np.argmax(expected))

    def reset(self) -> None:
        pass  # Thompson Sampling accumulates across episodes


# ---------------------------------------------------------------------------
# Deep Q-Network (optional — requires PyTorch)
# ---------------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """Deep Q-Network with experience replay and target network.

    Requires PyTorch. If torch is not installed, __init__ raises ImportError.
    """

    def __init__(
        self,
        n_actions: int = 32,
        state_dim: int = 37,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        tau: float = 0.005,
        batch_size: int = 32,
        replay_size: int = 10000,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError(
                "DQNAgent requires PyTorch. Install it with `pip install torch` "
                "or use a simpler agent: create_agent('q-learning') or create_agent('ucb')."
            )

        self._torch = torch
        self._nn = nn

        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size

        # Networks
        self.policy_net = self._build_network(nn)
        self.target_net = self._build_network(nn)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=replay_size)

        # Track best action across training
        self._action_rewards = np.zeros(n_actions)
        self._action_counts = np.zeros(n_actions)

    def _build_network(self, nn: Any) -> Any:
        """Create the Q-network: 37 -> 128 -> 64 -> 32."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        )

    @property
    def name(self) -> str:
        return "DQN"

    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        if available_actions is None:
            available_actions = list(range(self.n_actions))

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        torch = self._torch
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(s).squeeze(0).numpy()

        masked = np.full(self.n_actions, -np.inf)
        for a in available_actions:
            masked[a] = q_values[a]
        return int(np.argmax(masked))

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.append((state, action, reward, next_state, done))

        # Track for get_best_action
        self._action_counts[action] += 1
        self._action_rewards[action] += reward

        self._train_step()

    def _train_step(self) -> None:
        """Sample a minibatch and perform one gradient step."""
        if len(self.replay_buffer) < self.batch_size:
            return

        torch = self._torch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values for chosen actions
        current_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update target network
        for p_target, p_policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(self.tau * p_policy.data + (1.0 - self.tau) * p_target.data)

    def get_best_action(self) -> int:
        torch = self._torch
        # Use a zero state (neutral) to get the policy network's best action
        with torch.no_grad():
            s = torch.zeros(1, self.state_dim)
            q_values = self.policy_net(s).squeeze(0).numpy()
        return int(np.argmax(q_values))

    def reset(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: Dict[str, type] = {
    "q-learning": EpsilonGreedyQAgent,
    "ucb": UCBAgent,
    "thompson": ThompsonSamplingAgent,
    "dqn": DQNAgent,
}


def create_agent(agent_type: str, n_actions: int = 32, **kwargs: Any) -> BaseAgent:
    """Create an RL agent by name.

    Args:
        agent_type: One of 'q-learning', 'ucb', 'thompson', 'dqn'.
        n_actions: Number of discrete actions (hyperparameter configs).
        **kwargs: Extra keyword arguments forwarded to the agent constructor.

    Returns:
        An initialized BaseAgent instance.

    Raises:
        ValueError: If agent_type is not recognized.
        ImportError: If 'dqn' is requested but PyTorch is not installed.
    """
    key = agent_type.lower().strip()
    if key not in _AGENT_REGISTRY:
        valid = ", ".join(sorted(_AGENT_REGISTRY.keys()))
        raise ValueError(f"Unknown agent type '{agent_type}'. Valid types: {valid}")

    cls = _AGENT_REGISTRY[key]
    return cls(n_actions=n_actions, **kwargs)
