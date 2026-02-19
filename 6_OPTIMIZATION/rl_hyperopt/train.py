"""
RL Hyperparameter Optimizer — Training Loop

Provides train_agent, evaluate_agent, compare_agents, and a convenience
find_optimal_config function that orchestrates the full pipeline.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .config import ACTION_SPACE, EPISODES, NUM_ACTIONS


def _unpack_reset(result):
    """Handle both gymnasium (obs, info) and classic gym (obs) reset returns."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _unpack_step(result):
    """Handle both gymnasium (obs, r, term, trunc, info) and gym (obs, r, done, info)."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info
    return result  # (obs, reward, done, info)


def train_agent(
    env: Any,
    agent: Any,
    episodes: int = EPISODES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train an RL agent on the DFS hyperparameter optimization environment.

    Args:
        env: A DFSHyperOptEnv instance with reset()/step() interface.
        agent: A BaseAgent instance with select_action/update/get_best_action.
        episodes: Number of training episodes to run.
        verbose: If True, print progress every 50 episodes.

    Returns:
        Dict with keys: rewards, running_avg, best_action, best_reward,
        best_config, convergence_episode.
    """
    episode_rewards: List[float] = []
    running_avgs: List[float] = []
    best_reward = -np.inf
    best_action = 0
    convergence_episode = 0
    window = 50
    patience = 100
    best_running_avg = -np.inf
    no_improve_count = 0

    for ep in range(episodes):
        state = _unpack_reset(env.reset())
        episode_total = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = _unpack_step(env.step(action))
            agent.update(state, action, reward, next_state, done)
            episode_total += reward
            state = next_state
            steps += 1

            # Track best single-step reward and its action
            if reward > best_reward:
                best_reward = reward
                best_action = action
                convergence_episode = ep

        episode_rewards.append(episode_total)

        # Running average
        if len(episode_rewards) >= window:
            avg = float(np.mean(episode_rewards[-window:]))
        else:
            avg = float(np.mean(episode_rewards))
        running_avgs.append(avg)

        # Agent reset handles per-episode bookkeeping (e.g. epsilon decay)
        agent.reset()

        # Early stopping check
        if avg > best_running_avg:
            best_running_avg = avg
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose and (ep + 1) % 50 == 0:
            agent_name = getattr(agent, "name", agent.__class__.__name__)
            print(f"  [{agent_name}] Episode {ep + 1}/{episodes}  "
                  f"avg_reward={avg:.4f}  best={best_reward:.4f}  "
                  f"eps={getattr(agent, 'epsilon', 'N/A')}")

        if no_improve_count >= patience and ep >= window + patience:
            if verbose:
                print(f"  Early stopping at episode {ep + 1} "
                      f"(no improvement for {patience} episodes)")
            break

    # Get the agent's final best action via its own policy
    final_best = agent.get_best_action()

    return {
        "rewards": episode_rewards,
        "running_avg": running_avgs,
        "best_action": final_best,
        "best_reward": best_reward,
        "best_config": ACTION_SPACE[final_best],
        "convergence_episode": convergence_episode,
    }


def evaluate_agent(
    env: Any,
    agent: Any,
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """Evaluate a trained agent using a greedy policy (no exploration).

    Args:
        env: A DFSHyperOptEnv instance.
        agent: A trained BaseAgent instance.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with mean_reward, std_reward, best_config.
    """
    # Save and disable exploration
    orig_epsilon = getattr(agent, "epsilon", None)
    if orig_epsilon is not None:
        agent.epsilon = 0.0

    episode_rewards: List[float] = []
    for _ in range(n_episodes):
        state = _unpack_reset(env.reset())
        total = 0.0
        done = False
        while not done:
            action = agent.get_best_action()
            next_state, reward, done, info = _unpack_step(env.step(action))
            total += reward
            state = next_state
        episode_rewards.append(total)

    # Restore exploration
    if orig_epsilon is not None:
        agent.epsilon = orig_epsilon

    best_action = agent.get_best_action()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "best_config": ACTION_SPACE[best_action],
    }


def compare_agents(
    env: Any,
    agent_dict: Dict[str, Any],
    episodes: int = EPISODES,
) -> Any:
    """Train and compare multiple agents on the same environment.

    Args:
        env: A DFSHyperOptEnv instance.
        agent_dict: Mapping of agent_name -> agent instance.
        episodes: Training episodes per agent.

    Returns:
        pandas DataFrame with comparison metrics, or dict if pandas unavailable.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for name, agent in agent_dict.items():
        print(f"\nTraining {name}...")
        agent.reset()
        train_result = train_agent(env, agent, episodes=episodes, verbose=True)
        eval_result = evaluate_agent(env, agent, n_episodes=50)

        results[name] = {
            "agent_name": name,
            "mean_reward": eval_result["mean_reward"],
            "std_reward": eval_result["std_reward"],
            "best_config": train_result["best_config"],
            "best_reward": train_result["best_reward"],
            "convergence_episode": train_result["convergence_episode"],
            "train_rewards": train_result["rewards"],
            "running_avg": train_result["running_avg"],
        }

    # Build comparison table
    rows = []
    for name, r in results.items():
        rows.append({
            "agent_name": name,
            "mean_reward": round(r["mean_reward"], 4),
            "std_reward": round(r["std_reward"], 4),
            "best_reward": round(r["best_reward"], 4),
            "convergence_episode": r["convergence_episode"],
            "best_config": json.dumps(r["best_config"]),
        })

    if pd is not None:
        df = pd.DataFrame(rows)
        df = df.sort_values("mean_reward", ascending=False).reset_index(drop=True)
        return df, results
    else:
        return rows, results


def plot_training_curves(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: str = "training_curves.png",
) -> None:
    """Plot reward curves for all agents.

    Args:
        results_dict: Mapping of agent_name -> result dict (must have 'running_avg').
        output_path: File path to save the plot.
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results_dict.items():
        avgs = data.get("running_avg", [])
        if avgs:
            ax.plot(range(len(avgs)), avgs, label=name, linewidth=1.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Running Avg Reward (window=50)")
    ax.set_title("RL Hyperparameter Optimizer — Training Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def find_optimal_config(
    data_path: Optional[str] = None,
    mode: str = "gpp",
    episodes: int = EPISODES,
) -> Dict[str, Any]:
    """Convenience function: load data, train agents, return best config.

    Args:
        data_path: Path to backtest_results.csv. Defaults to config.DATA_PATH.
        mode: Contest mode — 'gpp' or 'cash'.
        episodes: Number of training episodes.

    Returns:
        Dict with best_config, best_agent, comparison table, and all results.
    """
    from .config import DATA_PATH
    from .environment import DFSHyperOptEnv
    from .agent import EpsilonGreedyQAgent, UCBAgent, ThompsonSamplingAgent, create_agent

    path = data_path or DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Backtest data not found: {path}")

    env = DFSHyperOptEnv(csv_path=path, mode=mode)

    agents = {
        "Q-Learning": create_agent("q-learning"),
        "UCB": create_agent("ucb"),
        "Thompson": create_agent("thompson"),
    }

    comparison, all_results = compare_agents(env, agents, episodes=episodes)

    # Find the best agent by mean eval reward
    if pd is not None and hasattr(comparison, "iloc"):
        best_row = comparison.iloc[0]
        best_agent_name = best_row["agent_name"]
    else:
        best_row = max(comparison, key=lambda r: r["mean_reward"])
        best_agent_name = best_row["agent_name"]

    best_config = all_results[best_agent_name]["best_config"]

    print(f"\nBest agent: {best_agent_name}")
    print(f"Best config: {json.dumps(best_config, indent=2)}")

    return {
        "best_config": best_config,
        "best_agent": best_agent_name,
        "comparison": comparison,
        "all_results": all_results,
    }
