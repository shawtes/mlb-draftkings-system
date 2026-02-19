"""
RL Hyperparameter Optimizer — CLI Runner

Usage:
    python -m rl_hyperopt.run --mode gpp --agent all --episodes 500 --compare
    python -m rl_hyperopt.run --mode cash --agent ucb --episodes 300
    python -m rl_hyperopt.run --data-path /path/to/backtest_results.csv
"""

import argparse
import json
import os
import sys

# Ensure package is importable when run directly
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_pkg_dir)
for _p in [_parent_dir, _pkg_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from rl_hyperopt.config import ACTION_SPACE, DATA_PATH, EPISODES, NUM_ACTIONS
from rl_hyperopt.train import (
    compare_agents,
    evaluate_agent,
    find_optimal_config,
    plot_training_curves,
    train_agent,
)
from rl_hyperopt.environment import DFSHyperOptEnv
from rl_hyperopt.agent import create_agent

try:
    import pandas as pd
except ImportError:
    pd = None


AGENT_CHOICES = ["q-learning", "ucb", "thompson", "dqn", "all"]


def parse_args(argv: list = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RL Hyperparameter Optimizer for DFS lineup configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m rl_hyperopt.run --mode gpp --compare
  python -m rl_hyperopt.run --agent ucb --episodes 300
  python -m rl_hyperopt.run --data-path ./data/backtest_results.csv --mode cash
        """,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=f"Path to backtest_results.csv (default: auto-detect at {DATA_PATH})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gpp", "cash"],
        default="gpp",
        help="Contest mode: gpp or cash (default: gpp)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=AGENT_CHOICES,
        default="all",
        help="Agent type to train (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES,
        help=f"Number of training episodes (default: {EPISODES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./rl_output",
        help="Directory for output files (default: ./rl_output)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all agents and produce comparison table",
    )
    return parser.parse_args(argv)


def _create_agents(agent_name: str) -> dict:
    """Create agent instances by name.

    Args:
        agent_name: One of 'q-learning', 'ucb', 'thompson', 'dqn', 'all'.

    Returns:
        Dict mapping agent display names to agent instances.
    """
    if agent_name == "all":
        agents = {}
        for name in ["q-learning", "ucb", "thompson"]:
            agents[name.replace("-", "_").title().replace("_", "-")] = create_agent(name)
        # DQN is optional (requires more deps)
        try:
            agents["DQN"] = create_agent("dqn")
        except Exception:
            print("DQN agent unavailable (may require additional dependencies), skipping.")
        return agents
    else:
        display_name = agent_name.replace("-", "_").title().replace("_", "-")
        return {display_name: create_agent(agent_name)}


def main(argv: list = None) -> None:
    """Main CLI entry point."""
    args = parse_args(argv)

    # Resolve data path
    data_path = args.data_path or DATA_PATH
    if not os.path.exists(data_path):
        print(f"ERROR: Backtest data not found at {data_path}")
        print("Run backtest_optimizer.py first to generate backtest_results.csv")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RL Hyperparameter Optimizer for DFS")
    print("=" * 60)
    print(f"  Mode:       {args.mode.upper()}")
    print(f"  Agent:      {args.agent}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Data:       {data_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Actions:    {NUM_ACTIONS} configurations")
    print("=" * 60)

    # Create environment
    env = DFSHyperOptEnv(csv_path=data_path, mode=args.mode)

    if args.compare or args.agent == "all":
        # Run full comparison
        agents = _create_agents("all")
        comparison, all_results = compare_agents(env, agents, episodes=args.episodes)

        # Print comparison table
        print("\n" + "=" * 60)
        print("AGENT COMPARISON RESULTS")
        print("=" * 60)
        if pd is not None and hasattr(comparison, "to_string"):
            print(comparison.to_string(index=False))
        else:
            for row in comparison:
                print(f"  {row['agent_name']:15s}  mean={row['mean_reward']:.4f}  "
                      f"std={row['std_reward']:.4f}  best={row['best_reward']:.4f}  "
                      f"converged@ep{row['convergence_episode']}")

        # Plot training curves
        curves_path = os.path.join(args.output_dir, "training_curves.png")
        plot_training_curves(all_results, output_path=curves_path)

        # Save comparison table
        table_path = os.path.join(args.output_dir, "comparison_table.csv")
        if pd is not None and hasattr(comparison, "to_csv"):
            comparison.to_csv(table_path, index=False)
        else:
            import csv
            with open(table_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
                writer.writeheader()
                writer.writerows(comparison)
        print(f"\nComparison table saved to {table_path}")

        # Find and save best config
        if pd is not None and hasattr(comparison, "iloc"):
            best_name = comparison.iloc[0]["agent_name"]
        else:
            best_name = max(comparison, key=lambda r: r["mean_reward"])["agent_name"]

        best_config = all_results[best_name]["best_config"]

    else:
        # Train single agent
        agents = _create_agents(args.agent)
        agent_name, agent = next(iter(agents.items()))

        print(f"\nTraining {agent_name}...")
        result = train_agent(env, agent, episodes=args.episodes, verbose=True)
        eval_result = evaluate_agent(env, agent, n_episodes=50)

        print(f"\n{'=' * 60}")
        print(f"RESULTS — {agent_name}")
        print(f"{'=' * 60}")
        print(f"  Eval mean reward:   {eval_result['mean_reward']:.4f}")
        print(f"  Eval std reward:    {eval_result['std_reward']:.4f}")
        print(f"  Best reward seen:   {result['best_reward']:.4f}")
        print(f"  Converged at ep:    {result['convergence_episode']}")
        print(f"  Best config:        {json.dumps(result['best_config'])}")

        # Plot single agent curve
        curves_path = os.path.join(args.output_dir, "training_curves.png")
        plot_training_curves(
            {agent_name: {"running_avg": result["running_avg"]}},
            output_path=curves_path,
        )

        best_config = result["best_config"]

    # Save best config
    config_path = os.path.join(args.output_dir, "best_config.json")
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved to {config_path}")

    print(f"\nRecommended {args.mode.upper()} configuration:")
    print(f"  Max Exposure:  {best_config['max_exposure']}%")
    print(f"  Stacking:      {'Enabled' if best_config['stack_enabled'] else 'Disabled'}"
          f"{' ' + str(best_config['stack_sizes']) if best_config['stack_enabled'] else ''}")
    print(f"  Min Unique:    {best_config['min_unique']}")
    print(f"  Min Salary:    ${best_config['min_salary']:,}")


if __name__ == "__main__":
    main()
