"""
RL Hyperparameter Optimizer — Reward Functions

Provides reward computation for the DFS hyperparameter optimization environment.
Supports z-score normalization, composite weighted rewards, regret-based rewards,
and shaped rewards with bonuses/penalties.
"""

import numpy as np

from .config import REWARD_WEIGHTS, REWARD_METRICS, METRIC_RANGES


def normalize_metrics(df):
    """Z-score normalize the reward-relevant metric columns in a DataFrame.

    Operates in-place, adding columns named ``<metric>_zscore`` for each metric
    found in METRIC_RANGES.

    Args:
        df: pandas DataFrame with columns matching METRIC_RANGES keys.

    Returns:
        The same DataFrame with added ``*_zscore`` columns.
    """
    for metric in METRIC_RANGES:
        if metric not in df.columns:
            continue
        col = df[metric].astype(float)
        mean = col.mean()
        std = col.std()
        if std == 0 or np.isnan(std):
            df[f"{metric}_zscore"] = 0.0
        else:
            df[f"{metric}_zscore"] = (col - mean) / std
    return df


def _normalize_single(value, metric):
    """Min-max normalize a single metric value to [0, 1] using known ranges."""
    r = METRIC_RANGES.get(metric)
    if r is None:
        return float(value)
    span = r["max"] - r["min"]
    if span == 0:
        return 0.0
    return max(0.0, min(1.0, (float(value) - r["min"]) / span))


def composite_reward(metrics_row, mode="gpp"):
    """Compute a weighted composite reward from a single metrics row.

    Args:
        metrics_row: dict-like with metric keys (e.g. a CSV DictReader row
            or a pandas Series). Values should be numeric strings or floats.
        mode: ``'gpp'`` or ``'cash'`` — selects the weight profile.

    Returns:
        Float reward in roughly [0, 1] range (sum of weighted normalized metrics).
    """
    weights = REWARD_WEIGHTS.get(mode, REWARD_WEIGHTS["gpp"])
    metrics = REWARD_METRICS.get(mode, REWARD_METRICS["gpp"])

    reward = 0.0
    for metric in metrics:
        raw = float(metrics_row.get(metric, 0.0))
        # For consistency, lower is better (lower std dev = more consistent).
        # Invert so that low consistency values produce high normalized scores.
        if metric == "consistency":
            r = METRIC_RANGES["consistency"]
            normed = 1.0 - _normalize_single(raw, metric)
        else:
            normed = _normalize_single(raw, metric)
        reward += weights[metric] * normed

    return reward


def regret_reward(metrics_row, best_so_far, mode="gpp"):
    """Negative regret: how much worse this action was vs. the best seen.

    Args:
        metrics_row: dict-like with metric values.
        best_so_far: float — highest composite reward achieved so far in episode.
        mode: ``'gpp'`` or ``'cash'``.

    Returns:
        Float <= 0. Zero means this action matched or beat the best.
    """
    current = composite_reward(metrics_row, mode=mode)
    return min(0.0, current - best_so_far)


def shaped_reward(metrics_row, mode="gpp", best_so_far=0.0):
    """Shaped reward combining composite score with bonuses and penalties.

    - Base: composite_reward (weighted normalized metrics)
    - Bonus: +0.2 if this action sets a new personal best for the episode
    - Penalty: -0.5 if the row represents an error (all metrics near zero)

    Args:
        metrics_row: dict-like with metric values.
        mode: ``'gpp'`` or ``'cash'``.
        best_so_far: float — best composite reward seen so far in episode.

    Returns:
        Tuple of (shaped_reward: float, new_best: float).
    """
    # Detect error rows: if key metrics are all zero or missing
    best_actual = float(metrics_row.get("best_actual", 0.0))
    avg_actual = float(metrics_row.get("avg_actual", 0.0))
    if best_actual == 0.0 and avg_actual == 0.0:
        return -0.5, best_so_far

    base = composite_reward(metrics_row, mode=mode)
    bonus = 0.0
    new_best = best_so_far

    if base > best_so_far:
        bonus = 0.2
        new_best = base

    return base + bonus, new_best
