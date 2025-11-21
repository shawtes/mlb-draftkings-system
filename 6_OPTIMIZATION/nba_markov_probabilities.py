import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors='coerce', utc=False, infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(series, errors='coerce')


def discretize_states(points: np.ndarray, method: str = "quantile", thresholds: Tuple[float, float] = (10.0, 20.0)) -> List[int]:
    if points is None or len(points) == 0:
        return []
    if method == "quantile":
        q1 = float(np.nanpercentile(points, 33))
        q2 = float(np.nanpercentile(points, 66))
        # Ensure unique and ordered bin edges
        if not np.isfinite(q1):
            q1 = 0.0
        if not np.isfinite(q2):
            q2 = q1
        if q2 <= q1:
            # Nudge q2 slightly above q1 to avoid duplicate edges
            eps = max(1e-6, abs(q1) * 1e-6 + 1e-6)
            q2 = q1 + eps
        bins = [-np.inf, q1, q2, np.inf]
    else:
        bins = [-np.inf, thresholds[0], thresholds[1], np.inf]
    return pd.cut(points, bins=bins, labels=[0, 1, 2]).astype(int).tolist()


def estimate_transition_matrix(states: List[int], n_states: int = 3, smoothing: float = 1.0) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        if a is None or b is None:
            continue
        counts[int(a), int(b)] += 1.0
    # Laplace smoothing and row-normalize
    P = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + smoothing * n_states)
    return P


def state_emissions(points: np.ndarray, states: List[int], n_states: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    means = np.zeros(n_states)
    variances = np.ones(n_states)
    if points is None or len(points) == 0 or len(states) == 0:
        return means, variances
    mask_valid = ~np.isnan(points)
    points = points[mask_valid]
    states_np = np.array(states)[mask_valid[: len(states)]] if len(states) == len(mask_valid) else np.array(states)[: len(points)]
    for s in range(n_states):
        vals = points[states_np == s]
        if len(vals) > 1:
            means[s] = float(np.mean(vals))
            variances[s] = float(np.var(vals, ddof=1)) if len(vals) > 1 else float(np.var(vals))
        else:
            means[s] = float(np.mean(points)) if len(points) else 0.0
            variances[s] = float(np.var(points, ddof=1)) if len(points) > 1 else 4.0
    variances = np.clip(variances, 1e-6, None)
    return means, variances


def next_state_probs(P: np.ndarray, last_state: int) -> np.ndarray:
    e = np.zeros(P.shape[0])
    e[int(last_state)] = 1.0
    return e @ P


def expected_points_from_probs(probs: np.ndarray, state_means: np.ndarray) -> float:
    return float(np.dot(probs, state_means))


def prob_over_threshold(probs: np.ndarray, state_means: np.ndarray, state_vars: np.ndarray, threshold: float, samples: int = 4000, rng: np.random.Generator = np.random.default_rng()) -> float:
    if probs is None or state_means is None or state_vars is None:
        return 0.0
    draws: List[np.ndarray] = []
    for s, p in enumerate(probs):
        if p <= 0:
            continue
        n = max(1, int(samples * float(p)))
        scale = np.sqrt(max(1e-6, float(state_vars[s])))
        draws.append(rng.normal(loc=float(state_means[s]), scale=scale, size=n))
    if not draws:
        return 0.0
    sim = np.concatenate(draws)
    return float(np.mean(sim >= threshold))


def _detect_points_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        'FantasyPointsDraftKings', 'Fantasy_Points', 'FantasyPoints', 'DK_Points', 'Points'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in ['ID', 'DK_ID', 'PlayerID', 'player_id']:
        if c in df.columns:
            return c
    return None


def _detect_name_column(df: pd.DataFrame) -> Optional[str]:
    for c in ['Name', 'PlayerName', 'player_name']:
        if c in df.columns:
            return c
    return None


def load_historical_cache(cache_dir: Optional[str]) -> Optional[pd.DataFrame]:
    if not cache_dir:
        return None
    try:
        pkl_path = os.path.join(cache_dir, 'historical_3years.pkl')
        csv_path = os.path.join(cache_dir, 'historical_3years.csv')
        if os.path.exists(pkl_path):
            return pd.read_pickle(pkl_path)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        # optional metadata
        meta = os.path.join(cache_dir, 'config.json')
        if os.path.exists(meta):
            with open(meta, 'r') as f:
                _ = json.load(f)
    except Exception:
        return None
    return None


def build_player_histories(df_players: pd.DataFrame, hist_df: Optional[pd.DataFrame], min_games: int = 30) -> Dict[str, np.ndarray]:
    histories: Dict[str, np.ndarray] = {}
    if hist_df is None or hist_df.empty:
        return histories

    id_col_df = _detect_id_column(df_players)
    name_col_df = _detect_name_column(df_players)

    id_col_hist = _detect_id_column(hist_df)
    name_col_hist = _detect_name_column(hist_df)
    pts_col_hist = _detect_points_column(hist_df)

    date_col = None
    for c in ['Date', 'date', 'GameDate', 'game_date']:
        if c in hist_df.columns:
            date_col = c
            break
    if pts_col_hist is None:
        return histories

    # last 3 years filter
    now = datetime.now()
    cutoff = now - timedelta(days=365 * 3)
    if date_col:
        dates = _to_datetime_safe(hist_df[date_col])
        hist_df = hist_df.loc[(dates.notna()) & (dates >= cutoff) & (dates <= now)].copy()

    # Build lookup by ID then fallback to name
    for _, row in df_players.iterrows():
        key_id = str(row[id_col_df]) if id_col_df and pd.notna(row.get(id_col_df)) else None
        key_name = str(row[name_col_df]) if name_col_df and pd.notna(row.get(name_col_df)) else None

        player_hist = None
        if key_id and id_col_hist and id_col_hist in hist_df.columns:
            subset = hist_df.loc[hist_df[id_col_hist].astype(str) == key_id]
            if not subset.empty:
                player_hist = subset[pts_col_hist].astype(float).values
        if player_hist is None and key_name and name_col_hist and name_col_hist in hist_df.columns:
            subset = hist_df.loc[hist_df[name_col_hist].astype(str).str.upper() == key_name.upper()]
            if not subset.empty:
                player_hist = subset[pts_col_hist].astype(float).values

        if player_hist is not None and len(player_hist) >= min_games:
            histories[key_name or key_id] = np.array(player_hist, dtype=float)

    return histories


def compute_markov_for_series(points_series: np.ndarray, player_thresholds: Tuple[float, float, float] = (20.0, 25.0, 30.0)) -> Dict[str, float]:
    if points_series is None or len(points_series) < 2:
        return {}
    states = discretize_states(points_series, method="quantile")
    P = estimate_transition_matrix(states, n_states=3, smoothing=1.0)
    means, vars_ = state_emissions(points_series, states, n_states=3)
    last_state = states[-1]
    probs_next = next_state_probs(P, last_state)
    mc_expected = expected_points_from_probs(probs_next, means)
    prob_over = {f'MC_Prob_Over_{int(t)}': prob_over_threshold(probs_next, means, vars_, t) for t in player_thresholds}
    return {
        'MC_Expected': mc_expected,
        'MC_State_0_Prob': float(probs_next[0]),
        'MC_State_1_Prob': float(probs_next[1]),
        'MC_State_2_Prob': float(probs_next[2]),
        **prob_over,
    }


def apply_markov_adjustments(
    df_players: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    cache_dir: Optional[str] = None,
    blend_alpha: float = 0.25,
    min_games: int = 30,
    player_thresholds: Tuple[float, float, float] = (20.0, 25.0, 30.0),
) -> pd.DataFrame:
    if df_players is None or df_players.empty:
        return df_players

    # Load cache if not provided
    if history_df is None:
        history_df = load_historical_cache(cache_dir)

    histories = build_player_histories(df_players, history_df, min_games=min_games)

    # Attach Markov metrics per player
    mc_expected_list: List[float] = []
    prob_cols = [f'MC_Prob_Over_{int(t)}' for t in player_thresholds]
    prob_values: Dict[str, List[float]] = {c: [] for c in prob_cols}

    name_col = _detect_name_column(df_players) or 'Name'
    for _, row in df_players.iterrows():
        name = str(row.get(name_col, ''))
        hist_points = histories.get(name)
        if hist_points is None:
            mc_expected_list.append(np.nan)
            for c in prob_cols:
                prob_values[c].append(np.nan)
            continue
        metrics = compute_markov_for_series(hist_points, player_thresholds=player_thresholds)
        mc_expected_list.append(metrics.get('MC_Expected', np.nan))
        for c in prob_cols:
            prob_values[c].append(metrics.get(c, np.nan))

    df_players = df_players.copy()
    df_players['MC_Expected'] = mc_expected_list
    for c in prob_cols:
        df_players[c] = prob_values[c]

    # Blend projections if available
    if 'Predicted_DK_Points' in df_players.columns:
        base = df_players['Predicted_DK_Points'].astype(float)
        blend = np.where(~np.isnan(df_players['MC_Expected']),
                         (1.0 - blend_alpha) * base + blend_alpha * df_players['MC_Expected'],
                         base)
        df_players['Predicted_DK_Points'] = blend
        df_players['Predicted_DK_Points_MarkovBlend'] = blend
    else:
        # If no base projection, use MC when available
        if 'MC_Expected' in df_players.columns:
            df_players['Predicted_DK_Points'] = df_players['MC_Expected'].fillna(0.0)

    return df_players


