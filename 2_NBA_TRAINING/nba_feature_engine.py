"""
nba_feature_engine.py — NBA-specific feature engineering.

Adapts the 3-engine pattern from 1_CORE_TRAINING/feature_engine.py to NBA box
score data from nba_api.  Reuses the same mathematical frameworks (financial
rolling stats, GARCH volatility, copula dependencies, spectral analysis) but
with NBA stat columns instead of MLB sabermetrics.

Part of the NBA DFS Training Pipeline.
"""

import numpy as np
import pandas as pd
import time
import multiprocessing
import warnings
import os

from scipy import stats
from scipy.stats import skew, kurtosis

from nba_config import (
    NBA_COUNTING_STATS, NBA_COPULA_KEY_METRICS, NBA_MARCEL_SHRINK_COLS,
    NBA_LEAGUE_AVG, NBA_PROB_THRESHOLDS, COPULA_PREFIXES,
    PREGAME_LAG_STATS, PREGAME_LAG_WINDOWS, PREGAME_SHIFT_FEATURES,
)

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from scipy.stats import kendalltau, spearmanr
    SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    SCIPY_ADVANCED_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False

# Default parallelism
_N_FEATURE_JOBS = min(multiprocessing.cpu_count(), 12)


def set_feature_jobs(n):
    """Set the number of parallel workers for feature engineering."""
    global _N_FEATURE_JOBS
    _N_FEATURE_JOBS = n


# ============================================================================
# DraftKings NBA scoring
# ============================================================================

def compute_dk_fpts(df):
    """Compute DraftKings NBA fantasy points from box score columns.

    DK NBA formula:
        PTS×1.0 + FG3M×0.5 + REB×1.25 + AST×1.5
        + STL×2.0 + BLK×2.0 + TOV×(-0.5) + DD2×1.5 + TD3×3.0
    """
    pts = pd.to_numeric(df.get('PTS', 0), errors='coerce').fillna(0)
    fg3m = pd.to_numeric(df.get('FG3M', 0), errors='coerce').fillna(0)
    reb = pd.to_numeric(df.get('REB', 0), errors='coerce').fillna(0)
    ast = pd.to_numeric(df.get('AST', 0), errors='coerce').fillna(0)
    stl = pd.to_numeric(df.get('STL', 0), errors='coerce').fillna(0)
    blk = pd.to_numeric(df.get('BLK', 0), errors='coerce').fillna(0)
    tov = pd.to_numeric(df.get('TOV', 0), errors='coerce').fillna(0)
    dd2 = pd.to_numeric(df.get('DD2', 0), errors='coerce').fillna(0)
    td3 = pd.to_numeric(df.get('TD3', 0), errors='coerce').fillna(0)

    return (
        pts * 1.0
        + fg3m * 0.5
        + reb * 1.25
        + ast * 1.5
        + stl * 2.0
        + blk * 2.0
        + tov * (-0.5)
        + dd2 * 1.5
        + td3 * 3.0
    )


# ============================================================================
# NBA Advanced Stats (derived from box score)
# ============================================================================

def compute_advanced_stats(df):
    """Compute NBA advanced stats from raw box score columns.

    These are per-game derived stats, NOT rolling — rolling versions are
    computed later by the financial engine or prepare_pregame_features.
    """
    df = df.copy()

    # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
    denom = 2 * (df['FGA'] + 0.44 * df['FTA'])
    df['ts_pct'] = np.where(denom > 0, df['PTS'] / denom, np.nan)

    # Effective FG %: (FGM + 0.5 * FG3M) / FGA
    df['efg_pct'] = np.where(df['FGA'] > 0,
                              (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'],
                              np.nan)

    # AST/TOV ratio
    df['ast_tov_ratio'] = np.where(df['TOV'] > 0,
                                    df['AST'] / df['TOV'],
                                    df['AST'])  # if 0 TOV, ratio = AST count

    # Stocks (steals + blocks)
    df['stl_blk_sum'] = df['STL'] + df['BLK']

    # Per-minute rates (handle 0-minute games)
    minutes = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
    minutes_safe = np.where(minutes > 0, minutes, np.nan)
    df['pts_per_min'] = df['PTS'] / minutes_safe
    df['reb_per_min'] = df['REB'] / minutes_safe
    df['ast_per_min'] = df['AST'] / minutes_safe
    df['fantasy_per_min'] = df['calculated_dk_fpts'] / minutes_safe

    # Usage proxy: (FGA + 0.44*FTA + TOV) — possession usage without team data
    df['usage_proxy'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

    # FG3 attempt rate: FG3A / FGA
    df['fg3a_rate'] = np.where(df['FGA'] > 0, df['FG3A'] / df['FGA'], 0)

    # Free throw rate: FTA / FGA
    df['ft_rate'] = np.where(df['FGA'] > 0, df['FTA'] / df['FGA'], 0)

    return df


# ============================================================================
# Engine 1 — NBA Financial-Style Engine
# ============================================================================

def _financial_process_player(name, group, stat_cols, rolling_windows, date_col):
    """Process a single player for financial-style features."""
    new_features = {}

    for col in stat_cols:
        if col not in group.columns:
            continue
        for window in rolling_windows:
            new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
            new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
            new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
        if f'{col}_sma_28' in new_features:
            sma28 = new_features[f'{col}_sma_28']
            new_features[f'{col}_vs_sma_28'] = np.where(sma28 != 0, (group[col] / sma28) - 1, 0)

    # Bollinger bands on DK points
    for window in rolling_windows:
        mean = group['calculated_dk_fpts'].rolling(window).mean()
        std = group['calculated_dk_fpts'].rolling(window).std()
        upper = mean + (2 * std)
        lower = mean - (2 * std)
        new_features[f'dk_fpts_upper_band_{window}'] = upper
        new_features[f'dk_fpts_lower_band_{window}'] = lower
        band_range = upper - lower
        new_features[f'dk_fpts_band_width_{window}'] = np.where(
            mean != 0, band_range / mean, 0)
        new_features[f'dk_fpts_band_position_{window}'] = np.where(
            band_range != 0, (group['calculated_dk_fpts'] - lower) / band_range, 0.5)

    # Minutes volume analysis
    if 'MIN' in group.columns:
        min_roll = group['MIN'].rolling(28).mean()
        new_features['MIN_roll_mean_28'] = min_roll
        new_features['MIN_ratio'] = np.where(min_roll > 0, group['MIN'] / min_roll, 1)
        new_features['dk_fpts_MIN_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group['MIN'])

    # Usage volume analysis
    if 'usage_proxy' in group.columns:
        usg_roll = group['usage_proxy'].rolling(28).mean()
        new_features['usage_proxy_roll_mean_28'] = usg_roll
        new_features['usage_proxy_ratio'] = np.where(usg_roll > 0, group['usage_proxy'] / usg_roll, 1)

    # Calendar features
    new_features['day_of_week'] = group[date_col].dt.dayofweek
    new_features['month'] = group[date_col].dt.month
    new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)
    new_features['day_of_week_sin'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
    new_features['day_of_week_cos'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)

    return pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1)


class NBAFinancialEngine:
    """Financial-style rolling features adapted for NBA box score stats."""

    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = NBA_COUNTING_STATS + ['calculated_dk_fpts',
                                                     'ts_pct', 'efg_pct',
                                                     'usage_proxy', 'fantasy_per_min']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28, 45]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        date_col = 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])

        # Ensure required columns exist
        for col in self.stat_cols:
            if col not in df.columns:
                df[col] = 0

        stat_cols = self.stat_cols
        rolling_windows = self.rolling_windows
        groups = list(df.groupby('Name'))
        n_jobs = min(_N_FEATURE_JOBS, len(groups))

        if JOBLIB_AVAILABLE and n_jobs > 1:
            print(f"  Financial engine: {len(groups)} players across {n_jobs} cores")
            all_players_data = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_financial_process_player)(name, group, stat_cols, rolling_windows, date_col)
                for name, group in groups
            )
        else:
            all_players_data = [
                _financial_process_player(name, group, stat_cols, rolling_windows, date_col)
                for name, group in groups
            ]

        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        return enhanced_df


# ============================================================================
# Engine 2 — NBA Probabilistic Engine (GARCH, distributional, regime)
# ============================================================================

def _calculate_hurst(series):
    """Hurst exponent (simplified R/S method)."""
    try:
        data = np.array(series).flatten()
        data = data[~np.isnan(data)]
        if len(data) < 10:
            return 0.5
        mean_adj = data - np.mean(data)
        cumsum = np.cumsum(mean_adj)
        R = np.max(cumsum) - np.min(cumsum)
        S = np.std(data)
        if S == 0:
            return 0.5
        hurst = np.log(R / S) / np.log(len(data))
        return max(0, min(1, hurst))
    except Exception:
        return 0.5


def _calculate_drawdown_duration(drawdown_series):
    durations = []
    current_duration = 0
    for dd in drawdown_series:
        if dd < 0:
            current_duration += 1
        else:
            current_duration = 0
        durations.append(current_duration)
    return pd.Series(durations, index=drawdown_series.index)


def _safe_tail_ratio(x):
    if len(x) < 10:
        return np.nan
    try:
        q95, q50, q05 = x.quantile(0.95), x.quantile(0.5), x.quantile(0.05)
        denom = q50 - q05
        if abs(denom) < 1e-10:
            return 1.0
        result = (q95 - q50) / denom
        return result if np.isfinite(result) else 1.0
    except Exception:
        return np.nan


def _prob_player_all(name, group, min_observations, prob_thresholds):
    """Compute ALL probabilistic features for a single NBA player."""
    group = group.sort_values('date')
    fpts = group['calculated_dk_fpts']

    # === GARCH ===
    if len(group) < min_observations:
        group['garch_volatility'] = fpts.rolling(window=5).std()
        group['garch_conditional_volatility'] = group['garch_volatility']
        group['volatility_regime'] = 0
    else:
        fpts_clean = fpts.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        returns = fpts_clean.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if ARCH_AVAILABLE and len(returns) >= 20 and not returns.empty and returns.var() > 1e-10:
            try:
                if np.all(np.isfinite(returns)) and returns.std() > 1e-6:
                    scaled_returns = returns * 100
                    garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=True)
                    garch_fitted = garch_model.fit(disp='off', show_warning=False)
                    conditional_vol = garch_fitted.conditional_volatility / 100
                    conditional_vol = conditional_vol.replace([np.inf, -np.inf], np.nan).fillna(returns.std())
                    vol_series = pd.Series(index=group.index, dtype=float)
                    vol_series.iloc[1:len(conditional_vol)+1] = conditional_vol.values
                    group['garch_volatility'] = vol_series.bfill().fillna(returns.std())
                    group['garch_conditional_volatility'] = vol_series.bfill().fillna(returns.std())
                    vol_percentile = conditional_vol.rolling(window=10).rank(pct=True).fillna(0.5)
                    regime_series = pd.Series(index=group.index, dtype=int)
                    regime_series.iloc[1:len(vol_percentile)+1] = (vol_percentile > 0.7).astype(int)
                    group['volatility_regime'] = regime_series.fillna(0)
                else:
                    raise ValueError("Invalid returns")
            except Exception:
                fallback_vol = fpts_clean.rolling(window=10).std()
                group['garch_volatility'] = fallback_vol
                group['garch_conditional_volatility'] = fallback_vol
                group['volatility_regime'] = 0
        else:
            fallback_vol = fpts_clean.rolling(window=10).std()
            group['garch_volatility'] = fallback_vol
            group['garch_conditional_volatility'] = fallback_vol
            rolling_vol = fpts_clean.rolling(window=10).std().fillna(0)
            vol_threshold = rolling_vol.quantile(0.7) if not rolling_vol.empty else 0
            group['volatility_regime'] = (rolling_vol > vol_threshold).astype(int)

    # === Distributional ===
    for window in [7, 14, 30]:
        group[f'skewness_{window}d'] = fpts.rolling(window).apply(
            lambda x: skew(x) if len(x) >= 3 else np.nan, raw=False)
        group[f'kurtosis_{window}d'] = fpts.rolling(window).apply(
            lambda x: kurtosis(x) if len(x) >= 4 else np.nan, raw=False)
        group[f'var_95_{window}d'] = fpts.rolling(window).quantile(0.05)
        group[f'var_99_{window}d'] = fpts.rolling(window).quantile(0.01)
        group[f'expected_shortfall_{window}d'] = fpts.rolling(window).apply(
            lambda x: x[x <= x.quantile(0.05)].mean() if len(x) >= 5 else x.min(), raw=False)
    group['tail_ratio'] = fpts.rolling(30).apply(_safe_tail_ratio, raw=False)

    # NBA-specific probability thresholds (higher than MLB)
    for threshold in prob_thresholds:
        group[f'prob_exceed_{threshold}'] = fpts.rolling(30).apply(
            lambda x, t=threshold: (x > t).mean() if len(x) >= 5 else np.nan, raw=False)

    # === Regime ===
    short_ma = fpts.rolling(window=5).mean()
    long_ma = fpts.rolling(window=20).mean()
    group['bull_regime'] = (short_ma > long_ma).astype(int)
    with np.errstate(divide='ignore', invalid='ignore'):
        regime_strength = (short_ma - long_ma) / long_ma
        group['regime_strength'] = regime_strength.fillna(0).replace([np.inf, -np.inf], 0)
    momentum = fpts.pct_change(5)
    if len(momentum.dropna()) > 3 and momentum.std() > 0 and not momentum.isna().all():
        try:
            if len(momentum.dropna().unique()) >= 3:
                momentum_regime = pd.cut(momentum, bins=3, labels=[0, 1, 2], duplicates='drop')
                group['momentum_regime'] = momentum_regime.astype(float)
            else:
                med = momentum.median()
                group['momentum_regime'] = np.where(momentum > med, 2,
                                                     np.where(momentum < med, 0, 1)).astype(float)
        except (ValueError, TypeError):
            group['momentum_regime'] = np.where(momentum > 0, 2,
                                                 np.where(momentum < 0, 0, 1)).astype(float)
    else:
        group['momentum_regime'] = 1.0
    rolling_std = fpts.rolling(10).std()
    rolling_mean_val = fpts.rolling(10).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rolling_cv = rolling_std / rolling_mean_val
        rolling_cv = rolling_cv.fillna(0).replace([np.inf, -np.inf], 0)
        if len(rolling_cv.dropna()) > 0:
            group['consistency_regime'] = (rolling_cv < rolling_cv.quantile(0.33)).astype(int)
        else:
            group['consistency_regime'] = 0

    # === Advanced ===
    if len(group) >= min_observations:
        group['entropy'] = fpts.rolling(20).apply(
            lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) >= 5 else np.nan, raw=False)
        group['hurst_exponent'] = fpts.rolling(30).apply(
            lambda x: _calculate_hurst(x) if len(x) >= 10 else 0.5, raw=False)
        cumulative = fpts.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        group['max_drawdown'] = drawdown.rolling(20).min()
        group['current_drawdown'] = drawdown
        group['drawdown_duration'] = _calculate_drawdown_duration(drawdown)
        returns = fpts.pct_change().replace([np.inf, -np.inf], np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_mean = returns.rolling(20).mean()
            r_std = returns.rolling(20).std()
            group['rolling_sharpe'] = (r_mean / r_std).replace([np.inf, -np.inf], np.nan)
    else:
        for col in ['entropy', 'hurst_exponent', 'max_drawdown',
                    'current_drawdown', 'drawdown_duration', 'rolling_sharpe']:
            group[col] = np.nan

    return group


class NBAProbabilisticEngine:
    """GARCH volatility, distributional moments, correlations, regimes, advanced.
    Parallelized: per-player processing runs across all CPU cores."""

    def __init__(self, lookback_window=30, min_observations=10):
        self.lookback_window = lookback_window
        self.min_observations = min_observations

    def calculate_correlation_features(self, df):
        """Cross-player correlations (stays sequential — needs full df)."""
        print("Calculating correlation features...")
        top_players = df.groupby('Name')['calculated_dk_fpts'].sum().nlargest(50).index.tolist()
        correlation_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date').reset_index(drop=True)
            if name in top_players:
                correlations = []
                for other_player in top_players[:10]:
                    if other_player != name:
                        try:
                            other_group = df[df['Name'] == other_player].sort_values('date').reset_index(drop=True)
                            if len(other_group) > 0:
                                common_dates = set(group['date']).intersection(set(other_group['date']))
                                if len(common_dates) >= 10:
                                    player_series = group.set_index('date')['calculated_dk_fpts']
                                    other_series = other_group.set_index('date')['calculated_dk_fpts']
                                    player_series = player_series[~player_series.index.duplicated(keep='last')]
                                    other_series = other_series[~other_series.index.duplicated(keep='last')]
                                    rolling_corr = player_series.rolling(
                                        window=min(20, len(common_dates))).corr(other_series)
                                    corr_values = np.zeros(len(group))
                                    for i, date in enumerate(group['date']):
                                        if date in rolling_corr.index:
                                            val = rolling_corr.loc[date]
                                            corr_values[i] = val if not pd.isna(val) else 0
                                    correlations.append(corr_values)
                                else:
                                    correlations.append(np.zeros(len(group)))
                        except Exception:
                            correlations.append(np.zeros(len(group)))
                if correlations:
                    corr_array = np.array(correlations).T
                    group['avg_player_correlation'] = np.mean(corr_array, axis=1)
                    group['correlation_volatility'] = np.std(corr_array, axis=1) if corr_array.shape[1] > 1 else 0
                else:
                    group['avg_player_correlation'] = 0
                    group['correlation_volatility'] = 0
            else:
                group['avg_player_correlation'] = 0
                group['correlation_volatility'] = 0
            correlation_features.append(group)
        return pd.concat(correlation_features, ignore_index=True)

    def calculate_all_features(self, df):
        """Run all probabilistic features. GARCH/dist/regime/advanced are parallel."""
        print("Starting probabilistic feature engineering...")
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for probabilistic features")

        groups = list(df.groupby('Name'))
        n_jobs = min(_N_FEATURE_JOBS, len(groups))
        min_obs = self.min_observations

        if JOBLIB_AVAILABLE and n_jobs > 1:
            print(f"  Probabilistic engine: {len(groups)} players across {n_jobs} cores")
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(_prob_player_all)(name, group, min_obs, NBA_PROB_THRESHOLDS)
                for name, group in groups
            )
        else:
            results = [_prob_player_all(name, group, min_obs, NBA_PROB_THRESHOLDS)
                       for name, group in groups]

        df = pd.concat(results, ignore_index=True)
        df = self.calculate_correlation_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill()
        print("Probabilistic feature engineering completed.")
        return df


# ============================================================================
# Engine 3 — NBA Copula Engine (copula, EVT, network, spectral)
# ============================================================================

class NBACopulaEngine:
    """Copula parameters, extreme value theory, network features, spectral.
    Uses NBA key metrics instead of MLB batting stats."""

    def __init__(self, lookback_window=30, min_observations=15):
        self.lookback_window = lookback_window
        self.min_observations = min_observations

    def gaussian_copula_param(self, x, y):
        try:
            if len(x) < 5 or len(y) < 5:
                return 0.0
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) < 5:
                return 0.0
            tau, _ = kendalltau(x_clean, y_clean)
            rho = np.sin(np.pi * tau / 2)
            return rho if not np.isnan(rho) else 0.0
        except Exception:
            return 0.0

    def clayton_copula_param(self, x, y):
        try:
            if len(x) < 5 or len(y) < 5:
                return 0.0
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) < 5:
                return 0.0
            tau, _ = kendalltau(x_clean, y_clean)
            if tau <= 0:
                return 0.0
            theta = 2 * tau / (1 - tau)
            return max(0, theta) if not np.isnan(theta) else 0.0
        except Exception:
            return 0.0

    def tail_dependence_coefficient(self, x, y, tail='upper'):
        try:
            if len(x) < 10 or len(y) < 10:
                return 0.0
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean, y_clean = x[mask], y[mask]
            if len(x_clean) < 10:
                return 0.0
            x_ranks = stats.rankdata(x_clean) / len(x_clean)
            y_ranks = stats.rankdata(y_clean) / len(y_clean)
            threshold = 0.9 if tail == 'upper' else 0.1
            if tail == 'upper':
                condition = (x_ranks > threshold) & (y_ranks > threshold)
                tail_dep = np.sum(condition) / np.sum(x_ranks > threshold)
            else:
                condition = (x_ranks < threshold) & (y_ranks < threshold)
                tail_dep = np.sum(condition) / np.sum(x_ranks < threshold)
            return tail_dep if not np.isnan(tail_dep) else 0.0
        except Exception:
            return 0.0

    def _copula_evt_spectral_one_player(self, name, group):
        """Copula + EVT + spectral for one player."""
        group = group.sort_values('date')
        key_metrics = NBA_COPULA_KEY_METRICS

        # Copula features
        copula_dict = {}
        for i, metric1 in enumerate(key_metrics):
            for j, metric2 in enumerate(key_metrics[i+1:], i+1):
                if metric1 in group.columns and metric2 in group.columns:
                    gp, cp, ut, lt = [], [], [], []
                    for idx in range(len(group)):
                        si = max(0, idx - self.lookback_window)
                        w1 = group[metric1].iloc[si:idx+1]
                        w2 = group[metric2].iloc[si:idx+1]
                        if len(w1) >= self.min_observations:
                            gp.append(self.gaussian_copula_param(w1.values, w2.values))
                            cp.append(self.clayton_copula_param(w1.values, w2.values))
                            ut.append(self.tail_dependence_coefficient(w1.values, w2.values, 'upper'))
                            lt.append(self.tail_dependence_coefficient(w1.values, w2.values, 'lower'))
                        else:
                            gp.append(0.0); cp.append(0.0); ut.append(0.0); lt.append(0.0)
                    copula_dict[f'gaussian_copula_{metric1}_{metric2}'] = gp
                    copula_dict[f'clayton_copula_{metric1}_{metric2}'] = cp
                    copula_dict[f'upper_tail_dep_{metric1}_{metric2}'] = ut
                    copula_dict[f'lower_tail_dep_{metric1}_{metric2}'] = lt
        for feature, values in copula_dict.items():
            group[feature] = values

        # EVT features
        fpts = group['calculated_dk_fpts']
        block_maxima = [fpts.iloc[i:i+7].max() for i in range(0, len(fpts), 7)
                        if len(fpts.iloc[i:i+7]) > 0]
        if len(block_maxima) >= 10:
            bm = np.array(block_maxima)
            loc, sc, sh = np.mean(bm), np.std(bm), stats.skew(bm) / 3
            group['evt_location'] = loc
            group['evt_scale'] = sc
            group['evt_shape'] = sh
            group['evt_return_level'] = loc + sc * (-np.log(-np.log(0.95)))
            threshold = np.percentile(fpts, 90)
            group['exceedance_prob'] = (fpts > threshold).rolling(window=20).mean()
            group['extreme_value_index'] = sh
            excess = fpts[fpts > threshold] - threshold
            group['pot_threshold'] = threshold
            group['pot_excess_mean'] = excess.mean() if len(excess) > 0 else 0
            group['pot_excess_std'] = excess.std() if len(excess) > 0 else 0
        else:
            for c in ['evt_location', 'evt_shape', 'evt_return_level', 'exceedance_prob',
                       'extreme_value_index', 'pot_threshold', 'pot_excess_mean', 'pot_excess_std']:
                group[c] = 0
            group['evt_scale'] = 1

        # Spectral features
        if len(group) >= 20:
            fvals = fpts.values
            fft_values = np.fft.fft(fvals)
            freqs = np.fft.fftfreq(len(fvals))
            psd = np.abs(fft_values) ** 2
            di = np.argmax(psd[1:len(psd)//2]) + 1
            group['dominant_frequency'] = freqs[di]
            psd_norm = psd / np.sum(psd)
            group['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
            group['spectral_centroid'] = np.sum(freqs[:len(psd)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            cpsd = np.cumsum(psd[:len(psd)//2])
            ri = np.where(cpsd >= 0.85 * cpsd[-1])[0]
            group['spectral_rolloff'] = freqs[ri[0]] if len(ri) > 0 else 0
            rse = []
            for i in range(len(fvals)):
                si = max(0, i - 14)
                wd = fvals[si:i+1]
                if len(wd) >= 7:
                    wf = np.fft.fft(wd)
                    wp = np.abs(wf) ** 2
                    wpn = wp / np.sum(wp)
                    rse.append(-np.sum(wpn * np.log(wpn + 1e-10)))
                else:
                    rse.append(np.nan)
            group['rolling_spectral_entropy'] = rse
        else:
            for c in ['dominant_frequency', 'spectral_entropy', 'spectral_centroid',
                       'spectral_rolloff', 'rolling_spectral_entropy']:
                group[c] = np.nan
        return group

    def calculate_network_features(self, df):
        """Cross-player network features (sequential)."""
        print("Calculating network features...")
        top_players = df.groupby('Name')['calculated_dk_fpts'].sum().nlargest(30).index.tolist()
        network_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            if name in top_players:
                centrality_scores, clustering_coeffs = [], []
                for date in group['date']:
                    same_date_players = df[(df['date'] == date) & (df['Name'] != name)]
                    if len(same_date_players) >= 10:
                        correlations = []
                        for other_name in top_players[:20]:
                            if other_name != name:
                                other_data = same_date_players[same_date_players['Name'] == other_name]
                                if len(other_data) > 0:
                                    player_perf = group[group['date'] == date]['calculated_dk_fpts'].iloc[0] if len(group[group['date'] == date]) > 0 else 0
                                    other_perf = other_data['calculated_dk_fpts'].iloc[0]
                                    connection_strength = 1 / (1 + abs(player_perf - other_perf))
                                    correlations.append(connection_strength)
                        centrality = sum(correlations) / len(correlations) if correlations else 0
                        centrality_scores.append(centrality)
                        clustering_coeffs.append(np.std(correlations) if correlations else 0)
                    else:
                        centrality_scores.append(0)
                        clustering_coeffs.append(0)
                group['network_centrality'] = centrality_scores
                group['network_clustering'] = clustering_coeffs
                group['network_volatility'] = pd.Series(centrality_scores).rolling(window=5).std().fillna(0).values
                group['network_efficiency'] = pd.Series(centrality_scores).rolling(window=5).mean().fillna(0).values
            else:
                for col in ['network_centrality', 'network_clustering',
                            'network_volatility', 'network_efficiency']:
                    group[col] = 0
            network_features.append(group)
        return pd.concat(network_features, ignore_index=True)

    def calculate_all_advanced_features(self, df):
        """Copula + EVT + spectral parallelized; network sequential."""
        print("Starting advanced copula and dependency feature engineering...")
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for advanced features")

        groups = list(df.groupby('Name'))
        n_jobs = min(_N_FEATURE_JOBS, len(groups))

        if JOBLIB_AVAILABLE and n_jobs > 1:
            print(f"  Copula engine: {len(groups)} players across {n_jobs} cores")
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                delayed(self._copula_evt_spectral_one_player)(name, group)
                for name, group in groups
            )
        else:
            results = [self._copula_evt_spectral_one_player(name, group)
                       for name, group in groups]

        df = pd.concat(results, ignore_index=True)
        df = self.calculate_network_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill()
        print("Advanced copula and dependency feature engineering completed.")
        return df


# ============================================================================
# Sabermetric-equivalent feature engineering for NBA
# ============================================================================

def engineer_features(df, date_series=None):
    """Calculate advanced NBA stats + rolling features from raw box score data.

    NBA equivalent of MLB's engineer_features() — computes derived stats
    and lagged rolling DK point features.
    """
    if date_series is None:
        date_series = df['date']
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days

    # Compute advanced stats if not already present
    if 'ts_pct' not in df.columns:
        df = compute_advanced_stats(df)

    # Rolling DK point features with lag (shift(1) = no current game leakage)
    if 'calculated_dk_fpts' in df.columns:
        df['calculated_dk_fpts'] = df['calculated_dk_fpts'].replace([np.inf, -np.inf], np.nan)
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Replace inf but leave NaN for the preprocessor
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ============================================================================
# Post-processing: Copula PCA + Marcel shrinkage
# ============================================================================

def reduce_copula_features(df, n_components=8):
    """Replace raw copula/tail-dep columns with PCA components."""
    copula_cols = [c for c in df.columns if any(c.startswith(p) for p in COPULA_PREFIXES)]
    if len(copula_cols) < n_components or not PCA_AVAILABLE:
        print(f"Skipping copula PCA: {len(copula_cols)} columns, PCA_AVAILABLE={PCA_AVAILABLE}")
        return df

    print(f"Reducing {len(copula_cols)} copula columns to {n_components} PCA components...")
    copula_data = df[copula_cols].fillna(0).values
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(copula_data)

    df = df.drop(columns=copula_cols)
    for i in range(n_components):
        df[f'copula_pc_{i+1}'] = components[:, i]

    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"Copula PCA: {explained:.1f}% variance explained by {n_components} components")
    return df


def apply_marcel_shrinkage(df, min_games=50):
    """Apply Marcel-style shrinkage to NBA rate stats for small-sample players."""
    print("Applying Marcel shrinkage to small-sample players...")
    shrink_cols = [c for c in NBA_MARCEL_SHRINK_COLS if c in df.columns]

    if not shrink_cols:
        return df

    player_counts = df.groupby('Name')['date'].transform('count')
    reliability = player_counts / (player_counts + min_games)

    for col in shrink_cols:
        league_avg = NBA_LEAGUE_AVG.get(col.upper(), df[col].median())
        df[col] = reliability * df[col] + (1 - reliability) * league_avg

    n_shrunk = (player_counts < min_games).sum()
    print(f"Marcel shrinkage applied to {n_shrunk} rows across {len(shrink_cols)} features")
    return df


# ============================================================================
# Pre-game feature preparation (leakage-free)
# ============================================================================

def prepare_pregame_features(df, lag_stats=None, lag_windows=None, shift_features=None):
    """Convert same-game features into pre-game features.

    Same logic as MLB version: lagged rolling averages + shift(1).
    All output columns prefixed with 'pgm_' (pre-game).
    """
    if lag_stats is None:
        lag_stats = PREGAME_LAG_STATS
    if lag_windows is None:
        lag_windows = PREGAME_LAG_WINDOWS
    if shift_features is None:
        shift_features = PREGAME_SHIFT_FEATURES

    print("\n===== PREPARING PRE-GAME FEATURES =====")
    df = df.copy()
    df = df.sort_values(['Name', 'date'])

    # 1. Lagged rolling averages of per-game stats
    available_stats = [s for s in lag_stats if s in df.columns]
    print(f"  Computing lagged rolling averages for {len(available_stats)} stats "
          f"x {len(lag_windows)} windows...")

    for stat in available_stats:
        for window in lag_windows:
            col_name = f'pgm_{stat}_{window}'
            df[col_name] = df.groupby('Name')[stat].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )

    # 2. Shift existing probabilistic/rolling features by 1 game
    available_shift = [f for f in shift_features if f in df.columns]
    print(f"  Shifting {len(available_shift)} probabilistic/rolling features by 1 game...")

    for feat in available_shift:
        col_name = f'pgm_{feat}'
        df[col_name] = df.groupby('Name')[feat].transform(
            lambda x: x.shift(1)
        )

    # 3. Shift copula PCA components by 1 game
    copula_pcs = [c for c in df.columns if c.startswith('copula_pc_')]
    if copula_pcs:
        print(f"  Shifting {len(copula_pcs)} copula PCA components by 1 game...")
        for pc in copula_pcs:
            df[f'pgm_{pc}'] = df.groupby('Name')[pc].transform(
                lambda x: x.shift(1)
            )

    pgm_cols = [c for c in df.columns if c.startswith('pgm_')]
    print(f"  Created {len(pgm_cols)} pre-game features (pgm_* prefix)")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
