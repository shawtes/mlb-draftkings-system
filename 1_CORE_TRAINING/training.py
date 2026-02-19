"""
training.py — MLB DFS Training Pipeline v2

Thin orchestrator that imports from:
  - config.py          (CLI args, constants, feature lists)
  - feature_engine.py  (3 feature classes, DK scoring, sabermetrics)
  - model_builder.py   (ensemble, SHAP, Optuna, quantile models)
  - validator.py       (walk-forward CV, per-player eval, CQR)

The 3 engine classes are kept in-place below for backward compatibility
(other scripts that do `from training import EnhancedMLBFinancialStyleEngine`
will still work).

Usage:
  python training.py --data-path /path/to/merged_fangraphs_data.csv --output-dir ./output
  python training.py --data-path /path/to/data.csv --skip-hpo --n-splits 3
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
import os
import multiprocessing

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from scipy import stats
from scipy.stats import skew, kurtosis

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch package not available. GARCH features will be simplified.")

try:
    from scipy.stats import kendalltau, spearmanr
    from scipy.optimize import minimize
    SCIPY_ADVANCED_AVAILABLE = True
except ImportError:
    SCIPY_ADVANCED_AVAILABLE = False
    print("Warning: Advanced scipy features not available.")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.regime_switching import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Some advanced features will be simplified.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# BACKWARD-COMPATIBLE CLASS DEFINITIONS
# These classes are kept in-place so that existing code that imports from
# training.py continues to work. The canonical versions live in
# feature_engine.py.
# ============================================================================

class EnhancedMLBFinancialStyleEngine:
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28, 45]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])
        if 'PA' not in df.columns and 'PA.1' in df.columns:
            df['PA'] = df['PA.1']
        if 'AB' not in df.columns and 'AB.1' in df.columns:
            df['AB'] = df['AB.1']
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                print(f"Warning: Column '{col}' not found. Initialized with 0.")
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                if f'{col}_sma_28' in new_features:
                    new_features[f'{col}_vs_sma_28'] = (group[col] / new_features[f'{col}_sma_28']) - 1
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                new_features[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                if mean is not None and not mean.empty:
                    band_range = new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}']
                    new_features[f'dk_fpts_band_width_{window}'] = band_range / mean
                    new_features[f'dk_fpts_band_position_{window}'] = (group['calculated_dk_fpts'] - new_features[f'dk_fpts_lower_band_{window}']) / band_range
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    new_features[f'{vol_col}_ratio'] = group[vol_col] / new_features[f'{vol_col}_roll_mean_28']
                    new_features[f'dk_fpts_{vol_col}_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group[vol_col])
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns and 'PA' in group.columns and group['PA'].sum() > 0:
                    new_features[f'{col}_per_pa'] = group[col] / group['PA']
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)
            new_features['day_of_week_sin'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
            new_features['day_of_week_cos'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)
            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        return enhanced_df


class ProbabilisticMLBEngine:
    """GARCH volatility, distributional moments, correlations, regimes, advanced."""
    def __init__(self, lookback_window=30, min_observations=10):
        self.lookback_window = lookback_window
        self.min_observations = min_observations

    def calculate_garch_features(self, df):
        print("Calculating GARCH volatility features...")
        garch_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            if len(group) < self.min_observations:
                group['garch_volatility'] = group['calculated_dk_fpts'].rolling(window=5).std()
                group['garch_conditional_volatility'] = group['garch_volatility']
                group['volatility_regime'] = 0
            else:
                fpts_clean = group['calculated_dk_fpts'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
                returns = fpts_clean.pct_change().dropna()
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
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
            garch_features.append(group)
        return pd.concat(garch_features, ignore_index=True)

    def calculate_distributional_features(self, df):
        print("Calculating distributional features...")
        dist_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            for window in [7, 14, 30]:
                group[f'skewness_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(lambda x: skew(x) if len(x) >= 3 else 0)
                group[f'kurtosis_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(lambda x: kurtosis(x) if len(x) >= 4 else 0)
                group[f'var_95_{window}d'] = group['calculated_dk_fpts'].rolling(window).quantile(0.05)
                group[f'var_99_{window}d'] = group['calculated_dk_fpts'].rolling(window).quantile(0.01)
                group[f'expected_shortfall_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(
                    lambda x: x[x <= x.quantile(0.05)].mean() if len(x) >= 5 else x.min())
            def safe_tail_ratio(x):
                if len(x) < 10:
                    return 1.0
                try:
                    q95, q50, q05 = x.quantile(0.95), x.quantile(0.5), x.quantile(0.05)
                    denom = q50 - q05
                    if abs(denom) < 1e-10:
                        return 1.0
                    result = (q95 - q50) / denom
                    return result if np.isfinite(result) else 1.0
                except Exception:
                    return 1.0
            group['tail_ratio'] = group['calculated_dk_fpts'].rolling(30).apply(safe_tail_ratio)
            for threshold in [5, 10, 15, 20]:
                group[f'prob_exceed_{threshold}'] = group['calculated_dk_fpts'].rolling(30).apply(
                    lambda x: (x > threshold).mean() if len(x) >= 5 else 0)
            dist_features.append(group)
        return pd.concat(dist_features, ignore_index=True)

    def calculate_correlation_features(self, df):
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
                                    rolling_corr = player_series.rolling(window=min(20, len(common_dates))).corr(other_series)
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

    def calculate_regime_features(self, df):
        print("Calculating regime features...")
        regime_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            short_ma = group['calculated_dk_fpts'].rolling(window=5).mean()
            long_ma = group['calculated_dk_fpts'].rolling(window=20).mean()
            group['bull_regime'] = (short_ma > long_ma).astype(int)
            with np.errstate(divide='ignore', invalid='ignore'):
                regime_strength = (short_ma - long_ma) / long_ma
                group['regime_strength'] = regime_strength.fillna(0).replace([np.inf, -np.inf], 0)
            momentum = group['calculated_dk_fpts'].pct_change(5)
            if len(momentum.dropna()) > 3 and momentum.std() > 0 and not momentum.isna().all():
                try:
                    momentum_clean = momentum.dropna()
                    if len(momentum_clean.unique()) >= 3:
                        momentum_regime = pd.cut(momentum, bins=3, labels=[0, 1, 2], duplicates='drop')
                        group['momentum_regime'] = momentum_regime.astype(float)
                    else:
                        momentum_median = momentum.median()
                        group['momentum_regime'] = np.where(momentum > momentum_median, 2, np.where(momentum < momentum_median, 0, 1)).astype(float)
                except (ValueError, TypeError):
                    group['momentum_regime'] = np.where(momentum > 0, 2, np.where(momentum < 0, 0, 1)).astype(float)
            else:
                group['momentum_regime'] = 1.0
            rolling_std = group['calculated_dk_fpts'].rolling(10).std()
            rolling_mean = group['calculated_dk_fpts'].rolling(10).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rolling_cv = rolling_std / rolling_mean
                rolling_cv = rolling_cv.fillna(0).replace([np.inf, -np.inf], 0)
                if len(rolling_cv.dropna()) > 0:
                    cv_33_quantile = rolling_cv.quantile(0.33)
                    group['consistency_regime'] = (rolling_cv < cv_33_quantile).astype(int)
                else:
                    group['consistency_regime'] = 0
            regime_features.append(group)
        return pd.concat(regime_features, ignore_index=True)

    def calculate_advanced_features(self, df):
        print("Calculating advanced probabilistic features...")
        advanced_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            if len(group) >= self.min_observations:
                group['entropy'] = group['calculated_dk_fpts'].rolling(20).apply(
                    lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) >= 5 else 0)
                group['hurst_exponent'] = group['calculated_dk_fpts'].rolling(30).apply(
                    lambda x: self._calculate_hurst(x) if len(x) >= 10 else 0.5)
                cumulative = group['calculated_dk_fpts'].cumsum()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                group['max_drawdown'] = drawdown.rolling(20).min()
                group['current_drawdown'] = drawdown
                group['drawdown_duration'] = self._calculate_drawdown_duration(drawdown)
                returns = group['calculated_dk_fpts'].pct_change()
                returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    r_mean = returns.rolling(20).mean()
                    r_std = returns.rolling(20).std()
                    group['rolling_sharpe'] = r_mean / r_std
                    group['rolling_sharpe'] = group['rolling_sharpe'].fillna(0).replace([np.inf, -np.inf], 0)
            else:
                group['entropy'] = 0
                group['hurst_exponent'] = 0.5
                group['max_drawdown'] = 0
                group['current_drawdown'] = 0
                group['drawdown_duration'] = 0
                group['rolling_sharpe'] = 0
            advanced_features.append(group)
        return pd.concat(advanced_features, ignore_index=True)

    @staticmethod
    def _calculate_hurst(series):
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

    @staticmethod
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

    def calculate_all_features(self, df):
        print("Starting probabilistic feature engineering...")
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for probabilistic features")
        df = self.calculate_garch_features(df)
        df = self.calculate_distributional_features(df)
        df = self.calculate_correlation_features(df)
        df = self.calculate_regime_features(df)
        df = self.calculate_advanced_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        print("Probabilistic feature engineering completed.")
        return df


class AdvancedCopulaEngine:
    """Copula parameters, extreme value theory, network features, spectral."""
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

    def calculate_copula_features(self, df):
        print("Calculating copula dependency features...")
        key_metrics = ['HR', 'RBI', 'R', 'H', 'BB', 'SB', 'calculated_dk_fpts']
        copula_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            copula_dict = {}
            for i, metric1 in enumerate(key_metrics):
                for j, metric2 in enumerate(key_metrics[i+1:], i+1):
                    if metric1 in group.columns and metric2 in group.columns:
                        gaussian_params, clayton_params = [], []
                        upper_tail_deps, lower_tail_deps = [], []
                        for idx in range(len(group)):
                            start_idx = max(0, idx - self.lookback_window)
                            w1 = group[metric1].iloc[start_idx:idx+1]
                            w2 = group[metric2].iloc[start_idx:idx+1]
                            if len(w1) >= self.min_observations:
                                gaussian_params.append(self.gaussian_copula_param(w1.values, w2.values))
                                clayton_params.append(self.clayton_copula_param(w1.values, w2.values))
                                upper_tail_deps.append(self.tail_dependence_coefficient(w1.values, w2.values, 'upper'))
                                lower_tail_deps.append(self.tail_dependence_coefficient(w1.values, w2.values, 'lower'))
                            else:
                                gaussian_params.append(0.0)
                                clayton_params.append(0.0)
                                upper_tail_deps.append(0.0)
                                lower_tail_deps.append(0.0)
                        copula_dict[f'gaussian_copula_{metric1}_{metric2}'] = gaussian_params
                        copula_dict[f'clayton_copula_{metric1}_{metric2}'] = clayton_params
                        copula_dict[f'upper_tail_dep_{metric1}_{metric2}'] = upper_tail_deps
                        copula_dict[f'lower_tail_dep_{metric1}_{metric2}'] = lower_tail_deps
            for feature, values in copula_dict.items():
                group[feature] = values
            copula_features.append(group)
        return pd.concat(copula_features, ignore_index=True)

    def calculate_extreme_value_features(self, df):
        print("Calculating extreme value theory features...")
        evt_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            fpts = group['calculated_dk_fpts']
            block_size = 7
            block_maxima = []
            for i in range(0, len(fpts), block_size):
                block = fpts.iloc[i:i+block_size]
                if len(block) > 0:
                    block_maxima.append(block.max())
            if len(block_maxima) >= 10:
                block_maxima = np.array(block_maxima)
                location = np.mean(block_maxima)
                scale = np.std(block_maxima)
                shape = stats.skew(block_maxima) / 3
                return_level = location + scale * (-np.log(-np.log(0.95)))
                threshold = np.percentile(fpts, 90)
                exceedance_prob = (fpts > threshold).rolling(window=20).mean()
                group['evt_location'] = location
                group['evt_scale'] = scale
                group['evt_shape'] = shape
                group['evt_return_level'] = return_level
                group['exceedance_prob'] = exceedance_prob.fillna(0)
                group['extreme_value_index'] = shape
                excess = fpts[fpts > threshold] - threshold
                if len(excess) > 0:
                    group['pot_threshold'] = threshold
                    group['pot_excess_mean'] = excess.mean()
                    group['pot_excess_std'] = excess.std()
                else:
                    group['pot_threshold'] = 0
                    group['pot_excess_mean'] = 0
                    group['pot_excess_std'] = 0
            else:
                group['evt_location'] = 0
                group['evt_scale'] = 1
                group['evt_shape'] = 0
                group['evt_return_level'] = 0
                group['exceedance_prob'] = 0
                group['extreme_value_index'] = 0
                group['pot_threshold'] = 0
                group['pot_excess_mean'] = 0
                group['pot_excess_std'] = 0
            evt_features.append(group)
        return pd.concat(evt_features, ignore_index=True)

    def calculate_network_features(self, df):
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
                group['network_centrality'] = 0
                group['network_clustering'] = 0
                group['network_volatility'] = 0
                group['network_efficiency'] = 0
            network_features.append(group)
        return pd.concat(network_features, ignore_index=True)

    def calculate_spectral_features(self, df):
        print("Calculating spectral features...")
        spectral_features = []
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            if len(group) >= 20:
                fpts = group['calculated_dk_fpts'].values
                fft_values = np.fft.fft(fpts)
                freqs = np.fft.fftfreq(len(fpts))
                psd = np.abs(fft_values) ** 2
                dominant_freq_idx = np.argmax(psd[1:len(psd)//2]) + 1
                dominant_frequency = freqs[dominant_freq_idx]
                psd_norm = psd / np.sum(psd)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                spectral_centroid = np.sum(freqs[:len(psd)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
                cumsum_psd = np.cumsum(psd[:len(psd)//2])
                rolloff_threshold = 0.85 * cumsum_psd[-1]
                rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                group['dominant_frequency'] = dominant_frequency
                group['spectral_entropy'] = spectral_entropy
                group['spectral_centroid'] = spectral_centroid
                group['spectral_rolloff'] = spectral_rolloff
                rolling_se = []
                for i in range(len(fpts)):
                    start_idx = max(0, i - 14)
                    window_data = fpts[start_idx:i+1]
                    if len(window_data) >= 7:
                        w_fft = np.fft.fft(window_data)
                        w_psd = np.abs(w_fft) ** 2
                        w_psd_norm = w_psd / np.sum(w_psd)
                        rolling_se.append(-np.sum(w_psd_norm * np.log(w_psd_norm + 1e-10)))
                    else:
                        rolling_se.append(0)
                group['rolling_spectral_entropy'] = rolling_se
            else:
                group['dominant_frequency'] = 0
                group['spectral_entropy'] = 0
                group['spectral_centroid'] = 0
                group['spectral_rolloff'] = 0
                group['rolling_spectral_entropy'] = 0
            spectral_features.append(group)
        return pd.concat(spectral_features, ignore_index=True)

    def calculate_all_advanced_features(self, df):
        print("Starting advanced copula and dependency feature engineering...")
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for advanced features")
        df = self.calculate_copula_features(df)
        df = self.calculate_extreme_value_features(df)
        df = self.calculate_network_features(df)
        df = self.calculate_spectral_features(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        print("Advanced copula and dependency feature engineering completed.")
        return df


# ============================================================================
# Legacy helper functions (kept for backward compat)
# ============================================================================

def calculate_dk_fpts(row):
    singles = pd.to_numeric(row.get('1B', 0), errors='coerce') or 0
    doubles = pd.to_numeric(row.get('2B', 0), errors='coerce') or 0
    triples = pd.to_numeric(row.get('3B', 0), errors='coerce') or 0
    hr = pd.to_numeric(row.get('HR', 0), errors='coerce') or 0
    rbi = pd.to_numeric(row.get('RBI', 0), errors='coerce') or 0
    r = pd.to_numeric(row.get('R', 0), errors='coerce') or 0
    bb = pd.to_numeric(row.get('BB', 0), errors='coerce') or 0
    hbp = pd.to_numeric(row.get('HBP', 0), errors='coerce') or 0
    sb = pd.to_numeric(row.get('SB', 0), errors='coerce') or 0
    return (singles * 3 + doubles * 5 + triples * 8 + hr * 10 +
            rbi * 2 + r * 2 + bb * 2 + hbp * 2 + sb * 5)


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


def clean_infinite_values(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_median = df[col].median()
            fill_value = col_median if pd.notna(col_median) else 0
            df[col] = df[col].fillna(fill_value)
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            df[col] = df[col].fillna('Unknown')
    return df


def load_or_create_label_encoders(df, name_encoder_path, team_encoder_path):
    try:
        if os.path.exists(name_encoder_path):
            le_name = joblib.load(name_encoder_path)
            le_name.fit(df['Name'])
        else:
            raise FileNotFoundError
    except Exception:
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)
    try:
        if os.path.exists(team_encoder_path):
            le_team = joblib.load(team_encoder_path)
            le_team.fit(df['Team'])
        else:
            raise FileNotFoundError
    except Exception:
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    return le_name, le_team


# ============================================================================
# V2 MAIN — Walk-forward validated training pipeline
# ============================================================================

if __name__ == "__main__":
    # Add parent directory to path for local imports
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from config import (
        parse_args, HARDCODED_OPTIMAL_PARAMS,
        LEAGUE_AVG_WOBA, LEAGUE_AVG_HR_FLYBALL, WOBA_WEIGHTS,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, COPULA_PREFIXES,
        PREGAME_NUMERIC_FEATURES, MATCHUP_ALL_FEATURES,
    )
    from feature_engine import (
        calculate_dk_fpts as fe_calculate_dk_fpts,
        engineer_features as fe_engineer_features,
        concurrent_feature_engineering,
        reduce_copula_features,
        apply_marcel_shrinkage,
        prepare_pregame_features,
        EnhancedMLBFinancialStyleEngine as FE_FinancialEngine,
        ProbabilisticMLBEngine as FE_ProbEngine,
        AdvancedCopulaEngine as FE_CopulaEngine,
    )
    from model_builder import (
        build_preprocessor,
        build_ensemble,
        build_feature_selector,
        build_quantile_models,
        optuna_hpo,
        save_shap_importance,
        IndexSelector,
    )
    from validator import (
        WalkForwardValidator,
        per_player_evaluation,
        conformalized_quantile_regression,
        predict_with_quantiles,
    )

    # ---- 1. Parse CLI args ----
    args = parse_args()
    start_time = time.time()

    # Wire parallelism settings into feature_engine
    from feature_engine import set_feature_jobs
    set_feature_jobs(args.feature_jobs)

    # ---- Hardware summary ----
    hw = args.hardware
    print(f"\n{'='*60}")
    print(f"HARDWARE CONFIGURATION")
    print(f"{'='*60}")
    print(f"  CPU cores:      {hw['cpu_count']}")
    print(f"  RAM:            {hw['ram_gb']:.0f} GB" if hw['ram_gb'] else "  RAM:            unknown")
    print(f"  GPU:            {hw['gpu_name']} ({hw['gpu_mem_gb']:.1f} GB)" if hw['gpu_available'] else "  GPU:            none")
    print(f"  n_jobs:         {args.n_jobs}")
    print(f"  feature_jobs:   {args.feature_jobs}")
    print(f"  use_gpu:        {args.use_gpu}")
    print(f"  pregame mode:   {args.pregame}")
    print(f"  matchup engine: {args.matchup}")
    print(f"{'='*60}\n")

    # ---- 2. Resolve paths ----
    system_dir = os.path.dirname(script_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = script_dir  # default: same directory as script

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data path resolution
    data_path = args.data_path
    if data_path is None:
        # Try common locations
        candidates = [
            os.path.join(system_dir, 'data', 'merged_fangraphs_data.csv'),
            os.path.join(system_dir, 'merged_fangraphs_data.csv'),
            os.path.join(script_dir, 'merged_fangraphs_data.csv'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                data_path = candidate
                break
        if data_path is None:
            print("ERROR: No data path specified and no data file found.")
            print("Usage: python training.py --data-path /path/to/merged_fangraphs_data.csv")
            sys.exit(1)

    print(f"Data path: {data_path}")

    # ---- 3. Load data ----
    print("Loading dataset...")
    # Read CSV without forcing dtypes — columns vary across data sources
    df = pd.read_csv(data_path, low_memory=False)
    # Coerce known numeric columns if present
    for col, dt in [('inheritedRunners', 'float64'),
                     ('inheritedRunnersScored', 'float64'),
                     ('catchersInterference', 'int64'),
                     ('salary', 'int64')]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dt)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)
    print(f"Loaded {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")

    # Calculate DK fantasy points if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DK fantasy points...")
        df['calculated_dk_fpts'] = df.apply(fe_calculate_dk_fpts, axis=1)

    # Coerce object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # ---- 4. Label encoders ----
    name_encoder_path = os.path.join(output_dir, 'label_encoder_name_sep2.pkl')
    team_encoder_path = os.path.join(output_dir, 'label_encoder_team_sep2.pkl')
    le_name, le_team = load_or_create_label_encoders(df, name_encoder_path, team_encoder_path)

    # ---- 5. Feature engineering (3 engines + sabermetrics) ----
    print("\n===== FEATURE ENGINEERING =====")

    print("Running financial-style feature engineering...")
    financial_engine = FE_FinancialEngine()
    df = financial_engine.calculate_features(df)

    print("Running probabilistic feature engineering...")
    prob_engine = FE_ProbEngine(lookback_window=30, min_observations=10)
    df = prob_engine.calculate_all_features(df)

    print("Running advanced copula feature engineering...")
    copula_engine = FE_CopulaEngine(lookback_window=30, min_observations=15)
    df = copula_engine.calculate_all_advanced_features(df)

    # Sabermetric features
    chunksize = 50000
    df = concurrent_feature_engineering(df, chunksize)

    # ---- 5b. Post-processing: copula PCA + Marcel shrinkage ----
    df = reduce_copula_features(df, n_components=8)
    df = apply_marcel_shrinkage(df, min_games=50)

    # Final cleanup: replace inf, but do NOT blanket fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    print(f"Feature engineering complete. Shape: {df.shape}")

    # ---- 5c. Matchup engine: opposing pitcher features (if --matchup) ----
    if args.matchup:
        from matchup_engine import prepare_matchup_dataset
        pitcher_csv = args.pitcher_csv
        if pitcher_csv is None:
            # Auto-detect pitcher CSV next to batter data
            data_dir = os.path.dirname(data_path)
            candidate = os.path.join(data_dir, 'merged_fangraphs_data_pitchers.csv')
            if os.path.exists(candidate):
                pitcher_csv = candidate
        cache_dir = args.matchup_cache_dir
        if cache_dir is None:
            cache_dir = os.path.join(output_dir, 'matchup_cache')
        df = prepare_matchup_dataset(
            df,
            pitcher_csv=pitcher_csv,
            cache_dir=cache_dir,
        )
        print(f"After matchup engine: {df.shape}")

    # ---- 6. Pre-game feature preparation (if --pregame) ----
    if args.pregame:
        print(f"\n{'='*60}")
        print(f"PREGAME MODE: Using only pre-game features (leakage-free)")
        print(f"{'='*60}")
        df = prepare_pregame_features(df)

        # Use PREGAME_NUMERIC_FEATURES (only features knowable before first pitch)
        numeric_features = [f for f in PREGAME_NUMERIC_FEATURES if f in df.columns]
        # Also add any pgm_ copula PCA that might exist
        pgm_copula_pcs = [c for c in df.columns if c.startswith('pgm_copula_pc_')]
        numeric_features = list(set(numeric_features + pgm_copula_pcs))
        # Add matchup features if enabled (opp_* and matchup_* are leakage-free)
        if args.matchup:
            matchup_in_df = [f for f in MATCHUP_ALL_FEATURES if f in df.columns]
            # Also grab any opp_*/matchup_* columns that exist but aren't in the static list
            dynamic_matchup = [c for c in df.columns
                               if (c.startswith('opp_') or c.startswith('matchup_'))
                               and c not in numeric_features]
            numeric_features = list(set(numeric_features + matchup_in_df + dynamic_matchup))
            print(f"  Added {len(matchup_in_df) + len(dynamic_matchup)} matchup features")
    else:
        # Legacy mode: all features including same-game (LEAKY, for comparison only)
        print("\nWARNING: Using ALL features including same-game stats. Model will LEAK.")
        numeric_features = [f for f in NUMERIC_FEATURES if f in df.columns]
        copula_pcs = [c for c in df.columns if c.startswith('copula_pc_')]
        numeric_features = list(set(numeric_features + copula_pcs))

    categorical_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    target = df['calculated_dk_fpts'].copy()
    features = df[numeric_features + categorical_features].copy()
    date_series = df['date'].copy()

    # Only require target to be non-NaN; let the preprocessor (SimpleImputer)
    # handle missing feature values.  For pregame mode, also require at least
    # SOME lagged data so we're not predicting from pure imputation.
    valid_mask = target.notna()
    if args.pregame:
        pgm_cols_in_features = [c for c in numeric_features if c.startswith('pgm_') or c.startswith('lag_')]
        if pgm_cols_in_features:
            pgm_any_valid = features[pgm_cols_in_features].notna().any(axis=1)
            valid_mask = valid_mask & pgm_any_valid
        n_dropped = (~valid_mask).sum()
        print(f"Dropped {n_dropped} rows (missing target or no lagged data)")

    df = df[valid_mask].reset_index(drop=True)
    target = target[valid_mask].reset_index(drop=True)
    features = features[valid_mask].reset_index(drop=True)
    date_series = date_series[valid_mask].reset_index(drop=True)
    print(f"Dataset after filtering: {len(df)} rows")

    # ---- 7-9. Validation (pregame vs legacy path) ----
    wf_validator = WalkForwardValidator(n_splits=args.n_splits, gap_days=args.gap_days)

    def build_and_fit_fn(X_train, y_train):
        ensemble = build_ensemble(HARDCODED_OPTIMAL_PARAMS,
                                  use_gpu=args.use_gpu, n_jobs=args.n_jobs)
        ensemble.fit(X_train, y_train)
        return ensemble

    if args.pregame:
        # PREGAME PATH: preprocessor + selector fitted INSIDE each fold
        print("\n===== WALK-FORWARD VALIDATION (PREGAME, IN-FOLD PREPROCESSING) =====")

        metrics_df, oos_predictions_df, (last_preprocessor, last_selector) = \
            wf_validator.validate_pregame(
                df=df,
                features_raw=features,
                y=target.values,
                build_preprocess_fn=build_preprocessor,
                build_model_fn=build_and_fit_fn,
                date_col='date',
                n_features=args.n_features,
            )

        # For pregame, use the last fold's preprocessor/selector to process
        # the full dataset for final model training
        preprocessor = last_preprocessor
        selector = last_selector
        # Sanity check: ensure preprocessor and selector were fit inside fold
        if not hasattr(preprocessor, 'transformers_'):
            raise RuntimeError("Preprocessor returned by validate_pregame() was never fit")
        if not hasattr(selector, 'get_support') and not hasattr(selector, 'indices_'):
            raise RuntimeError("Selector returned by validate_pregame() was never fit")
        features_preprocessed = preprocessor.transform(features)
        features_selected = selector.transform(features_preprocessed)
    else:
        # LEGACY PATH: global preprocessor + selector (leaky but backward-compatible)
        print("\n===== FEATURE SELECTION (LEGACY) =====")
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        features_preprocessed = preprocessor.fit_transform(features)
        print(f"Preprocessed shape: {features_preprocessed.shape}")

        selector = build_feature_selector(
            features_preprocessed, target.values,
            n_features=args.n_features,
        )
        features_selected = selector.transform(features_preprocessed)
        print(f"Selected features shape: {features_selected.shape}")

        print("\n===== WALK-FORWARD VALIDATION (LEGACY) =====")
        metrics_df, oos_predictions_df = wf_validator.validate(
            df=df,
            X=features_selected,
            y=target.values,
            build_and_fit_fn=build_and_fit_fn,
            date_col='date',
        )

    # Save OOS validation results
    oos_path = os.path.join(output_dir, 'oos_validation_results.csv')
    metrics_df.to_csv(oos_path, index=False)
    print(f"OOS validation results saved to {oos_path}")

    # Per-player evaluation
    if not oos_predictions_df.empty:
        player_eval_df = per_player_evaluation(oos_predictions_df, min_samples=10)
        player_eval_path = os.path.join(output_dir, 'player_evaluation.csv')
        player_eval_df.to_csv(player_eval_path, index=False)
        print(f"Player evaluation saved to {player_eval_path}")

    # ---- 10. Optuna HPO (optional) ----
    if not args.skip_hpo:
        print("\n===== OPTUNA HYPERPARAMETER OPTIMIZATION =====")
        # Use last fold's split for HPO
        splits = list(wf_validator.split(df, 'date'))
        if splits:
            last_train_idx, last_test_idx = splits[-1]
            # For pregame, need to re-preprocess these splits
            if args.pregame:
                X_train_raw = features.iloc[last_train_idx]
                X_test_raw = features.iloc[last_test_idx]
                hpo_preprocessor = build_preprocessor(
                    features.select_dtypes(include=[np.number]).columns.tolist(),
                    features.select_dtypes(include=['object', 'category']).columns.tolist(),
                )
                X_train_proc = hpo_preprocessor.fit_transform(X_train_raw)
                X_test_proc = hpo_preprocessor.transform(X_test_raw)
                hpo_selector = build_feature_selector(
                    X_train_proc, target.values[last_train_idx],
                    n_features=args.n_features,
                )
                X_hpo_train = hpo_selector.transform(X_train_proc)
                X_hpo_val = hpo_selector.transform(X_test_proc)
            else:
                X_hpo_train = features_selected[last_train_idx]
                X_hpo_val = features_selected[last_test_idx]
            y_hpo_train = target.values[last_train_idx]
            y_hpo_val = target.values[last_test_idx]
            best_params = optuna_hpo(
                X_hpo_train, y_hpo_train,
                X_hpo_val, y_hpo_val,
                n_trials=args.optuna_trials,
                use_gpu=args.use_gpu,
                n_jobs=args.n_jobs,
            )
        else:
            best_params = HARDCODED_OPTIMAL_PARAMS
    else:
        print("Skipping HPO (--skip-hpo). Using hard-coded params.")
        best_params = HARDCODED_OPTIMAL_PARAMS

    # ---- 11. Train FINAL model on ALL data ----
    print("\n===== FINAL MODEL TRAINING =====")
    print("Training final ensemble on ALL data with best params...")
    final_ensemble = build_ensemble(best_params, use_gpu=args.use_gpu, n_jobs=args.n_jobs)
    final_ensemble.fit(features_selected, target.values)
    print("Final ensemble trained.")

    # Build complete pipeline for serialization
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_ensemble),
    ])

    # ---- 12. Make predictions on full data ----
    print("Making predictions on full dataset...")
    all_predictions = final_ensemble.predict(features_selected)

    mae, mse, r2 = evaluate_model(target, all_predictions)
    print(f"Full-data MAE: {mae:.4f}  MSE: {mse:.4f}  R2: {r2:.4f}")

    if not metrics_df.empty:
        oos_mae = metrics_df['mae'].mean()
        oos_r2 = metrics_df['r2'].mean()
        print(f"OOS MAE: {oos_mae:.4f}  OOS R2: {oos_r2:.4f}")
        if args.pregame:
            print(f"  (Pregame R2 ~0.02-0.15 expected for honest game-level DFS prediction)")

    # ---- 13. Quantile models ----
    print("\n===== QUANTILE REGRESSION =====")
    quantile_models = build_quantile_models(features_selected, target.values,
                                                use_gpu=args.use_gpu, n_jobs=args.n_jobs)

    # CQR calibration using last fold's test set
    cqr_adjustment = 0.0
    if quantile_models is not None:
        splits = list(wf_validator.split(df, 'date'))
        if splits:
            _, cal_idx = splits[-1]
            X_cal = features_selected[cal_idx]
            y_cal = target.values[cal_idx]
            cqr_adjustment, cqr_coverage = conformalized_quantile_regression(
                quantile_models, X_cal, y_cal, alpha=0.10,
            )

    # ---- 14. Save all artifacts ----
    print("\n===== SAVING ARTIFACTS =====")

    # a) Final predictions
    final_results_df = pd.DataFrame({
        'Name': features['Name'].values if 'Name' in features.columns else df['Name'].values,
        'Date': date_series.values,
        'Actual': target.values,
        'Predicted': all_predictions,
    })
    final_predictions_path = os.path.join(output_dir, 'final_predictions.csv')
    final_results_df.to_csv(final_predictions_path, index=False)
    print(f"Saved {final_predictions_path}")

    # b) Predictions with probabilities (from quantile models)
    if quantile_models is not None:
        quantile_preds_df = predict_with_quantiles(quantile_models, features_selected, cqr_adjustment)
        if quantile_preds_df is not None:
            final_with_probs = pd.concat([final_results_df.reset_index(drop=True),
                                          quantile_preds_df.reset_index(drop=True)], axis=1)
            probs_path = os.path.join(output_dir, 'final_predictions_with_probabilities.csv')
            final_with_probs.to_csv(probs_path, index=False)
            print(f"Saved {probs_path}")

            # c) Probability summary
            prob_summary = final_results_df[['Name', 'Date', 'Predicted']].copy()
            prob_summary['prediction_lower_80'] = quantile_preds_df['prediction_lower_80'].values
            prob_summary['prediction_upper_80'] = quantile_preds_df['prediction_upper_80'].values
            prob_summary['prediction_std'] = quantile_preds_df['prediction_std'].values
            prob_summary_path = os.path.join(output_dir, 'probability_summary.csv')
            prob_summary.to_csv(prob_summary_path, index=False)
            print(f"Saved {prob_summary_path}")

    # d) Model pipeline
    model_pipeline_path = os.path.join(output_dir, 'batters_final_ensemble_model_pipeline.pkl')
    joblib.dump(complete_pipeline, model_pipeline_path)
    print(f"Saved {model_pipeline_path}")

    # e) Quantile models
    if quantile_models is not None:
        quantile_path = os.path.join(output_dir, 'quantile_models.pkl')
        joblib.dump({'models': quantile_models, 'cqr_adjustment': cqr_adjustment}, quantile_path)
        print(f"Saved {quantile_path}")

    # f) Label encoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("Label encoders saved.")

    # g) Scaler (standalone copy for downstream use)
    scaler_path = os.path.join(output_dir, 'scaler_sep2.pkl')
    # Extract the fitted scaler from the preprocessor
    try:
        fitted_scaler = preprocessor.transformers_[0][1].named_steps['scaler']
        joblib.dump(fitted_scaler, scaler_path)
        print(f"Saved {scaler_path}")
    except Exception as e:
        print(f"Could not save standalone scaler: {e}")

    # h) Feature importances (SHAP-based)
    feature_importance_csv = os.path.join(output_dir, 'feature_importances.csv')
    feature_importance_plot = os.path.join(output_dir, 'feature_importances_plot.png')

    # Get feature names for the selected subset
    try:
        num_feat_names = numeric_features
        cat_feat_names = list(
            preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        )
        all_feat_names = num_feat_names + cat_feat_names
        selected_feat_names = [all_feat_names[i] for i in selector.selected_indices
                               if i < len(all_feat_names)]
    except Exception:
        selected_feat_names = [f'feature_{i}' for i in range(features_selected.shape[1])]

    # Use the LightGBM base model for SHAP if available
    try:
        base_model = final_ensemble.named_estimators_.get('lgbm') or final_ensemble.named_estimators_.get('gbr')
    except Exception:
        base_model = final_ensemble

    sample_size = min(5000, features_selected.shape[0])
    save_shap_importance(
        base_model,
        features_selected[:sample_size],
        selected_feat_names,
        feature_importance_csv,
        feature_importance_plot,
    )

    # i) Full dataset with features
    final_dataset_path = os.path.join(output_dir, 'battersfinal_dataset_with_features.csv')
    df.to_csv(final_dataset_path, index=False)
    print(f"Saved {final_dataset_path}")

    # j) Per-player quant profile (most recent observation of high-signal quant columns)
    QUANT_PROFILE_COLS = [
        'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
        'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
        'entropy', 'hurst_exponent', 'rolling_sharpe',
        'avg_player_correlation', 'correlation_volatility',
        'evt_return_level', 'exceedance_prob',
    ]
    available_quant_cols = [c for c in QUANT_PROFILE_COLS if c in df.columns]
    if available_quant_cols:
        quant_profile = (
            df.sort_values('date', ascending=False)
              .drop_duplicates('Name', keep='first')
              [['Name'] + available_quant_cols]
              .copy()
        )
        quant_profile_path = os.path.join(output_dir, 'player_quant_profile.csv')
        quant_profile.to_csv(quant_profile_path, index=False)
        print(f"Saved {quant_profile_path} ({len(quant_profile)} players, {len(available_quant_cols)} quant columns)")
    else:
        print("No quant profile columns found in data; skipping player_quant_profile.csv")

    # ---- Summary ----
    total_time = time.time() - start_time
    mode_label = "PREGAME (leakage-free)" if args.pregame else "LEGACY (all features)"
    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE V2 COMPLETE — {mode_label}")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Data: {len(df)} rows, {df.shape[1]} columns")
    print(f"Features selected: {features_selected.shape[1]}")
    print(f"Full-data MAE: {mae:.4f}")
    if not metrics_df.empty:
        print(f"OOS MAE (mean): {metrics_df['mae'].mean():.4f}")
        print(f"OOS R2 (mean):  {metrics_df['r2'].mean():.4f}")
        if args.pregame:
            print(f"\nPREGAME REALITY CHECK:")
            print(f"  R2 ~0.02-0.15 = good honest model (game-level DFS is high-variance)")
            print(f"  R2 > 0.50 = likely still leaking same-game data")
            print(f"  Baseline MAE: {metrics_df['baseline_mae'].mean():.4f} (train mean)")
    print(f"Output dir: {output_dir}")
    print(f"Artifacts: pipeline, quantile models, predictions, feature importances")
    print(f"{'='*60}")
