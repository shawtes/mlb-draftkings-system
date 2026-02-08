import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
import torch
print(torch.cuda.is_available())
import xgboost as xgb
print(xgb.__version__)
print(xgb.get_config())

print(xgb.Booster) # Should not error if installed correctly
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor

# Suppress specific pandas warnings related to runtime operations
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings
import multiprocessing
import os
import torch
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
from scipy.special import factorial
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
        
        # --- Preprocessing ---
        # Ensure date is datetime and sort
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])

        # Standardize opportunity columns
        if 'PA' not in df.columns and 'PA.1' in df.columns:
            df['PA'] = df['PA.1']
        if 'AB' not in df.columns and 'AB.1' in df.columns:
            df['AB'] = df['AB.1']
            
        # Ensure base columns exist
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                print(f"Warning: Column '{col}' not found. Initialized with 0.")

        # Group by player
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # --- Momentum Features (like RSI, MACD) ---
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    # Rolling means (SMA)
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    # Exponential rolling means (EMA)
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    # Rate of Change (Momentum)
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                # Performance vs moving average
                if f'{col}_sma_28' in new_features:
                    new_features[f'{col}_vs_sma_28'] = (group[col] / new_features[f'{col}_sma_28']) - 1
            
            # --- Volatility Features (like Bollinger Bands) ---
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                new_features[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                if mean is not None and not mean.empty:
                    new_features[f'dk_fpts_band_width_{window}'] = (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}']) / mean
                    new_features[f'dk_fpts_band_position_{window}'] = (group['calculated_dk_fpts'] - new_features[f'dk_fpts_lower_band_{window}']) / (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}'])

            # --- "Volume" (PA/AB) based Features ---
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    new_features[f'{vol_col}_ratio'] = group[vol_col] / new_features[f'{vol_col}_roll_mean_28']
                    new_features[f'dk_fpts_{vol_col}_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group[vol_col])

            # --- Interaction / Ratio Features ---
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns and 'PA' in group.columns and group['PA'].sum() > 0:
                    new_features[f'{col}_per_pa'] = group[col] / group['PA']
            
            # --- Temporal Features ---
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)
            new_features['day_of_week_sin'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
            new_features['day_of_week_cos'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)

            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        # Final cleanup        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df

class ProbabilisticMLBEngine:
    """
    Advanced probabilistic feature engineering for MLB fantasy point prediction.
    Includes GARCH volatility modeling, copula dependencies, and statistical features.
    """
    
    def __init__(self, lookback_window=30, min_observations=10):
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        
    def calculate_garch_features(self, df):
        """Calculate GARCH volatility features for each player"""
        print("Calculating GARCH volatility features...")
        
        garch_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            if len(group) < self.min_observations:
                # Not enough data for GARCH, use simplified volatility
                group['garch_volatility'] = group['calculated_dk_fpts'].rolling(window=5).std().fillna(0)
                group['garch_conditional_volatility'] = group['garch_volatility']
                group['volatility_regime'] = 0  # Low volatility regime
            else:
                # Calculate returns with robust handling of invalid values
                fpts_clean = group['calculated_dk_fpts'].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
                returns = fpts_clean.pct_change().dropna()
                
                # Additional validation: remove infinite and NaN values from returns
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Check if we have valid returns for GARCH
                if ARCH_AVAILABLE and len(returns) >= 20 and not returns.empty and returns.var() > 1e-10:
                    try:
                        # Ensure returns are finite and have reasonable variance
                        if np.all(np.isfinite(returns)) and returns.std() > 1e-6:
                            # Scale returns to avoid numerical issues
                            scaled_returns = returns * 100
                            
                            # Fit GARCH(1,1) model with robust settings
                            garch_model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=True)
                            garch_fitted = garch_model.fit(disp='off', show_warning=False)
                            
                            # Get conditional volatility
                            conditional_vol = garch_fitted.conditional_volatility / 100
                            
                            # Ensure conditional volatility is finite
                            conditional_vol = conditional_vol.replace([np.inf, -np.inf], np.nan).fillna(returns.std())
                            
                            # Pad with NaN for alignment
                            vol_series = pd.Series(index=group.index, dtype=float)
                            vol_series.iloc[1:len(conditional_vol)+1] = conditional_vol.values
                            group['garch_volatility'] = vol_series.bfill().fillna(returns.std())
                            group['garch_conditional_volatility'] = vol_series.bfill().fillna(returns.std())
                            
                            # Volatility regime (high/low based on historical percentiles)
                            vol_percentile = conditional_vol.rolling(window=10).rank(pct=True).fillna(0.5)
                            regime_series = pd.Series(index=group.index, dtype=int)
                            regime_series.iloc[1:len(vol_percentile)+1] = (vol_percentile > 0.7).astype(int)
                            group['volatility_regime'] = regime_series.fillna(0)
                        else:
                            raise ValueError("Returns contain invalid values or have insufficient variance")
                        
                    except Exception as e:
                        print(f"GARCH fitting failed for {name}: {e}")
                        # Fallback to rolling volatility
                        fallback_vol = fpts_clean.rolling(window=10).std().fillna(0)
                        group['garch_volatility'] = fallback_vol
                        group['garch_conditional_volatility'] = fallback_vol
                        group['volatility_regime'] = 0
                else:
                    # Simplified volatility features
                    fallback_vol = fpts_clean.rolling(window=10).std().fillna(0)
                    group['garch_volatility'] = fallback_vol
                    group['garch_conditional_volatility'] = fallback_vol
                    
                    # Simple volatility regime based on rolling std
                    rolling_vol = fpts_clean.rolling(window=10).std().fillna(0)
                    vol_threshold = rolling_vol.quantile(0.7) if not rolling_vol.empty else 0
                    group['volatility_regime'] = (rolling_vol > vol_threshold).astype(int)
            
            garch_features.append(group)
        
        return pd.concat(garch_features, ignore_index=True)
    
    def calculate_distributional_features(self, df):
        """Calculate distributional features for each player"""
        print("Calculating distributional features...")
        
        dist_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            # Rolling statistical moments
            for window in [7, 14, 30]:
                group[f'skewness_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(lambda x: skew(x) if len(x) >= 3 else 0)
                group[f'kurtosis_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(lambda x: kurtosis(x) if len(x) >= 4 else 0)
                
                # Value at Risk (VaR) and Expected Shortfall (ES)
                group[f'var_95_{window}d'] = group['calculated_dk_fpts'].rolling(window).quantile(0.05)
                group[f'var_99_{window}d'] = group['calculated_dk_fpts'].rolling(window).quantile(0.01)
                group[f'expected_shortfall_{window}d'] = group['calculated_dk_fpts'].rolling(window).apply(
                    lambda x: x[x <= x.quantile(0.05)].mean() if len(x) >= 5 else x.min()
                )
            
            # Tail risk measures
            def safe_tail_ratio(x):
                if len(x) < 10:
                    return 1.0
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        q95 = x.quantile(0.95)
                        q50 = x.quantile(0.5)
                        q05 = x.quantile(0.05)
                        denominator = q50 - q05
                        if abs(denominator) < 1e-10:
                            return 1.0
                        result = (q95 - q50) / denominator
                        return result if np.isfinite(result) else 1.0
                except:
                    return 1.0
            
            group['tail_ratio'] = group['calculated_dk_fpts'].rolling(30).apply(safe_tail_ratio)
            
            # Probability of exceeding thresholds (dynamic)
            for threshold in [5, 10, 15, 20]:
                group[f'prob_exceed_{threshold}'] = group['calculated_dk_fpts'].rolling(30).apply(
                    lambda x: (x > threshold).mean() if len(x) >= 5 else 0
                )
            
            dist_features.append(group)
        
        return pd.concat(dist_features, ignore_index=True)
    
    def calculate_correlation_features(self, df):
        """Calculate dynamic correlation features between players"""
        print("Calculating correlation features...")
        
        # Get top players by total fantasy points for correlation analysis
        top_players = df.groupby('Name')['calculated_dk_fpts'].sum().nlargest(50).index.tolist()
        
        correlation_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date').reset_index(drop=True)
            
            if name in top_players:
                # Calculate correlations with other top players
                correlations = []
                
                for other_player in top_players[:10]:  # Top 10 for efficiency
                    if other_player != name:
                        try:
                            other_group = df[df['Name'] == other_player].sort_values('date').reset_index(drop=True)
                            
                            if len(other_group) > 0:
                                # Align the data by using common date range
                                common_dates = set(group['date']).intersection(set(other_group['date']))
                                
                                if len(common_dates) >= 10:  # Need at least 10 common dates
                                    # Create aligned series for correlation calculation
                                    player_series = group.set_index('date')['calculated_dk_fpts']
                                    other_series = other_group.set_index('date')['calculated_dk_fpts']
                                    
                                    # Remove duplicates by keeping the last occurrence
                                    player_series = player_series[~player_series.index.duplicated(keep='last')]
                                    other_series = other_series[~other_series.index.duplicated(keep='last')]
                                    
                                    # Calculate rolling correlation only on common dates
                                    rolling_corr = player_series.rolling(window=min(20, len(common_dates))).corr(other_series)
                                    
                                    # Create a proper correlation array for this player
                                    corr_values = np.zeros(len(group))
                                    
                                    # Map correlation values to the group's date positions
                                    for i, date in enumerate(group['date']):
                                        if date in rolling_corr.index:
                                            corr_values[i] = rolling_corr.loc[date] if not pd.isna(rolling_corr.loc[date]) else 0
                                    
                                    correlations.append(corr_values)
                                else:
                                    # Not enough common dates, use zeros
                                    correlations.append(np.zeros(len(group)))
                                
                        except Exception as e:
                            print(f"Error calculating correlation between {name} and {other_player}: {e}")
                            # Use zeros as fallback
                            correlations.append(np.zeros(len(group)))
                
                if correlations and len(correlations) > 0:
                    try:
                        # Convert to numpy array for easier handling
                        corr_array = np.array(correlations).T  # Transpose to have players as columns
                        
                        # Calculate average correlation
                        avg_correlation = np.mean(corr_array, axis=1) if corr_array.shape[1] > 0 else np.zeros(len(group))
                        group['avg_player_correlation'] = avg_correlation
                        
                        # Calculate correlation volatility
                        corr_vol = np.std(corr_array, axis=1) if corr_array.shape[1] > 1 else np.zeros(len(group))
                        group['correlation_volatility'] = corr_vol
                    except Exception as e:
                        print(f"Error in correlation aggregation for {name}: {e}")
                        group['avg_player_correlation'] = 0
                        group['correlation_volatility'] = 0
                else:
                    group['avg_player_correlation'] = 0
                    group['correlation_volatility'] = 0
            else:
                group['avg_player_correlation'] = 0
                group['correlation_volatility'] = 0
            
            correlation_features.append(group)
        
        return pd.concat(correlation_features, ignore_index=True)
    
    def calculate_regime_features(self, df):
        """Calculate regime-based features"""
        print("Calculating regime features...")
        
        regime_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            # Performance regimes based on moving averages
            short_ma = group['calculated_dk_fpts'].rolling(window=5).mean()
            long_ma = group['calculated_dk_fpts'].rolling(window=20).mean()
            
            # Regime indicators
            group['bull_regime'] = (short_ma > long_ma).astype(int)
            
            # Regime strength with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                regime_strength = (short_ma - long_ma) / long_ma
                group['regime_strength'] = regime_strength.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Momentum regimes with robust binning
            momentum = group['calculated_dk_fpts'].pct_change(5)
            
            # Check if momentum has sufficient variation for binning
            if len(momentum.dropna()) > 3 and momentum.std() > 0 and not momentum.isna().all():
                try:
                    # Use quantile-based binning for more robust results
                    momentum_clean = momentum.dropna()
                    if len(momentum_clean.unique()) >= 3:
                        momentum_regime = pd.cut(momentum, bins=3, labels=[0, 1, 2], duplicates='drop')
                        group['momentum_regime'] = momentum_regime.astype(float)
                    else:
                        # Not enough unique values, use simple thresholding
                        momentum_median = momentum.median()
                        group['momentum_regime'] = np.where(
                            momentum > momentum_median, 2,
                            np.where(momentum < momentum_median, 0, 1)
                        ).astype(float)
                except (ValueError, TypeError):
                    # Fallback to simple thresholding based on sign
                    group['momentum_regime'] = np.where(
                        momentum > 0, 2,
                        np.where(momentum < 0, 0, 1)
                    ).astype(float)
            else:
                # Default to neutral regime (1) when insufficient data
                group['momentum_regime'] = 1.0
            
            # Consistency regime with safe calculations
            rolling_std = group['calculated_dk_fpts'].rolling(10).std()
            rolling_mean = group['calculated_dk_fpts'].rolling(10).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rolling_cv = rolling_std / rolling_mean
                rolling_cv = rolling_cv.fillna(0).replace([np.inf, -np.inf], 0)
                
                # Safe quantile calculation
                if len(rolling_cv.dropna()) > 0:
                    cv_33_quantile = rolling_cv.quantile(0.33)
                    group['consistency_regime'] = (rolling_cv < cv_33_quantile).astype(int)
                else:
                    group['consistency_regime'] = 0
            
            regime_features.append(group)
        
        return pd.concat(regime_features, ignore_index=True)
    
    def calculate_advanced_features(self, df):
        """Calculate advanced probabilistic features"""
        print("Calculating advanced probabilistic features...")
        
        advanced_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            if len(group) >= self.min_observations:
                # Information theoretic measures
                group['entropy'] = group['calculated_dk_fpts'].rolling(20).apply(
                    lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) >= 5 else 0
                )
                
                # Hurst exponent (simplified)
                group['hurst_exponent'] = group['calculated_dk_fpts'].rolling(30).apply(
                    lambda x: self._calculate_hurst(x) if len(x) >= 10 else 0.5
                )
                
                # Drawdown features
                cumulative = group['calculated_dk_fpts'].cumsum()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                
                group['max_drawdown'] = drawdown.rolling(20).min()
                group['current_drawdown'] = drawdown
                group['drawdown_duration'] = self._calculate_drawdown_duration(drawdown)
                
                # Sharpe ratio (risk-adjusted return)
                returns = group['calculated_dk_fpts'].pct_change()
                returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rolling_mean = returns.rolling(20).mean()
                    rolling_std = returns.rolling(20).std()
                    group['rolling_sharpe'] = rolling_mean / rolling_std
                    group['rolling_sharpe'] = group['rolling_sharpe'].fillna(0).replace([np.inf, -np.inf], 0)
                
            else:
                # Default values for insufficient data
                group['entropy'] = 0
                group['hurst_exponent'] = 0.5
                group['max_drawdown'] = 0
                group['current_drawdown'] = 0
                group['drawdown_duration'] = 0
                group['rolling_sharpe'] = 0
            
            advanced_features.append(group)
        
        return pd.concat(advanced_features, ignore_index=True)
    
    def _calculate_hurst(self, series):
        """Calculate Hurst exponent (simplified R/S method)"""
        try:
            if len(series) < 10:
                return 0.5
            
            # Convert to numpy array and remove NaN
            data = np.array(series).flatten()
            data = data[~np.isnan(data)]
            
            if len(data) < 10:
                return 0.5
            
            # Calculate mean-adjusted series
            mean_adj = data - np.mean(data)
            
            # Calculate cumulative sum
            cumsum = np.cumsum(mean_adj)
            
            # Calculate range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Calculate standard deviation
            S = np.std(data)
            
            if S == 0:
                return 0.5
            
            # Hurst exponent (simplified)
            hurst = np.log(R/S) / np.log(len(data))
            
            # Bound between 0 and 1
            return max(0, min(1, hurst))
        
        except:
            return 0.5
    
    def _calculate_drawdown_duration(self, drawdown_series):
        """Calculate duration of current drawdown"""
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
        """Calculate all probabilistic features"""
        print("Starting probabilistic feature engineering...")
        
        # Ensure we have the required column
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for probabilistic features")
        
        # Apply all feature calculations
        df = self.calculate_garch_features(df)
        df = self.calculate_distributional_features(df)
        df = self.calculate_correlation_features(df)
        df = self.calculate_regime_features(df)
        df = self.calculate_advanced_features(df)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        
        print("Probabilistic feature engineering completed.")
        return df

class AdvancedCopulaEngine:
    """
    Advanced copula and dependency modeling for MLB fantasy points.
    Includes copula parameters, extreme value theory, and network features.
    """
    
    def __init__(self, lookback_window=30, min_observations=15):
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        
    def gaussian_copula_param(self, x, y):
        """Estimate Gaussian copula parameter using Kendall's tau"""
        try:
            if len(x) < 5 or len(y) < 5:
                return 0.0
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 5:
                return 0.0
                
            # Calculate Kendall's tau
            tau, _ = kendalltau(x_clean, y_clean)
            
            # Convert to Gaussian copula parameter
            rho = np.sin(np.pi * tau / 2)
            return rho if not np.isnan(rho) else 0.0
        except:
            return 0.0
    
    def clayton_copula_param(self, x, y):
        """Estimate Clayton copula parameter"""
        try:
            if len(x) < 5 or len(y) < 5:
                return 0.0
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 5:
                return 0.0
                
            # Calculate Kendall's tau
            tau, _ = kendalltau(x_clean, y_clean)
            
            # Convert to Clayton copula parameter
            if tau <= 0:
                return 0.0
            theta = 2 * tau / (1 - tau)
            return max(0, theta) if not np.isnan(theta) else 0.0
        except:
            return 0.0
    
    def tail_dependence_coefficient(self, x, y, tail='upper'):
        """Calculate tail dependence coefficient"""
        try:
            if len(x) < 10 or len(y) < 10:
                return 0.0
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 10:
                return 0.0
            
            # Convert to ranks (empirical distribution)
            x_ranks = stats.rankdata(x_clean) / len(x_clean)
            y_ranks = stats.rankdata(y_clean) / len(y_clean)
            
            # Calculate tail dependence
            threshold = 0.9 if tail == 'upper' else 0.1
            
            if tail == 'upper':
                condition = (x_ranks > threshold) & (y_ranks > threshold)
                tail_dep = np.sum(condition) / np.sum(x_ranks > threshold)
            else:
                condition = (x_ranks < threshold) & (y_ranks < threshold)
                tail_dep = np.sum(condition) / np.sum(x_ranks < threshold)
            
            return tail_dep if not np.isnan(tail_dep) else 0.0
        except:
            return 0.0
    
    def calculate_copula_features(self, df):
        """Calculate copula-based dependency features"""
        print("Calculating copula dependency features...")
        
        # Key performance metrics for copula analysis
        key_metrics = ['HR', 'RBI', 'R', 'H', 'BB', 'SB', 'calculated_dk_fpts']
        
        copula_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            # Initialize copula features
            copula_dict = {}
            
            # Calculate copula parameters between different stats
            for i, metric1 in enumerate(key_metrics):
                for j, metric2 in enumerate(key_metrics[i+1:], i+1):
                    if metric1 in group.columns and metric2 in group.columns:
                        # Rolling copula parameters
                        gaussian_params = []
                        clayton_params = []
                        upper_tail_deps = []
                        lower_tail_deps = []
                        
                        for idx in range(len(group)):
                            start_idx = max(0, idx - self.lookback_window)
                            window_data1 = group[metric1].iloc[start_idx:idx+1]
                            window_data2 = group[metric2].iloc[start_idx:idx+1]
                            
                            if len(window_data1) >= self.min_observations:
                                # Gaussian copula
                                gauss_param = self.gaussian_copula_param(window_data1.values, window_data2.values)
                                gaussian_params.append(gauss_param)
                                
                                # Clayton copula
                                clayton_param = self.clayton_copula_param(window_data1.values, window_data2.values)
                                clayton_params.append(clayton_param)
                                
                                # Tail dependence
                                upper_tail = self.tail_dependence_coefficient(window_data1.values, window_data2.values, 'upper')
                                lower_tail = self.tail_dependence_coefficient(window_data1.values, window_data2.values, 'lower')
                                upper_tail_deps.append(upper_tail)
                                lower_tail_deps.append(lower_tail)
                            else:
                                gaussian_params.append(0.0)
                                clayton_params.append(0.0)
                                upper_tail_deps.append(0.0)
                                lower_tail_deps.append(0.0)
                        
                        # Store rolling features
                        copula_dict[f'gaussian_copula_{metric1}_{metric2}'] = gaussian_params
                        copula_dict[f'clayton_copula_{metric1}_{metric2}'] = clayton_params
                        copula_dict[f'upper_tail_dep_{metric1}_{metric2}'] = upper_tail_deps
                        copula_dict[f'lower_tail_dep_{metric1}_{metric2}'] = lower_tail_deps
            
            # Add to group
            for feature, values in copula_dict.items():
                group[feature] = values
            
            copula_features.append(group)
        
        return pd.concat(copula_features, ignore_index=True)
    
    def calculate_extreme_value_features(self, df):
        """Calculate extreme value theory features"""
        print("Calculating extreme value theory features...")
        
        evt_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            # Extreme value features for fantasy points
            fpts = group['calculated_dk_fpts']
            
            # Block maxima approach
            block_size = 7  # Weekly blocks
            block_maxima = []
            
            for i in range(0, len(fpts), block_size):
                block = fpts.iloc[i:i+block_size]
                if len(block) > 0:
                    block_maxima.append(block.max())
            
            # Fit generalized extreme value distribution parameters (simplified)
            if len(block_maxima) >= 10:
                # Location, scale, shape parameters (simplified estimation)
                block_maxima = np.array(block_maxima)
                
                # Location parameter (mean of block maxima)
                location = np.mean(block_maxima)
                
                # Scale parameter (std of block maxima)
                scale = np.std(block_maxima)
                
                # Shape parameter (simplified using skewness)
                shape = stats.skew(block_maxima) / 3
                
                # Return level (expected maximum in next period)
                return_level = location + scale * (-np.log(-np.log(0.95)))
                
                # Exceedance probability
                threshold = np.percentile(fpts, 90)
                exceedance_prob = (fpts > threshold).rolling(window=20).mean()
                
                # Extreme value index
                extreme_value_index = pd.Series([shape] * len(group), index=group.index)
                
                # Add features
                group['evt_location'] = location
                group['evt_scale'] = scale
                group['evt_shape'] = shape
                group['evt_return_level'] = return_level
                group['exceedance_prob'] = exceedance_prob.fillna(0)
                group['extreme_value_index'] = extreme_value_index
                
                # Peak over threshold features
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
                # Default values for insufficient data
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
        """Calculate network-based features using player interactions"""
        print("Calculating network features...")
        
        # Get top players for network analysis
        top_players = df.groupby('Name')['calculated_dk_fpts'].sum().nlargest(30).index.tolist()
        
        network_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            if name in top_players:
                # Calculate network centrality measures
                centrality_scores = []
                clustering_coeffs = []
                
                for date in group['date']:
                    # Get other players' performance on the same date
                    same_date_players = df[(df['date'] == date) & (df['Name'] != name)]
                    
                    if len(same_date_players) >= 10:
                        # Calculate correlation with other players
                        correlations = []
                        for other_name in top_players[:20]:  # Top 20 for efficiency
                            if other_name != name:
                                other_data = same_date_players[same_date_players['Name'] == other_name]
                                if len(other_data) > 0:
                                    # Simple correlation based on performance difference from mean
                                    player_perf = group[group['date'] == date]['calculated_dk_fpts'].iloc[0] if len(group[group['date'] == date]) > 0 else 0
                                    other_perf = other_data['calculated_dk_fpts'].iloc[0] if len(other_data) > 0 else 0
                                    
                                    # Network connection strength
                                    connection_strength = 1 / (1 + abs(player_perf - other_perf))
                                    correlations.append(connection_strength)
                        
                        # Network centrality (sum of connections)
                        centrality = sum(correlations) / len(correlations) if correlations else 0
                        centrality_scores.append(centrality)
                        
                        # Clustering coefficient (simplified)
                        clustering = np.std(correlations) if correlations else 0
                        clustering_coeffs.append(clustering)
                    else:
                        centrality_scores.append(0)
                        clustering_coeffs.append(0)
                
                # Add network features
                group['network_centrality'] = centrality_scores
                group['network_clustering'] = clustering_coeffs
                
                # Network volatility
                group['network_volatility'] = pd.Series(centrality_scores).rolling(window=5).std().fillna(0).values
                
                # Network efficiency
                group['network_efficiency'] = pd.Series(centrality_scores).rolling(window=5).mean().fillna(0).values
                
            else:
                # Default values for non-top players
                group['network_centrality'] = 0
                group['network_clustering'] = 0
                group['network_volatility'] = 0
                group['network_efficiency'] = 0
            
            network_features.append(group)
        
        return pd.concat(network_features, ignore_index=True)
    
    def calculate_spectral_features(self, df):
        """Calculate spectral analysis features"""
        print("Calculating spectral features...")
        
        spectral_features = []
        
        for name, group in df.groupby('Name'):
            group = group.sort_values('date')
            
            if len(group) >= 20:
                # FFT-based features
                fpts = group['calculated_dk_fpts'].values
                
                # Apply FFT
                fft_values = np.fft.fft(fpts)
                freqs = np.fft.fftfreq(len(fpts))
                
                # Power spectral density
                psd = np.abs(fft_values) ** 2
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(psd[1:len(psd)//2]) + 1
                dominant_frequency = freqs[dominant_freq_idx]
                
                # Spectral entropy
                psd_norm = psd / np.sum(psd)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs[:len(psd)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
                
                # Spectral rolloff
                cumsum_psd = np.cumsum(psd[:len(psd)//2])
                rolloff_threshold = 0.85 * cumsum_psd[-1]
                rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                
                # Add spectral features
                group['dominant_frequency'] = dominant_frequency
                group['spectral_entropy'] = spectral_entropy
                group['spectral_centroid'] = spectral_centroid
                group['spectral_rolloff'] = spectral_rolloff
                
                # Spectral features over time (rolling)
                rolling_spectral_entropy = []
                for i in range(len(fpts)):
                    start_idx = max(0, i - 14)  # 2-week window
                    window_data = fpts[start_idx:i+1]
                    
                    if len(window_data) >= 7:
                        window_fft = np.fft.fft(window_data)
                        window_psd = np.abs(window_fft) ** 2
                        window_psd_norm = window_psd / np.sum(window_psd)
                        window_entropy = -np.sum(window_psd_norm * np.log(window_psd_norm + 1e-10))
                        rolling_spectral_entropy.append(window_entropy)
                    else:
                        rolling_spectral_entropy.append(0)
                
                group['rolling_spectral_entropy'] = rolling_spectral_entropy
                
            else:
                # Default values for insufficient data
                group['dominant_frequency'] = 0
                group['spectral_entropy'] = 0
                group['spectral_centroid'] = 0
                group['spectral_rolloff'] = 0
                group['rolling_spectral_entropy'] = 0
            
            spectral_features.append(group)
        
        return pd.concat(spectral_features, ignore_index=True)
    
    def calculate_all_advanced_features(self, df):
        """Calculate all advanced copula and dependency features"""
        print("Starting advanced copula and dependency feature engineering...")
        
        # Ensure we have the required columns
        if 'calculated_dk_fpts' not in df.columns:
            raise ValueError("calculated_dk_fpts column required for advanced features")
        
        # Apply all advanced feature calculations
        df = self.calculate_copula_features(df)
        df = self.calculate_extreme_value_features(df)
        df = self.calculate_network_features(df)
        df = self.calculate_spectral_features(df)
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        
        print("Advanced copula and dependency feature engineering completed.")
        return df

# Define constants for calculations
# CONFIGURATION: Using hard-coded optimal parameters for fast training
HARDCODED_OPTIMAL_PARAMS = {
    'model__final_estimator__n_estimators': 200,
    'model__final_estimator__max_depth': 6,
    'model__final_estimator__learning_rate': 0.1,
    'model__final_estimator__subsample': 0.8,
    'model__final_estimator__colsample_bytree': 0.9,
    'model__final_estimator__min_child_weight': 3,
    'model__final_estimator__gamma': 0.1,
    'model__final_estimator__reg_alpha': 0.1,
    'model__final_estimator__reg_lambda': 1.0,
}

# League averages for 2020 to 2024
league_avg_wOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315
}

league_avg_HR_FlyBall = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141
}

# wOBA weights for 2020 to 2024
wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

selected_features = [
     'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+', 
    'flyBalls', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
    'singles', 'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI','AB', 'WAR'  
]

engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast', 'Name_encoded', 'team_encoded','wRC+', 'wRAA', 'wOBA',   
]
selected_features += engineered_features

def calculate_dk_fpts(row):
    # Ensure all required columns are present and numeric, defaulting to 0
    # This prevents errors if a stat column is missing from a row
    singles = pd.to_numeric(row.get('1B', 0), errors='coerce')
    doubles = pd.to_numeric(row.get('2B', 0), errors='coerce')
    triples = pd.to_numeric(row.get('3B', 0), errors='coerce')
    hr = pd.to_numeric(row.get('HR', 0), errors='coerce')
    rbi = pd.to_numeric(row.get('RBI', 0), errors='coerce')
    r = pd.to_numeric(row.get('R', 0), errors='coerce')
    bb = pd.to_numeric(row.get('BB', 0), errors='coerce')
    hbp = pd.to_numeric(row.get('HBP', 0), errors='coerce')
    sb = pd.to_numeric(row.get('SB', 0), errors='coerce')

    return (singles * 3 + doubles * 5 + triples * 8 + hr * 10 +
            rbi * 2 + r * 2 + bb * 2 + hbp * 2 + sb * 5)

def engineer_features(df, date_series=None):
    if date_series is None:
        date_series = df['date']
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    # Extract date features
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days

    # Define default values to handle years not present in the lookup tables
    default_wOBA = 0.317  # A reasonable league average
    default_HR_FlyBall = 0.143 # A reasonable league average
    default_wOBA_weights = wOBA_weights[2022] # Use a recent year as default

    # Calculate key statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics (with safe fallbacks for missing years)
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA.get(x['year'], default_wOBA)) / 1.15) * x['AB'] if x['AB'] > 0 else 0, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA.get(x['year'], default_wOBA) * 100) if x['AB'] > 0 and league_avg_wOBA.get(x['year'], default_wOBA) > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) if league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) > 0 else 0, axis=1)

    # Calculate singles
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    # Calculate wOBA using year-specific weights (with safe fallbacks)
    df['wOBA_Statcast'] = df.apply(lambda x: (
        wOBA_weights.get(x['year'], default_wOBA_weights)['BB'] * x.get('BB', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HBP'] * x.get('HBP', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['1B'] * x.get('1B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['2B'] * x.get('2B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['3B'] * x.get('3B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HR'] * x.get('HR', 0)
    ) / (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) if (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) > 0 else 0, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = df.apply(lambda x: (
        x.get('1B', 0) + (2 * x.get('2B', 0)) + (3 * x.get('3B', 0)) + (4 * x.get('HR', 0))
    ) / x.get('AB', 1) if x.get('AB', 1) > 0 else 0, axis=1)

    # Calculate RAR_Statcast (Runs Above Replacement)
    df['RAR_Statcast'] = df['WAR'] * 10 if 'WAR' in df.columns else 0

    # Calculate Offense_Statcast
    df['Offense_Statcast'] = df['wRAA'] + df['BsR'] if 'BsR' in df.columns else df['wRAA']

    # Calculate Dollars_Statcast
    WAR_conversion_factor = 8.0  # Example conversion factor, can be adjusted
    df['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor if 'WAR' in df.columns else 0

    # Calculate WPA/LI_Statcast
    df['WPA/LI_Statcast'] = df['WPA/LI'] if 'WPA/LI' in df.columns else 0

    # Calculate rolling statistics if 'calculated_dk_fpts' is present
    if 'calculated_dk_fpts' in df.columns:
        # Clean the calculated_dk_fpts column first to prevent NaN propagation
        df['calculated_dk_fpts'] = df['calculated_dk_fpts'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).min().fillna(0))
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).max().fillna(0))
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().fillna(0))

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1).fillna(0))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).max().shift(1).fillna(0))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).min().shift(1).fillna(0))

    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def process_chunk(chunk, date_series=None):
    return engineer_features(chunk, date_series)

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    start_time = time.time()
    
    # Use sequential processing when GPU is available to avoid CUDA context conflicts
    if torch.cuda.is_available():
        print("GPU detected - using sequential processing for feature engineering")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_date_series = date_series[i*chunksize:(i+1)*chunksize]
            processed_chunk = process_chunk(chunk, chunk_date_series)
            processed_chunks.append(processed_chunk)
    else:
        max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers for stability
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            processed_chunks = list(executor.map(process_chunk, chunks, [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using default values.")
            default_row = pd.DataFrame([{col: 0 for col in df.columns if col != 'calculated_dk_fpts'}])
            default_row['date'] = prediction_date
            default_row['Name'] = player
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use all available data, up to 45 most recent games
            player_df = player_df.head(20)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

def process_predictions(chunk, pipeline):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    # Clean the features to ensure no infinite or excessively large values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, train_data['Name'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Model evaluation completed.")
    return mae, mse, r2, mape

def calculate_probability_predictions(model, features, thresholds, n_bootstrap=100):
    """
    Calculate probability predictions for exceeding various fantasy point thresholds.
    
    Args:
        model: Trained sklearn model
        features: Feature matrix for prediction
        thresholds: List of fantasy point thresholds to calculate probabilities for
        n_bootstrap: Number of bootstrap samples for uncertainty estimation
    
    Returns:
        Dictionary with probability predictions for each threshold
    """
    print(f"Calculating probability predictions for {len(thresholds)} thresholds...")
    
    # Ensure features are numpy arrays on CPU to avoid device mismatch
    if hasattr(features, 'values'):
        features = features.values
    features = np.asarray(features, dtype=np.float32)
    
    # Get base predictions
    base_predictions = model.predict(features)
    
    # Calculate model residuals for uncertainty estimation
    # For production, we estimate uncertainty using bootstrap sampling
    probabilities = {}
    
    # Bootstrap sampling for uncertainty estimation
    n_samples = features.shape[0]
    bootstrap_predictions = []
    
    print(f"Performing {n_bootstrap} bootstrap samples for uncertainty estimation...")
    
    # Generate bootstrap samples
    for i in range(n_bootstrap):
        # Create bootstrap sample indices
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Get bootstrap predictions (adding small random noise to simulate uncertainty)
        # This is a simplified approach - in production you might use quantile regression
        noise_std = np.std(base_predictions) * 0.1  # 10% of prediction std as noise
        bootstrap_pred = base_predictions + np.random.normal(0, noise_std, n_samples)
        bootstrap_predictions.append(bootstrap_pred)
    
    # Convert to numpy array for easier manipulation
    bootstrap_predictions = np.array(bootstrap_predictions)
    
    # Calculate probabilities for each threshold
    for threshold in thresholds:
        # Count how many bootstrap samples exceed the threshold for each player
        exceed_counts = np.sum(bootstrap_predictions > threshold, axis=0)
        probabilities[f'prob_over_{threshold}'] = exceed_counts / n_bootstrap
    
    # Also calculate prediction intervals
    lower_percentile = np.percentile(bootstrap_predictions, 10, axis=0)
    upper_percentile = np.percentile(bootstrap_predictions, 90, axis=0)
    
    probabilities['prediction_lower_80'] = lower_percentile
    probabilities['prediction_upper_80'] = upper_percentile
    probabilities['prediction_std'] = np.std(bootstrap_predictions, axis=0)
    
    print("Probability predictions calculated successfully.")
    return probabilities

def save_feature_importance(pipeline, output_csv_path, output_plot_path):
    print("Saving feature importances...")
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    selector = pipeline.named_steps['selector']

    # Because of the nested stacking model, we need to access an inner model
    # to get feature importances related to the original features.
    # We will use the GradientBoostingRegressor from the base models.
    # The path is: final_model -> stacking_model -> gb_model
    try:
        stacking_model_estimator = model.named_estimators_['stacking']
        gb_model = stacking_model_estimator.named_estimators_['gb']
        
        if hasattr(gb_model, 'feature_importances_'):
            feature_importances = gb_model.feature_importances_
        else:
            raise AttributeError("The GradientBoostingRegressor model does not have feature_importances_.")
    except (KeyError, AttributeError) as e:
        print(f"Could not retrieve feature importances from the nested GradientBoostingRegressor: {e}")
        print("Falling back to Lasso coefficients as a proxy for importance.")
        try:
            stacking_model_estimator = model.named_estimators_['stacking']
            lasso_model = stacking_model_estimator.named_estimators_['lasso']
            if hasattr(lasso_model, 'coef_'):
                feature_importances = np.abs(lasso_model.coef_)
            else:
                raise AttributeError("The Lasso model does not have coef_.")
        except (KeyError, AttributeError) as e_lasso:
            raise ValueError(f"Could not retrieve feature importances from any base model. Lasso error: {e_lasso}")
    
    # Get all feature names from the preprocessor
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    all_feature_names = np.concatenate([numeric_features, cat_features])
    
    # Get the mask of selected features from the selector
    support_mask = selector.get_support()
    
    # Get the names of ONLY the selected features
    selected_feature_names = all_feature_names[support_mask]

    if len(feature_importances) != len(selected_feature_names):
        raise ValueError(f"The number of feature importances ({len(feature_importances)}) does not match the number of selected feature names ({len(selected_feature_names)}).")
    
    feature_importance_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Importance': feature_importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    feature_importance_df.to_csv(output_csv_path, index=False)
    print(f"Feature importances saved to {output_csv_path}")

    # Plot top 25 features for readability
    top_25_features = feature_importance_df.head(550)

    plt.figure(figsize=(12, 10))
    plt.barh(top_25_features['Feature'], top_25_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 25 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()
    print(f"Feature importance plot saved to {output_plot_path}")

# =============================================================================
# TRAINING CONFIGURATION - PRODUCTION MODE
# =============================================================================
# This script uses hard-coded optimal parameters for fast and reliable training.
# The parameters below have been pre-optimized for MLB DraftKings fantasy point
# prediction and provide consistent performance across different datasets.
# =============================================================================

# Define final_model outside of the main block
base_models = [
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('svr', SVR()),
    ('gb', GradientBoostingRegressor())
]
# ...existing code...

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Use CPU for XGBoost to avoid device mismatch issues with pandas/numpy data
# GPU acceleration can cause device mismatch warnings when working with CPU-based data structures
print("Using CPU for XGBoost to ensure compatibility with pandas/numpy data structures...")
xgb_params = {
    'tree_method': 'hist',
    'device': 'cpu',
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

meta_model = XGBRegressor(**xgb_params)


stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

# Voting Regressor
voting_model = VotingRegressor(
    estimators=base_models
)

# Combine all models into a final ensemble pipeline
ensemble_models = [
    ('stacking', stacking_model),
    ('voting', voting_model)
]
# ...existing code...
final_model = StackingRegressor(
    estimators=ensemble_models,
    final_estimator=XGBRegressor(**xgb_params)
)

# ...existing code...

def clean_infinite_values(df):
    """Clean infinite and NaN values from dataframe with robust error handling"""
    import warnings
    
    # Suppress specific pandas warnings about NaN operations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        # Replace inf and -inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, replace NaN with the mean of the column (or 0 if mean is NaN)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_mean = df[col].mean()
            # If mean is NaN (all values are NaN), use 0 as fallback
            fill_value = col_mean if pd.notna(col_mean) else 0
            df[col] = df[col].fillna(fill_value)
        
        # For non-numeric columns, replace NaN with a placeholder value (e.g., 'Unknown')
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            df[col] = df[col].fillna('Unknown')
    
    return df

# The paths for saving and loading LabelEncoders and Scalers will be set in the main function

def load_or_create_label_encoders(df, name_encoder_path, team_encoder_path):
    # Handle version compatibility by recreating encoders if needed
    try:
        if os.path.exists(name_encoder_path):
            le_name = joblib.load(name_encoder_path)
            # Test if the encoder works with current version
            le_name.fit(df['Name'])
        else:
            raise FileNotFoundError("Name encoder not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new name encoder due to compatibility issues...")
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)

    try:
        if os.path.exists(team_encoder_path):
            le_team = joblib.load(team_encoder_path)
            # Test if the encoder works with current version
            le_team.fit(df['Team'])
        else:
            raise FileNotFoundError("Team encoder not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new team encoder due to compatibility issues...")
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    return le_name, le_team

def load_or_create_scaler(df, numeric_features, scaler_path):
    # Force recreation of scaler to avoid version compatibility issues
    # Remove existing scaler file if it exists
    if os.path.exists(scaler_path):
        print("Removing existing scaler due to version compatibility...")
        os.remove(scaler_path)
    
    scaler = StandardScaler()
    # Don't modify the original dataframe, just fit the scaler
    scaler.fit(df[numeric_features])
    joblib.dump(scaler, scaler_path)
    print("New scaler created and saved.")
    return scaler

def process_fold(fold_data):
    fold, (train_index, test_index), X, y, date_series, numeric_features, categorical_features, final_model = fold_data
    print(f"Processing fold {fold}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Feature engineering is now done on the full dataset beforehand.
    # We will just clean the data within the fold to be safe.
    X_train = clean_infinite_values(X_train.copy())
    X_test = clean_infinite_values(X_test.copy())

    # Prepare preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit preprocessor on training data and transform both train and test
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Feature selection
    selector = SelectKBest(f_regression, k=min(550, X_train_preprocessed.shape[1]))
    X_train_selected = selector.fit_transform(X_train_preprocessed, y_train)
    X_test_selected = selector.transform(X_test_preprocessed)

    # Prepare and fit the model
    model = final_model  # Your stacking model
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Evaluate the model
    mae, mse, r2, mape = evaluate_model(y_test, y_pred)
    
    # Create a DataFrame with predictions, actual values, names, and dates
    results_df = pd.DataFrame({
        'Name': X.iloc[test_index]['Name'],
        'Date': date_series.iloc[test_index],
        'Actual': y_test,
        'Predicted': y_pred
    })

    return mae, mse, r2, mape, results_df

if __name__ == "__main__":
    start_time = time.time()
    
    # Set up proper directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    system_dir = os.path.dirname(script_dir)  # Go up one level to MLB_DRAFTKINGS_SYSTEM
    predictions_dir = os.path.join(system_dir, '2_PREDICTIONS')
    models_dir = os.path.join(system_dir, '3_MODELS')
    analysis_dir = os.path.join(system_dir, '7_ANALYSIS')
    
    # Create output directories if they don't exist
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Debug: Print all directories to confirm they exist
    print(f"Predictions directory: {predictions_dir} (exists: {os.path.exists(predictions_dir)})")
    print(f"Models directory: {models_dir} (exists: {os.path.exists(models_dir)})")
    print(f"Analysis directory: {analysis_dir} (exists: {os.path.exists(analysis_dir)})")
    
    # Additional debug: Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"System directory: {system_dir}")
    
    # SIMPLE SOLUTION: Save everything in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Set the paths for encoders and scalers (save in script directory)
    name_encoder_path = os.path.join(script_directory, 'label_encoder_name_sep2.pkl')
    team_encoder_path = os.path.join(script_directory, 'label_encoder_team_sep2.pkl')
    scaler_path = os.path.join(script_directory, 'scaler_sep2.pkl')
    
    print("Loading dataset...")
    df = pd.read_csv('C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)

    # Calculate calculated_dk_fpts if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("calculated_dk_fpts column not found. Calculating now...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Load or create LabelEncoders
    le_name, le_team = load_or_create_label_encoders(df, name_encoder_path, team_encoder_path)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    # --- New Financial-Style Feature Engineering Step ---
    print("Starting financial-style feature engineering...")
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    print("Financial-style feature engineering complete.")
    
    # --- New Probabilistic Feature Engineering Step ---
    print("Starting probabilistic feature engineering...")
    prob_engine = ProbabilisticMLBEngine(lookback_window=30, min_observations=10)
    df = prob_engine.calculate_all_features(df)
    print("Probabilistic feature engineering complete.")
    
    # --- Advanced Copula and Dependency Feature Engineering ---
    print("Starting advanced copula and dependency feature engineering...")
    advanced_engine = AdvancedCopulaEngine(lookback_window=30, min_observations=15)
    df = advanced_engine.calculate_all_advanced_features(df)
    print("Advanced copula and dependency feature engineering complete.")
    # --- End of New Steps ---

    chunksize = 50000
    df = concurrent_feature_engineering(df, chunksize)

    # --- Centralized Data Cleaning ---
    print("Cleaning final dataset of any infinite or NaN values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # --- End of Cleaning Step ---

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Define the list of all selected and engineered features
    features = selected_features + ['date']

    # Define numeric and categorical features
    numeric_features = [
        'wOBA', 'BABIP', 'ISO',  'wRAA', 'wRC', 'wRC+', 'flyBalls', 'year', 
        'month', 'day',
        'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
        'rolling_mean_fpts_49', 
        'wOBA_Statcast',
        'SLG_Statcast', 'RAR_Statcast', 'Offense_Statcast', 'Dollars_Statcast',
        'WPA/LI_Statcast', 'Off', 'WAR', 'Dol', 'RAR',    
        'RE24', 'REW', 'SLG', 'WPA/LI','AB',
        # Basic probabilistic features
        'garch_volatility', 'garch_conditional_volatility', 'volatility_regime',
        'skewness_7d', 'skewness_14d', 'skewness_30d',
        'kurtosis_7d', 'kurtosis_14d', 'kurtosis_30d',
        'var_95_7d', 'var_95_14d', 'var_95_30d',
        'var_99_7d', 'var_99_14d', 'var_99_30d',
        'expected_shortfall_7d', 'expected_shortfall_14d', 'expected_shortfall_30d',
        'tail_ratio', 'prob_exceed_5', 'prob_exceed_10', 'prob_exceed_15', 'prob_exceed_20',
        'avg_player_correlation', 'correlation_volatility',
        'bull_regime', 'regime_strength', 'momentum_regime', 'consistency_regime',
        'entropy', 'hurst_exponent', 'max_drawdown', 'current_drawdown', 
        'drawdown_duration', 'rolling_sharpe',
        # Advanced copula and dependency features
        'gaussian_copula_HR_RBI', 'gaussian_copula_HR_R', 'gaussian_copula_HR_H',
        'gaussian_copula_HR_BB', 'gaussian_copula_HR_SB', 'gaussian_copula_HR_calculated_dk_fpts',
        'gaussian_copula_RBI_R', 'gaussian_copula_RBI_H', 'gaussian_copula_RBI_BB',
        'gaussian_copula_RBI_SB', 'gaussian_copula_RBI_calculated_dk_fpts',
        'gaussian_copula_R_H', 'gaussian_copula_R_BB', 'gaussian_copula_R_SB',
        'gaussian_copula_R_calculated_dk_fpts', 'gaussian_copula_H_BB',
        'gaussian_copula_H_SB', 'gaussian_copula_H_calculated_dk_fpts',
        'gaussian_copula_BB_SB', 'gaussian_copula_BB_calculated_dk_fpts',
        'gaussian_copula_SB_calculated_dk_fpts',
        'clayton_copula_HR_RBI', 'clayton_copula_HR_R', 'clayton_copula_HR_H',
        'clayton_copula_HR_BB', 'clayton_copula_HR_SB', 'clayton_copula_HR_calculated_dk_fpts',
        'clayton_copula_RBI_R', 'clayton_copula_RBI_H', 'clayton_copula_RBI_BB',
        'clayton_copula_RBI_SB', 'clayton_copula_RBI_calculated_dk_fpts',
        'clayton_copula_R_H', 'clayton_copula_R_BB', 'clayton_copula_R_SB',
        'clayton_copula_R_calculated_dk_fpts', 'clayton_copula_H_BB',
        'clayton_copula_H_SB', 'clayton_copula_H_calculated_dk_fpts',
        'clayton_copula_BB_SB', 'clayton_copula_BB_calculated_dk_fpts',
        'clayton_copula_SB_calculated_dk_fpts',
        'upper_tail_dep_HR_RBI', 'upper_tail_dep_HR_R', 'upper_tail_dep_HR_H',
        'upper_tail_dep_HR_BB', 'upper_tail_dep_HR_SB', 'upper_tail_dep_HR_calculated_dk_fpts',
        'upper_tail_dep_RBI_R', 'upper_tail_dep_RBI_H', 'upper_tail_dep_RBI_BB',
        'upper_tail_dep_RBI_SB', 'upper_tail_dep_RBI_calculated_dk_fpts',
        'upper_tail_dep_R_H', 'upper_tail_dep_R_BB', 'upper_tail_dep_R_SB',
        'upper_tail_dep_R_calculated_dk_fpts', 'upper_tail_dep_H_BB',
        'upper_tail_dep_H_SB', 'upper_tail_dep_H_calculated_dk_fpts',
        'upper_tail_dep_BB_SB', 'upper_tail_dep_BB_calculated_dk_fpts',
        'upper_tail_dep_SB_calculated_dk_fpts',
        'lower_tail_dep_HR_RBI', 'lower_tail_dep_HR_R', 'lower_tail_dep_HR_H',
        'lower_tail_dep_HR_BB', 'lower_tail_dep_HR_SB', 'lower_tail_dep_HR_calculated_dk_fpts',
        'lower_tail_dep_RBI_R', 'lower_tail_dep_RBI_H', 'lower_tail_dep_RBI_BB',
        'lower_tail_dep_RBI_SB', 'lower_tail_dep_RBI_calculated_dk_fpts',
        'lower_tail_dep_R_H', 'lower_tail_dep_R_BB', 'lower_tail_dep_R_SB',
        'lower_tail_dep_R_calculated_dk_fpts', 'lower_tail_dep_H_BB',
        'lower_tail_dep_H_SB', 'lower_tail_dep_H_calculated_dk_fpts',
        'lower_tail_dep_BB_SB', 'lower_tail_dep_BB_calculated_dk_fpts',
        'lower_tail_dep_SB_calculated_dk_fpts',
        # Extreme value theory features
        'evt_location', 'evt_scale', 'evt_shape', 'evt_return_level',
        'exceedance_prob', 'extreme_value_index', 'pot_threshold',
        'pot_excess_mean', 'pot_excess_std',
        # Network features
        'network_centrality', 'network_clustering', 'network_volatility', 'network_efficiency',
        # Spectral features
        'dominant_frequency', 'spectral_entropy', 'spectral_centroid', 
        'spectral_rolloff', 'rolling_spectral_entropy'
    ]

    categorical_features = ['Name', 'Team']

    # Debug prints to check feature lists and data types
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Data types in DataFrame:")
    print(df.dtypes)

    # Load or create Scaler
    scaler = load_or_create_scaler(df, numeric_features, scaler_path)

    # Define transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor that includes both numeric and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Before fitting the preprocessor
    print("Preparing features for preprocessing...")
    
    # It's important to drop the target from the features AFTER all engineering is complete
    if 'calculated_dk_fpts' in df.columns:
        features = df.drop(columns=['calculated_dk_fpts'])
        target = df['calculated_dk_fpts']
    else:
        # Fallback or error if the target column is still missing
        raise KeyError("'calculated_dk_fpts' not found in DataFrame columns after all processing.")        
    date_series = df['date']
    
    # Clean the data
    features = clean_infinite_values(features.copy())
    
    # Ensure all engineered features are created before selecting them
    features = features[numeric_features + categorical_features]

    # Debug print to check data types in features DataFrame
    print("Data types in features DataFrame before preprocessing:")
    print(features.dtypes)

    # Fit preprocessor and transform features
    print("Fitting preprocessor and transforming features...")
    features_preprocessed = preprocessor.fit_transform(features)
    
    # Feature selection
    print("Performing feature selection...")
    selector = SelectKBest(f_regression, k=min(550, features_preprocessed.shape[1]))
    features_selected = selector.fit_transform(features_preprocessed, target)
    
    print(f"Selected {features_selected.shape[1]} features out of {features_preprocessed.shape[1]}")

    # =============================================================================
    # APPLY HARD-CODED OPTIMAL PARAMETERS AND TRAIN MODEL
    # =============================================================================
    
    print("Using hard-coded optimal parameters for fast training...")
    print("Optimal parameters:")
    for param, value in HARDCODED_OPTIMAL_PARAMS.items():
        print(f"  {param}: {value}")
    
    # Create a complete pipeline
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])
    
    # Apply hard-coded parameters to the pipeline
    complete_pipeline.set_params(**HARDCODED_OPTIMAL_PARAMS)
    
    # Train the model with hard-coded parameters
    print("Training final ensemble model with hard-coded parameters...")
    complete_pipeline.fit(features, target)
    
    # Make predictions using the trained model
    print("Making predictions on training data...")
    all_predictions = complete_pipeline.predict(features)

    # Calculate probability predictions for various DraftKings thresholds
    print("Calculating probability predictions for fantasy point thresholds...")
    probability_thresholds = [5, 10, 15, 20, 25, 30, 35, 40]
    
    # Use the trained model for probability predictions
    probability_predictions = calculate_probability_predictions(
        complete_pipeline.named_steps['model'], 
        complete_pipeline.named_steps['selector'].transform(
            complete_pipeline.named_steps['preprocessor'].transform(features)
        ), 
        probability_thresholds
    )

    # Evaluate the model on training data (for reference)
    mae, mse, r2, mape = evaluate_model(target, all_predictions)
    
    print(f'Training MAE: {mae:.4f}')
    print(f'Training MSE: {mse:.4f}')
    print(f'Training R2: {r2:.4f}')
    print(f'Training MAPE: {mape:.4f}%')

    # Create a DataFrame with all predictions, actual values, names, and dates
    final_results_df = pd.DataFrame({
        'Name': features['Name'],
        'Date': date_series,
        'Actual': target,
        'Predicted': all_predictions
    })
    
    # Add probability predictions to the results DataFrame
    print("Adding probability predictions to results...")
    for key, probs in probability_predictions.items():
        final_results_df[key] = probs
    
    # Save the final results with probability predictions
    # Set file save paths (save in script directory)
    final_predictions_with_probs_path = os.path.join(script_directory, 'final_predictions_with_probabilities.csv')
    final_results_path = os.path.join(script_directory, 'final_results.csv')
    analysis_path = os.path.join(script_directory, 'training_analysis.csv')
    print(f"Saving final predictions with probabilities to: {final_predictions_with_probs_path}")
    
    final_results_df.to_csv(final_predictions_with_probs_path, index=False)
    print("Final predictions with probabilities saved.")
    
    # Also save a summary of probability predictions
    prob_summary = pd.DataFrame({
        'Name': features['Name'],
        'Date': date_series,
        'Predicted_FPTS': all_predictions,
        'Prob_Over_5': probability_predictions['prob_over_5'],
        'Prob_Over_10': probability_predictions['prob_over_10'],
        'Prob_Over_15': probability_predictions['prob_over_15'],
        'Prob_Over_20': probability_predictions['prob_over_20'],
        'Prob_Over_25': probability_predictions['prob_over_25'],
        'Prediction_Lower_80': probability_predictions['prediction_lower_80'],
        'Prediction_Upper_80': probability_predictions['prediction_upper_80'],
        'Prediction_Std': probability_predictions['prediction_std']
    })
    
    prob_summary_path = os.path.join(script_directory, 'probability_summary.csv')
    print(f"Saving probability summary to: {prob_summary_path}")
    
    prob_summary.to_csv(prob_summary_path, index=False)
    print("Probability summary saved.")

    # Save the legacy format for backwards compatibility
    final_predictions_path = os.path.join(script_directory, 'final_predictions.csv')
    print(f"Saving legacy final predictions to: {final_predictions_path}")
    
    final_results_df[['Name', 'Date', 'Actual', 'Predicted']].to_csv(final_predictions_path, index=False)
    print("Legacy final predictions saved.")

    # Save the complete pipeline
    model_pipeline_path = os.path.join(script_directory, 'batters_final_ensemble_model_pipeline.pkl')
    print(f"Saving model pipeline to: {model_pipeline_path}")
    joblib.dump(complete_pipeline, model_pipeline_path)
    print("Final model pipeline saved.")

    # Save the final data to a CSV file
    final_dataset_path = os.path.join(script_directory, 'battersfinal_dataset_with_features.csv')
    print(f"Saving final dataset to: {final_dataset_path}")
    
    df.to_csv(final_dataset_path, index=False)
    print("Final dataset with all features saved.")

    # Save the LabelEncoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("LabelEncoders saved.")

    # Save feature importance with the updated pipeline structure
    feature_importance_csv_path = os.path.join(script_directory, 'feature_importances.csv')
    feature_importance_plot_path = os.path.join(script_directory, 'feature_importances_plot.png')
    save_feature_importance(complete_pipeline, feature_importance_csv_path, feature_importance_plot_path)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")

    # =============================================================================
    # ADVANCED PROBABILISTIC FEATURE ENGINEERING - GARCH, COPULA, ETC.
    # =============================================================================
    # This section adds advanced probabilistic feature engineering using GARCH
    # models for volatility, copula for dependency modeling, and other statistical
    # features for enhanced predictive performance.
    # =============================================================================

    # --- New Advanced Probabilistic Feature Engineering Step ---
    print("Starting advanced probabilistic feature engineering...")
    prob_engine = ProbabilisticMLBEngine(lookback_window=30, min_observations=10)
    
    # Calculate GARCH features
    df = prob_engine.calculate_garch_features(df)
    
    # Calculate distributional features
    df = prob_engine.calculate_distributional_features(df)
    
    # Calculate correlation features
    df = prob_engine.calculate_correlation_features(df)
    
    # Calculate regime features
    df = prob_engine.calculate_regime_features(df)
    
    # Calculate advanced features
    df = prob_engine.calculate_advanced_features(df)
    # --- End of New Step ---

    # --- Centralized Data Cleaning - Final Pass ---
    print("Final cleaning of dataset after advanced feature engineering...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # --- End of Final Cleaning ---

    # Save the final data to a CSV file after advanced feature engineering
    final_dataset_advanced_path = os.path.join(script_directory, 'battersfinal_dataset_with_features_advanced.csv')
    print(f"Saving final dataset with advanced features to: {final_dataset_advanced_path}")
    
    df.to_csv(final_dataset_advanced_path, index=False)
    print("Final dataset with advanced features saved.")

    print("All processing complete.")