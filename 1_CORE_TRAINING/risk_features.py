"""
Risk-Based Feature Engineering
Implements advanced risk metrics and tail risk measures for better risk-adjusted predictions
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
import logging

logger = logging.getLogger(__name__)

class RiskFeatureEngine:
    """
    Advanced risk-based feature engineering
    """
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
    
    def add_risk_features(self, df):
        """
        Add comprehensive risk-based features
        """
        try:
            logger.info("⚠️ Calculating risk-based features...")
            
            # Calculate returns first
            df['returns'] = df['close'].pct_change()
            
            # Standard risk measures
            df = self._add_volatility_measures(df)
            
            # Downside risk measures
            df = self._add_downside_risk_measures(df)
            
            # Value at Risk (VaR) and Expected Shortfall
            df = self._add_var_measures(df)
            
            # Distribution-based features
            df = self._add_distribution_features(df)
            
            # Tail risk measures
            df = self._add_tail_risk_measures(df)
            
            # Maximum drawdown features
            df = self._add_drawdown_features(df)
            
            # Risk-adjusted return measures
            df = self._add_risk_adjusted_returns(df)
            
            # Correlation-based risk
            df = self._add_correlation_risk_features(df)
            
            logger.info("✅ Added risk-based features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding risk features: {str(e)}")
            return df
    
    def _add_volatility_measures(self, df):
        """Add various volatility measures"""
        try:
            returns = df['returns']
            
            # Standard volatility (multiple windows)
            for window in [5, 10, 20, 60]:
                df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
                
                # Volatility percentile ranking
                df[f'vol_percentile_{window}'] = df[f'volatility_{window}'].rolling(252).rank(pct=True)
            
            # EWMA volatility (exponentially weighted)
            for alpha in [0.94, 0.97]:
                df[f'ewma_vol_{int(alpha*100)}'] = returns.ewm(alpha=alpha).std() * np.sqrt(252)
            
            # Realized volatility (sum of squared returns)
            for window in [5, 20]:
                df[f'realized_vol_{window}'] = np.sqrt(
                    (returns ** 2).rolling(window).sum() * 252
                )
            
            # Parkinson estimator (uses high-low range)
            df['parkinson_vol'] = np.sqrt(
                252 * 0.361 * (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
            )
            
            # Garman-Klass estimator (more efficient)
            hl = np.log(df['high'] / df['low'])
            co = np.log(df['close'] / df['open'])
            
            df['garman_klass_vol'] = np.sqrt(
                252 * (0.5 * hl**2 - (2*np.log(2) - 1) * co**2).rolling(20).mean()
            )
            
            # Volatility of volatility
            df['vol_of_vol'] = df['volatility_20'].rolling(20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility measures: {str(e)}")
            return df
    
    def _add_downside_risk_measures(self, df):
        """Add downside risk and semi-deviation measures"""
        try:
            returns = df['returns']
            
            # Downside deviation
            for window in [20, 60]:
                negative_returns = returns.where(returns < 0, 0)
                df[f'downside_deviation_{window}'] = np.sqrt(
                    (negative_returns ** 2).rolling(window).mean() * 252
                )
                
                # Upside deviation  
                positive_returns = returns.where(returns > 0, 0)
                df[f'upside_deviation_{window}'] = np.sqrt(
                    (positive_returns ** 2).rolling(window).mean() * 252
                )
                
                # Upside/downside ratio
                df[f'upside_downside_ratio_{window}'] = (
                    df[f'upside_deviation_{window}'] / 
                    (df[f'downside_deviation_{window}'] + 1e-8)
                )
            
            # Sortino ratio components
            for window in [20, 60]:
                mean_return = returns.rolling(window).mean() * 252
                downside_dev = df[f'downside_deviation_{window}']
                df[f'sortino_ratio_{window}'] = mean_return / (downside_dev + 1e-8)
            
            # Lower partial moments
            for window in [20, 60]:
                # Second lower partial moment (semi-variance)
                target_return = 0  # Use 0 as target
                below_target = returns.where(returns < target_return, 0)
                df[f'lpm2_{window}'] = (below_target ** 2).rolling(window).mean()
                
                # Third lower partial moment (downside skewness)
                df[f'lpm3_{window}'] = (below_target ** 3).rolling(window).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating downside risk measures: {str(e)}")
            return df
    
    def _add_var_measures(self, df):
        """Add Value at Risk and Expected Shortfall measures"""
        try:
            returns = df['returns']
            
            for window in [20, 60]:
                for confidence in self.confidence_levels:
                    # Historical VaR
                    var_hist = returns.rolling(window).quantile(1 - confidence)
                    df[f'var_{int(confidence*100)}_{window}'] = -var_hist
                    
                    # Expected Shortfall (Conditional VaR)
                    es_hist = returns.rolling(window).apply(
                        lambda x: -x[x <= x.quantile(1 - confidence)].mean() 
                        if len(x[x <= x.quantile(1 - confidence)]) > 0 else 0,
                        raw=False
                    )
                    df[f'expected_shortfall_{int(confidence*100)}_{window}'] = es_hist
                    
                    # Parametric VaR (assuming normal distribution)
                    mean_ret = returns.rolling(window).mean()
                    std_ret = returns.rolling(window).std()
                    z_score = stats.norm.ppf(confidence)
                    df[f'parametric_var_{int(confidence*100)}_{window}'] = -(mean_ret - z_score * std_ret)
            
            # VaR ratio (historical vs parametric)
            for confidence in [0.95, 0.99]:
                for window in [20, 60]:
                    hist_var = df[f'var_{int(confidence*100)}_{window}']
                    param_var = df[f'parametric_var_{int(confidence*100)}_{window}']
                    df[f'var_ratio_{int(confidence*100)}_{window}'] = hist_var / (param_var + 1e-8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating VaR measures: {str(e)}")
            return df
    
    def _add_distribution_features(self, df):
        """Add return distribution characteristics"""
        try:
            returns = df['returns']
            
            for window in [20, 60]:
                # Skewness (asymmetry)
                df[f'skewness_{window}'] = returns.rolling(window).apply(
                    lambda x: skew(x.dropna()) if len(x.dropna()) > 3 else 0,
                    raw=False
                )
                
                # Kurtosis (tail heaviness)
                df[f'kurtosis_{window}'] = returns.rolling(window).apply(
                    lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0,
                    raw=False
                )
                
                # Excess kurtosis
                df[f'excess_kurtosis_{window}'] = df[f'kurtosis_{window}'] - 3
                
                # Jarque-Bera normality test statistic
                df[f'jarque_bera_{window}'] = returns.rolling(window).apply(
                    lambda x: jarque_bera(x.dropna())[0] if len(x.dropna()) > 7 else 0,
                    raw=False
                )
            
            # Distribution moments ratios
            for window in [20, 60]:
                # Skewness-kurtosis ratio
                df[f'skew_kurt_ratio_{window}'] = (
                    df[f'skewness_{window}'] / (abs(df[f'kurtosis_{window}']) + 1e-8)
                )
                
                # Coefficient of variation
                mean_ret = returns.rolling(window).mean()
                std_ret = returns.rolling(window).std()
                df[f'coeff_variation_{window}'] = std_ret / (abs(mean_ret) + 1e-8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating distribution features: {str(e)}")
            return df
    
    def _add_tail_risk_measures(self, df):
        """Add tail risk and extreme value measures"""
        try:
            returns = df['returns']
            
            for window in [20, 60]:
                # Tail ratio (99th percentile / 1st percentile)
                q99 = returns.rolling(window).quantile(0.99)
                q01 = returns.rolling(window).quantile(0.01)
                df[f'tail_ratio_{window}'] = q99 / (abs(q01) + 1e-8)
                
                # Extreme positive and negative returns frequency
                extreme_threshold = returns.rolling(window).std() * 2
                
                extreme_positive = (returns > extreme_threshold).rolling(window).sum()
                extreme_negative = (returns < -extreme_threshold).rolling(window).sum()
                
                df[f'extreme_positive_freq_{window}'] = extreme_positive / window
                df[f'extreme_negative_freq_{window}'] = extreme_negative / window
                df[f'extreme_asymmetry_{window}'] = (
                    extreme_positive - extreme_negative
                ) / window
                
                # Maximum/minimum returns in window
                df[f'max_return_{window}'] = returns.rolling(window).max()
                df[f'min_return_{window}'] = returns.rolling(window).min()
                df[f'return_range_{window}'] = (
                    df[f'max_return_{window}'] - df[f'min_return_{window}']
                )
            
            # Tail dependence proxies
            for window in [60]:
                # Upper tail dependence (correlation in extreme up moves)
                threshold_up = returns.rolling(window).quantile(0.9)
                extreme_up = returns > threshold_up
                df[f'upper_tail_events_{window}'] = extreme_up.rolling(window).sum()
                
                # Lower tail dependence (correlation in extreme down moves)
                threshold_down = returns.rolling(window).quantile(0.1)
                extreme_down = returns < threshold_down
                df[f'lower_tail_events_{window}'] = extreme_down.rolling(window).sum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating tail risk measures: {str(e)}")
            return df
    
    def _add_drawdown_features(self, df):
        """Add maximum drawdown and recovery features"""
        try:
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['returns']).cumprod()
            
            # Running maximum (peak)
            df['running_max'] = df['cumulative_returns'].expanding().max()
            
            # Current drawdown
            df['current_drawdown'] = (df['cumulative_returns'] / df['running_max']) - 1
            
            # Maximum drawdown over different periods
            for window in [20, 60, 252]:
                rolling_max = df['cumulative_returns'].rolling(window).max()
                rolling_drawdown = (df['cumulative_returns'] / rolling_max) - 1
                df[f'max_drawdown_{window}'] = rolling_drawdown.rolling(window).min()
            
            # Drawdown duration
            df['in_drawdown'] = (df['current_drawdown'] < 0).astype(int)
            df['drawdown_duration'] = df['in_drawdown'].groupby(
                (df['in_drawdown'] != df['in_drawdown'].shift(1)).cumsum()
            ).cumsum()
            
            # Time to recovery (days since last peak)
            df['days_since_peak'] = (
                (df['cumulative_returns'] != df['running_max']).astype(int)
                .groupby((df['cumulative_returns'] == df['running_max']).cumsum())
                .cumsum()
            )
            
            # Drawdown recovery ratio
            for window in [60]:
                avg_drawdown = abs(df['current_drawdown'].rolling(window).mean())
                recovery_time = df['days_since_peak'].rolling(window).mean()
                df[f'recovery_ratio_{window}'] = avg_drawdown / (recovery_time + 1e-8)
            
            # Ulcer Index (drawdown volatility)
            for window in [20, 60]:
                df[f'ulcer_index_{window}'] = np.sqrt(
                    (df['current_drawdown'] ** 2).rolling(window).mean()
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating drawdown features: {str(e)}")
            return df
    
    def _add_risk_adjusted_returns(self, df):
        """Add risk-adjusted return measures"""
        try:
            returns = df['returns']
            
            for window in [20, 60]:
                mean_return = returns.rolling(window).mean() * 252
                volatility = returns.rolling(window).std() * np.sqrt(252)
                
                # Sharpe ratio (risk-free rate assumed to be 0)
                df[f'sharpe_ratio_{window}'] = mean_return / (volatility + 1e-8)
                
                # Calmar ratio (return / max drawdown)
                if f'max_drawdown_{window}' in df.columns:
                    df[f'calmar_ratio_{window}'] = mean_return / (
                        abs(df[f'max_drawdown_{window}']) + 1e-8
                    )
                
                # Sterling ratio
                if f'ulcer_index_{window}' in df.columns:
                    df[f'sterling_ratio_{window}'] = mean_return / (
                        df[f'ulcer_index_{window}'] + 1e-8
                    )
                
                # Information ratio (return / tracking error)
                benchmark_return = 0  # Assume 0 benchmark
                tracking_error = np.sqrt(
                    ((returns - benchmark_return) ** 2).rolling(window).mean() * 252
                )
                df[f'information_ratio_{window}'] = mean_return / (tracking_error + 1e-8)
            
            # Treynor ratio components (would need market beta)
            # Martin ratio (return / Ulcer index)
            for window in [60]:
                if f'ulcer_index_{window}' in df.columns:
                    mean_return = returns.rolling(window).mean() * 252
                    df[f'martin_ratio_{window}'] = mean_return / (
                        df[f'ulcer_index_{window}'] + 1e-8
                    )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted returns: {str(e)}")
            return df
    
    def _add_correlation_risk_features(self, df):
        """Add correlation-based risk features"""
        try:
            returns = df['returns']
            
            # Rolling correlation with lagged returns (serial correlation)
            for lag in [1, 2, 3, 5]:
                for window in [20, 60]:
                    lagged_returns = returns.shift(lag)
                    df[f'serial_corr_lag{lag}_{window}'] = returns.rolling(window).corr(
                        lagged_returns
                    )
            
            # Correlation stability
            for window in [60]:
                # Standard deviation of serial correlations
                serial_corr_cols = [col for col in df.columns 
                                  if col.startswith(f'serial_corr_') and col.endswith(f'_{window}')]
                if serial_corr_cols:
                    df[f'correlation_stability_{window}'] = df[serial_corr_cols].std(axis=1)
            
            # Autocorrelation decay
            autocorr_1 = returns.rolling(60).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0, 
                raw=False
            )
            autocorr_5 = returns.rolling(60).apply(
                lambda x: x.autocorr(lag=5) if len(x.dropna()) > 5 else 0,
                raw=False            )
            
            df['autocorr_decay'] = autocorr_1 - autocorr_5
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk features: {str(e)}")
            return df
    
    def add_regime_risk_features(self, df):
        """Add risk features that change by market regime"""
        try:
            returns = df['returns']
              # Define volatility regimes with proper index alignment
            vol_20 = returns.rolling(20).std() * np.sqrt(252)
            
            # Create regime classification with proper reindex and category handling
            vol_20_clean = vol_20.dropna()
            if len(vol_20_clean) > 10:  # Ensure sufficient data for quantiles
                vol_regime_temp = pd.qcut(vol_20_clean, q=3, labels=[0, 1, 2])  # Low, Med, High vol
                vol_regime = pd.Series(index=df.index, dtype='float64')  # Use float64 instead of category
                vol_regime.loc[vol_regime_temp.index] = vol_regime_temp.astype(float)
            else:
                # Not enough data for regime classification
                vol_regime = pd.Series(1.0, index=df.index)  # Default to medium regime
            
            # Risk metrics by regime
            for regime in [0, 1, 2]:
                # Use proper boolean indexing with aligned indices
                regime_mask = (vol_regime == regime).fillna(False)
                
                if regime_mask.sum() > 10:  # Ensure sufficient data
                    # Conditional VaR by regime
                    regime_returns = returns.loc[regime_mask]
                    if len(regime_returns) > 0:
                        conditional_var = regime_returns.quantile(0.05)
                        
                        # Use loc for proper assignment with boolean indexing
                        df[f'conditional_var_regime_{regime}'] = np.nan
                        df.loc[regime_mask, f'conditional_var_regime_{regime}'] = -conditional_var
                        
                        # Regime-specific Sharpe ratio
                        regime_sharpe = (regime_returns.mean() * 252) / (
                            regime_returns.std() * np.sqrt(252) + 1e-8
                        )
                        df[f'regime_sharpe_{regime}'] = np.nan
                        df.loc[regime_mask, f'regime_sharpe_{regime}'] = regime_sharpe
                else:
                    # Fill with NaN if insufficient data
                    df[f'conditional_var_regime_{regime}'] = np.nan
                    df[f'regime_sharpe_{regime}'] = np.nan
            
            # Risk regime transition probabilities with proper alignment
            regime_changes = vol_regime.diff() != 0
            df['regime_instability'] = regime_changes.rolling(20).sum() / 20
            
            # Current regime risk
            current_regime = vol_regime.iloc[-1] if len(vol_regime.dropna()) > 0 else 1
            df['current_vol_regime'] = vol_regime.fillna(1)  # Fill missing with default regime
            
            logger.info("✅ Added regime-based risk features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding regime risk features: {str(e)}")
            return df

# Global instance
risk_engine = RiskFeatureEngine()
