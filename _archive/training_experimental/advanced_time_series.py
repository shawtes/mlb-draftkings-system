"""
Advanced Time Series Models for ML Trading
Based on Stefan Jansen's "Machine Learning for Algorithmic Trading" - Chapter 9

This module implements:
1. GARCH Models: For volatility forecasting and risk management
2. Vector Autoregression (VAR): For multi-asset dependencies
3. Cointegration: For pairs trading and statistical arbitrage
4. Advanced Risk Metrics: VaR, Maximum Drawdown, Regime Detection
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

# Core time series libraries
try:
    from arch import arch_model
    from arch.univariate import GARCH, ConstantMean, Normal
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Financial libraries
try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GARCHResults:
    """Results container for GARCH model analysis"""
    model_name: str
    fitted_model: object
    volatility_forecast: np.ndarray
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    volatility_clustering_score: float

@dataclass
class VARResults:
    """Results container for VAR model analysis"""
    model: object
    fitted_values: pd.DataFrame
    residuals: pd.DataFrame
    forecast: pd.DataFrame
    impulse_responses: Dict[str, np.ndarray]
    forecast_error_variance: pd.DataFrame
    granger_causality: Dict[str, float]
    cointegration_rank: int
    correlation_matrix: pd.DataFrame

@dataclass
class CointegrationResults:
    """Results container for cointegration analysis"""
    pairs: List[Tuple[str, str]]
    cointegration_vectors: Dict[str, np.ndarray]
    p_values: Dict[str, float]
    critical_values: Dict[str, Dict[str, float]]
    half_life: Dict[str, float]
    spread_series: Dict[str, pd.Series]
    z_scores: Dict[str, pd.Series]
    trading_signals: Dict[str, pd.Series]
    sharpe_ratios: Dict[str, float]

class AdvancedTimeSeriesModels:
    """
    Advanced Time Series Models for sophisticated trading strategies
    """
    
    def __init__(self, enable_garch=True, enable_var=True, enable_cointegration=True):
        self.enable_garch = enable_garch and ARCH_AVAILABLE
        self.enable_var = enable_var and STATSMODELS_AVAILABLE
        self.enable_cointegration = enable_cointegration and STATSMODELS_AVAILABLE
        
        # Model storage
        self.garch_models = {}
        self.var_model = None
        self.cointegration_results = None
        
        # Risk management parameters
        self.var_confidence_levels = [0.95, 0.99]
        self.regime_threshold = 0.02  # 2% threshold for regime detection
        
        logger.info(f"🧠 Advanced Time Series Models initialized:")
        logger.info(f"   GARCH Models: {'✅' if self.enable_garch else '❌'}")
        logger.info(f"   VAR Models: {'✅' if self.enable_var else '❌'}")
        logger.info(f"   Cointegration: {'✅' if self.enable_cointegration else '❌'}")
    
    def fit_garch_model(self, returns: pd.Series, symbol: str, 
                       model_type: str = 'GARCH', p: int = 1, q: int = 1) -> GARCHResults:
        """
        Fit GARCH model for volatility forecasting
        
        Args:
            returns: Return series
            symbol: Asset symbol
            model_type: 'GARCH', 'EGARCH', 'GJR-GARCH'
            p: ARCH lag order
            q: GARCH lag order
        """
        if not self.enable_garch:
            logger.warning("GARCH models not available - install arch package")
            return None
            
        try:
            # Remove any NaN values
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 100:
                logger.warning(f"Insufficient data for GARCH model: {len(returns_clean)} observations")
                return None
            
            # Scale returns to percentage
            returns_pct = returns_clean * 100
            
            # Fit GARCH model
            if model_type == 'GARCH':
                model = arch_model(returns_pct, vol='GARCH', p=p, q=q, rescale=False)
            elif model_type == 'EGARCH':
                model = arch_model(returns_pct, vol='EGARCH', p=p, o=1, q=q, rescale=False)
            elif model_type == 'GJR-GARCH':
                from arch.univariate import GARCH
                model = arch_model(returns_pct, vol='GARCH', p=p, o=1, q=q, rescale=False)
            else:
                raise ValueError(f"Unknown GARCH model type: {model_type}")
            
            # Fit the model
            fitted_model = model.fit(disp='off', show_warning=False)
            
            # Extract results
            conditional_volatility = fitted_model.conditional_volatility / 100  # Convert back to decimal
            standardized_residuals = fitted_model.std_resid
            
            # Forecast volatility (1-step ahead)
            forecast = fitted_model.forecast(horizon=1, reindex=False)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
            
            # Calculate VaR and Expected Shortfall
            var_95 = np.percentile(returns_clean, 5)
            var_99 = np.percentile(returns_clean, 1)
            es_95 = returns_clean[returns_clean <= var_95].mean()
            
            # Volatility clustering score (persistence)
            volatility_clustering = self._calculate_volatility_clustering(conditional_volatility)
            
            # Store the model
            self.garch_models[symbol] = fitted_model
            
            results = GARCHResults(
                model_name=f"{model_type}({p},{q})",
                fitted_model=fitted_model,
                volatility_forecast=volatility_forecast,
                conditional_volatility=conditional_volatility.values,
                standardized_residuals=standardized_residuals.values,
                log_likelihood=fitted_model.loglikelihood,
                aic=fitted_model.aic,
                bic=fitted_model.bic,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                volatility_clustering_score=volatility_clustering
            )
            
            logger.info(f"✅ GARCH model fitted for {symbol}: AIC={results.aic:.2f}, Vol={volatility_forecast:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fitting GARCH model for {symbol}: {str(e)}")
            return None
    
    def fit_var_model(self, price_data: pd.DataFrame, maxlags: int = 15) -> VARResults:
        """
        Fit Vector Autoregression model for multi-asset dependencies
        
        Args:
            price_data: DataFrame with price series for multiple assets
            maxlags: Maximum number of lags to consider
        """
        if not self.enable_var:
            logger.warning("VAR models not available - install statsmodels")
            return None
            
        try:
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if len(returns) < 100:
                logger.warning(f"Insufficient data for VAR model: {len(returns)} observations")
                return None
                
            if returns.shape[1] < 2:
                logger.warning("VAR requires at least 2 variables")
                return None
            
            # Fit VAR model
            model = VAR(returns)
            
            # Select optimal lag length
            lag_order_results = model.select_order(maxlags=maxlags)
            optimal_lags = lag_order_results.aic
            
            # Fit with optimal lags
            fitted_model = model.fit(optimal_lags)
            
            # Generate forecasts
            forecast_steps = 5
            forecast = fitted_model.forecast(returns.values[-optimal_lags:], steps=forecast_steps)
            forecast_df = pd.DataFrame(forecast, columns=returns.columns)
            
            # Calculate impulse response functions
            impulse_responses = {}
            irf = fitted_model.irf(periods=10)
            for i, col in enumerate(returns.columns):
                impulse_responses[col] = irf.irfs[:, :, i]
            
            # Forecast error variance decomposition
            fevd = fitted_model.fevd(periods=10)
            
            # Granger causality tests
            granger_causality = {}
            for col in returns.columns:
                other_cols = [c for c in returns.columns if c != col]
                if other_cols:
                    gc_test = fitted_model.test_causality(col, other_cols[0], kind='f')
                    granger_causality[f"{other_cols[0]}_causes_{col}"] = gc_test.pvalue
            
            # Test for cointegration
            cointegration_rank = self._johansen_cointegration_test(price_data)
            
            # Store the model
            self.var_model = fitted_model
            
            results = VARResults(
                model=fitted_model,
                fitted_values=pd.DataFrame(fitted_model.fittedvalues, 
                                         index=returns.index[optimal_lags:], 
                                         columns=returns.columns),
                residuals=pd.DataFrame(fitted_model.resid, 
                                     index=returns.index[optimal_lags:], 
                                     columns=returns.columns),
                forecast=forecast_df,
                impulse_responses=impulse_responses,
                forecast_error_variance=fevd.decomp,
                granger_causality=granger_causality,
                cointegration_rank=cointegration_rank,
                correlation_matrix=returns.corr()
            )
            
            logger.info(f"✅ VAR model fitted with {optimal_lags} lags for {len(returns.columns)} assets")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fitting VAR model: {str(e)}")
            return None
    
    def analyze_cointegration(self, price_data: pd.DataFrame, 
                            significance_level: float = 0.05) -> CointegrationResults:
        """
        Perform cointegration analysis for pairs trading
        
        Args:
            price_data: DataFrame with price series for multiple assets
            significance_level: Statistical significance level
        """
        if not self.enable_cointegration:
            logger.warning("Cointegration analysis not available - install statsmodels")
            return None
            
        try:
            # Find all possible pairs
            symbols = price_data.columns.tolist()
            pairs = [(symbols[i], symbols[j]) for i in range(len(symbols)) 
                    for j in range(i+1, len(symbols))]
            
            cointegration_vectors = {}
            p_values = {}
            critical_values = {}
            half_life = {}
            spread_series = {}
            z_scores = {}
            trading_signals = {}
            sharpe_ratios = {}
            
            for pair in pairs:
                try:
                    s1, s2 = pair
                    price1 = price_data[s1].dropna()
                    price2 = price_data[s2].dropna()
                    
                    # Ensure same length
                    common_index = price1.index.intersection(price2.index)
                    if len(common_index) < 50:
                        continue
                        
                    price1 = price1.loc[common_index]
                    price2 = price2.loc[common_index]
                    
                    # Perform cointegration test
                    coint_score, p_value, crit_values = coint(price1, price2)
                    
                    if p_value <= significance_level:
                        # Calculate cointegration vector (hedge ratio)
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression()
                        lr.fit(price2.values.reshape(-1, 1), price1.values)
                        hedge_ratio = lr.coef_[0]
                        
                        # Calculate spread
                        spread = price1 - hedge_ratio * price2
                        
                        # Calculate half-life of mean reversion
                        hl = self._calculate_half_life(spread)
                        
                        # Generate z-scores
                        z_score = (spread - spread.mean()) / spread.std()
                        
                        # Generate trading signals
                        signals = self._generate_cointegration_signals(z_score)
                        
                        # Calculate strategy Sharpe ratio
                        returns1 = price1.pct_change().dropna()
                        returns2 = price2.pct_change().dropna()
                        strategy_returns = self._calculate_pairs_returns(
                            returns1, returns2, signals, hedge_ratio
                        )
                        sharpe = self._calculate_sharpe_ratio(strategy_returns)
                        
                        # Store results
                        pair_key = f"{s1}_{s2}"
                        cointegration_vectors[pair_key] = hedge_ratio
                        p_values[pair_key] = p_value
                        critical_values[pair_key] = {
                            '1%': crit_values[0], '5%': crit_values[1], '10%': crit_values[2]
                        }
                        half_life[pair_key] = hl
                        spread_series[pair_key] = spread
                        z_scores[pair_key] = z_score
                        trading_signals[pair_key] = signals
                        sharpe_ratios[pair_key] = sharpe
                        
                        logger.info(f"✅ Cointegrated pair found: {pair_key} (p={p_value:.4f}, Sharpe={sharpe:.2f})")
                    
                except Exception as e:
                    logger.debug(f"Error analyzing pair {pair}: {str(e)}")
                    continue
            
            # Store results
            self.cointegration_results = CointegrationResults(
                pairs=pairs,
                cointegration_vectors=cointegration_vectors,
                p_values=p_values,
                critical_values=critical_values,
                half_life=half_life,
                spread_series=spread_series,
                z_scores=z_scores,
                trading_signals=trading_signals,
                sharpe_ratios=sharpe_ratios
            )
            
            logger.info(f"✅ Cointegration analysis complete: {len(cointegration_vectors)} pairs found")
            return self.cointegration_results
            
        except Exception as e:
            logger.error(f"❌ Error in cointegration analysis: {str(e)}")
            return None
    
    def detect_volatility_regimes(self, returns: pd.Series, 
                                 window: int = 21) -> pd.Series:
        """
        Detect volatility regimes using rolling volatility
        """
        try:
            rolling_vol = returns.rolling(window=window).std()
            vol_mean = rolling_vol.mean()
            vol_std = rolling_vol.std()
            
            # Define regimes: 0=Low, 1=Normal, 2=High
            regimes = pd.Series(index=returns.index, dtype=int)
            regimes[rolling_vol <= vol_mean - vol_std] = 0  # Low volatility
            regimes[(rolling_vol > vol_mean - vol_std) & 
                   (rolling_vol < vol_mean + vol_std)] = 1  # Normal volatility
            regimes[rolling_vol >= vol_mean + vol_std] = 2  # High volatility
            
            return regimes.fillna(1)  # Default to normal regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {str(e)}")
            return pd.Series(index=returns.index, data=1)  # Default to normal regime
    
    def calculate_dynamic_var(self, returns: pd.Series, 
                            confidence_level: float = 0.95,
                            window: int = 252) -> pd.Series:
        """
        Calculate dynamic Value at Risk using rolling window
        """
        try:
            var_series = returns.rolling(window=window).quantile(1 - confidence_level)
            return var_series
            
        except Exception as e:
            logger.error(f"Error calculating dynamic VaR: {str(e)}")
            return pd.Series(index=returns.index, data=np.nan)
    
    def calculate_maximum_drawdown(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        """
        try:
            # Calculate cumulative returns
            cumulative = (1 + price_series.pct_change()).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum drawdown
            max_dd = drawdown.min()
            
            # Duration of maximum drawdown
            dd_duration = self._calculate_drawdown_duration(drawdown)
            
            return {
                'max_drawdown': max_dd,
                'current_drawdown': drawdown.iloc[-1],
                'max_drawdown_duration': dd_duration,
                'recovery_time': self._calculate_recovery_time(drawdown)
            }
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {str(e)}")
            return {'max_drawdown': 0, 'current_drawdown': 0, 
                   'max_drawdown_duration': 0, 'recovery_time': 0}
    
    def optimize_kelly_criterion(self, returns: pd.Series, 
                                window: int = 252) -> pd.Series:
        """
        Calculate optimal position sizing using Kelly Criterion
        """
        try:
            def kelly_fraction(rets):
                if len(rets) < 10:
                    return 0.1  # Conservative default
                    
                mean_return = rets.mean()
                variance = rets.var()
                
                if variance <= 0:
                    return 0.1
                    
                # Kelly fraction = (expected return - risk-free rate) / variance
                # Assuming risk-free rate = 0 for simplicity
                kelly = mean_return / variance
                
                # Cap Kelly fraction for risk management
                return max(0.01, min(0.25, kelly))  # Between 1% and 25%
            
            kelly_series = returns.rolling(window=window).apply(kelly_fraction)
            return kelly_series.fillna(0.1)
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return pd.Series(index=returns.index, data=0.1)
    
    # Helper methods
    def _calculate_volatility_clustering(self, volatility: pd.Series) -> float:
        """Calculate volatility clustering score"""
        try:
            # Measure persistence in volatility (autocorrelation)
            vol_changes = volatility.diff().dropna()
            if len(vol_changes) < 10:
                return 0.5
                
            # Calculate first-order autocorrelation
            correlation = vol_changes.autocorr(lag=1)
            return abs(correlation) if not np.isnan(correlation) else 0.5
            
        except:
            return 0.5
    
    def _johansen_cointegration_test(self, price_data: pd.DataFrame) -> int:
        """Perform Johansen cointegration test"""
        try:
            if not STATSMODELS_AVAILABLE:
                return 0
                
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            # Remove any columns with insufficient data
            clean_data = price_data.dropna()
            if clean_data.shape[0] < 50 or clean_data.shape[1] < 2:
                return 0
                
            # Perform Johansen test
            result = coint_johansen(clean_data.values, det_order=0, k_ar_diff=1)
            
            # Count number of cointegrating relationships at 5% significance
            coint_rank = 0
            for i in range(len(result.lr1)):
                if result.lr1[i] > result.cvt[i, 1]:  # 5% critical value
                    coint_rank += 1
                    
            return coint_rank
            
        except:
            return 0
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion"""
        try:
            # Use AR(1) model to estimate mean reversion speed
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            common_index = spread_lag.index.intersection(spread_diff.index)
            if len(common_index) < 10:
                return np.inf
                
            spread_lag = spread_lag.loc[common_index]
            spread_diff = spread_diff.loc[common_index]
            
            # Regression: Δy_t = α + β*y_{t-1} + ε_t
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(spread_lag.values.reshape(-1, 1), spread_diff.values)
            
            beta = lr.coef_[0]
            
            if beta >= 0:
                return np.inf  # No mean reversion
                
            # Half-life = -ln(2) / ln(1 + β)
            half_life = -np.log(2) / np.log(1 + beta)
            return max(1, half_life)  # At least 1 period
            
        except:
            return np.inf
    
    def _generate_cointegration_signals(self, z_scores: pd.Series, 
                                      entry_threshold: float = 2.0,
                                      exit_threshold: float = 0.5) -> pd.Series:
        """Generate trading signals based on z-scores"""
        try:
            signals = pd.Series(index=z_scores.index, data=0)
            
            # Long signal when z-score is below -entry_threshold
            signals[z_scores <= -entry_threshold] = 1
            
            # Short signal when z-score is above entry_threshold
            signals[z_scores >= entry_threshold] = -1
            
            # Exit signals when z-score returns to normal range
            signals[abs(z_scores) <= exit_threshold] = 0
            
            # Forward fill signals to maintain positions
            signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
            
            return signals
            
        except:
            return pd.Series(index=z_scores.index, data=0)
    
    def _calculate_pairs_returns(self, returns1: pd.Series, returns2: pd.Series,
                               signals: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate returns for pairs trading strategy"""
        try:
            # Align all series
            common_index = returns1.index.intersection(returns2.index).intersection(signals.index)
            
            if len(common_index) < 10:
                return pd.Series()
                
            r1 = returns1.loc[common_index]
            r2 = returns2.loc[common_index]
            sig = signals.loc[common_index].shift(1).fillna(0)  # Use previous signal
            
            # Strategy returns: long asset1, short hedge_ratio * asset2
            strategy_returns = sig * (r1 - hedge_ratio * r2)
            
            return strategy_returns
            
        except:
            return pd.Series()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 10:
                return 0.0
                
            excess_returns = returns - risk_free_rate
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods"""
        try:
            is_dd = drawdown < 0
            dd_periods = is_dd.astype(int).groupby((~is_dd).cumsum()).cumsum()
            return dd_periods.max() if len(dd_periods) > 0 else 0
            
        except:
            return 0
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> int:
        """Calculate average recovery time from drawdowns"""
        try:
            # Find drawdown periods and recovery times
            is_dd = drawdown < 0
            dd_groups = is_dd.astype(int).groupby((~is_dd).cumsum())
            
            recovery_times = []
            for name, group in dd_groups:
                if group.sum() > 0:  # This was a drawdown period
                    recovery_times.append(len(group))
            
            return int(np.mean(recovery_times)) if recovery_times else 0
            
        except:
            return 0

# Global instance
advanced_time_series = AdvancedTimeSeriesModels()

def analyze_volatility_with_garch(price_data: pd.DataFrame, 
                                symbols: List[str] = None) -> Dict[str, GARCHResults]:
    """
    Convenience function to analyze volatility for multiple assets
    """
    if symbols is None:
        symbols = price_data.columns.tolist()
        
    results = {}
    
    for symbol in symbols:
        if symbol in price_data.columns:
            try:
                # Calculate returns
                returns = price_data[symbol].pct_change().dropna()
                
                # Fit GARCH model
                garch_result = advanced_time_series.fit_garch_model(returns, symbol)
                
                if garch_result:
                    results[symbol] = garch_result
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol} with GARCH: {str(e)}")
                
    return results

def analyze_multi_asset_dependencies(price_data: pd.DataFrame) -> VARResults:
    """
    Convenience function to analyze multi-asset dependencies with VAR
    """
    try:
        return advanced_time_series.fit_var_model(price_data)
    except Exception as e:
        logger.error(f"Error in multi-asset VAR analysis: {str(e)}")
        return None

def find_cointegrated_pairs(price_data: pd.DataFrame) -> CointegrationResults:
    """
    Convenience function to find cointegrated pairs for arbitrage
    """
    try:
        return advanced_time_series.analyze_cointegration(price_data)
    except Exception as e:
        logger.error(f"Error in cointegration analysis: {str(e)}")
        return None 