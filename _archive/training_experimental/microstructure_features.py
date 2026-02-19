"""
Market Microstructure Feature Engineering
Captures order flow, liquidity, and price impact patterns for better trade execution
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
import logging

logger = logging.getLogger(__name__)

class MarketMicrostructureEngine:
    """
    Advanced market microstructure feature engineering
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
    
    def add_microstructure_features(self, df):
        """
        Add comprehensive microstructure features
        """
        try:
            logger.info("🔬 Calculating market microstructure features...")
            
            # Price impact and efficiency features
            df = self._add_price_impact_features(df)
            
            # Volume profile features
            df = self._add_volume_profile_features(df)
            
            # Liquidity proxy features  
            df = self._add_liquidity_features(df)
            
            # Tick-level features (price changes)
            df = self._add_tick_features(df)
            
            # Market efficiency features
            df = self._add_efficiency_features(df)
            
            # Order flow imbalance proxies
            df = self._add_orderflow_features(df)
            
            # Volatility clustering features
            df = self._add_volatility_clustering_features(df)
            
            logger.info("✅ Added market microstructure features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {str(e)}")
            return df
    
    def _add_price_impact_features(self, df):
        """Add price impact and market impact features"""
        try:
            # Price impact per unit volume
            price_change = df['close'].pct_change()
            volume_normalized = df['volume'] / df['volume'].rolling(20).mean()
            
            df['price_impact'] = price_change / (volume_normalized + 1e-8)
            df['price_impact_smoothed'] = df['price_impact'].rolling(5).mean()
            
            # Amihud illiquidity measure (daily return / dollar volume)
            dollar_volume = df['close'] * df['volume']
            df['amihud_illiquidity'] = abs(price_change) / (dollar_volume + 1e-8)
            df['amihud_illiquidity_ma'] = df['amihud_illiquidity'].rolling(20).mean()
            
            # Price efficiency (how quickly prices adjust)
            for lag in [1, 2, 3, 5]:
                lagged_return = price_change.shift(lag)
                df[f'return_autocorr_{lag}'] = price_change.rolling(50).corr(lagged_return)
            
            # Price impact asymmetry (up moves vs down moves)
            up_moves = price_change > 0
            down_moves = price_change < 0
            
            df['upside_price_impact'] = (df['price_impact'] * up_moves).rolling(20).mean()
            df['downside_price_impact'] = (df['price_impact'] * down_moves).rolling(20).mean()
            df['price_impact_asymmetry'] = df['upside_price_impact'] - df['downside_price_impact']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating price impact features: {str(e)}")
            return df
    
    def _add_volume_profile_features(self, df):
        """Add volume profile and distribution features"""
        try:
            # Ensure we have required columns
            required_cols = ['volume', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing required column {col} for volume profile features")
                    return df
            
            # Volume profile features with proper alignment and type checking
            volume_mean_20 = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (volume_mean_20 + 1e-8)  # Avoid division by zero
              # Volume-price correlation with proper error handling
            try:
                df['volume_price_trend'] = df['volume'].rolling(10).corr(df['close']).fillna(0)
            except Exception:
                # Fallback if correlation fails
                df['volume_price_trend'] = 0.0
            
            # Volume at price levels
            volume_sum_20 = df['volume'].rolling(20).sum()
            weighted_price_sum = (df['close'] * df['volume']).rolling(20).sum()
            df['volume_weighted_price'] = weighted_price_sum / (volume_sum_20 + 1e-8)
            df['price_volume_divergence'] = df['close'] - df['volume_weighted_price']
            
            # Unusual volume detection with proper statistics
            volume_mean_50 = df['volume'].rolling(50).mean()
            volume_std_50 = df['volume'].rolling(50).std()
            volume_zscore = (df['volume'] - volume_mean_50) / (volume_std_50 + 1e-8)
            df['unusual_volume'] = (abs(volume_zscore) > 2).astype(int)
            df['volume_zscore'] = volume_zscore.fillna(0)
            
            # Volume acceleration with proper handling
            volume_pct_change = df['volume'].pct_change().fillna(0)
            df['volume_acceleration'] = volume_pct_change.diff().fillna(0)
            
            # Volume-price confirmation
            price_direction = np.sign(df['close'].pct_change().fillna(0))
            volume_direction = np.sign(volume_pct_change)
            df['volume_price_confirmation'] = (price_direction == volume_direction).astype(int)
            
            # On-Balance Volume momentum
            price_changes = df['close'].diff().fillna(0)
            obv_changes = np.where(price_changes > 0, df['volume'], 
                                 np.where(price_changes < 0, -df['volume'], 0))
            df['obv'] = pd.Series(obv_changes).cumsum()
            df['obv_momentum'] = df['obv'].pct_change(10).fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume profile features: {str(e)}")
            return df
    
    def _add_liquidity_features(self, df):
        """Add liquidity proxy features"""
        try:
            # Bid-ask spread proxy (using high-low range)
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            df['spread_proxy_ma'] = df['spread_proxy'].rolling(20).mean()
            
            # Market depth proxy (inverse of price impact)
            df['depth_proxy'] = 1 / (abs(df['price_impact']) + 1e-8)
            df['depth_proxy_normalized'] = df['depth_proxy'] / df['depth_proxy'].rolling(50).mean()
            
            # Liquidity premium (compensation for illiquidity)
            returns_1d = df['close'].pct_change()
            df['liquidity_premium'] = returns_1d / (df['amihud_illiquidity'] + 1e-8)
            
            # Roll's effective spread estimator
            price_changes = df['close'].diff()
            df['rolls_spread'] = 2 * np.sqrt(abs(price_changes.rolling(20).cov(price_changes.shift(1))))
            
            # Market impact cost
            df['market_impact_cost'] = df['spread_proxy'] + df['amihud_illiquidity']
            
            # Liquidity regime classification
            liquidity_score = 1 / (df['spread_proxy_ma'] + df['amihud_illiquidity_ma'] + 1e-8)
            liquidity_percentiles = liquidity_score.rolling(100).rank(pct=True)
            
            df['high_liquidity_regime'] = (liquidity_percentiles > 0.8).astype(int)
            df['low_liquidity_regime'] = (liquidity_percentiles < 0.2).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating liquidity features: {str(e)}")
            return df
    
    def _add_tick_features(self, df):
        """Add tick-level price movement features"""
        try:
            # Tick direction and runs
            price_changes = df['close'].diff()
            tick_direction = np.sign(price_changes)
            
            # Count consecutive ticks in same direction
            df['uptick_run'] = (tick_direction == 1).astype(int).groupby(
                (tick_direction != 1).cumsum()).cumsum()
            df['downtick_run'] = (tick_direction == -1).astype(int).groupby(
                (tick_direction != -1).cumsum()).cumsum()
            
            # Tick imbalance
            for window in [10, 20, 50]:
                upticks = (tick_direction == 1).rolling(window).sum()
                downticks = (tick_direction == -1).rolling(window).sum()
                df[f'tick_imbalance_{window}'] = (upticks - downticks) / window
            
            # Price level stickiness
            price_levels = np.round(df['close'], 2)  # Round to nearest cent
            df['price_level_changes'] = (price_levels != price_levels.shift(1)).astype(int)
            df['price_stickiness'] = 1 - df['price_level_changes'].rolling(20).mean()
            
            # Momentum in tick direction
            df['tick_momentum'] = tick_direction.rolling(5).sum()
            df['tick_momentum_change'] = df['tick_momentum'].diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating tick features: {str(e)}")
            return df
    
    def _add_efficiency_features(self, df):
        """Add market efficiency and mean reversion features"""
        try:
            returns = df['close'].pct_change()
            
            # Variance ratio test components
            for period in [2, 4, 8, 16]:
                returns_period = df['close'].pct_change(period)
                var_1 = returns.rolling(50).var()
                var_period = returns_period.rolling(50).var()
                
                df[f'variance_ratio_{period}'] = var_period / (period * var_1 + 1e-8)
            
            # Hurst exponent proxy (persistence measure)
            def hurst_proxy(series, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]
            
            if len(df) > 50:
                df['hurst_exponent'] = returns.rolling(50).apply(
                    lambda x: hurst_proxy(x.values) if len(x.dropna()) > 20 else 0.5,
                    raw=False
                )
            else:
                df['hurst_exponent'] = 0.5
            
            # Mean reversion indicators
            price_zscore = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            df['mean_reversion_signal'] = -price_zscore  # Negative z-score suggests buying opportunity
            
            # Efficiency ratio (Kaufman)
            for period in [10, 20]:
                direction = abs(df['close'] - df['close'].shift(period))
                volatility = abs(df['close'].diff()).rolling(period).sum()
                df[f'efficiency_ratio_{period}'] = direction / (volatility + 1e-8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating efficiency features: {str(e)}")
            return df
    
    def _add_orderflow_features(self, df):
        """Add order flow imbalance proxy features"""
        try:
            # Price and volume interaction
            returns = df['close'].pct_change()
            volume_changes = df['volume'].pct_change()
            
            # Order flow imbalance proxy
            df['orderflow_imbalance'] = returns * volume_changes
            df['orderflow_imbalance_ma'] = df['orderflow_imbalance'].rolling(10).mean()
            
            # Buying/selling pressure proxies
            # Assumption: High volume + price increase = buying pressure
            buying_pressure = (returns > 0) & (volume_changes > 0)
            selling_pressure = (returns < 0) & (volume_changes > 0)
            
            for window in [5, 20]:
                df[f'buying_pressure_{window}'] = buying_pressure.rolling(window).mean()
                df[f'selling_pressure_{window}'] = selling_pressure.rolling(window).mean()
                df[f'pressure_imbalance_{window}'] = (
                    df[f'buying_pressure_{window}'] - df[f'selling_pressure_{window}']
                )
            
            # Volume-weighted directional movement
            up_volume = np.where(returns > 0, df['volume'], 0)
            down_volume = np.where(returns < 0, df['volume'], 0)
            
            df['net_volume'] = up_volume - down_volume
            df['net_volume_ma'] = df['net_volume'].rolling(20).mean()
            
            # Order flow persistence
            df['orderflow_persistence'] = df['orderflow_imbalance'].rolling(10).apply(
                lambda x: len([i for i in range(1, len(x)) if x.iloc[i] * x.iloc[i-1] > 0]) / max(len(x)-1, 1),
                raw=False
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating order flow features: {str(e)}")
            return df
    
    def _add_volatility_clustering_features(self, df):
        """Add volatility clustering and GARCH-like features"""
        try:
            returns = df['close'].pct_change()
            squared_returns = returns ** 2
            
            # Volatility clustering measures
            for window in [5, 10, 20]:
                vol_ma = squared_returns.rolling(window).mean()
                df[f'volatility_ma_{window}'] = np.sqrt(vol_ma)
                
                # Volatility of volatility
                df[f'vol_of_vol_{window}'] = df[f'volatility_ma_{window}'].rolling(window).std()
            
            # ARCH effect test components
            for lag in [1, 2, 3, 5]:
                df[f'squared_return_lag_{lag}'] = squared_returns.shift(lag)
                df[f'vol_autocorr_{lag}'] = squared_returns.rolling(50).corr(
                    squared_returns.shift(lag)
                )
            
            # Volatility regime switching
            short_vol = df['volatility_ma_5']
            long_vol = df['volatility_ma_20']
            
            df['vol_regime_switch'] = (short_vol / long_vol).rolling(10).std()
            df['high_vol_regime'] = (short_vol > long_vol * 1.5).astype(int)
            df['low_vol_regime'] = (short_vol < long_vol * 0.7).astype(int)
            
            # Volatility mean reversion
            vol_zscore = (short_vol - long_vol) / (long_vol.rolling(50).std() + 1e-8)
            df['vol_mean_reversion'] = -vol_zscore
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility clustering features: {str(e)}")
            return df
    
    def add_regime_features(self, df):
        """Add market regime classification features"""
        try:
            returns = df['close'].pct_change()
            
            # Combine multiple regime indicators
            # 1. Volatility regime
            vol_20 = returns.rolling(20).std()
            vol_regime = pd.qcut(vol_20.dropna(), q=3, labels=['low_vol', 'med_vol', 'high_vol'])
            
            # 2. Trend regime
            sma_short = df['close'].rolling(10).mean()
            sma_long = df['close'].rolling(50).mean()
            trend_regime = pd.cut(
                (sma_short - sma_long) / sma_long * 100,
                bins=[-np.inf, -2, 2, np.inf],
                labels=['downtrend', 'sideways', 'uptrend']
            )
            
            # 3. Liquidity regime (from previous calculations)
            if 'high_liquidity_regime' in df.columns:
                liquidity_regime = np.where(
                    df['high_liquidity_regime'] == 1, 'high_liq',
                    np.where(df['low_liquidity_regime'] == 1, 'low_liq', 'med_liq')
                )
            else:
                liquidity_regime = 'med_liq'
            
            # Combine regimes
            df['market_regime'] = (
                vol_regime.astype(str) + '_' + 
                trend_regime.astype(str) + '_' + 
                pd.Series(liquidity_regime, index=df.index).astype(str)
            )
            
            # One-hot encode key regime combinations
            common_regimes = df['market_regime'].value_counts().head(10).index
            for regime in common_regimes:
                df[f'regime_{regime.replace(" ", "_").lower()}'] = (
                    df['market_regime'] == regime
                ).astype(int)
            
            logger.info("✅ Added market regime features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding regime features: {str(e)}")
            return df

# Global instance
microstructure_engine = MarketMicrostructureEngine()
