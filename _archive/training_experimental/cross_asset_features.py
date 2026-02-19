"""
Cross-Asset Feature Engineering for Cryptocurrency Trading
Captures relationships between different assets, correlations, and market regimes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class CrossAssetFeatureEngine:
    """
    Advanced cross-asset feature engineering
    """
    
    def __init__(self):
        self.asset_correlations = {}
        self.market_beta_cache = {}
        self.sector_mappings = {
            # Major cryptocurrencies
            'BTC': 'store_of_value',
            'ETH': 'smart_contract_platform', 
            'BNB': 'exchange_token',
            'ADA': 'smart_contract_platform',
            'SOL': 'smart_contract_platform',
            'DOT': 'interoperability',
            'LINK': 'oracle',
            'UNI': 'defi',
            'AAVE': 'defi',
            'COMP': 'defi',
            # Add more as needed
        }
    
    def add_cross_asset_features(self, df, symbol, market_data_dict=None):
        """
        Add cross-asset correlation and beta features
        
        Args:
            df: Main asset DataFrame
            symbol: Current asset symbol (e.g., 'BTC-USD')
            market_data_dict: Dict of {symbol: DataFrame} for other assets
        """
        try:
            if market_data_dict is None or len(market_data_dict) < 2:
                logger.warning("Insufficient market data for cross-asset features")
                return self._add_sector_features(df, symbol)
            
            # Calculate asset returns
            df = self._calculate_returns(df)
            
            # Add market beta features
            df = self._add_market_beta_features(df, symbol, market_data_dict)
            
            # Add correlation features
            df = self._add_correlation_features(df, symbol, market_data_dict)
            
            # Add relative strength features  
            df = self._add_relative_strength_features(df, symbol, market_data_dict)
            
            # Add market regime features
            df = self._add_market_regime_features(df, symbol, market_data_dict)
            
            # Add sector rotation features
            df = self._add_sector_features(df, symbol)
            
            logger.info(f"✅ Added cross-asset features for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error adding cross-asset features: {str(e)}")
            return df
    
    def _calculate_returns(self, df):
        """Calculate various return periods"""
        periods = [1, 5, 10, 20, 60]
        for period in periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        return df
    
    def _add_market_beta_features(self, df, symbol, market_data_dict):
        """Add beta coefficients vs major assets"""
        try:
            # Calculate beta vs BTC (crypto market proxy)
            if 'BTC-USD' in market_data_dict and symbol != 'BTC-USD':
                btc_data = market_data_dict['BTC-USD'].copy()
                btc_data['btc_return'] = btc_data['close'].pct_change()
                
                # Merge on timestamp/index
                merged = pd.merge_asof(df.sort_index(), btc_data[['btc_return']].sort_index(), 
                                     left_index=True, right_index=True, direction='nearest')
                
                if 'return_1' in merged.columns and 'btc_return' in merged.columns:
                    # Rolling beta calculation
                    for window in [20, 60, 120]:
                        covariance = merged['return_1'].rolling(window).cov(merged['btc_return'])
                        btc_variance = merged['btc_return'].rolling(window).var()
                        df[f'beta_btc_{window}'] = covariance / btc_variance
                        
                        # Beta stability (how much beta changes)
                        df[f'beta_btc_{window}_std'] = df[f'beta_btc_{window}'].rolling(20).std()
            
            # Calculate beta vs ETH (alt-coin proxy)
            if 'ETH-USD' in market_data_dict and symbol != 'ETH-USD':
                eth_data = market_data_dict['ETH-USD'].copy()
                eth_data['eth_return'] = eth_data['close'].pct_change()
                
                merged = pd.merge_asof(df.sort_index(), eth_data[['eth_return']].sort_index(),
                                     left_index=True, right_index=True, direction='nearest')
                
                if 'return_1' in merged.columns and 'eth_return' in merged.columns:
                    for window in [20, 60]:
                        covariance = merged['return_1'].rolling(window).cov(merged['eth_return'])
                        eth_variance = merged['eth_return'].rolling(window).var()
                        df[f'beta_eth_{window}'] = covariance / eth_variance
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating beta features: {str(e)}")
            return df
    
    def _add_correlation_features(self, df, symbol, market_data_dict):
        """Add rolling correlation features"""
        try:
            # Major correlation pairs
            major_pairs = ['BTC-USD', 'ETH-USD', 'BNB-USD']
            
            for pair in major_pairs:
                if pair in market_data_dict and pair != symbol:
                    pair_data = market_data_dict[pair].copy()
                    pair_data['pair_return'] = pair_data['close'].pct_change()
                    
                    merged = pd.merge_asof(df.sort_index(), pair_data[['pair_return']].sort_index(),
                                         left_index=True, right_index=True, direction='nearest')
                    
                    if 'return_1' in merged.columns and 'pair_return' in merged.columns:
                        # Rolling correlations
                        for window in [10, 20, 60]:
                            correlation = merged['return_1'].rolling(window).corr(merged['pair_return'])
                            df[f'corr_{pair.replace("-", "_").lower()}_{window}'] = correlation
                            
                            # Correlation regime (high/low correlation periods)
                            corr_median = correlation.rolling(120).median()
                            df[f'high_corr_regime_{pair.replace("-", "_").lower()}'] = (
                                correlation > corr_median
                            ).astype(int)
            
            # Average correlation with market
            corr_cols = [col for col in df.columns if col.startswith('corr_') and col.endswith('_20')]
            if corr_cols:
                df['avg_market_correlation'] = df[corr_cols].mean(axis=1)
                df['correlation_dispersion'] = df[corr_cols].std(axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating correlation features: {str(e)}")
            return df
    
    def _add_relative_strength_features(self, df, symbol, market_data_dict):
        """Add relative strength vs other assets"""
        try:
            # Compare performance vs BTC
            if 'BTC-USD' in market_data_dict and symbol != 'BTC-USD':
                btc_data = market_data_dict['BTC-USD'].copy()
                
                # Relative strength over different periods
                for period in [5, 20, 60]:
                    asset_performance = df['close'].pct_change(period)
                    btc_performance = btc_data['close'].pct_change(period)
                    
                    # Align timestamps
                    aligned_btc = pd.merge_asof(df.sort_index(), 
                                              btc_performance.to_frame('btc_perf').sort_index(),
                                              left_index=True, right_index=True, direction='nearest')
                    
                    if 'btc_perf' in aligned_btc.columns:
                        df[f'relative_strength_btc_{period}'] = asset_performance - aligned_btc['btc_perf']
                        
                        # Relative strength ranking
                        df[f'rs_btc_{period}_rank'] = (
                            df[f'relative_strength_btc_{period}'].rolling(120).rank(pct=True)
                        )
            
            # Sector relative strength
            base_symbol = symbol.split('-')[0]  # Remove -USD
            sector = self.sector_mappings.get(base_symbol, 'other')
            
            # Compare vs sector peers
            sector_peers = [s for s, sect in self.sector_mappings.items() 
                          if sect == sector and f"{s}-USD" in market_data_dict and f"{s}-USD" != symbol]
            
            if sector_peers:
                peer_returns = []
                for peer in sector_peers[:3]:  # Limit to top 3 peers
                    peer_symbol = f"{peer}-USD"
                    if peer_symbol in market_data_dict:
                        peer_return = market_data_dict[peer_symbol]['close'].pct_change(20)
                        peer_returns.append(peer_return)
                
                if peer_returns:
                    # Average sector performance
                    sector_avg_return = pd.concat(peer_returns, axis=1).mean(axis=1)
                    asset_return_20 = df['close'].pct_change(20)
                    
                    # Sector relative strength
                    aligned_sector = pd.merge_asof(df.sort_index(),
                                                 sector_avg_return.to_frame('sector_avg').sort_index(),
                                                 left_index=True, right_index=True, direction='nearest')
                    
                    if 'sector_avg' in aligned_sector.columns:
                        df['sector_relative_strength'] = asset_return_20 - aligned_sector['sector_avg']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating relative strength features: {str(e)}")
            return df
    
    def _add_market_regime_features(self, df, symbol, market_data_dict):
        """Add market regime detection features"""
        try:
            # Market-wide volatility regime
            if 'BTC-USD' in market_data_dict:
                btc_data = market_data_dict['BTC-USD'].copy()
                btc_volatility = btc_data['close'].pct_change().rolling(20).std() * np.sqrt(365) * 100
                
                # Volatility regime classification
                vol_quantiles = btc_volatility.rolling(252).quantile([0.33, 0.67])
                
                aligned_vol = pd.merge_asof(df.sort_index(),
                                          btc_volatility.to_frame('btc_vol').sort_index(),
                                          left_index=True, right_index=True, direction='nearest')
                
                if 'btc_vol' in aligned_vol.columns:
                    df['market_vol_regime'] = pd.cut(aligned_vol['btc_vol'], 
                                                   bins=[-np.inf, 50, 100, np.inf],
                                                   labels=[0, 1, 2]).astype(float)
                    
                    # High volatility periods
                    df['high_vol_regime'] = (aligned_vol['btc_vol'] > 80).astype(int)
            
            # Trend regime based on multiple assets
            major_assets = ['BTC-USD', 'ETH-USD']
            trend_signals = []
            
            for asset in major_assets:
                if asset in market_data_dict:
                    asset_data = market_data_dict[asset].copy()
                    
                    # Simple trend: price vs moving average
                    sma_50 = asset_data['close'].rolling(50).mean()
                    trend_signal = (asset_data['close'] > sma_50).astype(int)
                    trend_signals.append(trend_signal)
            
            if trend_signals:
                # Market trend consensus
                aligned_trends = []
                for signal in trend_signals:
                    aligned = pd.merge_asof(df.sort_index(),
                                         signal.to_frame('trend').sort_index(),
                                         left_index=True, right_index=True, direction='nearest')
                    if 'trend' in aligned.columns:
                        aligned_trends.append(aligned['trend'])
                
                if aligned_trends:
                    df['market_trend_consensus'] = pd.concat(aligned_trends, axis=1).mean(axis=1)
                    df['strong_bull_regime'] = (df['market_trend_consensus'] > 0.8).astype(int)
                    df['strong_bear_regime'] = (df['market_trend_consensus'] < 0.2).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating market regime features: {str(e)}")
            return df
    
    def _add_sector_features(self, df, symbol):
        """Add sector-based features"""
        try:
            base_symbol = symbol.split('-')[0]
            sector = self.sector_mappings.get(base_symbol, 'other')
            
            # One-hot encode sector
            df['sector_store_of_value'] = (sector == 'store_of_value').astype(int)
            df['sector_smart_contract'] = (sector == 'smart_contract_platform').astype(int)
            df['sector_defi'] = (sector == 'defi').astype(int)
            df['sector_exchange_token'] = (sector == 'exchange_token').astype(int)
            df['sector_other'] = (sector == 'other').astype(int)
            
            # Market cap tier (you can customize this)
            large_cap = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT']
            df['large_cap_asset'] = (base_symbol in large_cap).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding sector features: {str(e)}")
            return df
    
    def create_market_factor_features(self, market_data_dict, n_factors=3):
        """
        Create market factor features using PCA on multiple assets
        """
        try:
            if len(market_data_dict) < 3:
                logger.warning("Need at least 3 assets for factor analysis")
                return {}
            
            # Collect returns from all assets
            returns_data = {}
            for symbol, data in market_data_dict.items():
                returns_data[symbol] = data['close'].pct_change().dropna()
            
            # Create aligned returns matrix
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) < 50:
                logger.warning("Insufficient aligned data for factor analysis")
                return {}
            
            # Apply PCA
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_df)
            
            pca = PCA(n_components=n_factors)
            factors = pca.fit_transform(returns_scaled)
            
            # Create factor DataFrames
            factor_features = {}
            for i in range(n_factors):
                factor_features[f'market_factor_{i+1}'] = pd.Series(
                    factors[:, i], 
                    index=returns_df.index
                )
            
            logger.info(f"✅ Created {n_factors} market factors")
            logger.info(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
            
            return factor_features
            
        except Exception as e:
            logger.error(f"Error creating market factors: {str(e)}")
            return {}

# Global instance
cross_asset_engine = CrossAssetFeatureEngine()
