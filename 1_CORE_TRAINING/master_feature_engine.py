"""
Master Advanced Feature Engineering System
Combines all advanced feature engineering techniques for maximum ML performance
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Import our custom feature engines
from temporal_features import temporal_engine
from cross_asset_features import cross_asset_engine
from microstructure_features import microstructure_engine
from risk_features import risk_engine

# Enhanced features from existing system
try:
    from enhanced_features import enhanced_feature_engine
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logging.warning("Enhanced features module not available")

logger = logging.getLogger(__name__)

class MasterFeatureEngine:
    """
    Master feature engineering system combining all advanced techniques
    """
    
    def __init__(self, enable_all_features=True):
        self.enable_all_features = enable_all_features
        
        # Feature categories
        self.feature_categories = {
            'temporal': True,
            'cross_asset': True,
            'microstructure': True,
            'risk': True,
            'enhanced_technical': ENHANCED_AVAILABLE,
        }
        
        # Performance tracking
        self.feature_computation_times = {}
        self.feature_counts = {}
        
        logger.info("🧠 Master Feature Engine initialized")
        logger.info(f"   Available categories: {list(self.feature_categories.keys())}")
    
    def engineer_all_features(self, 
                            df: pd.DataFrame, 
                            symbol: str,
                            market_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                            feature_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Engineer all advanced features for a given asset
        
        Args:
            df: Main asset OHLCV DataFrame
            symbol: Asset symbol (e.g., 'BTC-USD')
            market_data_dict: Dictionary of other assets' data for cross-asset features
            feature_categories: List of feature categories to include (None = all)
        
        Returns:
            DataFrame with all engineered features
        """
        try:
            if df is None or df.empty:
                logger.error("Empty DataFrame provided")
                return pd.DataFrame()
            
            logger.info(f"🚀 Engineering advanced features for {symbol}")
            logger.info(f"   Input data shape: {df.shape}")
            
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return result_df
            
            # Convert to numeric and clean data
            result_df = self._clean_data(result_df)
            
            if len(result_df) < 50:
                logger.warning(f"Insufficient data for feature engineering: {len(result_df)} rows")
                return result_df
            
            # Determine which feature categories to compute
            categories_to_compute = feature_categories or list(self.feature_categories.keys())
            categories_to_compute = [cat for cat in categories_to_compute 
                                   if self.feature_categories.get(cat, False)]
            
            logger.info(f"   Computing features for categories: {categories_to_compute}")
            
            initial_features = len(result_df.columns)
            
            # 1. Enhanced Technical Indicators (from existing system)
            if 'enhanced_technical' in categories_to_compute and ENHANCED_AVAILABLE:
                result_df = self._add_enhanced_technical_features(result_df, symbol)
            
            # 2. Temporal Features
            if 'temporal' in categories_to_compute:
                result_df = self._add_temporal_features(result_df)
            
            # 3. Cross-Asset Features
            if 'cross_asset' in categories_to_compute and market_data_dict:
                result_df = self._add_cross_asset_features(result_df, symbol, market_data_dict)
            
            # 4. Market Microstructure Features
            if 'microstructure' in categories_to_compute:
                result_df = self._add_microstructure_features(result_df)
            
            # 5. Risk-Based Features
            if 'risk' in categories_to_compute:
                result_df = self._add_risk_features(result_df)
            
            # 6. Feature Interactions (Advanced)
            if self.enable_all_features:
                result_df = self._add_feature_interactions(result_df)
            
            # 7. Feature Selection and Cleaning
            result_df = self._clean_final_features(result_df)
            
            final_features = len(result_df.columns)
            added_features = final_features - initial_features
            
            logger.info(f"✅ Feature engineering complete for {symbol}")
            logger.info(f"   Final data shape: {result_df.shape}")
            logger.info(f"   Added {added_features} new features")
            logger.info(f"   Total features: {final_features}")
            
            # Store feature metadata
            self.feature_counts[symbol] = {
                'initial': initial_features,
                'final': final_features,
                'added': added_features,
                'categories': categories_to_compute
            }
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in master feature engineering for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return df
    
    def _clean_data(self, df):
        """Clean and prepare data for feature engineering"""
        try:
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN in OHLCV
            df = df.dropna(subset=numeric_cols)
            
            # Remove duplicate timestamps if index is timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                df = df[~df.index.duplicated(keep='first')]
            
            # Sort by timestamp/index
            df = df.sort_index()
            
            # Basic validation
            invalid_data = (
                (df['high'] < df['low']) |
                (df['close'] <= 0) |
                (df['volume'] < 0)
            )
            
            if invalid_data.any():
                logger.warning(f"Removing {invalid_data.sum()} invalid data points")
                df = df[~invalid_data]
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def _add_enhanced_technical_features(self, df, symbol):
        """Add enhanced technical indicators"""
        try:
            if not ENHANCED_AVAILABLE:
                return df
            
            import time
            start_time = time.time()
            
            # Use existing enhanced feature engine
            df = enhanced_feature_engine.calculate_enhanced_features(df, symbol)
            
            elapsed = time.time() - start_time
            self.feature_computation_times['enhanced_technical'] = elapsed
            logger.info(f"   ✅ Enhanced technical features: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding enhanced technical features: {str(e)}")
            return df
    
    def _add_temporal_features(self, df):
        """Add temporal/time-based features"""
        try:
            import time
            start_time = time.time()
            
            # Add temporal features
            df = temporal_engine.add_temporal_features(df)
            df = temporal_engine.add_fourier_features(df)
            df = temporal_engine.add_time_decay_features(df)
            
            elapsed = time.time() - start_time
            self.feature_computation_times['temporal'] = elapsed
            logger.info(f"   ✅ Temporal features: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {str(e)}")
            return df
    
    def _add_cross_asset_features(self, df, symbol, market_data_dict):
        """Add cross-asset correlation features"""
        try:
            import time
            start_time = time.time()
            
            # Add cross-asset features
            df = cross_asset_engine.add_cross_asset_features(df, symbol, market_data_dict)
            
            # Add market factor features
            factor_features = cross_asset_engine.create_market_factor_features(market_data_dict)
            
            # Merge factor features
            for factor_name, factor_series in factor_features.items():
                # Align with main DataFrame
                aligned_factor = pd.merge_asof(
                    df.sort_index(), 
                    factor_series.to_frame(factor_name).sort_index(),
                    left_index=True, right_index=True, direction='nearest'
                )
                if factor_name in aligned_factor.columns:
                    df[factor_name] = aligned_factor[factor_name]
            
            elapsed = time.time() - start_time
            self.feature_computation_times['cross_asset'] = elapsed
            logger.info(f"   ✅ Cross-asset features: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding cross-asset features: {str(e)}")
            return df
    
    def _add_microstructure_features(self, df):
        """Add market microstructure features"""
        try:
            import time
            start_time = time.time()
            
            # Add microstructure features
            df = microstructure_engine.add_microstructure_features(df)
            df = microstructure_engine.add_regime_features(df)
            
            elapsed = time.time() - start_time
            self.feature_computation_times['microstructure'] = elapsed
            logger.info(f"   ✅ Microstructure features: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding microstructure features: {str(e)}")
            return df
    
    def _add_risk_features(self, df):
        """Add risk-based features"""
        try:
            import time
            start_time = time.time()
            
            # Add risk features
            df = risk_engine.add_risk_features(df)
            df = risk_engine.add_regime_risk_features(df)
            
            elapsed = time.time() - start_time
            self.feature_computation_times['risk'] = elapsed
            logger.info(f"   ✅ Risk features: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding risk features: {str(e)}")
            return df
    
    def _add_feature_interactions(self, df):
        """Add advanced feature interactions"""
        try:
            import time
            start_time = time.time()
            
            # Select key features for interactions
            key_features = self._get_key_features_for_interactions(df)
            
            if len(key_features) >= 2:
                # Price-volume interactions
                if 'close' in df.columns and 'volume' in df.columns:
                    df['price_volume_interaction'] = df['close'] * np.log1p(df['volume'])
                
                # Volatility-momentum interactions
                vol_features = [col for col in df.columns if 'volatility' in col.lower()][:3]
                momentum_features = [col for col in df.columns if any(x in col.lower() for x in ['momentum', 'return'])][:3]
                
                for vol_feat in vol_features:
                    for mom_feat in momentum_features:
                        if vol_feat in df.columns and mom_feat in df.columns:
                            interaction_name = f"{vol_feat}_x_{mom_feat}"[:50]  # Limit name length
                            df[interaction_name] = df[vol_feat] * df[mom_feat]
                
                # Risk-return interactions
                risk_features = [col for col in df.columns if any(x in col.lower() for x in ['var', 'drawdown', 'risk'])][:2]
                return_features = [col for col in df.columns if 'return' in col.lower()][:2]
                
                for risk_feat in risk_features:
                    for ret_feat in return_features:
                        if risk_feat in df.columns and ret_feat in df.columns:
                            interaction_name = f"{risk_feat}_div_{ret_feat}"[:50]
                            df[interaction_name] = df[risk_feat] / (abs(df[ret_feat]) + 1e-8)
            
            elapsed = time.time() - start_time
            self.feature_computation_times['interactions'] = elapsed
            logger.info(f"   ✅ Feature interactions: {elapsed:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding feature interactions: {str(e)}")
            return df
    
    def _get_key_features_for_interactions(self, df):
        """Identify key features for interaction creation"""
        try:
            # Exclude basic OHLCV and timestamp columns
            exclude_patterns = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime']
            
            key_features = []
            for col in df.columns:
                if not any(pattern in col.lower() for pattern in exclude_patterns):
                    if df[col].dtype in ['float64', 'int64']:
                        # Check if feature has reasonable variance
                        if df[col].std() > 1e-8:
                            key_features.append(col)
            
            return key_features[:20]  # Limit to top 20 features to avoid explosion
            
        except Exception as e:
            logger.error(f"Error identifying key features: {str(e)}")
            return []
    
    def _clean_final_features(self, df):
        """Clean and finalize features"""
        try:
            # Remove features with too many NaN values
            nan_threshold = 0.5  # Remove features with >50% NaN
            for col in df.columns:
                if df[col].isnull().sum() / len(df) > nan_threshold:
                    df = df.drop(col, axis=1)
                    logger.debug(f"Removed feature {col} (too many NaNs)")
            
            # Fill remaining NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    # Forward fill, then backward fill, then fill with 0
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Remove features with zero variance
            for col in numeric_cols:
                if col in df.columns and df[col].std() < 1e-10:
                    df = df.drop(col, axis=1)
                    logger.debug(f"Removed feature {col} (zero variance)")
            
            logger.info(f"   🧹 Final cleanup complete: {df.shape[1]} features retained")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning final features: {str(e)}")
            return df
    
    def get_feature_importance_categories(self, df):
        """Categorize features by type for importance analysis"""
        try:
            feature_categories = {
                'price_based': [],
                'volume_based': [],
                'volatility_based': [],
                'momentum_based': [],
                'temporal_based': [],
                'risk_based': [],
                'microstructure_based': [],
                'cross_asset_based': [],
                'interaction_based': []
            }
            
            for col in df.columns:
                col_lower = col.lower()
                
                if any(x in col_lower for x in ['price', 'close', 'open', 'high', 'low']):
                    feature_categories['price_based'].append(col)
                elif any(x in col_lower for x in ['volume', 'obv']):
                    feature_categories['volume_based'].append(col)
                elif any(x in col_lower for x in ['vol', 'atr', 'volatility']):
                    feature_categories['volatility_based'].append(col)
                elif any(x in col_lower for x in ['momentum', 'roc', 'macd', 'return']):
                    feature_categories['momentum_based'].append(col)
                elif any(x in col_lower for x in ['hour', 'day', 'month', 'session', 'time']):
                    feature_categories['temporal_based'].append(col)
                elif any(x in col_lower for x in ['var', 'risk', 'drawdown', 'sharpe', 'sortino']):
                    feature_categories['risk_based'].append(col)
                elif any(x in col_lower for x in ['liquidity', 'spread', 'impact', 'tick']):
                    feature_categories['microstructure_based'].append(col)
                elif any(x in col_lower for x in ['corr', 'beta', 'factor', 'relative']):
                    feature_categories['cross_asset_based'].append(col)
                elif '_x_' in col_lower or '_div_' in col_lower:
                    feature_categories['interaction_based'].append(col)
            
            return feature_categories
            
        except Exception as e:
            logger.error(f"Error categorizing features: {str(e)}")
            return {}
    
    def get_computation_summary(self):
        """Get summary of feature computation performance"""
        total_time = sum(self.feature_computation_times.values())
        
        summary = {
            'total_computation_time': total_time,
            'category_times': self.feature_computation_times,
            'feature_counts': self.feature_counts
        }
        
        logger.info("📊 Feature Engineering Summary:")
        logger.info(f"   Total time: {total_time:.2f}s")
        for category, time_taken in self.feature_computation_times.items():
            percentage = (time_taken / total_time * 100) if total_time > 0 else 0
            logger.info(f"   {category}: {time_taken:.2f}s ({percentage:.1f}%)")
        
        return summary

# Global instance
master_feature_engine = MasterFeatureEngine()

# Convenience function for easy integration
def engineer_advanced_features(df, symbol, market_data_dict=None, categories=None):
    """
    Convenience function to engineer all advanced features
    
    Args:
        df: OHLCV DataFrame
        symbol: Asset symbol
        market_data_dict: Optional dict of other assets' data
        categories: Optional list of feature categories to include
    
    Returns:
        DataFrame with advanced features
    """
    return master_feature_engine.engineer_all_features(
        df, symbol, market_data_dict, categories
    )
