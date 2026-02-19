"""
Advanced Temporal Feature Engineering for Cryptocurrency Trading
Captures time-based patterns, seasonality, and cyclical behaviors
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from scipy import signal
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)

class TemporalFeatureEngine:
    """
    Advanced temporal feature engineering for crypto trading
    """
    
    def __init__(self):
        self.crypto_timezone = pytz.UTC
    
    def add_temporal_features(self, df):
        """
        Add comprehensive temporal features
        """
        try:
            # Ensure we have timestamp
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.warning("No timestamp column found, using index")
                df['timestamp'] = pd.to_datetime(df.index)
            
            # Basic time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Cyclical encoding (preserves temporal relationships)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Market session features (crypto is 24/7 but has patterns)
            # Asian session: 21:00-06:00 UTC
            # European session: 06:00-15:00 UTC  
            # US session: 13:00-22:00 UTC
            df['asian_session'] = ((df['hour'] >= 21) | (df['hour'] <= 6)).astype(int)
            df['european_session'] = ((df['hour'] >= 6) & (df['hour'] <= 15)).astype(int)
            df['us_session'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
            df['overlap_eu_us'] = ((df['hour'] >= 13) & (df['hour'] <= 15)).astype(int)
            
            # Weekend effect (lower volume typically)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Month-end effects (often higher volatility)
            df['month_end'] = (df['day_of_month'] >= 28).astype(int)
            
            # Time since last major event (you can customize these)
            df = self._add_event_features(df)
            
            # Seasonal momentum
            df = self._add_seasonal_momentum(df)
            
            # Time-based volatility regimes
            df = self._add_time_volatility_regimes(df)
            
            logger.info("✅ Added temporal features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {str(e)}")
            return df
    
    def _add_event_features(self, df):
        """Add features based on known market events"""
        # Options expiry dates (monthly, quarterly)
        df['days_to_month_end'] = (df['timestamp'].dt.days_in_month - df['timestamp'].dt.day)
        df['is_expiry_week'] = (df['days_to_month_end'] <= 7).astype(int)
        
        # Quarter end effects
        quarter_end_months = [3, 6, 9, 12]
        df['is_quarter_end'] = (df['month'].isin(quarter_end_months) & 
                               (df['days_to_month_end'] <= 7)).astype(int)
        
        return df
    
    def _add_seasonal_momentum(self, df):
        """Add seasonal momentum features"""
        # Calculate rolling returns at different periods
        for period in [7, 14, 30, 90]:  # Weekly, bi-weekly, monthly, quarterly
            df[f'return_{period}d'] = df['close'].pct_change(period)
              # Seasonal ranking (how does current period compare to historical)
            df[f'seasonal_rank_{period}d'] = (
                df[f'return_{period}d'].rolling(window=252, min_periods=period)
                .rank(pct=True)
            )        
        return df
    
    def _add_time_volatility_regimes(self, df):
        """Add time-based volatility regime features"""
        try:
            # Calculate hourly volatility patterns with safe groupby
            hourly_returns = df['close'].pct_change()
            if not hourly_returns.isna().all() and len(df) > 1:
                hourly_vol = df.groupby(df['hour'])['close'].pct_change().std()
                # Ensure hourly_vol is a pandas Series and handle NaN values
                if hasattr(hourly_vol, 'fillna'):
                    hourly_vol = hourly_vol.fillna(0)
                elif isinstance(hourly_vol, (int, float, np.number)):
                    # Convert single value to Series
                    hourly_vol = pd.Series([hourly_vol if not pd.isna(hourly_vol) else 0], 
                                         index=[df['hour'].iloc[0]])
                df['hourly_vol_regime'] = df['hour'].map(hourly_vol).fillna(0)
            else:
                df['hourly_vol_regime'] = 0
            
            # Day of week volatility patterns with safe groupby
            daily_returns = df['close'].pct_change()
            if not daily_returns.isna().all() and len(df) > 1:
                daily_vol = df.groupby(df['day_of_week'])['close'].pct_change().std()
                # Ensure daily_vol is a pandas Series and handle NaN values
                if hasattr(daily_vol, 'fillna'):
                    daily_vol = daily_vol.fillna(0)
                elif isinstance(daily_vol, (int, float, np.number)):
                    # Convert single value to Series
                    daily_vol = pd.Series([daily_vol if not pd.isna(daily_vol) else 0], 
                                        index=[df['day_of_week'].iloc[0]])
                df['daily_vol_regime'] = df['day_of_week'].map(daily_vol).fillna(0)
            else:
                df['daily_vol_regime'] = 0
            
            return df
        except Exception as e:
            logger.error(f"Error in time volatility regimes: {str(e)}")
            # Add default columns if there's an error
            df['hourly_vol_regime'] = 0
            df['daily_vol_regime'] = 0
            return df
    
    def add_fourier_features(self, df, n_components=5):
        """
        Add Fourier transform features to capture cyclical patterns
        """
        try:
            if len(df) < 100:
                logger.warning("Insufficient data for Fourier features")
                return df
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Apply FFT
            fft_values = fft(returns.values)
            frequencies = fftfreq(len(returns))
            
            # Get the most significant frequency components
            magnitude = np.abs(fft_values)
            top_freq_indices = np.argsort(magnitude)[-n_components:]
            
            # Create features from top frequency components
            for i, freq_idx in enumerate(top_freq_indices):
                if freq_idx < len(fft_values):
                    frequency = frequencies[freq_idx]
                    phase = np.angle(fft_values[freq_idx])
                    
                    # Create cyclical features
                    time_index = np.arange(len(df))
                    df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * frequency * time_index + phase)
                    df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * frequency * time_index + phase)
            
            logger.info(f"✅ Added {n_components} Fourier components")
            return df
            
        except Exception as e:
            logger.error(f"Error adding Fourier features: {str(e)}")
            return df
    
    def add_time_decay_features(self, df):
        """
        Add time decay features (recent data is more important)
        """
        try:
            # Time since start (for trend analysis)
            df['time_index'] = np.arange(len(df))
            df['time_normalized'] = df['time_index'] / len(df)
            
            # Exponential time decay weights
            decay_factors = [0.9, 0.95, 0.99]
            for decay in decay_factors:
                weights = decay ** np.arange(len(df))[::-1]  # Recent data gets higher weight
                df[f'time_weight_{int(decay*100)}'] = weights
              # Recency features for price levels with safe division
            for period in [5, 10, 20, 50]:
                recent_high = df['high'].rolling(window=period).max()
                recent_low = df['low'].rolling(window=period).min()
                
                # Safe division to avoid divide by zero
                range_diff = recent_high - recent_low
                df[f'price_position_{period}'] = np.where(
                    range_diff != 0,
                    (df['close'] - recent_low) / range_diff,
                    0.5  # Default to middle position if no range
                )
                
                df[f'time_since_high_{period}'] = (
                    df.groupby((df['high'] == recent_high).cumsum()).cumcount()
                )
                df[f'time_since_low_{period}'] = (
                    df.groupby((df['low'] == recent_low).cumsum()).cumcount()
                )
            
            logger.info("✅ Added time decay features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding time decay features: {str(e)}")
            return df

# Global instance
temporal_engine = TemporalFeatureEngine()
