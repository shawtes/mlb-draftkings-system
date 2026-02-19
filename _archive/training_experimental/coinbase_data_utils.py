#!/usr/bin/env python3
"""
Coinbase Data Utilities
Simple module to provide get_coinbase_data and calculate_indicators functions
"""

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

def get_coinbase_data(symbol='BTC-USD', granularity=3600, days=7):
    """
    Fetch historical data from Coinbase with improved rate limit handling.
    """
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    headers = {'Accept': 'application/json'}
    
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        
        params = {
            'granularity': granularity,
            'start': start.isoformat(),
            'end': end.isoformat()
        }
        
        max_retries = 3
        base_delay = 1
        
        for retry in range(max_retries):
            try:
                r = requests.get(url, headers=headers, params=params)
                
                if r.status_code == 429:  # Rate limit hit
                    delay = base_delay * (2 ** retry)
                    logger.warning(f"Rate limit hit, waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                elif r.status_code != 200:
                    logger.error(f"Error {r.status_code} fetching data for {symbol}: {r.text}")
                    return None
                    
                data = r.json()
                if not data:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                    
                df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df = df.astype({
                    'timestamp': 'float64',
                    'low': 'float64',
                    'high': 'float64',
                    'open': 'float64',
                    'close': 'float64',
                    'volume': 'float64'
                })
                
                # Convert timestamp and sort
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {retry + 1} failed for {symbol}: {str(e)}")
                if retry == max_retries - 1:
                    raise
                time.sleep(base_delay * (retry + 1))
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_indicators(df, symbol=None):
    """Calculate technical indicators"""
    try:
        if df is None or df.empty:
            logger.warning("Empty dataframe passed to calculate_indicators")
            return df
            
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Basic indicators
        df['EMA12'] = df['close'].ewm(span=12).mean()
        df['EMA26'] = df['close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # Stochastic Oscillator
        df['%K'] = ((df['close'] - df['low'].rolling(window=14).min()) / 
                    (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Lagging indicators
        df['lag_1'] = df['close'].shift(1)
        df['lag_2'] = df['close'].shift(2)
        df['lag_3'] = df['close'].shift(3)
        
        # Volume indicators
        df['OBV'] = calculate_obv(df)
        
        # Volatility
        df['ATR'] = calculate_atr(df)
        df['rolling_std_10'] = df['close'].rolling(window=10).std()
        
        # Price change indicators
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['price_vs_sma20'] = (df['close'] - df['MA20']) / df['MA20']
        
        logger.info(f"Calculated {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} indicators")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index, dtype=float)

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    try:
        obv = []
        obv_value = 0
        
        for i in range(len(df)):
            if i == 0:
                obv.append(df['volume'].iloc[i])
                obv_value = df['volume'].iloc[i]
            else:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv_value += df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv_value -= df['volume'].iloc[i]
                obv.append(obv_value)
        
        return pd.Series(obv, index=df.index)
    except Exception as e:
        logger.error(f"Error calculating OBV: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(index=df.index, dtype=float)
