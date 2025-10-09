#!/usr/bin/env python3
"""
Simple Train and Evaluate Script
Directly trains and evaluates regression models without complex imports.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
import logging
from coinbase.rest import Granularity

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTrainEvaluate:
    """Simple training and evaluation for regression models"""
    
    def __init__(self):
        self.models_dir = os.path.join(current_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.results = []
        
    def get_test_symbols(self):
        """Get test symbols for training"""
        return [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
            'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD'
        ]
    
    def get_data_and_indicators(self, symbol, granularity=3600, days=30):
        """Get data and calculate basic indicators"""
        try:
            from config import KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
            from coinbase.rest import RESTClient
            
            # Initialize client
            client = RESTClient(
                api_key=KEY_NAME,
                api_secret=PRIVATE_KEY_PEM,
                base_url=BASE_URL
            )
            
            # Calculate period
            end = datetime.now()
            start = end - pd.Timedelta(days=days)
            
            # Get candles
            # Map granularity seconds to Coinbase enum
            granularity_map = {
                60: Granularity.ONE_MINUTE,
                300: Granularity.FIVE_MINUTE,
                900: Granularity.FIFTEEN_MINUTE,
                1800: Granularity.THIRTY_MINUTE,
                3600: Granularity.ONE_HOUR,
                7200: Granularity.TWO_HOUR,
                21600: Granularity.SIX_HOUR,
                86400: Granularity.ONE_DAY
            }
            
            cb_granularity = granularity_map.get(granularity, Granularity.ONE_HOUR)
            
            response = client.get_candles(
                product_id=symbol,
                start=start.isoformat(),
                end=end.isoformat(),
                granularity=cb_granularity
            )
            
            if not hasattr(response, 'candles') or not response.candles:
                return None
                
            # Convert to DataFrame
            candles_data = []
            for candle in response.candles:
                candles_data.append({
                    'timestamp': pd.to_datetime(candle.start, unit='s'),
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                })
            
            df = pd.DataFrame(candles_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate basic indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # SMA
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['upper_band'] = df['bb_middle'] + (bb_std * 2)
            df['lower_band'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['ATR'] = df['tr'].rolling(14).mean()
            
            # Stochastic
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['%D'] = df['%K'].rolling(3).mean()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None
    
    def train_single_model(self, symbol, granularity=3600):
        """Train a single regression model"""
        try:
            print(f"🔨 Training {symbol} regression model...")
            
            # Get data
            df = self.get_data_and_indicators(symbol, granularity, days=60)
            if df is None or len(df) < 100:
                print(f"❌ Insufficient data for {symbol}")
                return False
            
            # Prepare features
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
            
            # Create target (next price change percentage)
            df['next_price'] = df['close'].shift(-1)
            df['target'] = ((df['next_price'] - df['close']) / df['close']) * 100
            df = df.dropna()
            
            if len(df) < 50:
                print(f"❌ Insufficient data after processing for {symbol}")
                return False
            
            # Prepare data
            X = df[feature_columns].fillna(0)
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            
            # Save model
            symbol_clean = symbol.replace('-', '')
            model_filename = f"{symbol_clean}_{granularity}_regressor.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(rf_model, model_path)
            
            print(f"✅ {symbol} model saved to {model_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {str(e)}")
            return False
    
    def evaluate_single_model(self, symbol, granularity=3600):
        """Evaluate a single model"""
        try:
            # Load model
            symbol_clean = symbol.replace('-', '')
            model_filename = f"{symbol_clean}_{granularity}_regressor.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            
            if not os.path.exists(model_path):
                return None
                
            model = joblib.load(model_path)
            
            # Get fresh test data
            df = self.get_data_and_indicators(symbol, granularity, days=14)
            if df is None or len(df) < 20:
                return None
            
            # Prepare features (last 50% as test)
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
            
            # Create target
            df['next_price'] = df['close'].shift(-1)
            df['target'] = ((df['next_price'] - df['close']) / df['close']) * 100
            df = df.dropna()
            
            # Use last 50% as test data
            test_size = len(df) // 2
            test_df = df[-test_size:].copy()
            
            X_test = test_df[feature_columns].fillna(0)
            y_actual = test_df['target'].values
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)
            
            # Correlation
            correlation, p_value = pearsonr(y_actual, y_pred)
            
            # Direction accuracy
            actual_direction = np.sign(y_actual)
            pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Threshold accuracy
            within_1pct = np.mean(np.abs(y_actual - y_pred) <= 1.0) * 100
            within_2pct = np.mean(np.abs(y_actual - y_pred) <= 2.0) * 100
            
            result = {
                'symbol': symbol,
                'granularity': granularity,
                'timeframe': f"{granularity//3600}h" if granularity >= 3600 else f"{granularity//60}m",
                'test_samples': len(test_df),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'correlation': correlation,
                'direction_accuracy': direction_accuracy,
                'within_1pct': within_1pct,
                'within_2pct': within_2pct,
                'model_file': model_filename
            }
            
            print(f"📊 {symbol}: MAE={mae:.4f}%, R²={r2:.3f}, Dir={direction_accuracy:.1f}%")
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {symbol}: {str(e)}")
            return None
    
    def train_and_evaluate_models(self, count=15):
        """Train and evaluate multiple models"""
        print("🎯 Simple Train and Evaluate 15 Regression Models")
        print("=" * 60)
        
        symbols = self.get_test_symbols()
        timeframes = [900, 3600]  # 15m, 1h
        
        # Plan training
        training_plan = []
        for symbol in symbols:
            for granularity in timeframes:
                training_plan.append((symbol, granularity))
                if len(training_plan) >= count:
                    break
            if len(training_plan) >= count:
                break
        
        print(f"📋 Training {len(training_plan)} models...")
        
        # Train models
        trained_count = 0
        for i, (symbol, granularity) in enumerate(training_plan, 1):
            timeframe_name = f"{granularity//3600}h" if granularity >= 3600 else f"{granularity//60}m"
            print(f"\n[{i:2d}/{len(training_plan)}] Training {symbol} ({timeframe_name})...")
            
            if self.train_single_model(symbol, granularity):
                trained_count += 1
            
            time.sleep(1)  # Small delay
        
        print(f"\n📊 Training Summary: {trained_count}/{len(training_plan)} models trained")
        
        if trained_count == 0:
            print("❌ No models were trained successfully")
            return
        
        # Wait and evaluate
        print(f"\n⏱️ Waiting 3 seconds before evaluation...")
        time.sleep(3)
        
        print(f"\n🔍 Evaluating Models...")
        print("=" * 60)
        
        # Evaluate models
        results = []
        for symbol, granularity in training_plan:
            result = self.evaluate_single_model(symbol, granularity)
            if result:
                results.append(result)
        
        self.results = results
        
        if results:
            # Print summary
            self.print_summary()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_train_eval_{timestamp}.csv"
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")
            
        print(f"\n✅ Training and evaluation complete!")
        return len(results) > 0
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Models Evaluated: {len(df)}")
        print(f"   Average MAE: {df['mae'].mean():.4f}%")
        print(f"   Average R²: {df['r2_score'].mean():.3f}")
        print(f"   Average Correlation: {df['correlation'].mean():.3f}")
        print(f"   Average Direction Accuracy: {df['direction_accuracy'].mean():.1f}%")
        
        print(f"\n🏆 BEST PERFORMERS:")
        
        # Best R²
        best_r2 = df.loc[df['r2_score'].idxmax()]
        print(f"   Best R²: {best_r2['symbol']} ({best_r2['timeframe']}) = {best_r2['r2_score']:.3f}")
        
        # Best MAE
        best_mae = df.loc[df['mae'].idxmin()]
        print(f"   Best MAE: {best_mae['symbol']} ({best_mae['timeframe']}) = {best_mae['mae']:.4f}%")
        
        # Best direction
        best_dir = df.loc[df['direction_accuracy'].idxmax()]
        print(f"   Best Direction: {best_dir['symbol']} ({best_dir['timeframe']}) = {best_dir['direction_accuracy']:.1f}%")
        
        # Good models
        good_r2 = df[df['r2_score'] > 0]
        good_direction = df[df['direction_accuracy'] > 50]
        print(f"   Models with positive R²: {len(good_r2)}/{len(df)}")
        print(f"   Models with >50% direction: {len(good_direction)}/{len(df)}")

def main():
    """Main function"""
    trainer = SimpleTrainEvaluate()
    trainer.train_and_evaluate_models(count=15)

if __name__ == "__main__":
    main() 