#!/usr/bin/env python3
"""
Standalone Train and Evaluate 15 Regression Models
No dependencies on maybe.py - completely self-contained.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandaloneTrainEval:
    """Completely standalone training and evaluation"""
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.results = []
        
    def get_symbols(self):
        """Get symbols for training"""
        return [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
            'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD'
        ]
    
    def get_coinbase_data(self, symbol, days=30):
        """Get data from Coinbase API"""
        try:
            # Try different import approaches for Coinbase
            try:
                from coinbase.rest import RESTClient
                # Try to import config
                try:
                    from config import KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
                except ImportError:
                    # Try lk.py
                    try:
                        from lk import KEY_NAME, PRIVATE_KEY_PEM, BASE_URL
                    except ImportError:
                        print("❌ Could not import Coinbase credentials")
                        return None
                        
                client = RESTClient(
                    api_key=KEY_NAME,
                    api_secret=PRIVATE_KEY_PEM,
                    base_url=BASE_URL
                )
                
                # Calculate time range
                end = datetime.now()
                start = end - timedelta(days=days)
                
                # Try different granularity formats
                try:
                    # Try new API format
                    response = client.get_candles(
                        product_id=symbol,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        granularity="ONE_HOUR"
                    )
                except Exception as e:
                    if "granularity" in str(e):
                        # Try older format
                        response = client.get_candles(
                            product_id=symbol,
                            start=start.isoformat(),
                            end=end.isoformat(),
                            granularity=3600
                        )
                    else:
                        raise e
                
                if not hasattr(response, 'candles') or not response.candles:
                    return None
                    
                # Convert to DataFrame
                data = []
                for candle in response.candles:
                    data.append({
                        'timestamp': pd.to_datetime(candle.start, unit='s'),
                        'open': float(candle.open),
                        'high': float(candle.high),
                        'low': float(candle.low),
                        'close': float(candle.close),
                        'volume': float(candle.volume)
                    })
                
                df = pd.DataFrame(data)
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
                
            except ImportError as e:
                print(f"❌ Coinbase SDK not available: {str(e)}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
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
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['upper_band'] = bb_middle + (bb_std * 2)
            df['lower_band'] = bb_middle - (bb_std * 2)
            
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
            print(f"❌ Error calculating indicators: {str(e)}")
            return df
    
    def train_model(self, symbol):
        """Train a single model"""
        try:
            print(f"🔨 Training {symbol}...")
            
            # Get data
            df = self.get_coinbase_data(symbol, days=90)
            if df is None or len(df) < 200:
                print(f"❌ Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
                return False
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if len(df) < 100:
                print(f"❌ Insufficient data after indicators for {symbol}: {len(df)} rows")
                return False
            
            # Feature columns
            features = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
            
            # Create target (next price change percentage)
            df['next_price'] = df['close'].shift(-1)
            df['target'] = ((df['next_price'] - df['close']) / df['close']) * 100
            df = df.dropna()
            
            if len(df) < 50:
                print(f"❌ Insufficient data after target creation for {symbol}: {len(df)} rows")
                return False
            
            # Prepare data
            X = df[features].fillna(0)
            y = df['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Test model quickly
            y_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            # Save model
            symbol_clean = symbol.replace('-', '')
            model_filename = f"{symbol_clean}_regressor.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(model, model_path)
            
            print(f"✅ {symbol} trained: MAE={test_mae:.4f}%, R²={test_r2:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Error training {symbol}: {str(e)}")
            return False
    
    def evaluate_model(self, symbol):
        """Evaluate a trained model"""
        try:
            # Load model
            symbol_clean = symbol.replace('-', '')
            model_filename = f"{symbol_clean}_regressor.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            
            if not os.path.exists(model_path):
                return None
                
            model = joblib.load(model_path)
            
            # Get fresh test data (different from training)
            df = self.get_coinbase_data(symbol, days=30)
            if df is None or len(df) < 50:
                return None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            if len(df) < 20:
                return None
            
            # Use only recent data for testing
            test_df = df[-20:].copy()  # Last 20 points
            
            # Features
            features = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
            
            # Create target
            test_df['next_price'] = test_df['close'].shift(-1)
            test_df['target'] = ((test_df['next_price'] - test_df['close']) / test_df['close']) * 100
            test_df = test_df.dropna()
            
            if len(test_df) < 10:
                return None
            
            # Prepare test data
            X_test = test_df[features].fillna(0)
            y_actual = test_df['target'].values
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)
            
            # Correlation
            try:
                correlation, _ = pearsonr(y_actual, y_pred)
            except:
                correlation = 0.0
            
            # Direction accuracy
            actual_direction = np.sign(y_actual)
            pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Threshold accuracy
            within_1pct = np.mean(np.abs(y_actual - y_pred) <= 1.0) * 100
            within_2pct = np.mean(np.abs(y_actual - y_pred) <= 2.0) * 100
            
            result = {
                'symbol': symbol,
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
    
    def run_training_and_evaluation(self, count=15):
        """Run complete training and evaluation"""
        print("🎯 Standalone Train and Evaluate 15 Regression Models")
        print("=" * 65)
        
        symbols = self.get_symbols()
        
        # Limit to requested count
        symbols = symbols[:count]
        
        print(f"📋 Planning to train {len(symbols)} models:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2d}. {symbol}")
        
        # Training phase
        print(f"\n🚀 Training Phase...")
        print("=" * 40)
        
        trained_count = 0
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i:2d}/{len(symbols)}] Training {symbol}...")
            if self.train_model(symbol):
                trained_count += 1
            time.sleep(1)  # Brief delay
        
        print(f"\n📊 Training Summary: {trained_count}/{len(symbols)} models trained successfully")
        
        if trained_count == 0:
            print("❌ No models were trained successfully")
            return
        
        # Evaluation phase
        print(f"\n🔍 Evaluation Phase...")
        print("=" * 40)
        
        results = []
        for symbol in symbols:
            result = self.evaluate_model(symbol)
            if result:
                results.append(result)
        
        self.results = results
        
        if results:
            self.print_summary()
            self.save_results()
        else:
            print("❌ No models could be evaluated")
        
        print(f"\n✅ Complete! Trained {trained_count} models, evaluated {len(results)} models")
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n📊 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Models Evaluated: {len(df)}")
        print(f"Average MAE: {df['mae'].mean():.4f}%")
        print(f"Average R²: {df['r2_score'].mean():.3f}")
        print(f"Average RMSE: {df['rmse'].mean():.4f}%")
        print(f"Average Correlation: {df['correlation'].mean():.3f}")
        print(f"Average Direction Accuracy: {df['direction_accuracy'].mean():.1f}%")
        print(f"Average Within 1%: {df['within_1pct'].mean():.1f}%")
        print(f"Average Within 2%: {df['within_2pct'].mean():.1f}%")
        
        print(f"\n🏆 BEST PERFORMERS:")
        
        # Best R²
        best_r2 = df.loc[df['r2_score'].idxmax()]
        print(f"Best R²: {best_r2['symbol']} = {best_r2['r2_score']:.3f}")
        
        # Best MAE  
        best_mae = df.loc[df['mae'].idxmin()]
        print(f"Best MAE: {best_mae['symbol']} = {best_mae['mae']:.4f}%")
        
        # Best Direction
        best_dir = df.loc[df['direction_accuracy'].idxmax()]
        print(f"Best Direction: {best_dir['symbol']} = {best_dir['direction_accuracy']:.1f}%")
        
        # Performance stats
        positive_r2 = df[df['r2_score'] > 0]
        good_direction = df[df['direction_accuracy'] > 50]
        
        print(f"\n📈 PERFORMANCE STATS:")
        print(f"Models with positive R²: {len(positive_r2)}/{len(df)} ({len(positive_r2)/len(df)*100:.1f}%)")
        print(f"Models with >50% direction accuracy: {len(good_direction)}/{len(df)} ({len(good_direction)/len(df)*100:.1f}%)")
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"standalone_eval_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to {filename}")

def main():
    """Main function"""
    trainer = StandaloneTrainEval()
    trainer.run_training_and_evaluation(count=15)

if __name__ == "__main__":
    main() 