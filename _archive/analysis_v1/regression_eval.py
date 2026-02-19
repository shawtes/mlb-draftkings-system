#!/usr/bin/env python3
"""
Regression Model Evaluation - Price Prediction Focus
Evaluates only regression models for price prediction accuracy.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import get_coinbase_data, calculate_indicators

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegressionEvaluator:
    """Focused regression model evaluator for price predictions"""
    
    def __init__(self):
        self.models_dir = os.path.join(current_dir, 'models')
        self.results = []
        
    def get_regression_models(self):
        """Get only regression models (_model.pkl and _regressor.pkl files)"""
        models = {}
        
        if not os.path.exists(self.models_dir):
            print(f"❌ Models directory not found: {self.models_dir}")
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_model.pkl') or filename.endswith('_regressor.pkl'):  # Both patterns
                if filename.endswith('_model.pkl'):
                    parts = filename.replace('_model.pkl', '').split('_')
                elif filename.endswith('_regressor.pkl'):
                    parts = filename.replace('_regressor.pkl', '').split('_')
                
                if len(parts) >= 2:
                    granularity = int(parts[-1])
                    symbol_parts = parts[:-1]
                    symbol = '_'.join(symbol_parts)
                    
                    # Fix symbol format - handle different naming patterns
                    if symbol == 'USDT':
                        symbol = 'USDT-USD'
                    elif symbol == '00':
                        continue
                    elif symbol.endswith('USD') and not symbol.endswith('-USD'):
                        # Convert BTCUSD to BTC-USD
                        if len(symbol) > 3:
                            base = symbol[:-3]
                            symbol = f"{base}-USD"
                    elif not symbol.endswith('-USD') and not symbol.endswith('USD'):
                        symbol = symbol + '-USD'
                    
                    if symbol.startswith('-') or len(symbol) < 4:
                        continue
                    
                    if symbol not in models:
                        models[symbol] = []
                    models[symbol].append({
                        'granularity': granularity,
                        'filename': filename
                    })
        
        for symbol in models:
            models[symbol].sort(key=lambda x: x['granularity'])
            
        return models
    
    def get_model_features(self, model_path):
        """Get expected model features"""
        try:
            model = joblib.load(joblib.load(model_path)
            
            if hasattr(model, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_names_in_'):
                return list(model.feature_names_in_)
            
            # For older models, try to get from n_features_in_
            if hasattr(model, 'n_features_in_'):
                # Return a reasonable default set that matches the expected count
                default_features = [
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'sma_20', 'sma_50', 'upper_band', 'lower_band',
                    'ATR', '%K', '%D', 'volume'
                ]
                
                # Extend if model expects more features
                expected_count = model.n_features_in_
                if expected_count > len(default_features):
                    additional_features = [
                        'price_momentum_3', 'price_momentum_5', 'volume_ratio',
                        'bb_position', 'rsi_momentum', 'volume_sma_ratio',
                        'price_change_1h', 'price_change_4h', 'volatility'
                    ]
                    default_features.extend(additional_features[:expected_count - len(default_features)])
                
                return default_features[:expected_count]
            
            # Fallback feature set
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
        except Exception as e:
            print(f"⚠️ Could not load model features: {str(e)}")
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band'
            ]
    
    def evaluate_regression_model(self, symbol, model_info, test_days=7):
        """Evaluate a regression model for price predictions"""
        try:
            granularity = model_info['granularity']
            filename = model_info['filename']
            
            granularity_name = f"{granularity//3600}h" if granularity >= 3600 else f"{granularity//60}m"
            
            print(f"📈 Evaluating {symbol} ({granularity_name}) regression model")
            
            # Load model
            model_path = os.path.join(self.models_dir, filename)
            model = joblib.load(model_path)
            
            # Get data
            df = get_coinbase_data(symbol, granularity, days=test_days + 7)
            if df.empty:
                print(f"❌ No data for {symbol}")
                return None
            
            # Calculate indicators
            df = calculate_indicators(df)
            df = df.dropna()
            
            if len(df) < 20:
                print(f"❌ Insufficient data: {len(df)} rows")
                return None
            
            # Split data - use more recent data for testing
            test_size = min(len(df) // 3, test_days * 24)
            test_df = df[-test_size:].copy()
            
            if len(test_df) < 10:
                print(f"❌ Test set too small: {len(test_df)} rows")
                return None
            
            # Get features
            expected_features = self.get_model_features(model_path)
            available_features = [f for f in expected_features if f in test_df.columns]
            missing_features = [f for f in expected_features if f not in test_df.columns]
            
            # Create missing features with default values
            if missing_features:
                print(f"⚠️ Creating {len(missing_features)} missing features with defaults: {missing_features[:3]}...")
                for feature in missing_features:
                    if 'momentum' in feature:
                        # Price momentum features
                        if 'price_momentum_3' == feature:
                            test_df[feature] = test_df['close'].pct_change(3).fillna(0)
                        elif 'price_momentum_5' == feature:
                            test_df[feature] = test_df['close'].pct_change(5).fillna(0)
                        else:
                            test_df[feature] = test_df['close'].pct_change().fillna(0)
                    elif 'volume_ratio' == feature:
                        # Volume ratio feature
                        volume_sma = test_df['volume'].rolling(10).mean()
                        test_df[feature] = (test_df['volume'] / volume_sma).fillna(1.0)
                    elif 'bb_position' == feature:
                        # Bollinger Band position
                        if 'upper_band' in test_df.columns and 'lower_band' in test_df.columns:
                            bb_range = test_df['upper_band'] - test_df['lower_band']
                            test_df[feature] = ((test_df['close'] - test_df['lower_band']) / bb_range).fillna(0.5)
                        else:
                            test_df[feature] = 0.5
                    elif 'volatility' == feature:
                        # Price volatility
                        test_df[feature] = test_df['close'].pct_change().rolling(20).std().fillna(0.02)
                    else:
                        # Default to zero for unknown features
                        test_df[feature] = 0.0
                
                # Update available features
                available_features = expected_features
            
            if len(available_features) < len(expected_features) * 0.5:
                print(f"❌ Too few features available: {len(available_features)}/{len(expected_features)}")
                return None
            
            # Prepare target variable - next price percentage change
            test_df['next_price'] = test_df['close'].shift(-1)
            test_df['price_change_pct'] = ((test_df['next_price'] - test_df['close']) / test_df['close']) * 100
            test_df = test_df.dropna()
            
            if len(test_df) == 0:
                print(f"❌ No valid test data after price calculation")
                return None
            
            X_test = test_df[available_features].fillna(0)
            y_actual = test_df['price_change_pct'].values
            
            # Make predictions
            if not hasattr(model, 'predict'):
                print(f"❌ Model has no predict method")
                return None
                
            y_pred = model.predict(X_test)
            
            # Calculate regression metrics
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_actual, y_pred)
            
            # Correlation
            correlation, p_value = pearsonr(y_actual, y_pred)
            
            # Direction accuracy (sign prediction)
            actual_direction = np.sign(y_actual)
            pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Additional regression-specific metrics
            mean_actual = np.mean(np.abs(y_actual))
            mean_pred = np.mean(np.abs(y_pred))
            std_actual = np.std(y_actual)
            std_pred = np.std(y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_actual - y_pred) / np.where(np.abs(y_actual) > 0.01, y_actual, 1))) * 100
            
            # Price prediction accuracy within thresholds
            threshold_1pct = np.mean(np.abs(y_actual - y_pred) <= 1.0) * 100  # Within 1%
            threshold_2pct = np.mean(np.abs(y_actual - y_pred) <= 2.0) * 100  # Within 2%
            threshold_5pct = np.mean(np.abs(y_actual - y_pred) <= 5.0) * 100  # Within 5%
            
            # Explained variance score
            explained_var = 1 - (np.var(y_actual - y_pred) / np.var(y_actual))
            
            result = {
                'symbol': symbol,
                'granularity': granularity,
                'granularity_name': granularity_name,
                'test_samples': len(test_df),
                'features_used': len(available_features),
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'correlation': correlation,
                'direction_accuracy': direction_accuracy,
                'mean_actual': mean_actual,
                'mean_pred': mean_pred,
                'std_actual': std_actual,
                'std_pred': std_pred,
                'mape': mape,
                'threshold_1pct': threshold_1pct,
                'threshold_2pct': threshold_2pct,
                'threshold_5pct': threshold_5pct,
                'explained_variance': explained_var,
                'model_file': filename
            }
            
            print(f"✅ {symbol} ({granularity_name}): MAE={mae:.4f}%, R²={r2:.3f}, Dir={direction_accuracy:.1f}%")
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {symbol}: {str(e)}")
            return None
    
    def evaluate_all_regression_models(self, test_days=7):
        """Evaluate all regression models"""
        models = self.get_regression_models()
        
        if not models:
            print("❌ No regression models found")
            return []
        
        print(f"🔍 Found {len(models)} symbols with regression models")
        
        all_results = []
        for symbol, model_list in models.items():
            print(f"\n📊 Evaluating {symbol}...")
            for model_info in model_list:
                result = self.evaluate_regression_model(symbol, model_info, test_days)
                if result:
                    all_results.append(result)
        
        self.results = all_results
        return all_results
    
    def print_regression_summary(self):
        """Print summary of regression evaluation results"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        print(f"\n📊 REGRESSION MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Models Evaluated: {len(df)}")
        print(f"Average MAE: {df['mae'].mean():.4f}%")
        print(f"Average R²: {df['r2_score'].mean():.3f}")
        print(f"Average RMSE: {df['rmse'].mean():.4f}%")
        print(f"Average Correlation: {df['correlation'].mean():.3f}")
        print(f"Average Direction Accuracy: {df['direction_accuracy'].mean():.1f}%")
        print(f"Average Within 1%: {df['threshold_1pct'].mean():.1f}%")
        print(f"Average Within 2%: {df['threshold_2pct'].mean():.1f}%")
        
        print(f"\n🏆 BEST PERFORMERS:")
        
        # Best R²
        best_r2 = df.loc[df['r2_score'].idxmax()]
        print(f"Best R²: {best_r2['symbol']} ({best_r2['granularity_name']}) = {best_r2['r2_score']:.3f}")
        
        # Best MAE
        best_mae = df.loc[df['mae'].idxmin()]
        print(f"Best MAE: {best_mae['symbol']} ({best_mae['granularity_name']}) = {best_mae['mae']:.4f}%")
        
        # Best Direction Accuracy
        best_dir = df.loc[df['direction_accuracy'].idxmax()]
        print(f"Best Direction: {best_dir['symbol']} ({best_dir['granularity_name']}) = {best_dir['direction_accuracy']:.1f}%")
        
        # Performance distribution
        positive_r2 = df[df['r2_score'] > 0]
        good_direction = df[df['direction_accuracy'] > 50]
        good_mae = df[df['mae'] < 2.0]
        
        print(f"\n📈 PERFORMANCE DISTRIBUTION:")
        print(f"Models with positive R²: {len(positive_r2)}/{len(df)} ({len(positive_r2)/len(df)*100:.1f}%)")
        print(f"Models with >50% direction accuracy: {len(good_direction)}/{len(df)} ({len(good_direction)/len(df)*100:.1f}%)")
        print(f"Models with MAE < 2%: {len(good_mae)}/{len(df)} ({len(good_mae)/len(df)*100:.1f}%)")
    
    def save_regression_results(self, filename=None):
        """Save results to CSV"""
        if not self.results:
            print("❌ No results to save")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regression_eval_{timestamp}.csv"
        
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Main function to run regression evaluation"""
    evaluator = RegressionEvaluator()
    results = evaluator.evaluate_all_regression_models(test_days=7)
    
    if results:
        evaluator.print_regression_summary()
        evaluator.save_regression_results()
    else:
        print("❌ No models found to evaluate")

if __name__ == "__main__":
    main()