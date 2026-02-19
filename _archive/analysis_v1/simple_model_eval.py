#!/usr/bin/env python3
"""
Simple Model Evaluation - Standard ML Metrics
Evaluates models using MAE, R², MSE, RMSE, direction accuracy, and correlation.
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

class SimpleModelEvaluator:
    """Simple ML metrics evaluator"""
    
    def __init__(self):
        self.models_dir = os.path.join(current_dir, 'models')
        self.results = []
        
    def get_available_models(self):
        """Get list of available models"""
        models = {}
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('_model.pkl') or filename.endswith('_clf.pkl'):
                if filename.endswith('_model.pkl'):
                    parts = filename.replace('_model.pkl', '').split('_')
                    model_type = 'regression'
                elif filename.endswith('_clf.pkl'):
                    parts = filename.replace('_clf.pkl', '').split('_')
                    model_type = 'classification'
                
                if len(parts) >= 2:
                    granularity = int(parts[-1])
                    symbol_parts = parts[:-1]
                    symbol = '_'.join(symbol_parts)
                    
                    # Fix symbol format
                    if symbol == 'USDT':
                        symbol = 'USDT-USD'
                    elif symbol == '00':
                        continue
                    elif not symbol.endswith('-USD') and not symbol.endswith('USD'):
                        symbol = symbol + '-USD'
                    elif symbol.endswith('USD') and not symbol.endswith('-USD'):
                        symbol = symbol.replace('USD', '-USD')
                    
                    if symbol.startswith('-') or len(symbol) < 4:
                        continue
                    
                    if symbol not in models:
                        models[symbol] = []
                    models[symbol].append({
                        'granularity': granularity,
                        'type': model_type,
                        'filename': filename
                    })
        
        for symbol in models:
            models[symbol].sort(key=lambda x: x['granularity'])
            
        return models
    
    def get_model_features(self, model_path):
        """Get model features"""
        try:
            model = joblib.load(joblib.load(model_path)
            
            if hasattr(model, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_names_in_'):
                return list(model.feature_names_in_)
            
            # Default feature set
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'ATR', '%K', '%D', 'volume'
            ]
        except:
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band'
            ]
    
    def evaluate_model(self, symbol, model_info, test_days=7):
        """Evaluate a single model with standard ML metrics"""
        try:
            granularity = model_info['granularity']
            model_type = model_info['type']
            filename = model_info['filename']
            
            granularity_name = f"{granularity//3600}h" if granularity >= 3600 else f"{granularity//60}m"
            
            print(f"📊 Evaluating {symbol} ({granularity_name}, {model_type})")
            
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
            
            if len(df) < 10:
                print(f"❌ Insufficient data: {len(df)} rows")
                return None
            
            # Split data
            test_size = min(len(df) // 3, test_days * 24)
            test_df = df[-test_size:].copy()
            
            if len(test_df) < 5:
                print(f"❌ Test set too small: {len(test_df)} rows")
                return None
            
            # Get features
            expected_features = self.get_model_features(model_path)
            feature_columns = [f for f in expected_features if f in test_df.columns]
            
            if len(feature_columns) < len(expected_features) * 0.5:
                print(f"❌ Too few features: {len(feature_columns)}/{len(expected_features)}")
                return None
            
            # Prepare target variable (price change percentage)
            test_df['next_price'] = test_df['close'].shift(-1)
            test_df['price_change_pct'] = ((test_df['next_price'] - test_df['close']) / test_df['close']) * 100
            test_df = test_df.dropna()
            
            if len(test_df) == 0:
                print(f"❌ No valid test data after price calculation")
                return None
            
            X_test = test_df[feature_columns].fillna(0)
            y_actual = test_df['price_change_pct'].values
            
            # Make predictions
            if model_type == 'regression':
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    print(f"❌ Model has no predict method")
                    return None
            else:  # classification
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.shape[1] >= 3:
                        # Convert to price change: SELL=-2%, HOLD=0%, BUY=+2%
                        y_pred = (y_pred_proba[:, 2] - y_pred_proba[:, 0]) * 2.0
                    else:
                        y_pred = (y_pred_proba[:, 1] - y_pred_proba[:, 0]) * 1.0
                elif hasattr(model, 'predict'):
                    y_pred_class = model.predict(X_test)
                    y_pred = np.where(y_pred_class == 2, 2.0,
                                     np.where(y_pred_class == 1, 0.0, -2.0))
                else:
                    print(f"❌ Classification model has no predict method")
                    return None
            
            # Calculate standard ML metrics
            mae = mean_absolute_error(y_actual, y_pred)
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            
            # R² score (coefficient of determination)
            r2 = r2_score(y_actual, y_pred)
            
            # Correlation coefficient
            correlation, p_value = pearsonr(y_actual, y_pred)
            
            # Direction accuracy
            actual_direction = np.sign(y_actual)
            pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Additional metrics
            mean_actual = np.mean(np.abs(y_actual))
            mean_pred = np.mean(np.abs(y_pred))
            std_actual = np.std(y_actual)
            std_pred = np.std(y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error) - careful with zeros
            mape = np.mean(np.abs((y_actual - y_pred) / np.where(y_actual != 0, y_actual, 1))) * 100
            
            result = {
                'symbol': symbol,
                'granularity': granularity,
                'granularity_name': granularity_name,
                'model_type': model_type,
                'test_samples': len(test_df),
                'features_used': len(feature_columns),
                
                # Core ML metrics
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'correlation': correlation,
                'correlation_p_value': p_value,
                'direction_accuracy': direction_accuracy,
                'mape': mape,
                
                # Distribution metrics
                'mean_actual': mean_actual,
                'mean_pred': mean_pred,
                'std_actual': std_actual,
                'std_pred': std_pred,
                
                'model_class': type(model).__name__
            }
            
            print(f"✅ {symbol} ({granularity_name}): MAE={mae:.4f}, R²={r2:.3f}, Dir_Acc={direction_accuracy:.1f}%")
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating {symbol}: {str(e)}")
            return None
    
    def evaluate_all_models(self, test_days=7):
        """Evaluate all available models"""
        print("🎯 Simple Model Evaluation - Standard ML Metrics")
        print("=" * 60)
        
        available_models = self.get_available_models()
        if not available_models:
            print("❌ No models found")
            return []
        
        print(f"📊 Found models for {len(available_models)} symbols")
        
        results = []
        total_models = sum(len(model_list) for model_list in available_models.values())
        evaluated = 0
        
        for symbol, model_list in available_models.items():
            for model_info in model_list:
                result = self.evaluate_model(symbol, model_info, test_days)
                if result:
                    results.append(result)
                evaluated += 1
                print(f"Progress: {evaluated}/{total_models}")
        
        self.results = results
        return results
    
    def print_summary(self):
        """Print summary of evaluation results"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\n📈 OVERALL METRICS:")
        print(f"   Models Evaluated: {len(df)}")
        print(f"   Average MAE: {df['mae'].mean():.4f}")
        print(f"   Average RMSE: {df['rmse'].mean():.4f}")
        print(f"   Average R²: {df['r2_score'].mean():.3f}")
        print(f"   Average Correlation: {df['correlation'].mean():.3f}")
        print(f"   Average Direction Accuracy: {df['direction_accuracy'].mean():.1f}%")
        
        print(f"\n🏆 BEST PERFORMING MODELS:")
        
        # Best by R²
        best_r2 = df.loc[df['r2_score'].idxmax()]
        print(f"   Best R²: {best_r2['symbol']} ({best_r2['granularity_name']}) = {best_r2['r2_score']:.3f}")
        
        # Best by MAE (lowest)
        best_mae = df.loc[df['mae'].idxmin()]
        print(f"   Best MAE: {best_mae['symbol']} ({best_mae['granularity_name']}) = {best_mae['mae']:.4f}")
        
        # Best by correlation
        best_corr = df.loc[df['correlation'].idxmax()]
        print(f"   Best Correlation: {best_corr['symbol']} ({best_corr['granularity_name']}) = {best_corr['correlation']:.3f}")
        
        # Best by direction accuracy
        best_dir = df.loc[df['direction_accuracy'].idxmax()]
        print(f"   Best Direction: {best_dir['symbol']} ({best_dir['granularity_name']}) = {best_dir['direction_accuracy']:.1f}%")
        
        # Performance by model type
        if 'model_type' in df.columns:
            print(f"\n📊 BY MODEL TYPE:")
            type_stats = df.groupby('model_type').agg({
                'mae': 'mean',
                'r2_score': 'mean',
                'correlation': 'mean',
                'direction_accuracy': 'mean'
            }).round(4)
            
            for model_type, stats in type_stats.iterrows():
                print(f"   {model_type.upper()}:")
                print(f"     MAE: {stats['mae']:.4f}")
                print(f"     R²: {stats['r2_score']:.3f}")
                print(f"     Correlation: {stats['correlation']:.3f}")
                print(f"     Direction Acc: {stats['direction_accuracy']:.1f}%")
        
        print("\n" + "="*70)
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if not self.results:
            print("❌ No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_eval_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"💾 Results saved to {filename}")

def main():
    """Main evaluation function"""
    # Parse arguments
    test_days = 7
    if len(sys.argv) > 1:
        try:
            test_days = int(sys.argv[1])
        except ValueError:
            print(f"Invalid test days, using default: {test_days}")
    
    print(f"📅 Using {test_days} days of test data")
    
    evaluator = SimpleModelEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_all_models(test_days)
    
    if results:
        # Print summary
        evaluator.print_summary()
        
        # Save results
        evaluator.save_results()
        
        print(f"\n✅ Evaluation complete! {len(results)} models evaluated.")
    else:
        print("❌ No models could be evaluated")

if __name__ == "__main__":
    main() 