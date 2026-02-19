#!/usr/bin/env python3
"""
Model Evaluation Script
Comprehensive evaluation of ML trading models across different symbols and timeframes.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import (
    get_coinbase_data, 
    calculate_indicators, 
    get_cached_symbols,
    train_price_prediction_model,
    make_ml_decision,
    get_price_prediction_for_granularity
)
from init_database import get_db_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self):
        self.db_path = get_db_path()
        self.models_dir = os.path.join(current_dir, 'models')
        self.granularities = {
            '15m': 900,
            '1h': 3600,
            '4h': 14400
        }
        self.coinbase_fees = 0.006  # 0.6% fee
        self.results = {}
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def get_available_models(self):
        """Get list of available trained models"""
        models = {}
        
        for filename in os.listdir(self.models_dir):
            # Handle both regression models (_model.pkl) and classification models (_clf.pkl)
            if filename.endswith('_model.pkl') or filename.endswith('_clf.pkl'):
                if filename.endswith('_model.pkl'):
                    parts = filename.replace('_model.pkl', '').split('_')
                    model_type = 'regression'
                elif filename.endswith('_clf.pkl'):
                    parts = filename.replace('_clf.pkl', '').split('_')
                    model_type = 'classification'
                
                if len(parts) >= 2:
                    # Better symbol parsing
                    granularity = int(parts[-1])
                    symbol_parts = parts[:-1]
                    
                    # Reconstruct symbol with proper format
                    symbol = '_'.join(symbol_parts)
                    
                    # Fix common symbol issues
                    if symbol == 'USDT':
                        symbol = 'USDT-USD'
                    elif symbol == '00':
                        continue  # Skip invalid symbols
                    elif not symbol.endswith('-USD') and not symbol.endswith('USD'):
                        symbol = symbol + '-USD'
                    elif symbol.endswith('USD') and not symbol.endswith('-USD'):
                        symbol = symbol.replace('USD', '-USD')
                    
                    # Skip clearly invalid symbols
                    if symbol.startswith('-') or len(symbol) < 4:
                        continue
                    
                    if symbol not in models:
                        models[symbol] = []
                    models[symbol].append({'granularity': granularity, 'type': model_type, 'filename': filename})
        
        # Sort granularities
        for symbol in models:
            models[symbol].sort(key=lambda x: x['granularity'])
            
        return models
    
    def get_model_features(self, model_path):
        """Try to determine what features a model was trained with"""
        try:
            model = joblib.load(joblib.load(model_path)
            
            # Check if model has feature_names_in_ attribute (sklearn models)
            if hasattr(model, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_names_in_'):
                return list(model.feature_names_in_)
            
            # Check if it's a pipeline with feature names
            if hasattr(model, 'feature_names_'):
                return list(model.feature_names_)
            
            # For older models, try to guess from model type
            if hasattr(model, 'n_features_'):
                n_features = model.n_features_
                # Return a default set based on feature count
                if n_features == 12:
                    return [
                        'rsi', 'macd', 'macd_signal', 'macd_hist',
                        'sma_20', 'sma_50', 'upper_band', 'lower_band',
                        'ATR', '%K', '%D', 'volume'
                    ]
                elif n_features == 11:
                    return [
                        'rsi', 'macd', 'macd_signal', 'macd_hist',
                        'sma_20', 'sma_50', 'upper_band', 'lower_band',
                        'ATR', '%K', '%D'
                    ]
            
            # Default feature set for unknown models
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band'
            ]
            
        except Exception as e:
            logger.warning(f"Could not determine model features: {str(e)}")
            # Return basic feature set
            return [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band'
            ]
    
    def train_missing_models(self, symbols=None, force_retrain=False):
        """Train models for symbols that don't have them"""
        logger.info("🤖 Checking for missing models and training if needed...")
        
        if symbols is None:
            # Get some popular symbols to train
            symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD']
        
        trained_count = 0
        
        for symbol in symbols:
            for granularity_name, granularity in self.granularities.items():
                model_path = os.path.join(self.models_dir, f"{symbol}_{granularity}_model.pkl")
                
                if not os.path.exists(model_path) or force_retrain:
                    logger.info(f"🔧 Training regression model for {symbol} ({granularity_name})...")
                    try:
                        # Train the price prediction model
                        model = train_price_prediction_model(symbol, granularity)
                        if model is not None:
                            trained_count += 1
                            logger.info(f"✅ Successfully trained {symbol} ({granularity_name})")
                        else:
                            logger.warning(f"❌ Failed to train {symbol} ({granularity_name})")
                    except Exception as e:
                        logger.error(f"❌ Error training {symbol} ({granularity_name}): {str(e)}")
                else:
                    logger.debug(f"✅ Model already exists for {symbol} ({granularity_name})")
        
        logger.info(f"🎉 Training complete: {trained_count} new models trained")
        return trained_count
    
    def evaluate_model_accuracy(self, symbol, model_info, test_days=7):
        """Evaluate model accuracy on recent data"""
        try:
            granularity = model_info['granularity']
            model_type = model_info['type']
            filename = model_info['filename']
            
            logger.info(f"📊 Evaluating {symbol} {model_type} model ({granularity}s)...")
            
            # Load model
            model_path = os.path.join(self.models_dir, filename)
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return None
            
            # Determine what features this model expects
            expected_features = self.get_model_features(model_path)
            logger.info(f"📋 Model expects {len(expected_features)} features: {expected_features}")
            
            model = joblib.load(model_path)
            
            # Get recent data for testing
            df = get_coinbase_data(symbol, granularity, days=test_days + 7)  # Extra days for indicators
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Calculate indicators
            df = calculate_indicators(df)
            df = df.dropna()
            
            if len(df) < 10:
                logger.warning(f"Insufficient data for evaluation: {len(df)} rows")
                return None
            
            # Split into train/test
            test_size = min(len(df) // 3, test_days * 24)  # Use last 1/3 or test_days worth
            train_df = df[:-test_size]
            test_df = df[-test_size:]
            
            if len(test_df) < 5:
                logger.warning(f"Test set too small: {len(test_df)} rows")
                return None
            
            # Use the features that the model expects
            feature_columns = []
            for feature in expected_features:
                if feature in test_df.columns:
                    feature_columns.append(feature)
                else:
                    logger.warning(f"Expected feature '{feature}' not available in data")
            
            if len(feature_columns) < len(expected_features) * 0.5:  # Need at least 50% of expected features
                logger.warning(f"Too few expected features available: {len(feature_columns)}/{len(expected_features)}")
                return None
            
            X_test = test_df[feature_columns].fillna(0)
            
            # Calculate actual price changes
            test_df = test_df.copy()
            test_df['next_price'] = test_df['close'].shift(-1)
            test_df['actual_change'] = ((test_df['next_price'] - test_df['close']) / test_df['close']) * 100
            test_df = test_df.dropna()
            
            if len(test_df) == 0:
                logger.warning("No valid test data after price change calculation")
                return None
            
            X_test = test_df[feature_columns].fillna(0)
            y_actual = test_df['actual_change'].values
            
            # Make predictions based on model type
            if model_type == 'regression':
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    logger.warning(f"Regression model for {symbol} doesn't have predict method")
                    return None
            else:  # classification
                if hasattr(model, 'predict_proba'):
                    # For classification, convert probabilities to price predictions
                    y_pred_proba = model.predict_proba(X_test)
                    # Convert class probabilities to expected price changes
                    # Assuming classes are [0, 1, 2] for [SELL, HOLD, BUY]
                    if y_pred_proba.shape[1] >= 3:
                        # Map probabilities to price changes: SELL=-2%, HOLD=0%, BUY=+2%
                        y_pred = (y_pred_proba[:, 2] - y_pred_proba[:, 0]) * 2.0
                    else:
                        # Binary classification: map to -1% or +1%
                        y_pred = (y_pred_proba[:, 1] - y_pred_proba[:, 0]) * 1.0
                elif hasattr(model, 'predict'):
                    # Simple classification predictions
                    y_pred_class = model.predict(X_test)
                    # Convert classes to price changes
                    y_pred = np.where(y_pred_class == 2, 2.0,  # BUY -> +2%
                                     np.where(y_pred_class == 1, 0.0,  # HOLD -> 0%
                                             -2.0))  # SELL -> -2%
                else:
                    logger.warning(f"Classification model for {symbol} doesn't have predict method")
                    return None
            
            # Calculate metrics
            mae = np.mean(np.abs(y_actual - y_pred))
            rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
            
            # Direction accuracy (most important for trading)
            actual_direction = np.sign(y_actual)
            pred_direction = np.sign(y_pred)
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Trading simulation
            portfolio_value = 10000  # Start with $10k
            trades = 0
            winning_trades = 0
            total_fees = 0
            
            for i in range(len(y_pred)):
                prediction = y_pred[i]
                actual = y_actual[i]
                current_price = test_df.iloc[i]['close']
                
                # Trading logic: only trade if prediction is strong enough
                threshold = 0.5 if model_type == 'classification' else 1.0
                if abs(prediction) > threshold:
                    trades += 1
                    
                    # Calculate position size (10% of portfolio)
                    position_size = portfolio_value * 0.1
                    quantity = position_size / current_price
                    
                    # Calculate fees
                    fee = position_size * self.coinbase_fees
                    total_fees += fee * 2  # Buy and sell fees
                    
                    # Calculate profit/loss
                    profit_loss = (actual / 100) * position_size
                    portfolio_value += profit_loss - (fee * 2)
                    
                    if profit_loss > fee * 2:  # Profit after fees
                        winning_trades += 1
            
            win_rate = (winning_trades / trades * 100) if trades > 0 else 0
            total_return = ((portfolio_value - 10000) / 10000) * 100
            
            # Calculate profit factor
            profits = []
            losses = []
            threshold = 0.5 if model_type == 'classification' else 1.0
            for i in range(len(y_pred)):
                if abs(y_pred[i]) > threshold:
                    actual_return = y_actual[i]
                    if actual_return > 0:
                        profits.append(actual_return)
                    else:
                        losses.append(abs(actual_return))
            
            profit_factor = (sum(profits) / sum(losses)) if losses else float('inf')
            
            result = {
                'symbol': symbol,
                'granularity': granularity,
                'granularity_name': self.get_granularity_name(granularity),
                'model_type': model_type,
                'test_periods': len(test_df),
                'test_days': test_days,
                'mae': mae,
                'rmse': rmse,
                'direction_accuracy': direction_accuracy,
                'total_trades': trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'total_fees': total_fees,
                'profit_factor': profit_factor,
                'final_portfolio': portfolio_value,
                'avg_prediction': np.mean(np.abs(y_pred)),
                'avg_actual': np.mean(np.abs(y_actual)),
                'feature_count': len(feature_columns),
                'expected_features': len(expected_features),
                'model_class': type(model).__name__
            }
            
            logger.info(f"✅ {symbol} ({self.get_granularity_name(granularity)}, {model_type}): "
                       f"Direction Accuracy: {direction_accuracy:.1f}%, "
                       f"Win Rate: {win_rate:.1f}%, "
                       f"Return: {total_return:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {symbol} model: {str(e)}")
            return None
    
    def get_granularity_name(self, granularity):
        """Convert granularity seconds to readable name"""
        for name, seconds in self.granularities.items():
            if seconds == granularity:
                return name
        return f"{granularity}s"
    
    def evaluate_all_models(self, test_days=7, train_missing=True):
        """Evaluate all available models"""
        logger.info("🎯 Starting comprehensive model evaluation...")
        
        available_models = self.get_available_models()
        
        # Train missing models if requested
        if train_missing and len(available_models) < 3:
            logger.info("🤖 Few models found, training some popular symbols...")
            trained = self.train_missing_models()
            if trained > 0:
                # Refresh available models after training
                available_models = self.get_available_models()
        
        total_models = sum(len(model_list) for model_list in available_models.values())
        
        logger.info(f"📊 Found models for {len(available_models)} symbols, {total_models} total models")
        
        results = []
        evaluated = 0
        
        for symbol, model_list in available_models.items():
            for model_info in model_list:
                result = self.evaluate_model_accuracy(symbol, model_info, test_days)
                if result:
                    results.append(result)
                evaluated += 1
                
                logger.info(f"Progress: {evaluated}/{total_models} models evaluated")
        
        self.results = results
        return results
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📊 MODEL EVALUATION SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Total Models Evaluated: {len(df)}")
        print(f"   Average Direction Accuracy: {df['direction_accuracy'].mean():.1f}%")
        print(f"   Average Win Rate: {df['win_rate'].mean():.1f}%")
        print(f"   Average Return: {df['total_return'].mean():.2f}%")
        print(f"   Average MAE: {df['mae'].mean():.4f}")
        print(f"   Average RMSE: {df['rmse'].mean():.4f}")
        
        # Best performing models
        print(f"\n🏆 TOP 5 MODELS BY DIRECTION ACCURACY:")
        top_accuracy = df.nlargest(5, 'direction_accuracy')
        for _, row in top_accuracy.iterrows():
            print(f"   {row['symbol']} ({row['granularity_name']}): "
                  f"{row['direction_accuracy']:.1f}% accuracy, "
                  f"{row['win_rate']:.1f}% win rate, "
                  f"{row['total_return']:+.2f}% return")
        
        print(f"\n💰 TOP 5 MODELS BY RETURN:")
        top_return = df.nlargest(5, 'total_return')
        for _, row in top_return.iterrows():
            print(f"   {row['symbol']} ({row['granularity_name']}): "
                  f"{row['total_return']:+.2f}% return, "
                  f"{row['direction_accuracy']:.1f}% accuracy, "
                  f"{row['total_trades']} trades")
        
        # Performance by timeframe
        print(f"\n⏰ PERFORMANCE BY TIMEFRAME:")
        timeframe_stats = df.groupby('granularity_name').agg({
            'direction_accuracy': 'mean',
            'win_rate': 'mean',
            'total_return': 'mean',
            'total_trades': 'mean'
        }).round(2)
        
        for timeframe, stats in timeframe_stats.iterrows():
            print(f"   {timeframe}: "
                  f"{stats['direction_accuracy']:.1f}% accuracy, "
                  f"{stats['win_rate']:.1f}% win rate, "
                  f"{stats['total_return']:+.2f}% avg return, "
                  f"{stats['total_trades']:.0f} avg trades")
        
        # Models that need attention
        print(f"\n⚠️  MODELS NEEDING ATTENTION:")
        poor_models = df[df['direction_accuracy'] < 55]
        if len(poor_models) > 0:
            for _, row in poor_models.iterrows():
                print(f"   {row['symbol']} ({row['granularity_name']}): "
                      f"{row['direction_accuracy']:.1f}% accuracy - Consider retraining")
        else:
            print("   ✅ All models performing above 55% accuracy threshold")
        
        # Profitable models
        profitable = df[df['total_return'] > 2.0]  # More than 2% return after fees
        print(f"\n💎 HIGHLY PROFITABLE MODELS ({len(profitable)} models with >2% return):")
        if len(profitable) > 0:
            for _, row in profitable.iterrows():
                print(f"   {row['symbol']} ({row['granularity_name']}): "
                      f"{row['total_return']:+.2f}% return, "
                      f"PF: {row['profit_factor']:.2f}")
        else:
            print("   📊 No models with >2% return in test period")
        
        print("\n" + "="*80)
    
    def save_results_to_file(self, filename=None):
        """Save detailed results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_evaluation_{timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logger.info(f"💾 Results saved to {filename}")
        
        # Also save JSON for detailed analysis
        json_filename = filename.replace('.csv', '.json')
        with open(json_filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"💾 Detailed results saved to {json_filename}")
    
    def compare_live_performance(self, days=3):
        """Compare model predictions with actual recent performance"""
        logger.info(f"🔍 Comparing live performance over last {days} days...")
        
        live_results = []
        
        # Get symbols with recent activity
        symbols_to_test = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD']
        
        for symbol in symbols_to_test:
            for granularity_name, granularity in self.granularities.items():
                try:
                    # Check if model exists
                    model_path = os.path.join(self.models_dir, f"{symbol}_{granularity}_model.pkl")
                    if not os.path.exists(model_path):
                        continue
                    
                    # Get recent data
                    df = get_coinbase_data(symbol, granularity, days=days + 2)
                    if df.empty or len(df) < 10:
                        continue
                    
                    # Calculate indicators
                    df = calculate_indicators(df)
                    df = df.dropna()
                    
                    if len(df) < 5:
                        continue
                    
                    # Use last few periods for testing
                    test_periods = min(days * (86400 // granularity), len(df) - 1)
                    
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for i in range(len(df) - test_periods, len(df) - 1):
                        try:
                            # Make prediction for this point
                            prediction = get_price_prediction_for_granularity(symbol, granularity)
                            if prediction is None:
                                continue
                            
                            # Get actual price change
                            current_price = df.iloc[i]['close']
                            next_price = df.iloc[i + 1]['close']
                            actual_change = ((next_price - current_price) / current_price) * 100
                            
                            # Check if direction was correct
                            pred_direction = 1 if prediction['predicted_change'] > 0 else -1
                            actual_direction = 1 if actual_change > 0 else -1
                            
                            if pred_direction == actual_direction:
                                correct_predictions += 1
                            total_predictions += 1
                            
                        except Exception as e:
                            logger.warning(f"Error testing prediction for {symbol}: {str(e)}")
                            continue
                    
                    if total_predictions > 0:
                        accuracy = (correct_predictions / total_predictions) * 100
                        live_results.append({
                            'symbol': symbol,
                            'timeframe': granularity_name,
                            'live_accuracy': accuracy,
                            'predictions_tested': total_predictions,
                            'correct_predictions': correct_predictions
                        })
                        
                        logger.info(f"📊 {symbol} ({granularity_name}): "
                                   f"{accuracy:.1f}% live accuracy "
                                   f"({correct_predictions}/{total_predictions})")
                
                except Exception as e:
                    logger.warning(f"Error in live comparison for {symbol}: {str(e)}")
                    continue
        
        if live_results:
            print(f"\n🔴 LIVE PERFORMANCE COMPARISON (Last {days} days):")
            print("-" * 60)
            for result in live_results:
                print(f"{result['symbol']} ({result['timeframe']}): "
                      f"{result['live_accuracy']:.1f}% accuracy "
                      f"({result['correct_predictions']}/{result['predictions_tested']} correct)")
        else:
            logger.warning("No live performance data available")
        
        return live_results

def main():
    """Main evaluation function"""
    print("🎯 ML Model Evaluation System")
    print("=" * 50)
    
    # Parse command line arguments
    test_days = 7
    if len(sys.argv) > 1:
        try:
            test_days = int(sys.argv[1])
            print(f"📅 Using {test_days} days for testing")
        except ValueError:
            print(f"⚠️ Invalid test days argument, using default: {test_days}")
    
    evaluator = ModelEvaluator()
    
    # Check for available models
    available_models = evaluator.get_available_models()
    if not available_models:
        print("❌ No trained models found. Training some models first...")
        trained = evaluator.train_missing_models()
        if trained == 0:
            print("❌ Could not train any models. Please check your setup.")
            return
        # Refresh available models after training
        available_models = evaluator.get_available_models()
    
    print(f"📊 Found models for {len(available_models)} symbols")
    for symbol, model_list in available_models.items():
        timeframes = [f"{evaluator.get_granularity_name(m['granularity'])} ({m['type']})" for m in model_list]
        print(f"   {symbol}: {', '.join(timeframes)}")
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation with {test_days} days of test data...")
    results = evaluator.evaluate_all_models(test_days, train_missing=False)  # Don't train again
    
    if results:
        # Generate summary report
        evaluator.generate_summary_report()
        
        # Save results
        evaluator.save_results_to_file()
        
        # Compare with live performance
        evaluator.compare_live_performance(days=3)
        
        print(f"\n✅ Evaluation complete! {len(results)} models evaluated.")
        print("💡 Use the saved CSV/JSON files for detailed analysis.")
        
    else:
        print("❌ No models could be evaluated successfully.")

if __name__ == "__main__":
    main() 