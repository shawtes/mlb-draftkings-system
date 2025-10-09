#!/usr/bin/env python3
"""
ML Model Metrics Evaluation
Provides comprehensive regression metrics for price prediction models
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import get_coinbase_data, calculate_indicators
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelEvaluator:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        self.results = []
        
    def prepare_features(self, df, target_periods=1):
        """Prepare features and target for ML training"""
        try:
            # Calculate future returns (target)
            df['future_return'] = df['close'].shift(-target_periods) / df['close'] - 1
            df['future_return_pct'] = df['future_return'] * 100
            
            # Feature columns
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'upper_band', 'lower_band',
                'stoch_k', 'stoch_d', 'williams_r', 'cci',
                'ATR', 'OBV'
            ]
            
            # Add price-based features
            df['price_ma_ratio'] = df['close'] / df['sma_20']
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Add momentum features
            df['momentum_1'] = df['close'].pct_change(1) * 100
            df['momentum_5'] = df['close'].pct_change(5) * 100
            df['momentum_10'] = df['close'].pct_change(10) * 100
            
            feature_columns.extend([
                'price_ma_ratio', 'volume_ma_ratio', 'high_low_ratio', 'close_open_ratio',
                'momentum_1', 'momentum_5', 'momentum_10'
            ])
            
            # Clean data
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient data: {len(df)} rows")
                return None, None
            
            # Prepare features and target
            X = df[feature_columns].fillna(0)
            y = df['future_return_pct'].fillna(0)
            
            # Remove extreme outliers
            q1, q3 = y.quantile([0.01, 0.99])
            mask = (y >= q1) & (y <= q3)
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None, None
    
    def calculate_metrics(self, y_true, y_pred, symbol, model_name):
        """Calculate comprehensive regression metrics"""
        try:
            # Basic regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100
            
            # MAE as percentage of mean absolute target
            mae_percent = (mae / np.mean(np.abs(y_true))) * 100
            
            # Directional accuracy
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_true == direction_pred) * 100
            
            # Hit rate (predictions within 1% of actual)
            within_1_pct = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
            within_2_pct = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
            
            # Prediction distribution
            pred_std = np.std(y_pred)
            true_std = np.std(y_true)
            prediction_range = np.max(y_pred) - np.min(y_pred)
            
            # Bias metrics
            mean_error = np.mean(y_pred - y_true)
            bias_ratio = mean_error / np.mean(np.abs(y_true)) * 100
            
            metrics = {
                'symbol': symbol,
                'model': model_name,
                'samples': len(y_true),
                'r2_score': round(r2, 4),
                'mae': round(mae, 4),
                'mae_percent': round(mae_percent, 2),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 2),
                'directional_accuracy': round(directional_accuracy, 2),
                'within_1_pct': round(within_1_pct, 2),
                'within_2_pct': round(within_2_pct, 2),
                'mean_error': round(mean_error, 4),
                'bias_percent': round(bias_ratio, 2),
                'pred_std': round(pred_std, 4),
                'true_std': round(true_std, 4),
                'prediction_range': round(prediction_range, 4),
                'model_quality': self._assess_model_quality(r2, mae_percent, directional_accuracy),
                'trading_viability': self._assess_trading_viability(directional_accuracy, within_1_pct, r2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None
    
    def _assess_model_quality(self, r2, mae_percent, directional_accuracy):
        """Assess overall model quality"""
        if r2 > 0.3 and mae_percent < 50 and directional_accuracy > 60:
            return "EXCELLENT"
        elif r2 > 0.1 and mae_percent < 70 and directional_accuracy > 55:
            return "GOOD"
        elif r2 > 0.05 and mae_percent < 90 and directional_accuracy > 52:
            return "FAIR"
        else:
            return "POOR"
    
    def _assess_trading_viability(self, directional_accuracy, within_1_pct, r2):
        """Assess trading viability"""
        if directional_accuracy > 60 and within_1_pct > 30 and r2 > 0.1:
            return "HIGHLY_VIABLE"
        elif directional_accuracy > 55 and within_1_pct > 20 and r2 > 0.05:
            return "VIABLE"
        elif directional_accuracy > 52 and r2 > 0.02:
            return "MARGINAL"
        else:
            return "NOT_VIABLE"
    
    def evaluate_symbol(self, symbol, timeframes=[1]):
        """Evaluate ML models for a symbol across different prediction timeframes"""
        logger.info(f"\n🔍 Evaluating ML models for {symbol}")
        
        try:
            # Get data
            df = get_coinbase_data(symbol, 3600, days=60)  # 60 days of hourly data
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            for timeframe in timeframes:
                logger.info(f"📊 Evaluating {timeframe}h prediction timeframe for {symbol}")
                
                # Prepare features
                X, y = self.prepare_features(df, target_periods=timeframe)
                
                if X is None or len(X) < 100:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}h timeframe")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, shuffle=False
                )
                
                logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                
                # Train and evaluate each model
                for model_name, model in self.models.items():
                    try:
                        logger.info(f"🤖 Training {model_name}...")
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        # Calculate metrics for training set
                        train_metrics = self.calculate_metrics(
                            y_train, y_pred_train, symbol, f"{model_name}_train_{timeframe}h"
                        )
                        
                        # Calculate metrics for test set
                        test_metrics = self.calculate_metrics(
                            y_test, y_pred_test, symbol, f"{model_name}_test_{timeframe}h"
                        )
                        
                        if train_metrics:
                            self.results.append(train_metrics)
                            
                        if test_metrics:
                            self.results.append(test_metrics)
                            
                            # Log key metrics
                            logger.info(f"✅ {model_name} ({timeframe}h) Results:")
                            logger.info(f"   R² Score: {test_metrics['r2_score']}")
                            logger.info(f"   MAE: {test_metrics['mae']}")
                            logger.info(f"   MAE%: {test_metrics['mae_percent']}%")
                            logger.info(f"   RMSE: {test_metrics['rmse']}")
                            logger.info(f"   MAPE: {test_metrics['mape']}%")
                            logger.info(f"   Directional Accuracy: {test_metrics['directional_accuracy']}%")
                            logger.info(f"   Model Quality: {test_metrics['model_quality']}")
                            logger.info(f"   Trading Viability: {test_metrics['trading_viability']}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error training {model_name}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"❌ Error evaluating {symbol}: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        if not self.results:
            logger.warning("No results to report")
            return
        
        df_results = pd.DataFrame(self.results)
        
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE ML MODEL EVALUATION REPORT")
        logger.info("="*80)
        
        # Overall summary
        test_results = df_results[df_results['model'].str.contains('_test_')]
        
        logger.info(f"\n📈 OVERALL SUMMARY:")
        logger.info(f"   Total Models Evaluated: {len(test_results)}")
        logger.info(f"   Average R² Score: {test_results['r2_score'].mean():.4f}")
        logger.info(f"   Average MAE: {test_results['mae'].mean():.4f}")
        logger.info(f"   Average MAE%: {test_results['mae_percent'].mean():.2f}%")
        logger.info(f"   Average RMSE: {test_results['rmse'].mean():.4f}")
        logger.info(f"   Average MAPE: {test_results['mape'].mean():.2f}%")
        logger.info(f"   Average Directional Accuracy: {test_results['directional_accuracy'].mean():.2f}%")
        
        # Best performing models
        logger.info(f"\n🏆 TOP PERFORMING MODELS (by R² Score):")
        top_models = test_results.nlargest(5, 'r2_score')[['symbol', 'model', 'r2_score', 'mae', 'mae_percent', 'directional_accuracy', 'trading_viability']]
        for _, row in top_models.iterrows():
            logger.info(f"   {row['symbol']} - {row['model']}: R²={row['r2_score']:.4f}, MAE%={row['mae_percent']:.1f}%, Dir={row['directional_accuracy']:.1f}%, Viability={row['trading_viability']}")
        
        # Model comparison
        logger.info(f"\n🔬 MODEL TYPE COMPARISON:")
        model_comparison = test_results.groupby(test_results['model'].str.extract('([^_]+)')[0]).agg({
            'r2_score': ['mean', 'std'],
            'mae_percent': ['mean', 'std'],
            'directional_accuracy': ['mean', 'std']
        }).round(4)
        
        for model_type in model_comparison.index:
            logger.info(f"   {model_type}:")
            logger.info(f"      R² Score: {model_comparison.loc[model_type, ('r2_score', 'mean')]:.4f} ± {model_comparison.loc[model_type, ('r2_score', 'std')]:.4f}")
            logger.info(f"      MAE%: {model_comparison.loc[model_type, ('mae_percent', 'mean')]:.2f}% ± {model_comparison.loc[model_type, ('mae_percent', 'std')]:.2f}%")
            logger.info(f"      Dir Acc: {model_comparison.loc[model_type, ('directional_accuracy', 'mean')]:.2f}% ± {model_comparison.loc[model_type, ('directional_accuracy', 'std')]:.2f}%")
        
        # Trading viability analysis
        logger.info(f"\n💰 TRADING VIABILITY ANALYSIS:")
        viability_counts = test_results['trading_viability'].value_counts()
        for viability, count in viability_counts.items():
            percentage = count / len(test_results) * 100
            logger.info(f"   {viability}: {count} models ({percentage:.1f}%)")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS TABLE:")
        display_columns = [
            'symbol', 'model', 'r2_score', 'mae', 'mae_percent', 'rmse', 'mape',
            'directional_accuracy', 'within_1_pct', 'model_quality', 'trading_viability'
        ]
        print(test_results[display_columns].to_string(index=False))
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"ml_evaluation_results_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)
        logger.info(f"\n💾 Detailed results saved to: {results_path}")
        
        logger.info("="*80)

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Comprehensive ML Model Evaluation")
    
    evaluator = MLModelEvaluator()
    
    # Symbols to evaluate
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"]
    
    # Prediction timeframes (in hours)
    timeframes = [1, 4, 24]  # 1h, 4h, 24h predictions
    
    logger.info(f"📊 Evaluating {len(symbols)} symbols across {len(timeframes)} timeframes")
    logger.info(f"🎯 Symbols: {', '.join(symbols)}")
    logger.info(f"⏰ Timeframes: {timeframes} hours")
    
    # Evaluate each symbol
    for symbol in symbols:
        try:
            evaluator.evaluate_symbol(symbol, timeframes)
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {symbol}: {str(e)}")
            continue
    
    # Generate comprehensive report
    evaluator.generate_report()
    
    logger.info("🎉 ML Model Evaluation completed!")

if __name__ == "__main__":
    main() 