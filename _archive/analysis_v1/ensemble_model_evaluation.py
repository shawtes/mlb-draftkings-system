#!/usr/bin/env python3
"""
Ensemble ML Model Evaluation
Advanced ensemble methods with stacking and voting regressors
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

# Import ML models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using alternatives")
    # Create a placeholder class
    class XGBRegressor:
        def __init__(self, **kwargs):
            # Fallback to GradientBoosting
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        def fit(self, X, y):
            return self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleMLEvaluator:
    def __init__(self):
        self.results = []
        self.scaler = StandardScaler()
        
    def create_ensemble_models(self):
        """Create advanced ensemble models"""
        
        # Base models for stacking
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.1)),
            ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
            ('svr', SVR(kernel='rbf', C=1.0, gamma='scale')),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)),
            ('bagging', BaggingRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=8), 
                n_estimators=10, 
                random_state=42
            ))
        ]

        # Meta model for stacking
        meta_model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)

        # Stacking Regressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3  # 3-fold cross-validation
        )

        # Voting Regressor
        voting_model = VotingRegressor(
            estimators=[
                ('lr', LinearRegression()),
                ('ridge', Ridge(alpha=1.0)),
                ('lasso', Lasso(alpha=0.1)),
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)),
                ('bagging', BaggingRegressor(
                    base_estimator=DecisionTreeRegressor(max_depth=8), 
                    n_estimators=10,
                    random_state=42
                ))
            ]
        )

        # Ensemble of ensembles
        ensemble_models = [
            ('stacking', stacking_model),
            ('voting', voting_model)
        ]
        
        final_model = StackingRegressor(
            estimators=ensemble_models,
            final_estimator=XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
            cv=3
        )

        # All models to evaluate
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42),
            'StackingRegressor': stacking_model,
            'VotingRegressor': voting_model,
            'EnsembleOfEnsembles': final_model
        }
        
        return models
        
    def prepare_features(self, df, target_periods=1):
        """Prepare features and target for ML training"""
        try:
            # Calculate future returns (target)
            df['future_return'] = df['close'].shift(-target_periods) / df['close'] - 1
            df['future_return_pct'] = df['future_return'] * 100
            
            # Get all available numeric columns except price data and future targets
            exclude_columns = {
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'future_return', 'future_return_pct'
            }
            
            # Get all numeric columns that are not in exclude list
            all_columns = set(df.columns)
            available_features = []
            
            for col in all_columns:
                if col not in exclude_columns:
                    # Check if column is numeric
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        available_features.append(col)
                    except (ValueError, TypeError):
                        continue  # Skip non-numeric columns
            
            logger.info(f"Found {len(available_features)} potential feature columns")
            
            # If we have enhanced features, use them; otherwise fall back to basic technical indicators
            if len(available_features) > 50:  # Enhanced features available
                feature_columns = available_features
                logger.info("Using enhanced feature set")
            else:
                # Fall back to basic features that should exist
                basic_features = []
                possible_basic = [
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'sma_20', 'sma_50', 'upper_band', 'lower_band',
                    'stoch_k', 'stoch_d', '%K', '%D',  # Try both naming conventions
                    'williams_r', 'cci', 'ATR', 'atr',  # Try both cases
                    'OBV', 'obv'  # Try both cases
                ]
                
                for feature in possible_basic:
                    if feature in df.columns:
                        basic_features.append(feature)
                
                if basic_features:
                    feature_columns = basic_features
                    logger.info(f"Using {len(basic_features)} basic technical indicators")
                else:
                    logger.error("No suitable features found")
                    return None, None
            
            # Additional ratio features if base columns exist
            if 'close' in df.columns:
                if 'sma_20' in df.columns:
                    df['price_ma20_ratio'] = df['close'] / df['sma_20']
                    feature_columns.append('price_ma20_ratio')
                if 'sma_50' in df.columns:
                    df['price_ma50_ratio'] = df['close'] / df['sma_50']
                    feature_columns.append('price_ma50_ratio')
                if 'volume' in df.columns:
                    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                    feature_columns.append('volume_ma_ratio')
                
                # Price action features
                df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
                df['close_open_ratio'] = df['close'] / df['open']
                feature_columns.extend(['high_low_ratio', 'close_open_ratio'])
            
            # Momentum features
            for period in [1, 3, 5]:  # Reduced periods for stability
                momentum_col = f'momentum_{period}'
                df[momentum_col] = df['close'].pct_change(period) * 100
                feature_columns.append(momentum_col)
            
            # Clean data
            df = df.dropna()
            
            if len(df) < 100:
                logger.warning(f"Insufficient data: {len(df)} rows")
                return None, None
            
            # Prepare features and target, only using columns that exist
            existing_features = [col for col in feature_columns if col in df.columns]
            logger.info(f"Using {len(existing_features)} features: {existing_features[:10]}...")  # Log first 10
            
            X = df[existing_features].fillna(method='ffill').fillna(0)
            y = df['future_return_pct'].fillna(0)
            
            # Remove extreme outliers (more conservative)
            q1, q99 = y.quantile([0.05, 0.95])  # 5th and 95th percentiles
            mask = (y >= q1) & (y <= q99)
            X = X[mask]
            y = y[mask]
            
            # Remove any remaining infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
            logger.info(f"Target range: {y.min():.3f}% to {y.max():.3f}%")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def calculate_metrics(self, y_true, y_pred, symbol, model_name):
        """Calculate comprehensive regression metrics"""
        try:
            # Basic regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Avoid division by zero
            y_true_abs = np.abs(y_true)
            y_true_abs_mean = np.mean(y_true_abs)
            
            if y_true_abs_mean > 0:
                mape = np.mean(y_true_abs / y_true_abs_mean * np.abs(y_true - y_pred)) * 100
                mae_percent = (mae / y_true_abs_mean) * 100
            else:
                mape = float('inf')
                mae_percent = float('inf')
            
            # Directional accuracy
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_true == direction_pred) * 100
            
            # Hit rates
            within_1_pct = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
            within_2_pct = np.mean(np.abs(y_true - y_pred) <= 2.0) * 100
            within_5_pct = np.mean(np.abs(y_true - y_pred) <= 5.0) * 100
            
            # Advanced metrics
            pred_std = np.std(y_pred)
            true_std = np.std(y_true)
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
            
            # Bias metrics
            mean_error = np.mean(y_pred - y_true)
            bias_ratio = mean_error / y_true_abs_mean * 100 if y_true_abs_mean > 0 else 0
            
            # Trading-specific metrics
            profitable_predictions = np.sum((y_pred > 0) & (y_true > 0)) + np.sum((y_pred < 0) & (y_true < 0))
            total_directional_predictions = np.sum(np.abs(y_pred) > 0.5)  # Only confident predictions
            
            if total_directional_predictions > 0:
                profitable_ratio = profitable_predictions / total_directional_predictions * 100
            else:
                profitable_ratio = 0
            
            metrics = {
                'symbol': symbol,
                'model': model_name,
                'samples': len(y_true),
                'r2_score': round(r2, 4),
                'mae': round(mae, 4),
                'mae_percent': round(mae_percent, 2) if not np.isinf(mae_percent) else 999.99,
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 2) if not np.isinf(mape) else 999.99,
                'directional_accuracy': round(directional_accuracy, 2),
                'within_1_pct': round(within_1_pct, 2),
                'within_2_pct': round(within_2_pct, 2),
                'within_5_pct': round(within_5_pct, 2),
                'correlation': round(correlation, 4),
                'mean_error': round(mean_error, 4),
                'bias_percent': round(bias_ratio, 2),
                'pred_std': round(pred_std, 4),
                'true_std': round(true_std, 4),
                'profitable_ratio': round(profitable_ratio, 2),
                'model_quality': self._assess_model_quality(r2, mae_percent, directional_accuracy),
                'trading_viability': self._assess_trading_viability(directional_accuracy, within_1_pct, r2, profitable_ratio)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None
    
    def _assess_model_quality(self, r2, mae_percent, directional_accuracy):
        """Assess overall model quality"""
        if r2 > 0.2 and mae_percent < 80 and directional_accuracy > 60:
            return "EXCELLENT"
        elif r2 > 0.1 and mae_percent < 100 and directional_accuracy > 55:
            return "GOOD"
        elif r2 > 0.05 and mae_percent < 150 and directional_accuracy > 52:
            return "FAIR"
        else:
            return "POOR"
    
    def _assess_trading_viability(self, directional_accuracy, within_1_pct, r2, profitable_ratio):
        """Assess trading viability"""
        if directional_accuracy > 60 and within_1_pct > 30 and r2 > 0.1 and profitable_ratio > 60:
            return "HIGHLY_VIABLE"
        elif directional_accuracy > 55 and within_1_pct > 20 and r2 > 0.05 and profitable_ratio > 55:
            return "VIABLE"
        elif directional_accuracy > 52 and r2 > 0.02 and profitable_ratio > 50:
            return "MARGINAL"
        else:
            return "NOT_VIABLE"
    
    def evaluate_symbol(self, symbol, timeframes=[1]):
        """Evaluate ensemble models for a symbol"""
        logger.info(f"\n🔍 Evaluating ensemble models for {symbol}")
        
        try:
            # Get data
            df = get_coinbase_data(symbol, 3600, days=90)  # 90 days for better training
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            # Create models
            models = self.create_ensemble_models()
            
            for timeframe in timeframes:
                logger.info(f"📊 Evaluating {timeframe}h prediction timeframe for {symbol}")
                
                # Prepare features
                X, y = self.prepare_features(df, target_periods=timeframe)
                
                if X is None or len(X) < 200:  # Need more data for ensemble
                    logger.warning(f"Insufficient data for {symbol} {timeframe}h timeframe: {len(X) if X is not None else 0} samples")
                    continue
                
                # Scale features for better performance
                X_scaled = self.scaler.fit_transform(X)
                
                # Split data (temporal split to avoid lookahead bias)
                split_idx = int(len(X_scaled) * 0.7)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                
                # Train and evaluate each model
                for model_name, model in models.items():
                    try:
                        logger.info(f"🤖 Training {model_name}...")
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        # Calculate metrics for test set (most important)
                        test_metrics = self.calculate_metrics(
                            y_test, y_pred_test, symbol, f"{model_name}_test_{timeframe}h"
                        )
                        
                        if test_metrics:
                            self.results.append(test_metrics)
                            
                            # Log key metrics
                            logger.info(f"✅ {model_name} ({timeframe}h) Results:")
                            logger.info(f"   R² Score: {test_metrics['r2_score']}")
                            logger.info(f"   MAE: {test_metrics['mae']:.4f} ({test_metrics['mae_percent']:.1f}%)")
                            logger.info(f"   RMSE: {test_metrics['rmse']:.4f}")
                            logger.info(f"   Directional Accuracy: {test_metrics['directional_accuracy']:.1f}%")
                            logger.info(f"   Profitable Ratio: {test_metrics['profitable_ratio']:.1f}%")
                            logger.info(f"   Within 1%: {test_metrics['within_1_pct']:.1f}%")
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
        logger.info("🚀 ADVANCED ENSEMBLE ML MODEL EVALUATION REPORT")
        logger.info("="*80)
        
        # Overall summary
        logger.info(f"\n📈 OVERALL SUMMARY:")
        logger.info(f"   Total Models Evaluated: {len(df_results)}")
        logger.info(f"   Average R² Score: {df_results['r2_score'].mean():.4f}")
        logger.info(f"   Average MAE: {df_results['mae'].mean():.4f}")
        logger.info(f"   Average RMSE: {df_results['rmse'].mean():.4f}")
        logger.info(f"   Average Directional Accuracy: {df_results['directional_accuracy'].mean():.2f}%")
        logger.info(f"   Average Profitable Ratio: {df_results['profitable_ratio'].mean():.2f}%")
        
        # Best performing models
        logger.info(f"\n🏆 TOP PERFORMING MODELS (by R² Score):")
        top_models = df_results.nlargest(10, 'r2_score')[
            ['symbol', 'model', 'r2_score', 'mae', 'directional_accuracy', 'profitable_ratio', 'trading_viability']
        ]
        for _, row in top_models.iterrows():
            logger.info(f"   {row['symbol']} - {row['model']}: R²={row['r2_score']:.4f}, "
                       f"Dir.Acc={row['directional_accuracy']:.1f}%, "
                       f"Profit.Ratio={row['profitable_ratio']:.1f}%, "
                       f"Viability={row['trading_viability']}")
        
        # Model type comparison
        logger.info(f"\n🔬 MODEL TYPE COMPARISON:")
        model_comparison = df_results.groupby(df_results['model'].str.extract('([^_]+)')[0]).agg({
            'r2_score': ['mean', 'std', 'max'],
            'directional_accuracy': ['mean', 'std', 'max'],
            'profitable_ratio': ['mean', 'std', 'max']
        }).round(4)
        
        for model_type in model_comparison.index:
            logger.info(f"   {model_type}:")
            logger.info(f"      R² Score: {model_comparison.loc[model_type, ('r2_score', 'mean')]:.4f} "
                       f"± {model_comparison.loc[model_type, ('r2_score', 'std')]:.4f} "
                       f"(max: {model_comparison.loc[model_type, ('r2_score', 'max')]:.4f})")
            logger.info(f"      Dir Acc: {model_comparison.loc[model_type, ('directional_accuracy', 'mean')]:.2f}% "
                       f"± {model_comparison.loc[model_type, ('directional_accuracy', 'std')]:.2f}% "
                       f"(max: {model_comparison.loc[model_type, ('directional_accuracy', 'max')]:.2f}%)")
        
        # Trading viability analysis
        logger.info(f"\n💰 TRADING VIABILITY ANALYSIS:")
        viability_counts = df_results['trading_viability'].value_counts()
        for viability, count in viability_counts.items():
            percentage = count / len(df_results) * 100
            logger.info(f"   {viability}: {count} models ({percentage:.1f}%)")
        
        # Ensemble vs individual model comparison
        ensemble_models = df_results[df_results['model'].str.contains('Stacking|Voting|Ensemble')]
        individual_models = df_results[~df_results['model'].str.contains('Stacking|Voting|Ensemble')]
        
        if len(ensemble_models) > 0 and len(individual_models) > 0:
            logger.info(f"\n🔀 ENSEMBLE vs INDIVIDUAL MODELS:")
            logger.info(f"   Ensemble Models - Avg R²: {ensemble_models['r2_score'].mean():.4f}, "
                       f"Avg Dir.Acc: {ensemble_models['directional_accuracy'].mean():.2f}%")
            logger.info(f"   Individual Models - Avg R²: {individual_models['r2_score'].mean():.4f}, "
                       f"Avg Dir.Acc: {individual_models['directional_accuracy'].mean():.2f}%")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS TABLE:")
        display_columns = [
            'symbol', 'model', 'r2_score', 'mae', 'directional_accuracy', 
            'profitable_ratio', 'within_1_pct', 'trading_viability'
        ]
        print(df_results[display_columns].to_string(index=False))
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"ensemble_evaluation_results_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)
        logger.info(f"\n💾 Detailed results saved to: {results_path}")
        
        logger.info("="*80)

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Advanced Ensemble ML Model Evaluation")
    
    evaluator = EnsembleMLEvaluator()
    
    # Symbols to evaluate
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]  # Start with 3 for faster testing
    
    # Prediction timeframes (in hours)
    timeframes = [1, 4, 24]  # 1h, 4h, 24h predictions
    
    logger.info(f"📊 Evaluating {len(symbols)} symbols across {len(timeframes)} timeframes")
    logger.info(f"🎯 Symbols: {', '.join(symbols)}")
    logger.info(f"⏰ Timeframes: {timeframes} hours")
    logger.info(f"🤖 Models: Linear, Ridge, Lasso, RandomForest, XGBoost, Stacking, Voting, EnsembleOfEnsembles")
    
    # Evaluate each symbol
    for symbol in symbols:
        try:
            evaluator.evaluate_symbol(symbol, timeframes)
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {symbol}: {str(e)}")
            continue
    
    # Generate comprehensive report
    evaluator.generate_report()
    
    logger.info("🎉 Advanced Ensemble ML Model Evaluation completed!")

if __name__ == "__main__":
    main() 