#!/usr/bin/env python3
"""
Model Performance Comparison: Ensemble vs LightGBM Momentum Models
================================================================

This script provides comprehensive comparison between:
1. Ensemble models (RandomForest + Voting + Stacking)
2. LightGBM momentum-based models 

For cryptocurrency trading performance evaluation.
"""

import os
import sys
import sqlite3
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import time

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb

# Try to import LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM is available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available - install with: pip install lightgbm")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from maybe import get_coinbase_data, calculate_indicators, get_db_path
    from init_database import init_database
except ImportError as e:
    logger.warning(f"Could not import local modules: {e}")
    # Create dummy functions if imports fail
    def get_coinbase_data(symbol, granularity, days):
        return pd.DataFrame()
    def calculate_indicators(df):
        return df
    def get_db_path():
        return "trading_bot.db"

class ModelPerformanceComparator:
    """
    Comprehensive model performance comparison framework
    """
    
    def __init__(self, results_dir: str = "model_comparison_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize performance tracking
        self.performance_results = {
            'ensemble_models': {},
            'lightgbm_models': {},
            'comparison_summary': {},
            'trading_performance': {}
        }
        
        # Model configurations
        self.ensemble_configs = self._get_ensemble_configs()
        self.lightgbm_configs = self._get_lightgbm_configs()
        
        logger.info(f"📊 Model Performance Comparator initialized")
        logger.info(f"📁 Results directory: {self.results_dir}")
    
    def _get_ensemble_configs(self) -> Dict[str, Any]:
        """Get ensemble model configurations"""
        return {
            'random_forest': {
                'classifier': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'regressor': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            },
            'voting_ensemble': {
                'classifier': VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                        ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42))
                    ],
                    voting='soft'
                ),
                'regressor': VotingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42))
                    ]
                )
            },
            'stacking_ensemble': {
                'classifier': StackingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=30, random_state=42)),
                        ('xgb', xgb.XGBClassifier(n_estimators=30, random_state=42))
                    ],
                    final_estimator=LogisticRegression(),
                    cv=3
                ),
                'regressor': StackingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(n_estimators=30, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=30, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=30, random_state=42))
                    ],
                    final_estimator=LinearRegression(),
                    cv=3
                )
            }
        }
    
    def _get_lightgbm_configs(self) -> Dict[str, Any]:
        """Get LightGBM model configurations"""
        if not LIGHTGBM_AVAILABLE:
            return {}
            
        return {
            'lightgbm_momentum': {
                'classifier': LGBMClassifier(
                    boosting_type='gbdt',
                    objective='binary',
                    metric='binary_logloss',
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.9,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    verbose=-1,
                    random_state=42
                ),
                'regressor': LGBMRegressor(
                    boosting_type='gbdt',
                    objective='regression',
                    metric='rmse',
                    num_leaves=31,
                    learning_rate=0.05,
                    feature_fraction=0.9,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    verbose=-1,
                    random_state=42
                )
            },
            'lightgbm_optimized': {
                'classifier': LGBMClassifier(
                    boosting_type='gbdt',
                    objective='binary',
                    num_leaves=127,
                    learning_rate=0.07,
                    feature_fraction=0.4,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    max_depth=10,
                    min_child_samples=20,
                    reg_alpha=0.15,
                    reg_lambda=0.525,
                    verbose=-1,
                    random_state=42
                ),
                'regressor': LGBMRegressor(
                    boosting_type='gbdt',
                    objective='regression',
                    num_leaves=127,
                    learning_rate=0.07,
                    feature_fraction=0.4,
                    bagging_fraction=0.8,
                    bagging_freq=5,
                    max_depth=10,
                    min_child_samples=20,
                    reg_alpha=0.15,
                    reg_lambda=0.525,
                    verbose=-1,
                    random_state=42
                )
            },
            'lightgbm_ensemble': {
                'regressor': VotingRegressor(
                    estimators=[
                        ('lgbm1', LGBMRegressor(
                            boosting_type='gbdt',
                            num_leaves=127,
                            learning_rate=0.07,
                            feature_fraction=0.4,
                            max_depth=10,
                            random_state=42,
                            verbose=-1
                        )),
                        ('lgbm2', LGBMRegressor(
                            boosting_type='gbdt',
                            num_leaves=7,
                            learning_rate=0.08,
                            feature_fraction=0.4,
                            max_depth=7,
                            random_state=43,
                            verbose=-1
                        )),
                        ('lgbm3', LGBMRegressor(
                            boosting_type='gbdt',
                            num_leaves=255,
                            learning_rate=0.11,
                            feature_fraction=0.4,
                            max_depth=10,
                            random_state=44,
                            verbose=-1
                        ))
                    ],
                    weights=[0.35, 0.35, 0.30]
                )
            }
        }
    
    def prepare_data(self, symbols: List[str] = ["BTC-USD", "ETH-USD"], 
                    granularity: int = 3600, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for model comparison
        """
        logger.info(f"📊 Preparing data for {len(symbols)} symbols...")
        
        datasets = {}
        
        for symbol in symbols:
            try:
                logger.info(f"📈 Fetching data for {symbol}...")
                
                # Get raw data
                df = get_coinbase_data(symbol, granularity, days)
                
                if df.empty:
                    logger.warning(f"❌ No data available for {symbol}")
                    continue
                
                # Calculate technical indicators
                df = calculate_indicators(df)
                
                # Add momentum features
                df = self._add_momentum_features(df)
                
                # Add price targets and signals
                df = self._add_trading_targets(df)
                
                # Clean data
                df = df.dropna()
                
                if len(df) < 50:
                    logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} rows")
                    continue
                
                datasets[symbol] = df
                logger.info(f"✅ Prepared {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error preparing data for {symbol}: {str(e)}")
                continue
        
        logger.info(f"✅ Data preparation complete for {len(datasets)} symbols")
        return datasets
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-specific features"""
        try:
            # Price momentum
            df['price_momentum_5'] = df['close'].pct_change(5)
            df['price_momentum_10'] = df['close'].pct_change(10)
            df['price_momentum_20'] = df['close'].pct_change(20)
            
            # Volume momentum
            df['volume_momentum_5'] = df['volume'].pct_change(5)
            df['volume_momentum_10'] = df['volume'].pct_change(10)
            
            # RSI momentum
            if 'rsi' in df.columns:
                df['rsi_momentum'] = df['rsi'].diff(5)
                df['rsi_acceleration'] = df['rsi_momentum'].diff()
            
            # MACD momentum
            if 'macd' in df.columns:
                df['macd_momentum'] = df['macd'].diff(3)
                df['macd_signal_momentum'] = df['macd_signal'].diff(3) if 'macd_signal' in df.columns else 0
            
            # Bollinger Band momentum
            if 'upper_band' in df.columns and 'lower_band' in df.columns:
                df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['close']
                df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
                df['bb_momentum'] = df['bb_position'].diff(5)
            
            # Moving average convergence/divergence
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                df['ma_convergence'] = (df['sma_20'] - df['sma_50']) / df['close']
                df['ma_momentum'] = df['ma_convergence'].diff(5)
            
            # Volatility momentum
            df['volatility'] = df['close'].rolling(10).std()
            df['volatility_momentum'] = df['volatility'].pct_change(5)
            
            # Price acceleration
            df['price_acceleration'] = df['price_momentum_5'].diff()
            
            logger.debug("✅ Added momentum features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding momentum features: {str(e)}")
            return df
    
    def _add_trading_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading targets and signals"""
        try:
            # Future price changes for regression
            df['future_return_1h'] = df['close'].pct_change(-1)  # Next period return
            df['future_return_4h'] = df['close'].pct_change(-4)  # 4 periods ahead
            df['future_return_24h'] = df['close'].pct_change(-24)  # 24 periods ahead
            
            # Classification targets (buy/sell signals)
            df['signal_1h'] = (df['future_return_1h'] > 0.001).astype(int)  # >0.1% gain
            df['signal_4h'] = (df['future_return_4h'] > 0.005).astype(int)  # >0.5% gain
            df['signal_24h'] = (df['future_return_24h'] > 0.01).astype(int)  # >1% gain
            
            # Strong signal targets
            df['strong_signal_1h'] = (df['future_return_1h'] > 0.002).astype(int)  # >0.2% gain
            df['strong_signal_4h'] = (df['future_return_4h'] > 0.01).astype(int)   # >1% gain
            
            logger.debug("✅ Added trading targets")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding trading targets: {str(e)}")
            return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for model training"""
        # Exclude target columns and timestamp
        exclude_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'future_return_1h', 'future_return_4h', 'future_return_24h',
            'signal_1h', 'signal_4h', 'signal_24h', 
            'strong_signal_1h', 'strong_signal_4h'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isna().sum() < len(df) * 0.5]
        
        logger.debug(f"📊 Selected {len(feature_cols)} feature columns")
        return feature_cols
    
    def train_and_evaluate_models(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train and evaluate all models on all datasets
        """
        logger.info("🚀 Starting model training and evaluation...")
        
        results = {
            'ensemble_models': {},
            'lightgbm_models': {},
            'model_comparison': {},
            'training_times': {}
        }
        
        for symbol, df in datasets.items():
            logger.info(f"📈 Evaluating models for {symbol}...")
            
            # Prepare features and targets
            feature_cols = self.get_feature_columns(df)
            X = df[feature_cols].fillna(0)
            
            # Define targets
            targets = {
                'classification': {
                    'signal_1h': df['signal_1h'],
                    'signal_4h': df['signal_4h'],
                    'strong_signal_1h': df['strong_signal_1h']
                },
                'regression': {
                    'future_return_1h': df['future_return_1h'],
                    'future_return_4h': df['future_return_4h'],
                    'future_return_24h': df['future_return_24h']
                }
            }
            
            symbol_results = {
                'ensemble_models': {},
                'lightgbm_models': {},
                'data_info': {
                    'total_samples': len(df),
                    'features': len(feature_cols),
                    'feature_names': feature_cols[:10]  # Store first 10 for reference
                }
            }
            
            # Split data chronologically for time series
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            
            # Evaluate ensemble models
            logger.info(f"🎯 Evaluating ensemble models for {symbol}...")
            for model_name, config in self.ensemble_configs.items():
                symbol_results['ensemble_models'][model_name] = self._evaluate_model_config(
                    model_name, config, X_train, X_test, targets, split_idx, df
                )
            
            # Evaluate LightGBM models
            if LIGHTGBM_AVAILABLE:
                logger.info(f"⚡ Evaluating LightGBM models for {symbol}...")
                for model_name, config in self.lightgbm_configs.items():
                    symbol_results['lightgbm_models'][model_name] = self._evaluate_model_config(
                        model_name, config, X_train, X_test, targets, split_idx, df
                    )
            else:
                logger.warning("⚠️ LightGBM not available, skipping LightGBM models")
            
            results[symbol] = symbol_results
        
        # Generate comparison summary
        results['comparison_summary'] = self._generate_comparison_summary(results)
        
        logger.info("✅ Model training and evaluation complete!")
        return results
    
    def _evaluate_model_config(self, model_name: str, config: Dict[str, Any], 
                             X_train: pd.DataFrame, X_test: pd.DataFrame,
                             targets: Dict[str, Dict[str, pd.Series]], 
                             split_idx: int, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a specific model configuration"""
        
        model_results = {
            'classification': {},
            'regression': {},
            'training_time': {},
            'prediction_time': {}
        }
        
        # Evaluate classification tasks
        if 'classifier' in config:
            for target_name, y_full in targets['classification'].items():
                y_train, y_test = y_full[:split_idx], y_full[split_idx:]
                
                # Skip if not enough positive samples
                if y_train.sum() < 10 or y_test.sum() < 5:
                    continue
                
                try:
                    start_time = time.time()
                    model = config['classifier']
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    start_time = time.time()
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    prediction_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    
                    model_results['classification'][target_name] = metrics
                    model_results['training_time'][f'classification_{target_name}'] = training_time
                    model_results['prediction_time'][f'classification_{target_name}'] = prediction_time
                    
                except Exception as e:
                    logger.error(f"❌ Error evaluating {model_name} classifier on {target_name}: {str(e)}")
                    continue
        
        # Evaluate regression tasks
        if 'regressor' in config:
            for target_name, y_full in targets['regression'].items():
                y_train, y_test = y_full[:split_idx], y_full[split_idx:]
                
                # Remove NaN values
                valid_idx = ~(y_train.isna() | y_test.isna())
                if valid_idx.sum() < 20:
                    continue
                
                try:
                    start_time = time.time()
                    model = config['regressor']
                    model.fit(X_train[valid_idx[:len(X_train)]], y_train[valid_idx[:len(y_train)]])
                    training_time = time.time() - start_time
                    
                    start_time = time.time()
                    y_pred = model.predict(X_test)
                    prediction_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                    
                    model_results['regression'][target_name] = metrics
                    model_results['training_time'][f'regression_{target_name}'] = training_time
                    model_results['prediction_time'][f'regression_{target_name}'] = prediction_time
                    
                except Exception as e:
                    logger.error(f"❌ Error evaluating {model_name} regressor on {target_name}: {str(e)}")
                    continue
        
        return model_results
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                        y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        try:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'positive_rate': y_pred.mean(),
                'true_positive_rate': y_true.mean(),
                'samples': len(y_true)
            }
        except Exception as e:
            logger.error(f"❌ Error calculating classification metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        try:
            # Remove NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {'error': 'No valid predictions'}
            
            return {
                'mse': mean_squared_error(y_true_clean, y_pred_clean),
                'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
                'mae': mean_absolute_error(y_true_clean, y_pred_clean),
                'r2_score': r2_score(y_true_clean, y_pred_clean),
                'mean_error': np.mean(y_pred_clean - y_true_clean),
                'samples': len(y_true_clean)
            }
        except Exception as e:
            logger.error(f"❌ Error calculating regression metrics: {str(e)}")
            return {'error': str(e)}
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary between ensemble and LightGBM models"""
        
        summary = {
            'performance_comparison': {},
            'speed_comparison': {},
            'best_models': {},
            'recommendations': []
        }
        
        try:
            # Aggregate results across symbols
            ensemble_metrics = {'classification': {}, 'regression': {}}
            lightgbm_metrics = {'classification': {}, 'regression': {}}
            
            for symbol, symbol_results in results.items():
                if symbol in ['comparison_summary']:
                    continue
                
                # Aggregate ensemble model performance
                if 'ensemble_models' in symbol_results:
                    for model_name, model_results in symbol_results['ensemble_models'].items():
                        if 'classification' in model_results:
                            for target, metrics in model_results['classification'].items():
                                if isinstance(metrics, dict) and 'f1_score' in metrics:
                                    key = f"{model_name}_{target}"
                                    if key not in ensemble_metrics['classification']:
                                        ensemble_metrics['classification'][key] = []
                                    ensemble_metrics['classification'][key].append(metrics['f1_score'])
                        
                        if 'regression' in model_results:
                            for target, metrics in model_results['regression'].items():
                                if isinstance(metrics, dict) and 'r2_score' in metrics:
                                    key = f"{model_name}_{target}"
                                    if key not in ensemble_metrics['regression']:
                                        ensemble_metrics['regression'][key] = []
                                    ensemble_metrics['regression'][key].append(metrics['r2_score'])
                
                # Aggregate LightGBM model performance
                if 'lightgbm_models' in symbol_results:
                    for model_name, model_results in symbol_results['lightgbm_models'].items():
                        if 'classification' in model_results:
                            for target, metrics in model_results['classification'].items():
                                if isinstance(metrics, dict) and 'f1_score' in metrics:
                                    key = f"{model_name}_{target}"
                                    if key not in lightgbm_metrics['classification']:
                                        lightgbm_metrics['classification'][key] = []
                                    lightgbm_metrics['classification'][key].append(metrics['f1_score'])
                        
                        if 'regression' in model_results:
                            for target, metrics in model_results['regression'].items():
                                if isinstance(metrics, dict) and 'r2_score' in metrics:
                                    key = f"{model_name}_{target}"
                                    if key not in lightgbm_metrics['regression']:
                                        lightgbm_metrics['regression'][key] = []
                                    lightgbm_metrics['regression'][key].append(metrics['r2_score'])
            
            # Calculate average performance
            ensemble_avg = {}
            lightgbm_avg = {}
            
            for task_type in ['classification', 'regression']:
                ensemble_avg[task_type] = {
                    key: np.mean(values) for key, values in ensemble_metrics[task_type].items()
                }
                lightgbm_avg[task_type] = {
                    key: np.mean(values) for key, values in lightgbm_metrics[task_type].items()
                }
            
            summary['performance_comparison'] = {
                'ensemble_average': ensemble_avg,
                'lightgbm_average': lightgbm_avg
            }
            
            # Determine best models
            if ensemble_avg['classification'] or lightgbm_avg['classification']:
                best_classification = max(
                    list(ensemble_avg['classification'].items()) + list(lightgbm_avg['classification'].items()),
                    key=lambda x: x[1]
                )
                summary['best_models']['classification'] = best_classification
            
            if ensemble_avg['regression'] or lightgbm_avg['regression']:
                best_regression = max(
                    list(ensemble_avg['regression'].items()) + list(lightgbm_avg['regression'].items()),
                    key=lambda x: x[1]
                )
                summary['best_models']['regression'] = best_regression
            
            # Generate recommendations
            summary['recommendations'] = self._generate_recommendations(
                ensemble_avg, lightgbm_avg
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating comparison summary: {str(e)}")
            summary['error'] = str(e)
        
        return summary
    
    def _generate_recommendations(self, ensemble_avg: Dict, lightgbm_avg: Dict) -> List[str]:
        """Generate model recommendations based on performance"""
        recommendations = []
        
        try:
            # Compare classification performance
            if ensemble_avg['classification'] and lightgbm_avg['classification']:
                ens_class_avg = np.mean(list(ensemble_avg['classification'].values()))
                lgb_class_avg = np.mean(list(lightgbm_avg['classification'].values()))
                
                if lgb_class_avg > ens_class_avg * 1.05:  # 5% better
                    recommendations.append(
                        f"🚀 LightGBM models show {((lgb_class_avg - ens_class_avg) / ens_class_avg * 100):.1f}% "
                        "better classification performance"
                    )
                elif ens_class_avg > lgb_class_avg * 1.05:
                    recommendations.append(
                        f"🎯 Ensemble models show {((ens_class_avg - lgb_class_avg) / lgb_class_avg * 100):.1f}% "
                        "better classification performance"
                    )
                else:
                    recommendations.append("📊 Classification performance is similar between model types")
            
            # Compare regression performance
            if ensemble_avg['regression'] and lightgbm_avg['regression']:
                ens_reg_avg = np.mean(list(ensemble_avg['regression'].values()))
                lgb_reg_avg = np.mean(list(lightgbm_avg['regression'].values()))
                
                if lgb_reg_avg > ens_reg_avg * 1.05:
                    recommendations.append(
                        f"📈 LightGBM models show {((lgb_reg_avg - ens_reg_avg) / ens_reg_avg * 100):.1f}% "
                        "better regression performance"
                    )
                elif ens_reg_avg > lgb_reg_avg * 1.05:
                    recommendations.append(
                        f"📉 Ensemble models show {((ens_reg_avg - lgb_reg_avg) / lgb_reg_avg * 100):.1f}% "
                        "better regression performance"
                    )
                else:
                    recommendations.append("📊 Regression performance is similar between model types")
            
            # Add specific recommendations
            if LIGHTGBM_AVAILABLE:
                recommendations.extend([
                    "⚡ LightGBM models are typically faster and more memory efficient",
                    "🎯 Ensemble models provide better interpretability and robustness",
                    "🔄 Consider using LightGBM for frequent retraining scenarios",
                    "🎪 Use ensemble models for critical trading decisions requiring stability"
                ])
            else:
                recommendations.append("📦 Install LightGBM for comprehensive comparison: pip install lightgbm")
        
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {str(e)}")
            recommendations.append(f"❌ Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save comparison results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {str(e)}")
            return None
    
    def create_comparison_report(self, results: Dict[str, Any]) -> str:
        """Create a comprehensive comparison report"""
        
        report_lines = [
            "=" * 80,
            "MODEL PERFORMANCE COMPARISON REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 EXECUTIVE SUMMARY",
            "-" * 40
        ]
        
        try:
            # Add executive summary
            if 'comparison_summary' in results and 'recommendations' in results['comparison_summary']:
                for recommendation in results['comparison_summary']['recommendations']:
                    report_lines.append(f"• {recommendation}")
            
            report_lines.extend([
                "",
                "🎯 DETAILED RESULTS BY SYMBOL",
                "-" * 40
            ])
            
            # Add detailed results for each symbol
            for symbol, symbol_results in results.items():
                if symbol == 'comparison_summary':
                    continue
                
                report_lines.extend([
                    f"",
                    f"📈 {symbol}",
                    f"{'─' * len(symbol) + '─' * 2}"
                ])
                
                if 'data_info' in symbol_results:
                    info = symbol_results['data_info']
                    report_lines.append(f"Data: {info['total_samples']} samples, {info['features']} features")
                
                # Ensemble model results
                if 'ensemble_models' in symbol_results:
                    report_lines.append("\n🎪 Ensemble Models:")
                    for model_name, model_results in symbol_results['ensemble_models'].items():
                        report_lines.append(f"  • {model_name}")
                        
                        if 'classification' in model_results:
                            for target, metrics in model_results['classification'].items():
                                if isinstance(metrics, dict) and 'f1_score' in metrics:
                                    report_lines.append(
                                        f"    - {target}: F1={metrics['f1_score']:.3f}, "
                                        f"Acc={metrics['accuracy']:.3f}"
                                    )
                        
                        if 'regression' in model_results:
                            for target, metrics in model_results['regression'].items():
                                if isinstance(metrics, dict) and 'r2_score' in metrics:
                                    report_lines.append(
                                        f"    - {target}: R²={metrics['r2_score']:.3f}, "
                                        f"RMSE={metrics['rmse']:.6f}"
                                    )
                
                # LightGBM model results
                if 'lightgbm_models' in symbol_results and symbol_results['lightgbm_models']:
                    report_lines.append("\n⚡ LightGBM Models:")
                    for model_name, model_results in symbol_results['lightgbm_models'].items():
                        report_lines.append(f"  • {model_name}")
                        
                        if 'classification' in model_results:
                            for target, metrics in model_results['classification'].items():
                                if isinstance(metrics, dict) and 'f1_score' in metrics:
                                    report_lines.append(
                                        f"    - {target}: F1={metrics['f1_score']:.3f}, "
                                        f"Acc={metrics['accuracy']:.3f}"
                                    )
                        
                        if 'regression' in model_results:
                            for target, metrics in model_results['regression'].items():
                                if isinstance(metrics, dict) and 'r2_score' in metrics:
                                    report_lines.append(
                                        f"    - {target}: R²={metrics['r2_score']:.3f}, "
                                        f"RMSE={metrics['rmse']:.6f}"
                                    )
            
            # Add best models summary
            if 'comparison_summary' in results and 'best_models' in results['comparison_summary']:
                report_lines.extend([
                    "",
                    "🏆 BEST PERFORMING MODELS",
                    "-" * 40
                ])
                
                best_models = results['comparison_summary']['best_models']
                if 'classification' in best_models:
                    model_name, score = best_models['classification']
                    report_lines.append(f"📊 Best Classification: {model_name} (F1={score:.3f})")
                
                if 'regression' in best_models:
                    model_name, score = best_models['regression']
                    report_lines.append(f"📈 Best Regression: {model_name} (R²={score:.3f})")
            
            report_lines.extend([
                "",
                "=" * 80,
                "End of Report",
                "=" * 80
            ])
            
        except Exception as e:
            logger.error(f"❌ Error creating report: {str(e)}")
            report_lines.extend([
                "",
                f"❌ Error creating detailed report: {str(e)}",
                "=" * 80
            ])
        
        return "\n".join(report_lines)
    
    def run_full_comparison(self, symbols: List[str] = ["BTC-USD", "ETH-USD"], 
                          granularity: int = 3600, days: int = 30) -> Dict[str, Any]:
        """
        Run the complete model comparison pipeline
        """
        logger.info("🚀 Starting full model performance comparison...")
        
        try:
            # Prepare data
            datasets = self.prepare_data(symbols, granularity, days)
            
            if not datasets:
                logger.error("❌ No datasets available for comparison")
                return {'error': 'No datasets available'}
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(datasets)
            
            # Save results
            results_file = self.save_results(results)
            
            # Create and save report
            report = self.create_comparison_report(results)
            report_file = self.results_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"✅ Full comparison complete!")
            logger.info(f"📄 Report saved to: {report_file}")
            logger.info(f"📊 Results saved to: {results_file}")
            
            # Print summary to console
            print("\n" + "=" * 80)
            print("MODEL PERFORMANCE COMPARISON SUMMARY")
            print("=" * 80)
            
            if 'comparison_summary' in results and 'recommendations' in results['comparison_summary']:
                print("\n🎯 KEY FINDINGS:")
                for recommendation in results['comparison_summary']['recommendations']:
                    print(f"  • {recommendation}")
            
            print(f"\n📊 Analyzed {len(datasets)} symbols with {len(self.ensemble_configs)} ensemble models", end="")
            if LIGHTGBM_AVAILABLE:
                print(f" and {len(self.lightgbm_configs)} LightGBM models")
            else:
                print(" (LightGBM not available)")
            
            print(f"\n📄 Full report: {report_file}")
            print("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in full comparison: {str(e)}")
            return {'error': str(e)}


def main():
    """Main function to run the comparison"""
    print("🚀 Model Performance Comparison: Ensemble vs LightGBM")
    print("=" * 60)
    
    # Initialize comparator
    comparator = ModelPerformanceComparator()
    
    # Configure symbols and parameters
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    granularity = 3600  # 1 hour
    days = 30  # 30 days of data
    
    print(f"📊 Configuration:")
    print(f"  • Symbols: {', '.join(symbols)}")
    print(f"  • Granularity: {granularity}s ({granularity//3600}h)")
    print(f"  • Data Period: {days} days")
    print(f"  • LightGBM Available: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    print()
    
    # Run comparison
    results = comparator.run_full_comparison(symbols, granularity, days)
    
    if 'error' in results:
        print(f"❌ Comparison failed: {results['error']}")
        return
    
    print("\n✅ Comparison completed successfully!")
    print("\nTo view detailed results, check the generated report file.")


if __name__ == "__main__":
    main() 