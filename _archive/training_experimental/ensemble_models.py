#!/usr/bin/env python3
"""
Ensemble Models for Trading Integration
======================================

This module integrates the best-performing ensemble models from the 
performance comparison framework into the main trading system.

Based on comparison results:
- Ensemble models show 21.2% better regression performance
- Better interpretability and robustness for critical trading decisions
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb

# Local imports
try:
    from maybe import get_coinbase_data, calculate_indicators
    from model_performance_comparison import ModelPerformanceComparator
except ImportError as e:
    logging.warning(f"Could not import local modules: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class EnsembleModelManager:
    """
    Manages ensemble models for trading decisions
    """
    
    def __init__(self, models_dir: str = "enhanced_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize model configurations
        self.ensemble_models = {}
        self.model_metadata = {}
        self.scaler = None
        
        # Trading signal thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.75,
            'medium_confidence': 0.60,
            'low_confidence': 0.50
        }
        
        logger.info(f"🎯 Ensemble Model Manager initialized")
        logger.info(f"📁 Models directory: {self.models_dir}")
    
    def build_ensemble_models(self) -> Dict[str, Any]:
        """Build optimized ensemble models for trading"""
        
        ensemble_configs = {
            'random_forest_ensemble': {
                'classifier': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'regressor': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
            },
            'voting_ensemble': {
                'classifier': VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=42)),
                        ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=8, random_state=42))
                    ],
                    voting='soft'
                ),
                'regressor': VotingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42))
                    ]
                )
            },
            'stacking_ensemble': {
                'classifier': StackingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=6, random_state=42)),
                        ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=6, random_state=42))
                    ],
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=5
                ),
                'regressor': StackingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)),
                        ('gb', GradientBoostingRegressor(n_estimators=80, max_depth=6, random_state=42)),
                        ('xgb', xgb.XGBRegressor(n_estimators=80, max_depth=6, random_state=42))
                    ],
                    final_estimator=LinearRegression(),
                    cv=5
                )
            }
        }
        
        self.ensemble_models = ensemble_configs
        logger.info(f"✅ Built {len(ensemble_configs)} ensemble model configurations")
        return ensemble_configs
    
    def prepare_training_data(self, symbols: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"], 
                            granularity: int = 3600, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Prepare training data with enhanced features"""
        
        datasets = {}
        
        for symbol in symbols:
            try:
                logger.info(f"📈 Preparing data for {symbol}...")
                
                # Get data
                df = get_coinbase_data(symbol, granularity, days)
                if df is None or len(df) < 100:
                    logger.warning(f"❌ Insufficient data for {symbol}")
                    continue
                
                # Calculate comprehensive indicators
                df = calculate_indicators(df)
                if df is None:
                    logger.warning(f"❌ Failed to calculate indicators for {symbol}")
                    continue
                
                # Add momentum features
                df = self._add_enhanced_momentum_features(df)
                
                # Add trading targets
                df = self._add_trading_targets(df)
                
                # Remove rows with NaN values
                df = df.dropna()
                
                if len(df) < 50:
                    logger.warning(f"❌ Insufficient clean data for {symbol}")
                    continue
                
                datasets[symbol] = df
                logger.info(f"✅ Prepared {len(df)} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"❌ Error preparing data for {symbol}: {str(e)}")
                continue
        
        return datasets
    
    def _add_enhanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced momentum and technical features"""
        
        # Price momentum features
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Volume momentum
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_momentum'] = df['volume'] / df['volume_sma_10'] - 1
        
        # RSI momentum
        if 'RSI_14' in df.columns:
            df['rsi_momentum'] = df['RSI_14'].diff()
            df['rsi_acceleration'] = df['rsi_momentum'].diff()
        
        # MACD momentum
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['macd_momentum'] = df['MACD'] - df['MACD_Signal']
            df['macd_momentum_change'] = df['macd_momentum'].diff()
        
        # Bollinger Band features
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_momentum'] = df['bb_position'].diff()
        
        # Moving average convergence
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['ma_convergence'] = df['SMA_20'] / df['SMA_50'] - 1
            df['ma_convergence_momentum'] = df['ma_convergence'].diff()
        
        # Volatility features
        df['volatility_10'] = df['close'].rolling(10).std()
        df['volatility_momentum'] = df['volatility_10'].pct_change()
        
        # Price acceleration
        df['price_acceleration'] = df['close'].diff().diff()
        
        return df
    
    def _add_trading_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading targets for different timeframes"""
        
        # Future returns for regression
        df['return_1h'] = df['close'].shift(-1) / df['close'] - 1
        df['return_4h'] = df['close'].shift(-4) / df['close'] - 1
        df['return_24h'] = df['close'].shift(-24) / df['close'] - 1
        
        # Classification targets (buy/sell signals)
        df['signal_1h'] = (df['return_1h'] > 0.01).astype(int)  # 1% threshold
        df['signal_4h'] = (df['return_4h'] > 0.02).astype(int)  # 2% threshold
        df['strong_signal_1h'] = (df['return_1h'] > 0.02).astype(int)  # Strong signal: 2%
        
        return df
    
    def train_ensemble_models(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train ensemble models on prepared data"""
        
        if not self.ensemble_models:
            self.build_ensemble_models()
        
        training_results = {}
        
        # Combine all datasets
        all_data = []
        for symbol, df in datasets.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_data.append(df_copy)
        
        if not all_data:
            logger.error("❌ No training data available")
            return {}
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Combined training data: {len(combined_df)} rows")
        
        # Prepare features and targets
        feature_columns = self._get_feature_columns(combined_df)
        
        # Handle missing values
        X = combined_df[feature_columns].fillna(0)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        # Train models for each configuration
        for model_name, config in self.ensemble_models.items():
            logger.info(f"🎯 Training {model_name}...")
            
            try:
                model_results = {}
                
                # Train classification models
                for target in ['signal_1h', 'signal_4h', 'strong_signal_1h']:
                    if target in combined_df.columns:
                        y = combined_df[target].fillna(0)
                        
                        # Split data chronologically
                        split_idx = int(len(X_scaled) * 0.8)
                        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Train classifier
                        classifier = config['classifier']
                        classifier.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = classifier.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        model_results[f'{target}_classifier'] = {
                            'model': classifier,
                            'accuracy': accuracy,
                            'features': feature_columns
                        }
                        
                        logger.info(f"✅ {target} classifier accuracy: {accuracy:.3f}")
                
                # Train regression models
                for target in ['return_1h', 'return_4h', 'return_24h']:
                    if target in combined_df.columns:
                        y = combined_df[target].fillna(0)
                        
                        # Split data chronologically
                        split_idx = int(len(X_scaled) * 0.8)
                        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                        y_train, y_test = y[:split_idx], y[split_idx:]
                        
                        # Train regressor
                        regressor = config['regressor']
                        regressor.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = regressor.predict(X_test)
                        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                        
                        model_results[f'{target}_regressor'] = {
                            'model': regressor,
                            'rmse': rmse,
                            'features': feature_columns
                        }
                        
                        logger.info(f"✅ {target} regressor RMSE: {rmse:.4f}")
                
                training_results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        # Save models
        self._save_models(training_results)
        
        logger.info(f"✅ Training complete for {len(training_results)} model types")
        return training_results
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get relevant feature columns for training"""
        
        exclude_columns = [
            'symbol', 'return_1h', 'return_4h', 'return_24h',
            'signal_1h', 'signal_4h', 'strong_signal_1h',
            'open', 'high', 'low', 'close', 'volume'  # Raw OHLCV data
        ]
        
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns and 
                          not col.startswith('Unnamed')]
        
        return feature_columns
    
    def get_ensemble_trading_decision(self, symbol: str, 
                                    model_type: str = 'stacking_ensemble') -> Dict[str, Any]:
        """Get trading decision from ensemble models"""
        
        try:
            # Get recent data
            df = get_coinbase_data(symbol, granularity=3600, days=7)
            if df is None or len(df) < 50:
                return self._get_default_decision()
            
            # Calculate indicators and features
            df = calculate_indicators(df)
            df = self._add_enhanced_momentum_features(df)
            
            # Get latest row
            latest_data = df.iloc[-1:][self._get_feature_columns(df)].fillna(0)
            
            # Scale features
            if self.scaler is not None:
                latest_scaled = self.scaler.transform(latest_data)
                latest_scaled = pd.DataFrame(latest_scaled, columns=latest_data.columns)
            else:
                latest_scaled = latest_data
            
            # Load models if not in memory
            if model_type not in self.ensemble_models:
                self._load_models()
            
            # Get predictions from different models
            predictions = {}
            
            try:
                # Classification predictions
                for target in ['signal_1h', 'signal_4h', 'strong_signal_1h']:
                    model_key = f'{target}_classifier'
                    if model_key in self.model_metadata.get(model_type, {}):
                        model = self.model_metadata[model_type][model_key]['model']
                        
                        # Get probability predictions
                        prob_pred = model.predict_proba(latest_scaled)
                        if len(prob_pred[0]) > 1:
                            predictions[target] = {
                                'probability': prob_pred[0][1],  # Probability of positive class
                                'prediction': model.predict(latest_scaled)[0]
                            }
                
                # Regression predictions
                for target in ['return_1h', 'return_4h', 'return_24h']:
                    model_key = f'{target}_regressor'
                    if model_key in self.model_metadata.get(model_type, {}):
                        model = self.model_metadata[model_type][model_key]['model']
                        
                        return_pred = model.predict(latest_scaled)[0]
                        predictions[target] = {
                            'predicted_return': return_pred
                        }
                
                # Generate combined decision
                decision = self._generate_ensemble_decision(predictions)
                
                logger.info(f"🎯 Ensemble decision for {symbol}: {decision['action']} "
                          f"(confidence: {decision['confidence']:.2%})")
                
                return decision
                
            except Exception as e:
                logger.error(f"❌ Error making prediction: {str(e)}")
                return self._get_default_decision()
                
        except Exception as e:
            logger.error(f"❌ Error getting ensemble decision for {symbol}: {str(e)}")
            return self._get_default_decision()
    
    def _generate_ensemble_decision(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading decision from ensemble predictions"""
        
        # Initialize decision components
        buy_signals = 0
        sell_signals = 0
        confidence_scores = []
        
        # Analyze classification signals
        for signal_type in ['signal_1h', 'signal_4h', 'strong_signal_1h']:
            if signal_type in predictions:
                prob = predictions[signal_type]['probability']
                pred = predictions[signal_type]['prediction']
                
                if pred == 1:  # Buy signal
                    buy_signals += 1
                    confidence_scores.append(prob)
                else:  # Hold/sell signal
                    sell_signals += 1
                    confidence_scores.append(1 - prob)
        
        # Analyze regression predictions
        expected_returns = []
        for return_type in ['return_1h', 'return_4h', 'return_24h']:
            if return_type in predictions:
                ret = predictions[return_type]['predicted_return']
                expected_returns.append(ret)
        
        # Calculate weighted decision
        avg_expected_return = np.mean(expected_returns) if expected_returns else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Determine action
        if buy_signals > sell_signals and avg_expected_return > 0.005:  # 0.5% threshold
            action = "BUY"
            confidence = min(avg_confidence * (1 + abs(avg_expected_return) * 10), 0.95)
        elif sell_signals > buy_signals or avg_expected_return < -0.005:  # -0.5% threshold
            action = "SELL"
            confidence = min(avg_confidence * (1 + abs(avg_expected_return) * 10), 0.95)
        else:
            action = "HOLD"
            confidence = 0.5
        
        # Apply confidence thresholds
        if confidence < self.confidence_thresholds['low_confidence']:
            action = "HOLD"
        
        return {
            'action': action,
            'confidence': confidence,
            'expected_return': avg_expected_return,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'raw_predictions': predictions,
            'model_type': 'ensemble',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_decision(self) -> Dict[str, Any]:
        """Get default decision when models fail"""
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'expected_return': 0.0,
            'buy_signals': 0,
            'sell_signals': 0,
            'raw_predictions': {},
            'model_type': 'default',
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_models(self, training_results: Dict[str, Any]):
        """Save trained models to disk"""
        
        for model_name, results in training_results.items():
            model_path = self.models_dir / f"{model_name}_ensemble.pkl"
            
            try:
                # Save model and metadata
                model_data = {
                    'models': results,
                    'scaler': self.scaler,
                    'timestamp': datetime.now().isoformat(),
                    'feature_columns': results[list(results.keys())[0]]['features'] if results else []
                }
                
                joblib.dump(model_data, model_path)
                logger.info(f"💾 Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"❌ Error saving {model_name}: {str(e)}")
    
    def _load_models(self):
        """Load models from disk"""
        
        try:
            for model_file in self.models_dir.glob("*_ensemble.pkl"):
                model_name = model_file.stem.replace("_ensemble", "")
                
                model_data = joblib.load(model_file)
                self.model_metadata[model_name] = model_data['models']
                
                if self.scaler is None and 'scaler' in model_data:
                    self.scaler = model_data['scaler']
                
                logger.info(f"📥 Loaded {model_name} from {model_file}")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")

# Global ensemble manager instance
ensemble_manager = EnsembleModelManager()

def get_ensemble_prediction(symbol: str, model_type: str = 'stacking_ensemble') -> Dict[str, Any]:
    """
    Main function to get ensemble trading prediction
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        model_type: Type of ensemble model to use
        
    Returns:
        Dictionary with trading decision and confidence
    """
    return ensemble_manager.get_ensemble_trading_decision(symbol, model_type)

def train_ensemble_models_for_trading(symbols: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]):
    """
    Train ensemble models for trading
    
    Args:
        symbols: List of trading symbols to train on
    """
    logger.info("🚀 Starting ensemble model training for trading...")
    
    # Prepare data
    datasets = ensemble_manager.prepare_training_data(symbols)
    
    if not datasets:
        logger.error("❌ No training data available")
        return
    
    # Train models
    results = ensemble_manager.train_ensemble_models(datasets)
    
    if results:
        logger.info("✅ Ensemble model training completed successfully")
        return results
    else:
        logger.error("❌ Ensemble model training failed")
        return None

if __name__ == "__main__":
    # Example usage
    print("🎯 Ensemble Models for Trading")
    print("=" * 50)
    
    # Train models
    train_ensemble_models_for_trading()
    
    # Test prediction
    decision = get_ensemble_prediction("BTC-USD")
    print(f"\n📊 Sample decision: {decision}") 