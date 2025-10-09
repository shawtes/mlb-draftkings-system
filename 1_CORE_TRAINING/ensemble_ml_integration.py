#!/usr/bin/env python3
"""
Ensemble ML Integration for Trading Dashboard
============================================

This module integrates the best-performing ensemble models to replace 
the current ML decision system in the trading dashboard.

Based on performance comparison results:
✅ Ensemble models show superior performance
✅ Better interpretability and robustness
✅ Optimized for cryptocurrency trading
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
import sqlite3
from typing import Dict, List, Tuple, Optional, Any

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Global constants
ENSEMBLE_MODELS_DIR = Path("enhanced_models/ensemble")
ENSEMBLE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

class EnsembleTradingML:
    """
    Advanced Ensemble ML System for Trading Decisions
    Replaces the existing ML system with high-performance ensemble models
    """
    
    def __init__(self, models_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else ENSEMBLE_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ensemble models
        self.ensemble_models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # Performance thresholds
        self.confidence_thresholds = {
            'buy': 0.65,      # High confidence for buying
            'sell': 0.60,     # Medium confidence for selling
            'strong_buy': 0.80,  # Very high confidence
            'strong_sell': 0.75  # High confidence for selling
        }
        
        # Risk management
        self.max_position_risk = 0.02  # 2% max risk per position
        self.profit_probability_threshold = 0.58  # 58% min profit probability
        
        logger.info(f"🎯 Ensemble Trading ML System initialized")
        logger.info(f"📁 Models directory: {self.models_dir}")
        
        # Try to load existing models
        self._load_existing_models()
    
    def build_optimized_ensemble_models(self) -> Dict[str, Any]:
        """Build optimized ensemble models based on performance comparison results"""
        
        # Best performing configurations from comparison
        ensemble_configs = {
            'primary_stacking': {
                'classifier': StackingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(
                            n_estimators=150, max_depth=12, min_samples_split=4,
                            min_samples_leaf=2, max_features='sqrt', random_state=42,
                            class_weight='balanced', n_jobs=-1
                        )),
                        ('gb', GradientBoostingClassifier(
                            n_estimators=120, max_depth=8, learning_rate=0.1,
                            min_samples_split=4, random_state=42
                        )),
                        ('xgb', xgb.XGBClassifier(
                            n_estimators=100, max_depth=8, learning_rate=0.1,
                            random_state=42, eval_metric='logloss'
                        ))
                    ],
                    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
                    cv=5,
                    stack_method='predict_proba'
                ),
                'regressor': StackingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(
                            n_estimators=150, max_depth=12, min_samples_split=4,
                            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
                        )),
                        ('gb', GradientBoostingRegressor(
                            n_estimators=120, max_depth=8, learning_rate=0.1,
                            min_samples_split=4, random_state=42
                        )),
                        ('xgb', xgb.XGBRegressor(
                            n_estimators=100, max_depth=8, learning_rate=0.1,
                            random_state=42
                        ))
                    ],
                    final_estimator=LinearRegression(),
                    cv=5
                )
            },
            'voting_ensemble': {
                'classifier': VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(
                            n_estimators=120, max_depth=10, random_state=42,
                            class_weight='balanced', n_jobs=-1
                        )),
                        ('gb', GradientBoostingClassifier(
                            n_estimators=100, max_depth=8, random_state=42
                        )),
                        ('xgb', xgb.XGBClassifier(
                            n_estimators=80, max_depth=8, random_state=42,
                            eval_metric='logloss'
                        ))
                    ],
                    voting='soft'
                ),
                'regressor': VotingRegressor(
                    estimators=[
                        ('rf', RandomForestRegressor(
                            n_estimators=120, max_depth=10, random_state=42, n_jobs=-1
                        )),
                        ('gb', GradientBoostingRegressor(
                            n_estimators=100, max_depth=8, random_state=42
                        )),
                        ('xgb', xgb.XGBRegressor(
                            n_estimators=80, max_depth=8, random_state=42
                        ))
                    ]
                )
            }
        }
        
        self.ensemble_models = ensemble_configs
        logger.info(f"✅ Built {len(ensemble_configs)} optimized ensemble configurations")
        return ensemble_configs
    
    def make_ensemble_trading_decision(self, symbol: str, granularity: int = 3600, 
                                     investment_amount: float = 100.0) -> Tuple[str, float]:
        """
        Primary ML decision function - replaces make_ml_decision and make_enhanced_ml_decision
        
        Returns:
            tuple: (decision, confidence) where decision is 'BUY', 'SELL', or 'HOLD'
        """
        try:
            # Skip stablecoins - they don't have meaningful price movements
            stablecoins = ['USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD', 'TUSD-USD', 'USDP-USD']
            if symbol in stablecoins:
                logger.info(f"⏭️ Skipping stablecoin {symbol} - no meaningful price movement for ML analysis")
                return 'HOLD', 0.0
            
            logger.info(f"🎯 Ensemble ML decision for {symbol} (granularity: {granularity}s)")
            
            # Get comprehensive prediction
            prediction = self.get_comprehensive_ensemble_prediction(symbol, granularity, investment_amount)
            
            if prediction is None:
                logger.warning(f"❌ Failed to get prediction for {symbol}")
                return 'HOLD', 0.0
            
            # Extract decision and confidence
            decision = prediction.get('action', 'HOLD')
            confidence = prediction.get('confidence', 0.0)
            
            # Log decision details
            logger.info(f"🤖 {symbol}: {decision} | Confidence: {confidence:.1%}")
            if 'profit_probability' in prediction:
                logger.info(f"💰 Profit Probability: {prediction['profit_probability']:.1%}")
            if 'expected_return' in prediction:
                logger.info(f"📈 Expected Return: {prediction['expected_return']:.2%}")
            
            # Apply risk management filters
            if decision == 'BUY' and confidence >= self.confidence_thresholds['buy']:
                profit_prob = prediction.get('profit_probability', 0.0)
                if profit_prob >= self.profit_probability_threshold:
                    logger.info(f"✅ {symbol}: BUY confirmed | High confidence & profit probability")
                    return decision, confidence
                else:
                    logger.info(f"⏭️ {symbol}: BUY rejected | Low profit probability {profit_prob:.1%}")
                    return 'HOLD', confidence * 0.5
            
            elif decision == 'SELL' and confidence >= self.confidence_thresholds['sell']:
                logger.info(f"📤 {symbol}: SELL confirmed | Sufficient confidence")
                return decision, confidence
            
            else:
                logger.info(f"⏸️ {symbol}: HOLD | Decision: {decision}, Confidence: {confidence:.1%}")
                return 'HOLD', confidence
            
        except Exception as e:
            logger.error(f"❌ Ensemble trading decision error for {symbol}: {str(e)}")
            return 'HOLD', 0.0
    
    def get_comprehensive_ensemble_prediction(self, symbol: str, granularity: int = 3600, 
                                            investment_amount: float = 100.0) -> Dict[str, Any]:
        """Get comprehensive prediction with multiple ensemble models"""
        
        try:
            # Import required functions
            from maybe import get_coinbase_data, calculate_indicators
            
            # Get data
            df = get_coinbase_data(symbol, granularity, days=60)
            if df is None or len(df) < 50:
                logger.warning(f"❌ Insufficient data for {symbol}")
                return None
            
            # Calculate indicators and features
            df = calculate_indicators(df)
            df = self._add_ensemble_features(df)
            df = df.dropna()
            
            if len(df) < 30:
                logger.warning(f"❌ Insufficient clean data for {symbol}")
                return None
            
            # Prepare features
            feature_data = self._prepare_features(df, symbol)
            if feature_data is None:
                return None
            
            # Get ensemble predictions
            ensemble_predictions = self._get_ensemble_predictions(feature_data, symbol)
            
            # Analyze and combine predictions
            final_prediction = self._analyze_ensemble_consensus(ensemble_predictions, df, symbol, investment_amount)
            
            return final_prediction
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive prediction for {symbol}: {str(e)}")
            return None
    
    def _add_ensemble_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced features optimized for ensemble models"""
        
        # Price momentum features
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # Technical indicator enhancements
        if 'rsi' in df.columns:
            df['rsi_momentum'] = df['rsi'].diff()
            df['rsi_divergence'] = (df['close'].pct_change(14) > 0).astype(int) - (df['rsi'].diff(14) > 0).astype(int)
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_momentum'] = df['macd'].diff()
            df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & 
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # Bollinger Band features
        if 'upper_band' in df.columns and 'lower_band' in df.columns:
            df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['close']
            df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)
        
        # Market structure features
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_strength'] = df['close'].rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        """Prepare features for ensemble predictions"""
        
        # Define feature columns
        base_features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'upper_band', 'lower_band',
            'volume', '%K', '%D', 'OBV', 'ATR'
        ]
        
        momentum_features = [f'momentum_{p}' for p in [3, 5, 10, 20]]
        volatility_features = [f'volatility_{p}' for p in [3, 5, 10, 20]]
        
        ensemble_features = [
            'volume_ratio', 'volume_momentum', 'rsi_momentum', 'rsi_divergence',
            'macd_momentum', 'macd_crossover', 'bb_width', 'bb_position', 'bb_squeeze',
            'higher_high', 'lower_low', 'trend_strength'
        ]
        
        all_features = base_features + momentum_features + volatility_features + ensemble_features
        
        # Filter available features
        available_features = [f for f in all_features if f in df.columns]
        
        if len(available_features) < 10:
            logger.warning(f"❌ Insufficient features for {symbol}: {len(available_features)}")
            return None
        
        # Get latest features
        latest_features = df[available_features].iloc[-1:].fillna(0)
        
        # Get training data for the last 200 periods
        train_data = df[available_features].iloc[-200:].fillna(0)
        
        return {
            'latest': latest_features.values,
            'training': train_data.values,
            'feature_names': available_features
        }
    
    def _get_ensemble_predictions(self, feature_data: Dict[str, np.ndarray], symbol: str) -> Dict[str, Any]:
        """Get predictions from all ensemble models"""
        
        predictions = {}
        
        # Load or train models
        model_types = ['primary_stacking', 'voting_ensemble']
        
        for model_type in model_types:
            try:
                # Try to load existing model
                model_path = self.models_dir / f"{symbol}_{model_type}_classifier.pkl"
                
                if model_path.exists():
                    # Load existing model
                    model_data = joblib.load(joblib.load(model_path)
                    classifier = model_data['model']
                    scaler = model_data.get('scaler')
                    
                    # Scale features if scaler exists
                    features = feature_data['latest']
                    if scaler is not None:
                        features = scaler.transform(features)
                    
                    # Get predictions
                    prediction = classifier.predict(features)[0]
                    probabilities = classifier.predict_proba(features)[0]
                    confidence = np.max(probabilities)
                    
                    predictions[model_type] = {
                        'prediction': prediction, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/confidence': confidence,
                        'probabilities': probabilities
                    }
                    
                    logger.debug(f"✅ {model_type}: prediction={prediction}, confidence={confidence:.3f}")
                
                else:
                    # Train new model if none exists
                    logger.info(f"🔄 Training new {model_type} model for {symbol}")
                    self._train_ensemble_model(symbol, model_type, feature_data)
                    
                    # Retry prediction after training
                    if model_path.exists():
                        model_data = joblib.load(joblib.load(model_path)
                        classifier = model_data['model']
                        scaler = model_data.get('scaler')
                        
                        features = feature_data['latest']
                        if scaler is not None:
                            features = scaler.transform(features)
                        
                        prediction = classifier.predict(features)[0]
                        probabilities = classifier.predict_proba(features)[0]
                        confidence = np.max(probabilities)
                        
                        predictions[model_type] = {
                            'prediction': prediction, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/confidence': confidence,
                            'probabilities': probabilities
                        }
                
            except Exception as e:
                logger.warning(f"⚠️ Error with {model_type} for {symbol}: {str(e)}")
                continue
        
        return predictions
    
    def _train_ensemble_model(self, symbol: str, model_type: str, feature_data: Dict[str, np.ndarray]):
        """Train ensemble model for specific symbol"""
        
        try:
            # Import required functions
            from maybe import get_coinbase_data, calculate_indicators
            
            # Get extended training data
            df = get_coinbase_data(symbol, 3600, days=90)
            if df is None or len(df) < 100:
                logger.warning(f"❌ Insufficient training data for {symbol}")
                return
            
            # Prepare training data
            df = calculate_indicators(df)
            df = self._add_ensemble_features(df)
            
            # Create targets
            df['future_return'] = df['close'].pct_change(-1)  # Next period return
            df['signal'] = (df['future_return'] > 0.001).astype(int)  # Buy signal if >0.1% gain
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"❌ Insufficient clean training data for {symbol}")
                return
            
            # Prepare features and targets
            X = df[feature_data['feature_names']].values
            y = df['signal'].values
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get model configuration
            if model_type not in self.ensemble_models:
                self.build_optimized_ensemble_models()
            
            model_config = self.ensemble_models[model_type]
            classifier = model_config['classifier']
            
            # Train model
            logger.info(f"🔄 Training {model_type} classifier for {symbol}...")
            classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ {model_type} trained for {symbol} | Accuracy: {accuracy:.3f}")
            
            # Save model
            model_data = {
                'model': classifier,
                'scaler': scaler,
                'accuracy': accuracy,
                'feature_names': feature_data['feature_names'],
                'training_date': datetime.now().isoformat(),
                'symbol': symbol,
                'model_type': model_type
            }
            
            model_path = self.models_dir / f"{symbol}_{model_type}_classifier.pkl"
            joblib.dump(model_data, model_path)
            
            logger.info(f"💾 Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_type} for {symbol}: {str(e)}")
    
    def _analyze_ensemble_consensus(self, predictions: Dict[str, Any], df: pd.DataFrame,
                                  symbol: str, investment_amount: float) -> Dict[str, Any]:
        """Analyze consensus across ensemble models"""
        
        if not predictions:
            return self._create_hold_decision("No model predictions available")
        
        # Collect votes and confidences
        buy_votes = 0
        sell_votes = 0
        confidences = []
        
        for model_type, pred_data in predictions.items():
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            
            if prediction == 1:  # Buy signal
                buy_votes += confidence  # Weight by confidence
            else:  # Sell/Hold signal
                sell_votes += confidence
                
            confidences.append(confidence)
        
        # Calculate consensus
        total_weight = buy_votes + sell_votes
        if total_weight == 0:
            return self._create_hold_decision("No weighted consensus")
        
        buy_consensus = buy_votes / total_weight
        avg_confidence = np.mean(confidences)
        
        # Determine action
        if buy_consensus >= 0.6 and avg_confidence >= self.confidence_thresholds['buy']:
            action = 'BUY'
            final_confidence = min(buy_consensus * avg_confidence, 0.95)
        elif buy_consensus <= 0.4 and avg_confidence >= self.confidence_thresholds['sell']:
            action = 'SELL'
            final_confidence = min((1 - buy_consensus) * avg_confidence, 0.95)
        else:
            action = 'HOLD'
            final_confidence = avg_confidence * 0.5
        
        # Calculate profit probability and expected return
        current_price = df['close'].iloc[-1]
        profit_probability = self._calculate_profit_probability(df, action, final_confidence)
        expected_return = self._calculate_expected_return(df, action, final_confidence)
        
        result = {
            'action': action,
            'confidence': final_confidence,
            'profit_probability': profit_probability,
            'expected_return': expected_return,
            'buy_consensus': buy_consensus,
            'model_count': len(predictions),
            'current_price': current_price,
            'investment_amount': investment_amount,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }
        
        logger.info(f"🎯 Ensemble consensus for {symbol}: {action} ({final_confidence:.1%} confidence)")
        logger.info(f"💰 Profit probability: {profit_probability:.1%}, Expected return: {expected_return:.2%}")
        
        return result
    
    def _calculate_profit_probability(self, df: pd.DataFrame, action: str, confidence: float) -> float:
        """Calculate probability of profit based on historical performance"""
        
        if action == 'HOLD':
            return 0.5
        
        # Look at recent performance for similar signals
        df_recent = df.tail(100).copy()
        df_recent['future_return'] = df_recent['close'].pct_change(-1)
        
        if action == 'BUY':
            # Historical success rate for buy signals
            positive_returns = (df_recent['future_return'] > 0.001).sum()
            total_periods = len(df_recent.dropna())
            base_probability = positive_returns / total_periods if total_periods > 0 else 0.5
        else:  # SELL
            # Historical success rate for avoiding losses
            negative_returns = (df_recent['future_return'] < -0.001).sum()
            total_periods = len(df_recent.dropna())
            base_probability = negative_returns / total_periods if total_periods > 0 else 0.5
        
        # Adjust by confidence
        adjusted_probability = base_probability + (confidence - 0.5) * 0.3
        return np.clip(adjusted_probability, 0.1, 0.9)
    
    def _calculate_expected_return(self, df: pd.DataFrame, action: str, confidence: float) -> float:
        """Calculate expected return based on historical data"""
        
        if action == 'HOLD':
            return 0.0
        
        # Look at recent volatility and trends
        df_recent = df.tail(50).copy()
        avg_volatility = df_recent['close'].pct_change().std()
        
        if action == 'BUY':
            # Expected positive return based on confidence and volatility
            base_return = avg_volatility * 2.0  # Target 2x the volatility
            expected_return = base_return * confidence
        else:  # SELL
            # Expected return from avoiding losses
            expected_return = avg_volatility * confidence * 0.5
        
        return np.clip(expected_return, -0.05, 0.05)  # Limit to ±5%
    
    def _create_hold_decision(self, reason: str) -> Dict[str, Any]:
        """Create a HOLD decision with reason"""
        return {
            'action': 'HOLD',
            'confidence': 0.2,
            'profit_probability': 0.5,
            'expected_return': 0.0,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def _load_existing_models(self):
        """Load any existing trained models"""
        try:
            model_files = list(self.models_dir.glob("*_classifier.pkl"))
            loaded_count = 0
            
            for model_file in model_files:
                try:
                    model_data = joblib.load(joblib.load(model_file)
                    # Store metadata
                    symbol = model_data.get('symbol', 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/unknown')
                    model_type = model_data.get('model_type', 'unknown')
                    
                    if symbol not in self.model_metadata:
                        self.model_metadata[symbol] = {}
                    
                    self.model_metadata[symbol][model_type] = {
                        'accuracy': model_data.get('accuracy', 0.0),
                        'training_date': model_data.get('training_date'),
                        'path': str(model_file)
                    }
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not load model {model_file}: {str(e)}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"✅ Loaded metadata for {loaded_count} existing models")
            else:
                logger.info("📝 No existing models found - will train on first use")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading existing models: {str(e)}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ensemble models"""
        return {
            'models_directory': str(self.models_dir),
            'available_models': self.model_metadata,
            'confidence_thresholds': self.confidence_thresholds,
            'total_models': sum(len(models) for models in self.model_metadata.values()),
            'system_status': 'ready'
        }


# Global ensemble system instance
ensemble_ml_system = EnsembleTradingML()

def make_ensemble_ml_decision(symbol: str, granularity: int = 3600) -> Tuple[str, float]:
    """
    Replacement function for make_ml_decision and make_enhanced_ml_decision
    
    This function maintains the same interface as the original functions
    but uses the new ensemble ML system underneath.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        granularity: Time granularity in seconds (default: 3600 = 1 hour)
    
    Returns:
        tuple: (decision, confidence) where decision is 'BUY', 'SELL', or 'HOLD'
    """
    global ensemble_ml_system
    
    try:
        # Use the ensemble system
        decision, confidence = ensemble_ml_system.make_ensemble_trading_decision(
            symbol=symbol,
            granularity=granularity,
            investment_amount=100.0
        )
        
        logger.info(f"🎯 Ensemble ML Decision for {symbol}: {decision} ({confidence:.1%})")
        return decision, confidence
        
    except Exception as e:
        logger.error(f"❌ Ensemble ML decision error for {symbol}: {str(e)}")
        return 'HOLD', 0.0

def make_enhanced_ensemble_ml_decision(symbol: str, granularity: int = 3600, 
                                     investment_amount: float = 100.0) -> Dict[str, Any]:
    """
    Enhanced replacement function that returns detailed prediction information
    
    Returns:
        dict: Comprehensive prediction data including profit probability, expected returns, etc.
    """
    global ensemble_ml_system
    
    try:
        # Skip stablecoins - they don't have meaningful price movements
        stablecoins = ['USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD', 'TUSD-USD', 'USDP-USD']
        if symbol in stablecoins:
            logger.info(f"⏭️ Skipping stablecoin {symbol} - no meaningful price movement for ML analysis")
            return {
                'action': 'HOLD',
                'confidence': 0.2,
                'profit_probability': 0.5,
                'expected_return': 0.0,
                'reason': f'Stablecoin {symbol} skipped - no volatility for ML',
                'timestamp': datetime.now().isoformat()
            }
        
        # Get comprehensive prediction
        prediction = ensemble_ml_system.get_comprehensive_ensemble_prediction(
            symbol=symbol,
            granularity=granularity,
            investment_amount=investment_amount
        )
        
        if prediction is None:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'profit_probability': 0.5,
                'expected_return': 0.0,
                'reason': 'No prediction available',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"🎯 Enhanced Ensemble ML Decision for {symbol}: {prediction['action']} "
                   f"({prediction['confidence']:.1%} confidence, "
                   f"{prediction['profit_probability']:.1%} profit probability)")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Enhanced ensemble ML decision error for {symbol}: {str(e)}")
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'profit_probability': 0.5,
            'expected_return': 0.0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def train_ensemble_models_for_symbols(symbols: List[str] = None) -> Dict[str, Any]:
    """Train ensemble models for specified symbols"""
    
    if symbols is None:
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"]
    
    global ensemble_ml_system
    
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"🔄 Training ensemble models for {symbol}...")
            
            # Build models if not already built
            if not ensemble_ml_system.ensemble_models:
                ensemble_ml_system.build_optimized_ensemble_models()
            
            # Trigger training by making a prediction
            decision, confidence = make_ensemble_ml_decision(symbol)
            
            results[symbol] = {
                'status': 'completed',
                'decision': decision,
                'confidence': confidence
            }
            
            logger.info(f"✅ Training completed for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Training failed for {symbol}: {str(e)}")
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

def get_ensemble_system_status() -> Dict[str, Any]:
    """Get status of the ensemble ML system"""
    global ensemble_ml_system
    return ensemble_ml_system.get_model_status()

# Initialize the ensemble system
logger.info("🚀 Ensemble ML Integration System initialized and ready") 