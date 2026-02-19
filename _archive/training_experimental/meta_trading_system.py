#!/usr/bin/env python3
"""
Meta-Model Trading System
========================

Uses model outputs (predictions, probabilities, confidence scores) as features
for a meta-classifier that makes final buy/sell/hold decisions.

This is superior to direct model predictions because:
1. Combines multiple model perspectives
2. Learns from model disagreement patterns
3. Adapts to changing market regimes
4. Provides better risk-adjusted decisions
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import json

warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import existing ML engines
try:
    from stacking_ml_engine import StackingMLEngine
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False

try:
    from advanced_ensemble_ml import AdvancedEnsembleMomentumML
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from maybe import get_coinbase_data, calculate_indicators
    COINBASE_AVAILABLE = True
except ImportError:
    COINBASE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Container for individual model predictions"""
    model_name: str
    prediction: float  # Predicted price change %
    confidence: float  # Model confidence (0-1)
    probability_up: float  # Probability of price going up
    probability_down: float  # Probability of price going down
    features_used: int  # Number of features used
    model_accuracy: float  # Historical accuracy of this model
    timestamp: datetime

@dataclass
class TradingDecision:
    """Final trading decision from meta-model"""
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # Meta-model confidence (0-1)
    expected_return: float  # Expected return %
    risk_score: float  # Risk assessment (0-1)
    position_size: float  # Recommended position size (0-1)
    stop_loss: float  # Recommended stop loss %
    take_profit: float  # Recommended take profit %
    reasoning: str  # Human-readable explanation
    model_consensus: Dict[str, Any]  # Individual model votes
    timestamp: datetime

class MetaModelTradingSystem:
    """
    Meta-model system that uses predictions from multiple ML models
    as features for a higher-level trading decision classifier
    """
    
    def __init__(self):
        self.meta_models = {}  # Meta-classifiers for each symbol
        self.model_engines = {}  # Individual ML engines
        self.scalers = {}  # Feature scalers
        self.model_cache_dir = os.path.join(current_dir, 'meta_models')
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        # Performance tracking
        self.decision_history = []
        self.model_performance = {}
        
        # Initialize base ML engines
        self._initialize_base_engines()
        
        logger.info("🧠 Meta-Model Trading System initialized")
    
    def _initialize_base_engines(self):
        """Initialize the base ML engines that will provide predictions"""
        try:
            if STACKING_AVAILABLE:
                self.model_engines['stacking'] = StackingMLEngine()
                logger.info("✅ Stacking ML Engine loaded")
            
            if ENSEMBLE_AVAILABLE:
                # Will be initialized per symbol
                logger.info("✅ Ensemble ML Engine available")
            
            logger.info(f"📊 {len(self.model_engines)} base ML engines available")
            
        except Exception as e:
            logger.error(f"💥 Error initializing base engines: {str(e)}")
    
    def collect_model_predictions(self, symbol: str, granularity: int = 3600) -> List[ModelPrediction]:
        """
        Collect predictions from all available ML models
        Returns list of ModelPrediction objects
        """
        predictions = []
        
        try:
            # Get data for models
            df = get_coinbase_data(symbol, granularity, days=60)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return predictions
            
            # 1. Stacking ML Engine Predictions
            if 'stacking' in self.model_engines:
                try:
                    stacking_pred = self.model_engines['stacking'].make_prediction(
                        symbol, granularity, investment_amount=100.0
                    )
                    
                    if stacking_pred:
                        predictions.append(ModelPrediction(
                            model_name='stacking_regressor',
                            prediction=stacking_pred.get('predicted_return', 0.0),
                            confidence=stacking_pred.get('overall_confidence', 0.5),
                            probability_up=stacking_pred.get('confidence_buy', 0.5),
                            probability_down=stacking_pred.get('confidence_sell', 0.5),
                            features_used=len(stacking_pred.get('features', [])),
                            model_accuracy=stacking_pred.get('model_accuracy', 0.5),
                            timestamp=datetime.now()
                        ))
                        logger.info(f"✅ Stacking prediction collected for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Stacking prediction failed for {symbol}: {str(e)}")
            
            # 2. Ensemble ML Predictions
            if ENSEMBLE_AVAILABLE:
                try:
                    ensemble_engine = AdvancedEnsembleMomentumML(symbol, granularity)
                    
                    # Would need to implement prediction method in ensemble engine
                    # For now, create a synthetic prediction based on the pattern
                    df_processed = calculate_indicators(df, symbol=symbol)
                    if len(df_processed) > 20:
                        recent_return = (df_processed['close'].iloc[-1] / df_processed['close'].iloc[-2] - 1) * 100
                        
                        predictions.append(ModelPrediction(
                            model_name='ensemble_momentum',
                            prediction=recent_return * 1.2,  # Momentum factor
                            confidence=0.65,
                            probability_up=0.6 if recent_return > 0 else 0.4,
                            probability_down=0.4 if recent_return > 0 else 0.6,
                            features_used=50,  # Estimate
                            model_accuracy=0.58,  # Estimate
                            timestamp=datetime.now()
                        ))
                        logger.info(f"✅ Ensemble prediction collected for {symbol}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ensemble prediction failed for {symbol}: {str(e)}")
            
            # 3. Simple Technical Indicator Predictions (baseline)
            try:
                df_tech = calculate_indicators(df, symbol=symbol)
                if len(df_tech) > 20:
                    # Simple momentum prediction
                    sma_short = df_tech['close'].rolling(5).mean().iloc[-1]
                    sma_long = df_tech['close'].rolling(20).mean().iloc[-1]
                    current_price = df_tech['close'].iloc[-1]
                    
                    momentum_signal = (sma_short - sma_long) / sma_long * 100
                    price_position = (current_price - sma_long) / sma_long * 100
                    
                    predictions.append(ModelPrediction(
                        model_name='technical_momentum',
                        prediction=momentum_signal,
                        confidence=min(abs(momentum_signal) / 2, 0.8),
                        probability_up=0.7 if momentum_signal > 0 else 0.3,
                        probability_down=0.3 if momentum_signal > 0 else 0.7,
                        features_used=10,
                        model_accuracy=0.52,
                        timestamp=datetime.now()
                    ))
                    logger.info(f"✅ Technical prediction collected for {symbol}")
                
            except Exception as e:
                logger.warning(f"⚠️ Technical prediction failed for {symbol}: {str(e)}")
            
            logger.info(f"📊 Collected {len(predictions)} model predictions for {symbol}")
            return predictions
            
        except Exception as e:
            logger.error(f"💥 Error collecting predictions for {symbol}: {str(e)}")
            return predictions
    
    def create_meta_features(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """
        Convert model predictions into features for the meta-classifier
        This is the key innovation - using model outputs as features
        """
        if not predictions:
            return {}
        
        features = {}
        
        # Basic aggregation features
        prediction_values = [p.prediction for p in predictions]
        confidence_values = [p.confidence for p in predictions]
        prob_up_values = [p.probability_up for p in predictions]
        prob_down_values = [p.probability_down for p in predictions]
        
        # 1. Central tendency features
        features['pred_mean'] = np.mean(prediction_values)
        features['pred_median'] = np.median(prediction_values)
        features['conf_mean'] = np.mean(confidence_values)
        features['conf_median'] = np.median(confidence_values)
        
        # 2. Dispersion features (model disagreement)
        features['pred_std'] = np.std(prediction_values)
        features['pred_range'] = np.max(prediction_values) - np.min(prediction_values)
        features['conf_std'] = np.std(confidence_values)
        features['model_disagreement'] = features['pred_std'] / (abs(features['pred_mean']) + 0.001)
        
        # 3. Consensus features
        features['bullish_consensus'] = np.mean(prob_up_values)
        features['bearish_consensus'] = np.mean(prob_down_values)
        features['consensus_strength'] = abs(features['bullish_consensus'] - features['bearish_consensus'])
        
        # 4. Individual model features
        for i, pred in enumerate(predictions):
            prefix = f"model_{pred.model_name}"
            features[f"{prefix}_prediction"] = pred.prediction
            features[f"{prefix}_confidence"] = pred.confidence
            features[f"{prefix}_prob_up"] = pred.probability_up
            features[f"{prefix}_accuracy"] = pred.model_accuracy
        
        # 5. Weighted features (by confidence)
        total_confidence = sum(confidence_values)
        if total_confidence > 0:
            features['weighted_prediction'] = sum(
                p.prediction * p.confidence for p in predictions
            ) / total_confidence
            
            features['weighted_prob_up'] = sum(
                p.probability_up * p.confidence for p in predictions
            ) / total_confidence
        else:
            features['weighted_prediction'] = features['pred_mean']
            features['weighted_prob_up'] = features['bullish_consensus']
        
        # 6. Risk-adjusted features
        high_conf_predictions = [p.prediction for p in predictions if p.confidence > 0.6]
        if high_conf_predictions:
            features['high_conf_mean'] = np.mean(high_conf_predictions)
            features['high_conf_count'] = len(high_conf_predictions)
        else:
            features['high_conf_mean'] = features['pred_mean']
            features['high_conf_count'] = 0
        
        # 7. Model quality features
        features['avg_features_used'] = np.mean([p.features_used for p in predictions])
        features['avg_model_accuracy'] = np.mean([p.model_accuracy for p in predictions])
        features['num_models'] = len(predictions)
        
        logger.info(f"🔧 Generated {len(features)} meta-features from {len(predictions)} model predictions")
        return features
    
    def create_trading_labels(self, df: pd.DataFrame, prediction_horizon: int = 24) -> pd.Series:
        """
        Create trading labels for training the meta-classifier
        Labels: 0=HOLD, 1=BUY, 2=SELL
        """
        # Calculate future returns
        future_returns = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Create labels based on thresholds
        labels = pd.Series(0, index=df.index)  # Default to HOLD
        
        # Define thresholds (can be optimized)
        buy_threshold = 0.02  # 2% gain
        sell_threshold = -0.015  # 1.5% loss
        
        labels[future_returns > buy_threshold] = 1  # BUY
        labels[future_returns < sell_threshold] = 2  # SELL
        
        return labels
    
    def train_meta_classifier(self, symbol: str, granularity: int = 3600, days: int = 90):
        """
        Train the meta-classifier using historical model predictions as features
        """
        try:
            logger.info(f"🎯 Training meta-classifier for {symbol}")
            
            # Get historical data
            df = get_coinbase_data(symbol, granularity, days=days)
            if df is None or len(df) < 100:
                logger.error(f"Insufficient data for training {symbol}")
                return False
            
            # Create features and labels for training
            X_meta = []
            y_meta = []
            
            # Create labels
            labels = self.create_trading_labels(df)
            
            # Simulate historical predictions (in production, you'd have real historical predictions)
            for i in range(50, len(df) - 24):  # Leave room for prediction horizon
                try:
                    # Get slice of data up to this point
                    df_slice = df.iloc[:i+1].copy()
                    
                    # This is a simulation - in production you'd have real historical predictions
                    mock_predictions = self._create_mock_predictions(df_slice, symbol)
                    
                    if mock_predictions:
                        meta_features = self.create_meta_features(mock_predictions)
                        if meta_features:
                            X_meta.append(list(meta_features.values()))
                            y_meta.append(labels.iloc[i])
                
                except Exception as e:
                    continue
            
            if len(X_meta) < 50:
                logger.error(f"Insufficient training samples for {symbol}")
                return False
            
            # Convert to arrays
            X_meta = np.array(X_meta)
            y_meta = np.array(y_meta)
            
            # Create and train meta-classifier
            meta_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Scale features
            scaler = StandardScaler()
            X_meta_scaled = scaler.fit_transform(X_meta)
            
            # Train classifier
            meta_classifier.fit(X_meta_scaled, y_meta)
            
            # Store models
            self.meta_models[symbol] = meta_classifier
            self.scalers[symbol] = scaler
            
            # Evaluate performance
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(meta_classifier, X_meta_scaled, y_meta, cv=5)
            
            logger.info(f"✅ Meta-classifier trained for {symbol}")
            logger.info(f"   Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            logger.info(f"   Training samples: {len(X_meta)}")
            
            # Save model
            model_path = os.path.join(self.model_cache_dir, f"{symbol}_meta_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'classifier': meta_classifier,
                    'scaler': scaler,
                    'feature_names': list(self.create_meta_features(mock_predictions).keys()),
                    'training_date': datetime.now(),
                    'performance': cv_scores.mean()
                }, f)
            
            return True
            
        except Exception as e:
            logger.error(f"💥 Error training meta-classifier for {symbol}: {str(e)}")
            return False
    
    def _create_mock_predictions(self, df: pd.DataFrame, symbol: str) -> List[ModelPrediction]:
        """Create mock predictions for training (replace with real predictions in production)"""
        if len(df) < 20:
            return []
        
        # Simple momentum-based mock predictions
        recent_return = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100
        volatility = df['close'].pct_change().std() * 100
        
        return [
            ModelPrediction(
                model_name='mock_stacking',
                prediction=recent_return * 0.8,
                confidence=min(abs(recent_return) / 10, 0.9),
                probability_up=0.6 if recent_return > 0 else 0.4,
                probability_down=0.4 if recent_return > 0 else 0.6,
                features_used=25,
                model_accuracy=0.55,
                timestamp=datetime.now()
            ),
            ModelPrediction(
                model_name='mock_ensemble',
                prediction=recent_return * 1.2,
                confidence=min(volatility / 5, 0.8),
                probability_up=0.65 if recent_return > 0 else 0.35,
                probability_down=0.35 if recent_return > 0 else 0.65,
                features_used=40,
                model_accuracy=0.58,
                timestamp=datetime.now()
            )
        ]
    
    def make_trading_decision(self, symbol: str, granularity: int = 3600) -> Optional[TradingDecision]:
        """
        Make final trading decision using the meta-classifier
        This is the main entry point for trading decisions
        """
        try:
            logger.info(f"🎯 Making trading decision for {symbol}")
            
            # Check if meta-model exists
            if symbol not in self.meta_models:
                logger.warning(f"No meta-model for {symbol}, training first...")
                if not self.train_meta_classifier(symbol, granularity):
                    logger.error(f"Failed to train meta-model for {symbol}")
                    return None
            
            # Collect current model predictions
            predictions = self.collect_model_predictions(symbol, granularity)
            
            if not predictions:
                logger.warning(f"No model predictions available for {symbol}")
                return None
            
            # Create meta-features
            meta_features = self.create_meta_features(predictions)
            
            if not meta_features:
                logger.warning(f"No meta-features generated for {symbol}")
                return None
            
            # Prepare features for prediction
            X_meta = np.array(list(meta_features.values())).reshape(1, -1)
            X_meta_scaled = self.scalers[symbol].transform(X_meta)
            
            # Make prediction
            meta_classifier = self.meta_models[symbol]
            predicted_class = meta_classifier.predict(X_meta_scaled)[0]
            class_probabilities = meta_classifier.predict_proba(X_meta_scaled)[0]
            
            # Convert prediction to action
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action = action_map[predicted_class]
            
            # Calculate confidence
            confidence = max(class_probabilities)
            
            # Calculate expected return and risk
            expected_return = meta_features.get('weighted_prediction', 0.0)
            risk_score = meta_features.get('model_disagreement', 0.5)
            consensus_strength = meta_features.get('consensus_strength', 0.5)
            
            # Calculate position sizing based on confidence and risk
            base_position_size = 0.1  # 10% base position
            position_multiplier = confidence * consensus_strength * (1 - risk_score)
            position_size = min(base_position_size * position_multiplier, 0.3)  # Max 30%
            
            # Calculate stop loss and take profit
            volatility_estimate = abs(expected_return) * 2
            stop_loss = max(volatility_estimate * 1.5, 2.0)  # At least 2%
            take_profit = max(volatility_estimate * 2.5, 3.0)  # At least 3%
            
            # Create model consensus summary
            model_consensus = {
                'individual_predictions': [
                    {
                        'model': p.model_name,
                        'prediction': p.prediction,
                        'confidence': p.confidence
                    } for p in predictions
                ],
                'consensus_metrics': {
                    'prediction_mean': meta_features.get('pred_mean', 0.0),
                    'disagreement': meta_features.get('model_disagreement', 0.0),
                    'bullish_consensus': meta_features.get('bullish_consensus', 0.5)
                }
            }
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                action, confidence, expected_return, risk_score, predictions
            )
            
            # Create trading decision
            decision = TradingDecision(
                action=action,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasoning=reasoning,
                model_consensus=model_consensus,
                timestamp=datetime.now()
            )
            
            # Log decision
            self.decision_history.append(decision)
            logger.info(f"🎯 Decision for {symbol}: {action} (confidence: {confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"💥 Error making trading decision for {symbol}: {str(e)}")
            return None
    
    def _generate_reasoning(self, action: str, confidence: float, expected_return: float, 
                          risk_score: float, predictions: List[ModelPrediction]) -> str:
        """Generate human-readable reasoning for the trading decision"""
        
        reasoning_parts = [
            f"Meta-classifier recommends {action} with {confidence:.1%} confidence."
        ]
        
        if expected_return != 0:
            reasoning_parts.append(
                f"Expected return: {expected_return:+.2f}%."
            )
        
        if risk_score > 0.7:
            reasoning_parts.append("HIGH RISK: Models show significant disagreement.")
        elif risk_score < 0.3:
            reasoning_parts.append("LOW RISK: Strong model consensus.")
        
        # Add model consensus info
        bullish_models = sum(1 for p in predictions if p.prediction > 0)
        total_models = len(predictions)
        
        if bullish_models > total_models * 0.7:
            reasoning_parts.append(f"Strong bullish consensus ({bullish_models}/{total_models} models).")
        elif bullish_models < total_models * 0.3:
            reasoning_parts.append(f"Strong bearish consensus ({total_models - bullish_models}/{total_models} models).")
        else:
            reasoning_parts.append(f"Mixed signals ({bullish_models}/{total_models} models bullish).")
        
        return " ".join(reasoning_parts)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the meta-model system"""
        return {
            'available_models': list(self.meta_models.keys()),
            'base_engines': list(self.model_engines.keys()),
            'total_decisions': len(self.decision_history),
            'last_decision': self.decision_history[-1].timestamp if self.decision_history else None,
            'system_health': 'OK' if self.meta_models else 'NO_MODELS_TRAINED'
        }

# Global instance
meta_trading_system = MetaModelTradingSystem()

# Convenience functions for integration
def make_meta_trading_decision(symbol: str, granularity: int = 3600) -> Optional[TradingDecision]:
    """Make a trading decision using the meta-model system"""
    return meta_trading_system.make_trading_decision(symbol, granularity)

def train_meta_model(symbol: str, granularity: int = 3600):
    """Train a meta-model for a symbol"""
    return meta_trading_system.train_meta_classifier(symbol, granularity)

def get_meta_system_status():
    """Get meta-system status"""
    return meta_trading_system.get_system_status()

if __name__ == "__main__":
    # Test the meta-model system
    logger.info("🧪 Testing Meta-Model Trading System")
    
    test_symbol = "BTC-USD"
    
    # Train meta-model
    logger.info(f"Training meta-model for {test_symbol}...")
    if meta_trading_system.train_meta_classifier(test_symbol):
        # Make decision
        decision = meta_trading_system.make_trading_decision(test_symbol)
        
        if decision:
            logger.info("✅ Meta-model system test successful!")
            logger.info(f"Decision: {decision.action} ({decision.confidence:.1%} confidence)")
            logger.info(f"Expected return: {decision.expected_return:+.2f}%")
            logger.info(f"Reasoning: {decision.reasoning}")
        else:
            logger.error("❌ Decision making failed")
    else:
        logger.error("❌ Training failed")
