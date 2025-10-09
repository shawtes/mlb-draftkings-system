#!/usr/bin/env python3
"""
Advanced Ensemble ML for Momentum Trading
==========================================

Uses sophisticated stacking and voting ensembles for better predictions
Based on Stefan Jansen's ML for Algorithmic Trading principles
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import (
    StackingRegressor, VotingRegressor, RandomForestRegressor, 
    BaggingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)

class AdvancedEnsembleMomentumML:
    """
    Advanced ensemble ML system for momentum trading
    Combines multiple algorithms using stacking and voting
    """
    
    def __init__(self, symbol, granularity=3600):
        self.symbol = symbol
        self.granularity = granularity
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # Model performance tracking
        self.model_scores = {}
        self.training_history = []
        
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('lasso', Lasso(alpha=0.1)),
            ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
            ('bagging', BaggingRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=5), 
                n_estimators=10, 
                random_state=42
            ))
        ]
        
        # Add SVR with scaled features
        base_models.append(('svr', SVR(kernel='rbf', C=1.0, gamma='scale')))
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', XGBRegressor(
                objective='reg:squarederror',
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42
            )))
        
        return base_models
    
    def create_ensemble_models(self):
        """Create advanced ensemble models"""
        base_models = self.create_base_models()
        
        # Meta model for stacking
        if XGBOOST_AVAILABLE:
            meta_model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=30,
                max_depth=4,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42
            )
        else:
            meta_model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42)
        
        # Stacking Regressor
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=3  # Use 3-fold CV for stacking
        )
        
        # Voting Regressor (simpler ensemble)
        voting_models = [model for model in base_models if model[0] != 'svr']  # SVR needs scaling
        voting_model = VotingRegressor(estimators=voting_models)
        
        # Final ensemble of ensembles
        ensemble_models = [
            ('stacking', stacking_model),
            ('voting', voting_model)
        ]
        
        # Final meta-ensemble
        if XGBOOST_AVAILABLE:
            final_meta = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=20,
                max_depth=3,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42
            )
        else:
            final_meta = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        
        final_model = StackingRegressor(
            estimators=ensemble_models,
            final_estimator=final_meta,
            cv=3
        )
        
        return {
            'stacking': stacking_model,
            'voting': voting_model,
            'final_ensemble': final_model
        }
    
    def engineer_momentum_features(self, df):
        """Enhanced momentum feature engineering"""
        # Price momentum features
        df['momentum_1h'] = df['close'].pct_change(1)
        df['momentum_4h'] = df['close'].pct_change(4)
        df['momentum_12h'] = df['close'].pct_change(12)
        df['momentum_24h'] = df['close'].pct_change(24)
        df['momentum_1w'] = df['close'].pct_change(168)  # 1 week
        
        # Acceleration (2nd derivative)
        df['price_acceleration'] = df['momentum_1h'].diff()
        df['momentum_acceleration'] = df['momentum_4h'].diff()
        
        # Volume momentum
        df['volume_momentum'] = df['volume'].pct_change(4)
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        
        # Volatility features
        df['volatility_4h'] = df['close'].rolling(4).std() / df['close'].rolling(4).mean()
        df['volatility_24h'] = df['close'].rolling(24).std() / df['close'].rolling(24).mean()
        
        # Trend strength
        df['trend_strength_short'] = (df['close'] - df['close'].rolling(12).mean()) / df['close'].rolling(12).std()
        df['trend_strength_medium'] = (df['close'] - df['close'].rolling(24).mean()) / df['close'].rolling(24).std()
        df['trend_strength_long'] = (df['close'] - df['close'].rolling(48).mean()) / df['close'].rolling(48).std()
        
        # Support/Resistance momentum
        df['resistance_break'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(int)
        df['support_break'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(int)
        
        # RSI momentum
        df['rsi'] = self.calculate_rsi(df['close'])
        df['rsi_momentum'] = df['rsi'].diff()
        
        # MACD features
        df = self.add_macd_features(df)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def add_macd_features(self, df):
        """Add MACD features"""
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def create_targets(self, df, horizons=[1, 4, 12]):
        """Create multiple prediction targets"""
        targets = {}
        
        for horizon in horizons:
            # Forward returns
            targets[f'return_{horizon}h'] = df['close'].shift(-horizon) / df['close'] - 1
            
            # Direction (up/down)
            targets[f'direction_{horizon}h'] = (targets[f'return_{horizon}h'] > 0).astype(int)
            
            # Magnitude (absolute return)
            targets[f'magnitude_{horizon}h'] = targets[f'return_{horizon}h'].abs()
        
        return targets
    
    def train_ensemble_models(self, df):
        """Train the ensemble models"""
        logger.info(f"🎯 Training advanced ensemble models for {self.symbol}")
        
        # Feature engineering
        df = self.engineer_momentum_features(df)
        
        # Select features
        feature_cols = [
            'momentum_1h', 'momentum_4h', 'momentum_12h', 'momentum_24h', 'momentum_1w',
            'price_acceleration', 'momentum_acceleration',
            'volume_momentum', 'volume_price_correlation',
            'volatility_4h', 'volatility_24h',
            'trend_strength_short', 'trend_strength_medium', 'trend_strength_long',
            'resistance_break', 'support_break',
            'rsi', 'rsi_momentum',
            'macd', 'macd_signal', 'macd_histogram'
        ]
        
        # Create targets BEFORE cleaning
        targets = self.create_targets(df)
        
        # Add targets to dataframe
        for target_name, target_values in targets.items():
            df[target_name] = target_values
        
        # Only include return targets for training
        return_targets = [name for name in targets.keys() if name.startswith('return_')]
        
        # Clean data - only require features and return targets
        required_cols = feature_cols + return_targets
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 100:
            logger.error(f"Insufficient data for {self.symbol}: {len(df_clean)} rows")
            return False
        
        logger.info(f"📊 Training on {len(df_clean)} samples with {len(feature_cols)} features")
        
        # Prepare data
        X = df_clean[feature_cols]
        self.feature_names = feature_cols
        
        # Scale features for SVR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['features'] = scaler
        
        # Train models for each return target
        ensemble_models = self.create_ensemble_models()
        
        for target_name in return_targets:
            y = df_clean[target_name]
            
            logger.info(f"🚀 Training ensemble for {target_name}")
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            trained_models = {}
            scores = {}
            
            for model_name, model in ensemble_models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_scaled, y, 
                        cv=tscv, 
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    
                    # Train final model
                    model.fit(X_scaled, y)
                    
                    # Store model and score
                    trained_models[model_name] = model
                    scores[model_name] = -cv_scores.mean()
                    
                    logger.info(f"   ✅ {model_name}: MSE = {scores[model_name]:.6f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ {model_name} failed: {e}")
                    continue
            
            # Store best models
            if trained_models:
                self.models[target_name] = trained_models
                self.model_scores[target_name] = scores
                
                # Find best model
                best_model = min(scores, key=scores.get)
                logger.info(f"   🏆 Best model for {target_name}: {best_model}")
        
        return len(self.models) > 0
    
    def predict_ensemble(self, df):
        """Generate ensemble predictions"""
        if not self.models:
            logger.error("No trained models available")
            return None
        
        # Feature engineering
        df = self.engineer_momentum_features(df)
        
        # Prepare features
        X = df[self.feature_names].iloc[-1:].fillna(0)
        X_scaled = self.scalers['features'].transform(X)
        
        predictions = {}
        
        for target_name, models in self.models.items():
            target_predictions = {}
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    target_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    continue
            
            if target_predictions:
                # Use the best model's prediction (or ensemble average)
                best_model = min(self.model_scores[target_name], key=self.model_scores[target_name].get)
                
                predictions[target_name] = {
                    'best_prediction': target_predictions.get(best_model, 0),
                    'ensemble_average': np.mean(list(target_predictions.values())),
                    'all_predictions': target_predictions,
                    'best_model': best_model
                }
        
        return predictions
    
    def get_trading_signal(self, predictions):
        """Convert predictions to trading signals"""
        if not predictions:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Focus on 4-hour predictions (good balance of accuracy and utility)
        target_key = 'return_4h'
        
        if target_key not in predictions:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        pred_data = predictions[target_key]
        predicted_return = pred_data['best_prediction']
        ensemble_avg = pred_data['ensemble_average']
        
        # Calculate confidence based on model agreement
        all_preds = list(pred_data['all_predictions'].values())
        pred_std = np.std(all_preds)
        confidence = max(0.1, 1.0 - (pred_std / max(abs(predicted_return), 0.01)))
        
        # Determine action
        threshold = 0.005  # 0.5% threshold for action
        
        if predicted_return > threshold:
            action = 'BUY'
        elif predicted_return < -threshold:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Boost confidence if ensemble agrees
        if abs(predicted_return - ensemble_avg) < 0.002:  # Models agree
            confidence *= 1.2
        
        confidence = min(0.95, confidence)  # Cap at 95%
        
        return {
            'action': action,
            'confidence': confidence,
            'predicted_return': predicted_return,
            'ensemble_average': ensemble_avg,
            'model_agreement': 1.0 - pred_std,
            'best_model': pred_data['best_model']
        }

def create_advanced_ensemble_ml(symbol, granularity=3600):
    """Factory function to create advanced ensemble ML system"""
    return AdvancedEnsembleMomentumML(symbol, granularity)

def train_advanced_ensemble(symbol, granularity=3600, days=90):
    """Train advanced ensemble model for symbol"""
    try:
        # Get data
        from maybe import get_coinbase_data
        df = get_coinbase_data(symbol, granularity, days=days)
        
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        # Create and train ensemble
        ensemble = create_advanced_ensemble_ml(symbol, granularity)
        success = ensemble.train_ensemble_models(df)
        
        if success:
            logger.info(f"✅ Advanced ensemble trained for {symbol}")
            return ensemble
        else:
            logger.error(f"❌ Failed to train ensemble for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error training advanced ensemble for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Test the advanced ensemble system
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Advanced Ensemble ML System")
    print("=" * 50)
    
    # Test with BTC-USD
    ensemble = train_advanced_ensemble('BTC-USD', 3600, 60)
    
    if ensemble:
        print("✅ Ensemble training successful!")
        
        # Test predictions
        from maybe import get_coinbase_data
        df = get_coinbase_data('BTC-USD', 3600, days=30)
        
        if df is not None:
            predictions = ensemble.predict_ensemble(df)
            signal = ensemble.get_trading_signal(predictions)
            
            print(f"🎯 Trading Signal: {signal['action']}")
            print(f"📊 Confidence: {signal['confidence']:.1%}")
            print(f"📈 Predicted Return: {signal['predicted_return']:.3%}")
            print(f"🤖 Best Model: {signal['best_model']}")
        else:
            print("❌ Failed to get prediction data")
    else:
        print("❌ Ensemble training failed") 