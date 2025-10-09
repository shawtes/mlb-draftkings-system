#!/usr/bin/env python3
"""
Meta Trading Classifier - Final Decision Layer
Takes ensemble model predictions and makes final BUY/SELL/HOLD decisions
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetaTradingClassifier:
    """
    Meta classifier that makes final trading decisions based on:
    1. Price predictions from ensemble model
    2. Market conditions
    3. Risk factors
    4. Technical indicators
    """
    
    def __init__(self, profit_threshold=0.02, stop_loss_threshold=-0.015, model_path=None):
        self.profit_threshold = profit_threshold  # 2% minimum profit target
        self.stop_loss_threshold = stop_loss_threshold  # -1.5% stop loss
        self.model_path = model_path or 'meta_trading_classifier.joblib'
        self.classifier = None
        self.is_trained = False
        self.feature_names = [
            'predicted_return', 'prediction_confidence', 'current_price',
            'rsi', 'bb_position', 'macd_signal', 'volume_ratio',
            'volatility', 'trend_strength', 'support_resistance_score'
        ]
        
    def prepare_features(self, price_prediction_data):
        """
        Prepare features for the meta classifier
        
        Args:
            price_prediction_data (dict): Contains predictions and market data
            
        Returns:
            np.array: Feature vector for classification
        """
        try:
            # Extract features from prediction data
            predicted_price = price_prediction_data.get('predicted_price', 0)
            current_price = price_prediction_data.get('current_price', 0)
            confidence = price_prediction_data.get('confidence', 0)
            
            # Calculate predicted return
            if current_price > 0:
                predicted_return = (predicted_price - current_price) / current_price
            else:
                predicted_return = 0
                
            # Technical indicators
            rsi = price_prediction_data.get('rsi', 50)
            bb_position = price_prediction_data.get('bb_position', 0.5)  # 0-1 position in Bollinger Bands
            macd_signal = price_prediction_data.get('macd_signal', 0)  # MACD - Signal line
            volume_ratio = price_prediction_data.get('volume_ratio', 1.0)  # Current/Average volume
            volatility = price_prediction_data.get('volatility', 0.02)  # Historical volatility
            trend_strength = price_prediction_data.get('trend_strength', 0)  # -1 to 1
            support_resistance_score = price_prediction_data.get('support_resistance_score', 0)  # Distance from S/R levels
            
            features = np.array([
                predicted_return,
                confidence,
                current_price,
                rsi,
                bb_position,
                macd_signal,
                volume_ratio,
                volatility,
                trend_strength,
                support_resistance_score
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return default features
            return np.zeros((1, len(self.feature_names)))
    
    def create_training_data(self, historical_predictions, historical_outcomes):
        """
        Create training data from historical predictions and actual outcomes
        
        Args:
            historical_predictions (list): Historical prediction data
            historical_outcomes (list): Actual trading outcomes (1=profitable, 0=unprofitable, -1=loss)
            
        Returns:
            tuple: (X, y) training data
        """
        try:
            X = []
            y = []
            
            for pred_data, outcome in zip(historical_predictions, historical_outcomes):
                features = self.prepare_features(pred_data).flatten()
                X.append(features)
                
                # Convert outcome to trading decision
                # 1 = BUY, 0 = HOLD, -1 = SELL (but we'll use 0, 1, 2 for sklearn)
                if outcome > self.profit_threshold:
                    y.append(1)  # BUY (was profitable)
                elif outcome < self.stop_loss_threshold:
                    y.append(2)  # SELL (was a loss)
                else:
                    y.append(0)  # HOLD (neutral)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return np.array([]), np.array([])
    
    def train(self, historical_predictions, historical_outcomes):
        """
        Train the meta classifier
        
        Args:
            historical_predictions (list): Historical prediction data
            historical_outcomes (list): Actual trading outcomes
        """
        try:
            logger.info("Training meta trading classifier...")
            
            # Create training data
            X, y = self.create_training_data(historical_predictions, historical_outcomes)
            
            if len(X) < 10:
                logger.warning("Insufficient training data for meta classifier")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.classifier.fit(X_train, y_train)
              # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Meta classifier accuracy: {accuracy:.3f}")
            logger.info("Classification Report:")
            logger.info(classification_report(y_test, y_pred, target_names=['HOLD', 'BUY', 'SELL']))
            
            # Feature importance
            importance = self.classifier.feature_importances_
            for name, imp in zip(self.feature_names, importance):
                logger.info(f"Feature importance - {name}: {imp:.3f}")
            
            self.is_trained = True
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training meta classifier: {e}")
            return False
    
    def predict_decision(self, price_prediction_data):
        """
        Make final trading decision based on ensemble predictions
        
        Args:
            price_prediction_data (dict): Prediction data from ensemble model
            
        Returns:
            dict: Final trading decision with confidence
        """
        try:
            if not self.is_trained and not self.load_model():
                # Train the model if it doesn't exist
                logger.info("No trained meta classifier found. Training new model...")
                success = train_meta_classifier_from_history()
                if not success:
                    raise RuntimeError("Failed to train meta classifier")
                self.load_model()  # Load the newly trained model
            
            # Prepare features
            features = self.prepare_features(price_prediction_data)
            
            # Get prediction
            decision_idx = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            
            # Convert to decision
            decisions = ['HOLD', 'BUY', 'SELL']
            decision = decisions[decision_idx]
            confidence = probabilities[decision_idx]
            
            # Additional validation
            predicted_return = features[0][0]  # First feature is predicted return
            
            # Override if predicted return is too small for BUY
            if decision == 'BUY' and predicted_return < 0.01:  # Less than 1% return
                decision = 'HOLD'
                confidence = 0.5
                
            # Override if loss is too large for any non-SELL decision
            if predicted_return < -0.02 and decision != 'SELL':  # More than 2% loss
                decision = 'SELL'
                confidence = 0.8
            
            result = {
                'decision': decision,
                'confidence': float(confidence),
                'predicted_return': float(predicted_return),
                'probabilities': {
                    'HOLD': float(probabilities[0]),
                    'BUY': float(probabilities[1]),
                    'SELL': float(probabilities[2])
                },                'model_used': 'meta_classifier'
            }
            
            logger.info(f"Meta decision: {decision} (confidence: {confidence:.3f}, return: {predicted_return:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error making meta decision: {e}")
            raise RuntimeError(f"Meta classifier failed: {e}")
    
    def save_model(self):
        """Save the trained model"""
        try:
            if self.classifier and self.is_trained:
                joblib.dump(joblib.dump({
                    'classifier': self.classifier, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_names': self.feature_names,
                    'profit_threshold': self.profit_threshold,
                    'stop_loss_threshold': self.stop_loss_threshold,
                    'trained_at': datetime.now().isoformat()
                }, self.model_path)
                logger.info(f"Meta classifier saved to {self.model_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load a trained model"""
        try:
            if os.path.exists(self.model_path):
                data = joblib.load(self.model_path)
                self.classifier = data['classifier']
                self.feature_names = data.get('feature_names', self.feature_names)
                self.profit_threshold = data.get('profit_threshold', self.profit_threshold)
                self.stop_loss_threshold = data.get('stop_loss_threshold', self.stop_loss_threshold)
                self.is_trained = True
                logger.info(f"Meta classifier loaded from {self.model_path}")
                return True
            else:
                logger.info("No saved meta classifier found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_stats(self):
        """Get classifier statistics"""
        stats = {
            'is_trained': self.is_trained,
            'profit_threshold': self.profit_threshold,
            'stop_loss_threshold': self.stop_loss_threshold,
            'feature_count': len(self.feature_names),
            'model_path': self.model_path
        }
        
        if self.classifier and self.is_trained:
            stats['n_estimators'] = self.classifier.n_estimators
            stats['feature_importances'] = dict(zip(self.feature_names, self.classifier.feature_importances_))
        
        return stats

def get_ensemble_price_prediction(symbol, investment_amount=100):
    """
    Get price prediction from the actual ensemble model
    
    Args:
        symbol (str): Trading symbol
        investment_amount (float): Investment amount
        
    Returns:
        dict: Ensemble prediction data for meta classifier
    """
    try:
        # Try to import and use the actual ensemble models
        try:
            from ensemble_models import get_ensemble_prediction
            ensemble_result = get_ensemble_prediction(symbol)
            logger.info(f"Using ensemble_models for {symbol}")
        except ImportError:
            try:
                from advanced_ensemble_ml import AdvancedEnsembleMomentumML
                ensemble_engine = AdvancedEnsembleMomentumML(symbol)
                from maybe import get_coinbase_data
                df = get_coinbase_data(symbol, 3600, days=7)
                ensemble_result = ensemble_engine.predict_ensemble(df)
                logger.info(f"Using advanced_ensemble_ml for {symbol}")
            except ImportError:
                try:
                    from maybe import make_price_prediction
                    ensemble_result = make_price_prediction(symbol, investment_amount=investment_amount)
                    logger.info(f"Using maybe.py price prediction for {symbol}")
                except ImportError:
                    logger.error("No ensemble model available")
                    # Create minimal fallback prediction for testing
                    try:
                        from maybe import get_coinbase_data
                        df = get_coinbase_data(symbol, 3600, days=1)
                        if df is not None and not df.empty:
                            current_price = df['close'].iloc[-1]
                            # Very small prediction for testing
                            ensemble_result = {'predicted_price': current_price * 1.0005, 'confidence': 0.1}
                            logger.warning(f"Using minimal fallback prediction for {symbol}")
                        else:
                            return None
                    except Exception:
                        return None
        
        if not ensemble_result:
            logger.warning(f"No ensemble prediction available for {symbol}")
            return None
        
        # Get current market data
        try:
            from maybe import get_coinbase_data, calculate_indicators
            df = get_coinbase_data(symbol, 3600, days=1)
            if df is None or df.empty:
                logger.error(f"No market data for {symbol}")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Calculate technical indicators
            df_indicators = calculate_indicators(df)
            latest_row = df_indicators.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
        
        # Extract prediction data from ensemble result
        if isinstance(ensemble_result, dict):
            # Handle different ensemble result formats
            predicted_price = ensemble_result.get('predicted_price')
            if predicted_price is None:
                # Try different keys
                for key in ['target_price', 'price_prediction', 'best_prediction', 'ensemble_average']:
                    if key in ensemble_result:
                        predicted_price = ensemble_result[key]
                        break
            
            if predicted_price is None:
                # Try to extract from nested structures                if 'predictions' in ensemble_result:
                    pred_dict = ensemble_result['predictions']
                    if isinstance(pred_dict, dict):
                        predicted_price = list(pred_dict.values())[0] if pred_dict else None
            
            confidence = ensemble_result.get('confidence', 0.5)
            if confidence is None:
                confidence = ensemble_result.get('overall_confidence', 0.5)
        else:
            # Handle non-dict results
            predicted_price = float(ensemble_result) if ensemble_result else None
            confidence = 0.5
        
        if predicted_price is None or predicted_price <= 0:
            logger.warning(f"Invalid predicted price for {symbol}: {predicted_price}")
            # Use current price as fallback for testing
            predicted_price = current_price * 1.001  # Small 0.1% prediction
            logger.warning(f"Using fallback predicted price: {predicted_price}")
            confidence = 0.1  # Low confidence for fallback
        
        # Prepare data for meta classifier
        prediction_data = {
            'predicted_price': float(predicted_price),
            'current_price': float(current_price),
            'confidence': float(confidence),
            'rsi': float(latest_row.get('RSI', 50)),
            'bb_position': 0.5,  # Would need Bollinger Bands calculation
            'macd_signal': float(latest_row.get('MACD', 0)) - float(latest_row.get('Signal_Line', 0)),
            'volume_ratio': 1.0,  # Would need volume analysis
            'volatility': float(df['close'].pct_change().std()) if len(df) > 1 else 0.02,
            'trend_strength': float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) if len(df) > 10 else 0,
            'support_resistance_score': 0.0  # Would need S/R level calculation
        }
        
        logger.info(f"Ensemble prediction for {symbol}: price=${predicted_price:.4f}, current=${current_price:.4f}")
        return prediction_data
        
    except Exception as e:
        logger.error(f"Error getting ensemble prediction for {symbol}: {e}")
        return None

# Global instance
meta_classifier = MetaTradingClassifier()

def get_meta_decision(ensemble_prediction_data=None, symbol=None, investment_amount=100):
    """
    Convenience function to get meta trading decision
    
    Args:
        ensemble_prediction_data (dict): Prediction data from ensemble model (optional)
        symbol (str): Trading symbol (if ensemble_prediction_data not provided)
        investment_amount (float): Investment amount (if ensemble_prediction_data not provided)
        
    Returns:
        dict: Final trading decision
    """
    if ensemble_prediction_data is None:
        if symbol is None:
            raise ValueError("Either ensemble_prediction_data or symbol must be provided")
        
        # Get prediction from actual ensemble model
        ensemble_prediction_data = get_ensemble_price_prediction(symbol, investment_amount)
        if ensemble_prediction_data is None:
            raise RuntimeError(f"Failed to get ensemble prediction for {symbol}")
    
    return meta_classifier.predict_decision(ensemble_prediction_data)

def train_meta_classifier_from_history():
    """
    Train the meta classifier using historical data
    This should be called with actual historical trading data
    """
    try:
        # This is a placeholder - in practice, you would load historical data
        # from your database or trading logs
        
        logger.info("Loading historical trading data for meta classifier training...")
        
        # Example: Load from database or CSV
        # historical_predictions = load_historical_predictions()
        # historical_outcomes = load_historical_outcomes()
        
        # For now, create dummy data for demonstration
        logger.warning("Using dummy data for meta classifier training - replace with real data")
        
        historical_predictions = []
        historical_outcomes = []
        
        # Generate some example training data
        np.random.seed(42)
        for i in range(100):
            pred_data = {
                'predicted_price': 50000 + np.random.normal(0, 5000),
                'current_price': 50000,
                'confidence': np.random.uniform(0.3, 0.9),
                'rsi': np.random.uniform(20, 80),
                'bb_position': np.random.uniform(0, 1),
                'macd_signal': np.random.normal(0, 100),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'volatility': np.random.uniform(0.01, 0.05),
                'trend_strength': np.random.uniform(-1, 1),
                'support_resistance_score': np.random.uniform(-0.02, 0.02)
            }
            
            # Simulate outcome based on predicted return
            predicted_return = (pred_data['predicted_price'] - pred_data['current_price']) / pred_data['current_price']
            
            # Add some noise to simulate real trading outcomes
            actual_outcome = predicted_return + np.random.normal(0, 0.01)
            
            historical_predictions.append(pred_data)
            historical_outcomes.append(actual_outcome)
        
        # Train the classifier
        success = meta_classifier.train(historical_predictions, historical_outcomes)
        
        if success:
            logger.info("Meta classifier training completed successfully")
            return True
        else:
            logger.error("Meta classifier training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error training meta classifier: {e}")
        return False

def test_complete_meta_system(symbol='BTC-USD', investment_amount=100):
    """
    Test the complete meta classifier system with real ensemble predictions
    
    Args:
        symbol (str): Trading symbol to test
        investment_amount (float): Investment amount
        
    Returns:
        dict: Complete test results
    """
    try:
        logger.info(f"🤖 Testing complete meta system for {symbol}")
        logger.info("=" * 60)
        
        # Step 1: Get ensemble price prediction
        logger.info("📊 Step 1: Getting ensemble price prediction...")
        ensemble_data = get_ensemble_price_prediction(symbol, investment_amount)
        
        if ensemble_data is None:
            logger.error("❌ Failed to get ensemble prediction")
            return {'success': False, 'error': 'No ensemble prediction available'}
        
        logger.info(f"✅ Ensemble prediction received:")
        logger.info(f"   Predicted price: ${ensemble_data['predicted_price']:.4f}")
        logger.info(f"   Current price: ${ensemble_data['current_price']:.4f}")
        logger.info(f"   Expected return: {((ensemble_data['predicted_price'] / ensemble_data['current_price']) - 1) * 100:.2f}%")
        logger.info(f"   Ensemble confidence: {ensemble_data['confidence']:.3f}")
        
        # Step 2: Train meta classifier if needed
        logger.info("🧠 Step 2: Ensuring meta classifier is trained...")
        if not meta_classifier.is_trained and not meta_classifier.load_model():
            success = train_meta_classifier_from_history()
            if not success:
                logger.error("❌ Failed to train meta classifier")
                return {'success': False, 'error': 'Meta classifier training failed'}
        
        # Step 3: Get meta decision
        logger.info("🎯 Step 3: Getting meta classifier decision...")
        meta_decision = meta_classifier.predict_decision(ensemble_data)
        
        logger.info(f"✅ Meta classifier decision:")
        logger.info(f"   Final decision: {meta_decision['decision']}")
        logger.info(f"   Meta confidence: {meta_decision['confidence']:.3f}")
        logger.info(f"   Predicted return: {meta_decision['predicted_return']:.4f}")
        logger.info(f"   Model used: {meta_decision['model_used']}")
        
        # Step 4: Calculate trade recommendation
        logger.info("💰 Step 4: Calculating trade recommendation...")
        
        if meta_decision['decision'] == 'BUY':
            position_size = min(investment_amount, investment_amount * meta_decision['confidence'])
            expected_profit = position_size * meta_decision['predicted_return']
            
            logger.info(f"📈 BUY RECOMMENDATION:")
            logger.info(f"   Recommended position size: ${position_size:.2f}")
            logger.info(f"   Expected profit: ${expected_profit:.2f}")
            logger.info(f"   Risk level: {1 - meta_decision['confidence']:.1%}")
            
        elif meta_decision['decision'] == 'SELL':
            logger.info(f"📉 SELL RECOMMENDATION:")
            logger.info(f"   Sell any existing positions")
            logger.info(f"   Expected loss prevention: {abs(meta_decision['predicted_return']) * 100:.1f}%")
            
        else:
            logger.info(f"⏸️ HOLD RECOMMENDATION:")
            logger.info(f"   No action recommended")
            logger.info(f"   Market conditions unclear")
        
        result = {
            'success': True,
            'symbol': symbol,
            'ensemble_prediction': ensemble_data,
            'meta_decision': meta_decision,
            'recommendation': {
                'action': meta_decision['decision'],
                'confidence': meta_decision['confidence'],
                'expected_return': meta_decision['predicted_return'],
                'risk_level': 1 - meta_decision['confidence']
            }
        }
        
        logger.info("🎉 Complete meta system test successful!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in complete meta system test: {e}")
        return {
            'success': False,
            'error': str(e),
            'symbol': symbol
        }

if __name__ == "__main__":
    # Test the complete meta classifier system
    logger.info("🚀 Meta Trading Classifier - Complete System Test")
    logger.info("=" * 70)
    
    # Test with real ensemble integration
    result = test_complete_meta_system('BTC-USD', 100)
    
    if result['success']:
        logger.info("✅ Complete system test PASSED!")
    else:
        logger.error(f"❌ Complete system test FAILED: {result['error']}")
    
    # Show final stats
    stats = meta_classifier.get_stats()
    logger.info(f"📊 Final meta classifier stats: {stats}")
