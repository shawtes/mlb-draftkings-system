#!/usr/bin/env python3
"""
Stacking Regressor ML Engine
High-performance ML engine using the best-performing StackingRegressor from ensemble tests
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging
import pickle
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to path for maybe module
app_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'app')
app_dir = os.path.abspath(app_dir)
if os.path.exists(app_dir):
    sys.path.insert(0, app_dir)
    logger.info(f"Added app directory to path: {app_dir}")
else:
    logger.warning(f"App directory not found: {app_dir}")

try:
    from maybe import get_coinbase_data, calculate_indicators
    logger.info("✅ Successfully imported data functions from maybe module")
except ImportError as e:
    logger.warning(f"⚠️ Could not import from maybe module: {e}")
    # Import from our fallback utility
    try:
        from coinbase_data_utils import get_coinbase_data, calculate_indicators
        logger.info("✅ Using fallback coinbase_data_utils")
    except ImportError:
        logger.error("❌ No data utilities available")
        # Create dummy functions as final fallback
        def get_coinbase_data(symbol='BTC-USD', granularity=3600, days=7):
            logger.warning("Using dummy get_coinbase_data function")
            return None
        
        def calculate_indicators(df, symbol=None):
            logger.warning("Using dummy calculate_indicators function")
            return df

# Import ML models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = GradientBoostingRegressor

# Try to import quantitative finance enhancements
try:
    # Add app directory to path for quantitative finance modules
    app_dir = os.path.join(os.path.dirname(current_dir), '..', 'app')
    if os.path.exists(app_dir):
        sys.path.insert(0, app_dir)
    
    from quant_ml_integration import QuantitativeMLIntegrator
    QUANT_FINANCE_AVAILABLE = True
    logger.info("✅ Quantitative finance enhancements available")
except ImportError as e:
    QUANT_FINANCE_AVAILABLE = False
    logger.warning(f"⚠️ Quantitative finance enhancements not available: {e}")
    # Create dummy class
    class QuantitativeMLIntegrator:
        def __init__(self): pass
        def enhance_ml_prediction(self, *args, **kwargs): 
            return {'enhanced_confidence': kwargs.get('base_confidence', 0.5)}

class StackingMLEngine:
    """Advanced ML Engine using StackingRegressor - the best performer from ensemble tests"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.last_performance = {}
        self.model_cache_dir = "model_cache"
        
        # Initialize quantitative finance integrator if available
        if QUANT_FINANCE_AVAILABLE:
            try:
                self.quant_integrator = QuantitativeMLIntegrator()
                logger.info("🔬 Quantitative finance integrator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize quantitative finance integrator: {e}")
                self.quant_integrator = None
        else:
            self.quant_integrator = None
        
        # Create cache directory
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)
        
        logger.info("🧠 StackingML Engine initialized")
    
    def create_stacking_model(self):
        """Create the high-performance StackingRegressor model"""
        
        # Base models for stacking (optimized from ensemble tests)
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=1.0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42, max_depth=8) if XGBOOST_AVAILABLE 
                    else GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=8))
        ]
        
        # StackingRegressor with Ridge as final estimator (best configuration)
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=0.1),
            cv=3,
            n_jobs=1
        )
        
        return stacking_model
    
    def prepare_features(self, df, symbol):
        """Prepare features for StackingRegressor"""
        try:
            logger.info(f"🔧 Preparing features for {symbol}")
            
            # Calculate future returns (target)
            df['future_return_pct'] = (df['close'].shift(-1) / df['close'] - 1) * 100
            
            # Get all numeric feature columns (from enhanced indicators)
            exclude_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'future_return_pct'}
            
            feature_columns = []
            for col in df.columns:
                if col not in exclude_columns and df[col].dtype in ['float64', 'int64']:
                    feature_columns.append(col)
            
            # If no enhanced features, create basic ones
            if len(feature_columns) == 0:
                df['price_change'] = df['close'].pct_change() * 100
                df['volume_change'] = df['volume'].pct_change() * 100
                df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
                df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
                
                # Basic technical indicators
                df['sma_5'] = df['close'].rolling(5).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
                
                feature_columns = ['price_change', 'volume_change', 'high_low_ratio', 
                                 'price_position', 'price_vs_sma20']
            
            # Clean data
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                return None, None, []
            
            # Prepare features and target
            X = df[feature_columns].fillna(0)
            y = df['future_return_pct'].fillna(0)
            
            # Remove extreme outliers (keep 90% of data)
            q5, q95 = y.quantile([0.05, 0.95])
            mask = (y >= q5) & (y <= q95)
            X = X[mask]
            y = y[mask]
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"✅ Prepared {len(X)} samples with {X.shape[1]} features for {symbol}")
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"❌ Error preparing features for {symbol}: {str(e)}")
            return None, None, []
    
    def train_model(self, symbol, granularity=3600, days=None):
        """Train StackingRegressor model for a symbol"""
        try:
            logger.info(f"🚀 Training StackingRegressor for {symbol}")
            
            # Calculate appropriate days based on granularity to stay under 300 candle limit
            if days is None:
                max_days = min(300 * granularity // (24 * 3600), 14)  # Max 14 days, respect API limits
                days = max(3, max_days)  # Minimum 3 days for training
            
            # Get data
            df = get_coinbase_data(symbol, granularity, days=days)
            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return False
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            # Prepare features
            X, y, feature_columns = self.prepare_features(df, symbol)
            
            if X is None or len(X) < 50:
                logger.error(f"Insufficient data for training {symbol}")
                return False
            
            # Store feature columns
            self.feature_columns = feature_columns
            
            # Split data (temporal split)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train StackingRegressor
            model = self.create_stacking_model()
            
            logger.info(f"🎯 Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions and evaluate
            y_pred = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Directional accuracy
            direction_true = np.sign(y_test)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_true == direction_pred)
            
            # Correlation
            correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
            
            performance = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'directional_accuracy': directional_accuracy,
                'correlation': correlation,
                'samples': len(y_test),
                'trained_at': datetime.now()
            }
            
            logger.info(f"✅ StackingRegressor trained for {symbol}")
            logger.info(f"   📊 R² Score: {r2:.4f}")
            logger.info(f"   📏 MAE: {mae:.4f}")
            logger.info(f"   🧭 Directional Accuracy: {directional_accuracy:.1%}")
            logger.info(f"   🔗 Correlation: {correlation:.4f}")
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.last_performance[symbol] = performance
            
            # Save to disk
            self._save_model(symbol, model, scaler, feature_columns, performance)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training model for {symbol}: {str(e)}")
            return False
    
    def _save_model(self, symbol, model, scaler, feature_columns, performance):
        """Save model and metadata to disk"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'performance': performance,
                'version': 'stacking_v1.0'
            }
            
            filepath = os.path.join(self.model_cache_dir, f"{symbol.replace('-', '_')}_stacking.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Model saved for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model for {symbol}: {str(e)}")
    
    def _load_model(self, symbol):
        """Load model from disk"""
        try:
            filepath = os.path.join(self.model_cache_dir, f"{symbol.replace('-', '_')}_stacking.pkl")
            
            if not os.path.exists(filepath):
                return False
            
            # Check if model is recent (within 24 hours)
            file_age = time.time() - os.path.getmtime(filepath)
            if file_age > 86400:  # 24 hours
                logger.info(f"⏰ Model for {symbol} is stale ({file_age/3600:.1f}h old), will retrain")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[symbol] = model_data['model']
            self.scalers[symbol] = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.last_performance[symbol] = model_data['performance']
            
            logger.info(f"📂 Loaded cached model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model for {symbol}: {str(e)}")
            return False
    
    def make_prediction(self, symbol, granularity=3600, investment_amount=100.0):
        """Make enhanced prediction using StackingRegressor"""
        try:
            logger.info(f"🔮 Making StackingRegressor prediction for {symbol}")
            
            # Load or train model
            if symbol not in self.models:
                if not self._load_model(symbol):
                    logger.info(f"🚀 Training new model for {symbol}")
                    if not self.train_model(symbol, granularity):
                        return None
            
            # Get recent data for prediction (limit to 7 days for hourly data to stay under 300 candle limit)
            days_to_fetch = min(7, 300 * granularity // (24 * 3600))  # Calculate max days for granularity
            df = get_coinbase_data(symbol, granularity, days=days_to_fetch)
            if df is None or df.empty:
                logger.error(f"No data available for prediction on {symbol}")
                return None
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            # Prepare features (same as training)
            X, _, _ = self.prepare_features(df, symbol)
            if X is None:
                return None
            
            # Get latest features for prediction
            latest_features = X.iloc[-1:][self.feature_columns]
            
            # Scale features
            scaler = self.scalers[symbol]
            latest_features_scaled = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[symbol]
            predicted_return = model.predict(latest_features_scaled)[0]
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            predicted_price = current_price * (1 + predicted_return / 100)
            
            # Calculate confidence based on model performance and prediction strength
            performance = self.last_performance.get(symbol, {})
            base_confidence = performance.get('directional_accuracy', 0.5)
            prediction_strength = min(abs(predicted_return) / 5.0, 1.0)  # Normalize to 0-1
            ml_confidence = (base_confidence + prediction_strength) / 2
            
            # Determine preliminary action
            action = "BUY" if predicted_return > 0.5 else "SELL" if predicted_return < -0.5 else "HOLD"
            
            # Apply quantitative finance enhancements if available
            enhanced_confidence = ml_confidence
            quant_analysis = None
            
            if self.quant_integrator and len(df) >= 100:
                try:
                    logger.info(f"🔬 Applying quantitative finance enhancements for {symbol}")
                    
                    # Get enhanced ML prediction with quantitative analysis
                    quant_result = self.quant_integrator.enhance_ml_prediction(
                        symbol=symbol,
                        df=df,
                        ml_decision=action,
                        ml_confidence=ml_confidence
                    )
                    
                    if quant_result:
                        enhanced_confidence = quant_result.get('enhanced_confidence', ml_confidence)
                        action = quant_result.get('enhanced_decision', action)
                        quant_analysis = {
                            'confidence_adjustment': quant_result.get('confidence_adjustment', 0),
                            'position_multiplier': quant_result.get('position_size_multiplier', 1.0),
                            'regime': quant_result.get('market_regime', 'UNKNOWN'),
                            'risk_metrics': quant_result.get('risk_metrics', {}),
                            'enhanced': True
                        }
                        
                        logger.info(f"✅ Quantitative enhancements applied for {symbol}")
                        logger.info(f"   Original: {quant_result.get('original_decision', 'N/A')} ({quant_result.get('original_confidence', 0):.1%})")
                        logger.info(f"   Enhanced: {action} ({enhanced_confidence:.1%})")
                        logger.info(f"   Confidence Adjustment: {quant_result.get('confidence_adjustment', 0):+.1%}")
                    else:
                        logger.warning(f"⚠️ No quantitative enhancement result for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Quantitative finance enhancement failed for {symbol}: {str(e)}")
                    quant_analysis = {'enhanced': False, 'error': str(e)}
            else:
                if not self.quant_integrator:
                    logger.debug(f"📊 Quantitative finance not available for {symbol}")
                else:
                    logger.debug(f"📊 Insufficient data for quantitative analysis ({len(df)} < 100)")
            
            overall_confidence = enhanced_confidence
            
            # Calculate profit estimation
            fees = 0.0035 * 2  # Buy + sell fees (0.35% each)
            gross_profit_pct = abs(predicted_return)
            net_profit_pct = gross_profit_pct - (fees * 100)
            expected_profit_usd = (investment_amount * net_profit_pct / 100) if action != "HOLD" else 0
            
            # Profit probability based on directional accuracy
            profit_probability = performance.get('directional_accuracy', 0.5)
            
            prediction = {
                'symbol': symbol,
                'action': action,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return_pct': predicted_return,
                'overall_confidence': overall_confidence,
                'ml_confidence': ml_confidence,
                'expected_profit_pct': net_profit_pct,
                'expected_profit_usd': expected_profit_usd,
                'profit_probability': profit_probability,
                'model_performance': {
                    'r2_score': performance.get('r2_score', 0),
                    'directional_accuracy': performance.get('directional_accuracy', 0),
                    'correlation': performance.get('correlation', 0)
                },
                'quantitative_analysis': quant_analysis,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'StackingRegressor',
                'investment_amount': investment_amount
            }
            
            logger.info(f"✅ StackingRegressor prediction for {symbol}:")
            logger.info(f"   🎯 Action: {action}")
            logger.info(f"   💰 Expected Return: {predicted_return:+.2f}%")
            logger.info(f"   🎪 Confidence: {overall_confidence:.1%}")
            logger.info(f"   💵 Expected Profit: ${expected_profit_usd:+.2f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Error making prediction for {symbol}: {str(e)}")
            return None
    
    def get_multi_timeframe_predictions(self, symbol):
        """Get predictions across multiple timeframes"""
        timeframes = {
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        predictions = {}
        
        for tf_name, granularity in timeframes.items():
            try:
                logger.info(f"📊 Getting {tf_name} prediction for {symbol}")
                prediction = self.make_prediction(symbol, granularity)
                
                if prediction:
                    predictions[tf_name] = {
                        'prediction': prediction,
                        'confidence': prediction['overall_confidence'],
                        'timeframe': tf_name,
                        'granularity': granularity
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Error getting {tf_name} prediction for {symbol}: {str(e)}")
                continue
        
        return predictions
    
    def get_model_info(self, symbol):
        """Get information about the model for a symbol"""
        if symbol in self.last_performance:
            perf = self.last_performance[symbol]
            return {
                'model_type': 'StackingRegressor',
                'performance': perf,
                'features_count': len(self.feature_columns),
                'last_trained': perf.get('trained_at'),
                'available': True
            }
        else:
            return {
                'model_type': 'StackingRegressor',
                'available': False,
                'message': 'Model not trained yet'
            }

# Global engine instance
stacking_engine = StackingMLEngine()

# Convenience functions for dashboard integration
def make_enhanced_ml_decision(symbol, granularity=3600, investment_amount=100.0):
    """Make enhanced ML decision using StackingRegressor"""
    return stacking_engine.make_prediction(symbol, granularity, investment_amount)

def get_multi_timeframe_predictions(symbol):
    """Get multi-timeframe predictions using StackingRegressor"""
    return stacking_engine.get_multi_timeframe_predictions(symbol)

def train_price_prediction_models(symbol, granularity=3600):
    """Train StackingRegressor models"""
    return stacking_engine.train_model(symbol, granularity)

def get_model_performance(symbol):
    """Get model performance information"""
    return stacking_engine.get_model_info(symbol)

if __name__ == "__main__":
    # Test the engine
    logger.info("🧪 Testing StackingML Engine")
    
    # Test on BTC-USD
    test_symbol = "BTC-USD"
    
    # Train model
    if stacking_engine.train_model(test_symbol):
        # Make prediction
        prediction = stacking_engine.make_prediction(test_symbol, investment_amount=100.0)
        
        if prediction:
            logger.info("🎉 StackingML Engine test successful!")
            logger.info(f"Test prediction: {prediction['action']} with {prediction['overall_confidence']:.1%} confidence")
        else:
            logger.error("❌ Prediction failed")
    else:
        logger.error("❌ Training failed")