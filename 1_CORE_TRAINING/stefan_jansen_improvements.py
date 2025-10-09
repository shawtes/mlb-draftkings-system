"""
Stefan Jansen ML for Algorithmic Trading Improvements
=====================================================

Key improvements based on "Machine Learning for Algorithmic Trading" by Stefan Jansen:

1. Momentum-specific feature engineering (Chapter 4)
2. Ensemble methods for better predictions (Chapter 7)
3. Online learning for adaptation (Chapter 8)
4. Risk-adjusted returns and portfolio optimization (Chapter 5)
5. Alternative data integration (Chapter 3)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from scipy import signal
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# Enhanced Feature Engineering System
class EnhancedFeatureEngine:
    def __init__(self, enable_denoising=True, enable_alpha_factors=True):
        self.enable_denoising = enable_denoising and KALMAN_AVAILABLE
        self.enable_alpha_factors = enable_alpha_factors
        self.scalers = {}
        self.pca_models = {}

    def calculate_enhanced_features(self, df, symbol=None, timeframe='1h'):
        """
        Main function to calculate all enhanced features
        """
        try:
            if df is None or df.empty:
                raise ValueError("Input DataFrame is empty")

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Make a copy to avoid modifying the original
            df = df.copy()

            # Remove rows with NaN values in OHLCV
            df = df.dropna(subset=required_cols)

            if len(df) < 50:
                raise ValueError("Insufficient data for feature calculation")

            # Example: Add basic technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()

            return df

        except Exception as e:
            logger.error(f"Error in calculate_enhanced_features: {str(e)}")
            return df  # Return original DataFrame on error

# Global instance of EnhancedFeatureEngine
enhanced_feature_engine = EnhancedFeatureEngine()

def calculate_enhanced_indicators(df, symbol=None, timeframe='1h'):
    """
    Wrapper function to calculate enhanced indicators
    """
    try:
        return enhanced_feature_engine.calculate_enhanced_features(df, symbol, timeframe)
    except Exception as e:
        logger.error(f"Error in calculate_enhanced_indicators: {str(e)}")
        return df

def calculate_momentum_features(df):
    """
    Enhanced momentum features based on Stefan Jansen's book
    Chapter 4: Alpha Factor Research
    """
    # === MOMENTUM FACTORS ===
    
    # Multiple timeframe momentum (Jansen Chapter 4)
    df['momentum_1w'] = df['close'] / df['close'].shift(5) - 1      # 1-week momentum
    df['momentum_1m'] = df['close'] / df['close'].shift(20) - 1     # 1-month momentum
    df['momentum_3m'] = df['close'] / df['close'].shift(60) - 1     # 3-month momentum
    df['momentum_6m'] = df['close'] / df['close'].shift(120) - 1    # 6-month momentum
    df['momentum_12m'] = df['close'] / df['close'].shift(252) - 1   # 1-year momentum
    
    # Price acceleration (second and third derivatives)
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['price_jerk'] = df['price_acceleration'].diff()
    
    # === VOLUME-BASED MOMENTUM ===
    
    # Volume-Price Trend (VPT) - Jansen uses this extensively
    df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
    df['vpt_momentum'] = df['vpt'].pct_change(10)
    
    # Accumulation/Distribution Line
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    df['ad_line'] = money_flow_volume.cumsum()
    df['ad_momentum'] = df['ad_line'].pct_change(10)
    
    # On Balance Volume momentum
    obv_direction = np.where(df['close'] > df['close'].shift(1), df['volume'], 
                           np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['obv'] = obv_direction.cumsum()
    df['obv_momentum'] = df['obv'].pct_change(10)
    
    # === BREAKOUT AND TREND FEATURES ===
    
    # Multiple timeframe breakouts
    for period in [10, 20, 50]:
        df[f'breakout_high_{period}'] = (df['close'] > df['high'].rolling(period).max().shift(1)).astype(int)
        df[f'breakout_low_{period}'] = (df['close'] < df['low'].rolling(period).min().shift(1)).astype(int)
        
    # Trend strength indicators
    df['trend_strength_short'] = abs(df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close']
    df['trend_strength_long'] = abs(df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close']
    
    # === RELATIVE STRENGTH ===
    
    # Multiple timeframe relative strength
    for period in [10, 20, 50]:
        df[f'relative_strength_{period}'] = df['close'] / df['close'].rolling(period).mean() - 1
    
    # === VOLATILITY-ADJUSTED MOMENTUM ===
    
    # Sharpe-like momentum (risk-adjusted)
    for period in [10, 20]:
        returns = df['close'].pct_change()
        rolling_mean = returns.rolling(period).mean()
        rolling_std = returns.rolling(period).std()
        df[f'sharpe_momentum_{period}'] = rolling_mean / rolling_std
    
    # === MONEY FLOW INDICATORS ===
    
    # Chaikin Money Flow
    money_flow_volume = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['cmf'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
    
    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow))
    
    return df

def create_momentum_ensemble_model(df, symbol, granularity):
    """
    Ensemble model specifically designed for momentum trading
    Based on Stefan Jansen Chapter 7: Linear Models
    """
    
    # Add momentum features
    df = calculate_momentum_features(df)
    
    # Feature selection for momentum (focus on momentum indicators)
    momentum_features = [
        'momentum_1w', 'momentum_1m', 'momentum_3m', 'momentum_6m',
        'price_acceleration', 'vpt_momentum', 'ad_momentum', 'obv_momentum',
        'breakout_high_10', 'breakout_high_20', 'breakout_low_10', 'breakout_low_20',
        'trend_strength_short', 'trend_strength_long',
        'relative_strength_10', 'relative_strength_20', 'relative_strength_50',
        'sharpe_momentum_10', 'sharpe_momentum_20',
        'cmf', 'mfi', 'rsi'
    ]
    
    # Create target (next period return)
    # For momentum trading, we want to predict continuation
    horizon = max(1, granularity // 3600)  # Adjust horizon based on granularity
    df['target'] = df['close'].shift(-horizon) / df['close'] - 1
    
    # Clean data
    df = df.dropna()
    
    if len(df) < 100:
        logger.warning(f"Insufficient data for ensemble model: {len(df)} rows")
        return None
    
    # Prepare features and target
    available_features = [f for f in momentum_features if f in df.columns]
    X = df[available_features].fillna(0)
    y = df['target'].fillna(0)
    
    # Train/test split (time series aware)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features for better ensemble performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble of momentum-focused models
    models = {
        'gradient_boost': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        ),
        'ridge': Ridge(
            alpha=1.0,
            random_state=42
        )
    }
    
    # Train models
    trained_models = {}
    model_weights = {}
    
    for name, model in models.items():
        try:
            if name == 'ridge':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate model performance for weighting
            mae = mean_absolute_error(y_test, y_pred)
            direction_accuracy = np.mean((y_test > 0) == (y_pred > 0))
            
            # Weight models based on direction accuracy (more important for momentum)
            weight = direction_accuracy
            
            trained_models[name] = model
            model_weights[name] = weight
            
            logger.info(f"{name}: MAE={mae:.4f}, Direction Accuracy={direction_accuracy:.1%}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Save ensemble
    ensemble_data = {
        'models': trained_models,
        'weights': model_weights,
        'scaler': scaler,
        'features': available_features,
        'horizon': horizon,
        'symbol': symbol,
        'granularity': granularity
    }
    
    return ensemble_data

def predict_with_momentum_ensemble(ensemble_data, df):
    """
    Make predictions using the momentum ensemble
    """
    try:
        # Add momentum features
        df = calculate_momentum_features(df)
        
        # Prepare features
        features = ensemble_data['features']
        X = df[features].iloc[-1:].fillna(0)
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in ensemble_data['models'].items():
            try:
                if name == 'ridge':
                    X_scaled = ensemble_data['scaler'].transform(X)
                    pred = model.predict(X_scaled)[0]
                else:
                    pred = model.predict(X)[0]
                
                predictions[name] = pred
                
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
        
        # Weighted ensemble prediction
        if predictions:
            weights = ensemble_data['weights']
            weighted_prediction = sum(pred * weights.get(name, 0) for name, pred in predictions.items())
            
            # Calculate confidence based on model agreement
            pred_values = list(predictions.values())
            confidence = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-6))
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'predicted_return': weighted_prediction,
                'confidence': confidence,
                'individual_predictions': predictions,
                'horizon_hours': ensemble_data['horizon'] * ensemble_data['granularity'] / 3600
            }
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
    
    return None

def create_momentum_override_logic(symbol, momentum_score, price_change_pct, ml_decision):
    """
    Stefan Jansen's approach: Use multiple signals and risk management
    Override ML decision for strong momentum signals
    """
    
    # Strong momentum thresholds (based on Jansen's factor research)
    strong_momentum_score = 85
    strong_price_change = 5.0  # 5%
    
    # Very strong momentum thresholds
    very_strong_momentum_score = 95
    very_strong_price_change = 10.0  # 10%
    
    # Check for momentum override conditions
    if momentum_score >= very_strong_momentum_score and abs(price_change_pct) >= very_strong_price_change:
        # Very strong momentum - override any ML decision
        action = 'BUY' if price_change_pct > 0 else 'SELL'
        confidence = 0.9
        reason = f"Very strong momentum override (score: {momentum_score}, change: {price_change_pct:+.1f}%)"
        
        logger.info(f"🚀 {symbol}: {reason}")
        
        return {
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'override': True
        }
    
    elif momentum_score >= strong_momentum_score and abs(price_change_pct) >= strong_price_change:
        # Strong momentum - override only HOLD decisions
        if ml_decision.get('action') == 'HOLD':
            action = 'BUY' if price_change_pct > 0 else 'SELL'
            confidence = 0.75
            reason = f"Strong momentum override of HOLD (score: {momentum_score}, change: {price_change_pct:+.1f}%)"
            
            logger.info(f"⚡ {symbol}: {reason}")
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'override': True
            }
    
    # No override - use ML decision
    return {
        'action': ml_decision.get('action', 'HOLD'),
        'confidence': ml_decision.get('confidence', 0.0),
        'reason': 'ML decision (no momentum override)',
        'override': False
    }

# Additional Stefan Jansen improvements to implement:

def calculate_factor_scores(df):
    """
    Calculate factor scores as in Jansen Chapter 4
    """
    scores = {}
    
    # Momentum factor
    scores['momentum'] = df['momentum_3m'].iloc[-1] if 'momentum_3m' in df.columns else 0
    
    # Volatility factor  
    returns = df['close'].pct_change()
    scores['volatility'] = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0
    
    # Volume factor
    scores['volume'] = (df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] - 1) if len(df) > 20 else 0
    
    # Quality factor (consistency of returns)
    if len(returns) > 60:
        scores['quality'] = -returns.rolling(60).std().iloc[-1]  # Lower volatility = higher quality
    else:
        scores['quality'] = 0
    
    return scores

def risk_adjusted_position_sizing(portfolio_value, volatility, target_vol=0.15):
    """
    Risk parity position sizing based on Jansen Chapter 5
    """
    if volatility <= 0:
        return 0
    
    # Target position size based on volatility targeting
    position_size = (target_vol / volatility) * portfolio_value
    
    # Cap at reasonable limits
    max_position = portfolio_value * 0.2  # Max 20% per position
    min_position = portfolio_value * 0.01  # Min 1% per position
    
    return max(min_position, min(max_position, position_size))