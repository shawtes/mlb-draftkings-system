"""
ML IMPROVEMENT GUIDE FOR MOMENTUM TRADING
========================================

Based on Stefan Jansen's "Machine Learning for Algorithmic Trading"

This guide shows you HOW TO IMPROVE YOUR ML specifically for momentum trading.
"""

# ============================================================================
# 🎯 KEY ML IMPROVEMENTS FOR YOUR MOMENTUM SYSTEM
# ============================================================================

"""
PROBLEM: Your current ML models are trained for MEAN REVERSION (buying when price goes down)
SOLUTION: Train MOMENTUM-SPECIFIC models that identify trend CONTINUATION patterns

Here are the 10 most impactful improvements:
"""

# ============================================================================
# 1. 🚀 FIX YOUR TRAINING TARGETS (HIGHEST IMPACT)
# ============================================================================

def create_momentum_targets_stefan_jansen(df, lookforward_periods=12):
    """
    Stefan Jansen Chapter 4: Create momentum-specific targets
    
    Your current targets probably look for reversals.
    Momentum targets should look for CONTINUATION.
    """
    # Forward returns for momentum prediction
    df['forward_return_1h'] = df['close'].shift(-1) / df['close'] - 1
    df['forward_return_4h'] = df['close'].shift(-4) / df['close'] - 1
    df['forward_return_12h'] = df['close'].shift(-12) / df['close'] - 1
    
    # Momentum continuation target (key difference!)
    df['momentum_strength'] = df['close'].pct_change(24)  # 24h momentum
    df['momentum_continues'] = (
        (df['momentum_strength'] > 0.02) &  # Strong upward momentum
        (df['forward_return_4h'] > 0.01)    # Continues upward
    ).astype(int)
    
    # Breakout continuation target
    df['breakout_target'] = (
        (df['close'] > df['close'].rolling(20).max().shift(1)) &  # New highs
        (df['forward_return_4h'] > 0.005)  # Continues after breakout
    ).astype(int)
    
    return df

# ============================================================================
# 2. 🧠 USE TIME-SERIES AWARE MODELS (Stefan Jansen Chapter 6)
# ============================================================================

"""
PROBLEM: You're using RandomForest/Ridge which ignore time order
SOLUTION: Use LightGBM, XGBoost, or LSTM that understand time sequences
"""

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def create_momentum_model_stefan_jansen():
    """
    Stefan Jansen recommendation: Use gradient boosting for time series
    """
    model = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=0,
        random_state=42
    )
    return model

# ============================================================================
# 3. 📊 MOMENTUM-SPECIFIC FEATURES (Stefan Jansen Chapter 4)
# ============================================================================

def calculate_stefan_jansen_momentum_features(df):
    """
    Stefan Jansen Chapter 4: Factor-based features for momentum
    """
    # === MOMENTUM FACTORS ===
    df['momentum_1w'] = df['close'] / df['close'].shift(7*24) - 1  # 1 week
    df['momentum_1m'] = df['close'] / df['close'].shift(30*24) - 1  # 1 month
    df['momentum_3m'] = df['close'] / df['close'].shift(90*24) - 1  # 3 months
    
    # === ACCELERATION (2nd derivative) ===
    df['price_acceleration'] = df['close'].pct_change().diff()
    df['volume_acceleration'] = df['volume'].pct_change().diff()
    
    # === TREND STRENGTH ===
    df['trend_strength_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['trend_strength_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
    
    # === MOMENTUM QUALITY ===
    df['momentum_consistency'] = df['close'].pct_change().rolling(20).apply(
        lambda x: (x > 0).sum() / len(x)  # % of positive days
    )
    
    # === VOLUME CONFIRMATION ===
    df['volume_price_trend'] = (df['volume'] * df['close'].pct_change()).rolling(10).sum()
    
    return df

# ============================================================================
# 4. 🎯 WALK-FORWARD VALIDATION (Stefan Jansen Chapter 7)
# ============================================================================

def walk_forward_validation_stefan_jansen(df, model, features, target, window_size=1000):
    """
    Stefan Jansen Chapter 7: Proper time series validation
    
    NEVER use random train/test split for time series!
    """
    predictions = []
    
    for i in range(window_size, len(df)):
        # Train on past data only
        train_data = df.iloc[i-window_size:i]
        test_data = df.iloc[i:i+1]
        
        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        
        # Train and predict
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_test)[0][1]
        predictions.append(pred)
    
    return predictions

# ============================================================================
# 5. 🎪 ENSEMBLE METHODS (Stefan Jansen Chapter 8)
# ============================================================================

def create_momentum_ensemble_stefan_jansen():
    """
    Stefan Jansen Chapter 8: Combine multiple models for better predictions
    """
    from sklearn.ensemble import VotingClassifier
    
    # Different models for different aspects
    momentum_model = LGBMClassifier(random_state=42)
    breakout_model = XGBClassifier(random_state=43) 
    volume_model = LGBMClassifier(random_state=44)
    
    ensemble = VotingClassifier([
        ('momentum', momentum_model),
        ('breakout', breakout_model),
        ('volume', volume_model)
    ], voting='soft')
    
    return ensemble

# ============================================================================
# 6. 📈 TRADING-SPECIFIC EVALUATION (Stefan Jansen Chapter 5)
# ============================================================================

def evaluate_trading_performance_stefan_jansen(predictions, returns, transaction_costs=0.001):
    """
    Stefan Jansen Chapter 5: Evaluate models based on TRADING performance, not accuracy
    """
    # Convert predictions to signals
    signals = (predictions > 0.5).astype(int)
    
    # Calculate strategy returns
    strategy_returns = signals.shift(1) * returns - transaction_costs * signals.diff().abs()
    
    # Trading-specific metrics
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365*24)
    max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
    hit_rate = (strategy_returns > 0).mean()
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'total_return': strategy_returns.sum()
    }

# ============================================================================
# 7. 🔄 ONLINE LEARNING (Stefan Jansen Chapter 9)
# ============================================================================

def create_online_learning_system_stefan_jansen():
    """
    Stefan Jansen Chapter 9: Continuously update models with new data
    """
    from sklearn.linear_model import SGDClassifier
    
    # Online learning model that updates with each new data point
    online_model = SGDClassifier(
        loss='log',  # For probability predictions
        learning_rate='adaptive',
        eta0=0.01,
        random_state=42
    )
    
    return online_model

# ============================================================================
# 🎯 IMMEDIATE ACTION PLAN FOR YOUR SYSTEM
# ============================================================================

"""
STEP 1: Fix your training targets (highest impact)
- Change from predicting reversals to predicting continuations
- Use create_momentum_targets_stefan_jansen()

STEP 2: Switch to time-series aware models
- Replace RandomForest with LightGBM
- Use create_momentum_model_stefan_jansen()

STEP 3: Add momentum-specific features
- Use calculate_stefan_jansen_momentum_features()
- Focus on trend strength and momentum quality

STEP 4: Implement proper validation
- Use walk_forward_validation_stefan_jansen()
- Never use random train/test split

STEP 5: Evaluate on trading metrics
- Use evaluate_trading_performance_stefan_jansen()
- Optimize for Sharpe ratio, not accuracy

NEXT IMPROVEMENTS:
- Ensemble methods (combine multiple models)
- Online learning (continuous updates)
- Alternative data (sentiment, order book)
"""

# ============================================================================
# 🚀 QUICK FIX FOR YOUR CURRENT SYSTEM
# ============================================================================

def quick_momentum_fix(symbol, granularity=3600):
    """
    Quick fix to make your current system more momentum-friendly
    """
    try:
        # Get data
        from maybe import get_coinbase_data
        df = get_coinbase_data(symbol, granularity, days=90)
        
        # Add momentum features
        df = calculate_stefan_jansen_momentum_features(df)
        
        # Create momentum targets
        df = create_momentum_targets_stefan_jansen(df)
        
        # Use momentum-aware model
        model = create_momentum_model_stefan_jansen()
        
        # Train on momentum continuation
        feature_cols = ['momentum_1w', 'momentum_1m', 'trend_strength_20', 'momentum_consistency']
        X = df[feature_cols].dropna()
        y = df['momentum_continues'].iloc[:len(X)]
        
        # Simple train/predict (use walk-forward in production)
        model.fit(X[:-10], y[:-10])
        prediction = model.predict_proba(X[-1:])
        
        return prediction[0][1]  # Probability of momentum continuation
        
    except Exception as e:
        print(f"Error in momentum fix: {e}")
        return 0.5

if __name__ == "__main__":
    print("📚 Stefan Jansen ML Improvements Loaded!")
    print("🎯 Use these functions to upgrade your momentum trading ML")
    
    # Test the quick fix
    try:
        prob = quick_momentum_fix('BTC-USD')
        print(f"✅ Momentum continuation probability: {prob:.1%}")
    except Exception as e:
        print(f"❌ Quick fix test failed: {e}") 