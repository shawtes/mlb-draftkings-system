#!/usr/bin/env python3
"""
Enhanced ML Predictor with Price Targets and Profitability Analysis
Instead of just BUY/SELL signals, predicts exact prices and calculates expected profit
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Import existing functions
from maybe import (
    get_coinbase_data,
    calculate_indicators,
    get_cached_symbols
)

# Enhanced ML Configuration
ENHANCED_MODELS_DIR = 'enhanced_models'
TRADING_FEES = {
    'limit_order': 0.005,  # 0.5% for limit orders (better than market orders)
    'market_order': 0.006,  # 0.6% for market orders
}
MIN_PROFIT_THRESHOLD = 0.02  # 2% minimum profit after fees
PREDICTION_HORIZONS = [1, 4, 24]  # 1h, 4h, 24h predictions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import meta classifier after logger is defined
try:
    from meta_trading_classifier import get_meta_decision, meta_classifier
    META_CLASSIFIER_AVAILABLE = True
    logger.info("[OK] Meta trading classifier imported successfully")
except ImportError as e:
    META_CLASSIFIER_AVAILABLE = False
    logger.error(f"[ERROR] Meta classifier not available: {e}")
    raise ImportError(f"Meta classifier is required but not available: {e}")

# Create models directory
os.makedirs(ENHANCED_MODELS_DIR, exist_ok=True)

class EnhancedMLPredictor:
    """
    Enhanced ML system that predicts exact price targets and calculates profitability
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'RSI', '%K', '%D', 'ATR', 'EMA12', 'EMA26', 'MACD', 'Signal_Line',
            'MA20', 'OBV', 'lag_1', 'lag_2', 'lag_3', 'rolling_std_10',
            'volume_sma', 'price_change_1h', 'price_change_4h', 'price_change_24h',
            'bb_position', 'rsi_momentum', 'volume_ratio'
        ]
    
    def prepare_enhanced_features(self, df):
        """Prepare enhanced feature set for price prediction"""
        try:
            if df.empty or len(df) < 50:
                return None
            
            # Start with basic indicators
            df = calculate_indicators(df)
            
            # Add enhanced features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['price_change_1h'] = df['close'].pct_change(1)
            df['price_change_4h'] = df['close'].pct_change(4) 
            df['price_change_24h'] = df['close'].pct_change(24)
            
            # Bollinger Band position (0-1, where 0.5 is middle)
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # RSI momentum (rate of change of RSI)
            df['rsi_momentum'] = df['RSI'].diff()
            
            # Volume ratio (current volume vs average)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill NaN values
            for col in self.feature_columns:
                if col in df.columns:
                    if col in ['RSI', '%K', '%D']:
                        df[col] = df[col].fillna(50.0)
                    elif col == 'bb_position':
                        df[col] = df[col].fillna(0.5)
                    elif 'change' in col or 'momentum' in col:
                        df[col] = df[col].fillna(0.0)
                    elif 'ratio' in col:
                        df[col] = df[col].fillna(1.0)
                    else:
                        df[col] = df[col].fillna(method='ffill').fillna(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def create_price_targets(self, df, horizons=[1, 4, 24]):
        """Create price targets for different time horizons"""
        targets = {}
        
        for h in horizons:
            # Future price target
            targets[f'target_{h}h'] = df['close'].shift(-h)
            # Price change percentage
            targets[f'change_{h}h'] = (df['close'].shift(-h) / df['close'] - 1) * 100
        
        return targets
    
    def train_enhanced_model(self, symbol, days=60):
        """Train enhanced model for price prediction"""
        try:
            logger.info(f"🤖 Training enhanced price prediction model for {symbol}...")
            
            # Get more historical data for training
            df = get_coinbase_data(symbol, granularity=3600, days=days)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} records")
                return None
            
            # Prepare features
            df = self.prepare_enhanced_features(df)
            if df is None:
                return None
            
            # Create targets for different horizons
            targets = self.create_price_targets(df, PREDICTION_HORIZONS)
            
            # Add targets to dataframe
            for target_name, target_values in targets.items():
                df[target_name] = target_values
            
            # Remove rows with NaN targets
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Too few complete records for {symbol}: {len(df)}")
                return None
            
            # Prepare feature matrix
            X = df[self.feature_columns]
            
            # Train models for each horizon
            models = {}
            performance = {}
            
            for horizon in PREDICTION_HORIZONS:
                target_col = f'target_{horizon}h'
                change_col = f'change_{horizon}h'
                
                if target_col in df.columns and change_col in df.columns:
                    # Price target model
                    y_price = df[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
                    
                    price_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    price_model.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = price_model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    models[f'price_{horizon}h'] = price_model
                    performance[f'price_{horizon}h'] = {'mse': mse, 'r2': r2}
                    
                    logger.info(f"✅ {symbol} {horizon}h price model: R² = {r2:.3f}, MSE = {mse:.3f}")
            
            # Save models
            model_file = os.path.join(ENHANCED_MODELS_DIR, f"{symbol.replace('-', '')}_enhanced.pkl")
            joblib.dump(joblib.dump({
                'models': models, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/performance': performance,
                'feature_columns': self.feature_columns,
                'trained_date': datetime.now(),
                'symbol': symbol
            }, model_file)
            
            self.models[symbol] = models
            logger.info(f"🎯 Enhanced model trained and saved for {symbol}")
            return models
            
        except Exception as e:
            logger.error(f"Error training enhanced model for {symbol}: {e}")
            return None
    
    def predict_price_targets(self, symbol, current_price=None):
        """Predict price targets for different horizons"""
        try:
            # Load model if not in memory
            if symbol not in self.models:
                model_file = os.path.join(ENHANCED_MODELS_DIR, f"{symbol.replace('-', '')}_enhanced.pkl")
                if os.path.exists(model_file):
                    model_data = joblib.load(model_file)
                    self.models[symbol] = model_data['models']
                else:
                    logger.info(f"No enhanced model found for {symbol}, training new one...")
                    models = self.train_enhanced_model(symbol)
                    if not models:
                        return None
            
            # Get current data
            df = get_coinbase_data(symbol, granularity=3600, days=3)
            if df is None or df.empty:
                logger.error(f"No current data for {symbol}")
                return None
            
            # Prepare features
            df = self.prepare_enhanced_features(df)
            if df is None:
                return None
            
            # Get latest features
            latest_features = df[self.feature_columns].iloc[-1:].fillna(0)
            
            if current_price is None:
                current_price = df['close'].iloc[-1]
            
            # Make predictions
            predictions = {}
            
            for horizon in PREDICTION_HORIZONS:
                model_key = f'price_{horizon}h'
                if model_key in self.models[symbol]:
                    model = self.models[symbol][model_key]
                    predicted_price = model.predict(latest_features)[0]
                    
                    predictions[f'{horizon}h'] = {
                        'target_price': predicted_price,
                        'expected_change': (predicted_price / current_price - 1) * 100,
                        'horizon_hours': horizon
                    }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting price targets for {symbol}: {e}")
            return None
    
    def calculate_trade_profitability(self, symbol, predictions, current_price, position_size=1000):
        """Calculate expected profitability including trading fees"""
        try:
            if not predictions:
                return None
            
            results = {}
            
            for horizon, pred in predictions.items():
                target_price = pred['target_price']
                expected_change = pred['expected_change']
                
                # Calculate fees for round trip (buy + sell)
                buy_fee = position_size * TRADING_FEES['limit_order']
                sell_fee = (position_size * (target_price / current_price)) * TRADING_FEES['limit_order']
                total_fees = buy_fee + sell_fee
                
                # Calculate gross profit/loss
                gross_profit = position_size * (target_price / current_price - 1)
                
                # Calculate net profit after fees
                net_profit = gross_profit - total_fees
                net_profit_pct = (net_profit / position_size) * 100
                
                # Risk-reward calculation
                fee_pct = (total_fees / position_size) * 100
                breakeven_change = fee_pct  # Need this much gain just to break even
                
                results[horizon] = {
                    'target_price': target_price,
                    'expected_change_pct': expected_change,
                    'gross_profit': gross_profit,
                    'total_fees': total_fees,
                    'net_profit': net_profit,
                    'net_profit_pct': net_profit_pct,
                    'fee_pct': fee_pct,
                    'breakeven_change': breakeven_change,
                    'is_profitable': net_profit_pct > MIN_PROFIT_THRESHOLD,
                    'profit_margin': net_profit_pct - breakeven_change
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating profitability for {symbol}: {e}")
            return None
    
    def should_trade(self, symbol, position_size=1000):
        """Complete trading decision with price targets and profitability analysis"""
        try:
            # Get current price
            df = get_coinbase_data(symbol, granularity=3600, days=1)
            if df is None or df.empty:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Get price predictions
            predictions = self.predict_price_targets(symbol, current_price)
            if not predictions:
                return None
            
            # Calculate profitability
            profitability = self.calculate_trade_profitability(symbol, predictions, current_price, position_size)
            if not profitability:
                return None
            
            # Find best opportunity
            best_opportunity = None
            best_profit_margin = -float('inf')
            
            for horizon, analysis in profitability.items():
                if analysis['is_profitable'] and analysis['profit_margin'] > best_profit_margin:
                    best_profit_margin = analysis['profit_margin']
                    best_opportunity = {
                        'horizon': horizon,
                        'decision': 'BUY' if analysis['expected_change_pct'] > analysis['breakeven_change'] else 'HOLD',
                        'confidence': min(95, 60 + analysis['profit_margin'] * 3),  # Scale confidence by profit margin
                        'target_price': analysis['target_price'],
                        'expected_profit_pct': analysis['net_profit_pct'],
                        'expected_profit_usd': analysis['net_profit'],
                        'total_fees': analysis['total_fees'],
                        'time_horizon': analysis,
                        'analysis': analysis
                    }
            
            # Apply meta classifier decision (required)
            if best_opportunity:
                try:
                    # Prepare data for meta classifier
                    df_indicators = calculate_indicators(df)
                    latest_row = df_indicators.iloc[-1]
                    
                    meta_input = {
                        'predicted_price': best_opportunity['target_price'],
                        'current_price': current_price,
                        'confidence': best_opportunity['confidence'] / 100.0,
                        'rsi': latest_row.get('RSI', 50),
                        'bb_position': 0.5,  # Would need Bollinger Bands calculation
                        'macd_signal': latest_row.get('MACD', 0) - latest_row.get('Signal_Line', 0),
                        'volume_ratio': 1.0,  # Would need volume analysis
                        'volatility': df['close'].pct_change().std() if len(df) > 1 else 0.02,
                        'trend_strength': (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) > 10 else 0,
                        'support_resistance_score': 0.0  # Would need S/R level calculation
                    }
                    
                    # Get meta classifier decision (required)
                    meta_decision = get_meta_decision(meta_input)
                    
                    # Override the original decision with meta classifier
                    if meta_decision['decision'] != 'HOLD':
                        best_opportunity['decision'] = meta_decision['decision']
                        best_opportunity['meta_confidence'] = meta_decision['confidence']
                        best_opportunity['meta_decision'] = meta_decision
                        best_opportunity['final_confidence'] = (best_opportunity['confidence'] + meta_decision['confidence'] * 100) / 2
                        
                        logger.info(f"Meta classifier override for {symbol}: {meta_decision['decision']} "
                                   f"(confidence: {meta_decision['confidence']:.3f})")
                    else:
                        best_opportunity['meta_decision'] = meta_decision
                        logger.info(f"Meta classifier confirms HOLD for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Meta classifier error for {symbol}: {e}")
                    raise RuntimeError(f"Meta classifier required but failed: {e}")
            
            return best_opportunity
            
        except Exception as e:
            logger.error(f"Error making trade decision for {symbol}: {e}")
            return None

# Global predictor instance
enhanced_predictor = EnhancedMLPredictor()

def make_enhanced_ml_decision(symbol, position_size=1000):
    """
    Enhanced ML decision with exact price targets and profitability analysis
    
    Returns:
        dict: {
            'decision': 'BUY'/'HOLD'/'SELL',
            'confidence': float (0-100),
            'target_price': float,
            'expected_profit_pct': float,
            'expected_profit_usd': float,
            'time_horizon': str,
            'analysis': dict with detailed breakdown
        }
    """
    return enhanced_predictor.should_trade(symbol, position_size)

def scan_market_with_profit_analysis(symbols=None, position_size=1000):
    """Scan market using enhanced ML with profit analysis"""
    try:
        if symbols is None:
            symbols = get_cached_symbols()
            symbols = [s for s in symbols if s.endswith('-USD')][:10]  # Limit for demo
        
        opportunities = []
        
        for symbol in symbols:
            logger.info(f"🔍 Analyzing {symbol} with enhanced ML...")
            decision = make_enhanced_ml_decision(symbol, position_size)
            
            if decision and decision['decision'] == 'BUY':
                opportunities.append({
                    'symbol': symbol,
                    'decision': decision['decision'],
                    'confidence': decision['confidence'],
                    'target_price': decision['target_price'],
                    'expected_profit_pct': decision['expected_profit_pct'],
                    'expected_profit_usd': decision['expected_profit_usd'],
                    'time_horizon': decision['time_horizon'],
                    'profit_margin': decision['analysis']['profit_margin']
                })
                
                logger.info(f"🎯 {symbol}: {decision['decision']} - "
                          f"Target: ${decision['target_price']:.4f}, "
                          f"Profit: {decision['expected_profit_pct']:.2f}% (${decision['expected_profit_usd']:.2f})")
        
        # Sort by profit margin
        opportunities.sort(key=lambda x: x['profit_margin'], reverse=True)
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Error scanning market with profit analysis: {e}")
        return []

if __name__ == "__main__":
    # Test the enhanced ML system
    logger.info("🚀 Testing Enhanced ML Predictor")
    
    # Test with a few symbols
    test_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for symbol in test_symbols:
        logger.info(f"\n📊 Testing {symbol}...")
        decision = make_enhanced_ml_decision(symbol, position_size=1000)
        
        if decision:
            print(f"\n{'='*50}")
            print(f"📈 {symbol} Enhanced ML Analysis")
            print(f"{'='*50}")
            print(f"🎯 Decision: {decision['decision']}")
            print(f"🔮 Confidence: {decision['confidence']:.1f}%")
            print(f"💰 Target Price: ${decision['target_price']:.4f}")
            print(f"📈 Expected Profit: {decision['expected_profit_pct']:.2f}% (${decision['expected_profit_usd']:.2f})")
            print(f"⏱️ Time Horizon: {decision['time_horizon']}")
            print(f"💳 Total Fees: ${decision['analysis']['total_fees']:.2f}")
            print(f"📊 Profit Margin: {decision['analysis']['profit_margin']:.2f}%")
        else:
            print(f"❌ No decision available for {symbol}") 