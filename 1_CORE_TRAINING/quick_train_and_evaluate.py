#!/usr/bin/env python3
"""
Quick Training and Evaluation Script
Trains models for a few symbols and provides comprehensive evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import train_price_prediction_models, get_coinbase_data, calculate_indicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_evaluation():
    """Quick training and evaluation"""
    logger.info("🚀 Starting quick train and evaluate session")
    
    # Symbols to train and evaluate
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    logger.info(f"📊 Training models for {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Train models for each symbol
    for symbol in symbols:
        logger.info(f"\n🔥 Training models for {symbol}...")
        try:
            # Train all timeframe models
            result = train_price_prediction_models(symbol, granularity=3600)
            if result:
                logger.info(f"✅ Successfully trained models for {symbol}")
            else:
                logger.warning(f"❌ Failed to train models for {symbol}")
        except Exception as e:
            logger.error(f"💥 Error training {symbol}: {str(e)}")
    
    # Quick evaluation without models directory dependency
    logger.info("\n📊 Performing quick evaluation...")
    evaluation_results = {}
    
    for symbol in symbols:
        logger.info(f"\n🔍 Evaluating {symbol}...")
        try:
            # Get data for evaluation
            df = get_coinbase_data(symbol, 3600, days=30)
            if df is None or df.empty:
                logger.warning(f"No data for {symbol}")
                continue
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            # Simple statistical analysis
            price_changes = df['close'].pct_change().dropna() * 100
            volatility = price_changes.std()
            mean_return = price_changes.mean()
            
            # Direction changes
            directions = np.sign(price_changes)
            direction_changes = (directions != directions.shift()).sum()
            direction_persistence = 1 - (direction_changes / len(directions))
            
            # Trend analysis
            sma_20 = df['close'].rolling(20).mean()
            trend_strength = ((df['close'] - sma_20) / sma_20 * 100).abs().mean()
            
            # Predictability indicators
            autocorr_1 = price_changes.autocorr(lag=1)
            autocorr_5 = price_changes.autocorr(lag=5)
            
            # RSI analysis
            rsi_values = df['rsi'].dropna()
            rsi_extreme_signals = ((rsi_values < 30) | (rsi_values > 70)).sum()
            
            results = {
                'symbol': symbol,
                'data_points': len(df),
                'volatility_pct': round(volatility, 2),
                'mean_return_pct': round(mean_return, 4),
                'direction_persistence': round(direction_persistence, 3),
                'trend_strength_pct': round(trend_strength, 2),
                'autocorr_1_day': round(autocorr_1, 3) if not pd.isna(autocorr_1) else 0,
                'autocorr_5_day': round(autocorr_5, 3) if not pd.isna(autocorr_5) else 0,
                'rsi_extreme_signals': rsi_extreme_signals,
                'predictability_score': round((abs(autocorr_1) + direction_persistence) / 2, 3)
            }
            
            evaluation_results[symbol] = results
            
            logger.info(f"📈 {symbol} Summary:")
            logger.info(f"   Data Points: {results['data_points']}")
            logger.info(f"   Volatility: {results['volatility_pct']:.2f}%")
            logger.info(f"   Mean Return: {results['mean_return_pct']:.4f}%")
            logger.info(f"   Direction Persistence: {results['direction_persistence']:.3f}")
            logger.info(f"   Predictability Score: {results['predictability_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {symbol}: {str(e)}")
    
    # Create summary report
    if evaluation_results:
        logger.info("\n" + "="*60)
        logger.info("📊 QUICK EVALUATION SUMMARY")
        logger.info("="*60)
        
        df_results = pd.DataFrame(evaluation_results.values())
        
        logger.info(f"\nAverage Volatility: {df_results['volatility_pct'].mean():.2f}%")
        logger.info(f"Average Predictability: {df_results['predictability_score'].mean():.3f}")
        logger.info(f"Most Predictable: {df_results.loc[df_results['predictability_score'].idxmax(), 'symbol']}")
        logger.info(f"Most Volatile: {df_results.loc[df_results['volatility_pct'].idxmax(), 'symbol']}")
        
        # Print detailed table
        print("\n📋 DETAILED RESULTS:")
        print(df_results.to_string(index=False))
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"quick_evaluation_results_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)
        logger.info(f"\n💾 Results saved to: {results_path}")
        
        logger.info("="*60)
    
    return evaluation_results

def analyze_trading_potential():
    """Analyze trading potential based on market characteristics"""
    logger.info("\n🎯 TRADING POTENTIAL ANALYSIS")
    logger.info("="*60)
    
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"]
    
    trading_analysis = []
    
    for symbol in symbols:
        try:
            df = get_coinbase_data(symbol, 3600, days=14)  # 2 weeks of hourly data
            if df is None or df.empty:
                continue
            
            df = calculate_indicators(df, symbol=symbol)
            
            # Calculate trading metrics
            price_changes = df['close'].pct_change().dropna() * 100
            
            # Volatility analysis
            daily_volatility = price_changes.std()
            
            # Trend analysis  
            sma_20 = df['close'].rolling(20).mean()
            current_trend = (df['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100
            
            # RSI analysis
            current_rsi = df['rsi'].iloc[-1]
            rsi_signal = "BUY" if current_rsi < 35 else "SELL" if current_rsi > 65 else "NEUTRAL"
            
            # MACD analysis
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            macd_signal = "BUY" if current_macd > current_signal else "SELL"
            
            # Price momentum
            momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100
            momentum_24 = (df['close'].iloc[-1] - df['close'].iloc[-25]) / df['close'].iloc[-25] * 100
            
            # Trading score (0-100)
            volatility_score = min(daily_volatility * 10, 50)  # More volatile = higher score up to 50
            trend_score = min(abs(current_trend) * 2, 30)  # Stronger trend = higher score up to 30
            signal_score = 20 if (rsi_signal == macd_signal and rsi_signal != "NEUTRAL") else 10
            
            trading_score = volatility_score + trend_score + signal_score
            
            # Recommendation
            if trading_score >= 70:
                recommendation = "HIGH_POTENTIAL"
            elif trading_score >= 50:
                recommendation = "MODERATE_POTENTIAL"
            else:
                recommendation = "LOW_POTENTIAL"
            
            analysis = {
                'symbol': symbol,
                'current_price': round(df['close'].iloc[-1], 4),
                'daily_volatility': round(daily_volatility, 2),
                'current_trend_pct': round(current_trend, 2),
                'current_rsi': round(current_rsi, 1),
                'momentum_5h_pct': round(momentum_5, 2),
                'momentum_24h_pct': round(momentum_24, 2),
                'rsi_signal': rsi_signal,
                'macd_signal': macd_signal,
                'trading_score': round(trading_score, 1),
                'recommendation': recommendation
            }
            
            trading_analysis.append(analysis)
            
            logger.info(f"📊 {symbol}:")
            logger.info(f"   Price: ${analysis['current_price']}")
            logger.info(f"   Volatility: {analysis['daily_volatility']:.2f}%")
            logger.info(f"   Trend: {analysis['current_trend_pct']:+.2f}%")
            logger.info(f"   RSI: {analysis['current_rsi']:.1f} ({analysis['rsi_signal']})")
            logger.info(f"   Trading Score: {analysis['trading_score']:.1f}/100 ({analysis['recommendation']})")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {str(e)}")
    
    if trading_analysis:
        df_trading = pd.DataFrame(trading_analysis)
        
        # Print summary
        logger.info(f"\n🏆 TOP TRADING OPPORTUNITIES:")
        top_opportunities = df_trading.nlargest(3, 'trading_score')
        for _, row in top_opportunities.iterrows():
            logger.info(f"   {row['symbol']}: {row['trading_score']:.1f}/100 ({row['recommendation']})")
        
        # Save trading analysis
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        trading_path = f"trading_analysis_{timestamp}.csv"
        df_trading.to_csv(trading_path, index=False)
        logger.info(f"\n💾 Trading analysis saved to: {trading_path}")
        
        print("\n📋 DETAILED TRADING ANALYSIS:")
        print(df_trading.to_string(index=False))
    
    return trading_analysis

def main():
    """Main function"""
    logger.info("🚀 Starting Quick Train and Evaluate Session")
    
    # Perform quick evaluation
    evaluation_results = quick_evaluation()
    
    # Analyze trading potential
    trading_analysis = analyze_trading_potential()
    
    logger.info("\n🎉 Quick evaluation completed!")
    logger.info("💡 This provides baseline analysis. For full ML evaluation, train more models first.")
    
    return evaluation_results, trading_analysis

if __name__ == "__main__":
    main() 