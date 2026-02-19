#!/usr/bin/env python3
"""
Simple ML-like Evaluation without Training
Provides comprehensive statistical and predictive analysis
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

from maybe import get_coinbase_data, calculate_indicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_predictive_metrics(df, symbol):
    """Calculate comprehensive predictive and statistical metrics"""
    try:
        # Price changes
        price_changes = df['close'].pct_change().dropna() * 100
        
        # Basic statistics
        mean_return = price_changes.mean()
        volatility = price_changes.std()
        skewness = price_changes.skew()
        kurtosis = price_changes.kurtosis()
        
        # Predictability metrics
        autocorr_1 = price_changes.autocorr(lag=1)
        autocorr_5 = price_changes.autocorr(lag=5)
        autocorr_24 = price_changes.autocorr(lag=24)
        
        # Direction prediction accuracy (using simple rules)
        directions = np.sign(price_changes)
        
        # RSI-based predictions
        rsi_predictions = []
        rsi_actual = []
        for i in range(len(df) - 1):
            if i >= 14:  # Need 14 periods for RSI
                current_rsi = df['rsi'].iloc[i]
                next_direction = directions.iloc[i + 1]
                
                # Simple RSI strategy: BUY when RSI < 35, SELL when RSI > 65
                if current_rsi < 35:
                    rsi_pred = 1  # Predict UP
                elif current_rsi > 65:
                    rsi_pred = -1  # Predict DOWN
                else:
                    rsi_pred = 0  # Predict NEUTRAL
                
                rsi_predictions.append(rsi_pred)
                rsi_actual.append(next_direction)
        
        # Calculate RSI accuracy
        rsi_predictions = np.array(rsi_predictions)
        rsi_actual = np.array(rsi_actual)
        
        rsi_accuracy = 0
        if len(rsi_predictions) > 0:
            correct_predictions = np.sum(np.sign(rsi_predictions) == np.sign(rsi_actual))
            rsi_accuracy = correct_predictions / len(rsi_predictions) * 100
        
        # MACD-based predictions
        macd_predictions = []
        macd_actual = []
        for i in range(len(df) - 1):
            if i >= 26:  # Need 26 periods for MACD
                current_macd = df['macd'].iloc[i]
                current_signal = df['macd_signal'].iloc[i]
                next_direction = directions.iloc[i + 1]
                
                # MACD strategy: BUY when MACD > Signal, SELL when MACD < Signal
                if current_macd > current_signal:
                    macd_pred = 1  # Predict UP
                else:
                    macd_pred = -1  # Predict DOWN
                
                macd_predictions.append(macd_pred)
                macd_actual.append(next_direction)
        
        # Calculate MACD accuracy
        macd_predictions = np.array(macd_predictions)
        macd_actual = np.array(macd_actual)
        
        macd_accuracy = 0
        if len(macd_predictions) > 0:
            correct_predictions = np.sum(np.sign(macd_predictions) == np.sign(macd_actual))
            macd_accuracy = correct_predictions / len(macd_predictions) * 100
        
        # Moving average crossover predictions
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        ma_predictions = []
        ma_actual = []
        for i in range(len(df) - 1):
            if i >= 50:  # Need 50 periods for MA50
                current_sma20 = sma_20.iloc[i]
                current_sma50 = sma_50.iloc[i]
                next_direction = directions.iloc[i + 1]
                
                # MA strategy: BUY when SMA20 > SMA50, SELL when SMA20 < SMA50
                if current_sma20 > current_sma50:
                    ma_pred = 1  # Predict UP
                else:
                    ma_pred = -1  # Predict DOWN
                
                ma_predictions.append(ma_pred)
                ma_actual.append(next_direction)
        
        # Calculate MA accuracy
        ma_predictions = np.array(ma_predictions)
        ma_actual = np.array(ma_actual)
        
        ma_accuracy = 0
        if len(ma_predictions) > 0:
            correct_predictions = np.sum(np.sign(ma_predictions) == np.sign(ma_actual))
            ma_accuracy = correct_predictions / len(ma_predictions) * 100
        
        # Price prediction using linear trend
        price_values = df['close'].values
        trend_predictions = []
        trend_actual = []
        
        for i in range(20, len(price_values) - 1):  # Use 20-period trend
            # Fit linear trend to last 20 periods
            x = np.arange(20)
            y = price_values[i-19:i+1]
            
            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]
            predicted_price = price_values[i] + slope
            actual_price = price_values[i + 1]
            
            # Calculate percentage prediction error
            pred_error = abs(predicted_price - actual_price) / actual_price * 100
            
            trend_predictions.append(pred_error)
            trend_actual.append(actual_price)
        
        # Calculate trend prediction metrics
        trend_mae = np.mean(trend_predictions) if trend_predictions else 0
        trend_rmse = np.sqrt(np.mean(np.array(trend_predictions) ** 2)) if trend_predictions else 0
        
        # Volatility prediction
        vol_window = 20
        vol_predictions = []
        vol_actual = []
        
        for i in range(vol_window, len(price_changes) - 1):
            # Predict next volatility based on historical
            hist_vol = price_changes.iloc[i-vol_window:i].std()
            actual_vol = abs(price_changes.iloc[i + 1])
            
            vol_predictions.append(hist_vol)
            vol_actual.append(actual_vol)
        
        vol_correlation = np.corrcoef(vol_predictions, vol_actual)[0, 1] if len(vol_predictions) > 1 else 0
        
        # Overall predictability score
        predictability_components = [
            abs(autocorr_1) * 25,  # 25% weight
            abs(autocorr_5) * 15,  # 15% weight
            rsi_accuracy * 0.3,    # 30% weight
            macd_accuracy * 0.2,   # 20% weight
            ma_accuracy * 0.1      # 10% weight
        ]
        
        predictability_score = sum([x for x in predictability_components if not np.isnan(x)])
        
        # Market efficiency score (lower = more efficient/harder to predict)
        efficiency_score = 100 - predictability_score
        
        metrics = {
            'symbol': symbol,
            'data_points': len(df),
            
            # Basic statistics
            'mean_return_pct': round(mean_return, 4),
            'volatility_pct': round(volatility, 2),
            'skewness': round(skewness, 3),
            'kurtosis': round(kurtosis, 3),
            
            # Autocorrelation (predictability indicators)
            'autocorr_1h': round(autocorr_1, 3) if not pd.isna(autocorr_1) else 0,
            'autocorr_5h': round(autocorr_5, 3) if not pd.isna(autocorr_5) else 0,
            'autocorr_24h': round(autocorr_24, 3) if not pd.isna(autocorr_24) else 0,
            
            # Directional accuracy metrics (simulating ML performance)
            'rsi_directional_accuracy': round(rsi_accuracy, 2),
            'macd_directional_accuracy': round(macd_accuracy, 2),
            'ma_directional_accuracy': round(ma_accuracy, 2),
            
            # Price prediction metrics
            'trend_mae_pct': round(trend_mae, 3),
            'trend_rmse_pct': round(trend_rmse, 3),
            'volatility_correlation': round(vol_correlation, 3),
            
            # Composite scores
            'predictability_score': round(predictability_score, 2),
            'market_efficiency_score': round(efficiency_score, 2),
            
            # Trading viability assessment
            'trading_viability': assess_trading_viability(rsi_accuracy, macd_accuracy, volatility, predictability_score)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
        return None

def assess_trading_viability(rsi_acc, macd_acc, volatility, pred_score):
    """Assess trading viability based on metrics"""
    avg_accuracy = (rsi_acc + macd_acc) / 2
    
    if avg_accuracy > 58 and volatility > 2.5 and pred_score > 55:
        return "HIGHLY_VIABLE"
    elif avg_accuracy > 55 and volatility > 2.0 and pred_score > 45:
        return "VIABLE"
    elif avg_accuracy > 52 and volatility > 1.5:
        return "MARGINAL"
    else:
        return "NOT_VIABLE"

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Comprehensive ML-Style Evaluation")
    
    # Symbols to evaluate
    symbols = [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
        "MATIC-USD", "LINK-USD", "UNI-USD", "AVAX-USD", "LTC-USD"
    ]
    
    logger.info(f"📊 Evaluating {len(symbols)} symbols with comprehensive metrics")
    
    all_metrics = []
    successful_evaluations = 0
    
    for symbol in symbols:
        logger.info(f"\n🔍 Evaluating {symbol}...")
        try:
            # Get data
            df = get_coinbase_data(symbol, 3600, days=30)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            # Calculate indicators
            df = calculate_indicators(df, symbol=symbol)
            
            # Calculate comprehensive metrics
            metrics = calculate_predictive_metrics(df, symbol)
            
            if metrics:
                all_metrics.append(metrics)
                successful_evaluations += 1
                
                logger.info(f"📈 {symbol} Results:")
                logger.info(f"   Volatility: {metrics['volatility_pct']:.2f}%")
                logger.info(f"   RSI Accuracy: {metrics['rsi_directional_accuracy']:.1f}%")
                logger.info(f"   MACD Accuracy: {metrics['macd_directional_accuracy']:.1f}%")
                logger.info(f"   Predictability: {metrics['predictability_score']:.1f}/100")
                logger.info(f"   Trading Viability: {metrics['trading_viability']}")
                
        except Exception as e:
            logger.error(f"❌ Error evaluating {symbol}: {str(e)}")
            continue
    
    if all_metrics:
        # Create comprehensive report
        df_metrics = pd.DataFrame(all_metrics)
        
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE ML-STYLE EVALUATION REPORT")
        logger.info("="*80)
        
        # Summary statistics
        logger.info(f"\n📊 SUMMARY STATISTICS:")
        logger.info(f"   Total Symbols Evaluated: {successful_evaluations}")
        logger.info(f"   Average Volatility: {df_metrics['volatility_pct'].mean():.2f}%")
        logger.info(f"   Average RSI Accuracy: {df_metrics['rsi_directional_accuracy'].mean():.1f}%")
        logger.info(f"   Average MACD Accuracy: {df_metrics['macd_directional_accuracy'].mean():.1f}%")
        logger.info(f"   Average Predictability Score: {df_metrics['predictability_score'].mean():.1f}/100")
        
        # Trading viability summary
        viability_counts = df_metrics['trading_viability'].value_counts()
        logger.info(f"\n💰 TRADING VIABILITY DISTRIBUTION:")
        for viability, count in viability_counts.items():
            percentage = count / len(df_metrics) * 100
            logger.info(f"   {viability}: {count} symbols ({percentage:.1f}%)")
        
        # Top performers
        logger.info(f"\n🏆 TOP PERFORMING SYMBOLS:")
        top_performers = df_metrics.nlargest(5, 'predictability_score')
        for _, row in top_performers.iterrows():
            logger.info(f"   {row['symbol']}: {row['predictability_score']:.1f}/100 ({row['trading_viability']})")
        
        # Detailed metrics table
        print(f"\n📋 DETAILED EVALUATION METRICS:")
        display_cols = [
            'symbol', 'volatility_pct', 'rsi_directional_accuracy', 
            'macd_directional_accuracy', 'predictability_score', 'trading_viability'
        ]
        print(df_metrics[display_cols].to_string(index=False))
        
        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"comprehensive_evaluation_{timestamp}.csv"
        df_metrics.to_csv(results_path, index=False)
        logger.info(f"\n💾 Detailed results saved to: {results_path}")
        
        # Performance insights
        logger.info(f"\n💡 KEY INSIGHTS:")
        
        highly_viable = df_metrics[df_metrics['trading_viability'] == 'HIGHLY_VIABLE']
        if len(highly_viable) > 0:
            logger.info(f"   🎯 {len(highly_viable)} symbols show high trading potential")
            logger.info(f"   📈 Best predictability: {df_metrics['predictability_score'].max():.1f}/100")
        
        high_vol = df_metrics[df_metrics['volatility_pct'] > 5.0]
        logger.info(f"   🔥 {len(high_vol)} symbols have >5% volatility (good for trading)")
        
        good_rsi = df_metrics[df_metrics['rsi_directional_accuracy'] > 55]
        logger.info(f"   📊 {len(good_rsi)} symbols show >55% RSI directional accuracy")
        
        logger.info("="*80)
        logger.info("🎉 Comprehensive evaluation completed!")
        logger.info("💡 These metrics simulate ML model performance using statistical analysis.")
        
        return df_metrics
    
    else:
        logger.warning("No successful evaluations completed")
        return None

if __name__ == "__main__":
    results = main() 