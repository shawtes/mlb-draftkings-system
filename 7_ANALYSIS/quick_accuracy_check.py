#!/usr/bin/env python3
"""
Quick ML Accuracy Check
Focused analysis of prediction magnitude vs actual returns
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import get_coinbase_data, make_enhanced_ml_decision
from stacking_ml_engine import stacking_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_accuracy_analysis(symbol='BTC-USD', hours=24):
    """Quick analysis of ML prediction accuracy vs actual returns"""
    
    logger.info(f"🔍 Quick accuracy check for {symbol} over {hours} hours")
    
    try:
        # Get recent historical data
        df = get_coinbase_data(symbol, granularity=3600, days=7)
        
        if df is None or len(df) < hours + 1:
            logger.error(f"Insufficient data for {symbol}")
            return
        
        # Use last N hours for testing
        test_data = df.tail(hours + 1)
        
        predictions = []
        actual_returns = []
        confidences = []
        actions = []
        
        logger.info("Making predictions and comparing to actual returns...")
        
        for i in range(hours):
            try:
                # Current price
                current_price = test_data.iloc[i]['close']
                next_price = test_data.iloc[i + 1]['close']
                
                # Actual return
                actual_return_pct = ((next_price - current_price) / current_price) * 100
                
                # Make ML prediction
                decision = make_enhanced_ml_decision(symbol, granularity=3600, investment_amount=100.0)
                
                if decision:
                    # Extract predicted return from target prices
                    target_prices = decision.get('target_prices', {})
                    confidence = decision.get('overall_confidence', 0)
                    action = decision.get('action', 'HOLD')
                    
                    # Try to get 1h prediction, fallback to others
                    predicted_price = target_prices.get('1h') or target_prices.get('15min') or target_prices.get('30min') or current_price
                    predicted_return_pct = ((predicted_price - current_price) / current_price) * 100
                    
                    predictions.append(predicted_return_pct)
                    actual_returns.append(actual_return_pct)
                    confidences.append(confidence)
                    actions.append(action)
                    
                    logger.info(f"Hour {i+1}: Predicted: {predicted_return_pct:+.3f}%, Actual: {actual_return_pct:+.3f}%, Confidence: {confidence:.1%}, Action: {action}")
                
            except Exception as e:
                logger.warning(f"Error at hour {i+1}: {str(e)}")
                continue
        
        if not predictions:
            logger.error("No valid predictions made")
            return
        
        # Calculate statistics
        pred_array = np.array(predictions)
        actual_array = np.array(actual_returns)
        
        mae = np.mean(np.abs(pred_array - actual_array))
        
        # Directional accuracy
        pred_direction = np.sign(pred_array)
        actual_direction = np.sign(actual_array)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Magnitude analysis
        pred_magnitude = np.mean(np.abs(pred_array))
        actual_magnitude = np.mean(np.abs(actual_array))
        magnitude_ratio = pred_magnitude / actual_magnitude if actual_magnitude > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(pred_array, actual_array)[0, 1] if len(pred_array) > 1 else 0
        
        # Action analysis
        buy_count = actions.count('BUY')
        sell_count = actions.count('SELL')
        hold_count = actions.count('HOLD')
        
        logger.info("\n" + "="*60)
        logger.info("QUICK ACCURACY ANALYSIS RESULTS")
        logger.info("="*60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Analysis Period: {hours} hours")
        logger.info(f"Valid Predictions: {len(predictions)}")
        logger.info("")
        logger.info("ACCURACY METRICS:")
        logger.info(f"  Mean Absolute Error: {mae:.4f}%")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.1%}")
        logger.info(f"  Correlation: {correlation:.3f}")
        logger.info("")
        logger.info("MAGNITUDE ANALYSIS:")
        logger.info(f"  Average Predicted Magnitude: {pred_magnitude:.4f}%")
        logger.info(f"  Average Actual Magnitude: {actual_magnitude:.4f}%")
        logger.info(f"  Magnitude Ratio (Pred/Actual): {magnitude_ratio:.3f}")
        logger.info("")
        logger.info("PREDICTION RANGES:")
        logger.info(f"  Predicted Returns: {np.min(pred_array):+.3f}% to {np.max(pred_array):+.3f}%")
        logger.info(f"  Actual Returns: {np.min(actual_array):+.3f}% to {np.max(actual_array):+.3f}%")
        logger.info("")
        logger.info("ACTION DISTRIBUTION:")
        logger.info(f"  BUY: {buy_count} ({buy_count/len(actions)*100:.1f}%)")
        logger.info(f"  SELL: {sell_count} ({sell_count/len(actions)*100:.1f}%)")
        logger.info(f"  HOLD: {hold_count} ({hold_count/len(actions)*100:.1f}%)")
        logger.info("")
        
        # Analyze if predictions are systematically too conservative
        if magnitude_ratio < 0.5:
            logger.info("⚠️  ISSUE IDENTIFIED: Predictions are VERY CONSERVATIVE")
            logger.info("   Models are predicting much smaller price movements than actually occur")
            logger.info("   This explains the consistently low expected returns")
        elif magnitude_ratio < 0.8:
            logger.info("⚠️  ISSUE IDENTIFIED: Predictions are CONSERVATIVE")
            logger.info("   Models are underestimating the magnitude of price movements")
        elif magnitude_ratio > 1.5:
            logger.info("⚠️  ISSUE IDENTIFIED: Predictions are AGGRESSIVE")
            logger.info("   Models are overestimating the magnitude of price movements")
        else:
            logger.info("✅ Prediction magnitude appears reasonable")
        
        # Analyze directional accuracy
        if directional_accuracy > 0.6:
            logger.info("✅ Good directional accuracy - models predict direction well")
        elif directional_accuracy > 0.5:
            logger.info("⚠️  Moderate directional accuracy - some directional skill")
        else:
            logger.info("❌ Poor directional accuracy - models struggle with direction")
        
        # Summary recommendation
        logger.info("\nRECOMMENDATION:")
        if magnitude_ratio < 0.7 and directional_accuracy > 0.55:
            logger.info("📊 Models have directional skill but are too conservative")
            logger.info("💡 Consider: Scaling up predictions by factor of {:.1f}x".format(1/magnitude_ratio))
            logger.info("💡 Consider: Lowering minimum return thresholds")
        elif magnitude_ratio < 0.7:
            logger.info("📊 Models are both conservative and lack directional skill")
            logger.info("💡 Consider: Retraining with different features or hyperparameters")
        else:
            logger.info("📊 Model performance appears reasonable")
        
        return {
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'magnitude_ratio': magnitude_ratio,
            'correlation': correlation,
            'pred_magnitude': pred_magnitude,
            'actual_magnitude': actual_magnitude,
            'action_distribution': {'BUY': buy_count, 'SELL': sell_count, 'HOLD': hold_count}
        }
        
    except Exception as e:
        logger.error(f"💥 Error in quick accuracy analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main function"""
    logger.info("🚀 Starting Quick ML Accuracy Check")
    
    # Test multiple symbols quickly
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for symbol in symbols:
        logger.info(f"\n{'='*80}")
        quick_accuracy_analysis(symbol, hours=12)  # 12 hours for quick test
        logger.info(f"{'='*80}")

if __name__ == "__main__":
    main() 