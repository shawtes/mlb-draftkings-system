#!/usr/bin/env python3
"""
ML Model Evaluation Script
Comprehensive evaluation of ML prediction accuracy against historical returns
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import get_coinbase_data, calculate_indicators, make_price_prediction, make_enhanced_ml_decision
from stacking_ml_engine import stacking_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelEvaluator:
    """Comprehensive ML Model Evaluator"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.detailed_predictions = []
        
    def evaluate_symbol_predictions(self, symbol, days_back=30, evaluation_days=7):
        """
        Evaluate ML predictions for a symbol over a historical period
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USD')
            days_back: How many days of data to use for training
            evaluation_days: How many days to evaluate predictions on
        """
        logger.info(f"🔍 Evaluating {symbol} - Training: {days_back} days, Testing: {evaluation_days} days")
        
        try:
            # Get historical data
            total_days = days_back + evaluation_days
            df = get_coinbase_data(symbol, granularity=3600, days=total_days)
            
            if df is None or len(df) < 24 * evaluation_days:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            # Split into training and evaluation periods
            train_end_idx = len(df) - (24 * evaluation_days)  # 24 hours per day
            train_df = df.iloc[:train_end_idx]
            eval_df = df.iloc[train_end_idx:]
            
            logger.info(f"📊 Training period: {len(train_df)} hours")
            logger.info(f"🧪 Evaluation period: {len(eval_df)} hours")
            
            # Train model on historical data
            logger.info(f"🚀 Training model for {symbol}...")
            train_success = stacking_engine.train_model(symbol, granularity=3600, days=days_back)
            
            if not train_success:
                logger.error(f"Failed to train model for {symbol}")
                return None
            
            # Make predictions for each hour in evaluation period
            predictions = []
            actual_returns = []
            timestamps = []
            
            for i in range(len(eval_df) - 1):
                try:
                    # Use data up to current point for prediction
                    current_data = df.iloc[:train_end_idx + i + 1]
                    
                    # Make enhanced ML prediction
                    enhanced_decision = make_enhanced_ml_decision(
                        symbol=symbol,
                        granularity=3600,
                        investment_amount=100.0
                    )
                    
                    if enhanced_decision:
                        # Get predicted return percentage
                        predicted_return = enhanced_decision.get('target_prices', {}).get('1h', 0)
                        current_price = enhanced_decision.get('current_price', 0)
                        
                        if predicted_return > 0 and current_price > 0:
                            predicted_return_pct = ((predicted_return - current_price) / current_price) * 100
                        else:
                            predicted_return_pct = 0
                        
                        # Calculate actual return
                        current_price_actual = eval_df.iloc[i]['close']
                        next_price_actual = eval_df.iloc[i + 1]['close']
                        actual_return_pct = ((next_price_actual - current_price_actual) / current_price_actual) * 100
                        
                        predictions.append(predicted_return_pct)
                        actual_returns.append(actual_return_pct)
                        timestamps.append(eval_df.iloc[i]['timestamp'])
                        
                        # Store detailed prediction data
                        self.detailed_predictions.append({
                            'symbol': symbol,
                            'timestamp': eval_df.iloc[i]['timestamp'],
                            'predicted_return_pct': predicted_return_pct,
                            'actual_return_pct': actual_return_pct,
                            'current_price': current_price_actual,
                            'next_price': next_price_actual,
                            'confidence': enhanced_decision.get('overall_confidence', 0),
                            'action': enhanced_decision.get('action', 'HOLD'),
                            'profit_probability': enhanced_decision.get('profit_probability', 0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error making prediction for {symbol} at step {i}: {str(e)}")
                    continue
            
            if len(predictions) == 0:
                logger.error(f"No valid predictions for {symbol}")
                return None
            
            # Calculate evaluation metrics
            results = self.calculate_evaluation_metrics(
                symbol, predictions, actual_returns, timestamps
            )
            
            logger.info(f"✅ Evaluation complete for {symbol}")
            logger.info(f"   📊 Predictions made: {len(predictions)}")
            logger.info(f"   🎯 MAE: {results['mae']:.4f}%")
            logger.info(f"   🧭 Directional Accuracy: {results['directional_accuracy']:.1%}")
            logger.info(f"   📈 Correlation: {results['correlation']:.3f}")
            
            self.evaluation_results[symbol] = results
            return results
            
        except Exception as e:
            logger.error(f"💥 Error evaluating {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def calculate_evaluation_metrics(self, symbol, predictions, actual_returns, timestamps):
        """Calculate comprehensive evaluation metrics"""
        
        pred_array = np.array(predictions)
        actual_array = np.array(actual_returns)
        
        # Basic accuracy metrics
        mae = np.mean(np.abs(pred_array - actual_array))
        rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2))
        
        # Directional accuracy
        pred_direction = np.sign(pred_array)
        actual_direction = np.sign(actual_array)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        # Correlation
        correlation = np.corrcoef(pred_array, actual_array)[0, 1] if len(pred_array) > 1 else 0
        
        # R-squared
        ss_res = np.sum((actual_array - pred_array) ** 2)
        ss_tot = np.sum((actual_array - np.mean(actual_array)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Return magnitude analysis
        pred_magnitude = np.mean(np.abs(pred_array))
        actual_magnitude = np.mean(np.abs(actual_array))
        magnitude_ratio = pred_magnitude / actual_magnitude if actual_magnitude != 0 else 0
        
        # Profit simulation (if we followed all predictions)
        simulated_profit = 0
        for pred, actual in zip(pred_array, actual_array):
            if abs(pred) > 0.5:  # Only trade if prediction > 0.5%
                if pred > 0:  # Predicted up, buy
                    simulated_profit += actual - 0.7  # Subtract trading fees
                elif pred < 0:  # Predicted down, sell/short
                    simulated_profit += abs(actual) - 0.7 if actual < 0 else -actual - 0.7
        
        # Profitable trade percentage
        profitable_trades = 0
        total_trades = 0
        
        for pred, actual in zip(pred_array, actual_array):
            if abs(pred) > 0.5:
                total_trades += 1
                if (pred > 0 and actual > 0.7) or (pred < 0 and actual < -0.7):
                    profitable_trades += 1
        
        profit_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            'symbol': symbol,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'r2_score': r2_score,
            'predicted_magnitude': pred_magnitude,
            'actual_magnitude': actual_magnitude,
            'magnitude_ratio': magnitude_ratio,
            'simulated_profit_pct': simulated_profit,
            'profit_rate': profit_rate,
            'total_predictions': len(predictions),
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'timestamps': timestamps,
            'predictions': predictions,
            'actual_returns': actual_returns
        }
    
    def evaluate_multiple_symbols(self, symbols, days_back=30, evaluation_days=7):
        """Evaluate multiple symbols and generate summary report"""
        logger.info(f"🔍 Evaluating {len(symbols)} symbols...")
        
        results_summary = []
        
        for symbol in symbols:
            logger.info(f"📊 Processing {symbol}...")
            result = self.evaluate_symbol_predictions(symbol, days_back, evaluation_days)
            
            if result:
                results_summary.append({
                    'symbol': result['symbol'],
                    'mae': result['mae'],
                    'directional_accuracy': result['directional_accuracy'],
                    'correlation': result['correlation'],
                    'magnitude_ratio': result['magnitude_ratio'],
                    'simulated_profit_pct': result['simulated_profit_pct'],
                    'profit_rate': result['profit_rate'],
                    'total_predictions': result['total_predictions']
                })
        
        # Generate summary statistics
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            
            logger.info("\n🎯 EVALUATION SUMMARY:")
            logger.info(f"   Average MAE: {summary_df['mae'].mean():.4f}%")
            logger.info(f"   Average Directional Accuracy: {summary_df['directional_accuracy'].mean():.1%}")
            logger.info(f"   Average Correlation: {summary_df['correlation'].mean():.3f}")
            logger.info(f"   Average Magnitude Ratio: {summary_df['magnitude_ratio'].mean():.3f}")
            logger.info(f"   Average Simulated Profit: {summary_df['simulated_profit_pct'].mean():.2f}%")
            logger.info(f"   Average Profit Rate: {summary_df['profit_rate'].mean():.1%}")
            
            # Identify best and worst performers
            best_performer = summary_df.loc[summary_df['directional_accuracy'].idxmax()]
            worst_performer = summary_df.loc[summary_df['directional_accuracy'].idxmin()]
            
            logger.info(f"\n🏆 Best Performer: {best_performer['symbol']}")
            logger.info(f"   Directional Accuracy: {best_performer['directional_accuracy']:.1%}")
            logger.info(f"   MAE: {best_performer['mae']:.4f}%")
            
            logger.info(f"\n⚠️ Worst Performer: {worst_performer['symbol']}")
            logger.info(f"   Directional Accuracy: {worst_performer['directional_accuracy']:.1%}")
            logger.info(f"   MAE: {worst_performer['mae']:.4f}%")
        
        return results_summary
    
    def generate_evaluation_report(self, output_file='ml_evaluation_report.txt'):
        """Generate detailed evaluation report"""
        logger.info(f"📝 Generating evaluation report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("ML MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            for symbol, results in self.evaluation_results.items():
                f.write(f"SYMBOL: {symbol}\n")
                f.write("-" * 30 + "\n")
                f.write(f"MAE: {results['mae']:.4f}%\n")
                f.write(f"RMSE: {results['rmse']:.4f}%\n")
                f.write(f"Directional Accuracy: {results['directional_accuracy']:.1%}\n")
                f.write(f"Correlation: {results['correlation']:.3f}\n")
                f.write(f"R² Score: {results['r2_score']:.3f}\n")
                f.write(f"Predicted Magnitude: {results['predicted_magnitude']:.4f}%\n")
                f.write(f"Actual Magnitude: {results['actual_magnitude']:.4f}%\n")
                f.write(f"Magnitude Ratio: {results['magnitude_ratio']:.3f}\n")
                f.write(f"Simulated Profit: {results['simulated_profit_pct']:.2f}%\n")
                f.write(f"Profit Rate: {results['profit_rate']:.1%}\n")
                f.write(f"Total Predictions: {results['total_predictions']}\n")
                f.write(f"Profitable Trades: {results['profitable_trades']}/{results['total_trades']}\n")
                f.write("\n")
        
        logger.info(f"✅ Report saved to {output_file}")

def main():
    """Main evaluation function"""
    evaluator = MLModelEvaluator()
    
    # List of symbols to evaluate
    test_symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD',
        'MATIC-USD', 'AVAX-USD', 'ALGO-USD', 'XRP-USD', 'LTC-USD'
    ]
    
    # Run evaluation
    logger.info("🚀 Starting ML Model Evaluation...")
    
    # Evaluate individual symbols
    for symbol in test_symbols[:3]:  # Start with 3 symbols for quick test
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING {symbol}")
        logger.info(f"{'='*60}")
        
        evaluator.evaluate_symbol_predictions(
            symbol=symbol,
            days_back=30,  # 30 days for training
            evaluation_days=3  # 3 days for evaluation
        )
    
    # Generate summary
    summary = evaluator.evaluate_multiple_symbols(test_symbols[:3], days_back=30, evaluation_days=3)
    
    # Generate report
    evaluator.generate_evaluation_report()
    
    logger.info("\n🎉 Evaluation complete!")
    logger.info("📊 Check ml_evaluation_report.txt for detailed results")

if __name__ == "__main__":
    main() 