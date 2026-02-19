#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script
Trains new models and evaluates them comprehensively across different symbols and timeframes.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import (
    get_coinbase_data, 
    calculate_indicators, 
    get_cached_symbols,
    train_price_prediction_model,
    make_ml_decision,
    get_price_prediction_for_granularity
)
from init_database import get_db_path
from evaluate_models import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedModelEvaluator(ModelEvaluator):
    """Enhanced model evaluator with training capabilities"""
    
    def __init__(self):
        super().__init__()
        self.top_symbols = [
            'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
            'DOT-USD', 'LINK-USD', 'AVAX-USD', 'ATOM-USD', 'XTZ-USD'
        ]
    
    def train_comprehensive_models(self, force_retrain=False):
        """Train regression models for top performing symbols"""
        logger.info("🚀 Training comprehensive model suite...")
        
        trained_models = []
        total_to_train = len(self.top_symbols) * len(self.granularities)
        current = 0
        
        for symbol in self.top_symbols:
            logger.info(f"📊 Training models for {symbol}...")
            
            for granularity_name, granularity in self.granularities.items():
                current += 1
                logger.info(f"🔧 [{current}/{total_to_train}] Training {symbol} {granularity_name} model...")
                
                model_path = os.path.join(self.models_dir, f"{symbol}_{granularity}_model.pkl")
                
                if os.path.exists(model_path) and not force_retrain:
                    logger.info(f"✅ Model already exists for {symbol} ({granularity_name})")
                    trained_models.append({'symbol': symbol, 'granularity': granularity, 'status': 'existing'})
                    continue
                
                try:
                    # Train the price prediction model
                    model = train_price_prediction_model(symbol, granularity)
                    if model is not None:
                        trained_models.append({'symbol': symbol, 'granularity': granularity, 'status': 'trained'})
                        logger.info(f"✅ Successfully trained {symbol} ({granularity_name})")
                    else:
                        trained_models.append({'symbol': symbol, 'granularity': granularity, 'status': 'failed'})
                        logger.warning(f"❌ Failed to train {symbol} ({granularity_name})")
                        
                except Exception as e:
                    trained_models.append({'symbol': symbol, 'granularity': granularity, 'status': 'error', 'error': str(e)})
                    logger.error(f"❌ Error training {symbol} ({granularity_name}): {str(e)}")
        
        # Summary
        trained_count = len([m for m in trained_models if m['status'] == 'trained'])
        existing_count = len([m for m in trained_models if m['status'] == 'existing'])
        failed_count = len([m for m in trained_models if m['status'] in ['failed', 'error']])
        
        logger.info(f"🎉 Training Summary:")
        logger.info(f"   ✅ New models trained: {trained_count}")
        logger.info(f"   📁 Existing models: {existing_count}")
        logger.info(f"   ❌ Failed models: {failed_count}")
        
        return trained_models
    
    def evaluate_model_performance_comparison(self, test_days=7):
        """Compare performance between different model types and timeframes"""
        logger.info("📊 Running comprehensive performance comparison...")
        
        # Get all available models
        all_results = self.evaluate_all_models(test_days, train_missing=False)
        
        if not all_results:
            logger.warning("No models to compare")
            return
        
        df = pd.DataFrame(all_results)
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Model type comparison
        if 'model_type' in df.columns:
            print(f"\n📊 PERFORMANCE BY MODEL TYPE:")
            type_comparison = df.groupby('model_type').agg({
                'direction_accuracy': ['mean', 'std', 'count'],
                'win_rate': ['mean', 'std'],
                'total_return': ['mean', 'std'],
                'total_trades': 'mean'
            }).round(2)
            
            for model_type in type_comparison.index:
                stats = type_comparison.loc[model_type]
                print(f"   {model_type.upper()}:")
                print(f"     Direction Accuracy: {stats['direction_accuracy']['mean']:.1f}% ± {stats['direction_accuracy']['std']:.1f}% ({stats['direction_accuracy']['count']} models)")
                print(f"     Win Rate: {stats['win_rate']['mean']:.1f}% ± {stats['win_rate']['std']:.1f}%")
                print(f"     Avg Return: {stats['total_return']['mean']:.2f}% ± {stats['total_return']['std']:.2f}%")
                print(f"     Avg Trades: {stats['total_trades']['mean']:.0f}")
        
        # Timeframe comparison
        print(f"\n⏰ PERFORMANCE BY TIMEFRAME:")
        timeframe_comparison = df.groupby('granularity_name').agg({
            'direction_accuracy': ['mean', 'std', 'count'],
            'win_rate': ['mean', 'std'],
            'total_return': ['mean', 'std'],
            'total_trades': 'mean'
        }).round(2)
        
        for timeframe in timeframe_comparison.index:
            stats = timeframe_comparison.loc[timeframe]
            print(f"   {timeframe}:")
            print(f"     Direction Accuracy: {stats['direction_accuracy']['mean']:.1f}% ± {stats['direction_accuracy']['std']:.1f}% ({stats['direction_accuracy']['count']} models)")
            print(f"     Win Rate: {stats['win_rate']['mean']:.1f}% ± {stats['win_rate']['std']:.1f}%")
            print(f"     Avg Return: {stats['total_return']['mean']:.2f}% ± {stats['total_return']['std']:.2f}%")
            print(f"     Avg Trades: {stats['total_trades']['mean']:.0f}")
        
        # Top performers overall
        print(f"\n🏆 TOP 10 PERFORMING MODELS:")
        # Sort by a combined score: direction accuracy + return
        df['performance_score'] = df['direction_accuracy'] + (df['total_return'] * 2)  # Weight return 2x
        top_performers = df.nlargest(10, 'performance_score')
        
        for idx, row in top_performers.iterrows():
            print(f"   {row['symbol']} ({row['granularity_name']}, {row['model_type']}): "
                  f"Score: {row['performance_score']:.1f} "
                  f"(Acc: {row['direction_accuracy']:.1f}%, Ret: {row['total_return']:.2f}%)")
        
        # Models worth deploying
        print(f"\n💎 MODELS RECOMMENDED FOR LIVE TRADING:")
        good_models = df[
            (df['direction_accuracy'] >= 60) & 
            (df['total_return'] >= 1.0) & 
            (df['win_rate'] >= 40)
        ]
        
        if len(good_models) > 0:
            for idx, row in good_models.iterrows():
                print(f"   ✅ {row['symbol']} ({row['granularity_name']}, {row['model_type']}): "
                      f"{row['direction_accuracy']:.1f}% accuracy, "
                      f"{row['win_rate']:.1f}% win rate, "
                      f"{row['total_return']:.2f}% return")
        else:
            print("   ⚠️ No models meet the criteria for live trading deployment")
            print("      Criteria: ≥60% accuracy, ≥1% return, ≥40% win rate")
        
        # Risk analysis
        print(f"\n⚠️ RISK ANALYSIS:")
        high_risk = df[df['total_return'] < -3.0]
        if len(high_risk) > 0:
            print(f"   🚨 High-risk models (>3% loss):")
            for idx, row in high_risk.iterrows():
                print(f"     {row['symbol']} ({row['granularity_name']}): {row['total_return']:.2f}% loss")
        else:
            print("   ✅ No high-risk models detected")
        
        low_accuracy = df[df['direction_accuracy'] < 45]
        if len(low_accuracy) > 0:
            print(f"   📉 Low accuracy models (<45%):")
            for idx, row in low_accuracy.iterrows():
                print(f"     {row['symbol']} ({row['granularity_name']}): {row['direction_accuracy']:.1f}% accuracy")
        
        print("\n" + "="*80)
        
        return df
    
    def generate_trading_recommendations(self):
        """Generate specific trading recommendations based on model performance"""
        logger.info("💡 Generating trading recommendations...")
        
        if not self.results:
            logger.warning("No results available for recommendations")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("💡 TRADING RECOMMENDATIONS")
        print("="*80)
        
        # Best symbols to trade
        symbol_performance = df.groupby('symbol').agg({
            'direction_accuracy': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean',
            'total_trades': 'mean'
        }).round(2)
        
        # Sort by combined performance
        symbol_performance['combined_score'] = (
            symbol_performance['direction_accuracy'] + 
            (symbol_performance['total_return'] * 2) +
            symbol_performance['win_rate']
        )
        
        top_symbols = symbol_performance.nlargest(5, 'combined_score')
        
        print(f"\n🎯 TOP 5 SYMBOLS TO TRADE:")
        for symbol, stats in top_symbols.iterrows():
            print(f"   {symbol}: Combined Score {stats['combined_score']:.1f}")
            print(f"     Accuracy: {stats['direction_accuracy']:.1f}%, "
                  f"Win Rate: {stats['win_rate']:.1f}%, "
                  f"Avg Return: {stats['total_return']:.2f}%")
        
        # Best timeframes
        timeframe_performance = df.groupby('granularity_name').agg({
            'direction_accuracy': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean'
        }).round(2)
        
        print(f"\n⏰ RECOMMENDED TIMEFRAMES:")
        for timeframe, stats in timeframe_performance.iterrows():
            recommendation = "🟢 Strong" if stats['direction_accuracy'] >= 55 else "🟡 Moderate" if stats['direction_accuracy'] >= 50 else "🔴 Weak"
            print(f"   {timeframe}: {recommendation} "
                  f"(Accuracy: {stats['direction_accuracy']:.1f}%, "
                  f"Avg Return: {stats['total_return']:.2f}%)")
        
        # Portfolio allocation suggestions
        profitable_models = df[df['total_return'] > 0].copy()
        if len(profitable_models) > 0:
            print(f"\n💰 SUGGESTED PORTFOLIO ALLOCATION:")
            profitable_models['weight'] = profitable_models['total_return'] / profitable_models['total_return'].sum()
            
            for idx, row in profitable_models.nlargest(5, 'total_return').iterrows():
                allocation = row['weight'] * 100
                print(f"   {row['symbol']} ({row['granularity_name']}): {allocation:.1f}% "
                      f"(Return: {row['total_return']:.2f}%)")
        else:
            print(f"\n💰 PORTFOLIO ALLOCATION:")
            print("   ⚠️ No profitable models found - recommend paper trading first")
        
        print("\n" + "="*80)

def main():
    """Enhanced evaluation main function"""
    print("🚀 Enhanced ML Model Evaluation & Training System")
    print("=" * 60)
    
    # Parse command line arguments
    test_days = 7
    force_retrain = False
    
    if len(sys.argv) > 1:
        if '--retrain' in sys.argv:
            force_retrain = True
            print("🔧 Force retraining enabled")
        
        try:
            test_days = int([arg for arg in sys.argv if arg.isdigit()][0])
            print(f"📅 Using {test_days} days for testing")
        except (IndexError, ValueError):
            print(f"📅 Using default {test_days} days for testing")
    
    evaluator = EnhancedModelEvaluator()
    
    # Step 1: Train comprehensive model suite
    print(f"\n🤖 Step 1: Training/Checking Model Suite")
    print("-" * 40)
    training_results = evaluator.train_comprehensive_models(force_retrain=force_retrain)
    
    # Step 2: Evaluate all models
    print(f"\n📊 Step 2: Comprehensive Model Evaluation")
    print("-" * 40)
    results_df = evaluator.evaluate_model_performance_comparison(test_days)
    
    if results_df is not None and len(results_df) > 0:
        # Step 3: Generate trading recommendations
        print(f"\n💡 Step 3: Trading Recommendations")
        print("-" * 40)
        evaluator.generate_trading_recommendations()
        
        # Step 4: Save detailed results
        print(f"\n💾 Step 4: Saving Results")
        print("-" * 40)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        filename = f"enhanced_evaluation_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"💾 Comprehensive results saved to {filename}")
        
        # Save JSON with training info
        json_data = {
            'evaluation_timestamp': timestamp,
            'test_days': test_days,
            'training_results': training_results,
            'model_results': evaluator.results,
            'summary_stats': {
                'total_models_evaluated': len(results_df),
                'avg_accuracy': results_df['direction_accuracy'].mean(),
                'avg_return': results_df['total_return'].mean(),
                'best_model': results_df.loc[results_df['direction_accuracy'].idxmax()].to_dict() if len(results_df) > 0 else None
            }
        }
        
        json_filename = f"enhanced_evaluation_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"💾 Detailed analysis saved to {json_filename}")
        
        print(f"\n✅ Enhanced evaluation complete!")
        print(f"📊 {len(results_df)} models evaluated across {len(evaluator.top_symbols)} symbols")
        print(f"💡 Check the generated CSV and JSON files for detailed analysis")
        
        # Quick summary
        if len(results_df) > 0:
            best_accuracy = results_df['direction_accuracy'].max()
            best_return = results_df['total_return'].max()
            profitable_count = len(results_df[results_df['total_return'] > 0])
            
            print(f"\n📈 Quick Summary:")
            print(f"   🎯 Best Accuracy: {best_accuracy:.1f}%")
            print(f"   💰 Best Return: {best_return:.2f}%")
            print(f"   📊 Profitable Models: {profitable_count}/{len(results_df)}")
            
    else:
        print("❌ No models could be evaluated successfully.")
        print("💡 Check your data connections and symbol availability.")

if __name__ == "__main__":
    main() 