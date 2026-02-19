#!/usr/bin/env python3
"""
Quick Model Training and Evaluation
Fast training and evaluation of a few models for testing.
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import train_price_prediction_model, get_coinbase_data, calculate_indicators
from evaluate_models import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_train_and_evaluate():
    """Quick training and evaluation of a few models"""
    print("🚀 Quick Model Training & Evaluation")
    print("=" * 50)
    
    # Test symbols - start with popular, liquid pairs
    test_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD']
    test_granularities = {'1h': 3600}  # Just 1h for speed
    
    print(f"📊 Testing {len(test_symbols)} symbols with {len(test_granularities)} timeframes")
    
    # Step 1: Train models
    print(f"\n🤖 Step 1: Training Models")
    print("-" * 30)
    
    trained_models = []
    
    for symbol in test_symbols:
        for name, granularity in test_granularities.items():
            print(f"🔧 Training {symbol} ({name}) model...")
            
            try:
                model = train_price_prediction_model(symbol, granularity)
                if model is not None:
                    trained_models.append({'symbol': symbol, 'granularity': granularity, 'name': name})
                    print(f"✅ Successfully trained {symbol} ({name})")
                else:
                    print(f"❌ Failed to train {symbol} ({name})")
            except Exception as e:
                print(f"❌ Error training {symbol} ({name}): {str(e)}")
    
    print(f"\n📈 Training Summary: {len(trained_models)}/{len(test_symbols) * len(test_granularities)} models trained")
    
    # Step 2: Quick evaluation
    if trained_models:
        print(f"\n📊 Step 2: Quick Evaluation")
        print("-" * 30)
        
        evaluator = ModelEvaluator()
        
        # Get available models (including newly trained ones)
        available_models = evaluator.get_available_models()
        
        print(f"📋 Found models for evaluation:")
        for symbol, model_list in available_models.items():
            if symbol in test_symbols:
                for model_info in model_list:
                    if model_info['type'] == 'regression':  # Focus on our new regression models
                        print(f"   {symbol}: {evaluator.get_granularity_name(model_info['granularity'])} ({model_info['type']})")
        
        # Evaluate with 3 days of test data
        test_days = 3
        results = []
        
        for symbol in test_symbols:
            if symbol in available_models:
                for model_info in available_models[symbol]:
                    if model_info['type'] == 'regression':  # Only test regression models
                        print(f"🔍 Evaluating {symbol} ({evaluator.get_granularity_name(model_info['granularity'])})...")
                        result = evaluator.evaluate_model_accuracy(symbol, model_info, test_days)
                        if result:
                            results.append(result)
        
        # Quick results summary
        if results:
            print(f"\n📊 Quick Results Summary")
            print("-" * 30)
            
            for result in results:
                print(f"{result['symbol']} ({result['granularity_name']}):")
                print(f"   Direction Accuracy: {result['direction_accuracy']:.1f}%")
                print(f"   Win Rate: {result['win_rate']:.1f}%")
                print(f"   Return: {result['total_return']:+.2f}%")
                print(f"   Trades: {result['total_trades']}")
                print()
            
            # Best performer
            best_accuracy = max(results, key=lambda x: x['direction_accuracy'])
            best_return = max(results, key=lambda x: x['total_return'])
            
            print(f"🏆 Best Accuracy: {best_accuracy['symbol']} ({best_accuracy['direction_accuracy']:.1f}%)")
            print(f"💰 Best Return: {best_return['symbol']} ({best_return['total_return']:+.2f}%)")
            
            # Save quick results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            import pandas as pd
            df = pd.DataFrame(results)
            filename = f"quick_evaluation_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")
            
        else:
            print("❌ No evaluation results available")
    
    else:
        print("❌ No models were trained successfully")
    
    print(f"\n✅ Quick evaluation complete!")

if __name__ == "__main__":
    quick_train_and_evaluate() 