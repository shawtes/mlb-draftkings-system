#!/usr/bin/env python3
"""
Quick Train 15 Models and Evaluate
Uses existing functions but imports them selectively to avoid syntax errors.
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def train_models_and_evaluate():
    """Train 15 models and evaluate them"""
    print("🎯 Quick Train 15 Models and Evaluate")
    print("=" * 50)
    
    # Define symbols and timeframes
    symbols = [
        'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
        'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD'
    ]
    
    timeframes = [900, 3600]  # 15m, 1h
    
    # Plan training
    training_plan = []
    for symbol in symbols:
        for granularity in timeframes:
            training_plan.append((symbol, granularity))
            if len(training_plan) >= 15:
                break
        if len(training_plan) >= 15:
            break
    
    print(f"📋 Training Plan: {len(training_plan)} models")
    for i, (symbol, granularity) in enumerate(training_plan, 1):
        timeframe_name = f"{granularity//3600}h" if granularity >= 3600 else f"{granularity//60}m"
        print(f"   {i:2d}. {symbol} ({timeframe_name})")
    
    # Import the training function carefully
    try:
        # First try to import specific training function from the working file
        print(f"\n🔧 Loading training functions...")
        
        # Try importing train_price_prediction_model directly
        # We'll use exec to avoid the syntax error in maybe.py
        with open('maybe.py', 'r') as f:
            content = f.read()
        
        # Extract just the training function and dependencies
        train_func_start = content.find('def train_price_prediction_model(symbol, granularity):')
        if train_func_start == -1:
            print("❌ Training function not found in maybe.py")
            return
        
        # For now, let's use the existing regression evaluator
        from regression_eval import RegressionEvaluator
        
        print("✅ Using regression evaluator to check existing models")
        evaluator = RegressionEvaluator()
        
        # Check what models we already have
        existing_models = evaluator.get_regression_models()
        print(f"📊 Found {len(existing_models)} existing model families")
        
        # If we have fewer than 15 models, we need to create new ones
        total_existing = sum(len(models) for models in existing_models.values())
        print(f"📈 Total existing models: {total_existing}")
        
        if total_existing < 15:
            print(f"🔨 Need to create {15 - total_existing} more models")
            
            # Use the direct training from the simple script
            print("🚀 Using direct training approach...")
            exec(open('simple_train_eval.py').read())
            return
        
        # Evaluate existing models
        print(f"\n🔍 Evaluating existing models...")
        results = evaluator.evaluate_all_regression_models(test_days=7)
        
        if results:
            print(f"✅ Evaluated {len(results)} models")
            
            # Print summary
            evaluator.print_regression_summary()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quick_eval_{timestamp}.csv"
            evaluator.save_regression_results(filename)
            print(f"\n💾 Results saved to {filename}")
        else:
            print("❌ No models could be evaluated")
        
    except Exception as e:
        print(f"❌ Error in training/evaluation: {str(e)}")
        print("🔄 Falling back to simple training approach...")
        
        # Fall back to our simple training
        try:
            # Import and run the simple trainer
            from simple_train_eval import SimpleTrainEvaluate
            trainer = SimpleTrainEvaluate()
            trainer.train_and_evaluate_models(count=15)
        except Exception as e2:
            print(f"❌ Fallback also failed: {str(e2)}")

if __name__ == "__main__":
    train_models_and_evaluate() 