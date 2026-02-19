#!/usr/bin/env python3
"""
Train and Evaluate 15 Regression Models
Trains regression models for multiple symbols and timeframes, then evaluates them.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import training and evaluation functions
from maybe import train_price_prediction_model, get_cached_symbols
from regression_eval import RegressionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainerAndEvaluator:
    """Train multiple regression models and evaluate them"""
    
    def __init__(self):
        self.trained_models = []
        self.evaluation_results = []
        
    def get_top_symbols(self, count=10):
        """Get top trading symbols"""
        try:
            # Get available symbols
            all_symbols = get_cached_symbols()
            
            # Popular crypto symbols for training
            priority_symbols = [
                'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
                'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD',
                'ALGO-USD', 'ATOM-USD', 'AVAX-USD', 'NEAR-USD', 'FTM-USD'
            ]
            
            # Filter to only available symbols
            available_priority = [s for s in priority_symbols if s in all_symbols]
            
            # Add more symbols if needed
            if len(available_priority) < count:
                additional = [s for s in all_symbols if s not in available_priority][:count - len(available_priority)]
                available_priority.extend(additional)
            
            return available_priority[:count]
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            # Fallback list
            return [
                'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'MATIC-USD',
                'DOT-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD', 'XLM-USD'
            ]
    
    def train_models(self, target_count=15):
        """Train regression models for multiple symbols and timeframes"""
        print("🚀 Training Regression Models for Price Prediction")
        print("=" * 60)
        
        # Get symbols to train
        symbols = self.get_top_symbols(8)  # Get 8 symbols
        
        # Timeframes to train (in seconds)
        timeframes = [
            (900, '15m'),    # 15 minutes
            (3600, '1h'),    # 1 hour
        ]
        
        # Plan training combinations
        training_plan = []
        for symbol in symbols:
            for granularity, name in timeframes:
                training_plan.append((symbol, granularity, name))
                if len(training_plan) >= target_count:
                    break
            if len(training_plan) >= target_count:
                break
        
        print(f"📋 Training Plan: {len(training_plan)} models")
        for i, (symbol, granularity, name) in enumerate(training_plan, 1):
            print(f"   {i:2d}. {symbol} ({name})")
        
        print(f"\n🎯 Starting training...")
        
        trained_count = 0
        failed_count = 0
        
        for i, (symbol, granularity, name) in enumerate(training_plan, 1):
            try:
                print(f"\n[{i:2d}/{len(training_plan)}] 🔨 Training {symbol} ({name})...")
                start_time = time.time()
                
                # Train the model
                result = train_price_prediction_model(symbol, granularity)
                
                if result and len(result) >= 2:
                    model_lr, model_rf = result[0], result[1]
                    if model_lr is not None or model_rf is not None:
                        trained_count += 1
                        elapsed = time.time() - start_time
                        print(f"✅ {symbol} ({name}) trained in {elapsed:.1f}s")
                        
                        self.trained_models.append({
                            'symbol': symbol,
                            'granularity': granularity,
                            'timeframe_name': name,
                            'trained_at': datetime.now(),
                            'training_time': elapsed
                        })
                    else:
                        failed_count += 1
                        print(f"❌ {symbol} ({name}) training failed - no models returned")
                else:
                    failed_count += 1
                    print(f"❌ {symbol} ({name}) training failed - invalid result")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ {symbol} ({name}) training failed: {str(e)}")
                
            # Small delay between trainings
            if i < len(training_plan):
                time.sleep(2)
        
        print(f"\n📊 Training Summary:")
        print(f"   ✅ Successfully trained: {trained_count}")
        print(f"   ❌ Failed: {failed_count}")
        print(f"   📈 Success rate: {(trained_count / len(training_plan) * 100):.1f}%")
        
        return trained_count > 0
    
    def evaluate_models(self, test_days=7):
        """Evaluate all trained regression models"""
        print(f"\n🔍 Evaluating Trained Regression Models")
        print("=" * 60)
        
        evaluator = RegressionEvaluator()
        results = evaluator.evaluate_all_regression_models(test_days)
        
        if results:
            # Print summary
            evaluator.print_regression_summary()
            
            # Save results with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trained_models_eval_{timestamp}.csv"
            evaluator.save_regression_results(filename)
            
            self.evaluation_results = results
            print(f"\n✅ Evaluation complete! {len(results)} models evaluated.")
            
            # Print training vs evaluation summary
            print(f"\n📋 TRAINING vs EVALUATION SUMMARY:")
            print(f"   Models trained: {len(self.trained_models)}")
            print(f"   Models evaluated: {len(results)}")
            
            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                
                # Best performers
                print(f"\n🏆 TOP PERFORMERS:")
                
                # Best R²
                best_r2 = df.loc[df['r2_score'].idxmax()]
                print(f"   Best R²: {best_r2['symbol']} ({best_r2['granularity_name']}) = {best_r2['r2_score']:.3f}")
                
                # Best MAE
                best_mae = df.loc[df['mae'].idxmin()]
                print(f"   Best MAE: {best_mae['symbol']} ({best_mae['granularity_name']}) = {best_mae['mae']:.4f}%")
                
                # Best direction accuracy
                best_dir = df.loc[df['direction_accuracy'].idxmax()]
                print(f"   Best Direction: {best_dir['symbol']} ({best_dir['granularity_name']}) = {best_dir['direction_accuracy']:.1f}%")
                
                # Models with positive R²
                positive_r2 = df[df['r2_score'] > 0]
                print(f"   Models with positive R²: {len(positive_r2)}/{len(df)}")
                
                # Models with >50% direction accuracy
                good_direction = df[df['direction_accuracy'] > 50]
                print(f"   Models with >50% direction accuracy: {len(good_direction)}/{len(df)}")
            
        else:
            print("❌ No models could be evaluated")
            
        return len(results) > 0

def main():
    """Main training and evaluation function"""
    print("🎯 Train and Evaluate 15 Regression Models")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = ModelTrainerAndEvaluator()
    
    # Step 1: Train models
    training_success = trainer.train_models(target_count=15)
    
    if not training_success:
        print("❌ No models were successfully trained")
        return
    
    # Small delay before evaluation
    print(f"\n⏱️ Waiting 5 seconds before evaluation...")
    time.sleep(5)
    
    # Step 2: Evaluate models
    evaluation_success = trainer.evaluate_models(test_days=7)
    
    if evaluation_success:
        print(f"\n🎉 Training and evaluation completed successfully!")
        print(f"📈 Check the CSV file for detailed results")
    else:
        print(f"\n⚠️ Training completed but evaluation failed")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 