#!/usr/bin/env python3
"""
Quick script to retrain all existing models with consistent feature names
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maybe import train_model_for_symbol
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrain_all_models():
    """Retrain all models with consistent feature names"""
    
    print("🔄 Retraining all ML models with consistent features...")
    
    # Models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        return
    
    # Find all existing model files
    existing_models = set()
    for file in os.listdir(models_dir):
        if file.endswith('_clf.pkl'):
            # Extract symbol from filename like "BTCUSD_3600_clf.pkl"
            parts = file.replace('_clf.pkl', '').split('_')
            if len(parts) >= 2:
                symbol_part = '_'.join(parts[:-1])  # Everything except the last part (granularity)
                granularity = parts[-1]
                
                # Convert back to proper symbol format
                symbol = symbol_part[:3] + '-' + symbol_part[3:]  # Add dash for USD pairs
                existing_models.add((symbol, int(granularity)))
    
    print(f"📊 Found {len(existing_models)} existing models to retrain")
    
    retrained_count = 0
    failed_count = 0
    
    for symbol, granularity in existing_models:
        try:
            print(f"🤖 Retraining {symbol} (granularity: {granularity}s)...")
            model = train_model_for_symbol(symbol, granularity)
            
            if model is not None:
                retrained_count += 1
                print(f"✅ Successfully retrained {symbol}")
            else:
                failed_count += 1
                print(f"❌ Failed to retrain {symbol}")
                
        except Exception as e:
            failed_count += 1
            print(f"❌ Error retraining {symbol}: {str(e)}")
    
    print(f"\n🎉 Retraining complete!")
    print(f"   ✅ Successfully retrained: {retrained_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📊 Total: {len(existing_models)}")

if __name__ == "__main__":
    retrain_all_models() 