#!/usr/bin/env python3
"""
Integration script for optimized ML training
This replaces the current training logic with an optimized version that only trains top 10 assets
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add parent directory to path for optimized_training import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

def get_optimized_training_assets(max_assets=10):
    """
    Get top assets for training using the optimized approach.
    This replaces the old method of training all active positions.
    """
    try:
        from optimized_training import get_top_assets_for_training
        from init_database import get_db_path
        
        db_path = get_db_path()
        return get_top_assets_for_training(max_assets=max_assets, db_path=db_path)
        
    except ImportError:
        logger.warning("⚠️ Optimized training not available, falling back to old method")
        # Fallback to current positions
        try:
            from init_database import get_db_path
            import sqlite3
            
            conn = sqlite3.connect(get_db_path())
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT symbol FROM positions 
                WHERE status IN ('active', 'full_hold', 'open')
                ORDER BY symbol LIMIT ?
            """, (max_assets,))
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return [{'symbol': symbol, 'score': 50.0, 'has_position': True} for symbol in symbols]
            
        except Exception as e:
            logger.error(f"💥 Fallback failed: {str(e)}")
            return []

def run_optimized_auto_training():
    """
    Run optimized auto-training that only trains top performing assets.
    This is meant to replace the auto_retrain_models function.
    """
    try:
        logger.info("🤖 Starting OPTIMIZED automatic model retraining...")
        
        # Get top assets for training
        top_assets = get_optimized_training_assets(max_assets=10)
        
        if not top_assets:
            logger.info("🤖 No top performing assets found for training")
            return False
        
        logger.info(f"🤖 OPTIMIZED: Auto-retraining models for TOP {len(top_assets)} assets: {', '.join([asset['symbol'] for asset in top_assets])}")
        
        # Import training function
        from maybe import train_model_for_symbol
        
        retrained_count = 0
        total_training_time = 0
        
        for asset in top_assets:
            symbol = asset['symbol']
            score = asset['score']
            
            try:
                training_start = time.time()
                logger.info(f"🤖 Auto-retraining model for {symbol} (Score: {score:.1f})...")
                
                # Train with appropriate granularity based on asset performance
                granularity = 3600  # Default 1-hour
                if score > 80:  # High-performance assets get more detailed training
                    granularity = 900  # 15-minute for top performers
                    logger.info(f"   🎯 Using 15-minute granularity for high-performance asset {symbol}")
                
                model = train_model_for_symbol(symbol, granularity=granularity)
                
                training_time = time.time() - training_start
                total_training_time += training_time
                
                if model is not None:
                    retrained_count += 1
                    logger.info(f"✅ Auto-retrained models for {symbol} in {training_time:.1f}s")
                else:
                    logger.warning(f"❌ Failed to auto-retrain models for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error auto-retraining {symbol}: {str(e)}")
        
        # Log optimization results
        avg_training_time = total_training_time / max(retrained_count, 1)
        logger.info(f"🎉 OPTIMIZED auto-retraining completed: {retrained_count}/{len(top_assets)} models updated")
        logger.info(f"⚡ Total training time: {total_training_time:.1f}s (avg: {avg_training_time:.1f}s per model)")
        
        # Estimate resource savings
        try:
            from maybe import get_cached_symbols
            total_symbols = len(get_cached_symbols())
            saved_time = (total_symbols - len(top_assets)) * avg_training_time
            logger.info(f"💡 Resource savings: ~{saved_time:.0f}s saved vs training all {total_symbols} assets")
        except:
            pass
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Error in optimized auto training: {str(e)}")
        return False

def test_optimization_integration():
    """Test the optimization integration"""
    print("🎯 Testing Optimized Training Integration")
    print("=" * 50)
    
    # Test getting top assets
    top_assets = get_optimized_training_assets(max_assets=5)  # Test with 5 assets
    
    if top_assets:
        print(f"✅ Found {len(top_assets)} top assets for training:")
        for i, asset in enumerate(top_assets):
            has_pos = "📍" if asset.get('has_position') else "  "
            print(f"   #{i+1}: {has_pos} {asset['symbol']} - Score: {asset['score']:.1f}")
        
        print(f"\n🚀 The system will now train models for these {len(top_assets)} assets")
        print("   instead of all available assets, saving significant time!")
        
        # Estimate savings
        try:
            from maybe import get_cached_symbols
            total_symbols = len(get_cached_symbols())
            savings_pct = ((total_symbols - len(top_assets)) / total_symbols) * 100
            print(f"\n💡 Estimated resource savings: {savings_pct:.1f}%")
            print(f"   Training {len(top_assets)} / {total_symbols} assets")
        except:
            pass
            
    else:
        print("❌ No assets found for optimization")
    
    return len(top_assets) > 0

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the integration
    success = test_optimization_integration()
    
    if success:
        print("\n🎉 Optimized training integration is ready!")
        print("\nTo use this in your Flask dashboard, replace the auto_retrain_models function")
        print("with run_optimized_auto_training() from this module.")
    else:
        print("\n❌ Optimization integration test failed") 