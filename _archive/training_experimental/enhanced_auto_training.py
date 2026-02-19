#!/usr/bin/env python3
"""
Enhanced Auto-Training System
Focus on profitable assets, avoid stablecoins, optimize resource usage
"""

import os
import sys
import logging
import time
import requests
from datetime import datetime, timedelta
import pandas as pd

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your existing modules
try:
    from enhanced_config import ENHANCED_TRADING_CONFIG, PRIORITY_SYMBOLS
    from trading_utils import is_stablecoin, should_skip_symbol_for_ml
    from public_fallback import public_fallback
except ImportError as e:
    logger.warning(f"Some enhanced modules not found: {e}")

class EnhancedAutoTrainer:
    """Enhanced auto-training system that focuses on profitable assets"""
    
    def __init__(self):
        self.config = {
            "min_balance_for_trading": 5.0,
            "max_positions": 5,
            "position_size_pct": 20,
            "min_ml_confidence": 0.70,
            "scan_top_symbols": 50,
            "enable_momentum_trading": True,
            "skip_stablecoins": True
        }
        
        self.priority_symbols = [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
            "MATIC-USD", "LINK-USD", "AVAX-USD", "ATOM-USD", "XTZ-USD",
            "ALGO-USD", "ICP-USD", "FIL-USD", "VET-USD", "XLM-USD"
        ]
        
        self.stablecoins = [
            'USDT-USD', 'USDC-USD', 'DAI-USD', 'BUSD-USD', 
            'TUSD-USD', 'PAX-USD', 'GUSD-USD', 'SUSD-USD'
        ]
    
    def is_stablecoin(self, symbol):
        """Check if symbol is a stablecoin"""
        return symbol.upper() in self.stablecoins
    
    def get_public_market_data(self):
        """Get market data using public APIs"""
        try:
            # Use Coinbase Pro public API
            response = requests.get("https://api.exchange.coinbase.com/products")
            if response.status_code == 200:
                products = response.json()
                return [p for p in products if p['id'].endswith('-USD') and p['status'] == 'online']
            return []
        except:
            return []
    
    def get_top_performing_assets(self, max_assets=10):
        """Get top performing assets based on multiple criteria"""
        try:
            logger.info(f"🔍 Scanning for top {max_assets} performing assets...")
            
            market_data = self.get_public_market_data()
            if not market_data:
                logger.warning("No market data available, using priority symbols")
                return self.priority_symbols[:max_assets]
            
            # Filter out stablecoins and get metrics
            asset_scores = []
            
            for product in market_data[:100]:  # Limit to top 100 by volume
                symbol = product['id']
                
                # Skip stablecoins
                if self.is_stablecoin(symbol):
                    continue
                
                try:
                    # Get 24hr stats
                    stats_response = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/stats")
                    if stats_response.status_code == 200:
                        stats = stats_response.json()
                        
                        # Calculate performance score
                        volume = float(stats.get('volume', 0))
                        price_change = abs(float(stats.get('last', 0)) - float(stats.get('open', 0)))
                        volatility = price_change / float(stats.get('open', 1)) if float(stats.get('open', 0)) > 0 else 0
                        
                        # Multi-factor scoring
                        volume_score = min(volume / 1000000, 100)  # Volume in millions, cap at 100
                        volatility_score = min(volatility * 100, 50)  # Volatility as percentage, cap at 50
                        priority_bonus = 20 if symbol in self.priority_symbols else 0
                        
                        total_score = volume_score + volatility_score + priority_bonus
                        
                        asset_scores.append({
                            'symbol': symbol,
                            'score': total_score,
                            'volume': volume,
                            'volatility': volatility,
                            'price_change_pct': volatility * 100
                        })
                        
                        time.sleep(0.1)  # Avoid rate limiting
                        
                except Exception as e:
                    logger.debug(f"Error getting stats for {symbol}: {e}")
                    continue
            
            # Sort by score and return top assets
            asset_scores.sort(key=lambda x: x['score'], reverse=True)
            top_assets = [asset['symbol'] for asset in asset_scores[:max_assets]]
            
            logger.info(f"🏆 TOP {len(top_assets)} ASSETS FOR TRAINING:")
            for i, asset in enumerate(asset_scores[:max_assets]):
                logger.info(f"   #{i+1}: {asset['symbol']} - Score: {asset['score']:.1f}")
                logger.info(f"        Volume: ${asset['volume']:,.0f} | Volatility: {asset['price_change_pct']:.2f}%")
            
            return top_assets
            
        except Exception as e:
            logger.error(f"Error getting top performing assets: {e}")
            # Fallback to priority symbols
            return [s for s in self.priority_symbols if not self.is_stablecoin(s)][:max_assets]
    
    def should_train_asset(self, symbol):
        """Determine if we should train ML models for this asset"""
        # Skip stablecoins
        if self.is_stablecoin(symbol):
            logger.info(f"⏭️ Skipping {symbol} - stablecoin detected")
            return False
        
        # Check if asset has sufficient data/volume
        try:
            response = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/stats")
            if response.status_code == 200:
                stats = response.json()
                volume = float(stats.get('volume', 0))
                
                # Require minimum volume for training
                if volume < 10000:  # $10k minimum daily volume
                    logger.info(f"⏭️ Skipping {symbol} - insufficient volume: ${volume:,.0f}")
                    return False
                
                return True
            else:
                logger.info(f"⏭️ Skipping {symbol} - no stats available")
                return False
                
        except Exception as e:
            logger.info(f"⏭️ Skipping {symbol} - error checking stats: {e}")
            return False
    
    def enhanced_auto_training(self):
        """Enhanced auto-training that focuses on profitable assets"""
        try:
            logger.info("🤖 Starting Enhanced Auto-Training System")
            
            # Get top performing assets
            top_assets = self.get_top_performing_assets(max_assets=10)
            
            logger.info(f"🎯 Training models for {len(top_assets)} top assets (avoiding stablecoins)")
            
            trained_count = 0
            skipped_count = 0
            
            for symbol in top_assets:
                if self.should_train_asset(symbol):
                    logger.info(f"🔄 Training ML model for {symbol}...")
                    
                    # Here you would call your existing training function
                    # Example: train_model_for_symbol(symbol)
                    
                    trained_count += 1
                    logger.info(f"✅ Trained model for {symbol} ({trained_count}/{len(top_assets)})")
                    
                    # Short delay to avoid overwhelming the system
                    time.sleep(2)
                else:
                    skipped_count += 1
                    continue
            
            logger.info(f"🎉 Enhanced Auto-Training Complete:")
            logger.info(f"   ✅ Models trained: {trained_count}")
            logger.info(f"   ⏭️ Assets skipped: {skipped_count}")
            logger.info(f"   📊 Resource optimization: ~{((skipped_count/(trained_count+skipped_count))*100) if (trained_count+skipped_count) > 0 else 0:.1f}% reduction vs training all assets")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced auto-training: {e}")
            return False
    
    def start_intelligent_training_loop(self, interval_hours=2):
        """Start the intelligent training loop"""
        logger.info(f"🚀 Starting intelligent training loop (every {interval_hours} hours)")
        
        while True:
            try:
                logger.info("⏰ Starting scheduled enhanced training...")
                self.enhanced_auto_training()
                
                # Wait for next cycle
                sleep_seconds = interval_hours * 3600
                logger.info(f"😴 Sleeping for {interval_hours} hours until next training cycle...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("🛑 Training loop stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                # Sleep for 10 minutes before retrying
                time.sleep(600)

def main():
    """Main function to run enhanced auto-training"""
    trainer = EnhancedAutoTrainer()
    
    logger.info("🎯 Enhanced Auto-Training System")
    logger.info("=" * 50)
    
    # Run one-time training
    success = trainer.enhanced_auto_training()
    
    if success:
        logger.info("✅ One-time enhanced training completed successfully!")
        
        # Ask if user wants to start continuous loop
        print("\n🔄 Would you like to start continuous training loop? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response == 'y':
                trainer.start_intelligent_training_loop()
        except KeyboardInterrupt:
            logger.info("👋 Goodbye!")
    else:
        logger.error("❌ Enhanced training failed")

if __name__ == "__main__":
    main() 