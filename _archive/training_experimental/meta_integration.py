#!/usr/bin/env python3
"""
Meta-Model Integration Layer
===========================

Integrates the meta-model trading system with the existing trading infrastructure.
Provides seamless integration while maintaining backward compatibility.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the meta-model system
try:
    from meta_trading_system import (
        meta_trading_system, 
        make_meta_trading_decision,
        train_meta_model,
        TradingDecision
    )
    META_SYSTEM_AVAILABLE = True
except ImportError as e:
    META_SYSTEM_AVAILABLE = False
    logging.warning(f"Meta-system not available: {e}")

# Import existing systems for fallback
try:
    from stacking_ml_engine import make_enhanced_ml_decision
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetaModelIntegrator:
    """
    Integration layer for the meta-model system
    Provides a unified interface for trading decisions
    """
    
    def __init__(self):
        self.use_meta_system = META_SYSTEM_AVAILABLE
        self.fallback_systems = []
        
        if STACKING_AVAILABLE:
            self.fallback_systems.append('stacking')
        
        logger.info(f"🔗 Meta-Model Integrator initialized")
        logger.info(f"   Meta-system available: {META_SYSTEM_AVAILABLE}")
        logger.info(f"   Fallback systems: {self.fallback_systems}")
    
    def make_trading_decision(self, symbol: str, granularity: int = 3600, 
                            investment_amount: float = 100.0) -> Dict[str, Any]:
        """
        Make a trading decision using the best available system
        
        Returns a standardized decision format compatible with existing code
        """
        try:
            # Try meta-system first
            if self.use_meta_system and META_SYSTEM_AVAILABLE:
                meta_decision = make_meta_trading_decision(symbol, granularity)
                
                if meta_decision:
                    # Convert to standard format
                    return self._convert_meta_decision(meta_decision, investment_amount)
                else:
                    logger.warning(f"Meta-system failed for {symbol}, trying fallback...")
            
            # Fallback to stacking system
            if STACKING_AVAILABLE:
                logger.info(f"Using stacking system fallback for {symbol}")
                stacking_decision = make_enhanced_ml_decision(symbol, granularity, investment_amount)
                
                if stacking_decision:
                    # Enhance stacking decision with meta-system insights
                    return self._enhance_stacking_decision(stacking_decision, symbol)
            
            # Final fallback - simple decision
            logger.warning(f"All systems failed for {symbol}, using simple fallback")
            return self._create_fallback_decision(symbol, investment_amount)
            
        except Exception as e:
            logger.error(f"💥 Error in trading decision for {symbol}: {str(e)}")
            return self._create_error_decision(symbol, str(e))
    
    def _convert_meta_decision(self, meta_decision: TradingDecision, 
                             investment_amount: float) -> Dict[str, Any]:
        """Convert meta-decision to standard format"""
        
        # Calculate investment amounts
        if meta_decision.action == 'BUY':
            buy_amount = investment_amount * meta_decision.position_size
            sell_amount = 0.0
        elif meta_decision.action == 'SELL':
            buy_amount = 0.0
            sell_amount = investment_amount * meta_decision.position_size
        else:  # HOLD
            buy_amount = 0.0
            sell_amount = 0.0
        
        return {
            'action': meta_decision.action,
            'predicted_return': meta_decision.expected_return,
            'overall_confidence': meta_decision.confidence,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'position_size': meta_decision.position_size,
            'stop_loss_pct': meta_decision.stop_loss,
            'take_profit_pct': meta_decision.take_profit,
            'reasoning': meta_decision.reasoning,
            'risk_score': meta_decision.risk_score,
            'model_consensus': meta_decision.model_consensus,
            'decision_source': 'meta_model',
            'timestamp': meta_decision.timestamp,
            
            # Compatibility fields
            'confidence_buy': meta_decision.confidence if meta_decision.action == 'BUY' else 0.0,
            'confidence_sell': meta_decision.confidence if meta_decision.action == 'SELL' else 0.0,
            'confidence_hold': meta_decision.confidence if meta_decision.action == 'HOLD' else 0.0,
            'model_accuracy': meta_decision.confidence,  # Approximate
            'features': list(meta_decision.model_consensus.get('consensus_metrics', {}).keys()),
            
            # Additional meta-system insights
            'meta_insights': {
                'model_disagreement': meta_decision.model_consensus.get('consensus_metrics', {}).get('disagreement', 0.0),
                'individual_predictions': meta_decision.model_consensus.get('individual_predictions', []),
                'consensus_strength': meta_decision.model_consensus.get('consensus_metrics', {}).get('bullish_consensus', 0.5)
            }
        }
    
    def _enhance_stacking_decision(self, stacking_decision: Dict[str, Any], 
                                 symbol: str) -> Dict[str, Any]:
        """Enhance stacking decision with meta-system insights where possible"""
        
        # Add meta-system metadata
        stacking_decision['decision_source'] = 'stacking_enhanced'
        stacking_decision['meta_system_available'] = META_SYSTEM_AVAILABLE
        
        # Try to get model predictions for additional insights
        if META_SYSTEM_AVAILABLE:
            try:
                predictions = meta_trading_system.collect_model_predictions(symbol)
                if predictions:
                    meta_features = meta_trading_system.create_meta_features(predictions)
                    
                    stacking_decision['meta_insights'] = {
                        'model_disagreement': meta_features.get('model_disagreement', 0.0),
                        'consensus_strength': meta_features.get('consensus_strength', 0.5),
                        'weighted_prediction': meta_features.get('weighted_prediction', 0.0),
                        'num_models_analyzed': meta_features.get('num_models', 1)
                    }
                    
                    # Adjust confidence based on model consensus
                    consensus_factor = 1.0 - meta_features.get('model_disagreement', 0.0)
                    stacking_decision['overall_confidence'] *= consensus_factor
                    
            except Exception as e:
                logger.warning(f"Could not enhance stacking decision with meta-insights: {e}")
        
        return stacking_decision
    
    def _create_fallback_decision(self, symbol: str, investment_amount: float) -> Dict[str, Any]:
        """Create a safe fallback decision"""
        return {
            'action': 'HOLD',
            'predicted_return': 0.0,
            'overall_confidence': 0.1,
            'buy_amount': 0.0,
            'sell_amount': 0.0,
            'position_size': 0.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0,
            'reasoning': 'Fallback decision - insufficient data or system unavailable',
            'risk_score': 1.0,  # Maximum risk for unknown situation
            'decision_source': 'fallback',
            'timestamp': datetime.now(),
            'confidence_buy': 0.0,
            'confidence_sell': 0.0,
            'confidence_hold': 0.1,
            'model_accuracy': 0.0,
            'features': []
        }
    
    def _create_error_decision(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Create an error decision"""
        return {
            'action': 'HOLD',
            'predicted_return': 0.0,
            'overall_confidence': 0.0,
            'buy_amount': 0.0,
            'sell_amount': 0.0,
            'position_size': 0.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 3.0,
            'reasoning': f'Error in decision making: {error_msg}',
            'risk_score': 1.0,
            'decision_source': 'error',
            'timestamp': datetime.now(),
            'error': error_msg,
            'confidence_buy': 0.0,
            'confidence_sell': 0.0,
            'confidence_hold': 0.0,
            'model_accuracy': 0.0,
            'features': []
        }
    
    def train_models(self, symbol: str, granularity: int = 3600) -> Dict[str, bool]:
        """Train all available models for a symbol"""
        results = {}
        
        # Train meta-model
        if META_SYSTEM_AVAILABLE:
            try:
                results['meta_model'] = train_meta_model(symbol, granularity)
                logger.info(f"Meta-model training for {symbol}: {'✅' if results['meta_model'] else '❌'}")
            except Exception as e:
                results['meta_model'] = False
                logger.error(f"Meta-model training failed for {symbol}: {e}")
        
        # Train stacking models
        if STACKING_AVAILABLE:
            try:
                from stacking_ml_engine import train_price_prediction_models
                results['stacking_model'] = train_price_prediction_models(symbol, granularity)
                logger.info(f"Stacking model training for {symbol}: {'✅' if results['stacking_model'] else '❌'}")
            except Exception as e:
                results['stacking_model'] = False
                logger.error(f"Stacking model training failed for {symbol}: {e}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'meta_system_available': META_SYSTEM_AVAILABLE,
            'stacking_available': STACKING_AVAILABLE,
            'fallback_systems': self.fallback_systems,
            'preferred_system': 'meta_model' if META_SYSTEM_AVAILABLE else 'stacking',
            'timestamp': datetime.now()
        }
        
        # Add meta-system specific status
        if META_SYSTEM_AVAILABLE:
            try:
                meta_status = meta_trading_system.get_system_status()
                status['meta_system_status'] = meta_status
            except Exception as e:
                status['meta_system_error'] = str(e)
        
        return status
    
    def compare_predictions(self, symbol: str, granularity: int = 3600) -> Dict[str, Any]:
        """
        Compare predictions from different systems for analysis
        Useful for model evaluation and debugging
        """
        comparison = {
            'symbol': symbol,
            'granularity': granularity,
            'timestamp': datetime.now(),
            'predictions': {}
        }
        
        # Meta-system prediction
        if META_SYSTEM_AVAILABLE:
            try:
                meta_decision = make_meta_trading_decision(symbol, granularity)
                if meta_decision:
                    comparison['predictions']['meta_model'] = {
                        'action': meta_decision.action,
                        'confidence': meta_decision.confidence,
                        'expected_return': meta_decision.expected_return,
                        'risk_score': meta_decision.risk_score
                    }
            except Exception as e:
                comparison['predictions']['meta_model'] = {'error': str(e)}
        
        # Stacking system prediction
        if STACKING_AVAILABLE:
            try:
                stacking_decision = make_enhanced_ml_decision(symbol, granularity, 100.0)
                if stacking_decision:
                    comparison['predictions']['stacking'] = {
                        'action': stacking_decision.get('action', 'HOLD'),
                        'confidence': stacking_decision.get('overall_confidence', 0.0),
                        'expected_return': stacking_decision.get('predicted_return', 0.0),
                        'risk_score': 1.0 - stacking_decision.get('overall_confidence', 0.0)
                    }
            except Exception as e:
                comparison['predictions']['stacking'] = {'error': str(e)}
        
        return comparison

# Global integrator instance
meta_integrator = MetaModelIntegrator()

# Main interface functions for backward compatibility
def make_integrated_trading_decision(symbol: str, granularity: int = 3600, 
                                   investment_amount: float = 100.0) -> Dict[str, Any]:
    """
    Main interface for trading decisions - uses the best available system
    This function can replace existing ML decision functions
    """
    return meta_integrator.make_trading_decision(symbol, granularity, investment_amount)

def train_all_models(symbol: str, granularity: int = 3600) -> Dict[str, bool]:
    """Train all available models for a symbol"""
    return meta_integrator.train_models(symbol, granularity)

def get_integrated_system_status() -> Dict[str, Any]:
    """Get status of all integrated systems"""
    return meta_integrator.get_system_status()

def compare_all_predictions(symbol: str, granularity: int = 3600) -> Dict[str, Any]:
    """Compare predictions from all available systems"""
    return meta_integrator.compare_predictions(symbol, granularity)

if __name__ == "__main__":
    # Test the integration
    logger.info("🧪 Testing Meta-Model Integration")
    
    test_symbol = "BTC-USD"
    
    # Test system status
    status = get_integrated_system_status()
    logger.info(f"System status: {status}")
    
    # Test decision making
    decision = make_integrated_trading_decision(test_symbol, investment_amount=100.0)
    logger.info(f"Decision: {decision['action']} (confidence: {decision['overall_confidence']:.2f})")
    
    # Test comparison
    comparison = compare_all_predictions(test_symbol)
    logger.info(f"Prediction comparison: {comparison}")
    
    logger.info("✅ Integration test completed")
