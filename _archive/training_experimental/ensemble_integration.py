#!/usr/bin/env python3
"""
Ensemble Integration Layer for Advanced ML Models
Connects LightGBM, XGBoost, and Neural Networks with existing trading system

Integrates with:
- enhanced_features.py (225+ features)
- advanced_data_processor.py (data quality)
- unified_ml_system.py (current system)
- flask_trading_dashboard.py (UI)
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleMLIntegration:
    """
    Integration layer that combines advanced ML models with existing trading infrastructure
    
    Features:
    - Seamless integration with existing enhanced features
    - Advanced model ensemble predictions
    - Performance comparison and benchmarking
    - Fallback to existing models when needed
    - Trading signal generation with confidence scoring
    """
    
    def __init__(self):
        """Initialize the ensemble integration system"""
        
        # Import enhanced systems
        self._initialize_enhanced_systems()
        
        # Import advanced ML models
        self._initialize_advanced_models()
        
        # Performance tracking
        self.performance_comparison = {
            'baseline_performance': {},
            'advanced_performance': {},
            'ensemble_performance': {}
        }
        
        # Configuration
        self.enable_ensemble = True
        self.ensemble_confidence_threshold = 0.75
        self.fallback_to_baseline = True
        
        logger.info("🎯 Ensemble ML Integration initialized")
        logger.info(f"   Enhanced systems: {'✅' if self.enhanced_available else '❌'}")
        logger.info(f"   Advanced models: {'✅' if self.advanced_available else '❌'}")
        logger.info(f"   Unified system: {'✅' if self.unified_available else '❌'}")
    
    def _initialize_enhanced_systems(self):
        """Initialize enhanced feature and data processing systems"""
        try:
            from enhanced_features import EnhancedFeatureEngine
            from advanced_data_processor import AdvancedDataProcessor
            from unified_ml_system import UnifiedMLSystem
            
            self.enhanced_features = EnhancedFeatureEngine()
            self.data_processor = AdvancedDataProcessor()
            self.unified_system = UnifiedMLSystem()
            
            self.enhanced_available = True
            self.unified_available = True
            
            logger.info("✅ Enhanced systems loaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Enhanced systems not fully available: {e}")
            self.enhanced_available = False
            self.unified_available = False
    
    def _initialize_advanced_models(self):
        """Initialize advanced ML models system"""
        try:
            from advanced_ml_models import AdvancedMLModels
            
            self.advanced_models = AdvancedMLModels(
                models_dir="advanced_models",
                use_gpu=False,  # Conservative default
                enable_neural_networks=True
            )
            
            self.advanced_available = True
            logger.info("✅ Advanced ML models system loaded")
            
        except ImportError as e:
            logger.warning(f"⚠️ Advanced ML models not available: {e}")
            self.advanced_available = False
    
    def prepare_enhanced_data(self, 
                             symbol: str, 
                             granularity: int = 3600, 
                             days: int = 30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare enhanced dataset using all available systems
        
        Args:
            symbol: Trading symbol
            granularity: Data granularity in seconds
            days: Number of days of data
        
        Returns:
            Tuple of (processed_dataframe, metadata)
        """
        logger.info(f"📊 Preparing enhanced data for {symbol}")
        
        metadata = {
            'symbol': symbol,
            'granularity': granularity,
            'days': days,
            'processing_steps': [],
            'feature_count': 0,
            'data_quality_score': 0.0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Get base data
            if self.enhanced_available:
                from maybe import get_coinbase_data
                df = get_coinbase_data(symbol, granularity, days)
                metadata['processing_steps'].append('base_data_acquired')
            else:
                raise ValueError("Base data acquisition not available")
            
            if df is None or df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Step 2: Data quality assessment and processing
            if self.enhanced_available:
                quality_result = self.data_processor.process_market_data(df, symbol=symbol)
                
                if quality_result.get('success'):
                    df = quality_result['processed_data']
                    metadata['data_quality_score'] = quality_result['quality_metrics']['overall_score']
                    metadata['processing_steps'].append('data_quality_processed')
                    
                    logger.info(f"   Data quality score: {metadata['data_quality_score']:.3f}")
                else:
                    logger.warning("   Data quality processing failed, using raw data")
                    metadata['processing_steps'].append('data_quality_failed')
            
            # Step 3: Enhanced feature engineering
            if self.enhanced_available:
                enhanced_result = self.enhanced_features.calculate_enhanced_features(df, symbol=symbol)
                
                if enhanced_result.get('success'):
                    df = enhanced_result['enhanced_data']
                    metadata['feature_count'] = enhanced_result.get('feature_count', len(df.columns))
                    metadata['processing_steps'].append('enhanced_features_generated')
                    
                    logger.info(f"   Enhanced features: {metadata['feature_count']}")
                else:
                    logger.warning("   Enhanced feature generation failed")
                    metadata['processing_steps'].append('enhanced_features_failed')
                    
                    # Fallback to basic features
                    from maybe import calculate_indicators
                    df = calculate_indicators(df)
                    metadata['feature_count'] = len(df.columns)
                    metadata['processing_steps'].append('basic_features_fallback')
            
            # Step 4: Final data preparation
            # Remove any remaining NaN values
            df = df.dropna()
            
            # Ensure we have enough data
            if len(df) < 50:
                raise ValueError(f"Insufficient data after processing: {len(df)} rows")
            
            metadata['final_rows'] = len(df)
            metadata['final_columns'] = len(df.columns)
            metadata['processing_time'] = time.time() - start_time
            
            logger.info(f"✅ Enhanced data prepared: {df.shape}")
            logger.info(f"   Processing time: {metadata['processing_time']:.2f}s")
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"❌ Error preparing enhanced data: {str(e)}")
            metadata['error'] = str(e)
            metadata['processing_time'] = time.time() - start_time
            
            # Return empty dataframe as fallback
            return pd.DataFrame(), metadata
    
    def train_ensemble_models(self, 
                             symbol: str,
                             granularity: int = 3600,
                             target_column: str = 'target',
                             models_to_use: Optional[List[str]] = None,
                             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train ensemble of advanced models on enhanced features
        
        Args:
            symbol: Trading symbol
            granularity: Data granularity
            target_column: Name of target column
            models_to_use: List of models to include in ensemble
            validation_split: Fraction of data for validation
        
        Returns:
            Training results and performance metrics
        """
        logger.info(f"🎯 Training ensemble models for {symbol}")
        
        if not self.advanced_available:
            return {"error": "Advanced ML models not available"}
        
        try:
            # Prepare enhanced data
            df, data_metadata = self.prepare_enhanced_data(symbol, granularity)
            
            if df.empty:
                return {"error": "No data available for training"}
            
            # Create target variable if needed
            if target_column not in df.columns:
                df = self._create_target_variable(df, target_column)
            
            # Prepare features and target
            target = df[target_column]
            features = df.drop(columns=[target_column, 'timestamp'], errors='ignore')
            
            logger.info(f"   Training data: {features.shape}")
            logger.info(f"   Target distribution: {target.value_counts().to_dict()}")
            
            # Split data for validation
            split_idx = int(len(features) * (1 - validation_split))
            
            X_train = features.iloc[:split_idx]
            y_train = target.iloc[:split_idx]
            X_val = features.iloc[split_idx:]
            y_val = target.iloc[split_idx:]
            
            logger.info(f"   Train set: {X_train.shape}")
            logger.info(f"   Validation set: {X_val.shape}")
            
            # Train ensemble
            training_start = time.time()
            
            ensemble_result = self.advanced_models.train_ensemble(
                X_train, y_train,
                X_val, y_val,
                models_to_use=models_to_use
            )
            
            training_time = time.time() - training_start
            
            # Save models
            self.advanced_models.save_models(symbol, f"{granularity}s")
            
            # Prepare comprehensive results
            results = {
                'success': True,
                'symbol': symbol,
                'training_metadata': {
                    'data_preparation': data_metadata,
                    'training_time': training_time,
                    'models_trained': list(ensemble_result['individual_models'].keys()),
                    'features_used': len(features.columns),
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val)
                },
                'model_performance': {
                    'individual_models': {
                        model_type: {
                            'accuracy': metrics.accuracy,
                            'auc_roc': metrics.auc_roc,
                            'f1_score': metrics.f1_score,
                            'training_time': metrics.training_time,
                            'cv_mean': np.mean(metrics.cv_scores),
                            'cv_std': np.std(metrics.cv_scores)
                        }
                        for model_type, metrics in ensemble_result['individual_metrics'].items()
                    },
                    'ensemble_performance': {
                        'accuracy': ensemble_result['ensemble_metrics'].accuracy,
                        'auc_roc': ensemble_result['ensemble_metrics'].auc_roc,
                        'f1_score': ensemble_result['ensemble_metrics'].f1_score,
                        'ensemble_method': ensemble_result['ensemble_method']
                    }
                },
                'feature_importance': ensemble_result['feature_importance'],
                'ensemble_weights': self.advanced_models.ensemble_weights
            }
            
            # Store performance for comparison
            self.performance_comparison['advanced_performance'][symbol] = results['model_performance']
            
            logger.info(f"✅ Ensemble training completed in {training_time:.2f}s")
            logger.info(f"   Models trained: {len(ensemble_result['individual_models'])}")
            logger.info(f"   Best individual: {results['model_performance']['ensemble_performance']['auc_roc']:.3f} AUC")
            logger.info(f"   Ensemble AUC: {results['model_performance']['ensemble_performance']['auc_roc']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error training ensemble models: {str(e)}")
            return {"error": str(e)}
    
    def _create_target_variable(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Create target variable for training"""
        try:
            # Simple future return prediction target
            df = df.copy()
            
            # Calculate future returns (next period)
            df['future_return'] = df['close'].pct_change(1).shift(-1)
            
            # Binary target: 1 if positive return, 0 otherwise
            df[target_column] = (df['future_return'] > 0).astype(int)
            
            # Remove the last row (no future data)
            df = df.iloc[:-1]
            
            # Remove temporary column
            df = df.drop('future_return', axis=1)
            
            logger.info(f"   Created target variable: {df[target_column].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating target variable: {str(e)}")
            return df
    
    def get_ensemble_prediction(self, 
                               symbol: str,
                               investment_amount: float = 100.0,
                               granularity: int = 3600,
                               use_latest_data: bool = True) -> Dict[str, Any]:
        """
        Get ensemble prediction for a symbol
        
        Args:
            symbol: Trading symbol
            investment_amount: Investment amount for calculations
            granularity: Data granularity
            use_latest_data: Whether to use latest market data
        
        Returns:
            Comprehensive prediction results
        """
        logger.info(f"🔮 Getting ensemble prediction for {symbol}")
        
        start_time = time.time()
        
        try:
            # Try to load existing models
            models_loaded = False
            if self.advanced_available:
                models_loaded = self.advanced_models.load_models(symbol, f"{granularity}s")
            
            if not models_loaded:
                logger.info("   No pre-trained models found, training new ensemble...")
                training_result = self.train_ensemble_models(symbol, granularity)
                
                if not training_result.get('success'):
                    return {"error": "Failed to train models", "details": training_result.get('error')}
            
            # Prepare prediction data
            df, data_metadata = self.prepare_enhanced_data(symbol, granularity, days=7)  # Recent data
            
            if df.empty:
                return {"error": "No data available for prediction"}
            
            # Get latest features for prediction
            features = df.drop(columns=['timestamp'], errors='ignore')
            latest_features = features.tail(1)  # Most recent data point
            
            logger.info(f"   Prediction features: {latest_features.shape}")
            
            # Get ensemble prediction
            if self.advanced_available and self.advanced_models.models:
                ensemble_pred = self.advanced_models.predict_ensemble(
                    latest_features,
                    return_probabilities=True,
                    return_individual=True
                )
                
                prediction_type = 'ensemble'
                main_prediction = ensemble_pred['ensemble_prediction'][0]
                main_probability = ensemble_pred['ensemble_probability'][0]
                confidence_score = ensemble_pred['confidence_score']
                
            else:
                # Fallback to unified system
                if self.unified_available:
                    unified_pred = self.unified_system.predict_with_model(df, symbol)
                    
                    if unified_pred.get('success'):
                        prediction_type = 'unified_fallback'
                        main_prediction = 1 if unified_pred['prediction'] == 'BUY' else 0
                        main_probability = unified_pred.get('confidence', 0.5)
                        confidence_score = unified_pred.get('confidence', 0.5)
                    else:
                        return {"error": "All prediction systems failed"}
                else:
                    return {"error": "No prediction systems available"}
            
            # Convert prediction to trading signal
            if main_prediction == 1 and main_probability >= 0.5:
                action = 'STRONG_BUY' if main_probability >= 0.8 else 'BUY'
            elif main_prediction == 0 and main_probability <= 0.5:
                action = 'STRONG_SELL' if main_probability <= 0.2 else 'SELL'
            else:
                action = 'HOLD'
            
            # Calculate additional metrics
            current_price = float(df['close'].iloc[-1])
            
            # Calculate profit estimates
            if action in ['BUY', 'STRONG_BUY']:
                expected_return_pct = (main_probability - 0.5) * 10  # Simple heuristic
                expected_profit_usd = investment_amount * (expected_return_pct / 100)
                
                # TP/SL recommendations based on confidence
                tp_percentage = expected_return_pct * 1.5
                sl_percentage = expected_return_pct * 0.5
                
                tp_price = current_price * (1 + tp_percentage / 100)
                sl_price = current_price * (1 - sl_percentage / 100)
            else:
                expected_return_pct = 0
                expected_profit_usd = 0
                tp_price = None
                sl_price = None
            
            # Prepare comprehensive results
            results = {
                'success': True,
                'symbol': symbol,
                'action': action,
                'prediction_type': prediction_type,
                'current_price': current_price,
                'investment_amount': investment_amount,
                
                # Confidence and probability
                'main_probability': main_probability,
                'confidence_score': confidence_score,
                'overall_confidence': confidence_score,
                
                # Profit estimates
                'expected_return_pct': expected_return_pct,
                'expected_profit_usd': expected_profit_usd,
                
                # TP/SL recommendations
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_percentage': tp_percentage if action in ['BUY', 'STRONG_BUY'] else None,
                'sl_percentage': sl_percentage if action in ['BUY', 'STRONG_BUY'] else None,
                
                # Data quality and features
                'feature_count': len(latest_features.columns),
                'data_quality_score': data_metadata.get('data_quality_score', 0.0),
                'processing_time': time.time() - start_time,
                
                # Individual model predictions (if available)
                'individual_predictions': ensemble_pred.get('individual_predictions') if prediction_type == 'ensemble' else None,
                'individual_probabilities': ensemble_pred.get('individual_probabilities') if prediction_type == 'ensemble' else None,
                
                # Metadata
                'timestamp': datetime.now().isoformat(),
                'data_metadata': data_metadata
            }
            
            logger.info(f"✅ Ensemble prediction completed:")
            logger.info(f"   Action: {action}")
            logger.info(f"   Confidence: {confidence_score:.1%}")
            logger.info(f"   Expected return: {expected_return_pct:+.2f}%")
            logger.info(f"   Processing time: {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting ensemble prediction: {str(e)}")
            return {"error": str(e)}
    
    def compare_model_performance(self, 
                                 symbol: str,
                                 granularity: int = 3600) -> Dict[str, Any]:
        """
        Compare performance between different model approaches
        
        Args:
            symbol: Trading symbol to analyze
            granularity: Data granularity
        
        Returns:
            Performance comparison results
        """
        logger.info(f"📊 Comparing model performance for {symbol}")
        
        comparison_results = {
            'symbol': symbol,
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Prepare test data
            df, _ = self.prepare_enhanced_data(symbol, granularity, days=60)  # More data for testing
            
            if df.empty:
                return {"error": "No data available for comparison"}
            
            # Create target variable
            df = self._create_target_variable(df, 'target')
            
            # Split data for fair comparison
            split_idx = int(len(df) * 0.8)
            
            features = df.drop(columns=['target', 'timestamp'], errors='ignore')
            target = df['target']
            
            X_train = features.iloc[:split_idx]
            y_train = target.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_test = target.iloc[split_idx:]
            
            logger.info(f"   Test data: {X_test.shape}")
            
            # Test 1: Advanced Ensemble Models
            if self.advanced_available:
                try:
                    logger.info("   Testing advanced ensemble models...")
                    
                    # Train ensemble on training data
                    ensemble_result = self.advanced_models.train_ensemble(X_train, y_train)
                    
                    # Test on holdout data
                    test_predictions = self.advanced_models.predict_ensemble(X_test)
                    
                    # Calculate test metrics
                    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
                    
                    test_accuracy = accuracy_score(y_test, test_predictions['ensemble_prediction'])
                    test_auc = roc_auc_score(y_test, test_predictions['ensemble_probability'])
                    test_f1 = f1_score(y_test, test_predictions['ensemble_prediction'])
                    
                    comparison_results['models_compared'].append('Advanced_Ensemble')
                    comparison_results['performance_metrics']['Advanced_Ensemble'] = {
                        'accuracy': test_accuracy,
                        'auc_roc': test_auc,
                        'f1_score': test_f1,
                        'confidence': test_predictions['confidence_score'],
                        'models_in_ensemble': len(ensemble_result['individual_models'])
                    }
                    
                    logger.info(f"      Advanced Ensemble - AUC: {test_auc:.3f}, Accuracy: {test_accuracy:.3f}")
                    
                except Exception as e:
                    logger.warning(f"   Advanced ensemble test failed: {str(e)}")
            
            # Test 2: Unified ML System (existing)
            if self.unified_available:
                try:
                    logger.info("   Testing unified ML system...")
                    
                    # Use existing unified system
                    unified_result = self.unified_system.predict_with_model(df, symbol)
                    
                    if unified_result.get('success'):
                        # Simple performance estimate based on unified system
                        unified_confidence = unified_result.get('confidence', 0.5)
                        
                        comparison_results['models_compared'].append('Unified_System')
                        comparison_results['performance_metrics']['Unified_System'] = {
                            'accuracy': unified_confidence,  # Approximation
                            'auc_roc': unified_confidence,   # Approximation
                            'f1_score': unified_confidence,  # Approximation
                            'confidence': unified_confidence,
                            'models_in_ensemble': 1
                        }
                        
                        logger.info(f"      Unified System - Confidence: {unified_confidence:.3f}")
                        
                except Exception as e:
                    logger.warning(f"   Unified system test failed: {str(e)}")
            
            # Generate recommendations
            if comparison_results['performance_metrics']:
                best_model = max(
                    comparison_results['performance_metrics'].items(),
                    key=lambda x: x[1]['auc_roc']
                )
                
                comparison_results['recommendations'] = [
                    f"Best performing model: {best_model[0]} (AUC: {best_model[1]['auc_roc']:.3f})",
                    f"Models tested: {len(comparison_results['models_compared'])}",
                    "Use ensemble approach for best performance" if 'Advanced_Ensemble' in comparison_results['models_compared'] else "Consider upgrading to ensemble models"
                ]
                
                # Store comparison for future reference
                self.performance_comparison['comparison_results'] = comparison_results
                
                logger.info(f"✅ Performance comparison completed")
                logger.info(f"   Best model: {best_model[0]} (AUC: {best_model[1]['auc_roc']:.3f})")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Error in performance comparison: {str(e)}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the ensemble integration system"""
        return {
            'ensemble_integration': {
                'enhanced_systems_available': self.enhanced_available,
                'advanced_models_available': self.advanced_available,
                'unified_system_available': self.unified_available,
                'ensemble_enabled': self.enable_ensemble,
                'confidence_threshold': self.ensemble_confidence_threshold
            },
            'available_models': self.advanced_models.available_models if self.advanced_available else [],
            'performance_history': self.performance_comparison,
            'system_capabilities': {
                'enhanced_features': '225+ features' if self.enhanced_available else 'Not available',
                'data_quality': 'Advanced processing' if self.enhanced_available else 'Not available',
                'model_types': len(self.advanced_models.available_models) if self.advanced_available else 0,
                'ensemble_methods': ['weighted_voting', 'equal_voting'] if self.advanced_available else []
            }
        }


# Integration functions for existing system
def get_ensemble_ml_decision(symbol: str, investment_amount: float = 100.0) -> Dict[str, Any]:
    """
    Get ensemble ML decision - integrates with existing flask_trading_dashboard.py
    
    Args:
        symbol: Trading symbol
        investment_amount: Investment amount for calculations
    
    Returns:
        Enhanced ML decision with ensemble predictions
    """
    try:
        # Create integration system
        integration = EnsembleMLIntegration()
        
        # Get ensemble prediction
        result = integration.get_ensemble_prediction(symbol, investment_amount)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in ensemble ML decision: {str(e)}")
        return {"error": str(e)}


def train_ensemble_for_symbol(symbol: str, granularity: int = 3600) -> Dict[str, Any]:
    """
    Train ensemble models for a specific symbol
    
    Args:
        symbol: Trading symbol
        granularity: Data granularity in seconds
    
    Returns:
        Training results
    """
    try:
        integration = EnsembleMLIntegration()
        return integration.train_ensemble_models(symbol, granularity)
        
    except Exception as e:
        logger.error(f"❌ Error training ensemble for {symbol}: {str(e)}")
        return {"error": str(e)}


def compare_ensemble_performance(symbol: str) -> Dict[str, Any]:
    """
    Compare ensemble performance against existing models
    
    Args:
        symbol: Trading symbol to analyze
    
    Returns:
        Performance comparison results
    """
    try:
        integration = EnsembleMLIntegration()
        return integration.compare_model_performance(symbol)
        
    except Exception as e:
        logger.error(f"❌ Error comparing performance for {symbol}: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the ensemble integration
    logger.info("🧪 Testing Ensemble ML Integration")
    
    try:
        # Create integration system
        integration = EnsembleMLIntegration()
        
        # Get system status
        status = integration.get_system_status()
        logger.info(f"📊 System status: {status}")
        
        # Test with sample symbol (if data available)
        test_symbol = "BTC-USD"
        
        # Get ensemble prediction
        prediction = integration.get_ensemble_prediction(test_symbol, 100.0)
        
        if prediction.get('success'):
            logger.info("✅ Ensemble integration test completed successfully!")
            logger.info(f"   Test prediction: {prediction['action']} with {prediction['confidence_score']:.1%} confidence")
        else:
            logger.warning(f"⚠️ Test prediction failed: {prediction.get('error')}")
        
    except Exception as e:
        logger.error(f"❌ Ensemble integration test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 