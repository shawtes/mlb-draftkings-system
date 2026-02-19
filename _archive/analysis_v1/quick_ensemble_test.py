#!/usr/bin/env python3
"""
Quick Ensemble Model Test
Fast evaluation of key ensemble models with comprehensive metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from maybe import get_coinbase_data, calculate_indicators
import logging

# Import ML models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using GradientBoosting")
    XGBRegressor = GradientBoostingRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ensemble_models():
    """Create key ensemble models for evaluation"""
    
    # Base models for ensembles - with consistent parameters
    base_models = [
        ('lr', LinearRegression()),
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)),
        ('xgb', XGBRegressor(n_estimators=50, random_state=42, max_depth=6) if XGBOOST_AVAILABLE else GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6))
    ]

    # Stacking Regressor with cross-validation
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=0.1),  # Regularized final estimator
        cv=3,
        n_jobs=1  # Prevent parallel issues
    )

    # Voting Regressor - will be handled separately with proper scaling
    voting_model = VotingRegressor(
        estimators=base_models,
        n_jobs=1  # Prevent parallel issues
    )

    # Models to evaluate
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, max_depth=8) if XGBOOST_AVAILABLE else GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=8),
        'StackingRegressor': stacking_model,
        'VotingRegressor': voting_model
    }
    
    return models

def prepare_features_simple(df):
    """Simple feature preparation that works with any dataset"""
    try:
        # Calculate future returns (target)
        df['future_return_pct'] = (df['close'].shift(-1) / df['close'] - 1) * 100
        
        # Get all numeric columns except core price data
        exclude_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'future_return_pct'}
        
        # Find all numeric feature columns
        feature_columns = []
        for col in df.columns:
            if col not in exclude_columns and df[col].dtype in ['float64', 'int64']:
                feature_columns.append(col)
        
        # Add basic price features if available
        if len(feature_columns) == 0:
            # Fallback to basic price features
            df['price_change'] = df['close'].pct_change() * 100
            df['volume_change'] = df['volume'].pct_change() * 100
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            feature_columns = ['price_change', 'volume_change', 'high_low_ratio', 'price_position']
        
        # Clean data
        df = df.dropna()
        
        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} rows")
            return None, None, []
        
        # Prepare features and target
        X = df[feature_columns].fillna(0)
        y = df['future_return_pct'].fillna(0)
        
        # Remove extreme outliers
        q1, q99 = y.quantile([0.05, 0.95])
        mask = (y >= q1) & (y <= q99)
        X = X[mask]
        y = y[mask]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        
        return X, y, feature_columns
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None, None, []

def calculate_comprehensive_metrics(y_true, y_pred, model_name):
    """Calculate all the metrics you requested"""
    try:
        # Validate inputs
        if len(y_true) != len(y_pred):
            logger.error(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
            return None
            
        # Remove any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            logger.error(f"No valid data points for {model_name}")
            return None
            
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 10:
            logger.warning(f"Insufficient valid data for {model_name}: {len(y_true_clean)} points")
            return None
        
        # Basic regression metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        
        # Handle R² calculation carefully
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
            # Cap extremely negative R² values
            if r2 < -1000:
                logger.warning(f"Extremely negative R² for {model_name}: {r2}, capping at -1000")
                r2 = -1000
        except Exception as e:
            logger.warning(f"Error calculating R² for {model_name}: {str(e)}")
            r2 = -999
        
        # MAE as percentage
        y_true_abs_mean = np.mean(np.abs(y_true_clean))
        mae_percent = (mae / y_true_abs_mean) * 100 if y_true_abs_mean > 0 else float('inf')
        
        # MAPE (Mean Absolute Percentage Error) - handle division by zero
        denominator = np.where(np.abs(y_true_clean) > 1e-8, y_true_clean, 1e-8)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / denominator)) * 100
        
        # Cap extreme MAPE values
        if mape > 10000:
            mape = 10000
        
        # Directional accuracy
        direction_true = np.sign(y_true_clean)
        direction_pred = np.sign(y_pred_clean)
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Hit rates
        within_1_pct = np.mean(np.abs(y_true_clean - y_pred_clean) <= 1.0) * 100
        within_2_pct = np.mean(np.abs(y_true_clean - y_pred_clean) <= 2.0) * 100
        
        # Correlation
        try:
            correlation = np.corrcoef(y_true_clean, y_pred_clean)[0, 1] if len(y_true_clean) > 1 else 0
            if np.isnan(correlation):
                correlation = 0
        except Exception:
            correlation = 0
        
        return {
            'model': model_name,
            'r2_score': round(r2, 4),
            'mae': round(mae, 4),
            'mae_percent': round(mae_percent, 2) if not np.isinf(mae_percent) else 999.99,
            'mse': round(mse, 4),
            'rmse': round(rmse, 4),
            'mape': round(mape, 2) if not np.isinf(mape) else 999.99,
            'directional_accuracy': round(directional_accuracy, 2),
            'within_1_pct': round(within_1_pct, 2),
            'within_2_pct': round(within_2_pct, 2),
            'correlation': round(correlation, 4),
            'samples': len(y_true_clean)
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
        return None

def validate_predictions(y_pred, model_name):
    """Validate predictions for numerical stability"""
    try:
        # Check for NaN or infinite values
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            logger.warning(f"❌ {model_name}: Invalid predictions detected (NaN/Inf)")
            return False
            
        # Check for extreme values
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        
        if pred_std > 1000 or abs(pred_mean) > 1000:
            logger.warning(f"❌ {model_name}: Extreme predictions detected (std={pred_std:.2f}, mean={pred_mean:.2f})")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ {model_name}: Error validating predictions: {str(e)}")
        return False

def train_and_predict_safe(model, X_train, y_train, X_test, model_name):
    """Safely train model and make predictions with validation"""
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Validate predictions
        if not validate_predictions(y_pred, model_name):
            return None
            
        return y_pred
        
    except Exception as e:
        logger.error(f"❌ {model_name}: Training/prediction error: {str(e)}")
        return None

def quick_ensemble_evaluation():
    """Quick evaluation of ensemble models"""
    logger.info("🚀 Starting Quick Ensemble ML Evaluation")
    
    # Test symbol
    symbol = "BTC-USD"
    
    logger.info(f"📊 Testing ensemble models on {symbol}")
    
    try:
        # Get data
        df = get_coinbase_data(symbol, 3600, days=30)  # 30 days for speed
        if df is None or df.empty:
            logger.error(f"No data available for {symbol}")
            return
        
        # Calculate indicators
        df = calculate_indicators(df, symbol=symbol)
        
        # Prepare features
        X, y, feature_names = prepare_features_simple(df)
        
        if X is None:
            logger.error("Feature preparation failed")
            return
        
        # Split data (temporal split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Create models
        models = create_ensemble_models()
        
        results = []
        
        # Evaluate each model
        for model_name, model in models.items():
            try:
                logger.info(f"🤖 Training {model_name}...")
                
                # Special handling for VotingRegressor
                if model_name == 'VotingRegressor':
                    # Use less aggressive scaling for VotingRegressor
                    scaler_voting = StandardScaler()
                    X_train_voting = scaler_voting.fit_transform(X_train)
                    X_test_voting = scaler_voting.transform(X_test)
                    
                    # Clip extreme values
                    X_train_voting = np.clip(X_train_voting, -5, 5)
                    X_test_voting = np.clip(X_test_voting, -5, 5)
                    
                    y_pred = train_and_predict_safe(model, X_train_voting, y_train, X_test_voting, model_name)
                else:
                    # Use standard scaled features for other models
                    y_pred = train_and_predict_safe(model, X_train_scaled, y_train, X_test_scaled, model_name)
                
                # Skip if prediction failed
                if y_pred is None:
                    logger.warning(f"⚠️ Skipping {model_name} due to prediction issues")
                    continue
                
                # Calculate metrics
                metrics = calculate_comprehensive_metrics(y_test, y_pred, model_name)
                
                if metrics:
                    results.append(metrics)
                    
                    # Log key metrics
                    logger.info(f"✅ {model_name} Results:")
                    logger.info(f"   R² Score: {metrics['r2_score']}")
                    logger.info(f"   MAE: {metrics['mae']:.4f} ({metrics['mae_percent']:.1f}%)")
                    logger.info(f"   MSE: {metrics['mse']:.4f}")
                    logger.info(f"   RMSE: {metrics['rmse']:.4f}")
                    logger.info(f"   MAPE: {metrics['mape']:.2f}%")
                    logger.info(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                    logger.info(f"   Within 1%: {metrics['within_1_pct']:.1f}%")
                    logger.info(f"   Correlation: {metrics['correlation']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Error with {model_name}: {str(e)}")
                continue
        
        # Summary report
        if results:
            logger.info("\n" + "="*60)
            logger.info("📊 QUICK ENSEMBLE EVALUATION SUMMARY")
            logger.info("="*60)
            
            df_results = pd.DataFrame(results)
            
            # Sort by R² score
            df_results = df_results.sort_values('r2_score', ascending=False)
            
            logger.info(f"\n🏆 MODEL RANKING (by R² Score):")
            for _, row in df_results.iterrows():
                logger.info(f"   {row['model']}: R²={row['r2_score']:.4f}, "
                           f"MAE={row['mae']:.4f}, "
                           f"Dir.Acc={row['directional_accuracy']:.1f}%")
            
            # Best model analysis
            best_model = df_results.iloc[0]
            logger.info(f"\n🥇 BEST MODEL: {best_model['model']}")
            logger.info(f"   📈 R² Score: {best_model['r2_score']:.4f}")
            logger.info(f"   📏 MAE: {best_model['mae']:.4f} ({best_model['mae_percent']:.1f}%)")
            logger.info(f"   📐 MSE: {best_model['mse']:.4f}")
            logger.info(f"   📊 RMSE: {best_model['rmse']:.4f}")
            logger.info(f"   🎯 MAPE: {best_model['mape']:.2f}%")
            logger.info(f"   🧭 Directional Accuracy: {best_model['directional_accuracy']:.1f}%")
            logger.info(f"   🎳 Within 1%: {best_model['within_1_pct']:.1f}%")
            logger.info(f"   🔗 Correlation: {best_model['correlation']:.4f}")
            
            # Ensemble vs individual comparison
            ensemble_models = df_results[df_results['model'].str.contains('Stacking|Voting')]
            individual_models = df_results[~df_results['model'].str.contains('Stacking|Voting')]
            
            if len(ensemble_models) > 0 and len(individual_models) > 0:
                logger.info(f"\n🔀 ENSEMBLE vs INDIVIDUAL COMPARISON:")
                logger.info(f"   Ensemble Models - Avg R²: {ensemble_models['r2_score'].mean():.4f}")
                logger.info(f"   Individual Models - Avg R²: {individual_models['r2_score'].mean():.4f}")
                logger.info(f"   Ensemble Advantage: {ensemble_models['r2_score'].mean() - individual_models['r2_score'].mean():+.4f}")
            
            # Display complete results table
            print(f"\n📋 COMPLETE RESULTS TABLE:")
            print(df_results[['model', 'r2_score', 'mae', 'mae_percent', 'mse', 'rmse', 'mape', 'directional_accuracy', 'within_1_pct']].to_string(index=False))
            
            logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error in evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    quick_ensemble_evaluation() 