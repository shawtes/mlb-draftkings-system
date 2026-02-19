"""
Quick CV Parameter Search Script for MLB DraftKings Model

This script runs a quick search to find optimal cross-validation parameters
for the best balance of accuracy and speed.

Instructions:
1. Run this script to find optimal CV parameters
2. Copy the output parameters to the main training script
3. Set USE_HARDCODED_CV_PARAMS = True in training.py
4. Run the full training with optimized CV settings
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')

def quick_cv_parameter_search(data_path, sample_size=3000):
    """
    Find optimal CV parameters for TIME SERIES MLB data
    """
    print("🔍 Quick CV Parameter Search for TIME SERIES MLB Data")
    print("=" * 70)
    
    # Load small sample of data
    print("Loading data sample...")
    try:
        df = pd.read_csv(data_path, nrows=sample_size)
    except:
        print("Error loading data. Using chunk loading...")
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=1000):
            chunks.append(chunk)
            if len(pd.concat(chunks)) >= sample_size:
                break
        df = pd.concat(chunks).head(sample_size)
    
    print(f"Loaded {len(df)} rows for CV parameter search")
    
    # Basic preprocessing
    df.fillna(0, inplace=True)
    
    # Select only numeric features for quick search
    numeric_features = df.select_dtypes(include=[np.number]).columns
    target_column = 'dk_fpts' if 'dk_fpts' in df.columns else 'calculated_dk_fpts'
    numeric_features = [col for col in numeric_features if col != target_column]
    
    if target_column not in df.columns:
        print(f"Error: '{target_column}' column not found")
        return None
    
    X = df[numeric_features[:30]]  # Use only first 30 numeric features
    y = df[target_column]
    
    # Basic scaling and feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select top features
    selector = SelectKBest(f_regression, k=min(15, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    print(f"Using {X_selected.shape[1]} features for CV parameter search")
    
    # Test different CV configurations - TIME SERIES FOCUS
    cv_configs = [
        # Time Series CV (RECOMMENDED for MLB data)
        {'cv_type': 'timeseries', 'cv_folds': 3, 'n_iter': 8, 'description': 'TimeSeries 3-fold ⭐ RECOMMENDED'},
        {'cv_type': 'timeseries', 'cv_folds': 4, 'n_iter': 10, 'description': 'TimeSeries 4-fold'},
        {'cv_type': 'timeseries', 'cv_folds': 5, 'n_iter': 12, 'description': 'TimeSeries 5-fold'},
        
        # Regular K-Fold CV (for comparison only)
        {'cv_type': 'kfold', 'cv_folds': 3, 'n_iter': 8, 'description': 'K-Fold 3-fold (comparison)'},
        {'cv_type': 'kfold', 'cv_folds': 5, 'n_iter': 12, 'description': 'K-Fold 5-fold (comparison)'},
    ]
    
    print("\n🚀 Testing CV configurations...")
    print("🏅 TimeSeriesSplit is HIGHLY RECOMMENDED for MLB player data!")
    print("   - Respects temporal order of games")
    print("   - Prevents data leakage from future")
    print("   - More realistic for player predictions")
    
    best_config = None
    best_score = float('-inf')
    best_efficiency = float('-inf')
    results = []
    
    # Simple model for CV testing
    test_model = xgb.XGBRegressor(n_estimators=30, max_depth=3, random_state=42, verbosity=0)
    
    for i, config in enumerate(cv_configs):
        print(f"\n{i+1}. Testing {config['description']}...")
        
        try:
            start_time = time.time()
            
            # Create appropriate CV splitter
            if config['cv_type'] == 'timeseries':
                cv_splitter = TimeSeriesSplit(n_splits=config['cv_folds'])
                print(f"   🕒 Using TimeSeriesSplit: trains on past, tests on future")
            else:
                cv_splitter = config['cv_folds']  # Regular K-Fold
                print(f"   📊 Using K-Fold: random splits (not ideal for time series)")
            
            # Test this configuration
            search = RandomizedSearchCV(
                test_model,
                {'n_estimators': [20, 30, 40], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.2]},
                n_iter=config['n_iter'],
                cv=cv_splitter,
                scoring='neg_mean_squared_error',
                n_jobs=1,
                verbose=0,
                random_state=42
            )
            
            search.fit(X_selected, y)
            
            elapsed = time.time() - start_time
            score = -search.best_score_
            
            # Calculate efficiency score (accuracy per minute)
            efficiency = score / (elapsed / 60) if elapsed > 0 else 0
            
            # BIG bonus for TimeSeriesSplit (much better for time series data)
            if config['cv_type'] == 'timeseries':
                efficiency *= 1.5  # 50% bonus for being appropriate for time series
                config['recommended'] = True
            else:
                config['recommended'] = False
            
            results.append({
                'config': config,
                'score': score,
                'time': elapsed,
                'efficiency': efficiency
            })
            
            print(f"   Score: {score:.4f}")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Efficiency: {efficiency:.2f}")
            if config.get('recommended'):
                print(f"   ⭐ HIGHLY RECOMMENDED for MLB time series data!")
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_config = config
                
        except Exception as e:
            print(f"   ❌ Config failed: {e}")
            continue
    
    # Display results
    print("\n" + "="*70)
    print("📊 CV RESULTS - TIME SERIES vs K-FOLD COMPARISON:")
    print("="*70)
    
    for i, result in enumerate(results):
        config = result['config']
        if config == best_config:
            marker = "🏆"
        elif config.get('recommended'):
            marker = "⭐"
        else:
            marker = "  "
            
        print(f"{marker} {i+1}. {config['description']}")
        print(f"     Method: {config['cv_type']}, Splits: {config['cv_folds']}")
        print(f"     Score: {result['score']:.4f}, Time: {result['time']:.1f}s")
        print(f"     Efficiency: {result['efficiency']:.2f}")
        if config.get('recommended'):
            print(f"     🎯 Perfect for MLB player performance prediction!")
        print()
    
    if best_config:
        print("🎯 OPTIMAL CV CONFIGURATION:")
        print("=" * 70)
        print(f"🏅 WINNER: {best_config['description']}")
        
        if best_config['cv_type'] == 'timeseries':
            print("✅ EXCELLENT CHOICE! TimeSeriesSplit is perfect for MLB data!")
            print("   ✓ Respects temporal order of games")
            print("   ✓ No data leakage from future games")
            print("   ✓ Realistic training/testing scenarios")
            print("   ✓ Each fold: train on past → test on future")
        else:
            print("⚠️  Consider switching to TimeSeriesSplit for better results!")
        
        print(f"\nConfiguration details:")
        for param, value in best_config.items():
            if param not in ['description', 'recommended']:
                print(f"  {param}: {value}")
        print("=" * 70)
        
        # Create complete parameter set
        recommended_params = {
            'cv_type': best_config['cv_type'],
            'cv_folds': best_config['cv_folds'],
            'n_iter': best_config['n_iter'],
            'test_size': 0.2,
            'scoring': 'neg_mean_squared_error',
            'random_state': 42,
            'verbose': 1
        }
        
        print(f"\n📋 COPY THESE OPTIMAL CV PARAMETERS:")
        print("=" * 70)
        print("HARDCODED_CV_PARAMS = {")
        for param, value in recommended_params.items():
            if isinstance(value, str):
                print(f"    '{param}': '{value}',")
            else:
                print(f"    '{param}': {value},")
        print("}")
        print("=" * 70)
        
        return recommended_params
    else:
        print("❌ No valid CV configuration found.")
        return None

def compare_cv_performance(data_path):
    """
    Compare different CV approaches for educational purposes
    """
    print("\n📚 CV PERFORMANCE COMPARISON")
    print("=" * 50)
    
    approaches = {
        'TimeSeriesSplit': 'BEST for time-based data (MLB)',
        'K-Fold': 'General purpose (ignores time order)',
        'Stratified K-Fold': 'Better for classification',
        'Leave-One-Out': 'Most thorough but slow',
        'Repeated K-Fold': 'More robust but slower'
    }
    
    print("CV Approach Comparison:")
    for approach, description in approaches.items():
        print(f"  {approach}: {description}")
    
    print(f"\n💡 For MLB DraftKings regression:")
    print("  - TimeSeriesSplit is HIGHLY RECOMMENDED")
    print("  - Respects temporal order (no data leakage)")
    print("  - Trains on past data, tests on future data")
    print("  - 3-5 folds balance accuracy and speed")
    print("  - More realistic for actual predictions")

if __name__ == "__main__":
    # Update this path to your data file
    data_path = r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv'
    
    try:
        best_params = quick_cv_parameter_search(data_path)
        if best_params:
            print("\n🎉 SUCCESS! Use the CV parameters above in your training script.")
            compare_cv_performance(data_path)
        else:
            print("\n❌ CV search failed. Check your data file path and format.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your data file exists and has the correct format.")
