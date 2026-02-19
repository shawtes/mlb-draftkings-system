"""
Quick Hyperparameter Search Script for MLB DraftKings Model

This script runs a quick hyperparameter search on a small sample of data
to find optimal parameters, then saves them for use in the full training.

Instructions:
1. Run this script first to find optimal parameters
2. Copy the output parameters to the main training script
3. Set USE_HARDCODED_PARAMS = True in training.py
4. Run the full training with hard-coded parameters
"""

import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')

def quick_hyperparameter_search(data_path, sample_size=5000):
    """
    Perform quick hyperparameter search on small sample
    """
    print("🔍 Quick Hyperparameter Search for MLB DraftKings Model")
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
    
    print(f"Loaded {len(df)} rows for hyperparameter search")
    
    # Basic preprocessing
    df.fillna(0, inplace=True)
    
    # Select only numeric features for quick search
    numeric_features = df.select_dtypes(include=[np.number]).columns
    target_column = 'dk_fpts' if 'dk_fpts' in df.columns else 'calculated_dk_fpts'
    numeric_features = [col for col in numeric_features if col != target_column]
    
    if target_column not in df.columns:
        print(f"Error: '{target_column}' column not found")
        print("Available columns:", df.columns.tolist())
        return None
    
    X = df[numeric_features[:50]]  # Use only first 50 numeric features
    y = df[target_column]
    
    # Basic scaling and feature selection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Select top features
    selector = SelectKBest(f_regression, k=min(20, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    print(f"Using {X_selected.shape[1]} features for hyperparameter search")
    
    # Simple model setup
    base_models = [
        ('ridge', Ridge(alpha=1.0)),
        ('xgb', xgb.XGBRegressor(n_estimators=30, random_state=42))
    ]
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=xgb.XGBRegressor(random_state=42),
        n_jobs=1
    )
    
    # Parameter grid
    param_grid = {
        'final_estimator__n_estimators': [50, 100, 150],
        'final_estimator__max_depth': [3, 5, 7],
        'final_estimator__learning_rate': [0.1, 0.15, 0.2],
        'final_estimator__subsample': [0.8, 0.9, 1.0]
    }
    
    # Search
    print("🚀 Starting hyperparameter search...")
    search = RandomizedSearchCV(
        stacking_model,
        param_grid,
        n_iter=15,
        cv=2,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1,
        random_state=42
    )
    
    start_time = time.time()
    search.fit(X_selected, y)
    elapsed = time.time() - start_time
    
    # Results
    print(f"\n✅ Search completed in {elapsed:.1f} seconds")
    print("\n" + "="*70)
    print("🎯 OPTIMAL PARAMETERS FOUND:")
    print("="*70)
    
    best_params = search.best_params_
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV Score: {-search.best_score_:.4f}")
    
    print("\n" + "="*70)
    print("📋 COPY THESE PARAMETERS TO YOUR TRAINING SCRIPT:")
    print("="*70)
    print("HARDCODED_OPTIMAL_PARAMS = {")
    for param, value in best_params.items():
        print(f"    '{param}': {value},")
    print("}")
    
    print("\n🔧 INSTRUCTIONS:")
    print("1. Copy the parameters above")
    print("2. Update HARDCODED_OPTIMAL_PARAMS in training.py")
    print("3. Set USE_HARDCODED_PARAMS = True in training.py")
    print("4. Run the full training script")
    print("="*70)
    
    # Save to file
    output_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/8_DOCUMENTATION/optimal_parameters.txt'
    with open(output_file, 'w') as f:
        f.write("OPTIMAL HYPERPARAMETERS FOR MLB DRAFTKINGS MODEL\n")
        f.write("=" * 50 + "\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest CV Score: {-search.best_score_:.4f}\n\n")
        f.write("Code to copy:\n")
        f.write("HARDCODED_OPTIMAL_PARAMS = {\n")
        for param, value in best_params.items():
            f.write(f"    '{param}': {value},\n")
        f.write("}\n")
    
    print(f"💾 Parameters saved to {output_file}")
    
    return best_params

if __name__ == "__main__":
    # Update this path to your data file
    data_path = r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv'
    
    try:
        best_params = quick_hyperparameter_search(data_path)
        if best_params:
            print("\n🎉 SUCCESS! Use the parameters above in your training script.")
        else:
            print("\n❌ Search failed. Check your data file path and format.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your data file exists and has the correct format.")
