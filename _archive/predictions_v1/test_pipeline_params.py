#!/usr/bin/env python3
"""
Quick test to verify the pipeline parameter structure
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def test_pipeline_parameters():
    """Test the pipeline parameter structure"""
    
    print("🔍 Testing Pipeline Parameter Structure...")
    
    # Create the same model structure as in training.backup.py
    base_models = [
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('svr', SVR()),
        ('gb', GradientBoostingRegressor())
    ]

    xgb_params = {
        'tree_method': 'hist',
        'device': 'cpu',
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }

    meta_model = XGBRegressor(**xgb_params)
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    voting_model = VotingRegressor(estimators=base_models)

    ensemble_models = [
        ('stacking', stacking_model),
        ('voting', voting_model)
    ]

    final_model = StackingRegressor(
        estimators=ensemble_models,
        final_estimator=XGBRegressor(**xgb_params)
    )

    # Create mock transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['feature1', 'feature2']),
            ('cat', categorical_transformer, ['cat1'])
        ])

    selector = SelectKBest(f_regression, k=10)

    # Create the complete pipeline
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    # Test the corrected parameter names
    CORRECTED_PARAMS = {
        'model__final_estimator__n_estimators': 200,
        'model__final_estimator__max_depth': 6,
        'model__final_estimator__learning_rate': 0.1,
        'model__final_estimator__subsample': 0.8,
        'model__final_estimator__colsample_bytree': 0.9,
        'model__final_estimator__min_child_weight': 3,
        'model__final_estimator__gamma': 0.1,
        'model__final_estimator__reg_alpha': 0.1,
        'model__final_estimator__reg_lambda': 1.0,
    }

    print("\n📊 Testing parameter setting...")
    
    try:
        # Test if we can set the parameters
        complete_pipeline.set_params(**CORRECTED_PARAMS)
        print("✅ Parameter setting successful!")
        
        # Verify the parameters were set
        final_estimator = complete_pipeline.named_steps['model'].final_estimator
        print(f"✅ n_estimators set to: {final_estimator.n_estimators}")
        print(f"✅ max_depth set to: {final_estimator.max_depth}")
        print(f"✅ learning_rate set to: {final_estimator.learning_rate}")
        
        print("\n🎯 Pipeline structure test: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Parameter setting failed: {e}")
        
        # Let's inspect the pipeline structure
        print("\n🔍 Pipeline structure inspection:")
        print(f"Pipeline steps: {list(complete_pipeline.named_steps.keys())}")
        
        model_step = complete_pipeline.named_steps['model']
        print(f"Model type: {type(model_step)}")
        
        if hasattr(model_step, 'final_estimator'):
            print(f"Final estimator type: {type(model_step.final_estimator)}")
            
        # Get all valid parameters
        print("\n📋 Valid parameters for the pipeline:")
        valid_params = complete_pipeline.get_params().keys()
        xgb_params = [p for p in valid_params if 'model__final_estimator' in p]
        for param in sorted(xgb_params)[:10]:  # Show first 10
            print(f"  {param}")
        
        return False

if __name__ == "__main__":
    success = test_pipeline_parameters()
    
    if success:
        print("\n🎉 SUCCESS: Pipeline parameters are correctly configured!")
        print("The training script should now work without parameter errors.")
    else:
        print("\n⚠️  Parameter configuration needs adjustment.")
        print("Check the pipeline structure and parameter names.")
