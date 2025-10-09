#!/usr/bin/env python3
"""
Test script to verify that all required functions are defined
"""
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import joblib
        print("✅ Core imports successful")
        
        # Test sklearn imports
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import StackingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
        print("✅ SKLearn imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_function_definitions():
    """Test that all required functions are defined"""
    print("\n🔍 Testing function definitions...")
    
    # Read the file content
    with open('TRAINING.BACKUP.7-1-2025.PY', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key function definitions
    required_functions = [
        'load_or_create_label_encoders',
        'load_or_create_scaler', 
        'clean_infinite_values',
        'evaluate_model',
        'ProbabilityPredictor',
        'create_enhanced_predictions_with_probabilities',
        'save_feature_importance'
    ]
    
    required_variables = [
        'final_model',
        'base_models',
        'device',
        'name_encoder_path',
        'team_encoder_path'
    ]
    
    # Check functions
    for func_name in required_functions:
        if f'def {func_name}(' in content or f'class {func_name}' in content:
            print(f"✅ {func_name} defined")
        else:
            print(f"❌ {func_name} NOT defined")
    
    # Check variables
    for var_name in required_variables:
        if f'{var_name} =' in content:
            print(f"✅ {var_name} defined")
        else:
            print(f"❌ {var_name} NOT defined")
    
    print("\n🎯 Function definition check complete!")

def test_main_execution_order():
    """Test that functions are defined before main execution"""
    print("\n📍 Testing execution order...")
    
    with open('TRAINING.BACKUP.7-1-2025.PY', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find position of main execution
    main_pos = content.find('if __name__ == "__main__":')
    if main_pos == -1:
        print("❌ Main execution block not found")
        return False
    
    # Check if functions are defined before main
    functions_before_main = [
        'load_or_create_label_encoders',
        'final_model =',
        'base_models =',
        'device ='
    ]
    
    for func_name in functions_before_main:
        func_pos = content.find(func_name)
        if func_pos != -1 and func_pos < main_pos:
            print(f"✅ {func_name} defined before main execution")
        else:
            print(f"❌ {func_name} NOT defined before main execution")
    
    print("\n📍 Execution order check complete!")

if __name__ == "__main__":
    print("🧪 Running comprehensive function tests...")
    
    # Test 1: Imports
    if not test_imports():
        print("❌ Import test failed")
        sys.exit(1)
    
    # Test 2: Function definitions
    test_function_definitions()
    
    # Test 3: Execution order
    test_main_execution_order()
    
    print("\n🎉 All tests completed!") 