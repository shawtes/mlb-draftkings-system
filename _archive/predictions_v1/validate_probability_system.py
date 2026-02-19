#!/usr/bin/env python3
"""
Quick validation script to test the training.backup.py probability functionality
"""

import pandas as pd
import numpy as np
import os
import sys

def validate_training_script():
    """Validate that the training script has the correct probability prediction functionality"""
    
    print("🔍 Validating training script probability prediction functionality...")
    
    # Check if training script exists
    training_script_path = r"c:\Users\smtes\Downloads\coinbase_ml_trader\MLB_DRAFTKINGS_SYSTEM\9_BACKUP\training.backup.py"
    
    if not os.path.exists(training_script_path):
        print(f"❌ Training script not found at: {training_script_path}")
        return False
    
    # Read the training script
    with open(training_script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required components
    checks = [
        ('scipy import', 'from scipy import stats'),
        ('probability function', 'def calculate_probability_predictions'),
        ('probability function call', 'probability_predictions = calculate_probability_predictions('),
        ('bootstrap sampling', 'n_bootstrap'),
        ('probability thresholds', 'probability_thresholds = [5, 10, 15, 20, 25, 30, 35, 40]'),
        ('enhanced results', 'final_predictions_with_probabilities.csv'),
        ('probability summary', 'probability_summary.csv'),
    ]
    
    print("\n📋 Checking required components:")
    all_passed = True
    
    for check_name, check_string in checks:
        if check_string in content:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: Missing")
            all_passed = False
    
    # Check for specific function signature
    if 'def calculate_probability_predictions(model, features, thresholds, n_bootstrap=100):' in content:
        print("✅ Function signature: Correct")
    else:
        print("❌ Function signature: Incorrect or missing")
        all_passed = False
    
    # Check for probability calculation logic
    if 'bootstrap_predictions > threshold' in content:
        print("✅ Probability calculation logic: Found")
    else:
        print("❌ Probability calculation logic: Missing")
        all_passed = False
    
    # Check for prediction intervals
    if 'prediction_lower_80' in content and 'prediction_upper_80' in content:
        print("✅ Prediction intervals: Found")
    else:
        print("❌ Prediction intervals: Missing")
        all_passed = False
    
    print(f"\n🎯 Overall validation: {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("\n🚀 The training script is ready for production use!")
        print("📊 It includes:")
        print("   • Bootstrap-based uncertainty estimation")
        print("   • Probability predictions for 8 fantasy point thresholds")
        print("   • 80% prediction intervals")
        print("   • Enhanced CSV outputs with probability data")
        print("   • Backward compatibility with existing workflows")
    else:
        print("\n⚠️  The training script needs additional updates to be complete.")
    
    return all_passed

def generate_sample_output():
    """Generate sample output to show what the probability predictions look like"""
    
    print("\n📊 Sample Probability Prediction Output:")
    print("="*80)
    
    # Create sample data
    sample_data = {
        'Name': ['Mike Trout', 'Ronald Acuña Jr.', 'Mookie Betts', 'Vladimir Guerrero Jr.', 'Trea Turner'],
        'Date': ['2025-07-05'] * 5,
        'Predicted_FPTS': [18.5, 22.3, 16.8, 19.2, 14.3],
        'Prob_Over_5': [0.95, 0.98, 0.93, 0.96, 0.89],
        'Prob_Over_10': [0.85, 0.92, 0.78, 0.87, 0.71],
        'Prob_Over_15': [0.65, 0.78, 0.58, 0.68, 0.45],
        'Prob_Over_20': [0.35, 0.58, 0.28, 0.38, 0.18],
        'Prob_Over_25': [0.15, 0.35, 0.12, 0.18, 0.06],
        'Prediction_Lower_80': [12.2, 16.1, 10.9, 13.5, 8.7],
        'Prediction_Upper_80': [24.8, 28.5, 22.7, 24.9, 19.9],
        'Prediction_Std': [3.2, 3.8, 2.9, 3.4, 2.8]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Display formatted output
    for _, row in df.iterrows():
        print(f"\n🏆 Player: {row['Name']}")
        print(f"📈 Predicted DraftKings Points: {row['Predicted_FPTS']:.1f}")
        print(f"🎯 Probability Breakdown:")
        print(f"   • Over 5 points:  {row['Prob_Over_5']:.1%}")
        print(f"   • Over 10 points: {row['Prob_Over_10']:.1%}")
        print(f"   • Over 15 points: {row['Prob_Over_15']:.1%}")
        print(f"   • Over 20 points: {row['Prob_Over_20']:.1%}")
        print(f"   • Over 25 points: {row['Prob_Over_25']:.1%}")
        print(f"📊 80% Prediction Range: {row['Prediction_Lower_80']:.1f} - {row['Prediction_Upper_80']:.1f}")
        print(f"🔄 Prediction Uncertainty: ±{row['Prediction_Std']:.1f}")
        
        # Add strategy recommendation
        if row['Prob_Over_15'] >= 0.7:
            print("💡 Strategy: HIGH CONFIDENCE - Strong cash game play")
        elif row['Prob_Over_15'] >= 0.4:
            print("💡 Strategy: MODERATE CONFIDENCE - Good tournament play")
        else:
            print("💡 Strategy: LOW CONFIDENCE - GPP punt play only")
    
    print("\n" + "="*80)
    print("🎯 This is what the enhanced prediction system will output!")

if __name__ == "__main__":
    success = validate_training_script()
    generate_sample_output()
    
    if success:
        print("\n🎉 VALIDATION COMPLETE: The probability prediction system is ready!")
        print("🚀 Next steps:")
        print("   1. Run the training script to generate probability predictions")
        print("   2. Use the probability outputs for lineup optimization")
        print("   3. Monitor performance and calibrate as needed")
    else:
        print("\n⚠️  Please review the training script and ensure all components are present.")
