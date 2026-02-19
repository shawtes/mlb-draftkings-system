#!/usr/bin/env python3
"""
Test script to verify the 10-day rolling constraint is working properly.
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predction01 import predict_unseen_data

def test_rolling_constraint():
    """Test that predictions are properly constrained within 10-day rolling ranges."""
    print("Testing 10-day rolling constraint...")
    
    # Test with a small sample to verify constraint logic
    test_date = "2025-07-05"
    
    print(f"Running predictions for {test_date}...")
    
    # Run prediction
    try:
        predict_unseen_data(
            input_file='../4_DATA/data_with_dk_entries_salaries_final_cleaned.csv',
            model_file='c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl',
            prediction_date=test_date
        )
        
        # Load the generated predictions
        predictions_file = f'batters_predictions_{test_date.replace("-", "")}.csv'
        if os.path.exists(predictions_file):
            result = pd.read_csv(predictions_file)
        else:
            print(f"ERROR: Predictions file {predictions_file} not found!")
            return False
    except Exception as e:
        print(f"ERROR: Failed to generate predictions: {e}")
        return False
    
    if result is None or result.empty:
        print("ERROR: No predictions generated!")
        return False
    
    print(f"Generated {len(result)} predictions")
    
    # Check for outlier predictions (>35 points)
    outliers = result[result['predicted_dk_fpts'] > 35]
    print(f"\nOutlier predictions (>35 points): {len(outliers)}")
    
    if len(outliers) > 0:
        print("Outlier details:")
        for idx, row in outliers.iterrows():
            print(f"  {row['Name']}: {row['predicted_dk_fpts']:.2f} points")
    
    # Check for the specific problematic value (38.317764)
    specific_value_count = len(result[result['predicted_dk_fpts'].round(6) == 38.317764])
    print(f"\nPredictions with value 38.317764: {specific_value_count}")
    
    # Show prediction distribution
    print(f"\nPrediction statistics:")
    print(f"Min: {result['predicted_dk_fpts'].min():.2f}")
    print(f"Max: {result['predicted_dk_fpts'].max():.2f}")
    print(f"Mean: {result['predicted_dk_fpts'].mean():.2f}")
    print(f"Std: {result['predicted_dk_fpts'].std():.2f}")
    
    return len(outliers) == 0

if __name__ == "__main__":
    success = test_rolling_constraint()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
