#!/usr/bin/env python3
"""
Efficient MAE calculator that handles large CSV files.
"""

import pandas as pd
import numpy as np

def load_data_efficiently():
    """Load data efficiently, checking columns first."""
    
    print("Loading prediction data...")
    # Load predictions (should be smaller)
    predictions_df = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS/batters_predictions_20240409.csv')
    print(f"Predictions loaded: {len(predictions_df)} rows")
    print(f"Prediction columns: {list(predictions_df.columns)}")
    
    # Parse date if needed
    if 'date' in predictions_df.columns:
        predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    
    print("\nLoading actual performance data (checking columns first)...")
    
    # Read just the first few rows to check columns
    sample_actual = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_fangraphs_logs_with_fpts.csv', nrows=5)
    print(f"Actual data columns: {list(sample_actual.columns)}")
    
    # Look for FPTS-like columns
    fpts_like_cols = [col for col in sample_actual.columns if 'fpts' in col.lower() or 'pts' in col.lower()]
    print(f"FPTS-like columns: {fpts_like_cols}")
    
    # Look for other potential fantasy point columns
    potential_cols = [col for col in sample_actual.columns if any(keyword in col.lower() for keyword in ['point', 'fantasy', 'dk', 'fd', 'score'])]
    print(f"Other potential fantasy point columns: {potential_cols}")
    
    # Now load the full actual data with only necessary columns
    needed_cols = ['Name', 'date'] + fpts_like_cols + potential_cols
    # Remove duplicates
    needed_cols = list(set(needed_cols))
    # Keep only columns that exist
    needed_cols = [col for col in needed_cols if col in sample_actual.columns]
    
    print(f"Loading actual data with columns: {needed_cols}")
    
    try:
        actual_df = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_fangraphs_logs_with_fpts.csv', usecols=needed_cols)
        print(f"Actual data loaded: {len(actual_df)} rows")
        
        # Parse date if needed
        if 'date' in actual_df.columns:
            actual_df['date'] = pd.to_datetime(actual_df['date']).dt.date
            
        return predictions_df, actual_df, fpts_like_cols + potential_cols
        
    except Exception as e:
        print(f"Error loading actual data: {e}")
        return None, None, []

def calculate_mae():
    """Calculate MAE between predictions and actual."""
    
    predictions_df, actual_df, potential_fpts_cols = load_data_efficiently()
    
    if predictions_df is None or actual_df is None:
        print("❌ Failed to load data")
        return
    
    # Try to merge on Name and date
    print("\nMerging datasets...")
    merged_df = pd.merge(
        predictions_df, 
        actual_df, 
        left_on=['Name', 'date'], 
        right_on=['Name', 'date'], 
        how='inner'
    )
    
    print(f"Merged dataset: {len(merged_df)} rows")
    print(f"Available columns in merged dataset: {list(merged_df.columns)}")
    
    # Find the best FPTS column
    fpts_column = None
    for col in potential_fpts_cols:
        if col in merged_df.columns:
            # Check if it's numeric and has reasonable values
            if merged_df[col].dtype in ['float64', 'int64']:
                print(f"Found potential FPTS column: {col}")
                print(f"  Min: {merged_df[col].min()}, Max: {merged_df[col].max()}, Mean: {merged_df[col].mean():.2f}")
                if fpts_column is None:  # Use the first valid one
                    fpts_column = col
    
    if fpts_column is None:
        print("❌ No suitable FPTS column found")
        return
    
    print(f"✅ Using '{fpts_column}' as actual fantasy points")
    
    if len(merged_df) == 0:
        print("❌ No matches found between predictions and actual data")
        return
    
    # Calculate MAE
    mae = (merged_df[fpts_column] - merged_df['predicted_dk_fpts']).abs().mean()
    
    # Calculate other statistics
    positive_diff_avg = (merged_df[fpts_column] - merged_df['predicted_dk_fpts'])[merged_df[fpts_column] > merged_df['predicted_dk_fpts']].mean()
    negative_diff_avg = (merged_df[fpts_column] - merged_df['predicted_dk_fpts'])[merged_df[fpts_column] < merged_df['predicted_dk_fpts']].mean()
    correlation = merged_df[fpts_column].corr(merged_df['predicted_dk_fpts'])
    
    # Print results
    print(f"\n📊 Model Performance Results:")
    print(f"MAE (Mean Absolute Error): {mae:.3f}")
    print(f"Average Under-prediction: {positive_diff_avg:.3f}")
    print(f"Average Over-prediction: {negative_diff_avg:.3f}")
    print(f"Correlation: {correlation:.3f}")
    
    # Show sample matches
    print(f"\nSample matched predictions vs actual:")
    sample_cols = ['Name', 'date', 'predicted_dk_fpts', fpts_column]
    if all(col in merged_df.columns for col in sample_cols):
        sample_df = merged_df[sample_cols].head(10)
        sample_df['difference'] = sample_df[fpts_column] - sample_df['predicted_dk_fpts']
        print(sample_df)

if __name__ == "__main__":
    calculate_mae()
