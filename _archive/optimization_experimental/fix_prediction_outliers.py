#!/usr/bin/env python3
"""
Fix Prediction Outliers - Remove unrealistic predictions and apply proper constraints
"""

import pandas as pd
import numpy as np
import sys
import os

def fix_prediction_outliers(input_file, output_file):
    """
    Fix prediction outliers by applying realistic constraints and removing faulty logic
    """
    print(f"🔧 Fixing prediction outliers in {input_file}")
    
    try:
        # Load the predictions
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} predictions")
        
        # Check if we have the required columns
        pred_col = None
        for col in ['predicted_dk_fpts', 'Predicted_DK_Points', 'Predicted_Points']:
            if col in df.columns:
                pred_col = col
                break
        
        if pred_col is None:
            print("❌ No prediction column found!")
            return False
        
        print(f"Using prediction column: {pred_col}")
        
        # Show original statistics
        print("\n📊 Original Prediction Statistics:")
        print(f"Min: {df[pred_col].min():.2f}")
        print(f"Max: {df[pred_col].max():.2f}")
        print(f"Mean: {df[pred_col].mean():.2f}")
        print(f"Median: {df[pred_col].median():.2f}")
        print(f"Std: {df[pred_col].std():.2f}")
        
        # Show current outliers
        outliers = df[df[pred_col] > 25]
        print(f"\n🚨 Current outliers (>25 points): {len(outliers)}")
        if len(outliers) > 0:
            # Show available columns for outliers
            available_cols = ['Name', pred_col]
            if 'FPTS' in df.columns:
                available_cols.append('FPTS')
            elif 'dk_fpts' in df.columns:
                available_cols.append('dk_fpts')
            elif 'calculated_dk_fpts' in df.columns:
                available_cols.append('calculated_dk_fpts')
            
            print(outliers[available_cols].head(10))
        
        # Apply realistic constraints for MLB DFS
        print("\n🔧 Applying realistic constraints...")
        
        # Method 1: Use historical performance as baseline
        historical_col = None
        for col in ['FPTS', 'dk_fpts', 'calculated_dk_fpts']:
            if col in df.columns:
                historical_col = col
                break
        
        if historical_col:
            # Use player's historical average ± reasonable range
            print(f"Using {historical_col} as baseline")
            df['baseline'] = df[historical_col].clip(lower=0, upper=25)  # Cap historical at 25
            df['realistic_min'] = np.maximum(0, df['baseline'] - 8)
            df['realistic_max'] = np.minimum(25, df['baseline'] + 8)
            
            # Apply constraints
            df[f'{pred_col}_fixed'] = np.clip(df[pred_col], df['realistic_min'], df['realistic_max'])
            
        else:
            # Method 2: Use statistical approach
            print("Using statistical approach for constraints")
            
            # Calculate reasonable bounds based on data distribution
            q25 = df[pred_col].quantile(0.25)
            q75 = df[pred_col].quantile(0.75)
            iqr = q75 - q25
            
            # Set realistic bounds (most MLB players score 0-20 points)
            realistic_min = 0
            realistic_max = min(25, q75 + 1.5 * iqr)  # Cap at 25 or Q75 + 1.5*IQR
            
            print(f"Realistic range: {realistic_min:.2f} to {realistic_max:.2f}")
            
            # Apply constraints
            df[f'{pred_col}_fixed'] = np.clip(df[pred_col], realistic_min, realistic_max)
        
        # Method 3: Handle identical predictions by adding small random variation
        print("\n🎲 Adding variation to identical predictions...")
        
        # Find groups with identical predictions
        identical_groups = df.groupby(f'{pred_col}_fixed').size()
        large_groups = identical_groups[identical_groups > 5]
        
        if len(large_groups) > 0:
            print(f"Found {len(large_groups)} groups with >5 identical predictions")
            
            for pred_value, count in large_groups.items():
                mask = df[f'{pred_col}_fixed'] == pred_value
                print(f"  Group {pred_value:.3f}: {count} players")
                
                # Add small random variation (±10% or ±1 point, whichever is smaller)
                variation = np.random.uniform(-1, 1, size=count)
                variation = np.clip(variation, -min(1, pred_value * 0.1), min(1, pred_value * 0.1))
                
                df.loc[mask, f'{pred_col}_fixed'] += variation
                
                # Ensure we don't go negative
                df.loc[mask, f'{pred_col}_fixed'] = np.maximum(0, df.loc[mask, f'{pred_col}_fixed'])
        
        # Final statistics
        print("\n📊 Fixed Prediction Statistics:")
        print(f"Min: {df[f'{pred_col}_fixed'].min():.2f}")
        print(f"Max: {df[f'{pred_col}_fixed'].max():.2f}")
        print(f"Mean: {df[f'{pred_col}_fixed'].mean():.2f}")
        print(f"Median: {df[f'{pred_col}_fixed'].median():.2f}")
        print(f"Std: {df[f'{pred_col}_fixed'].std():.2f}")
        
        # Show remaining outliers
        remaining_outliers = df[df[f'{pred_col}_fixed'] > 25]
        print(f"\n✅ Remaining outliers (>25 points): {len(remaining_outliers)}")
        
        # Replace the original column
        df[pred_col] = df[f'{pred_col}_fixed']
        df.drop(columns=[f'{pred_col}_fixed'], inplace=True)
        
        # Clean up temporary columns
        temp_cols = ['baseline', 'realistic_min', 'realistic_max']
        for col in temp_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        # Save fixed predictions
        df.to_csv(output_file, index=False)
        print(f"\n✅ Fixed predictions saved to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 MLB DFS Prediction Outlier Fix")
    print("=" * 50)

    # Define absolute paths for the required files
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_paths = [
        os.path.join(base_dir, "app", "merged_player_projections01.csv"),
        os.path.join(base_dir, "merged_player_projections01.csv"),
        os.path.join(base_dir, "4_DATA", "merged_player_projections01.csv"),
        os.path.join(base_dir, "4_DATA", "merged_player_projections.csv"),
        os.path.join(base_dir, "7_ANALYSIS", "batters_predictions_20250713.csv"),
        os.path.join(base_dir, "7_ANALYSIS", "batters_probability_predictions_202507013.csv"),
    ]

    # Check for file existence
    found_files = [file for file in file_paths if os.path.exists(file)]

    if not found_files:
        print("❌ No prediction files found!")
        print("Looking for files:")
        for file in file_paths:
            print(f"  - {file}")
        return

    print(f"Found {len(found_files)} prediction files:")
    for file in found_files:
        print(f"  - {file}")

    # Fix each file
    for input_file in found_files:
        output_file = input_file.replace('.csv', '_fixed.csv')
        print(f"\n🔧 Processing {input_file}")

        success = fix_prediction_outliers(input_file, output_file)

        if success:
            print(f"✅ Successfully fixed {input_file}")
        else:
            print(f"❌ Failed to fix {input_file}")

    print("\n🎯 Recommendations:")
    print("1. Use the '_fixed.csv' files in your optimizer")
    print("2. The original 5-game average constraint was too restrictive")
    print("3. Consider retraining your model with better data preprocessing")
    print("4. Set outlier_threshold = 25 in your optimizer to catch future issues")

if __name__ == "__main__":
    main()