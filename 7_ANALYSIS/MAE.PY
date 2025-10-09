import pandas as pd
import numpy as np

# Load the datasets
print("Loading prediction data...")
predictions_df = pd.read_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/batters_probability_predictions_20250707.csv')
print(f"Predictions loaded: {len(predictions_df)} rows")

print("Loading actual performance data...")
actual_df = pd.read_csv('C:/Users/smtes/FangraphsData/merged_fangraphs_data_output.csv')
print(f"Actual data loaded: {len(actual_df)} rows")

# Convert dates to datetime for proper matching
print("Converting dates...")

# Check for date column variations and standardize
if 'Date' in predictions_df.columns:
    predictions_df['date'] = pd.to_datetime(predictions_df['Date'], errors='coerce')
elif 'date' in predictions_df.columns:
    predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
else:
    print("Error: No date column found in predictions file")
    print("Available columns:", predictions_df.columns.tolist())
    exit()

if 'Date' in actual_df.columns:
    actual_df['date'] = pd.to_datetime(actual_df['Date'], errors='coerce')
elif 'date' in actual_df.columns:
    actual_df['date'] = pd.to_datetime(actual_df['date'], errors='coerce')
else:
    print("Error: No date column found in actual data file")
    print("Available columns:", actual_df.columns.tolist())
    exit()

# Show date ranges for debugging
print(f"Prediction date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
print(f"Actual data date range: {actual_df['date'].min()} to {actual_df['date'].max()}")

# Filter out pitchers if position column exists
if 'position' in predictions_df.columns:
    predictions_df = predictions_df[predictions_df['position'] != 'P']
    print(f"After removing pitchers: {len(predictions_df)} predictions")

# Merge datasets on both player name AND date
print("Merging datasets on Name and date...")
merged_df = pd.merge(
    predictions_df, 
    actual_df, 
    left_on=['Name', 'date'], 
    right_on=['Name', 'date'], 
    how='inner'
)

print(f"Merged dataset: {len(merged_df)} rows")

# Debug: Show available columns in merged dataset
print(f"Available columns in merged dataset: {list(merged_df.columns)}")

# Check if FPTS column exists and find the right column to use
fpts_column = None
print(f"Available columns in merged dataset: {len(merged_df.columns)} columns")

if 'dk_fpts' in merged_df.columns:
    print("✅ FPTS column found")
    fpts_column = 'dk_fpts'
elif 'calculated_dk_fpts' in merged_df.columns:
    print("✅ calculated_dk_fpts column found - using this as FPTS")
    fpts_column = 'calculated_dk_fpts'
else:
    print("❌ FPTS column not found")
    # Look for similar column names
    fpts_like_cols = [col for col in merged_df.columns if 'dk_fpts' in col.lower() or 'pts' in col.lower()]
    print(f"Columns containing 'dk_fpts' or 'pts': {fpts_like_cols}")
    
    # Try to use the first FPTS-like column found
    if fpts_like_cols:
        fpts_column = fpts_like_cols[0]
        print(f"Using column '{fpts_column}' as actual FPTS")
    else:
        print("No FPTS-like columns found.")
        # Try to find any numeric column that might be fantasy points
        numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        potential_fpts = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['point', 'score', 'fantasy', 'dk', 'fd'])]
        
        if potential_fpts:
            fpts_column = potential_fpts[0]
            print(f"Using column '{fpts_column}' as potential FPTS column")
        else:
            print("Available columns (first 20):", list(merged_df.columns)[:20])
            print(f"Available numeric columns: {numeric_cols[:10]}")

if fpts_column is None:
    print("❌ Could not identify FPTS column. Exiting.")
    exit(1)

print(f"Using '{fpts_column}' as the actual fantasy points column")

# Debug: Show some info about the FPTS column
print(f"FPTS column '{fpts_column}' statistics:")
print(f"  Min: {merged_df[fpts_column].min():.2f}")
print(f"  Max: {merged_df[fpts_column].max():.2f}")
print(f"  Mean: {merged_df[fpts_column].mean():.2f}")
print(f"  Non-null values: {merged_df[fpts_column].count()}/{len(merged_df)}")

# Check for predicted points column
predicted_column = None
if 'Predicted_DK_Points' in merged_df.columns:
    predicted_column = 'Predicted_DK_Points'
elif 'predicted_dk_fpts' in merged_df.columns:
    predicted_column = 'predicted_dk_fpts'
elif 'predicted_points' in merged_df.columns:
    predicted_column = 'predicted_points'
else:
    print("❌ Predicted points column not found in merged dataset")
    # Look for similar column names
    pred_like_cols = [col for col in merged_df.columns if 'predicted' in col.lower() or 'pred' in col.lower()]
    print(f"Columns containing 'predicted' or 'pred': {pred_like_cols}")

    if pred_like_cols:
        predicted_column = pred_like_cols[0]
        print(f"Using column '{predicted_column}' as predicted points")
    else:
        print("Available columns:", list(merged_df.columns))
        exit(1)

print(f"Using '{predicted_column}' as the predicted fantasy points column")

# Remove any rows with NaN values in the key columns
print("Cleaning data...")
before_clean = len(merged_df)
merged_df = merged_df.dropna(subset=[fpts_column, predicted_column])

# Remove rows where actual dk_fpts is 0 (player didn't play or had no production)
before_zero_filter = len(merged_df)
merged_df = merged_df[merged_df[fpts_column] != 0]
after_clean = len(merged_df)
print(f"After removing NaN values: {before_zero_filter} rows (removed {before_clean - before_zero_filter} rows)")
print(f"After removing zero dk_fpts: {after_clean} rows (removed {before_zero_filter - after_clean} rows)")

if len(merged_df) == 0:
    print("❌ No valid data remaining after cleaning. Check your data for missing values.")
    exit(1)

# Final validation before proceeding
if len(merged_df) == 0 or fpts_column is None:
    print("❌ Cannot proceed! Either no matches found or no FPTS column identified.")
    if len(merged_df) == 0:
        print("Check if:")
        print("- Player names match exactly between datasets")
        print("- Dates match exactly between datasets") 
        print("- The prediction date exists in actual data")
        
        # Show some sample data for debugging
        print("\nSample prediction data:")
        print(predictions_df[['Name', 'date', predicted_column]].head())
        
        print("\nSample actual data:")
        if 'Name' in actual_df.columns and 'date' in actual_df.columns:
            sample_cols = ['Name', 'date']
            # Add FPTS column if it exists
            for col in ['dk_fpts', 'calculated_dk_fpts']:
                if col in actual_df.columns:
                    sample_cols.append(col)
                    break
            print(actual_df[sample_cols].head())
        else:
            print("Available columns in actual data:", list(actual_df.columns)[:10])
            print(actual_df.head(2))
    else:
        print("No suitable FPTS column found in merged dataset")
    exit(1)
else:
    # Calculate comprehensive evaluation metrics
    print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS:")
    print(f"=" * 60)
    print(f"📊 Dataset: {len(merged_df)} valid predictions")
    print(f"📅 Date: {merged_df['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"🏀 Using '{fpts_column}' as actual fantasy points")
    print(f"🔮 Using '{predicted_column}' as predicted fantasy points")
    print(f"=" * 60)
    
    # Calculate primary metrics
    actual_points = merged_df[fpts_column].values
    predicted_points = merged_df[predicted_column].values
    
    mae = np.mean(np.abs(actual_points - predicted_points))
    rmse = np.sqrt(np.mean((actual_points - predicted_points) ** 2))
    r2 = 1 - (np.sum((actual_points - predicted_points) ** 2) / np.sum((actual_points - np.mean(actual_points)) ** 2))
    mape = np.mean(np.abs((actual_points - predicted_points) / actual_points)) * 100
    
    print(f"📈 Mean Absolute Error (MAE): {mae:.3f}")
    print(f"� Root Mean Square Error (RMSE): {rmse:.3f}")
    print(f"📊 R² Score: {r2:.3f}")
    print(f"📊 Mean Absolute Percentage Error (MAPE): {mape:.1f}%")
    print(f"=" * 60)
    
    # Calculate bias statistics
    differences = actual_points - predicted_points
    positive_diff_avg = differences[differences > 0].mean() if len(differences[differences > 0]) > 0 else 0
    negative_diff_avg = differences[differences < 0].mean() if len(differences[differences < 0]) > 0 else 0
    
    print(f"📊 Prediction Bias Analysis:")
    print(f"   Average Under-prediction: {positive_diff_avg:.3f}")
    print(f"   Average Over-prediction: {abs(negative_diff_avg):.3f}")
    print(f"   Model tends to: {'Over-predict' if np.mean(differences) < 0 else 'Under-predict'}")
    
    # Statistical summary
    correlation = np.corrcoef(actual_points, predicted_points)[0, 1]
    print(f"\n📊 Statistical Summary:")
    print(f"   Correlation: {correlation:.3f}")
    print(f"   Mean Actual: {actual_points.mean():.2f}")
    print(f"   Mean Predicted: {predicted_points.mean():.2f}")
    print(f"   Std Actual: {actual_points.std():.2f}")
    print(f"   Std Predicted: {predicted_points.std():.2f}")
    
    # Show best and worst predictions
    merged_df['abs_error'] = np.abs(merged_df[fpts_column] - merged_df[predicted_column])
    merged_df['error'] = merged_df[fpts_column] - merged_df[predicted_column]
    
    print(f"\n✅ Best Predictions (Lowest Absolute Error):")
    print(f"=" * 80)
    best_preds = merged_df.nsmallest(5, 'abs_error')[['Name', fpts_column, predicted_column, 'error', 'abs_error']]
    best_preds.columns = ['Name', 'Actual', 'Predicted', 'Error', 'Abs_Error']
    print(best_preds.to_string(index=False, float_format='%.2f'))
    
    print(f"\n❌ Worst Predictions (Highest Absolute Error):")
    print(f"=" * 80)
    worst_preds = merged_df.nlargest(5, 'abs_error')[['Name', fpts_column, predicted_column, 'error', 'abs_error']]
    worst_preds.columns = ['Name', 'Actual', 'Predicted', 'Error', 'Abs_Error']
    print(worst_preds.to_string(index=False, float_format='%.2f'))
    
    # Show a random sample of predictions
    print(f"\n📋 Random Sample of Predictions:")
    print(f"=" * 80)
    sample_preds = merged_df.sample(10)[['Name', fpts_column, predicted_column, 'error', 'abs_error']]
    sample_preds.columns = ['Name', 'Actual', 'Predicted', 'Error', 'Abs_Error']
    print(sample_preds.to_string(index=False, float_format='%.2f'))
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Your model achieved an MAE of {mae:.3f} and R² of {r2:.3f}")
    print(f"   This means on average, predictions are off by {mae:.1f} fantasy points")
    if r2 > 0.7:
        print(f"   ✅ Excellent predictive performance!")
    elif r2 > 0.5:
        print(f"   ✅ Good predictive performance!")
    elif r2 > 0.3:
        print(f"   ⚠️  Moderate predictive performance")
    else:
        print(f"   ❌ Poor predictive performance - consider model improvements")