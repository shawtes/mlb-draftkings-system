"""
Check if we have actual historical variance or synthetic variance
"""
import pandas as pd
import numpy as np

df = pd.read_csv('nba_training_data.csv')

# Check for actual historical stats
print("📊 Checking for actual vs synthetic variance...")
print(f"\nTotal records: {len(df)}")

# Look for actual stat columns
actual_cols = [col for col in df.columns if col.startswith('actual_')]
projected_cols = [col for col in df.columns if col.startswith('projected_')]

print(f"\nActual columns: {actual_cols}")
print(f"Projected columns: {len(projected_cols)}")

# Check if actuals match projections (synthetic)
if 'actual_points' in df.columns and 'projected_points' in df.columns:
    correlation = df['actual_points'].corr(df['projected_points'])
    print(f"\nCorrelation between actual and projected: {correlation:.3f}")
    
    if correlation > 0.95:
        print("⚠️  WARNING: Actuals are synthetic (based on projections)")
    else:
        print("✅ Actuals appear to be real historical data")

# Show how variance is calculated
if 'points_accuracy_std' in df.columns:
    print(f"\n📈 Sample variance values:")
    print(df[['player_name_proj', 'points_accuracy_std', 'projected_points']].head(10))
