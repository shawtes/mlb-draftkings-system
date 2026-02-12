import pandas as pd
import numpy as np


# Load the data
df = pd.read_csv('/Users/sineshawmesfintesfaye/mlb-draftkings-system/nba_training_data.csv')

# Calculate the skewness of the data (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
skewness = numeric_df.skew()

print("=" * 80)
print("SKEWNESS ANALYSIS FOR LOG TRANSFORMATION")
print("=" * 80)
print("\nOriginal Skewness Values:")
print(skewness.sort_values(ascending=False))
print("\n" + "=" * 80)

# Identify features that would benefit from log transformation
# Generally, features with |skewness| > 1 benefit from transformation
# Features with |skewness| > 2 strongly benefit
high_skew_threshold = 1.0
very_high_skew_threshold = 2.0

high_skew_features = skewness[abs(skewness) > high_skew_threshold].sort_values(ascending=False, key=abs)
very_high_skew_features = skewness[abs(skewness) > very_high_skew_threshold].sort_values(ascending=False, key=abs)

print(f"\nFeatures with high skewness (|skewness| > {high_skew_threshold}):")
print(f"Total: {len(high_skew_features)} features")
print("\nTop candidates for log transformation:")

# Check which features can be log-transformed (must be strictly positive or use log1p)
log_transformable = []
log1p_transformable = []
cannot_log = []

for feature in very_high_skew_features.index:
    col_data = numeric_df[feature].dropna()
    min_val = col_data.min()
    max_val = col_data.max()
    
    if min_val > 0:
        # Can use log transformation (use log1p for numerical stability, even for positive values)
        # log1p(x) = log(1+x), which is more stable than log(x) and handles values near zero better
        log_skew = np.log1p(col_data).skew()
        improvement = abs(skewness[feature]) - abs(log_skew)
        log_transformable.append({
            'feature': feature,
            'original_skew': skewness[feature],
            'log_skew': log_skew,
            'improvement': improvement,
            'min': min_val,
            'max': max_val
        })
    elif min_val >= -1:
        # Can use log1p transformation (handles zeros and small negatives)
        log_skew = np.log1p(col_data - min_val + 1).skew()
        improvement = abs(skewness[feature]) - abs(log_skew)
        log1p_transformable.append({
            'feature': feature,
            'original_skew': skewness[feature],
            'log1p_skew': log_skew,
            'improvement': improvement,
            'min': min_val,
            'max': max_val
        })
    else:
        # Cannot easily log transform (has significant negative values)
        cannot_log.append({
            'feature': feature,
            'original_skew': skewness[feature],
            'min': min_val,
            'max': max_val
        })

print("\n" + "-" * 80)
print("RECOMMENDED FOR LOG TRANSFORMATION (np.log1p):")
print("-" * 80)
if log_transformable:
    print("\nFeatures with strictly positive values (can use log1p):")
    for item in sorted(log_transformable, key=lambda x: abs(x['improvement']), reverse=True)[:15]:
        print(f"  {item['feature']:40s} | Skew: {item['original_skew']:8.3f} -> {item['log_skew']:8.3f} | Improvement: {item['improvement']:8.3f}")

if log1p_transformable:
    print("\nFeatures with zeros/small negatives (can use log1p with shift):")
    for item in sorted(log1p_transformable, key=lambda x: abs(x['improvement']), reverse=True)[:10]:
        print(f"  {item['feature']:40s} | Skew: {item['original_skew']:8.3f} -> {item['log1p_skew']:8.3f} | Improvement: {item['improvement']:8.3f}")

if cannot_log:
    print("\nFeatures that cannot be log-transformed (have significant negative values):")
    for item in cannot_log[:10]:
        print(f"  {item['feature']:40s} | Skew: {item['original_skew']:8.3f} | Range: [{item['min']:.2f}, {item['max']:.2f}]")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("\n1. Log transformation is recommended for features with |skewness| > 2")
print("2. Use np.log1p() instead of np.log() - it handles zeros and is more stable")
print("3. For features with negative values, consider:")
print("   - Box-Cox transformation (requires positive values)")
print("   - Yeo-Johnson transformation (handles negatives)")
print("   - Square root transformation for count data")
print("4. Apply transformation BEFORE scaling in your preprocessing pipeline")
print("\n5. Top 10 features that would benefit most from log transformation:")
top_candidates = sorted(log_transformable + log1p_transformable, 
                       key=lambda x: abs(x['improvement']), reverse=True)[:10]
for i, item in enumerate(top_candidates, 1):
    feat_name = item['feature']
    print(f"   {i:2d}. {feat_name}")