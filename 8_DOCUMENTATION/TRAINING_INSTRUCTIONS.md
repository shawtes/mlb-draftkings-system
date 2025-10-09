# MLB DraftKings Model Training Instructions

## Problem: 500+ features causing memory issues and crashes

## Solution: Two-step process

### Step 1: Find Optimal Parameters (FAST)
1. Run the quick parameter search script:
   ```
   python quick_param_search.py
   ```
   
2. This will:
   - Use only 5,000 rows of data
   - Use only 20 features
   - Find the best hyperparameters quickly
   - Save results to `optimal_parameters.txt`

### Step 2: Use Hard-coded Parameters (FULL TRAINING)
1. Copy the parameters from the output or `optimal_parameters.txt`
2. Update `HARDCODED_OPTIMAL_PARAMS` in `training.py`
3. Set `USE_HARDCODED_PARAMS = True` in `training.py`
4. Run the full training:
   ```
   python training.py
   ```

## Example Output:
```
🎯 OPTIMAL PARAMETERS FOUND:
================================
  final_estimator__n_estimators: 150
  final_estimator__max_depth: 5
  final_estimator__learning_rate: 0.15
  final_estimator__subsample: 0.9

📋 COPY THESE PARAMETERS:
HARDCODED_OPTIMAL_PARAMS = {
    'final_estimator__n_estimators': 150,
    'final_estimator__max_depth': 5,
    'final_estimator__learning_rate': 0.15,
    'final_estimator__subsample': 0.9,
}
```

## Configuration in training.py:
```python
# At the top of training.py
USE_HARDCODED_PARAMS = True  # Set to True after finding parameters

HARDCODED_OPTIMAL_PARAMS = {
    'final_estimator__n_estimators': 150,  # Update with your results
    'final_estimator__max_depth': 5,
    'final_estimator__learning_rate': 0.15,
    'final_estimator__subsample': 0.9,
}
```

## Memory Management Features Added:
- Automatic dataset reduction if > 100,000 rows
- Feature reduction to 150 max features
- Optimized chunk sizes for Omen 35L
- Memory usage monitoring

## Benefits:
- ✅ Fast parameter search (< 5 minutes)
- ✅ No memory crashes
- ✅ Optimal parameters for your data
- ✅ Full training with best settings
- ✅ Automatic fallbacks if training fails

## Troubleshooting:
- If quick search fails: Check data file path
- If full training fails: Uses automatic fallbacks
- If memory issues: Reduces dataset size automatically
- If crashes: Uses simpler models as fallbacks
