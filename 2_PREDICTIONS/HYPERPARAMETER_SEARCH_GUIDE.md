# 🔍 Hyperparameter Search & Cross-Validation Configuration Guide

## 📍 Location of Hyperparameter Search Parameters

### 1. **Configuration Variables** (Lines 32-55)

```python
# CONFIGURATION: Set USE_HARDCODED_PARAMS to True to skip hyperparameter search
USE_HARDCODED_PARAMS = True  # Set to False to enable hyperparameter search

# HARD-CODED OPTIMAL PARAMETERS (for fast training)
HARDCODED_OPTIMAL_PARAMS = {
    'final_estimator__n_estimators': 200,
    'final_estimator__max_depth': 6,
    'final_estimator__learning_rate': 0.1,
    'final_estimator__subsample': 0.8,
    'final_estimator__colsample_bytree': 0.9,
    'final_estimator__min_child_weight': 3,
    'final_estimator__gamma': 0.1,
    'final_estimator__reg_alpha': 0.1,
    'final_estimator__reg_lambda': 1.0,
}

# Cross-validation configuration
CV_CONFIG = {
    'cv_type': 'timeseries',  # 'timeseries' or 'kfold'
    'cv_folds': 3,  # Number of splits for TimeSeriesSplit
    'n_iter': 10,  # Number of iterations for RandomizedSearchCV
    'scoring': 'neg_mean_squared_error',
    'random_state': 42,
    'verbose': 1,
}
```

### 2. **Hyperparameter Search Space** (Lines 587-612)

```python
# Define hyperparameter search space for RandomizedSearchCV
HYPERPARAMETER_SEARCH_SPACE = {
    # XGBoost final estimator parameters
    'final_estimator__n_estimators': [100, 200, 300, 400, 500],
    'final_estimator__max_depth': [3, 4, 5, 6, 7, 8],
    'final_estimator__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'final_estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'final_estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'final_estimator__min_child_weight': [1, 3, 5, 7, 10],
    'final_estimator__gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'final_estimator__reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],
    'final_estimator__reg_lambda': [0.5, 1.0, 1.5, 2.0, 3.0],
    # Base model parameters (commented out for performance)
    # 'stacking__gb__n_estimators': [50, 100, 150, 200],
    # 'stacking__gb__max_depth': [3, 4, 5, 6],
    # 'stacking__gb__learning_rate': [0.01, 0.05, 0.1, 0.15],
}
```

### 3. **Hyperparameter Search Function** (Lines 614-675)

```python
def perform_hyperparameter_search(pipeline, X_train, y_train, search_space=None, cv_config=None):
    """
    Perform hyperparameter search using RandomizedSearchCV with TimeSeriesSplit
    """
    # Implementation with RandomizedSearchCV
    # Returns best_estimator after cross-validation
```

### 4. **Training Logic with Hyperparameter Search** (Lines 975-1010)

```python
# =============================================================================
# HYPERPARAMETER SEARCH OR HARD-CODED PARAMETERS
# =============================================================================

if USE_HARDCODED_PARAMS:
    # Use hard-coded optimal parameters for fast training
    print("Using hard-coded optimal parameters for fast training...")
    # Apply hard-coded parameters and train
else:
    # Perform hyperparameter search
    print("Performing hyperparameter search...")
    best_pipeline = perform_hyperparameter_search(...)
```

---

## 🎯 How to Use Hyperparameter Search

### **Method 1: Enable Hyperparameter Search (Slower, More Accurate)**

```python
# Change this line from True to False
USE_HARDCODED_PARAMS = False

# Optionally modify search parameters
CV_CONFIG = {
    'cv_type': 'timeseries',
    'cv_folds': 5,          # More folds for better validation
    'n_iter': 50,           # More iterations for better search
    'scoring': 'neg_mean_squared_error',
    'random_state': 42,
    'verbose': 2,           # More verbose output
}
```

### **Method 2: Use Hard-Coded Parameters (Faster, Pre-Optimized)**

```python
# Keep this as True for fast training
USE_HARDCODED_PARAMS = True

# Optionally modify the optimal parameters
HARDCODED_OPTIMAL_PARAMS = {
    'final_estimator__n_estimators': 300,  # Increase for potentially better performance
    'final_estimator__max_depth': 7,       # Increase for more complex patterns
    # ... other parameters
}
```

---

## 📊 Customization Options

### **1. Modify Search Space**

```python
# Add more parameter ranges
HYPERPARAMETER_SEARCH_SPACE = {
    'final_estimator__n_estimators': [100, 200, 300, 400, 500, 600, 700],  # More options
    'final_estimator__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],              # Wider range
    # Enable base model tuning
    'stacking__gb__n_estimators': [50, 100, 150, 200, 250],
    'stacking__gb__max_depth': [3, 4, 5, 6, 7],
    'stacking__gb__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
}
```

### **2. Modify Cross-Validation Strategy**

```python
CV_CONFIG = {
    'cv_type': 'timeseries',    # Best for time-series data
    'cv_folds': 5,              # More folds = better validation but slower
    'n_iter': 100,              # More iterations = more thorough search
    'scoring': 'neg_mean_squared_error',  # or 'neg_mean_absolute_error'
    'random_state': 42,
    'verbose': 2,               # 0=silent, 1=progress, 2=detailed
}
```

### **3. Add Custom Scoring Metrics**

```python
# Add custom scoring function
def custom_fantasy_score(y_true, y_pred):
    """Custom scoring function for fantasy sports"""
    mae = mean_absolute_error(y_true, y_pred)
    # Penalize underestimation more than overestimation
    underestimation_penalty = np.mean(np.maximum(0, y_true - y_pred) ** 2)
    return -(mae + underestimation_penalty)

# Use in CV_CONFIG
CV_CONFIG = {
    'scoring': custom_fantasy_score,  # Custom scoring
    # ... other parameters
}
```

---

## 🚀 Performance Recommendations

### **For Development/Testing:**
- `USE_HARDCODED_PARAMS = True` (fast)
- `n_iter = 10` (quick search if needed)
- `cv_folds = 3` (basic validation)

### **For Production Optimization:**
- `USE_HARDCODED_PARAMS = False` (thorough search)
- `n_iter = 50-100` (comprehensive search)
- `cv_folds = 5` (robust validation)

### **For Competition/Research:**
- `USE_HARDCODED_PARAMS = False`
- `n_iter = 200+` (exhaustive search)
- `cv_folds = 10` (very robust validation)
- Enable base model parameter tuning

---

## 📋 Expected Output

### **With Hyperparameter Search:**
```
Performing hyperparameter search...
Starting hyperparameter search with 50 iterations...
Search space size: 9 parameters
Using TimeSeriesSplit with 5 folds
Fitting hyperparameter search...
Best cross-validation score: -12.3456
Best hyperparameters:
  final_estimator__n_estimators: 300
  final_estimator__max_depth: 6
  final_estimator__learning_rate: 0.1
  ...

Top 5 parameter combinations:
Rank 1: Score = -12.3456 (±1.2345)
  final_estimator__n_estimators: 300
  final_estimator__max_depth: 6
  ...
```

### **With Hard-Coded Parameters:**
```
Using hard-coded optimal parameters for fast training...
Optimal parameters:
  final_estimator__n_estimators: 200
  final_estimator__max_depth: 6
  final_estimator__learning_rate: 0.1
  ...
Training final ensemble model with hard-coded parameters...
```

---

## 🎯 Summary

The hyperparameter search and cross-validation parameters are now **fully implemented** and located in:

1. **Lines 32-55**: Configuration variables and hard-coded parameters
2. **Lines 587-612**: Hyperparameter search space definition
3. **Lines 614-675**: Hyperparameter search function implementation
4. **Lines 975-1010**: Training logic with hyperparameter search integration

**To enable hyperparameter search**: Set `USE_HARDCODED_PARAMS = False`
**To use fast training**: Keep `USE_HARDCODED_PARAMS = True` (current setting)
