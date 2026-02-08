"""
MLB DraftKings Fantasy Points Prediction System
Ground-up implementation using NumPy and fundamental methods

This is a complete rewrite of the training system using only numpy and basic
mathematical operations, without high-level ML frameworks like sklearn, XGBoost, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
import os
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS AND PREPROCESSING
# =============================================================================

class StandardScalerNumPy:
    """Standard scaler implementation from scratch using numpy"""
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        """Fit the scaler to data"""
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        X = np.array(X)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform scaled data"""
        X = np.array(X)
        return X * self.std_ + self.mean_


class LabelEncoderNumPy:
    """Label encoder implementation from scratch"""
    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
        
    def fit(self, y):
        """Fit the encoder to labels"""
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """Transform labels to indices"""
        y = np.array(y)
        return np.array([self.class_to_index_.get(label, -1) for label in y])
    
    def fit_transform(self, y):
        """Fit and transform in one step"""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """Transform indices back to labels"""
        y = np.array(y)
        return np.array([self.classes_[idx] if 0 <= idx < len(self.classes_) else None for idx in y])


class OneHotEncoderNumPy:
    """One-hot encoder implementation from scratch"""
    def __init__(self):
        self.categories_ = None
        
    def fit(self, X):
        """Fit the encoder to data"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Transform data to one-hot encoded format"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        encoded_features = []
        for i in range(X.shape[1]):
            categories = self.categories_[i]
            column = X[:, i]
            one_hot = np.zeros((len(column), len(categories)))
            for j, cat in enumerate(categories):
                one_hot[:, j] = (column == cat).astype(float)
            encoded_features.append(one_hot)
        
        return np.hstack(encoded_features)
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


def impute_missing_values(X, strategy='mean'):
    """Simple imputation for missing values"""
    X = np.array(X, dtype=float)
    
    if strategy == 'mean':
        col_mean = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_mean[i] if not np.isnan(col_mean[i]) else 0
    elif strategy == 'median':
        col_median = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_median[i] if not np.isnan(col_median[i]) else 0
    else:  # constant
        X[np.isnan(X)] = 0
    
    return X


# =============================================================================
# MODEL IMPLEMENTATIONS FROM SCRATCH
# =============================================================================

class LinearRegressionNumPy:
    """Linear regression with optional L2 (Ridge) regularization"""
    def __init__(self, alpha=1.0, max_iter=1000, learning_rate=0.01):
        self.alpha = alpha  # L2 regularization strength
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """Fit the model using gradient descent with adaptive learning rate"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Adaptive learning rate
        lr = self.learning_rate
        prev_loss = float('inf')
        
        # Gradient descent with early stopping
        for iteration in range(self.max_iter):
            # Predictions
            y_pred = X @ self.weights + self.bias
            
            # Calculate loss
            loss = np.mean((y_pred - y) ** 2) + (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
            
            # Check for convergence
            if abs(prev_loss - loss) < 1e-6:
                break
            
            # Adapt learning rate
            if loss > prev_loss:
                lr *= 0.5  # Reduce learning rate if loss increased
            else:
                lr *= 1.01  # Slightly increase if improving
                
            lr = min(lr, self.learning_rate)  # Cap at initial learning rate
            prev_loss = loss
            
            # Compute gradients
            dw = (1 / n_samples) * (X.T @ (y_pred - y)) + (self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Clip gradients to prevent exploding gradients
            dw = np.clip(dw, -10, 10)
            db = np.clip(db, -10, 10)
            
            # Update parameters
            self.weights -= lr * dw
            self.bias -= lr * db
            
            # Check for NaN
            if np.any(np.isnan(self.weights)) or np.isnan(self.bias):
                # Reset to safe values
                self.weights = np.zeros(n_features)
                self.bias = 0
                break
            
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias


class DecisionTreeRegressorNumPy:
    """Simple decision tree regressor from scratch"""
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        
    def _mse(self, y):
        """Calculate mean squared error"""
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)
    
    def _best_split(self, X, y):
        """Find the best split for the data"""
        best_mse = float('inf')
        best_split = None
        
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature
            values = np.unique(X[:, feature_idx])
            
            # Try each threshold
            for threshold in values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate MSE for this split
                left_mse = self._mse(y[left_mask])
                right_mse = self._mse(y[right_mask])
                total_mse = left_mse + right_mse
                
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }
        
        return best_split
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {'value': np.mean(y)}
        
        # Find best split
        split = self._best_split(X, y)
        
        if split is None:
            return {'value': np.mean(y)}
        
        # Recursively build left and right subtrees
        left_tree = self._build_tree(X[split['left_mask']], y[split['left_mask']], depth + 1)
        right_tree = self._build_tree(X[split['right_mask']], y[split['right_mask']], depth + 1)
        
        return {
            'feature_idx': split['feature_idx'],
            'threshold': split['threshold'],
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Fit the decision tree"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, tree):
        """Predict for a single sample"""
        if 'value' in tree:
            return tree['value']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x, self.tree) for x in X])


class GradientBoostingRegressorNumPy:
    """Gradient boosting implementation from scratch"""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        """Fit the gradient boosting model"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Initial prediction is the mean
        self.initial_prediction = np.mean(y)
        
        # Current predictions
        current_predictions = np.full(len(y), self.initial_prediction)
        
        # Build trees
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for MSE)
            residuals = y - current_predictions
            
            # Fit a tree to the residuals
            tree = DecisionTreeRegressorNumPy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            current_predictions += self.learning_rate * predictions
            
            # Store the tree
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                mse = np.mean((y - current_predictions) ** 2)
                print(f"  Tree {i+1}/{self.n_estimators}, MSE: {mse:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X, dtype=float)
        predictions = np.full(len(X), self.initial_prediction)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions


class StackingRegressorNumPy:
    """Stacking ensemble regressor from scratch"""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        """Fit all base models and the meta model"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Fit base models and collect predictions
        base_predictions = []
        for i, (name, model) in enumerate(self.base_models):
            print(f"  Fitting base model {i+1}/{len(self.base_models)}: {name}")
            model.fit(X, y)
            preds = model.predict(X)
            base_predictions.append(preds)
        
        # Stack predictions as features for meta model
        meta_features = np.column_stack(base_predictions)
        
        # Fit meta model
        print(f"  Fitting meta model")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        X = np.array(X, dtype=float)
        
        # Get predictions from base models
        base_predictions = []
        for name, model in self.base_models:
            preds = model.predict(X)
            base_predictions.append(preds)
        
        # Stack predictions
        meta_features = np.column_stack(base_predictions)
        
        # Get final prediction from meta model
        return self.meta_model.predict(meta_features)


class VotingRegressorNumPy:
    """Voting ensemble regressor from scratch"""
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        """Fit all models"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        for i, (name, model) in enumerate(self.models):
            print(f"  Fitting model {i+1}/{len(self.models)}: {name}")
            model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Make predictions by averaging all models"""
        X = np.array(X, dtype=float)
        
        predictions = []
        for name, model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        
        # Average predictions
        return np.mean(predictions, axis=0)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class EnhancedMLBFinancialStyleEngine:
    """Financial-style feature engineering for MLB data"""
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        """Calculate rolling statistics and momentum features"""
        df = df.copy()
        
        # Ensure date is datetime and sort
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])

        # Standardize opportunity columns
        if 'PA' not in df.columns and 'PA.1' in df.columns:
            df['PA'] = df['PA.1']
        if 'AB' not in df.columns and 'AB.1' in df.columns:
            df['AB'] = df['AB.1']
            
        # Ensure base columns exist
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        # Group by player and calculate features
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # Rolling statistics for each stat
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    # Rolling mean
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    # Exponential moving average
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    # Rate of change
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
            
            # Volatility features (Bollinger Bands)
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_std_{window}'] = std
                new_features[f'dk_fpts_band_width_{window}'] = 2 * std

            # Opportunity features
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()

            # Ratio features
            for col in ['HR', 'RBI', 'BB', 'H', 'R']:
                if col in group.columns and 'PA' in group.columns:
                    pa_sum = group['PA'].replace(0, 1)  # Avoid division by zero
                    new_features[f'{col}_per_pa'] = group[col] / pa_sum
            
            # Temporal features
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)

            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        
        # Clean up
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        enhanced_df = enhanced_df.fillna(0)
        
        return enhanced_df


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Percentage Error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


# =============================================================================
# DRAFTKINGS POINTS CALCULATION
# =============================================================================

def calculate_dk_fpts(row):
    """Calculate DraftKings fantasy points"""
    points = 0
    
    # Single = 3 points
    points += row.get('1B', 0) * 3
    
    # Double = 5 points
    points += row.get('2B', 0) * 5
    
    # Triple = 8 points
    points += row.get('3B', 0) * 8
    
    # Home Run = 10 points
    points += row.get('HR', 0) * 10
    
    # RBI = 2 points
    points += row.get('RBI', 0) * 2
    
    # Run = 2 points
    points += row.get('R', 0) * 2
    
    # Walk = 2 points
    points += row.get('BB', 0) * 2
    
    # Hit by Pitch = 2 points
    points += row.get('HBP', 0) * 2
    
    # Stolen Base = 5 points
    points += row.get('SB', 0) * 5
    
    return points


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert date column
    date_col = 'game_date' if 'game_date' in df.columns else 'date'
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(by=['Name', date_col])
    
    # Calculate DraftKings points if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DraftKings fantasy points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
    
    # Handle missing values
    df = df.fillna(0)
    
    # Convert object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def engineer_features(df):
    """Apply feature engineering"""
    print("Starting feature engineering...")
    
    # Financial-style features
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    
    print("Feature engineering complete.")
    return df


def prepare_features_and_target(df, feature_cols, target_col='calculated_dk_fpts'):
    """Prepare features and target for training"""
    print("Preparing features and target...")
    
    # Ensure all feature columns exist
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            available_features.append(col)
        else:
            print(f"Warning: Feature '{col}' not found in dataframe")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            encoder = LabelEncoderNumPy()
            X[col] = encoder.fit_transform(X[col])
    
    # Convert to numpy arrays
    X = X.values.astype(float)
    y = y.values.astype(float)
    
    # Handle any remaining NaN or inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
    
    return X, y


def train_model(X, y):
    """Train the ensemble model"""
    print("\n" + "="*80)
    print("TRAINING ENSEMBLE MODEL")
    print("="*80)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScalerNumPy()
    X_scaled = scaler.fit_transform(X)
    
    # Define base models
    print("\nDefining base models...")
    base_models = [
        ('ridge_1', LinearRegressionNumPy(alpha=1.0, max_iter=1000, learning_rate=0.01)),
        ('ridge_2', LinearRegressionNumPy(alpha=10.0, max_iter=1000, learning_rate=0.01)),
        ('ridge_3', LinearRegressionNumPy(alpha=0.1, max_iter=1000, learning_rate=0.01)),
    ]
    
    # Train base models for stacking
    print("\nTraining base models for stacking ensemble...")
    for name, model in base_models:
        print(f"\nTraining {name}...")
        model.fit(X_scaled, y)
    
    # Train gradient boosting model
    print("\nTraining gradient boosting model...")
    gb_model = GradientBoostingRegressorNumPy(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=20
    )
    gb_model.fit(X_scaled, y)
    
    # Create voting ensemble
    print("\nCreating voting ensemble...")
    voting_models = base_models + [('gradient_boosting', gb_model)]
    voting_ensemble = VotingRegressorNumPy(voting_models)
    
    # Since models are already fitted, we don't need to fit again
    # Just prepare the voting ensemble structure
    voting_ensemble.models = voting_models
    
    # Create stacking ensemble with gradient boosting as meta model
    print("\nCreating stacking ensemble...")
    meta_model = LinearRegressionNumPy(alpha=1.0, max_iter=500, learning_rate=0.01)
    stacking_ensemble = StackingRegressorNumPy(base_models, meta_model)
    
    # Train stacking
    print("\nTraining stacking ensemble...")
    stacking_ensemble.fit(X_scaled, y)
    
    # Final ensemble: average of voting and stacking
    print("\nCreating final ensemble (average of voting and stacking)...")
    final_ensemble = {
        'voting': voting_ensemble,
        'stacking': stacking_ensemble,
        'scaler': scaler
    }
    
    print("\nModel training complete!")
    print("="*80)
    
    return final_ensemble


def predict_with_ensemble(ensemble, X):
    """Make predictions using the ensemble"""
    # Scale features
    X_scaled = ensemble['scaler'].transform(X)
    
    # Get predictions from both ensembles
    voting_pred = ensemble['voting'].predict(X_scaled)
    stacking_pred = ensemble['stacking'].predict(X_scaled)
    
    # Average the predictions
    final_pred = (voting_pred + stacking_pred) / 2
    
    return final_pred


def calculate_probability_predictions(y_pred, thresholds=[5, 10, 15, 20, 25, 30]):
    """
    Calculate probability of exceeding thresholds
    Uses a simple heuristic based on predicted values
    """
    probabilities = {}
    
    # Estimate standard deviation from predictions
    pred_std = np.std(y_pred)
    if pred_std == 0:
        pred_std = 5.0  # Default
    
    for threshold in thresholds:
        # Simple probability model: higher predicted value = higher probability
        z_scores = (y_pred - threshold) / pred_std
        # Use sigmoid-like transformation
        prob = 1 / (1 + np.exp(-z_scores))
        probabilities[f'prob_over_{threshold}'] = prob
    
    # Add prediction intervals
    probabilities['prediction_lower_80'] = y_pred - 1.28 * pred_std
    probabilities['prediction_upper_80'] = y_pred + 1.28 * pred_std
    probabilities['prediction_std'] = np.full_like(y_pred, pred_std)
    
    return probabilities


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*80)
    print("MLB DRAFTKINGS FANTASY POINTS PREDICTION SYSTEM")
    print("Ground-up NumPy Implementation")
    print("="*80 + "\n")
    
    # Data path (update this to match your environment)
    data_path = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    
    # Check if file exists, otherwise look for alternatives
    if not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
        print("Searching for alternative data files...")
        
        # Look for CSV files in current directory and parent directories
        possible_paths = [
            os.path.join(script_dir, 'merged_fangraphs_data.csv'),
            os.path.join(os.path.dirname(script_dir), 'merged_fangraphs_data.csv'),
            os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'merged_fangraphs_data.csv'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Found data file at: {data_path}")
                break
        else:
            print("ERROR: Could not find data file!")
            print("Please update the 'data_path' variable in the script.")
            exit(1)
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Define feature columns to use
    feature_cols = [
        'Name', 'Team',
        'wOBA', 'BABIP', 'ISO', 'wRAA', 'wRC', 'wRC+',
        'AB', 'PA', 'H', 'HR', 'RBI', 'R', 'BB', 'SB',
        '1B', '2B', '3B',
        'flyBalls',
        'HR_sma_7', 'HR_sma_14', 'HR_sma_28',
        'RBI_sma_7', 'RBI_sma_14', 'RBI_sma_28',
        'calculated_dk_fpts_sma_7', 'calculated_dk_fpts_sma_14', 'calculated_dk_fpts_sma_28',
        'calculated_dk_fpts_ema_7', 'calculated_dk_fpts_ema_14',
        'calculated_dk_fpts_std_7', 'calculated_dk_fpts_std_14', 'calculated_dk_fpts_std_28',
        'HR_per_pa', 'RBI_per_pa', 'BB_per_pa', 'H_per_pa',
        'day_of_week', 'month', 'is_weekend'
    ]
    
    # Prepare features and target
    X, y = prepare_features_and_target(df, feature_cols)
    
    # Split data into train and test (simple 80/20 split by time)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    ensemble_model = train_model(X_train, y_train)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred_test = predict_with_ensemble(ensemble_model, X_test)
    
    # Calculate metrics
    print("\nTest Set Performance:")
    metrics = calculate_metrics(y_test, y_pred_test)
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Make predictions on full dataset
    print("\nMaking predictions on full dataset...")
    y_pred_full = predict_with_ensemble(ensemble_model, X)
    
    # Calculate full dataset metrics
    print("\nFull Dataset Performance:")
    metrics_full = calculate_metrics(y, y_pred_full)
    print(f"  MAE:  {metrics_full['mae']:.4f}")
    print(f"  MSE:  {metrics_full['mse']:.4f}")
    print(f"  RMSE: {metrics_full['rmse']:.4f}")
    print(f"  R²:   {metrics_full['r2']:.4f}")
    print(f"  MAPE: {metrics_full['mape']:.2f}%")
    
    # Calculate probability predictions
    print("\nCalculating probability predictions...")
    prob_predictions = calculate_probability_predictions(y_pred_full)
    
    # Save results
    print("\nSaving results...")
    
    # Prepare results dataframe
    date_col = 'date' if 'date' in df.columns else 'game_date'
    results_df = pd.DataFrame({
        'Name': df['Name'],
        'Date': df[date_col],
        'Actual': y,
        'Predicted': y_pred_full
    })
    
    # Add probability predictions
    for key, values in prob_predictions.items():
        results_df[key] = values
    
    # Save to CSV
    output_path = os.path.join(script_dir, 'final_predictions_numpy.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Save probability summary
    prob_summary = pd.DataFrame({
        'Name': df['Name'],
        'Date': df[date_col],
        'Predicted_FPTS': y_pred_full,
        'Prob_Over_5': prob_predictions['prob_over_5'],
        'Prob_Over_10': prob_predictions['prob_over_10'],
        'Prob_Over_15': prob_predictions['prob_over_15'],
        'Prob_Over_20': prob_predictions['prob_over_20'],
        'Prediction_Lower_80': prob_predictions['prediction_lower_80'],
        'Prediction_Upper_80': prob_predictions['prediction_upper_80'],
        'Prediction_Std': prob_predictions['prediction_std']
    })
    
    prob_summary_path = os.path.join(script_dir, 'probability_summary_numpy.csv')
    prob_summary.to_csv(prob_summary_path, index=False)
    print(f"Probability summary saved to: {prob_summary_path}")
    
    # Save the model
    model_path = os.path.join(script_dir, 'ensemble_model_numpy.pkl')
    joblib.dump(ensemble_model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature columns for future use
    feature_info = {
        'feature_cols': feature_cols,
        'n_features': X.shape[1]
    }
    feature_info_path = os.path.join(script_dir, 'feature_info_numpy.pkl')
    joblib.dump(feature_info, feature_info_path)
    print(f"Feature info saved to: {feature_info_path}")
    
    # Execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80 + "\n")
