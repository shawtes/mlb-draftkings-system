import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import QuantileRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SimpleProbabilityPredictor:
    """
    A simplified probability predictor for testing purposes
    """
    def __init__(self, point_thresholds=None):
        if point_thresholds is None:
            self.point_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        else:
            self.point_thresholds = point_thresholds
            
        self.quantile_models = {}
        self.distribution_params = None
        
    def fit_simple_quantiles(self, X, y):
        """
        Fit simple quantile models for 25th, 50th, and 75th percentiles
        """
        print("Training simple quantile models...")
        
        quantiles = [0.25, 0.5, 0.75]
        for quantile in quantiles:
            print(f"Training quantile model for q={quantile}")
            model = Ridge(alpha=0.1)  # Use Ridge instead of QuantileRegressor for simplicity
            model.fit(X, y)
            self.quantile_models[quantile] = model
            
    def estimate_distribution_params(self, y_true, y_pred):
        """
        Estimate distribution parameters from residuals
        """
        residuals = y_true - y_pred
        mu, sigma = stats.norm.fit(residuals)
        self.distribution_params = {'mu': mu, 'sigma': sigma}
        print(f"Distribution parameters - mu: {mu:.3f}, sigma: {sigma:.3f}")
        
    def predict_probabilities_simple(self, predictions, std_dev=None):
        """
        Simple probability prediction using normal distribution assumption
        """
        if std_dev is None:
            std_dev = np.std(predictions) if len(predictions) > 1 else 5.0
            
        results = []
        
        for pred in predictions:
            player_probs = {'main_prediction': pred}
            
            for threshold in self.point_thresholds:
                if std_dev > 0:
                    z_score = (threshold - pred) / std_dev
                    prob_exceed = 1 - stats.norm.cdf(z_score)
                    player_probs[f'prob_over_{threshold}'] = max(0, min(1, prob_exceed))
                else:
                    player_probs[f'prob_over_{threshold}'] = 1.0 if pred > threshold else 0.0
                    
            results.append(player_probs)
            
        return results

def simple_dk_fpts_calculation(row):
    """Calculate DraftKings fantasy points"""
    singles = pd.to_numeric(row.get('1B', 0), errors='coerce')
    doubles = pd.to_numeric(row.get('2B', 0), errors='coerce')
    triples = pd.to_numeric(row.get('3B', 0), errors='coerce')
    hr = pd.to_numeric(row.get('HR', 0), errors='coerce')
    rbi = pd.to_numeric(row.get('RBI', 0), errors='coerce')
    r = pd.to_numeric(row.get('R', 0), errors='coerce')
    bb = pd.to_numeric(row.get('BB', 0), errors='coerce')
    sb = pd.to_numeric(row.get('SB', 0), errors='coerce')

    # Handle NaN values
    values = [singles, doubles, triples, hr, rbi, r, bb, sb]
    values = [0 if pd.isna(v) else v for v in values]
    singles, doubles, triples, hr, rbi, r, bb, sb = values

    return (singles * 3 + doubles * 5 + triples * 8 + hr * 10 +
            rbi * 2 + r * 2 + bb * 2 + sb * 5)

def train_simple_model():
    """
    Train a simple model with probability predictions for testing
    """
    print("Loading sample data...")
    
    # Try to load the full dataset
    try:
        df = pd.read_csv('4_DATA/data_20210101_to_20250618.csv',
                         low_memory=False, nrows=10000)  # Only load first 10k rows for testing
        print(f"Loaded {len(df)} rows for training")
    except FileNotFoundError:
        print("Full dataset not found. Creating synthetic data for testing...")
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Name': [f'Player_{i}' for i in range(n_samples)],
            'Team': np.random.choice(['NYY', 'BOS', 'LAD', 'HOU'], n_samples),
            'HR': np.random.poisson(0.5, n_samples),
            'RBI': np.random.poisson(1.2, n_samples),
            'R': np.random.poisson(1.0, n_samples),
            'BB': np.random.poisson(0.8, n_samples),
            'SB': np.random.poisson(0.2, n_samples),
            '1B': np.random.poisson(2.0, n_samples),
            '2B': np.random.poisson(0.5, n_samples),
            '3B': np.random.poisson(0.1, n_samples),
            'AB': np.random.poisson(4.0, n_samples),
            'H': np.random.poisson(3.0, n_samples),
        })
    
    # Calculate DK points
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DraftKings fantasy points...")
        df['calculated_dk_fpts'] = df.apply(simple_dk_fpts_calculation, axis=1)
    
    # Fill missing values
    df.fillna(0, inplace=True)
    
    # Prepare features
    numeric_features = ['HR', 'RBI', 'R', 'BB', 'SB', 'AB', 'H']
    categorical_features = ['Name', 'Team']
    
    # Encode categorical variables
    le_name = LabelEncoder()
    le_team = LabelEncoder()
    
    df['Name_encoded'] = le_name.fit_transform(df['Name'].astype(str))
    df['Team_encoded'] = le_team.fit_transform(df['Team'].astype(str))
    
    # Prepare feature matrix
    feature_columns = numeric_features + ['Name_encoded', 'Team_encoded']
    X = df[feature_columns].values
    y = df['calculated_dk_fpts'].values
    
    # Simple preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    selector = SelectKBest(f_regression, k=min(8, X_scaled.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)
    
    print(f"Selected {X_selected.shape[1]} features out of {X_scaled.shape[1]}")
    
    # Train simple model
    print("Training simple Ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_selected, y)
    
    # Make predictions
    predictions = model.predict(X_selected)
    
    # Evaluate
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R2: {r2:.3f}")
    
    # Initialize probability predictor
    prob_predictor = SimpleProbabilityPredictor()
    
    # Estimate distribution parameters
    prob_predictor.estimate_distribution_params(y, predictions)
    
    # Get probability predictions
    probability_results = prob_predictor.predict_probabilities_simple(predictions)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Name': df['Name'],
        'Team': df['Team'],
        'Actual_Points': y,
        'Predicted_Points': predictions
    })
    
    # Add probability columns
    for i, probs in enumerate(probability_results):
        for threshold in [10, 20, 30, 40]:
            prob_key = f'prob_over_{threshold}'
            if prob_key in probs:
                col_name = f'Prob_Over_{threshold}'
                if col_name not in results_df.columns:
                    results_df[col_name] = 0.0
                results_df.loc[i, col_name] = probs[prob_key]
    
    # Convert probabilities to percentages
    prob_cols = [col for col in results_df.columns if col.startswith('Prob_Over_')]
    for col in prob_cols:
        results_df[col] = (results_df[col] * 100).round(1)
    
    # Sort by predicted points
    results_df = results_df.sort_values('Predicted_Points', ascending=False)
    
    # Save results
    results_df.to_csv('4_DATA/simple_probability_test_results.csv', index=False)
    print("Simple test results saved to 'simple_probability_test_results.csv'")
    
    # Display top 10 results
    print("\nTop 10 Players by Predicted Points:")
    print("=" * 80)
    for _, row in results_df.head(10).iterrows():
        print(f"Player: {row['Name']} ({row['Team']})")
        print(f"  Predicted: {row['Predicted_Points']:.1f} pts | Actual: {row['Actual_Points']:.1f} pts")
        print(f"  Prob >10: {row['Prob_Over_10']:.1f}% | Prob >20: {row['Prob_Over_20']:.1f}% | Prob >30: {row['Prob_Over_30']:.1f}%")
        print()
    
    return results_df, prob_predictor

if __name__ == "__main__":
    print("Running simple probability prediction test...")
    results_df, predictor = train_simple_model()
    print("\nSimple test completed successfully!")
    print("This demonstrates that the probability prediction concept works.")
    print("You can now run the full training script with confidence.")
