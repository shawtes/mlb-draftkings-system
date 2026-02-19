import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings
import multiprocessing
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# GPU optimization settings
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['XGBOOST_VERBOSITY'] = '1'

print(f"GPU Available: {torch.cuda.is_available()}")
print(f"XGBoost Version: {xgb.__version__}")

class FastMLBEngine:
    """Simplified and faster version of the feature engineering engine"""
    def __init__(self):
        self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts']
        self.rolling_windows = [7, 14, 28]  # Reduced from [3, 7, 14, 28, 45]

    def calculate_features(self, df):
        df = df.copy()
        
        # Ensure date is datetime and sort
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])

        # Ensure base columns exist
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        # Group by player and calculate features more efficiently
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # Only essential rolling features
            for col in ['calculated_dk_fpts', 'HR', 'RBI', 'BB']:
                for window in self.rolling_windows:
                    new_features[f'{col}_mean_{window}'] = group[col].rolling(window, min_periods=1).mean()
                    if window == 28:  # Only calculate std for 28-day window
                        new_features[f'{col}_std_{window}'] = group[col].rolling(window, min_periods=1).std()
            
            # Essential temporal features
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['day_of_season'] = (group[date_col] - group[date_col].min()).dt.days

            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return enhanced_df

def calculate_dk_fpts(row):
    singles = pd.to_numeric(row.get('1B', 0), errors='coerce')
    doubles = pd.to_numeric(row.get('2B', 0), errors='coerce')
    triples = pd.to_numeric(row.get('3B', 0), errors='coerce')
    hr = pd.to_numeric(row.get('HR', 0), errors='coerce')
    rbi = pd.to_numeric(row.get('RBI', 0), errors='coerce')
    r = pd.to_numeric(row.get('R', 0), errors='coerce')
    bb = pd.to_numeric(row.get('BB', 0), errors='coerce')
    hbp = pd.to_numeric(row.get('HBP', 0), errors='coerce')
    sb = pd.to_numeric(row.get('SB', 0), errors='coerce')

    return (singles * 3 + doubles * 5 + triples * 8 + hr * 10 +
            rbi * 2 + r * 2 + bb * 2 + hbp * 2 + sb * 5)

def fast_feature_engineering(df):
    """Simplified feature engineering for speed"""
    # Basic stats
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    
    # Safe calculations with defaults
    df['wOBA'] = np.where(
        (df['AB'] + df['BB'] + df['SF'] + df['HBP']) > 0,
        (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + 
         df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] + df['SF'] + df['HBP']),
        0
    )
    
    df['BABIP'] = np.where(
        (df['AB'] - df['SO'] - df['HR'] + df['SF']) > 0,
        (df['H'] - df['HR']) / (df['AB'] - df['SO'] - df['HR'] + df['SF']),
        0
    )
    
    df['ISO'] = df['SLG'] - df['AVG']
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    return df

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, r2, mape

def clean_infinite_values(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    return df

# Simplified model (much faster than triple stacking)
def create_fast_model():
    """Create a simpler but still effective model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        xgb_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'reg:squarederror',
            'n_estimators': 100,  # Reduced from default
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_jobs': -1
        }
    else:
        xgb_params = {
            'tree_method': 'hist',
            'device': 'cpu',
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_jobs': -1
        }
    
    # Simple ensemble instead of triple stacking
    base_models = [
        ('gb', GradientBoostingRegressor(n_estimators=50, max_depth=4)),
        ('xgb', XGBRegressor(**xgb_params))
    ]
    
    return VotingRegressor(estimators=base_models)

if __name__ == "__main__":
    start_time = time.time()
    
    print("=== FAST TRAINING MODE ===")
    print("Loading dataset...")
    
    # Load smaller sample for testing
    df = pd.read_csv('4_DATA/data_20210101_to_20250618.csv',
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Take a sample for faster training (remove this line for full training)
    # df = df.sample(n=min(50000, len(df)), random_state=42)
    # print(f"Using sample size: {len(df)} rows for faster training")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)

    # Calculate calculated_dk_fpts if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DK fantasy points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    # Convert object columns to string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Create label encoders
    print("Creating label encoders...")
    le_name = LabelEncoder()
    le_team = LabelEncoder()
    
    df['Name_encoded'] = le_name.fit_transform(df['Name'])
    df['Team_encoded'] = le_team.fit_transform(df['Team'])

    # Fast feature engineering
    print("Starting fast feature engineering...")
    df = fast_feature_engineering(df)
    
    # Enhanced feature engineering (reduced)
    print("Enhanced feature engineering...")
    feature_engine = FastMLBEngine()
    df = feature_engine.calculate_features(df)
    print("Feature engineering complete.")

    # Clean data
    print("Cleaning data...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Define features (reduced set)
    numeric_features = [
        'wOBA', 'BABIP', 'ISO', 'year', 'month', 'day_of_week',
        'calculated_dk_fpts_mean_7', 'calculated_dk_fpts_mean_14', 'calculated_dk_fpts_mean_28',
        'calculated_dk_fpts_std_28', 'HR_mean_7', 'HR_mean_14', 'HR_mean_28',
        'RBI_mean_7', 'RBI_mean_14', 'RBI_mean_28', 'BB_mean_7', 'BB_mean_14', 'BB_mean_28',
        'AB', 'PA', 'H', 'HR', 'RBI', 'BB', 'SO', 'SB'
    ]
    
    categorical_features = ['Name', 'Team']

    # Prepare features
    print("Preparing features...")
    if 'calculated_dk_fpts' in df.columns:
        features = df.drop(columns=['calculated_dk_fpts'])
        target = df['calculated_dk_fpts']
    else:
        raise KeyError("'calculated_dk_fpts' not found in DataFrame columns.")
    
    # Clean features
    features = clean_infinite_values(features.copy())
    
    # Only use features that exist
    available_numeric = [f for f in numeric_features if f in features.columns]
    available_categorical = [f for f in categorical_features if f in features.columns]
    
    print(f"Using {len(available_numeric)} numeric and {len(available_categorical)} categorical features")

    # Create preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, available_numeric),
            ('cat', categorical_transformer, available_categorical)
        ])

    # Fit preprocessor and transform features
    print("Fitting preprocessor...")
    feature_subset = features[available_numeric + available_categorical]
    features_preprocessed = preprocessor.fit_transform(feature_subset)
    
    # Feature selection (reduced)
    print("Feature selection...")
    k = min(100, features_preprocessed.shape[1])  # Much smaller than 550
    selector = SelectKBest(f_regression, k=k)
    features_selected = selector.fit_transform(features_preprocessed, target)
    
    print(f"Selected {features_selected.shape[1]} features out of {features_preprocessed.shape[1]}")

    # Train fast model
    print("Training fast model...")
    fast_model = create_fast_model()
    fast_model.fit(features_selected, target)
    
    # Make predictions
    print("Making predictions...")
    all_predictions = fast_model.predict(features_selected)

    # Evaluate model
    mae, mse, r2, mape = evaluate_model(target, all_predictions)
    
    print(f'\nModel Performance:')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'R2: {r2:.4f}')
    print(f'MAPE: {mape:.4f}%')

    # Create complete pipeline
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', fast_model)
    ])

    # Save results
    print("Saving results...")
    final_results_df = pd.DataFrame({
        'Name': features['Name'],
        'Date': df['date'],
        'Actual': target,
        'Predicted': all_predictions
    })

    final_results_df.to_csv('2_PREDICTIONS/fast_predictions.csv', index=False)
    joblib.dump(joblib.dump(complete_pipeline, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/fast_model_pipeline.pkl')
    joblib.dump(joblib.dump(le_name, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/fast_label_encoder_name.pkl')
    joblib.dump(joblib.dump(le_team, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/fast_label_encoder_team.pkl')

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("Files saved:")
    print("- fast_predictions.csv")
    print("- fast_model_pipeline.pkl")
    print("- fast_label_encoder_name.pkl")
    print("- fast_label_encoder_team.pkl")
