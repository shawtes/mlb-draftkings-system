"""
Enhanced MLB DraftKings Fantasy Points Prediction Model

This script implements a comprehensive machine learning pipeline for predicting 
MLB DraftKings fantasy points using advanced stacking techniques and hyperparameter tuning.

Key Features:
- Multi-level stacking with 9 diverse base models
- Hyperparameter optimization using RandomizedSearchCV
- Financial-style feature engineering (RSI, MACD, Bollinger Bands)
- Probability predictions for different point thresholds
- GPU-accelerated XGBoost training when available
- Comprehensive model evaluation and performance metrics

Model Architecture:
1. Base Models: LinearRegression, Ridge, Lasso, DecisionTree, SVR, RandomForest, 
   XGBoost, Bagging, Polynomial Features
2. Stacking Level 1: StackingRegressor with XGBoost final estimator
3. Stacking Level 2: Final StackingRegressor with hyperparameter-tuned base model

Updated: July 2025 - Added comprehensive stacking and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import concurrent.futures
import time
import torch
print(torch.cuda.is_available())
import xgboost as xgb
print(xgb.__version__)
print(xgb.get_config())

print(xgb.Booster) # Should not error if installed correctly
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor, RandomForestClassifier, VotingClassifier, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, classification_report, accuracy_score
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import warnings
import multiprocessing
import os
import torch
from scipy import stats
from sklearn.linear_model import QuantileRegressor

# Omen 35L Specific Optimizations
def optimize_for_omen_35l():
    """
    Optimize settings specifically for HP Omen 35L with i7 processor
    """
    import psutil
    
    # Detect system specs
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🖥️ Omen 35L Detected:")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   RAM: {memory_gb:.1f} GB")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    
    # Optimize settings based on typical Omen 35L specs
    if cpu_count >= 8:  # i7 typically has 8+ logical cores
        recommended_workers = min(6, cpu_count - 2)  # Leave 2 cores for system
        print(f"   Recommended workers: {recommended_workers}")
        return {
            'n_jobs': recommended_workers,
            'max_workers': recommended_workers,
            'chunk_size': 30000 if memory_gb >= 16 else 20000,
            'hyperparameter_iterations': 25 if memory_gb >= 16 else 20
        }
    else:
        return {
            'n_jobs': 4,
            'max_workers': 4, 
            'chunk_size': 20000,
            'hyperparameter_iterations': 15
        }

# Get optimized settings for your hardware
omen_settings = optimize_for_omen_35l()

# GPU optimization settings
if torch.cuda.is_available():
    # Set environment variables for better GPU performance
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Reduce XGBoost verbosity for cleaner output
    os.environ['XGBOOST_VERBOSITY'] = '1'

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Also suppress XGBoost device warnings for cleaner output
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')
# Suppress sklearn version warnings for pickled objects
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

class EnhancedMLBFinancialStyleEngine:
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28, 45]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        
        # --- Preprocessing ---
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
                print(f"Warning: Column '{col}' not found. Initialized with 0.")

        # Group by player
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # --- Momentum Features (like RSI, MACD) ---
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    # Rolling means (SMA)
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    # Exponential rolling means (EMA)
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    # Rate of Change (Momentum)
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                # Performance vs moving average
                if f'{col}_sma_28' in new_features:
                    new_features[f'{col}_vs_sma_28'] = (group[col] / new_features[f'{col}_sma_28']) - 1
            
            # --- Volatility Features (like Bollinger Bands) ---
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                new_features[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                if mean is not None and not mean.empty:
                    new_features[f'dk_fpts_band_width_{window}'] = (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}']) / mean
                    new_features[f'dk_fpts_band_position_{window}'] = (group['calculated_dk_fpts'] - new_features[f'dk_fpts_lower_band_{window}']) / (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}'])

            # --- "Volume" (PA/AB) based Features ---
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    new_features[f'{vol_col}_ratio'] = group[vol_col] / new_features[f'{vol_col}_roll_mean_28']
                    new_features[f'dk_fpts_{vol_col}_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group[vol_col])

            # --- Interaction / Ratio Features ---
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns and 'PA' in group.columns and group['PA'].sum() > 0:
                    new_features[f'{col}_per_pa'] = group[col] / group['PA']
            
            # --- Temporal Features ---
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)
            new_features['day_of_week_sin'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
            new_features['day_of_week_cos'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)

            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        # Final cleanup        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df

# Define constants for calculations
# League averages for 2020 to 2024
league_avg_wOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315
}

league_avg_HR_FlyBall = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141
}

# wOBA weights for 2020 to 2024
wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

selected_features = [
     'wOBA', 'BABIP', 'ISO', 'FIP', 'wRAA', 'wRC', 'wRC+', 
    'flyBalls', 'year', 'month', 'day', 'day_of_week', 'day_of_season',
    'singles', 'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI','AB', 'WAR'  
]

engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast', 'Name_encoded', 'team_encoded','wRC+', 'wRAA', 'wOBA',   
]
selected_features += engineered_features

def calculate_dk_fpts(row):
    # Ensure all required columns are present and numeric, defaulting to 0
    # This prevents errors if a stat column is missing from a row
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

def engineer_features(df, date_series=None):
    if date_series is None:
        date_series = df['date']
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    # Extract date features
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days

    # Define default values to handle years not present in the lookup tables
    default_wOBA = 0.317  # A reasonable league average
    default_HR_FlyBall = 0.143 # A reasonable league average
    default_wOBA_weights = wOBA_weights[2022] # Use a recent year as default

    # Calculate key statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics (with safe fallbacks for missing years)
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA.get(x['year'], default_wOBA)) / 1.15) * x['AB'] if x['AB'] > 0 else 0, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA.get(x['year'], default_wOBA) * 100) if x['AB'] > 0 and league_avg_wOBA.get(x['year'], default_wOBA) > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) if league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) > 0 else 0, axis=1)

    # Calculate singles
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    # Calculate wOBA using year-specific weights (with safe fallbacks)
    df['wOBA_Statcast'] = df.apply(lambda x: (
        wOBA_weights.get(x['year'], default_wOBA_weights)['BB'] * x.get('BB', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HBP'] * x.get('HBP', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['1B'] * x.get('1B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['2B'] * x.get('2B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['3B'] * x.get('3B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HR'] * x.get('HR', 0)
    ) / (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) if (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) > 0 else 0, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = df.apply(lambda x: (
        x.get('1B', 0) + (2 * x.get('2B', 0)) + (3 * x.get('3B', 0)) + (4 * x.get('HR', 0))
    ) / x.get('AB', 1) if x.get('AB', 1) > 0 else 0, axis=1)

    # Calculate RAR_Statcast (Runs Above Replacement)
    df['RAR_Statcast'] = df['WAR'] * 10 if 'WAR' in df.columns else 0

    # Calculate Offense_Statcast
    df['Offense_Statcast'] = df['wRAA'] + df['BsR'] if 'BsR' in df.columns else df['wRAA']

    # Calculate Dollars_Statcast
    WAR_conversion_factor = 8.0  # Example conversion factor, can be adjusted
    df['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor if 'WAR' in df.columns else 0

    # Calculate WPA/LI_Statcast
    df['WPA/LI_Statcast'] = df['WPA/LI'] if 'WPA/LI' in df.columns else 0

    # Calculate rolling statistics if 'calculated_dk_fpts' is present
    if 'calculated_dk_fpts' in df.columns:
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    return df

def process_chunk(chunk, date_series=None):
    return engineer_features(chunk, date_series)

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    start_time = time.time()
    
    # Use sequential processing when GPU is available to avoid CUDA context conflicts
    if torch.cuda.is_available():
        print("GPU detected - using sequential processing for feature engineering")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_date_series = date_series[i*chunksize:(i+1)*chunksize]
            processed_chunk = process_chunk(chunk, chunk_date_series)
            processed_chunks.append(processed_chunk)
    else:
        max_workers = min(multiprocessing.cpu_count(), 4)  # Limit to 4 workers for stability
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            processed_chunks = list(executor.map(process_chunk, chunks, [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using default values.")
            default_row = pd.DataFrame([{col: 0 for col in df.columns if col != 'calculated_dk_fpts'}])
            default_row['date'] = prediction_date
            default_row['Name'] = player
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use all available data, up to 45 most recent games
            player_df = player_df.head(45)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

def process_predictions(chunk, pipeline):
    features = chunk.drop(columns=['calculated_dk_fpts'])
    # Clean the features to ensure no infinite or excessively large values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    features_preprocessed = pipeline.named_steps['preprocessor'].transform(features)
    features_selected = pipeline.named_steps['selector'].transform(features_preprocessed)
    chunk['predicted_dk_fpts'] = pipeline.named_steps['model'].predict(features_selected)
    return chunk

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        # Get all unique players from training data
        all_players = train_data['Name'].unique()
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, all_players, current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def evaluate_model(y_true, y_pred):
    print("Evaluating model...")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("Model evaluation completed.")
    return mae, mse, r2, mape

class ProbabilityPredictor:
    """
    A class to predict probabilities of achieving different point thresholds
    using quantile regression and distribution modeling.
    """
    def __init__(self, point_thresholds=None, quantiles=None):
        if point_thresholds is None:
            # DFS point thresholds in 5-point increments for realistic DraftKings analysis
            self.point_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        else:
            self.point_thresholds = point_thresholds
            
        if quantiles is None:
            # Quantiles for distribution modeling
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            self.quantiles = quantiles
            
        self.quantile_models = {}
        self.distribution_params = None
        
    def fit_quantile_models(self, X, y, preprocessor, selector):
        """
        Fit quantile regression models for different quantiles
        """
        print("Training quantile regression models...")
        
        # Transform features using the same preprocessor and selector
        X_transformed = preprocessor.transform(X)
        X_selected = selector.transform(X_transformed)
        
        for quantile in self.quantiles:
            print(f"Training quantile model for q={quantile}")
            model = QuantileRegressor(quantile=quantile, alpha=0.01, solver='highs')
            model.fit(X_selected, y)
            self.quantile_models[quantile] = model
            
    def predict_quantiles(self, X, preprocessor, selector):
        """
        Predict quantiles for given features
        """
        X_transformed = preprocessor.transform(X)
        X_selected = selector.transform(X_transformed)
        
        quantile_predictions = {}
        for quantile, model in self.quantile_models.items():
            quantile_predictions[quantile] = model.predict(X_selected)
            
        return quantile_predictions
        
    def estimate_distribution_params(self, y_true, y_pred):
        """
        Estimate distribution parameters from residuals
        """
        residuals = y_true - y_pred
        
        # Fit normal distribution to residuals
        mu, sigma = stats.norm.fit(residuals)
        self.distribution_params = {'mu': mu, 'sigma': sigma}
        
        print(f"Distribution parameters - mu: {mu:.3f}, sigma: {sigma:.3f}")
        
    def predict_probabilities(self, X, main_predictions, preprocessor, selector):
        """
        Predict probabilities of achieving different point thresholds
        """
        if not self.quantile_models:
            raise ValueError("Quantile models not fitted. Call fit_quantile_models first.")
            
        results = []
        
        # Get quantile predictions
        quantile_preds = self.predict_quantiles(X, preprocessor, selector)
        
        for i, main_pred in enumerate(main_predictions):
            player_probs = {'main_prediction': main_pred}
            
            # Method 1: Using quantile regression
            if len(self.quantile_models) >= 3:
                # Estimate distribution from quantiles
                q25 = quantile_preds[0.25][i] if 0.25 in quantile_preds else main_pred
                q50 = quantile_preds[0.5][i] if 0.5 in quantile_preds else main_pred
                q75 = quantile_preds[0.75][i] if 0.75 in quantile_preds else main_pred
                
                # Estimate standard deviation from IQR
                iqr = q75 - q25
                estimated_sigma = iqr / 1.35  # IQR ≈ 1.35 * σ for normal distribution
                
                for threshold in self.point_thresholds:
                    if estimated_sigma > 0:
                        # Probability of exceeding threshold using normal approximation
                        z_score = (threshold - main_pred) / estimated_sigma
                        prob_exceed = 1 - stats.norm.cdf(z_score)
                        player_probs[f'prob_over_{threshold}'] = max(0, min(1, prob_exceed))
                    else:
                        # If no variance, use deterministic approach
                        player_probs[f'prob_over_{threshold}'] = 1.0 if main_pred > threshold else 0.0
            
            # Method 2: Using residual distribution (if available)
            if self.distribution_params:
                sigma = self.distribution_params['sigma']
                for threshold in self.point_thresholds:
                    if sigma > 0:
                        z_score = (threshold - main_pred) / sigma
                        prob_exceed = 1 - stats.norm.cdf(z_score)
                        player_probs[f'prob_over_{threshold}_residual'] = max(0, min(1, prob_exceed))
            
            # Add quantile information
            for quantile, values in quantile_preds.items():
                player_probs[f'quantile_{int(quantile*100)}'] = values[i]
                
            results.append(player_probs)
            
        return results
        
    def create_probability_summary(self, probability_results, player_names):
        """
        Create a summary DataFrame with probability predictions
        """
        summary_data = []
        
        for i, (name, probs) in enumerate(zip(player_names, probability_results)):
            row = {'Name': name, 'Predicted_Points': probs['main_prediction']}
            
            # Add probability columns
            for threshold in self.point_thresholds:
                if f'prob_over_{threshold}' in probs:
                    row[f'Prob_Over_{threshold}'] = f"{probs[f'prob_over_{threshold}']:.1%}"
                    
            # Add quantile information
            for quantile in self.quantiles:
                quantile_key = f'quantile_{int(quantile*100)}'
                if quantile_key in probs:
                    row[f'Q{int(quantile*100)}'] = f"{probs[quantile_key]:.1f}"
                    
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

def create_enhanced_predictions_with_probabilities(complete_pipeline, probability_predictor, 
                                                 features, target, player_names):
    """
    Create predictions with probability estimates
    """
    print("Creating enhanced predictions with probabilities...")
    
    # Get main predictions
    main_predictions = complete_pipeline.predict(features)
    
    # Get probability predictions
    preprocessor = complete_pipeline.named_steps['preprocessor']
    selector = complete_pipeline.named_steps['selector']
    
    probability_results = probability_predictor.predict_probabilities(
        features, main_predictions, preprocessor, selector
    )
    
    # Create summary DataFrame
    prob_summary = probability_predictor.create_probability_summary(
        probability_results, player_names
    )
    
    return main_predictions, probability_results, prob_summary

# Define paths for saving and loading LabelEncoders and Scalers
name_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_name_sep2.pkl'
team_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_team_sep2.pkl'
scaler_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/scaler_sep2.pkl'

def clean_infinite_values(df):
    # Only process numeric columns to save memory
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        df[col].fillna(df[col].mean(), inplace=True)
    # For non-numeric columns, fill NaN with 'Unknown'
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col].fillna('Unknown', inplace=True)
    return df

def load_or_create_label_encoders(df):
    # Handle version compatibility by recreating encoders if needed
    try:
        if os.path.exists(name_encoder_path):
            le_name = joblib.load(name_encoder_path)
            # Test if the encoder works with current version
            le_name.fit(df['Name'])
        else:
            raise FileNotFoundError("Name encoder not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new name encoder due to compatibility issues...")
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)

    try:
        if os.path.exists(team_encoder_path):
            le_team = joblib.load(team_encoder_path)
            # Test if the encoder works with current version
            le_team.fit(df['Team'])
        else:
            raise FileNotFoundError("Team encoder not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new team encoder due to compatibility issues...")
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    return le_name, le_team

def load_or_create_scaler(df, numeric_features):
    # Force recreation of scaler to avoid version compatibility issues
    # Remove existing scaler file if it exists
    if os.path.exists(scaler_path):
        print("Removing existing scaler due to version compatibility...")
        os.remove(scaler_path)
    
    scaler = StandardScaler()
    # Don't modify the original dataframe, just fit the scaler
    scaler.fit(df[numeric_features])
    joblib.dump(scaler, scaler_path)
    print("New scaler created and saved.")
    return scaler

def save_feature_importance(pipeline, output_csv_path, output_plot_path):
    print("Saving feature importances...")
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    selector = pipeline.named_steps['selector']

    # Handle VotingRegressor - extract feature importances from base models
    feature_importances = None
    
    try:
        # First, try to get feature importances from XGBoost if available
        if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
            xgb_model = model.named_estimators_['xgb']
            if hasattr(xgb_model, 'feature_importances_'):
                feature_importances = xgb_model.feature_importances_
                print("Using XGBoost feature importances")
        
        # If XGBoost not available, try GradientBoosting
        elif hasattr(model, 'named_estimators_') and 'gb' in model.named_estimators_:
            gb_model = model.named_estimators_['gb']
            if hasattr(gb_model, 'feature_importances_'):
                feature_importances = gb_model.feature_importances_
                print("Using GradientBoosting feature importances")
        
        # Fallback to Lasso coefficients if tree-based models not available
        elif hasattr(model, 'named_estimators_') and 'lasso' in model.named_estimators_:
            lasso_model = model.named_estimators_['lasso']
            if hasattr(lasso_model, 'coef_'):
                feature_importances = np.abs(lasso_model.coef_)
                print("Using Lasso coefficients as feature importance proxy")
        
        # Final fallback to Ridge coefficients
        elif hasattr(model, 'named_estimators_') and 'ridge' in model.named_estimators_:
            ridge_model = model.named_estimators_['ridge']
            if hasattr(ridge_model, 'coef_'):
                feature_importances = np.abs(ridge_model.coef_)
                print("Using Ridge coefficients as feature importance proxy")
        
        # If it's not a VotingRegressor, try direct access
        elif hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            print("Using direct model feature importances")
        
        if feature_importances is None:
            raise ValueError("Could not extract feature importances from any base model")
            
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        # Create dummy importances as last resort
        n_features = selector.get_support().sum()
        feature_importances = np.ones(n_features) / n_features
        print("Using uniform feature importances as fallback")
    
    # Get all feature names from the preprocessor
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    all_feature_names = np.concatenate([numeric_features, cat_features])
    
    # Get the mask of selected features from the selector
    support_mask = selector.get_support()
    
    # Get the names of ONLY the selected features
    selected_feature_names = all_feature_names[support_mask]

    if len(feature_importances) != len(selected_feature_names):
        print(f"Warning: Feature importance length ({len(feature_importances)}) doesn't match selected features ({len(selected_feature_names)})")
        # Adjust lengths if needed
        min_len = min(len(feature_importances), len(selected_feature_names))
        feature_importances = feature_importances[:min_len]
        selected_feature_names = selected_feature_names[:min_len]
    
    feature_importance_df = pd.DataFrame({
        'Feature': selected_feature_names,
        'Importance': feature_importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    feature_importance_df.to_csv(output_csv_path, index=False)
    print(f"Feature importances saved to {output_csv_path}")

    # Plot top 25 features for readability
    top_25_features = feature_importance_df.head(25)

    plt.figure(figsize=(12, 10))
    plt.barh(top_25_features['Feature'], top_25_features['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 25 Feature Importances for DraftKings MLB Fantasy Points')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()
    print(f"Feature importance plot saved to {output_plot_path}")

# Define base models for stacking with comprehensive model variety
base_models = [
    ('lr', LinearRegression()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('dt', DecisionTreeRegressor()),
    ('svr', SVR()),
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('xgb', XGBRegressor(objective='reg:squarederror', n_jobs=-1)),
    ('bagging', BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10)),
    ('poly', make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
]

# Define stacking model with XGBoost final estimator
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)

# Define hyperparameter search space for the final model
param_grid = {
    'final_estimator__n_estimators': [50, 100, 200],
    'final_estimator__max_depth': [3, 5, 7],
    'final_estimator__learning_rate': [0.05, 0.1, 0.2],
    'final_estimator__subsample': [0.7, 0.8, 0.9]
}

# Function to create hyperparameter-tuned model
def create_hypertuned_model(X, y, cv_folds=3):
    """
    Create a hyperparameter-tuned stacking model using GridSearchCV
    """
    print("Starting hyperparameter tuning...")
    
    # Use optimized settings for Omen 35L
    n_iter = omen_settings.get('hyperparameter_iterations', 20)
    
    # Use RandomizedSearchCV for faster tuning with large parameter space
    search = RandomizedSearchCV(
        stacking_model,
        param_grid,
        n_iter=n_iter,  # Optimized for your hardware
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=1,  # Use single job to avoid conflicts
        verbose=1,
        random_state=42
    )
    
    print(f"🚀 Omen 35L Optimization: Testing {n_iter} parameter combinations")
    
    # Fit the search
    search.fit(X, y)
    
    print(f"Best parameters found: {search.best_params_}")
    print(f"Best cross-validation score: {-search.best_score_:.4f}")
    
    return search.best_estimator_

# Define final model with nested stacking
final_model = StackingRegressor(
    estimators=[('stacking', stacking_model)],
    final_estimator=XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.7
    )
)

# Check for GPU availability and update XGBoost models accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Update XGBoost models in base_models and stacking_model for GPU if available
if torch.cuda.is_available():
    xgb_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }
else:
    xgb_params = {
        'tree_method': 'hist',
        'device': 'cpu',
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }

# Update XGBoost models with GPU parameters
for i, (name, model) in enumerate(base_models):
    if name == 'xgb':
        base_models[i] = (name, XGBRegressor(**xgb_params))
        break

# Update stacking model XGBoost estimators
stacking_model.final_estimator.set_params(**xgb_params)
final_model.final_estimator.set_params(**xgb_params)

if __name__ == "__main__":
    start_time = time.time()
    
    print("Loading large dataset in chunks to avoid memory issues...")
    csv_path = r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv'
    chunksize = omen_settings.get('chunk_size', 25000)  # Optimized chunk size for Omen 35L
    print(f"🚀 Using optimized chunk size: {chunksize} for your Omen 35L")
    
    # Create an empty list to collect processed chunks
    chunks = []
    chunk_count = 0
    
    try:
        for chunk in pd.read_csv(csv_path, 
                                chunksize=chunksize, 
                                dtype={'inheritedRunners': 'float64', 
                                      'inheritedRunnersScored': 'float64', 
                                      'catchersInterference': 'int64', 
                                      'salary': 'int64'},
                                low_memory=False):
            chunk_count += 1
            print(f"Processing chunk {chunk_count} ({len(chunk)} rows)...")
            
            # Basic preprocessing for each chunk
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk.fillna(0, inplace=True)
            
            chunks.append(chunk)
            
            # Optional: Limit total chunks for testing (remove this line for full dataset)
            # if chunk_count >= 10:  # Process only first 10 chunks for testing
            #     break
        
        print(f"Concatenating {len(chunks)} chunks...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"Dataset loaded successfully! Total rows: {len(df)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying with smaller chunk size...")
        
        # Fallback with smaller chunks
        chunks = []
        chunksize = 10000  # Even smaller chunks
        
        for chunk in pd.read_csv(csv_path, 
                                chunksize=chunksize, 
                                dtype={'inheritedRunners': 'float64', 
                                      'inheritedRunnersScored': 'float64', 
                                      'catchersInterference': 'int64', 
                                      'salary': 'int64'},
                                low_memory=False):
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk.fillna(0, inplace=True)
            chunks.append(chunk)
            
        df = pd.concat(chunks, ignore_index=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(by=['Name', 'date'], inplace=True)

    # Calculate calculated_dk_fpts if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("calculated_dk_fpts column not found. Calculating now...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")

    # Load or create LabelEncoders
    le_name, le_team = load_or_create_label_encoders(df)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])

    # --- New Financial-Style Feature Engineering Step ---
    print("Starting financial-style feature engineering...")
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    print("Financial-style feature engineering complete.")
    # --- End of New Step ---

    chunksize = 20000
    df = concurrent_feature_engineering(df, chunksize)

    # --- Centralized Data Cleaning ---
    print("Cleaning final dataset of any infinite or NaN values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # --- End of Cleaning Step ---

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Define the list of all selected and engineered features
    features = selected_features + ['date']

    # Define numeric and categorical features
    numeric_features = [
        'wOBA', 'BABIP', 'ISO',  'wRAA', 'wRC', 'wRC+', 'flyBalls', 'year', 
        'month', 'day',
        'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
        'rolling_mean_fpts_49', 
        'wOBA_Statcast',
        'SLG_Statcast', 'RAR_Statcast', 'Offense_Statcast', 'Dollars_Statcast',
        'WPA/LI_Statcast', 'Off', 'WAR', 'Dol', 'RAR',    
        'RE24', 'REW', 'SLG', 'WPA/LI','AB'
    ]

    categorical_features = ['Name', 'Team']

    # Debug prints to check feature lists and data types
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Data types in DataFrame:")
    print(df.dtypes)

    # Load or create Scaler
    scaler = load_or_create_scaler(df, numeric_features)

    # Define transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor that includes both numeric and categorical transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Before fitting the preprocessor
    print("Preparing features for preprocessing...")
    
    # Ensure all engineered features are created before selecting them
    features = df[numeric_features + categorical_features]

    # Debug print to check data types in features DataFrame
    print("Data types in features DataFrame before preprocessing:")
    print(features.dtypes)

    # The main dataframe `df` is already cleaned, so no need to clean the `features` slice again.

    # Fit the preprocessor
    print("Fitting preprocessor...")
    preprocessed_features = preprocessor.fit_transform(features)
    n_features = preprocessed_features.shape[1]

    # Feature selection based on the actual number of features
    k = min(550, n_features)  # Select the minimum of 550 or the actual number of features

    selector = SelectKBest(f_regression, k=k)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    # Time series split removed - training on all data
    print("Training single model on all data...")

    # It's important to drop the target from the features AFTER all engineering is complete
    if 'calculated_dk_fpts' in df.columns:
        features = df.drop(columns=['calculated_dk_fpts'])
        target = df['calculated_dk_fpts']
    else:
        # Fallback or error if the target column is still missing
        raise KeyError("'calculated_dk_fpts' not found in DataFrame columns after all processing.")        
    date_series = df['date']
    
    # Clean the data
    features = clean_infinite_values(features.copy())
    
    # Prepare preprocessor
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
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit preprocessor and transform features
    print("Fitting preprocessor and transforming features...")
    features_preprocessed = preprocessor.fit_transform(features)
    
    # Feature selection
    print("Performing feature selection...")
    selector = SelectKBest(f_regression, k=min(550, features_preprocessed.shape[1]))
    features_selected = selector.fit_transform(features_preprocessed, target)
    
    print(f"Selected {features_selected.shape[1]} features out of {features_preprocessed.shape[1]}")
    
    # Train the final model with hyperparameter tuning
    print("Training final ensemble model with hyperparameter tuning...")
    
    training_start_time = time.time()
    success = False
    
    try:
        print("Step 1: Training stacking model with hyperparameter tuning...")
        print("This may take several minutes...")
        
        # First, train the hyperparameter-tuned stacking model
        tuned_stacking_model = create_hypertuned_model(features_selected, target, cv_folds=3)
        
        print("Step 2: Training final nested stacking model...")
        # Update final model with the tuned stacking model
        final_model.estimators = [('stacking', tuned_stacking_model)]
        
        # Train the final nested model
        final_model.fit(features_selected, target)
        
        elapsed = time.time() - training_start_time
        print(f"✅ Model training completed successfully in {elapsed:.1f} seconds!")
        
        # Display hyperparameter tuning results if available
        try:
            if hasattr(tuned_stacking_model, 'best_params_'):
                print("\nHyperparameter Tuning Results:")
                print("=" * 60)
                print("Best Parameters:")
                for param, value in tuned_stacking_model.best_params_.items():
                    print(f"  {param}: {value}")
                print(f"Best Cross-Validation Score: {-tuned_stacking_model.best_score_:.4f}")
                print("=" * 60)
        except Exception as e:
            print(f"Could not display hyperparameter results: {e}")
        
        success = True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        print("🔄 Falling back to simpler stacking model without hyperparameter tuning...")
        
        try:
            # Fallback to basic stacking model
            print("Training basic stacking model...")
            final_model = stacking_model
            final_model.fit(features_selected, target)
            print("✅ Basic stacking model training completed!")
            success = True
            
        except Exception as e2:
            print(f"❌ Stacking model fallback failed: {e2}")
            success = False
    
    # Fallback models if training fails
    if not success:
        print("🔄 Falling back to single XGBoost model...")
        
        try:
            # Fast XGBoost fallback
            fallback_model = XGBRegressor(
                n_estimators=30,
                max_depth=4,
                learning_rate=0.15,
                tree_method='hist',
                device=device,
                objective='reg:squarederror',
                random_state=42,
                n_jobs=2  # Limit parallelism to avoid hanging
            )
            final_model = fallback_model
            final_model.fit(features_selected, target)
            print("✅ XGBoost fallback model training completed!")
            
        except Exception as e2:
            print(f"❌ XGBoost fallback failed: {e2}")
            print("🔄 Using Random Forest as final fallback...")
            
            from sklearn.ensemble import RandomForestRegressor
            fallback_model = RandomForestRegressor(
                n_estimators=50, 
                max_depth=5, 
                random_state=42, 
                n_jobs=2  # Limit parallelism
            )
            final_model = fallback_model
            final_model.fit(features_selected, target)
            print("✅ Random Forest fallback completed!")
    
    # Initialize and train probability predictor
    print("Training probability prediction models...")
    probability_predictor = ProbabilityPredictor()
    probability_predictor.fit_quantile_models(features, target, preprocessor, selector)
      # Make predictions on the entire dataset for evaluation
    print("Making predictions on training data...")
    all_predictions = final_model.predict(features_selected)
    
    # Apply constraints for realistic MLB fantasy points (0 to 100 range)
    print("Applying realistic constraints to predictions...")
    all_predictions = np.clip(all_predictions, 0, 100)  # Clip to 0-100 range
    print(f"Prediction range after clipping: {all_predictions.min():.2f} to {all_predictions.max():.2f}")
    
    # Estimate distribution parameters from training data
    probability_predictor.estimate_distribution_params(target, all_predictions)

    # Evaluate the model on training data (for reference)
    mae, mse, r2, mape = evaluate_model(target, all_predictions)
    
    print(f'Training MAE: {mae:.4f}')
    print(f'Training MSE: {mse:.4f}')
    print(f'Training R2: {r2:.4f}')
    print(f'Training MAPE: {mape:.4f}%')

    # Create a complete pipeline that includes preprocessing and feature selection
    complete_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', final_model)
    ])

    # Create enhanced predictions with probabilities
    print("Creating enhanced predictions with probability estimates...")
    enhanced_predictions, probability_results, prob_summary = create_enhanced_predictions_with_probabilities(
        complete_pipeline, probability_predictor, features, target, features['Name']
    )

    # Create a DataFrame with all predictions, actual values, names, and dates
    final_results_df = pd.DataFrame({
        'Name': features['Name'],
        'Date': date_series,
        'Actual': target,
        'Predicted': all_predictions
    })

    # Save the final results
    final_results_df.to_csv('2_PREDICTIONS/final_predictions.csv', index=False)
    print("Final predictions saved.")
    
    # Save probability predictions
    prob_summary.to_csv('2_PREDICTIONS/probability_predictions.csv', index=False)
    print("Probability predictions saved.")
      # Display sample probability predictions
    print("\nSample DraftKings MLB Probability Predictions:")
    print("="*80)
    sample_df = prob_summary.head(10)
    for _, row in sample_df.iterrows():
        print(f"\nPlayer: {row['Name']}")
        print(f"Predicted DraftKings Points: {row['Predicted_Points']:.1f}")
        print("Probability of exceeding thresholds:")
        for threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            prob_col = f'Prob_Over_{threshold}'
            if prob_col in row:
                print(f"  > {threshold} points: {row[prob_col]}")
    print("="*80)

    # Save the complete pipeline and probability predictor
    joblib.dump(joblib.dump(complete_pipeline, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl')
    joblib.dump(joblib.dump(probability_predictor, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/probability_predictor01.pkl')
    print("Final model pipeline and probability predictor saved.")

    # Save the final data to a CSV file
    df.to_csv('7_ANALYSIS/battersfinal_dataset_with_features.csv', index=False)
    print("Final dataset with all features saved.")

    # Save the LabelEncoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("LabelEncoders saved.")

    # Save feature importance with the updated pipeline structure
    save_feature_importance(complete_pipeline, '7_ANALYSIS/feature_importances.csv', 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_importances_plot.png')

    # Save the trained model immediately after training
    print("Saving trained model...")
    try:
        model_save_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/trained_model.joblib'
        joblib.dump(final_model, model_save_path)
        print(f"✅ Model saved to {model_save_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save model: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total script execution time: {total_time:.2f} seconds.")

def display_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Display comprehensive model performance metrics
    """
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 60)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    
    # Display prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"Min Prediction: {predictions.min():.2f}")
    print(f"Max Prediction: {predictions.max():.2f}")
    print(f"Mean Prediction: {predictions.mean():.2f}")
    print(f"Std Prediction: {predictions.std():.2f}")
    
    print("=" * 60)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': predictions
    }

def display_hyperparameter_results(search_model, model_name="Hyperparameter Search"):
    """
    Display hyperparameter search results
    """
    print(f"\n{model_name} Results:")
    print("=" * 60)
    
    if hasattr(search_model, 'best_params_'):
        print("Best Parameters:")
        for param, value in search_model.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Cross-Validation Score: {-search_model.best_score_:.4f}")
        
        # Display top 5 parameter combinations
        if hasattr(search_model, 'cv_results_'):
            results_df = pd.DataFrame(search_model.cv_results_)
            top_5 = results_df.nlargest(5, 'mean_test_score')
            
            print("\nTop 5 Parameter Combinations:")
            for i, (idx, row) in enumerate(top_5.iterrows()):
                print(f"{i+1}. Score: {-row['mean_test_score']:.4f}")
                params = {k: v for k, v in row.items() if k.startswith('param_')}
                for param, value in params.items():
                    print(f"   {param.replace('param_', '')}: {value}")
                print()
    else:
        print("No hyperparameter search results available")
    
    print("=" * 60)