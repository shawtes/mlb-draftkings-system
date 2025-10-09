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
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, VotingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, classification_report, accuracy_score
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
import os
import torch
from scipy import stats
from sklearn.linear_model import QuantileRegressor

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

# Ensure GPU is utilized for XGBoost
if torch.cuda.is_available():
    xgb.set_config(verbosity=1, use_rmm=True)  # Enable RAPIDS Memory Manager for GPU acceleration

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

# Increase chunk size for larger memory utilization
CHUNK_SIZE = 100000  # Adjusted for 64 GB RAM

# Maximize parallel workers for concurrent tasks
MAX_WORKERS = multiprocessing.cpu_count()  # Utilize all available CPU cores

# Update concurrent feature engineering to use larger chunks and more workers
def concurrent_feature_engineering(df, chunksize=CHUNK_SIZE):
    print("Starting concurrent feature engineering...")
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    start_time = time.time()

    if torch.cuda.is_available():
        print("GPU detected - using sequential processing for feature engineering")
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_date_series = date_series[i*chunksize:(i+1)*chunksize]
            processed_chunk = process_chunk(chunk, chunk_date_series)
            processed_chunks.append(processed_chunk)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            processed_chunks = list(executor.map(process_chunk, chunks, [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Concurrent feature engineering completed in {total_time:.2f} seconds.")
    return pd.concat(processed_chunks)

# Optimize data types to reduce memory usage
def optimize_data_types(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

# Apply data type optimization after loading data
def load_and_optimize_data(file_path):
    df = pd.read_csv(file_path)
    df = optimize_data_types(df)
    return df

# Missing function definitions that are referenced in the code
def load_or_create_label_encoders(df):
    # Define paths for saving and loading LabelEncoders
    name_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_name.pkl'
    team_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_team.pkl'

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
            # Test if the scaler works with current version
            le_team.fit(df['Team'])
        else:
            raise FileNotFoundError("Team encoder not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new team encoder due to compatibility issues...")
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    return le_name, le_team

def load_or_create_scaler(df, numeric_features):
    """Load or create a scaler for numeric features"""
    scaler_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/scaler.pkl'
    
    try:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            # Test if the scaler works with current version
            scaler.fit(df[numeric_features])
        else:
            raise FileNotFoundError("Scaler not found")
    except (FileNotFoundError, Exception) as e:
        print("Creating new scaler due to compatibility issues...")
        scaler = StandardScaler()
        scaler.fit(df[numeric_features])
        joblib.dump(scaler, scaler_path)
    
    return scaler

def clean_infinite_values(df):
    """Clean infinite values from DataFrame"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mse, r2, mape

class ProbabilityPredictor:
    """Simple probability predictor for DraftKings fantasy points"""
    
    def __init__(self):
        self.quantile_models = {}
        self.mean_pred = None
        self.std_pred = None
    
    def fit_quantile_models(self, features, target, preprocessor, selector):
        """Fit quantile regression models"""
        print("Training quantile models...")
        
        # Simple approach: use the target statistics to estimate probabilities
        self.mean_pred = np.mean(target)
        self.std_pred = np.std(target)
        
        # Create simple quantile models
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            self.quantile_models[q] = np.quantile(target, q)
    
    def estimate_distribution_params(self, target, predictions):
        """Estimate distribution parameters"""
        self.mean_pred = np.mean(predictions)
        self.std_pred = np.std(predictions)
    
    def predict_probabilities(self, predictions, thresholds):
        """Predict probabilities for given thresholds"""
        probabilities = {}
        
        for threshold in thresholds:
            # Simple normal distribution assumption
            prob = 1 - stats.norm.cdf(threshold, predictions, self.std_pred)
            probabilities[f'Prob_Over_{threshold}'] = prob
        
        return probabilities

def create_enhanced_predictions_with_probabilities(complete_pipeline, probability_predictor, features, target, player_names):
    """Create enhanced predictions with probabilities"""
    
    # Make predictions
    predictions = complete_pipeline.predict(features)
    
    # Calculate probabilities for different thresholds
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40]
    prob_dict = probability_predictor.predict_probabilities(predictions, thresholds)
    
    # Create enhanced predictions DataFrame
    enhanced_predictions = pd.DataFrame({
        'Name': player_names,
        'Predicted_Points': predictions,
        'Actual_Points': target
    })
    
    # Add probability columns
    for threshold in thresholds:
        prob_col = f'Prob_Over_{threshold}'
        if prob_col in prob_dict:
            enhanced_predictions[prob_col] = prob_dict[prob_col]
    
    # Create probability results
    probability_results = prob_dict
    
    # Create probability summary (same as enhanced_predictions for simplicity)
    prob_summary = enhanced_predictions.copy()
    
    return enhanced_predictions, probability_results, prob_summary

def save_feature_importance(pipeline, csv_path, plot_path):
    """Save feature importance from the pipeline"""
    try:
        # Get the model from the pipeline
        model = pipeline.named_steps['model']
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("Model does not have feature importance attributes")
            return
        
        # Create feature importance DataFrame
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance saved to {csv_path}")
        
        # Create and save plot
        plt.figure(figsize=(10, 6))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")
        
    except Exception as e:
        print(f"Error saving feature importance: {e}")

# Define global variables used in the main execution
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# Define base models for ensemble (simplified to avoid XGBoost compatibility issues)
base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('lasso', Lasso(alpha=0.1)),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=2))
]

# Define final ensemble model
final_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=3,  # Reduced CV folds for faster training
    n_jobs=2  # Limited parallelism to avoid issues
)

# Define encoder paths
name_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_name.pkl'
team_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/label_encoder_team.pkl'

# --- Main script execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    print("🚀 Starting MLB Training Pipeline...")
    print(f"⏰ Pipeline started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- Q1: Data Loading ---
    print("\nQ1: Loading large dataset in chunks to avoid memory issues...")
    # Step 1: Data Loading
    step_start = time.time()
    csv_path = r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv'
    chunksize = 25000  # Process 25k rows at a time to manage memory
    
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
    
    # Timer for Step 1
    step_elapsed = time.time() - step_start
    print(f"✅ Step 1 completed in {step_elapsed:.1f} seconds")
    
    # --- Q2: Label Encoding ---
    print("\nQ2: Creating label encoders...")
    # Step 2: Label Encoding
    step_start = time.time()
    
    # Load or create LabelEncoders
    le_name, le_team = load_or_create_label_encoders(df)

    # Ensure 'Name_encoded' and 'Team_encoded' columns are created
    df['Name_encoded'] = le_name.transform(df['Name'])
    df['Team_encoded'] = le_team.transform(df['Team'])
    
    # Timer for Step 2
    step_elapsed = time.time() - step_start
    print(f"✅ Step 2 completed in {step_elapsed:.1f} seconds")

    # --- Q3: Financial-Style Feature Engineering ---
    print("\nQ3: Starting financial-style feature engineering...")
    # Step 3: Financial-Style Feature Engineering
    step_start = time.time()
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    print("Financial-style feature engineering complete.")
    
    # Timer for Step 3
    step_elapsed = time.time() - step_start
    print(f"✅ Step 3 completed in {step_elapsed:.1f} seconds")

    # Step 4: Concurrent Feature Engineering
    step_start = time.time()
    print("\n⚡ Step 4: Starting concurrent feature engineering...")
    
    chunksize = 20000
    df = concurrent_feature_engineering(df, chunksize)
    
    # Timer for Step 4
    step_elapsed = time.time() - step_start
    print(f"✅ Step 4 completed in {step_elapsed:.1f} seconds")

    # Step 5: Data Cleaning
    step_start = time.time()
    print("\n🧹 Step 5: Cleaning final dataset of any infinite or NaN values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)
    
    # Timer for Step 5
    step_elapsed = time.time() - step_start
    print(f"✅ Step 5 completed in {step_elapsed:.1f} seconds")

    # Step 6: Feature Preparation
    step_start = time.time()
    print("\n🔧 Step 6: Preparing features for preprocessing...")
    
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
    k = min(808, n_features)  # Select the minimum of 550 or the actual number of features

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
    
    # Timer for Step 6
    step_elapsed = time.time() - step_start
    print(f"✅ Step 6 completed in {step_elapsed:.1f} seconds")

    # Step 7: Model Training
    step_start = time.time()
    print("\n🤖 Step 7: Training final ensemble model...")
    
    training_start_time = time.time()
    success = False
    
    try:
        print(f"Training {type(final_model).__name__} with {len(base_models)} base models...")
        print("This may take several minutes...")
        
        # Train the model
        final_model.fit(features_selected, target)
        
        elapsed = time.time() - training_start_time
        print(f"✅ Model training completed successfully in {elapsed:.1f} seconds!")
        success = True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
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
    
    # Timer for Step 7
    step_elapsed = time.time() - step_start
    print(f"✅ Step 7 completed in {step_elapsed:.1f} seconds")

    # Step 8: Probability Prediction Training
    step_start = time.time()
    print("\n📊 Step 8: Training probability prediction models...")
    
    probability_predictor = ProbabilityPredictor()
    probability_predictor.fit_quantile_models(features, target, preprocessor, selector)
    
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
    
    # Timer for Step 8
    step_elapsed = time.time() - step_start
    print(f"✅ Step 8 completed in {step_elapsed:.1f} seconds")

    # Step 9: Final Predictions and Saving
    step_start = time.time()
    print("\n💾 Step 9: Making final predictions and saving results...")
    
    # Make predictions on the entire dataset
    print("Making final predictions on the entire dataset...")
    df['final_predictions'] = complete_pipeline.predict(features)
    
    # Save final predictions to CSV
    df[['Name', 'date', 'final_predictions']].to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS/final_predictions.csv', index=False)
    print("Final predictions saved.")
    
    # --- Probability Predictions ---
    print("Calculating probability predictions...")
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40]
    prob_predictions = pd.DataFrame()

    for threshold in thresholds:
        prob_col = f'Prob_Over_{threshold}'
        prob_predictions[prob_col] = probability_predictor.predict_probabilities(df['final_predictions'], [threshold])[prob_col]

    # Combine probability predictions with final predictions
    final_results = pd.concat([df[['Name', 'date', 'final_predictions']], prob_predictions], axis=1)

    # Save probability predictions to CSV
    final_results.to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/2_PREDICTIONS/probability_predictions.csv', index=False)
    print("Probability predictions saved.")

    # --- Feature Importance ---
    print("Calculating feature importance...")
    try:
        # Get the model from the pipeline
        model = complete_pipeline.named_steps['model']
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("Model does not have feature importance attributes")
            importances = None
        
        if importances is not None:
            # Create feature importance DataFrame
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save to CSV
            importance_df.to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_importances.csv', index=False)
            print("Feature importance saved.")
            
            # Create and save plot
            plt.figure(figsize=(10, 6))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/feature_importances_plot.png')
            plt.close()
            print("Feature importance plot saved.")
        
    except Exception as e:
        print(f"Error calculating feature importance: {e}")

    # Save the complete pipeline and probability predictor
    joblib.dump(complete_pipeline, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl')
    joblib.dump(probability_predictor, 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/probability_predictor01.pkl')
    print("Final model pipeline and probability predictor saved.")

    # Save the final data to a CSV
    df.to_csv('c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/battersfinal_dataset_with_features.csv', index=False)
    print("Final dataset with all features saved.")

    # Save the LabelEncoders
    joblib.dump(le_name, name_encoder_path)
    joblib.dump(le_team, team_encoder_path)
    print("LabelEncoders saved.")
    
    # Timer for Step 9
    step_elapsed = time.time() - step_start
    print(f"✅ Step 9 completed in {step_elapsed:.1f} seconds")

    # Final Pipeline Summary
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"🏁 Pipeline ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📈 Final model R² score: {r2:.4f}")
    print(f"🎯 Model predictions saved to: 2_PREDICTIONS/")
    print(f"📊 Analysis results saved to: 7_ANALYSIS/")
    print(f"🤖 Trained models saved to: 3_MODELS/")