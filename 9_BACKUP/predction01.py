import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
import warnings
import logging
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, QuantileRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

# Suppress specific pandas warnings related to runtime operations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Also suppress XGBoost device warnings for cleaner output
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')
# Suppress sklearn version warnings for pickled objects
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

# Reintroduce necessary definitions for league_avg_wOBA, league_avg_HR_FlyBall, and wOBA_weights
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

wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

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
        # Final cleanup
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df

def clean_infinite_values(df):
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, replace NaN with the mean of the column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # For non-numeric columns, replace NaN with a placeholder value (e.g., 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def calculate_dk_fpts(row):
    return (row['1B'] * 3 + row['2B'] * 5 + row['3B'] * 8 + row['HR'] * 10 +
            row['RBI'] * 2 + row['R'] * 2 + row['BB'] * 2 + row['HBP'] * 2 + row['SB'] * 5)

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
    df['week_of_season'] = (date_series - date_series.min()).dt.days // 7
    df['day_of_year'] = date_series.dt.dayofyear

    # Calculate key statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics
    logging.info(f"Year range in data: {df['year'].min()} to {df['year'].max()}")
    
    def safe_wRAA(row):
        year = row['year']
        if year not in league_avg_wOBA:
            logging.warning(f"Year {year} not found in league_avg_wOBA. Using {max(league_avg_wOBA.keys())} instead.")
            year = max(league_avg_wOBA.keys())
        return ((row['wOBA'] - league_avg_wOBA[year]) / 1.15) * row['AB'] if row['AB'] > 0 else 0

    df['wRAA'] = df.apply(safe_wRAA, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA.get(x['year'], league_avg_wOBA[2020]) * 100) if x['AB'] > 0 and league_avg_wOBA.get(x['year'], league_avg_wOBA[2020]) > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall.get(x['year'], league_avg_HR_FlyBall[2020]) if league_avg_HR_FlyBall.get(x['year'], league_avg_HR_FlyBall[2020]) > 0 else 0, axis=1)

    # Calculate singles
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    # Calculate wOBA using year-specific weights
    def safe_wOBA_Statcast(row):
        year = row['year']
        if year not in wOBA_weights:
            logging.warning(f"Year {year} not found in wOBA_weights. Using {max(wOBA_weights.keys())} instead.")
            year = max(wOBA_weights.keys())
        weights = wOBA_weights[year]
        numerator = (
            weights['BB'] * row['BB'] +
            weights['HBP'] * row['HBP'] +
            weights['1B'] * row['1B'] +
            weights['2B'] * row['2B'] +
            weights['3B'] * row['3B'] +
            weights['HR'] * row['HR']
        )
        denominator = row['AB'] + row['BB'] - row['IBB'] + row['SF'] + row['HBP']
        return numerator / denominator if denominator > 0 else 0

    df['wOBA_Statcast'] = df.apply(safe_wOBA_Statcast, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = df.apply(lambda x: (
        x['1B'] + (2 * x['2B']) + (3 * x['3B']) + (4 * x['HR'])
    ) / x['AB'] if x['AB'] > 0 else 0, axis=1)

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
        for window in [7, 10, 49]:  # Added window 10 for constraint purposes
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Remove 5-game average calculation - not in training script
    # df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Player consistency features (only add if they exist in training)
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    return df

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    
    # Apply financial-style features first
    print("Applying financial-style feature engineering...")
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    
    # Then use parallel processing for traditional features
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        processed_chunks = list(executor.map(engineer_features, chunks, 
                                           [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    
    result_df = pd.concat(processed_chunks)
    
    # Final cleanup of infinite values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.fillna(0)
    
    print("Concurrent feature engineering completed.")
    return result_df

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    
    # Calculate league averages for realistic defaults
    league_averages = df.select_dtypes(include=[np.number]).mean()
    league_std = df.select_dtypes(include=[np.number]).std()
    
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using league average values.")
            # Create default row with league averages instead of random values
            default_row = pd.DataFrame([league_averages]).copy()
            default_row['date'] = prediction_date
            default_row['Name'] = player
            # Use conservative estimate for unknown players
            default_row['calculated_dk_fpts'] = max(2.0, league_averages.get('calculated_dk_fpts', 5.0))
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use recent data for player averages (up to 5 most recent games)
            player_df = player_df.head(5)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and realistic
            if 'calculated_dk_fpts' in player_df.columns:
                dk_fpts_avg = player_df['calculated_dk_fpts'].mean()
                # Apply some variance but keep reasonable
                synthetic_row['calculated_dk_fpts'] = max(0, min(dk_fpts_avg, 25))  # Cap at 25 for baseline
            else:
                synthetic_row['calculated_dk_fpts'] = 5.0  # Conservative default
            
            # Handle categorical columns
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    
    # Final validation and cleanup
    synthetic_df = validate_synthetic_data(synthetic_df)
    
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    print(f"Synthetic data stats - calculated_dk_fpts range: {synthetic_df['calculated_dk_fpts'].min():.2f} to {synthetic_df['calculated_dk_fpts'].max():.2f}")
    
    return synthetic_df

def validate_synthetic_data(df):
    """Validate and clean synthetic data to ensure realistic values"""
    # Cap extreme values in key statistics
    df['calculated_dk_fpts'] = np.clip(df['calculated_dk_fpts'], 0, 30)
    
    # Remove 5_game_avg reference - not in training script
    # if '5_game_avg' in df.columns:
    #     df['5_game_avg'] = np.clip(df['5_game_avg'], 0, 30)
    
    # Ensure batting stats are within realistic ranges
    if 'AVG' in df.columns:
        df['AVG'] = np.clip(df['AVG'], 0, 0.500)
    if 'OBP' in df.columns:
        df['OBP'] = np.clip(df['OBP'], 0, 0.600)
    if 'SLG' in df.columns:
        df['SLG'] = np.clip(df['SLG'], 0, 1.000)
    
    # Clean infinite and NaN values
    df = clean_infinite_values(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def process_predictions(chunk, pipeline, player_adjustments):
    # Prepare features exactly as in training
    features = chunk.drop(columns=['calculated_dk_fpts'])

    # Clean the features to ensure no infinite or excessively large values
    features = clean_infinite_values(features)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    # Use the complete pipeline for prediction (this handles preprocessing and feature selection)
    raw_predictions = pipeline.predict(features)

    # Debug: Print raw prediction statistics
    print(f"Raw predictions - Min: {raw_predictions.min():.2f}, Max: {raw_predictions.max():.2f}, Mean: {raw_predictions.mean():.2f}")

    # Apply smart outlier handling instead of hard clipping
    chunk.loc[:, 'predicted_dk_fpts'] = apply_smart_prediction_constraints(raw_predictions, chunk)
    chunk.loc[:, 'predicted_dk_fpts'] = chunk.apply(lambda row: adjust_predictions(row, player_adjustments), axis=1)

    # Remove outliers with predictions > 35
    outlier_count = chunk[chunk['predicted_dk_fpts'] > 35].shape[0]
    if outlier_count > 0:
        print(f"Removing {outlier_count} outliers with predictions > 35.")
        chunk = chunk[chunk['predicted_dk_fpts'] <= 35]

    # Debug: Print statistics after outlier removal
    print(f"After outlier removal: {chunk.shape[0]} players")

    return chunk

def apply_smart_prediction_constraints(raw_predictions, chunk):
    """Apply strict constraints based on player's 10-day rolling range"""
    constrained_predictions = raw_predictions.copy()
    
    # Calculate realistic bounds based on player's 10-day rolling range
    for i, (idx, row) in enumerate(chunk.iterrows()):
        raw_pred = raw_predictions[i]
        
        # Get 10-day rolling min and max if available
        rolling_min_10 = row.get('rolling_min_fpts_10', None)
        rolling_max_10 = row.get('rolling_max_fpts_10', None)
        
        # If we have 10-day rolling data, use it as primary constraint
        if rolling_min_10 is not None and rolling_max_10 is not None and not (np.isnan(rolling_min_10) or np.isnan(rolling_max_10)):
            # Use 10-day range with minimal expansion (10% or 2 points max)
            range_expansion = min(2.0, (rolling_max_10 - rolling_min_10) * 0.1)
            lower_bound = max(0, rolling_min_10 - range_expansion)
            upper_bound = rolling_max_10 + range_expansion
            
            # Ensure minimum reasonable range of 3 points
            current_range = upper_bound - lower_bound
            if current_range < 3.0:
                center = (upper_bound + lower_bound) / 2
                lower_bound = max(0, center - 1.5)
                upper_bound = center + 1.5
            
            # Hard constraint within this range
            constrained_predictions[i] = np.clip(raw_pred, lower_bound, upper_bound)
            
        else:
            # For players without 10-day data, use 7-day range if available
            rolling_min_7 = row.get('rolling_min_fpts_7', None)
            rolling_max_7 = row.get('rolling_max_fpts_7', None)
            
            if rolling_min_7 is not None and rolling_max_7 is not None and not (np.isnan(rolling_min_7) or np.isnan(rolling_max_7)):
                # Use 7-day range with slightly more expansion
                range_expansion = min(3.0, (rolling_max_7 - rolling_min_7) * 0.2)
                lower_bound = max(0, rolling_min_7 - range_expansion)
                upper_bound = rolling_max_7 + range_expansion
                
                # Ensure minimum reasonable range of 4 points
                current_range = upper_bound - lower_bound
                if current_range < 4.0:
                    center = (upper_bound + lower_bound) / 2
                    lower_bound = max(0, center - 2.0)
                    upper_bound = center + 2.0
                
                constrained_predictions[i] = np.clip(raw_pred, lower_bound, upper_bound)
            else:
                # Fallback for players with no recent history - use conservative league-wide ranges
                # Use position-based reasonable ranges
                position = row.get('Position', 'OF')  # Default to OF
                if position in ['C']:
                    # Catchers tend to score less
                    constrained_predictions[i] = np.clip(raw_pred, 0, 12)
                elif position in ['1B', '3B', 'OF']:
                    # Power positions
                    constrained_predictions[i] = np.clip(raw_pred, 0, 18)
                elif position in ['SS', '2B']:
                    # Middle infield
                    constrained_predictions[i] = np.clip(raw_pred, 0, 15)
                else:
                    # Default fallback
                    constrained_predictions[i] = np.clip(raw_pred, 0, 15)
    
    # Final safety check - no prediction should exceed absolute maximum
    constrained_predictions = np.clip(constrained_predictions, 0, 35)
    
    print(f"Constrained predictions - Min: {constrained_predictions.min():.2f}, Max: {constrained_predictions.max():.2f}, Mean: {constrained_predictions.mean():.2f}")
    
    return constrained_predictions

def adjust_predictions(row, player_adjustments):
    """Adjust predictions based on player-specific average differences."""
    prediction = row['predicted_dk_fpts']
    player = row['Name']
    
    # Check if player_adjustments is empty or if player is not in adjustments
    if player_adjustments.empty or player not in player_adjustments.index:
        # No adjustments available, return original prediction
        return max(0, prediction)
    
    try:
        if prediction > row.get('calculated_dk_fpts', 0):
            adjustment = player_adjustments.loc[player, 'avg_positive_diff'] / 4  # Reduced adjustment factor
        else:
            adjustment = player_adjustments.loc[player, 'avg_negative_diff'] / 4  # Reduced adjustment factor
        
        adjusted_prediction = prediction - adjustment
    except (KeyError, TypeError):
        # If adjustment fails, use a small default adjustment
        if prediction > row.get('calculated_dk_fpts', 0):
            adjusted_prediction = prediction - 0.5
        else:
            adjusted_prediction = prediction + 0.5
    
    return max(0, adjusted_prediction)  # Ensure non-negative prediction

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, train_data['Name'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def predict_unseen_data(input_file, model_file, prediction_date):
    print("Loading dataset...")
    df = pd.read_csv(input_file,
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    # Debug: Check the first few date values and their format
    print(f"Sample date values from CSV:")
    print(df['date'].head(10).tolist())
    print(f"Date column data type: {df['date'].dtype}")
    
    # Try multiple approaches to parse dates
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            # Try with infer_datetime_format
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # Fallback: just use errors='coerce' to convert what we can
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for any failed conversions
    null_dates = df['date'].isnull().sum()
    if null_dates > 0:
        print(f"Warning: {null_dates} dates could not be parsed and were set to NaT")
        # Drop rows with null dates
        df = df.dropna(subset=['date'])
        print(f"Dropped rows with invalid dates. New shape: {df.shape}")
    
    prediction_date = pd.to_datetime(prediction_date)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range in dataset: {df['date'].min()} to {df['date'].max()}")
    print(f"Year range in dataset: {df['date'].dt.year.min()} to {df['date'].dt.year.max()}")
    print(f"Number of unique players: {df['Name'].nunique()}")
    
    # Get all unique players from the entire dataset
    all_players = df['Name'].unique()
      # No need to filter data up to the prediction date
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # Calculate calculated_dk_fpts if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DK Fantasy Points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")    # Load or create LabelEncoders - Updated paths to match training script
    name_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/label_encoder_name_sep2.pkl'
    team_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/label_encoder_team_sep2.pkl'
    scaler_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/scaler_sep2.pkl'    # Load or create LabelEncoders (handle version compatibility)
    try:
        if os.path.exists(name_encoder_path):
            le_name = joblib.load(name_encoder_path)
        else:
            le_name = LabelEncoder()
            le_name.fit(df['Name'])
            joblib.dump(le_name, name_encoder_path)
    except (FileNotFoundError, Exception) as e:
        print("Creating new name encoder due to compatibility issues...")
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)

    try:
        if os.path.exists(team_encoder_path):
            le_team = joblib.load(team_encoder_path)
        else:
            le_team = LabelEncoder()
            le_team.fit(df['Team'])
            joblib.dump(le_team, team_encoder_path)
    except (FileNotFoundError, Exception) as e:
        print("Creating new team encoder due to compatibility issues...")
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    # Update LabelEncoders with new players/teams - Fix this to match training approach
    # Instead of manually expanding encoder classes, use the training approach
    # that recreates encoders when needed
    
    # This ensures compatibility with the training pipeline
    try:
        # Test if encoders work with current data
        df['Name_encoded'] = le_name.transform(df['Name'])
        df['Team_encoded'] = le_team.transform(df['Team'])
    except ValueError as e:
        print(f"Encoder compatibility issue: {e}")
        print("Recreating encoders with current data...")
        # Recreate encoders with all data (training + new)
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)
        
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)
        
        # Now encode with new encoders
        df['Name_encoded'] = le_name.transform(df['Name'])
        df['Team_encoded'] = le_team.transform(df['Team'])    # Remove this scaler loading - the pipeline handles scaling internally
    # if os.path.exists(scaler_path):
    #     scaler = joblib.load(scaler_path)
    # else:
    #     raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # The pipeline includes all preprocessing steps

    chunksize = 20000  # Increased for better performance
    
    # --- New Financial-Style Feature Engineering Step ---
    print("Starting enhanced feature engineering with financial-style features...")
    df = concurrent_feature_engineering(df, chunksize)
    
    # --- Centralized Data Cleaning ---
    print("Cleaning final dataset of any infinite or NaN values...")
    df = clean_infinite_values(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print("Enhanced feature engineering complete.")

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Fix the categorical variable encoding to match training exactly
    # The training script uses 'Name' and 'Team' as categorical features
    # Remove these lines that override the proper encoding:
    # df['team_encoded'] = df['Team']
    # df['Name_encoded'] = df['Name']
    
    # The pipeline will handle the categorical encoding properly

    if df.empty:
        raise ValueError(f"No data available up to {prediction_date}")
    
    print(f"Data available up to {df['date'].max()}")

    print("Loading model...")
    pipeline = joblib.load(model_file)
    
    print("Model pipeline steps:")
    for step_name, step in pipeline.named_steps.items():
        print(f"- {step_name}: {type(step).__name__}")
    
    print(f"Processing date: {prediction_date}")
    
    # Create synthetic rows for all players for the prediction date
    current_df = create_synthetic_rows_for_all_players(df, all_players, prediction_date)
    
    if current_df.empty:
        print(f"No data available for date: {prediction_date}")
        return
      # Load player adjustments
    player_adjustments_path = '4_DATA/player_adjustments.csv'
    if os.path.exists(player_adjustments_path):
        player_adjustments = pd.read_csv(player_adjustments_path, index_col='Name')
    else:
        print("Player adjustments file not found. Using default adjustments.")
        player_adjustments = pd.DataFrame(columns=['avg_positive_diff', 'avg_negative_diff'])
      # Process predictions in chunks
    chunks = [current_df[i:i+chunksize] for i in range(0, current_df.shape[0], chunksize)]
    chunk_predictions = []
    all_features_for_prob = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_pred = process_predictions(chunk, pipeline, player_adjustments)
        chunk_predictions.append(chunk_pred)
        
        # Collect features for probability prediction
        features_for_prob = chunk.drop(columns=['calculated_dk_fpts'])
        features_for_prob = clean_infinite_values(features_for_prob)
        all_features_for_prob.append(features_for_prob)
    
    # Combine chunk predictions
    predictions = pd.concat(chunk_predictions)
    all_features = pd.concat(all_features_for_prob)
      # Create enhanced predictions with probabilities
    print("Creating enhanced DraftKings predictions with probability analysis...")
    enhanced_predictions, probability_results, prob_summary = create_enhanced_predictions_with_probabilities(
        pipeline, all_features, predictions['Name'], prediction_date
    )
    
    # Update predictions with enhanced values
    predictions['predicted_dk_fpts'] = enhanced_predictions
    
    print("Prediction statistics:")
    if 'predicted_dk_fpts' in predictions.columns:
        print(predictions['predicted_dk_fpts'].describe())
        print(f"Prediction range: {predictions['predicted_dk_fpts'].min():.2f} to {predictions['predicted_dk_fpts'].max():.2f}")
    else:
        print("Error: 'predicted_dk_fpts' column not found in predictions.")
        print("Available columns:", predictions.columns.tolist())    # Save main predictions
    output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/batters_predictions_{prediction_date.strftime("%Y%m%d")}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"Main predictions saved to {output_file}")
    
    # Save probability predictions
    prob_output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/batters_probability_predictions_{prediction_date.strftime("%Y%m%d")}.csv'
    prob_summary.to_csv(prob_output_file, index=False)
    print(f"Probability predictions saved to {prob_output_file}")
    
    # Display sample DraftKings probability predictions
    print("\nSample DraftKings MLB Probability Predictions:")
    print("="*80)
    sample_df = prob_summary.head(10)
    for _, row in sample_df.iterrows():
        print(f"\nPlayer: {row['Name']}")
        print(f"Predicted DraftKings Points: {row['Predicted_DK_Points']:.1f}")
        print("Probability of exceeding thresholds:")
        for threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            prob_col = f'Prob_Over_{threshold}'
            if prob_col in row:
                print(f"  > {threshold} points: {row[prob_col]}")
    print("="*80)
    
    # Print sample of main predictions
    print("\nSample main predictions:")
    print(predictions[['Name', 'predicted_dk_fpts', 'has_historical_data']].head(10))

    return predictions

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
        
    def predict_probabilities(self, X, main_predictions, preprocessor, selector):
        """
        Predict probabilities of achieving different point thresholds
        """
        results = []
          # Simple probability estimation based on prediction value and typical MLB variance
        typical_std = 8.0  # Typical standard deviation for MLB DraftKings points
        
        for i, main_pred in enumerate(main_predictions):
            player_probs = {'main_prediction': main_pred}
            
            for threshold in self.point_thresholds:
                if typical_std > 0:
                    # Probability of exceeding threshold using normal approximation
                    z_score = (threshold - main_pred) / typical_std
                    prob_exceed = 1 - stats.norm.cdf(z_score)
                    player_probs[f'prob_over_{threshold}'] = max(0, min(1, prob_exceed))
                else:
                    # If no variance, use deterministic approach
                    player_probs[f'prob_over_{threshold}'] = 1.0 if main_pred > threshold else 0.0
                    
            results.append(player_probs)
            
        return results
        
    def create_probability_summary(self, probability_results, player_names, prediction_date=None):
        """
        Create a summary DataFrame with probability predictions
        """
        summary_data = []
        
        for i, (name, probs) in enumerate(zip(player_names, probability_results)):
            row = {
                'Name': name, 
                'Date': prediction_date if prediction_date else pd.Timestamp.now().date(),
                'Predicted_DK_Points': probs['main_prediction']
            }
            
            # Add probability columns
            for threshold in self.point_thresholds:
                if f'prob_over_{threshold}' in probs:
                    row[f'Prob_Over_{threshold}'] = f"{probs[f'prob_over_{threshold}']:.1%}"
                    
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

def create_enhanced_predictions_with_probabilities(pipeline, features, player_names, prediction_date=None):
    """
    Create predictions with probability estimates - Updated to work with complete pipeline
    """
    print("Creating enhanced DraftKings predictions with probabilities...")
    
    # Get main predictions using the complete pipeline
    main_predictions = pipeline.predict(features)
    
    # Debug: Print raw prediction statistics
    print(f"Raw main predictions - Min: {main_predictions.min():.2f}, Max: {main_predictions.max():.2f}, Mean: {main_predictions.mean():.2f}")
    
    # Apply realistic constraints for MLB DraftKings fantasy points
    # Most MLB games result in 0-30 points, with exceptional games up to 45
    main_predictions = np.clip(main_predictions, 0, 45)
    
    # Additional outlier handling - if prediction is extremely high, apply log scaling
    for i in range(len(main_predictions)):
        if main_predictions[i] > 30:
            excess = main_predictions[i] - 30
            # Apply logarithmic scaling to reduce extreme values
            scaled_excess = np.log1p(excess) * 3
            main_predictions[i] = 30 + scaled_excess
    
    # Final safety constraint
    main_predictions = np.clip(main_predictions, 0, 40)
    
    print(f"Constrained main predictions - Min: {main_predictions.min():.2f}, Max: {main_predictions.max():.2f}, Mean: {main_predictions.mean():.2f}")
    
    # Get probability predictions - No need to extract preprocessor/selector separately
    # The pipeline handles everything internally
    probability_predictor = ProbabilityPredictor()
    probability_results = probability_predictor.predict_probabilities(
        features, main_predictions, None, None
    )
    
    # Create summary DataFrame with date
    prob_summary = probability_predictor.create_probability_summary(
        probability_results, player_names, prediction_date
    )
    
    return main_predictions, probability_results, prob_summary

if __name__ == "__main__":
    input_file = 'C:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_fangraphs_data.csv'
    model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    prediction_date = '2025-09-21'  # Example prediction date, can be changed as needed
    
    predict_unseen_data(input_file, model_file, prediction_date)