import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
import warnings
import logging
import os
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, QuantileRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
import warnings
from sklearn.model_selection import TimeSeriesSplit
import logging
import os
import torch
from scipy import stats
from sklearn.linear_model import QuantileRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # Final cleanup
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
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
    'wOBA_Statcast', 'SLG_Statcast', 'Off', 'WAR', 'Dol', 'RAR',     
    'RE24', 'REW', 'SLG', 'WPA/LI', 'AB', 'WAR'
]

engineered_features = [
    'wOBA_Statcast', 
    'SLG_Statcast', 'Offense_Statcast', 'RAR_Statcast', 'Dollars_Statcast', 
    'WPA/LI_Statcast', 'Name_encoded', 'team_encoded','wRC+', 'wRAA', 'wOBA',   
]
selected_features += engineered_features

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
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Calculate 5-game average
    df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Handle zero values in 5-game average
    df['5_game_avg'] = df['5_game_avg'].replace(0, np.nan).fillna(df['calculated_dk_fpts'].mean())
    
    # Debug: Print 5-game average calculation
    logging.info("5-game average calculation:")
    logging.info(df[['Name', 'date', 'calculated_dk_fpts', '5_game_avg']].head(10))
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Player consistency features
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    return df

def create_player_features_for_date(df, all_players, prediction_date):
    print(f"Creating player features for date: {prediction_date}...")
    synthetic_rows = []
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using randomized default values.")
            default_row = pd.DataFrame([{col: np.random.uniform(0, 1) for col in df.columns}])
            default_row['date'] = prediction_date
            default_row['Name'] = player
            default_row['calculated_dk_fpts'] = np.random.uniform(0, 5)  # Random value between 0 and 5
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use all available data, up to 20 most recent games
            player_df = player_df.head(20)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and calculated correctly
            if 'calculated_dk_fpts' in player_df.columns:
                synthetic_row['calculated_dk_fpts'] = player_df['calculated_dk_fpts'].mean()
            else:
                synthetic_row['calculated_dk_fpts'] = np.nan  # Placeholder, replace with actual calculation if needed
            
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    return synthetic_df

class WalkForwardPredictor:
    """
    Walk-forward prediction system to prevent data leakage and generate 
    predictions for a full year using a rolling window approach.
    """
    def __init__(self, train_window_days=365, prediction_horizon_days=1, min_train_samples=1000):
        self.train_window_days = train_window_days
        self.prediction_horizon_days = prediction_horizon_days
        self.min_train_samples = min_train_samples
        self.predictions_history = []
        self.model_performance = []
        
    def get_date_range_for_year(self, start_date, end_date):
        """Generate date range for walk-forward predictions"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate business days (excluding weekends when no MLB games typically occur)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to roughly match MLB season (April through October)
        mlb_season_dates = [
            d for d in date_range 
            if d.month >= 4 and d.month <= 10
        ]
        
        return mlb_season_dates
    
    def prepare_training_data(self, df, current_date):
        """
        Prepare training data up to current_date using rolling window
        """
        current_date = pd.to_datetime(current_date)
        
        # Define training window - only use data before current_date
        train_start = current_date - pd.Timedelta(days=self.train_window_days)
        train_end = current_date - pd.Timedelta(days=1)  # Don't include current day
        
        # Filter data for training window
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_data = df[train_mask].copy()
        
        print(f"Training data range: {train_start.date()} to {train_end.date()}")
        print(f"Training samples: {len(train_data)}")
        
        if len(train_data) < self.min_train_samples:
            print(f"Warning: Only {len(train_data)} training samples available (minimum: {self.min_train_samples})")
            return None
            
        return train_data
    
    def retrain_model(self, train_data, pipeline_template):
        """
        Retrain the model on the current training window
        """
        # Prepare features and target
        feature_cols = [col for col in train_data.columns if col not in ['calculated_dk_fpts', 'date', 'Name', 'Team']]
        X_train = train_data[feature_cols]
        y_train = train_data['calculated_dk_fpts']
        
        # Clean training data
        X_train = clean_infinite_values(X_train)
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train.fillna(0, inplace=True)
        
        # Clone the pipeline and retrain
        from sklearn.base import clone
        model_pipeline = clone(pipeline_template)
        
        try:
            # Fit the model
            model_pipeline.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred = model_pipeline.predict(X_train)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            print(f"Training MAE: {train_mae:.3f}, R2: {train_r2:.3f}")
            
            return model_pipeline
            
        except Exception as e:
            print(f"Error retraining model: {str(e)}")
            return None
    
    def make_prediction_for_date(self, df, current_date, model_pipeline):
        """
        Make predictions for a specific date
        """
        current_date = pd.to_datetime(current_date)
        
        # Get all unique players from historical data
        all_players = df['Name'].unique()
        
        # Create synthetic rows for prediction
        prediction_data = create_player_features_for_date(df, all_players, current_date)
        
        if prediction_data.empty:
            print(f"No prediction data available for {current_date.date()}")
            return None
            
        # Prepare features
        feature_cols = [col for col in prediction_data.columns if col not in ['calculated_dk_fpts', 'date', 'Name', 'Team']]
        X_pred = prediction_data[feature_cols]
        
        # Clean prediction data
        X_pred = clean_infinite_values(X_pred)
        X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_pred.fillna(0, inplace=True)
        
        try:
            # Make predictions
            predictions = model_pipeline.predict(X_pred)
            predictions = np.clip(predictions, 0, 100)  # Realistic DK points range
            
            # Create results dataframe
            results = pd.DataFrame({
                'Name': prediction_data['Name'],
                'Date': current_date.date(),
                'Predicted_DK_Points': predictions,
                'Has_Historical_Data': prediction_data.get('has_historical_data', True)
            })
            
            return results
            
        except Exception as e:
            print(f"Error making predictions for {current_date.date()}: {str(e)}")
            return None
    
    def run_walk_forward_prediction(self, df, pipeline_template, start_date, end_date, retrain_frequency=7):
        """
        Run walk-forward prediction for the entire date range
        """
        print(f"Starting walk-forward prediction from {start_date} to {end_date}")
        print(f"Training window: {self.train_window_days} days")
        print(f"Retraining frequency: {retrain_frequency} days")
        
        # Get prediction dates
        prediction_dates = self.get_date_range_for_year(start_date, end_date)
        print(f"Total prediction dates: {len(prediction_dates)}")
        
        all_predictions = []
        current_model = None
        last_train_date = None
        
        for i, current_date in enumerate(prediction_dates):
            print(f"\nProcessing date {i+1}/{len(prediction_dates)}: {current_date.date()}")
            
            # Check if we need to retrain
            should_retrain = (
                current_model is None or 
                last_train_date is None or 
                (current_date - last_train_date).days >= retrain_frequency
            )
            
            if should_retrain:
                print("Retraining model...")
                train_data = self.prepare_training_data(df, current_date)
                
                if train_data is not None:
                    current_model = self.retrain_model(train_data, pipeline_template)
                    last_train_date = current_date
                    
                    if current_model is None:
                        print("Failed to retrain model, skipping this date")
                        continue
                else:
                    print("Insufficient training data, skipping this date")
                    continue
            
            # Make prediction
            prediction_results = self.make_prediction_for_date(df, current_date, current_model)
            
            if prediction_results is not None:
                all_predictions.append(prediction_results)
                print(f"Generated {len(prediction_results)} predictions")
            else:
                print("Failed to generate predictions for this date")
        
        # Combine all predictions
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            print(f"\nWalk-forward prediction complete!")
            print(f"Total predictions generated: {len(final_predictions)}")
            print(f"Date range: {final_predictions['Date'].min()} to {final_predictions['Date'].max()}")
            print(f"Unique players: {final_predictions['Name'].nunique()}")
            
            return final_predictions
        else:
            print("No predictions generated!")
            return None

def run_walk_forward_prediction_pipeline(input_file, model_file, start_date, end_date, output_dir=None):
    """
    Complete walk-forward prediction pipeline
    """
    print("="*80)
    print("WALK-FORWARD PREDICTION PIPELINE")
    print("="*80)
    
    # Load and prepare data
    print("Loading dataset...")
    df = pd.read_csv(input_file,
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 
                           'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['Name', 'date'])
    
    print(f"Dataset loaded: {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique players: {df['Name'].nunique()}")
    
    # Calculate DK fantasy points if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DK Fantasy Points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
    
    # Basic preprocessing
    df = df.fillna(0)
    
    # Load model
    print("Loading model pipeline...")
    pipeline = joblib.load(model_file)
    
    # Create walk-forward predictor
    predictor = WalkForwardPredictor(
        train_window_days=365,
        min_train_samples=1000
    )
    
    # Run walk-forward prediction
    predictions = predictor.run_walk_forward_prediction(
        df=df,
        pipeline_template=pipeline,
        start_date=start_date,
        end_date=end_date,
        retrain_frequency=retrain_frequency,
        output_dir=output_dir
    )
    
    return predictions

def validate_walk_forward_predictions(predictions_df):
    """
    Validate the walk-forward predictions
    """
    print("\n=== Validating Walk-Forward Predictions ===")
    
    if predictions_df.empty:
        print("No predictions to validate")
        return
        
    # Basic statistics
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Unique players: {predictions_df['Name'].nunique()}")
    print(f"Date range: {predictions_df['prediction_date'].min()} to {predictions_df['prediction_date'].max()}")
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"Mean: {predictions_df['predicted_dk_fpts'].mean():.2f}")
    print(f"Std: {predictions_df['predicted_dk_fpts'].std():.2f}")
    print(f"Min: {predictions_df['predicted_dk_fpts'].min():.2f}")
    print(f"Max: {predictions_df['predicted_dk_fpts'].max():.2f}")
    
    # Check for realistic ranges
    unrealistic_high = (predictions_df['predicted_dk_fpts'] > 100).sum()
    unrealistic_low = (predictions_df['predicted_dk_fpts'] < 0).sum()
    
    print(f"\nData Quality Checks:")
    print(f"Predictions > 100: {unrealistic_high}")
    print(f"Predictions < 0: {unrealistic_low}")
    
    # Daily prediction counts
    daily_counts = predictions_df.groupby('prediction_date').size()
    print(f"\nDaily Prediction Counts:")
    print(f"Average predictions per day: {daily_counts.mean():.1f}")
    print(f"Min predictions per day: {daily_counts.min()}")
    print(f"Max predictions per day: {daily_counts.max()}")
    
    return True

def create_prediction_summary_report(predictions_df, output_dir=None):
    """
    Create a summary report of the walk-forward predictions
    """
    if output_dir is None:
        output_dir = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
        
    if predictions_df.empty:
        print("No predictions to summarize")
        return
        
    print("\n=== Creating Summary Report ===")
    
    # Player-level summary
    player_summary = predictions_df.groupby('Name').agg({
        'predicted_dk_fpts': ['count', 'mean', 'std', 'min', 'max'],
        'prediction_date': ['min', 'max']
    }).round(2)
    
    player_summary.columns = ['games_predicted', 'avg_predicted', 'std_predicted', 
                             'min_predicted', 'max_predicted', 'first_date', 'last_date']
    
    # Date-level summary
    date_summary = predictions_df.groupby('prediction_date').agg({
        'predicted_dk_fpts': ['count', 'mean', 'std', 'min', 'max'],
        'Name': 'nunique'
    }).round(2)
    
    date_summary.columns = ['players_predicted', 'avg_predicted', 'std_predicted', 
                           'min_predicted', 'max_predicted', 'unique_players']
    
    # Save summaries
    player_file = f'{output_dir}walk_forward_player_summary.csv'
    date_file = f'{output_dir}walk_forward_date_summary.csv'
    
    player_summary.to_csv(player_file)
    date_summary.to_csv(date_file)
    
    print(f"Player summary saved to: {player_file}")
    print(f"Date summary saved to: {date_file}")
    
    # Print top players by average prediction
    print(f"\nTop 10 Players by Average Predicted Points:")
    top_players = player_summary.sort_values('avg_predicted', ascending=False).head(10)
    for name, row in top_players.iterrows():
        print(f"{name}: {row['avg_predicted']:.2f} avg ({row['games_predicted']} games)")
    
    return player_summary, date_summary

def main():
    """Main execution function for walk-forward predictions"""
    # Configuration
    input_file = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    output_dir = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
    
    # Walk-forward prediction configuration
    start_date = '2024-04-01'  # Start of MLB season
    end_date = '2024-10-31'    # End of MLB season
    train_window_days = 365    # Use 1 year of historical data for training
    retrain_frequency = 7      # Retrain model every 7 days
    
    print("="*80)
    print("WALK-FORWARD PREDICTION PIPELINE - REFACTORED")
    print("="*80)
    print(f"Prediction period: {start_date} to {end_date}")
    print(f"Training window: {train_window_days} days")
    print(f"Retrain frequency: {retrain_frequency} days")
    print("="*80)
    
    try:
        # Load and prepare data
        print("Loading dataset...")
        df = pd.read_csv(input_file,
                         dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 
                               'catchersInterference': 'int64', 'salary': 'int64'},
                         low_memory=False)
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values(['Name', 'date'])
        
        print(f"Dataset loaded: {len(df)} records")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique players: {df['Name'].nunique()}")
        
        # Calculate DK fantasy points if not present
        if 'calculated_dk_fpts' not in df.columns:
            print("Calculating DK Fantasy Points...")
            df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)
        
        # Basic preprocessing
        df = df.fillna(0)
        
        # Load model
        print("Loading model pipeline...")
        pipeline = joblib.load(model_file)
        
        # Create walk-forward predictor
        predictor = WalkForwardPredictor(
            train_window_days=train_window_days,
            min_train_samples=1000
        )
        
        # Run walk-forward prediction
        predictions = predictor.run_walk_forward_prediction(
            df, pipeline, start_date, end_date, retrain_frequency
        )
        
        if predictions is not None:
            print("\nWalk-forward prediction completed successfully!")
            print(f"Generated {len(predictions)} predictions")
            
            # Validate predictions
            validate_walk_forward_predictions(predictions)
            
            # Create summary report
            create_prediction_summary_report(predictions, output_dir)
            
            print("\nPrediction data is ready for further analysis!")
            return predictions
        else:
            print("\nWalk-forward prediction failed!")
            return None
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    predictions = main()