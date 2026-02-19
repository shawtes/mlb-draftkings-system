"""
Efficient Data Preprocessing for HP Omen 35L

This script handles all the data preprocessing steps efficiently:
1. Memory-optimized CSV loading with chunking
2. Feature engineering in parallel chunks
3. Data cleaning and transformation
4. Preprocessing pipeline creation
5. Saving processed data for quick model training

Optimizations for HP Omen 35L:
- Chunked processing for large datasets
- Memory-efficient operations
- GPU/CPU optimization detection
- Parallel processing when beneficial
"""

import pandas as pd
import numpy as np
import time
import warnings
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import torch
import psutil

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

# League averages and wOBA weights from training.py
league_avg_wOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315,
    2025: 0.315  # Assuming similar to 2024
}

league_avg_HR_FlyBall = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141,
    2025: 0.141  # Assuming similar to 2024
}

# wOBA weights for different years
wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04},
    2025: {'BB': 0.64, 'HBP': 0.67, '1B': 0.83, '2B': 1.19, '3B': 1.51, 'HR': 2.03}  # Projected
}

def get_system_specs():
    """Get system specifications for optimization"""
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_available = torch.cuda.is_available()
    
    return {
        'cpu_cores': cpu_count,
        'memory_gb': memory_gb,
        'gpu_available': gpu_available,
        'chunk_size': 25000 if memory_gb >= 16 else 15000,
        'max_workers': min(4, cpu_count - 2)
    }

def calculate_dk_fpts(row):
    """Calculate DraftKings fantasy points from baseball stats"""
    try:
        singles = pd.to_numeric(row.get('1B', 0), errors='coerce')
        doubles = pd.to_numeric(row.get('2B', 0), errors='coerce')
        triples = pd.to_numeric(row.get('3B', 0), errors='coerce')
        hr = pd.to_numeric(row.get('HR', 0), errors='coerce')
        rbi = pd.to_numeric(row.get('RBI', 0), errors='coerce')
        r = pd.to_numeric(row.get('R', 0), errors='coerce')
        bb = pd.to_numeric(row.get('BB', 0), errors='coerce')
        hbp = pd.to_numeric(row.get('HBP', 0), errors='coerce')
        sb = pd.to_numeric(row.get('SB', 0), errors='coerce')

        # Handle NaN values
        singles = 0 if pd.isna(singles) else singles
        doubles = 0 if pd.isna(doubles) else doubles
        triples = 0 if pd.isna(triples) else triples
        hr = 0 if pd.isna(hr) else hr
        rbi = 0 if pd.isna(rbi) else rbi
        r = 0 if pd.isna(r) else r
        bb = 0 if pd.isna(bb) else bb
        hbp = 0 if pd.isna(hbp) else hbp
        sb = 0 if pd.isna(sb) else sb

        return (singles * 3 + doubles * 5 + triples * 8 + hr * 10 +
                rbi * 2 + r * 2 + bb * 2 + hbp * 2 + sb * 5)
    except Exception as e:
        print(f"Error calculating DK points: {e}")
        return 0

def engineer_advanced_features(df):
    """Advanced feature engineering from training.py"""
    print("Engineering advanced features...")
    
    # Ensure date column exists and is datetime
    if 'date' not in df.columns and 'game_date' in df.columns:
        df['date'] = df['game_date']
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_season'] = (df['date'] - df['date'].min()).dt.days

    # Define default values to handle years not present in the lookup tables
    default_wOBA = 0.317  # A reasonable league average
    default_HR_FlyBall = 0.143  # A reasonable league average
    default_wOBA_weights = wOBA_weights[2022]  # Use a recent year as default

    # Calculate DK fantasy points if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DraftKings fantasy points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    # Calculate singles first (needed for other calculations)
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
    df['1B'] = df['1B'].clip(lower=0)  # Ensure non-negative

    # Calculate key sabermetric statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics (with safe fallbacks for missing years)
    df['wRAA'] = df.apply(lambda x: ((x['wOBA'] - league_avg_wOBA.get(x['year'], default_wOBA)) / 1.15) * x['AB'] if x['AB'] > 0 else 0, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA.get(x['year'], default_wOBA) * 100) if x['AB'] > 0 and league_avg_wOBA.get(x['year'], default_wOBA) > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) if league_avg_HR_FlyBall.get(x['year'], default_HR_FlyBall) > 0 else 0, axis=1)

    # Calculate wOBA using year-specific weights (with safe fallbacks)
    df['wOBA_Statcast'] = df.apply(lambda x: (
        wOBA_weights.get(x['year'], default_wOBA_weights)['BB'] * x.get('BB', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HBP'] * x.get('HBP', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['1B'] * x.get('1B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['2B'] * x.get('2B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['3B'] * x.get('3B', 0) +
        wOBA_weights.get(x['year'], default_wOBA_weights)['HR'] * x.get('HR', 0)
    ) / (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) if (x.get('AB', 0) + x.get('BB', 0) - x.get('IBB', 0) + x.get('SF', 0) + x.get('HBP', 0)) > 0 else 0, axis=1)

    # Calculate SLG_Statcast
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

    # Basic rate stats with safe division
    df['BA'] = np.where(df['AB'] > 0, df['H'] / df['AB'], 0)
    df['OBP'] = np.where(
        (df['AB'] + df['BB'] + df['HBP'] + df['SF']) > 0,
        (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF']),
        0
    )
    df['SLG'] = np.where(
        df['AB'] > 0,
        (df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR']) / df['AB'],
        0
    )
    df['OPS'] = df['OBP'] + df['SLG']

    # Add is_weekend feature
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Clean infinite and NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def create_advanced_rolling_features(df, player_col='Name', target_col='calculated_dk_fpts'):
    """Create advanced rolling statistics matching training.py"""
    print("Creating advanced rolling features...")
    
    # Sort by player and date
    df = df.sort_values([player_col, 'date'])
    
    # Calculate rolling statistics if target column is present
    if target_col in df.columns:
        # Rolling windows for min/max/mean (matching training.py)
        for window in [7, 49]:
            df[f'rolling_min_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            df[f'rolling_max_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            df[f'rolling_mean_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Lag features (previous performance) - matching training.py
        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            df[f'lag_max_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).max().shift(1)
            )
            df[f'lag_min_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).min().shift(1)
            )
            
            # Additional rolling mean features for standard windows
            df[f'rolling_mean_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_fpts_{window}'] = df.groupby(player_col)[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

        # Rolling statistics for key offensive stats
        for stat in ['H', 'HR', 'RBI', 'R', 'BB', 'SB']:
            if stat in df.columns:
                for window in [3, 7, 14, 28]:
                    df[f'rolling_{stat.lower()}_{window}'] = df.groupby(player_col)[stat].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
    
    # Fill NaN values from shift operations
    df.fillna(0, inplace=True)
    
    return df

def load_data_efficiently(csv_path, chunk_size=25000):
    """Load large CSV file efficiently using chunking"""
    print(f"Loading data from: {csv_path}")
    print(f"Using chunk size: {chunk_size}")
    
    chunks = []
    chunk_count = 0
    
    try:
        # Define data types to reduce memory usage
        dtype_dict = {
            'inheritedRunners': 'float32',
            'inheritedRunnersScored': 'float32',
            'catchersInterference': 'int16',
            'salary': 'int32'
        }
        
        for chunk in pd.read_csv(csv_path, 
                                chunksize=chunk_size, 
                                dtype=dtype_dict,
                                low_memory=False):
            chunk_count += 1
            print(f"Processing chunk {chunk_count} ({len(chunk)} rows)...")
            
            # Basic preprocessing for each chunk
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk.fillna(0, inplace=True)
            
            # Convert object columns to string to avoid mixed types
            for col in chunk.select_dtypes(include=['object']).columns:
                chunk[col] = chunk[col].astype(str)
            
            chunks.append(chunk)
            
            # Optional: Limit chunks for testing
            # if chunk_count >= 5:  # Remove this line for full dataset
            #     break
        
        print(f"Concatenating {len(chunks)} chunks...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"Dataset loaded successfully! Total rows: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_preprocessing_pipeline(df):
    """Create preprocessing pipeline for features"""
    print("Creating preprocessing pipeline...")
    
    # Define feature categories (matching training.py)
    numeric_features = [
        # Date features
        'year', 'month', 'day', 'day_of_week', 'day_of_season', 'is_weekend',
        
        # Basic stats
        'H', 'HR', 'RBI', 'R', 'BB', 'SB', 'AB', 'PA', 'SO', '1B', '2B', '3B', 'HBP', 'SF', 'GDP',
        
        # Rate stats
        'BA', 'OBP', 'SLG', 'OPS', 'AVG', 'BABIP', 'ISO',
        
        # Advanced sabermetrics
        'wOBA', 'wRAA', 'wRC', 'wRC+', 'flyBalls', 'wOBA_Statcast', 'SLG_Statcast', 
        'RAR_Statcast', 'Offense_Statcast', 'Dollars_Statcast', 'WPA/LI_Statcast',
        
        # WAR and other advanced metrics
        'WAR', 'Off', 'Def', 'BsR', 'Dol', 'RAR', 'RE24', 'REW', 'FIP',
        
        # Rolling features for fantasy points (7, 49 day windows)
        'rolling_min_fpts_7', 'rolling_max_fpts_7', 'rolling_mean_fpts_7',
        'rolling_min_fpts_49', 'rolling_max_fpts_49', 'rolling_mean_fpts_49',
        
        # Rolling features for fantasy points (3, 7, 14, 28 day windows)
        'rolling_mean_fpts_3', 'rolling_mean_fpts_14', 'rolling_mean_fpts_28',
        'rolling_std_fpts_3', 'rolling_std_fpts_7', 'rolling_std_fpts_14', 'rolling_std_fpts_28',
        
        # Lag features
        'lag_mean_fpts_3', 'lag_mean_fpts_7', 'lag_mean_fpts_14', 'lag_mean_fpts_28',
        'lag_max_fpts_3', 'lag_max_fpts_7', 'lag_max_fpts_14', 'lag_max_fpts_28',
        'lag_min_fpts_3', 'lag_min_fpts_7', 'lag_min_fpts_14', 'lag_min_fpts_28',
        
        # Rolling statistics for key offensive stats
        'rolling_h_3', 'rolling_h_7', 'rolling_h_14', 'rolling_h_28',
        'rolling_hr_3', 'rolling_hr_7', 'rolling_hr_14', 'rolling_hr_28',
        'rolling_rbi_3', 'rolling_rbi_7', 'rolling_rbi_14', 'rolling_rbi_28',
        'rolling_r_3', 'rolling_r_7', 'rolling_r_14', 'rolling_r_28',
        'rolling_bb_3', 'rolling_bb_7', 'rolling_bb_14', 'rolling_bb_28',
        'rolling_sb_3', 'rolling_sb_7', 'rolling_sb_14', 'rolling_sb_28',
        
        # Plate discipline and contact metrics
        'BB%', 'K%', 'BB/K', 'SwStr%', 'O-Swing%', 'Z-Swing%', 'Contact%',
        
        # Batted ball metrics
        'GB%', 'FB%', 'LD%', 'IFFB%', 'HR/FB', 'GB/FB',
        
        # Pitch values and velocities (if available)
        'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF', 'wKN',
        'vFA (sc)', 'vFT (sc)', 'vFC (sc)', 'vSL (sc)', 'vCU (sc)', 'vCH (sc)',
        
        # Financial-style features from EnhancedMLBFinancialStyleEngine
        # Momentum features (SMA, EMA, ROC for all stats)
        'HR_sma_3', 'HR_sma_7', 'HR_sma_14', 'HR_sma_28', 'HR_sma_45',
        'RBI_sma_3', 'RBI_sma_7', 'RBI_sma_14', 'RBI_sma_28', 'RBI_sma_45',
        'BB_sma_3', 'BB_sma_7', 'BB_sma_14', 'BB_sma_28', 'BB_sma_45',
        'HR_ema_3', 'HR_ema_7', 'HR_ema_14', 'HR_ema_28', 'HR_ema_45',
        'RBI_ema_3', 'RBI_ema_7', 'RBI_ema_14', 'RBI_ema_28', 'RBI_ema_45',
        'BB_ema_3', 'BB_ema_7', 'BB_ema_14', 'BB_ema_28', 'BB_ema_45',
        'HR_roc_3', 'HR_roc_7', 'HR_roc_14', 'HR_roc_28', 'HR_roc_45',
        'RBI_roc_3', 'RBI_roc_7', 'RBI_roc_14', 'RBI_roc_28', 'RBI_roc_45',
        
        # Volatility features (Bollinger Band style)
        'dk_fpts_upper_band_3', 'dk_fpts_upper_band_7', 'dk_fpts_upper_band_14', 'dk_fpts_upper_band_28',
        'dk_fpts_lower_band_3', 'dk_fpts_lower_band_7', 'dk_fpts_lower_band_14', 'dk_fpts_lower_band_28',
        'dk_fpts_band_width_3', 'dk_fpts_band_width_7', 'dk_fpts_band_width_14', 'dk_fpts_band_width_28',
        'dk_fpts_band_position_3', 'dk_fpts_band_position_7', 'dk_fpts_band_position_14', 'dk_fpts_band_position_28',
        
        # Volume-based features (opportunity metrics)
        'PA_roll_mean_28', 'AB_roll_mean_28', 'PA_ratio', 'AB_ratio',
        'dk_fpts_PA_corr_28', 'dk_fpts_AB_corr_28',
        
        # Rate/efficiency features
        'HR_per_pa', 'RBI_per_pa', 'BB_per_pa', 'H_per_pa', 'SO_per_pa', 'R_per_pa',
        
        # Temporal features
        'is_weekend', 'day_of_week_sin', 'day_of_week_cos',
        
        # Miscellaneous advanced stats
        'WPA', 'pLI', 'phLI', 'WPA/LI', 'Clutch', 'Spd', 'UZR', 'wGDP'
    ]
    
    categorical_features = ['Name', 'Team']
    
    # Filter to only include features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor, numeric_features, categorical_features

def preprocess_data(csv_path, output_path):
    """Main preprocessing function"""
    print("🚀 Starting efficient data preprocessing for HP Omen 35L")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Get system specifications
    specs = get_system_specs()
    print(f"System: {specs['cpu_cores']} cores, {specs['memory_gb']:.1f} GB RAM, GPU: {specs['gpu_available']}")
    
    # Step 2: Load data efficiently
    df = load_data_efficiently(csv_path, specs['chunk_size'])
    if df is None:
        print("❌ Failed to load data!")
        return
    
    # Step 3: Engineer advanced features (including financial-style)
    print("Applying financial-style feature engineering...")
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    
    # Step 4: Engineer additional advanced features
    df = engineer_advanced_features(df)
    
    # Step 5: Create advanced rolling features
    df = create_advanced_rolling_features(df)
    
    # Step 6: Create preprocessing pipeline
    preprocessor, numeric_features, categorical_features = create_preprocessing_pipeline(df)
    
    # Step 7: Prepare features and target
    print("Preparing features and target...")
    
    # Remove target from features
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df['calculated_dk_fpts'].copy()
    
    # Clean features
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # Step 8: Fit preprocessing pipeline and transform data
    print("Fitting preprocessing pipeline...")
    X_preprocessed = preprocessor.fit_transform(X)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Preprocessed features: {X_preprocessed.shape[1]}")
    
    # Step 9: Feature selection (increased from 500 to match training.py expectations)
    print("Performing feature selection...")
    n_features_to_select = min(750, X_preprocessed.shape[1])  # Increased for more comprehensive features
    selector = SelectKBest(f_regression, k=n_features_to_select)
    X_selected = selector.fit_transform(X_preprocessed, y)
    
    print(f"Selected features: {X_selected.shape[1]}")
    
    # Step 10: Save processed data
    print("Saving processed data...")
    
    processed_data = {
        'X': X_selected,
        'y': y.values,
        'preprocessor': preprocessor,
        'selector': selector,
        'feature_names': all_features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'preprocessing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'original_shape': df.shape,
        'final_shape': X_selected.shape
    }
    
    joblib.dump(processed_data, output_path)
    
    # Step 11: Performance summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Data preprocessing completed successfully!")
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"📊 Original Data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"🔧 Processed Data: {X_selected.shape[0]} rows × {X_selected.shape[1]} features")
    print(f"💾 Saved to: {output_path}")
    print("=" * 60)

def main():
    """Main function"""
    csv_path = r'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/4_DATA/merged_output.csv'
    output_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/processed_training_data.joblib'
    
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        print("Please ensure the merged_output.csv file exists.")
        return
    
    preprocess_data(csv_path, output_path)

if __name__ == "__main__":
    main()
