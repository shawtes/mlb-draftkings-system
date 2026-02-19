"""
Walk-Forward Prediction System for MLB DraftKings
Refactored to prevent data leakage and focus on pure walk-forward predictions
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import DataConversionWarning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

class EnhancedMLBFinancialStyleEngine:
    """Enhanced feature engineering with financial-style indicators"""
    
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
        """Calculate financial-style features for MLB data"""
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
                logger.warning(f"Column '{col}' not found. Initialized with 0.")

        # Group by player and calculate features
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # Momentum Features (SMA, EMA, ROC)
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                
                # Performance vs moving average
                if f'{col}_sma_28' in new_features:
                    new_features[f'{col}_vs_sma_28'] = (group[col] / new_features[f'{col}_sma_28']) - 1
            
            # Volatility Features (Bollinger Bands)
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                new_features[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                
                if mean is not None and not mean.empty:
                    new_features[f'dk_fpts_band_width_{window}'] = (
                        new_features[f'dk_fpts_upper_band_{window}'] - 
                        new_features[f'dk_fpts_lower_band_{window}']
                    ) / mean
                    new_features[f'dk_fpts_band_position_{window}'] = (
                        group['calculated_dk_fpts'] - new_features[f'dk_fpts_lower_band_{window}']
                    ) / (
                        new_features[f'dk_fpts_upper_band_{window}'] - 
                        new_features[f'dk_fpts_lower_band_{window}']
                    )

            # Volume-based Features
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    new_features[f'{vol_col}_ratio'] = group[vol_col] / new_features[f'{vol_col}_roll_mean_28']
                    new_features[f'dk_fpts_{vol_col}_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group[vol_col])

            # Interaction Features
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns and 'PA' in group.columns and group['PA'].sum() > 0:
                    new_features[f'{col}_per_pa'] = group[col] / group['PA']
            
            # Temporal Features
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

class WalkForwardPredictor:
    """
    Walk-forward prediction system to prevent data leakage
    """
    
    def __init__(self, train_window_days=365, min_train_samples=1000):
        self.train_window_days = train_window_days
        self.min_train_samples = min_train_samples
        self.predictions_history = []
        self.model_performance = []
        
    def calculate_dk_fpts(self, row):
        """Calculate DraftKings fantasy points"""
        return (row['1B'] * 3 + row['2B'] * 5 + row['3B'] * 8 + row['HR'] * 10 +
                row['RBI'] * 2 + row['R'] * 2 + row['BB'] * 2 + row['HBP'] * 2 + row['SB'] * 5)
    
    def clean_infinite_values(self, df):
        """Clean infinite and NaN values from dataframe"""
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, replace NaN with the mean of the column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # For non-numeric columns, replace NaN with a placeholder value
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def engineer_basic_features(self, df):
        """Engineer basic statistical features"""
        df = df.copy()
        
        # Extract date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_season'] = (df['date'] - df['date'].min()).dt.days
        
        # Calculate singles
        df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
        
        # Calculate basic sabermetrics
        df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) 
                              if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
        df['ISO'] = df['SLG'] - df['AVG']
        
        # Calculate rolling statistics
        for window in [3, 7, 14, 28]:
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Player consistency features
        df['fpts_volatility'] = df['rolling_std_fpts_7'] / df['rolling_mean_fpts_7']
        
        df = df.fillna(0)
        return df
    
    def get_mlb_season_dates(self, start_date, end_date):
        """Generate MLB season dates (April through October)"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to MLB season months
        mlb_season_dates = [
            d for d in date_range 
            if d.month >= 4 and d.month <= 10
        ]
        
        return mlb_season_dates
    
    def prepare_training_data(self, df, current_date):
        """Prepare training data up to current_date using rolling window"""
        current_date = pd.to_datetime(current_date)
        
        # Define training window - only use data before current_date
        train_start = current_date - pd.Timedelta(days=self.train_window_days)
        train_end = current_date - pd.Timedelta(days=1)  # Don't include current day
        
        # Filter data for training window
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        train_data = df[train_mask].copy()
        
        logger.info(f"Training data range: {train_start.date()} to {train_end.date()}")
        logger.info(f"Training samples: {len(train_data)}")
        
        if len(train_data) < self.min_train_samples:
            logger.warning(f"Only {len(train_data)} training samples available (minimum: {self.min_train_samples})")
            return None
            
        return train_data
    
    def create_player_features_for_date(self, df, players, prediction_date):
        """Create feature rows for all players for a specific prediction date"""
        logger.info(f"Creating features for {len(players)} players on {prediction_date}")
        
        synthetic_rows = []
        prediction_date = pd.to_datetime(prediction_date)
        
        for player in players:
            player_df = df[df['Name'] == player].sort_values('date', ascending=False)
            
            if player_df.empty:
                # Create default row for players with no history
                logger.warning(f"No historical data for player {player}")
                default_row = pd.DataFrame([{col: 0 for col in df.columns}])
                default_row['date'] = prediction_date
                default_row['Name'] = player
                default_row['calculated_dk_fpts'] = 0
                default_row['has_historical_data'] = False
                synthetic_rows.append(default_row)
            else:
                # Use recent data (up to 20 games) to create features
                recent_data = player_df.head(20)
                
                # Calculate averages for numeric columns
                numeric_columns = recent_data.select_dtypes(include=[np.number]).columns
                numeric_averages = recent_data[numeric_columns].mean()
                
                synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
                synthetic_row['date'] = prediction_date
                synthetic_row['Name'] = player
                synthetic_row['has_historical_data'] = True
                
                # Handle non-numeric columns
                for col in recent_data.select_dtypes(include=['object']).columns:
                    if col not in ['date', 'Name']:
                        mode_val = recent_data[col].mode()
                        synthetic_row[col] = mode_val.iloc[0] if not mode_val.empty else recent_data[col].iloc[0]
                
                synthetic_rows.append(synthetic_row)
        
        if synthetic_rows:
            synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
            logger.info(f"Created {len(synthetic_rows)} synthetic rows for {prediction_date}")
            return synthetic_df
        else:
            logger.warning(f"No synthetic rows created for {prediction_date}")
            return pd.DataFrame()
    
    def retrain_model(self, train_data, pipeline_template):
        """Retrain the model on current training window"""
        # Prepare features and target
        feature_cols = [col for col in train_data.columns 
                       if col not in ['calculated_dk_fpts', 'date', 'Name', 'Team']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['calculated_dk_fpts']
        
        # Clean training data
        X_train = self.clean_infinite_values(X_train)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(0)
        
        # Clone and retrain pipeline
        model_pipeline = clone(pipeline_template)
        
        try:
            model_pipeline.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred = model_pipeline.predict(X_train)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            logger.info(f"Model retrained - MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
            
            return model_pipeline
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
            return None
    
    def make_predictions_for_date(self, model_pipeline, prediction_data):
        """Make predictions for a specific date"""
        if prediction_data.empty:
            return pd.DataFrame()
        
        # Prepare features
        feature_cols = [col for col in prediction_data.columns 
                       if col not in ['calculated_dk_fpts', 'date', 'Name', 'Team']]
        
        X_pred = prediction_data[feature_cols]
        
        # Clean prediction data
        X_pred = self.clean_infinite_values(X_pred)
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
        X_pred = X_pred.fillna(0)
        
        try:
            # Make predictions
            predictions = model_pipeline.predict(X_pred)
            predictions = np.clip(predictions, 0, 100)  # Realistic DK points range
            
            # Create results dataframe
            results = pd.DataFrame({
                'Name': prediction_data['Name'],
                'prediction_date': prediction_data['date'].iloc[0],
                'predicted_dk_fpts': predictions,
                'has_historical_data': prediction_data.get('has_historical_data', True)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return pd.DataFrame()
    
    def run_walk_forward_prediction(self, df, pipeline_template, start_date, end_date, 
                                   retrain_frequency=7, output_dir=None):
        """Run walk-forward prediction for the entire date range"""
        logger.info("="*80)
        logger.info("WALK-FORWARD PREDICTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Prediction period: {start_date} to {end_date}")
        logger.info(f"Training window: {self.train_window_days} days")
        logger.info(f"Retrain frequency: {retrain_frequency} days")
        
        # Get prediction dates
        prediction_dates = self.get_mlb_season_dates(start_date, end_date)
        logger.info(f"Total prediction dates: {len(prediction_dates)}")
        
        # Get all unique players
        all_players = df['Name'].unique()
        logger.info(f"Total players: {len(all_players)}")
        
        all_predictions = []
        current_model = None
        last_train_date = None
        
        for i, current_date in enumerate(prediction_dates):
            logger.info(f"\nProcessing date {i+1}/{len(prediction_dates)}: {current_date.date()}")
            
            # Check if we need to retrain
            should_retrain = (
                current_model is None or 
                last_train_date is None or 
                (current_date - last_train_date).days >= retrain_frequency
            )
            
            if should_retrain:
                logger.info("Retraining model...")
                train_data = self.prepare_training_data(df, current_date)
                
                if train_data is not None:
                    current_model = self.retrain_model(train_data, pipeline_template)
                    last_train_date = current_date
                    
                    if current_model is None:
                        logger.warning("Failed to retrain model, skipping this date")
                        continue
                else:
                    logger.warning("Insufficient training data, skipping this date")
                    continue
            
            # Create prediction data for all players
            prediction_data = self.create_player_features_for_date(df, all_players, current_date)
            
            if not prediction_data.empty:
                # Make predictions
                day_predictions = self.make_predictions_for_date(current_model, prediction_data)
                
                if not day_predictions.empty:
                    all_predictions.append(day_predictions)
                    logger.info(f"Generated {len(day_predictions)} predictions")
                else:
                    logger.warning("Failed to generate predictions for this date")
            else:
                logger.warning("No prediction data available for this date")
        
        # Combine all predictions
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            logger.info(f"\nWalk-forward prediction complete!")
            logger.info(f"Total predictions: {len(final_predictions)}")
            logger.info(f"Date range: {final_predictions['prediction_date'].min()} to {final_predictions['prediction_date'].max()}")
            logger.info(f"Unique players: {final_predictions['Name'].nunique()}")
            
            # Save predictions
            if output_dir is None:
                output_dir = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
            
            output_file = os.path.join(output_dir, f'walk_forward_predictions_{start_date}_{end_date}.csv')
            final_predictions.to_csv(output_file, index=False)
            logger.info(f"Predictions saved to: {output_file}")
            
            # Generate summary statistics
            self.generate_prediction_summary(final_predictions, output_dir)
            
            return final_predictions
        else:
            logger.error("No predictions generated!")
            return None
    
    def generate_prediction_summary(self, predictions_df, output_dir):
        """Generate summary statistics for predictions"""
        logger.info("\n=== PREDICTION SUMMARY ===")
        
        # Overall statistics
        logger.info(f"Total predictions: {len(predictions_df)}")
        logger.info(f"Unique players: {predictions_df['Name'].nunique()}")
        logger.info(f"Date range: {predictions_df['prediction_date'].min()} to {predictions_df['prediction_date'].max()}")
        
        # Prediction statistics
        pred_stats = predictions_df['predicted_dk_fpts'].describe()
        logger.info(f"\nPrediction Statistics:")
        logger.info(f"Mean: {pred_stats['mean']:.2f}")
        logger.info(f"Std: {pred_stats['std']:.2f}")
        logger.info(f"Min: {pred_stats['min']:.2f}")
        logger.info(f"Max: {pred_stats['max']:.2f}")
        
        # Daily prediction counts
        daily_counts = predictions_df.groupby('prediction_date').size()
        logger.info(f"\nDaily Statistics:")
        logger.info(f"Average predictions per day: {daily_counts.mean():.1f}")
        logger.info(f"Min predictions per day: {daily_counts.min()}")
        logger.info(f"Max predictions per day: {daily_counts.max()}")
        
        # Player-level summary
        player_summary = predictions_df.groupby('Name').agg({
            'predicted_dk_fpts': ['count', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        player_summary.columns = ['games_predicted', 'avg_predicted', 'std_predicted', 
                                 'min_predicted', 'max_predicted']
        
        # Save player summary
        player_file = os.path.join(output_dir, 'walk_forward_player_summary.csv')
        player_summary.to_csv(player_file)
        logger.info(f"Player summary saved to: {player_file}")
        
        # Top performers
        top_players = player_summary.sort_values('avg_predicted', ascending=False).head(10)
        logger.info(f"\nTop 10 Players by Average Predicted Points:")
        for name, row in top_players.iterrows():
            logger.info(f"{name}: {row['avg_predicted']:.2f} avg ({row['games_predicted']} games)")

def load_and_prepare_data(input_file):
    """Load and prepare the dataset"""
    logger.info("Loading dataset...")
    
    df = pd.read_csv(input_file,
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 
                           'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    null_dates = df['date'].isnull().sum()
    
    if null_dates > 0:
        logger.warning(f"{null_dates} dates could not be parsed and were removed")
        df = df.dropna(subset=['date'])
    
    # Sort by player and date
    df = df.sort_values(['Name', 'date'])
    
    logger.info(f"Dataset loaded: {len(df)} records")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique players: {df['Name'].nunique()}")
    
    return df

def main():
    """Main execution function"""
    # Configuration
    input_file = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    output_dir = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/'
    
    # Prediction configuration
    start_date = '2024-04-01'  # Start of MLB season
    end_date = '2024-10-31'    # End of MLB season
    train_window_days = 365    # Use 1 year of historical data
    retrain_frequency = 7      # Retrain every 7 days
    
    try:
        # Load data
        df = load_and_prepare_data(input_file)
        
        # Calculate DK fantasy points if not present
        predictor = WalkForwardPredictor(train_window_days=train_window_days)
        
        if 'calculated_dk_fpts' not in df.columns:
            logger.info("Calculating DK Fantasy Points...")
            df['calculated_dk_fpts'] = df.apply(predictor.calculate_dk_fpts, axis=1)
        
        # Engineer features
        logger.info("Engineering features...")
        financial_engine = EnhancedMLBFinancialStyleEngine()
        df = financial_engine.calculate_features(df)
        df = predictor.engineer_basic_features(df)
        
        # Final cleanup
        df = predictor.clean_infinite_values(df)
        df = df.fillna(0)
        
        # Load model
        logger.info("Loading model pipeline...")
        pipeline = joblib.load(model_file)
        
        # Run walk-forward prediction
        predictions = predictor.run_walk_forward_prediction(
            df=df,
            pipeline_template=pipeline,
            start_date=start_date,
            end_date=end_date,
            retrain_frequency=retrain_frequency,
            output_dir=output_dir
        )
        
        if predictions is not None:
            logger.info("Walk-forward prediction completed successfully!")
            return predictions
        else:
            logger.error("Walk-forward prediction failed!")
            return None
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return None

if __name__ == "__main__":
    predictions = main()
