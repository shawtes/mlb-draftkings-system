#!/usr/bin/env python3
"""
Quick Walk-Forward Prediction Generator
Creates a year's worth of predictions using a simplified approach
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class QuickWalkForwardPredictor:
    """Simplified walk-forward predictor for quick generation of training data"""
    
    def __init__(self, model_path, train_window_days=365):
        self.model_path = model_path
        self.train_window_days = train_window_days
        self.model = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_and_prepare_data(self, data_path):
        """Load and prepare the historical data"""
        print("Loading historical data...")
        
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Calculate DK points if not present
        if 'calculated_dk_fpts' not in df.columns:
            df['calculated_dk_fpts'] = (
                df['1B'] * 3 + df['2B'] * 5 + df['3B'] * 8 + df['HR'] * 10 +
                df['RBI'] * 2 + df['R'] * 2 + df['BB'] * 2 + df['HBP'] * 2 + df['SB'] * 5
            )
        
        # Sort by date
        df = df.sort_values(['Name', 'date'])
        
        print(f"Data loaded: {len(df)} records, {df['Name'].nunique()} players")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def create_basic_features(self, df):
        """Create basic features for prediction"""
        print("Creating basic features...")
        
        # Add basic rolling statistics
        for window in [7, 14, 30]:
            df[f'avg_fpts_{window}d'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'std_fpts_{window}d'] = df.groupby('Name')['calculated_dk_fpts'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        # Add basic rate stats
        df['hr_rate'] = df['HR'] / df['AB'].replace(0, 1)
        df['bb_rate'] = df['BB'] / df['AB'].replace(0, 1)
        df['sb_rate'] = df['SB'] / df['AB'].replace(0, 1)
        
        # Add date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_season'] = df['date'].dt.dayofyear
        
        # Fill missing values
        df = df.fillna(0)
        
        print("Basic features created")
        return df
    
    def get_prediction_features(self, df):
        """Get features for prediction"""
        feature_columns = [
            'avg_fpts_7d', 'avg_fpts_14d', 'avg_fpts_30d',
            'std_fpts_7d', 'std_fpts_14d', 'std_fpts_30d',
            'hr_rate', 'bb_rate', 'sb_rate',
            'day_of_week', 'month', 'day_of_season',
            'AB', 'HR', 'RBI', 'BB', 'SB', 'H', 'R'
        ]
        
        # Only use features that exist in the data
        available_features = [col for col in feature_columns if col in df.columns]
        
        return df[available_features]
    
    def generate_walk_forward_predictions(self, df, start_date, end_date):
        """Generate walk-forward predictions"""
        print(f"Generating walk-forward predictions from {start_date} to {end_date}")
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate prediction dates (business days only)
        prediction_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        prediction_dates = [d for d in prediction_dates if d.month in range(4, 11)]  # MLB season
        
        print(f"Will generate predictions for {len(prediction_dates)} dates")
        
        all_predictions = []
        
        for i, pred_date in enumerate(prediction_dates):
            if i % 10 == 0:
                print(f"Processing date {i+1}/{len(prediction_dates)}: {pred_date.date()}")
            
            # Get training data up to prediction date
            train_data = df[df['date'] < pred_date].copy()
            
            if len(train_data) < 100:  # Need minimum data
                continue
            
            # Get latest data for each player
            latest_data = train_data.groupby('Name').tail(1).copy()
            
            if len(latest_data) == 0:
                continue
            
            # Get features for prediction
            X_pred = self.get_prediction_features(latest_data)
            
            # Make predictions
            try:
                predictions = self.model.predict(X_pred)
                predictions = np.clip(predictions, 0, 100)  # Realistic range
                
                # Create results
                results = pd.DataFrame({
                    'Name': latest_data['Name'],
                    'Date': pred_date.date(),
                    'Predicted_DK_Points': predictions,
                    'Historical_Avg_7d': latest_data['avg_fpts_7d'],
                    'Historical_Avg_30d': latest_data['avg_fpts_30d']
                })
                
                all_predictions.append(results)
                
            except Exception as e:
                print(f"Error predicting for {pred_date.date()}: {e}")
                continue
        
        # Combine all predictions
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            print(f"Generated {len(final_predictions)} predictions")
            return final_predictions
        else:
            print("No predictions generated")
            return None
    
    def run_quick_predictions(self, data_path, start_date, end_date):
        """Run the complete quick prediction pipeline"""
        print("Starting Quick Walk-Forward Prediction Pipeline")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            return None
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        
        # Create features
        df = self.create_basic_features(df)
        
        # Generate predictions
        predictions = self.generate_walk_forward_predictions(df, start_date, end_date)
        
        return predictions

def main():
    # Configuration
    model_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    data_path = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    
    # Date range for predictions
    start_date = '2024-04-01'
    end_date = '2024-10-31'
    
    # Create predictor
    predictor = QuickWalkForwardPredictor(model_path)
    
    # Run predictions
    predictions = predictor.run_quick_predictions(data_path, start_date, end_date)
    
    if predictions is not None:
        # Save results
        output_file = f'2_PREDICTIONS/quick_walk_forward_predictions_{start_date}_{end_date}.csv'
        predictions.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Unique players: {predictions['Name'].nunique()}")
        print(f"Date range: {predictions['Date'].min()} to {predictions['Date'].max()}")
        
        # Show sample predictions
        print("\nSample predictions:")
        print(predictions.head(10))
        
        # Show top performers
        print("\nTop 10 predicted performers:")
        top_performers = predictions.nlargest(10, 'Predicted_DK_Points')
        for _, row in top_performers.iterrows():
            print(f"{row['Name']}: {row['Predicted_DK_Points']:.2f} points on {row['Date']}")
        
        return predictions
    else:
        print("Failed to generate predictions")
        return None

if __name__ == "__main__":
    predictions = main()
