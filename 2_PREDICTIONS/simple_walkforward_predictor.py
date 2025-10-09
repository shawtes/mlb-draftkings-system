"""
Simple Walk-Forward Prediction Runner
Clean implementation focused on preventing data leakage
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleWalkForwardPredictor:
    """Simple walk-forward predictor with no data leakage - uses pre-trained model only"""
    
    def __init__(self, min_train_samples=1000):
        self.min_train_samples = min_train_samples
    
    def calculate_dk_fpts(self, row):
        """Calculate DraftKings fantasy points"""
        return (row.get('1B', 0) * 3 + row.get('2B', 0) * 5 + row.get('3B', 0) * 8 + 
                row.get('HR', 0) * 10 + row.get('RBI', 0) * 2 + row.get('R', 0) * 2 + 
                row.get('BB', 0) * 2 + row.get('HBP', 0) * 2 + row.get('SB', 0) * 5)
    
    def get_prediction_dates(self, start_date, end_date):
        """Get MLB season dates for predictions"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate daily dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to MLB season (April through October)
        mlb_dates = [d for d in date_range if 4 <= d.month <= 10]
        
        return mlb_dates
    
    def create_prediction_features(self, df, players, prediction_date):
        """Create features for all players for prediction date using only historical data"""
        prediction_rows = []
        
        for player in players:
            # Get player's historical data (only before prediction date)
            player_data = df[(df['Name'] == player) & (df['date'] < prediction_date)].copy()
            
            if len(player_data) > 0:
                # Use recent averages (last 20 games before prediction date)
                recent_data = player_data.tail(20)
                
                # Calculate feature averages
                numeric_cols = recent_data.select_dtypes(include=[np.number]).columns
                averages = recent_data[numeric_cols].mean()
                
                # Create prediction row
                pred_row = averages.to_dict()
                pred_row['Name'] = player
                pred_row['date'] = prediction_date
                pred_row['has_history'] = True
                
            else:
                # Player with no history before prediction date - use defaults
                pred_row = {col: 0 for col in df.select_dtypes(include=[np.number]).columns}
                pred_row['Name'] = player
                pred_row['date'] = prediction_date
                pred_row['has_history'] = False
            
            prediction_rows.append(pred_row)
        
        return pd.DataFrame(prediction_rows)
    
    def run_walk_forward_predictions(self, df, model_pipeline, start_date, end_date, 
                                   output_file=None):
        """Run walk-forward predictions using pre-trained model only"""
        
        logger.info("Starting Walk-Forward Prediction")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("Using pre-trained model (no retraining)")
        
        # Get prediction dates
        prediction_dates = self.get_prediction_dates(start_date, end_date)
        all_players = df['Name'].unique()
        
        logger.info(f"Total dates: {len(prediction_dates)}")
        logger.info(f"Total players: {len(all_players)}")
        
        all_predictions = []
        
        for i, pred_date in enumerate(prediction_dates):
            logger.info(f"\nProcessing {i+1}/{len(prediction_dates)}: {pred_date.date()}")
            
            # Create prediction features using only data before prediction date
            historical_data = df[df['date'] < pred_date].copy()
            pred_features = self.create_prediction_features(historical_data, all_players, pred_date)
            
            if len(pred_features) > 0:
                # Make predictions
                feature_cols = [col for col in pred_features.columns 
                              if col not in ['Name', 'date', 'has_history']]
                
                X_pred = pred_features[feature_cols].fillna(0)
                
                try:
                    predictions = model_pipeline.predict(X_pred)
                    predictions = np.clip(predictions, 0, 100)  # Realistic range
                    
                    # Store results
                    results = pd.DataFrame({
                        'Name': pred_features['Name'],
                        'prediction_date': pred_date,
                        'predicted_dk_fpts': predictions,
                        'has_history': pred_features['has_history']
                    })
                    
                    all_predictions.append(results)
                    logger.info(f"Generated {len(results)} predictions")
                    
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
            else:
                logger.warning("No prediction features available for this date")
        
        # Combine all predictions
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            
            logger.info(f"\nWalk-forward complete!")
            logger.info(f"Total predictions: {len(final_predictions)}")
            logger.info(f"Date range: {final_predictions['prediction_date'].min()} to {final_predictions['prediction_date'].max()}")
            
            # Save results
            if output_file:
                final_predictions.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")
            
            # Print summary stats
            stats = final_predictions['predicted_dk_fpts'].describe()
            logger.info(f"\nPrediction Statistics:")
            logger.info(f"Mean: {stats['mean']:.2f}")
            logger.info(f"Std: {stats['std']:.2f}")
            logger.info(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            return final_predictions
        
        else:
            logger.error("No predictions generated!")
            return None

def main():
    """Main execution function"""
    
    # Configuration
    input_file = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    output_file = '2_PREDICTIONS/walk_forward_predictions.csv'
    
    # Date range for predictions
    start_date = '2024-04-01'
    end_date = '2024-10-31'
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(input_file, low_memory=False)
        
        # Parse dates and sort
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values(['Name', 'date'])
        
        logger.info(f"Data loaded: {len(df)} records")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Players: {df['Name'].nunique()}")
        
        # Calculate DK points if needed
        predictor = SimpleWalkForwardPredictor()
        
        if 'calculated_dk_fpts' not in df.columns:
            logger.info("Calculating DK Fantasy Points...")
            df['calculated_dk_fpts'] = df.apply(predictor.calculate_dk_fpts, axis=1)
        
        # Load model
        logger.info("Loading model...")
        model_pipeline = joblib.load(model_file)
        
        # Run predictions
        predictions = predictor.run_walk_forward_predictions(
            df=df,
            model_pipeline=model_pipeline,
            start_date=start_date,
            end_date=end_date,
            output_file=output_file
        )
        
        if predictions is not None:
            logger.info("Walk-forward prediction successful!")
            return predictions
        else:
            logger.error("Walk-forward prediction failed!")
            return None
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

if __name__ == "__main__":
    predictions = main()
