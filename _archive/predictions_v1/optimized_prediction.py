"""
Optimized Prediction Script for HP Omen 35L

This script uses the trained optimized model to make predictions on new data.
It includes:
1. Loading the optimized model and preprocessing pipeline
2. Making predictions on new player data
3. Probability estimation for different point thresholds
4. Performance evaluation and visualization

Optimized for HP Omen 35L system.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class OptimizedPredictor:
    """
    Optimized prediction class for MLB fantasy points
    """
    
    def __init__(self, model_path=None, metadata_path=None, processed_data_path=None):
        """Initialize the predictor with model and preprocessing components"""
        
        # Default paths
        if model_path is None:
            model_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/optimized_model.joblib'
        if metadata_path is None:
            metadata_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/model_metadata.joblib'
        if processed_data_path is None:
            processed_data_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/processed_training_data.joblib'
        
        self.model = None
        self.preprocessor = None
        self.selector = None
        self.metadata = None
        self.feature_names = None
        
        # Load model components
        self.load_model_components(model_path, metadata_path, processed_data_path)
    
    def load_model_components(self, model_path, metadata_path, processed_data_path):
        """Load all model components"""
        print("🔄 Loading optimized model components...")
        
        try:
            # Load the trained model
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from: {model_path}")
            
            # Load metadata
            self.metadata = joblib.load(metadata_path)
            print(f"✅ Metadata loaded: {self.metadata['model_type']}")
            
            # Load preprocessing components
            processed_data = joblib.load(processed_data_path)
            self.preprocessor = processed_data['preprocessor']
            self.selector = processed_data['selector']
            self.feature_names = processed_data['feature_names']
            
            print("✅ Preprocessing components loaded successfully")
            print(f"📊 Features: {len(self.feature_names)}")
            print(f"🔧 Model Type: {self.metadata['model_type']}")
            print(f"💾 GPU Used: {self.metadata['training_config']['use_gpu']}")
            
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            raise
    
    def prepare_features(self, df):
        """Prepare features from raw data"""
        print("🔧 Preparing features...")
        
        # Ensure we have the required features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            print(f"⚠️  Warning: Missing features will be filled with zeros: {missing_features[:5]}...")
            for feature in missing_features:
                df[feature] = 0
        
        # Select only the features used in training
        X = df[self.feature_names].copy()
        
        # Clean the data
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        
        return X
    
    def predict(self, X):
        """Make predictions using the optimized model"""
        print("🚀 Making predictions...")
        
        # Preprocess the features
        X_preprocessed = self.preprocessor.transform(X)
        
        # Apply feature selection
        X_selected = self.selector.transform(X_preprocessed)
        
        # Make predictions
        predictions = self.model.predict(X_selected)
        
        # Apply realistic constraints (0-100 DraftKings points)
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def predict_probabilities(self, X, point_thresholds=None):
        """
        Predict probabilities of exceeding point thresholds
        """
        if point_thresholds is None:
            point_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        print(f"🎯 Calculating probabilities for {len(point_thresholds)} thresholds...")
        
        # Get main predictions
        main_predictions = self.predict(X)
        
        # Estimate standard deviation from training data
        # This is a simplified approach - in practice, you'd use quantile regression
        estimated_std = 8.0  # Estimated standard deviation for MLB fantasy points
        
        probability_results = []
        
        for i, pred in enumerate(main_predictions):
            player_probs = {'predicted_points': pred}
            
            # Calculate probability of exceeding each threshold
            for threshold in point_thresholds:
                # Using normal distribution approximation
                z_score = (threshold - pred) / estimated_std
                prob_exceed = 1 - (0.5 * (1 + np.tanh(z_score / np.sqrt(2))))
                prob_exceed = max(0, min(1, prob_exceed))  # Clip to [0, 1]
                
                player_probs[f'prob_over_{threshold}'] = prob_exceed
            
            probability_results.append(player_probs)
        
        return probability_results
    
    def evaluate_predictions(self, y_true, y_pred, player_names=None):
        """Evaluate prediction performance"""
        print("📊 Evaluating prediction performance...")
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Create results dictionary
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred),
            'std_actual': np.std(y_true),
            'std_predicted': np.std(y_pred)
        }
        
        # Print results
        print("\n📈 Prediction Performance:")
        print("=" * 50)
        print(f"Mean Absolute Error: {mae:.3f}")
        print(f"Root Mean Squared Error: {rmse:.3f}")
        print(f"R² Score: {r2:.3f}")
        print(f"Mean Actual: {results['mean_actual']:.2f}")
        print(f"Mean Predicted: {results['mean_predicted']:.2f}")
        print("=" * 50)
        
        return results
    
    def create_prediction_report(self, df, predictions, probabilities=None, output_path=None):
        """Create a comprehensive prediction report"""
        print("📋 Creating prediction report...")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Name': df['Name'] if 'Name' in df.columns else range(len(predictions)),
            'Date': df['date'] if 'date' in df.columns else datetime.now().strftime('%Y-%m-%d'),
            'Predicted_Points': predictions
        })
        
        # Add probability columns if available
        if probabilities:
            for i, prob_dict in enumerate(probabilities):
                for key, value in prob_dict.items():
                    if key.startswith('prob_over_'):
                        threshold = key.split('_')[-1]
                        results_df.loc[i, f'Prob_Over_{threshold}'] = f"{value:.1%}"
        
        # Add actual points if available
        if 'calculated_dk_fpts' in df.columns:
            results_df['Actual_Points'] = df['calculated_dk_fpts']
            results_df['Prediction_Error'] = results_df['Predicted_Points'] - results_df['Actual_Points']
        
        # Sort by predicted points (descending)
        results_df = results_df.sort_values('Predicted_Points', ascending=False)
        
        # Save report if path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"📁 Report saved to: {output_path}")
        
        return results_df
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """Create prediction visualization plots"""
        print("📊 Creating prediction plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title('Actual vs Predicted Points')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Points')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of actual points
        axes[1, 0].hist(y_true, bins=30, alpha=0.7, color='blue', label='Actual')
        axes[1, 0].hist(y_pred, bins=30, alpha=0.7, color='red', label='Predicted')
        axes[1, 0].set_xlabel('Fantasy Points')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Points')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Plots saved to: {save_path}")
        
        plt.show()
    
    def predict_for_date(self, date_str, data_path=None):
        """
        Make predictions for a specific date
        
        Args:
            date_str: Date in format 'YYYY-MM-DD'
            data_path: Path to the data file (optional)
        
        Returns:
            Dictionary with predictions and analysis
        """
        print(f"🗓️  Loading data for date: {date_str}")
        
        # Default data path
        if data_path is None:
            data_path = '4_DATA/filtered_data.csv'
        
        try:
            # Load the full dataset
            df = pd.read_csv(data_path)
            print(f"📊 Total dataset size: {len(df)} records")
            
            # Filter for the specific date
            date_column = None
            if 'date' in df.columns:
                date_column = 'date'
            elif 'Date' in df.columns:
                date_column = 'Date'
            else:
                print("❌ No date column found in the dataset")
                return None
            
            # Convert date column to string for comparison
            df[date_column] = df[date_column].astype(str)
            df_date = df[df[date_column] == date_str].copy()
            
            if len(df_date) == 0:
                print(f"❌ No data found for date: {date_str}")
                # Get some sample dates to help with debugging
                unique_dates = df[date_column].unique()
                print(f"Available dates sample: {sorted(unique_dates)[:10]}")
                print(f"Total unique dates: {len(unique_dates)}")
                return None
            
            print(f"✅ Found {len(df_date)} player records for {date_str}")
            
            # Prepare features
            X_date = self.prepare_features(df_date)
            
            # Make predictions
            predictions = self.predict(X_date)
            
            # Calculate probabilities
            probabilities = self.predict_probabilities(X_date)
            
            # Create report
            report = self.create_prediction_report(df_date, predictions, probabilities)
            
            # Save predictions to CSV file
            output_filename = f"predictions_{date_str}.csv"
            output_path = os.path.join(os.getcwd(), output_filename)
            report.to_csv(output_path, index=False)
            print(f"📁 Predictions saved to: {output_path}")
            
            # Check if we have actual results for evaluation
            actual_points = None
            if 'dk_fpts' in df_date.columns:
                actual_points = df_date['dk_fpts'].values
                # Remove any NaN values
                valid_mask = ~np.isnan(actual_points)
                if valid_mask.sum() > 0:
                    actual_points = actual_points[valid_mask]
                    predictions_filtered = predictions[valid_mask]
                    results = self.evaluate_predictions(actual_points, predictions_filtered)
                else:
                    results = None
            elif 'calculated_dk_fpts' in df_date.columns:
                actual_points = df_date['calculated_dk_fpts'].values
                # Remove any NaN values
                valid_mask = ~np.isnan(actual_points)
                if valid_mask.sum() > 0:
                    actual_points = actual_points[valid_mask]
                    predictions_filtered = predictions[valid_mask]
                    results = self.evaluate_predictions(actual_points, predictions_filtered)
                else:
                    results = None
            else:
                results = None
            
            # Return comprehensive results
            return {
                'date': date_str,
                'player_count': len(df_date),
                'predictions': predictions,
                'probabilities': probabilities,
                'report': report,
                'evaluation': results,
                'raw_data': df_date
            }
            
        except Exception as e:
            print(f"❌ Error processing date {date_str}: {e}")
            return None

def main():
    """Main prediction function"""
    print("🚀 HP Omen 35L Optimized Prediction System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = OptimizedPredictor()
    
    # Load test data (using the processed training data as an example)
    print("\n📂 Loading test data...")
    processed_data_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/processed_training_data.joblib'
    processed_data = joblib.load(processed_data_path)
    
    # For demonstration, we'll use a subset of the training data
    # In practice, you would load new, unseen data
    X_test = processed_data['X'][:1000]  # First 1000 samples
    y_test = processed_data['y'][:1000]
    
    print(f"✅ Test data loaded: {X_test.shape[0]} samples")
    
    # Make predictions directly on preprocessed data
    print("\n🚀 Making predictions...")
    predictions = predictor.model.predict(X_test)
    predictions = np.clip(predictions, 0, 100)
    
    # Evaluate predictions
    results = predictor.evaluate_predictions(y_test, predictions)
    
    # Create visualization
    print("\n📊 Creating visualizations...")
    plot_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/prediction_plots.png'
    predictor.plot_predictions(y_test, predictions, plot_path)
    
    # Summary
    print("\n🎯 Prediction Summary:")
    print("=" * 50)
    print(f"🔧 Model Type: {predictor.metadata['model_type']}")
    print(f"📊 Test Samples: {len(predictions)}")
    print(f"📈 Mean Absolute Error: {results['mae']:.3f}")
    print(f"📉 R² Score: {results['r2']:.3f}")
    print(f"🎯 Prediction Range: {predictions.min():.1f} to {predictions.max():.1f}")
    print("=" * 50)
    
    print("\n✅ Prediction analysis complete!")

if __name__ == "__main__":
    main()
