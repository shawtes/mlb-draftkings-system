import pandas as pd
import numpy as np
import joblib
from scipy import stats
from sklearn.linear_model import QuantileRegressor
import warnings
warnings.filterwarnings('ignore')

class ProbabilityPredictor:
    """
    A class to predict probabilities of achieving different point thresholds
    using quantile regression and distribution modeling.
    """
    def __init__(self, point_thresholds=None, quantiles=None):
        if point_thresholds is None:
            # Common DFS point thresholds
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
        
    def predict_probabilities_simple(self, predictions, std_dev=None):
        """
        Simple probability prediction using normal distribution assumption
        """
        if std_dev is None:
            # Use a default standard deviation based on typical DFS point variance
            std_dev = np.std(predictions) if len(predictions) > 1 else 5.0
            
        results = []
        
        for pred in predictions:
            player_probs = {'main_prediction': pred}
            
            for threshold in self.point_thresholds:
                if std_dev > 0:
                    # Probability of exceeding threshold using normal approximation
                    z_score = (threshold - pred) / std_dev
                    prob_exceed = 1 - stats.norm.cdf(z_score)
                    player_probs[f'prob_over_{threshold}'] = max(0, min(1, prob_exceed))
                else:
                    # If no variance, use deterministic approach
                    player_probs[f'prob_over_{threshold}'] = 1.0 if pred > threshold else 0.0
                    
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
                    
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

def add_probabilities_to_merged_projections():
    """
    Add probability predictions to the existing merged projections
    """
    print("Loading merged projections...")
    
    # Load the merged projections
    try:
        df = pd.read_csv('4_DATA/merged_player_projections.csv')
        print(f"Loaded {len(df)} player projections")
        print("Columns:", df.columns.tolist())
    except FileNotFoundError:
        print("Error: merged_player_projections.csv not found. Please run the merge script first.")
        return
    
    # Check if we have the ML_Prediction column
    if 'ML_Prediction' not in df.columns:
        print("Error: ML_Prediction column not found in merged projections.")
        return
    
    # Handle NaN values in ML_Prediction by filling with PPG_Projection
    if 'PPG_Projection' in df.columns:
        df['ML_Prediction'] = df['ML_Prediction'].fillna(df['PPG_Projection'])
    
    # Remove rows where both predictions are NaN
    df = df.dropna(subset=['ML_Prediction'])
    print(f"After removing NaN predictions: {len(df)} players")
    
    # Get predictions
    predictions = df['ML_Prediction'].values
    player_names = df['Name'].values
    
    # Initialize probability predictor
    prob_predictor = ProbabilityPredictor()
    
    # Calculate standard deviation from the predictions for normal distribution assumption
    pred_std = np.std(predictions)
    print(f"Using standard deviation of {pred_std:.2f} for probability calculations")
    
    # Get probability predictions using simple method
    probability_results = prob_predictor.predict_probabilities_simple(predictions, pred_std)
    
    # Create probability summary
    prob_summary = prob_predictor.create_probability_summary(probability_results, player_names)
    
    # Merge with original dataframe
    enhanced_df = df.merge(prob_summary, on='Name', how='left')
    
    # Save enhanced projections
    enhanced_df.to_csv('4_DATA/enhanced_projections_with_probabilities.csv', index=False)
    print("Enhanced projections with probabilities saved!")
    
    # Display sample results
    print("\nSample Probability Predictions:")
    print("="*100)
    sample_df = enhanced_df.head(10)
    for _, row in sample_df.iterrows():
        if pd.notna(row.get('Predicted_Points')):
            print(f"\nPlayer: {row['Name']} ({row.get('Pos', 'Unknown')})")
            print(f"Predicted Points: {row['Predicted_Points']:.1f}")
            print("Probability of exceeding thresholds:")
            for threshold in [10, 20, 30, 40]:
                prob_col = f'Prob_Over_{threshold}'
                if prob_col in row and pd.notna(row[prob_col]):
                    print(f"  > {threshold} points: {row[prob_col]}")
    print("="*100)
    
    # Create a simplified version for DFS use with available columns
    available_columns = ['Name']
    if 'Pos' in df.columns:
        available_columns.append('Pos')
    if 'Team' in df.columns:
        available_columns.append('Team')
    if 'Salary' in df.columns:
        available_columns.append('Salary')
    if 'PPG_Projection' in df.columns:
        available_columns.append('PPG_Projection')
    
    available_columns.extend(['ML_Prediction', 'Predicted_Points'])
    available_columns.extend([f'Prob_Over_{t}' for t in [10, 20, 30, 40, 50]])
    
    # Only include columns that exist in the dataframe
    dfs_columns = [col for col in available_columns if col in enhanced_df.columns]
    dfs_df = enhanced_df[dfs_columns].copy()
    
    # Sort by predicted points
    dfs_df = dfs_df.sort_values('Predicted_Points', ascending=False)
    
    dfs_df.to_csv('4_DATA/dfs_projections_with_probabilities.csv', index=False)
    print("DFS-ready projections with probabilities saved!")
    
    return enhanced_df

def analyze_probability_distribution():
    """
    Analyze the probability distribution of predictions
    """
    try:
        df = pd.read_csv('4_DATA/enhanced_projections_with_probabilities.csv')
    except FileNotFoundError:
        print("Enhanced projections file not found. Running probability prediction first...")
        add_probabilities_to_merged_projections()
        df = pd.read_csv('4_DATA/enhanced_projections_with_probabilities.csv')
    
    print("\nProbability Distribution Analysis:")
    print("="*60)
    
    # Analysis by position (use Pos column if available)
    pos_col = 'Pos' if 'Pos' in df.columns else 'position'
    if pos_col in df.columns:
        for pos in df[pos_col].unique():
            if pd.notna(pos):
                pos_df = df[df[pos_col] == pos]
                if 'Predicted_Points' in pos_df.columns:
                    avg_pred = pos_df['Predicted_Points'].mean()
                    print(f"\n{pos} (n={len(pos_df)}):")
                    print(f"  Average Prediction: {avg_pred:.1f}")
                    
                    # Show average probabilities for key thresholds
                    for threshold in [20, 30, 40]:
                        prob_col = f'Prob_Over_{threshold}'
                        if prob_col in pos_df.columns:
                            # Convert percentage string back to float for calculation
                            probs = pos_df[prob_col].apply(lambda x: float(x.strip('%')) / 100 if pd.notna(x) and isinstance(x, str) and x.strip() != '' else 0)
                            avg_prob = probs.mean()
                            print(f"  Avg Prob > {threshold}: {avg_prob:.1%}")
    
    # Overall statistics
    if 'Predicted_Points' in df.columns:
        print(f"\nOverall Statistics:")
        print(f"  Total Players: {len(df)}")
        print(f"  Average Prediction: {df['Predicted_Points'].mean():.1f}")
        print(f"  Std Dev: {df['Predicted_Points'].std():.1f}")
        print(f"  Min: {df['Predicted_Points'].min():.1f}")
        print(f"  Max: {df['Predicted_Points'].max():.1f}")
    
    print("="*60)

if __name__ == "__main__":
    print("Adding probability predictions to merged projections...")
    
    # Add probabilities to merged projections
    enhanced_df = add_probabilities_to_merged_projections()
    
    if enhanced_df is not None:
        # Analyze probability distributions
        analyze_probability_distribution()
        
        print("\nFiles created:")
        print("- enhanced_projections_with_probabilities.csv (full data)")
        print("- dfs_projections_with_probabilities.csv (DFS-ready)")
        
        print("\nHow to interpret the probabilities:")
        print("- Prob_Over_X shows the probability that a player will score more than X points")
        print("- Higher percentages indicate more likely outcomes")
        print("- Use these probabilities to assess risk vs reward in DFS lineups")
