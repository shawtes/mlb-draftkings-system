#!/usr/bin/env python3
"""
MLB Prediction Debugging and Testing Script
Tests the improved prediction logic with realistic constraints
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prediction_ranges():
    """Test that predictions are within realistic ranges"""
    print("🏀 MLB PREDICTION TESTING SCRIPT")
    print("=" * 60)
    
    # Load your prediction script
    try:
        from predction01 import predict_unseen_data
        print("✅ Successfully imported prediction module")
    except ImportError as e:
        print(f"❌ Error importing prediction module: {e}")
        return
    
    # Test with the current data
    input_file = '4_DATA/filtered_data.csv'
    model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline01.pkl'
    prediction_date = '2025-07-05'
    
    print(f"\n📊 Testing predictions for {prediction_date}")
    print("-" * 40)
    
    try:
        # Run predictions
        predictions = predict_unseen_data(input_file, model_file, prediction_date)
        
        if predictions is not None and 'predicted_dk_fpts' in predictions.columns:
            pred_stats = predictions['predicted_dk_fpts'].describe()
            print("\n📈 PREDICTION STATISTICS:")
            print(pred_stats)
            
            # Check for unrealistic values
            high_predictions = predictions[predictions['predicted_dk_fpts'] > 35]
            extreme_predictions = predictions[predictions['predicted_dk_fpts'] > 50]
            
            print(f"\n🔍 OUTLIER ANALYSIS:")
            print(f"Total predictions: {len(predictions)}")
            print(f"Predictions > 35 points: {len(high_predictions)} ({len(high_predictions)/len(predictions)*100:.1f}%)")
            print(f"Predictions > 50 points: {len(extreme_predictions)} ({len(extreme_predictions)/len(predictions)*100:.1f}%)")
            
            if len(high_predictions) > 0:
                print(f"\nPlayers with predictions > 35 points:")
                print(high_predictions[['Name', 'predicted_dk_fpts']].head(10))
            
            # Show distribution
            print(f"\n📊 PREDICTION DISTRIBUTION:")
            bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
            pred_dist = pd.cut(predictions['predicted_dk_fpts'], bins=bins, include_lowest=True)
            print(pred_dist.value_counts().sort_index())
            
            # Sample of reasonable predictions
            reasonable_preds = predictions[
                (predictions['predicted_dk_fpts'] >= 5) & 
                (predictions['predicted_dk_fpts'] <= 25)
            ].head(10)
            
            print(f"\n✅ SAMPLE OF REASONABLE PREDICTIONS (5-25 points):")
            if len(reasonable_preds) > 0:
                print(reasonable_preds[['Name', 'predicted_dk_fpts', 'has_historical_data']])
            else:
                print("No predictions in the 5-25 range found")
                
        else:
            print("❌ No predictions generated or missing predicted_dk_fpts column")
            
    except Exception as e:
        print(f"❌ Error running predictions: {e}")
        import traceback
        traceback.print_exc()

def analyze_model_outputs():
    """Analyze raw model outputs vs constrained outputs"""
    print("\n🔬 MODEL OUTPUT ANALYSIS")
    print("=" * 60)
    
    # This would require access to your model pipeline
    # You can add this analysis to your main prediction script
    print("To analyze model outputs:")
    print("1. Add debug prints in process_predictions() to show raw vs constrained predictions")
    print("2. Check if model is outputting extreme values that need better scaling")
    print("3. Consider retraining model if raw outputs are consistently unrealistic")

def recommendations():
    """Provide recommendations for improving predictions"""
    print("\n💡 RECOMMENDATIONS FOR BETTER PREDICTIONS")
    print("=" * 60)
    
    recommendations = [
        "1. 📊 Data Quality: Ensure training data doesn't have outliers > 40 points",
        "2. 🎯 Target Engineering: Consider log-transforming target variable during training",
        "3. 🔧 Feature Scaling: Ensure all features are properly scaled/normalized",
        "4. 📈 Model Validation: Use cross-validation with realistic scoring metrics",
        "5. ⚙️ Hyperparameters: Tune model to minimize extreme predictions",
        "6. 🎲 Ensemble Methods: Use median instead of mean for ensemble predictions",
        "7. 📋 Post-processing: Apply position-specific caps (e.g., catchers typically score less)",
        "8. 🏟️ Context Features: Add ballpark factors, weather, matchup difficulty",
        "9. 📊 Confidence Intervals: Use prediction intervals instead of point estimates",
        "10. 🔄 Regular Updates: Retrain model monthly with recent data"
    ]
    
    for rec in recommendations:
        print(rec)

def create_improved_constraints():
    """Show examples of position-specific and context-aware constraints"""
    print("\n🎯 POSITION-SPECIFIC CONSTRAINTS")
    print("=" * 60)
    
    position_caps = {
        'C': {'typical_max': 25, 'absolute_max': 35},      # Catchers
        '1B': {'typical_max': 30, 'absolute_max': 40},     # First Base
        '2B': {'typical_max': 25, 'absolute_max': 35},     # Second Base  
        '3B': {'typical_max': 30, 'absolute_max': 40},     # Third Base
        'SS': {'typical_max': 25, 'absolute_max': 35},     # Shortstop
        'OF': {'typical_max': 30, 'absolute_max': 40},     # Outfield
        'DH': {'typical_max': 35, 'absolute_max': 45},     # Designated Hitter
    }
    
    print("Suggested position-based caps:")
    for pos, caps in position_caps.items():
        print(f"  {pos}: Typical max {caps['typical_max']}, Absolute max {caps['absolute_max']}")
    
    print("\n📊 You can implement this by adding position-based logic in apply_smart_prediction_constraints()")

if __name__ == "__main__":
    test_prediction_ranges()
    analyze_model_outputs()
    recommendations()
    create_improved_constraints()
    
    print("\n🏁 TESTING COMPLETE")
    print("=" * 60)
    print("Run your predictions again and check if the extreme values (100.0) are reduced!")
