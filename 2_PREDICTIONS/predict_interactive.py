#!/usr/bin/env python3
"""
Interactive MLB Prediction Tool
Simple interactive interface to predict for any specific day
"""

import pandas as pd
from datetime import datetime, timedelta
from optimized_prediction import OptimizedPredictor

def format_date(date_str):
    """Format date string to YYYY-MM-DD"""
    try:
        # Try different date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%Y/%m/%d', '%m/%d/%y']
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format works, raise error
        raise ValueError(f"Date format not recognized: {date_str}")
    except Exception as e:
        print(f"❌ Error formatting date: {e}")
        return None

def show_recent_dates(data_path, n=10):
    """Show recent available dates"""
    try:
        df = pd.read_csv(data_path)
        
        # Find the date column
        date_col = None
        for col in ['date', 'Date', 'DATE']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            print("❌ No date column found in dataset")
            return []
        
        # Get unique dates and sort them
        dates = sorted(df[date_col].dropna().unique())
        recent_dates = dates[-n:]
        
        print(f"\n📅 {len(recent_dates)} Most Recent Available Dates:")
        for i, date in enumerate(recent_dates, 1):
            print(f"  {i:2d}. {date}")
        
        return recent_dates
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return []

def main():
    print("🏆 MLB Interactive Prediction Tool")
    print("=" * 50)
    print("Enter a date to get MLB predictions for that day")
    print("Supported formats: YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY, etc.")
    print("Type 'recent' to see recent dates, or 'quit' to exit")
    print()
    
    data_path = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    predictor = None
    
    while True:
        try:
            user_input = input("📅 Enter date (or 'recent'/'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'recent':
                show_recent_dates(data_path)
                continue
            
            if not user_input:
                print("❌ Please enter a date")
                continue
            
            # Format the date
            formatted_date = format_date(user_input)
            if not formatted_date:
                continue
            
            print(f"\n🎯 Making predictions for: {formatted_date}")
            print("-" * 50)
            
            # Initialize predictor if needed
            if predictor is None:
                print("🔧 Loading prediction model...")
                predictor = OptimizedPredictor()
            
            # Make prediction
            results = predictor.predict_for_date(formatted_date, data_path)
            
            if results is None:
                print("❌ No predictions could be made")
                continue
            
            # Display results
            print(f"\n✅ Predictions completed for {results['date']}")
            print(f"📊 Total players: {results['player_count']}")
            
            # Get top predictions
            df_results = results['raw_data'].copy()
            df_results['predicted_points'] = results['predictions']
            df_results['confidence'] = results['probabilities']
            
            # Sort by predicted points
            df_top = df_results.nlargest(10, 'predicted_points')
            
            print(f"\n🏆 Top 10 Predictions:")
            print("-" * 70)
            
            for i, (idx, row) in enumerate(df_top.iterrows(), 1):
                name = row.get('Name', row.get('name', 'Unknown'))
                team = row.get('Team', row.get('team', 'N/A'))
                position = row.get('Position', row.get('position', 'N/A'))
                predicted = row['predicted_points']
                confidence = row['confidence']
                
                print(f"{i:2d}. {name} ({team} - {position})")
                print(f"    Predicted Points: {predicted:.2f}")
                
                # Handle confidence (might be dict or single value)
                if isinstance(confidence, dict):
                    # Get the highest confidence value
                    max_conf = max(confidence.values()) if confidence else 0
                    print(f"    Max Confidence: {max_conf:.2f}")
                else:
                    print(f"    Confidence: {confidence:.2f}")
                    if confidence > 1:
                        print(f"    Confidence %: {confidence:.1f}%")
                    else:
                        print(f"    Confidence %: {confidence:.1%}")
                
                # Show actual points if available
                if 'calculated_dk_fpts' in row and pd.notna(row['calculated_dk_fpts']):
                    actual = row['calculated_dk_fpts']
                    diff = actual - predicted
                    print(f"    Actual Points: {actual:.2f} (diff: {diff:+.2f})")
                
                print()
            
            # Show evaluation metrics if available
            if results['evaluation']:
                eval_metrics = results['evaluation']
                print("📈 Prediction Accuracy:")
                print(f"  • Mean Absolute Error: {eval_metrics['mae']:.2f}")
                print(f"  • Root Mean Square Error: {eval_metrics['rmse']:.2f}")
                print(f"  • R² Score: {eval_metrics['r2']:.3f}")
            
            # Ask if user wants to save results
            save_choice = input("\n💾 Save predictions to CSV? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                filename = f"predictions_{formatted_date.replace('-', '_')}.csv"
                df_results.to_csv(filename, index=False)
                print(f"✅ Predictions saved to: {filename}")
            
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different date")

if __name__ == "__main__":
    main()
