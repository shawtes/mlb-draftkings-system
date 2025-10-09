#!/usr/bin/env python3
"""
Daily MLB Prediction Tool
Easy interface to predict for any specific day using the trained model
"""

import argparse
import sys
import pandas as pd
from datetime import datetime, timedelta
from optimized_prediction import OptimizedPredictor

def format_date(date_str):
    """Format date string to YYYY-MM-DD"""
    try:
        # Try different date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%Y/%m/%d']
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

def list_available_dates(data_path):
    """List available dates in the dataset"""
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
        return dates
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description='Make MLB predictions for any specific day',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_day.py --date 2024-04-15
  python predict_day.py --date 04/15/2024
  python predict_day.py --list-dates
  python predict_day.py --recent 7
        """
    )
    
    parser.add_argument('--date', '-d', 
                       help='Date to predict for (YYYY-MM-DD, MM/DD/YYYY, etc.)')
    parser.add_argument('--list-dates', '-l', action='store_true',
                       help='List all available dates in the dataset')
    parser.add_argument('--recent', '-r', type=int, metavar='N',
                       help='Show N most recent dates available')
    parser.add_argument('--data-path', '-p', 
                       default='C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv',
                       help='Path to the data file')
    parser.add_argument('--top', '-t', type=int, default=10,
                       help='Number of top predictions to show (default: 10)')
    parser.add_argument('--save-report', '-s', 
                       help='Save prediction report to file')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print("🏆 MLB Daily Prediction Tool")
    print("=" * 50)
    
    # List available dates
    if args.list_dates or args.recent:
        print("\n📅 Available dates in dataset:")
        dates = list_available_dates(args.data_path)
        
        if not dates:
            print("❌ No dates found in dataset")
            return
        
        if args.recent:
            dates = dates[-args.recent:]
            print(f"📊 Showing {len(dates)} most recent dates:")
        
        for date in dates:
            print(f"  • {date}")
        
        if not args.date:
            return
    
    # Make prediction for specific date
    if args.date:
        formatted_date = format_date(args.date)
        if not formatted_date:
            return
        
        print(f"\n🎯 Making predictions for: {formatted_date}")
        print("-" * 50)
        
        try:
            # Initialize predictor
            predictor = OptimizedPredictor()
            
            # Make prediction
            results = predictor.predict_for_date(formatted_date, args.data_path)
            
            if results is None:
                print("❌ No predictions could be made")
                return
            
            # Display results
            print(f"\n✅ Predictions completed for {results['date']}")
            print(f"📊 Total players: {results['player_count']}")
            
            # Get top predictions
            df_results = results['raw_data'].copy()
            df_results['predicted_points'] = results['predictions']
            df_results['confidence'] = results['probabilities']
            
            # Sort by predicted points
            df_top = df_results.nlargest(args.top, 'predicted_points')
            
            print(f"\n🏆 Top {args.top} Predictions:")
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
            
            # Save report if requested
            if args.save_report:
                report_path = args.save_report
                if not report_path.endswith('.csv'):
                    report_path += '.csv'
                
                df_results.to_csv(report_path, index=False)
                print(f"\n💾 Full prediction report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
