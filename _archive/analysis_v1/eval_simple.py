#!/usr/bin/env python3
"""
Simple MLB Prediction Evaluation Interface
==========================================

This script provides a simple command-line interface for evaluating MLB predictions.
Just run it and follow the prompts to get evaluation results for any date.

Usage:
    python eval_simple.py
"""

import os
import sys
from datetime import datetime, timedelta

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_predictions import PredictionEvaluator

def main():
    """Simple interactive evaluation interface"""
    print("🚀 MLB Prediction Evaluation")
    print("=" * 40)
    print("This tool evaluates MLB predictions for any date.")
    print("Available data: 2005-2017 (sample dates)")
    print()
    
    # Initialize evaluator
    try:
        evaluator = PredictionEvaluator()
        print("✅ Prediction system loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading prediction system: {e}")
        return
    
    while True:
        print("\n" + "=" * 40)
        print("Choose an option:")
        print("1. Evaluate specific date")
        print("2. Evaluate recent days")
        print("3. Quick demo")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n📅 Evaluate Specific Date")
                print("Enter a date (YYYY-MM-DD format)")
                print("Example: 2017-05-19")
                date_str = input("Date: ").strip()
                
                if not date_str:
                    print("❌ Please enter a valid date")
                    continue
                
                print(f"\n🔍 Evaluating predictions for {date_str}...")
                result = evaluator.evaluate_date(date_str)
                
                if result:
                    print(f"\n✅ Evaluation completed successfully!")
                    
                    # Ask if user wants to see the detailed report
                    show_details = input("\nShow detailed prediction report? (y/n): ").strip().lower()
                    if show_details == 'y':
                        report = result.get('report')
                        if report is not None:
                            print(f"\n📋 Top 10 Predictions:")
                            print(report.head(10).to_string(index=False))
                
            elif choice == '2':
                print("\n📅 Evaluate Recent Days")
                print("Enter number of days to evaluate (from most recent available data)")
                
                try:
                    days = int(input("Number of days: ").strip())
                    if days <= 0:
                        print("❌ Please enter a positive number")
                        continue
                except ValueError:
                    print("❌ Please enter a valid number")
                    continue
                
                # For demo purposes, use dates from 2017
                base_date = datetime(2017, 8, 15)
                start_date = base_date - timedelta(days=days)
                
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = base_date.strftime('%Y-%m-%d')
                
                print(f"\n🔍 Evaluating predictions from {start_date_str} to {end_date_str}...")
                results = evaluator.evaluate_date_range(start_date_str, end_date_str)
                
                if results:
                    summary = evaluator.generate_summary_report(results)
                    evaluator.display_summary_report(summary)
                    evaluator.create_evaluation_plots(results)
                
            elif choice == '3':
                print("\n🎯 Quick Demo")
                print("Running evaluation on sample dates...")
                
                sample_dates = ["2017-05-19", "2017-06-15", "2017-07-01"]
                successful_evaluations = []
                
                for date_str in sample_dates:
                    print(f"\n📅 Evaluating {date_str}...")
                    result = evaluator.evaluate_date(date_str)
                    
                    if result:
                        successful_evaluations.append(result)
                        print(f"✅ {date_str} - SUCCESS")
                    else:
                        print(f"❌ {date_str} - FAILED")
                
                if successful_evaluations:
                    print(f"\n📊 Demo Summary:")
                    print(f"   Successfully evaluated {len(successful_evaluations)} dates")
                    
                    summary = evaluator.generate_summary_report(successful_evaluations)
                    evaluator.display_summary_report(summary)
                    
                    # Create plots
                    evaluator.create_evaluation_plots(successful_evaluations)
                    
            elif choice == '4':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
