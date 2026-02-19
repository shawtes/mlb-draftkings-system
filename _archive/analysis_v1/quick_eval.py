#!/usr/bin/env python3
"""
Quick Prediction Evaluation Demo
================================

This script demonstrates how to quickly evaluate predictions for specific dates
using the MLB prediction system.

Usage examples:
- python quick_eval.py --date 2024-06-15
- python quick_eval.py --recent-days 7
- python quick_eval.py --demo
"""

import os
import sys
from datetime import datetime, timedelta
import argparse

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_predictions import PredictionEvaluator

def demo_evaluation():
    """Run a demonstration of the evaluation system"""
    print("🎯 MLB Prediction Evaluation Demo")
    print("=" * 50)
    
    evaluator = PredictionEvaluator()
    
    # Try a few sample dates that should be available
    sample_dates = [
        "2017-05-19",
        "2017-06-15", 
        "2017-07-01",
        "2017-08-15"
    ]
    
    successful_evaluations = []
    
    for date_str in sample_dates:
        print(f"\n🔍 Testing evaluation for {date_str}")
        result = evaluator.evaluate_date(date_str)
        
        if result is not None:
            successful_evaluations.append(result)
            print(f"✅ Successfully evaluated {date_str}")
        else:
            print(f"❌ No data available for {date_str}")
    
    # Generate summary if we have results
    if successful_evaluations:
        print(f"\n📊 Demo completed with {len(successful_evaluations)} successful evaluations")
        
        # Create summary report
        summary = evaluator.generate_summary_report(successful_evaluations)
        evaluator.display_summary_report(summary)
        
        # Create plots
        evaluator.create_evaluation_plots(successful_evaluations)
        
    else:
        print("\n❌ No successful evaluations in demo")
        print("This might indicate:")
        print("1. No data available for the sample dates")
        print("2. Model not properly trained")
        print("3. Data path issues")

def evaluate_recent_days(days: int):
    """Evaluate predictions for the last N days"""
    print(f"🔍 Evaluating predictions for the last {days} days")
    print("=" * 50)
    
    evaluator = PredictionEvaluator()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Date range: {start_date_str} to {end_date_str}")
    
    # Evaluate the range
    results = evaluator.evaluate_date_range(start_date_str, end_date_str)
    
    if results:
        summary = evaluator.generate_summary_report(results)
        evaluator.display_summary_report(summary)
        evaluator.create_evaluation_plots(results)
    else:
        print("❌ No successful evaluations found")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Quick Prediction Evaluation")
    parser.add_argument('--date', type=str, help='Specific date to evaluate (YYYY-MM-DD)')
    parser.add_argument('--recent-days', type=int, help='Evaluate last N days')
    parser.add_argument('--demo', action='store_true', help='Run demo evaluation')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_evaluation()
    elif args.date:
        evaluator = PredictionEvaluator()
        evaluator.evaluate_date(args.date)
    elif args.recent_days:
        evaluate_recent_days(args.recent_days)
    else:
        print("🚀 Quick Prediction Evaluation")
        print("=" * 40)
        print("Usage:")
        print("  python quick_eval.py --date 2024-06-15")
        print("  python quick_eval.py --recent-days 7")
        print("  python quick_eval.py --demo")
        print("\nFor more options, use: python evaluate_predictions.py --help")

if __name__ == "__main__":
    main()
