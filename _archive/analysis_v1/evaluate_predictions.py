#!/usr/bin/env python3
"""
MLB Prediction Evaluation Interface
===================================

This script provides a comprehensive interface for evaluating MLB predictions
for any specific date or date range. It uses the OptimizedPredictor class
to make predictions and provides detailed evaluation metrics.

Author: MLB Prediction System
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_prediction import OptimizedPredictor

class PredictionEvaluator:
    """
    Comprehensive prediction evaluation interface
    """
    
    def __init__(self):
        """Initialize the prediction evaluator"""
        self.predictor = OptimizedPredictor()
        self.data_path = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
        
    def evaluate_date(self, date_str: str) -> Dict:
        """
        Evaluate predictions for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n🔍 Evaluating predictions for {date_str}")
        print("=" * 50)
        
        try:
            # Use the existing predict_for_date method
            result = self.predictor.predict_for_date(date_str)
            
            if result is None:
                print(f"❌ No data available for {date_str}")
                return None
                
            # Extract evaluation metrics
            evaluation = result.get('evaluation')
            predictions = result['predictions']
            
            if evaluation is None:
                print(f"✅ Successfully processed {len(predictions)} predictions")
                print(f"⚠️  No actual DK points available for comparison")
                print(f"🔢 Prediction Range: {predictions.min():.1f} to {predictions.max():.1f}")
                
                # Additional statistics
                print(f"\n📈 Prediction Statistics:")
                print(f"   Mean Prediction: {predictions.mean():.2f}")
                print(f"   Median Prediction: {np.median(predictions):.2f}")
                print(f"   Standard Deviation: {predictions.std():.2f}")
                
                return result
            
            # Display results with evaluation metrics
            print(f"✅ Successfully processed {len(predictions)} predictions")
            print(f"📊 Mean Absolute Error: {evaluation['mae']:.3f}")
            print(f"📈 Root Mean Square Error: {evaluation['rmse']:.3f}")
            print(f"🎯 R² Score: {evaluation['r2']:.3f}")
            print(f"📉 Mean Squared Error: {evaluation['mse']:.3f}")
            print(f"🔢 Prediction Range: {predictions.min():.1f} to {predictions.max():.1f}")
            
            # Additional statistics
            print(f"\n📈 Additional Statistics:")
            print(f"   Mean Prediction: {predictions.mean():.2f}")
            print(f"   Median Prediction: {np.median(predictions):.2f}")
            print(f"   Standard Deviation: {predictions.std():.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating date {date_str}: {e}")
            return None
    
    def evaluate_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Evaluate predictions for a date range
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of evaluation results
        """
        print(f"\n🔍 Evaluating predictions from {start_date} to {end_date}")
        print("=" * 60)
        
        results = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        total_days = (end_date_obj - current_date).days + 1
        processed_days = 0
        successful_days = 0
        
        while current_date <= end_date_obj:
            date_str = current_date.strftime('%Y-%m-%d')
            result = self.evaluate_date(date_str)
            
            if result is not None:
                results.append(result)
                successful_days += 1
            
            processed_days += 1
            current_date += timedelta(days=1)
            
            # Progress update
            if processed_days % 10 == 0:
                print(f"📊 Progress: {processed_days}/{total_days} days processed")
        
        print(f"\n✅ Range evaluation complete:")
        print(f"   📅 Total days: {total_days}")
        print(f"   ✅ Successful predictions: {successful_days}")
        print(f"   ❌ Failed predictions: {total_days - successful_days}")
        
        return results
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate a summary report from multiple evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        # Aggregate metrics (only for results with evaluation data)
        results_with_eval = [r for r in results if r.get('evaluation') is not None]
        
        if not results_with_eval:
            return {
                'total_evaluations': len(results),
                'evaluations_with_metrics': 0,
                'note': 'No evaluation metrics available (no actual DK points for comparison)'
            }
        
        all_mae = [r['evaluation']['mae'] for r in results_with_eval]
        all_rmse = [r['evaluation']['rmse'] for r in results_with_eval]
        all_r2 = [r['evaluation']['r2'] for r in results_with_eval]
        all_mse = [r['evaluation']['mse'] for r in results_with_eval]
        
        # Calculate summary statistics
        summary = {
            'total_evaluations': len(results),
            'evaluations_with_metrics': len(results_with_eval),
            'mae_stats': {
                'mean': np.mean(all_mae),
                'median': np.median(all_mae),
                'std': np.std(all_mae),
                'min': np.min(all_mae),
                'max': np.max(all_mae)
            },
            'rmse_stats': {
                'mean': np.mean(all_rmse),
                'median': np.median(all_rmse),
                'std': np.std(all_rmse),
                'min': np.min(all_rmse),
                'max': np.max(all_rmse)
            },
            'r2_stats': {
                'mean': np.mean(all_r2),
                'median': np.median(all_r2),
                'std': np.std(all_r2),
                'min': np.min(all_r2),
                'max': np.max(all_r2)
            }
        }
        
        return summary
    
    def display_summary_report(self, summary: Dict):
        """
        Display a formatted summary report
        
        Args:
            summary: Summary statistics dictionary
        """
        print("\n📊 SUMMARY REPORT")
        print("=" * 50)
        print(f"📈 Total Evaluations: {summary['total_evaluations']}")
        print(f"🎯 Evaluations with Metrics: {summary.get('evaluations_with_metrics', 0)}")
        
        if 'note' in summary:
            print(f"⚠️  {summary['note']}")
            print("=" * 50)
            return
        
        print()
        
        print("🎯 Mean Absolute Error (MAE):")
        mae = summary['mae_stats']
        print(f"   Mean: {mae['mean']:.3f}")
        print(f"   Median: {mae['median']:.3f}")
        print(f"   Std Dev: {mae['std']:.3f}")
        print(f"   Range: {mae['min']:.3f} to {mae['max']:.3f}")
        print()
        
        print("📈 Root Mean Square Error (RMSE):")
        rmse = summary['rmse_stats']
        print(f"   Mean: {rmse['mean']:.3f}")
        print(f"   Median: {rmse['median']:.3f}")
        print(f"   Std Dev: {rmse['std']:.3f}")
        print(f"   Range: {rmse['min']:.3f} to {rmse['max']:.3f}")
        print()
        
        print("🎯 R² Score:")
        r2 = summary['r2_stats']
        print(f"   Mean: {r2['mean']:.3f}")
        print(f"   Median: {r2['median']:.3f}")
        print(f"   Std Dev: {r2['std']:.3f}")
        print(f"   Range: {r2['min']:.3f} to {r2['max']:.3f}")
        print("=" * 50)
    
    def create_evaluation_plots(self, results: List[Dict], output_path: str = None):
        """
        Create visualization plots for evaluation results
        
        Args:
            results: List of evaluation results
            output_path: Path to save the plots
        """
        if not results:
            print("❌ No results to plot")
            return
        
        # Extract metrics for plotting
        dates = [r.get('date', f'Day {i+1}') for i, r in enumerate(results)]
        mae_values = [r['evaluation']['mae'] for r in results]
        rmse_values = [r['evaluation']['rmse'] for r in results]
        r2_values = [r['evaluation']['r2'] for r in results]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Evaluation Results', fontsize=16, fontweight='bold')
        
        # MAE over time
        axes[0, 0].plot(range(len(mae_values)), mae_values, 'b-', linewidth=2)
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[0, 0].set_xlabel('Evaluation Index')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE over time
        axes[0, 1].plot(range(len(rmse_values)), rmse_values, 'r-', linewidth=2)
        axes[0, 1].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        axes[0, 1].set_xlabel('Evaluation Index')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² over time
        axes[1, 0].plot(range(len(r2_values)), r2_values, 'g-', linewidth=2)
        axes[1, 0].set_title('R² Score', fontweight='bold')
        axes[1, 0].set_xlabel('Evaluation Index')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of MAE
        axes[1, 1].hist(mae_values, bins=min(20, len(mae_values)), alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('MAE Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/evaluation_plots.png'
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plots saved to: {output_path}")
        plt.close()

def main():
    """Main evaluation interface"""
    parser = argparse.ArgumentParser(description="MLB Prediction Evaluation Interface")
    parser.add_argument('--date', type=str, help='Specific date to evaluate (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='Start date for range evaluation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for range evaluation (YYYY-MM-DD)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PredictionEvaluator()
    
    print("🚀 MLB Prediction Evaluation System")
    print("=" * 60)
    
    if args.interactive or (not args.date and not args.start_date):
        # Interactive mode
        print("\n📋 Interactive Mode")
        print("Available options:")
        print("1. Evaluate single date")
        print("2. Evaluate date range")
        print("3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == '1':
                    date_str = input("Enter date (YYYY-MM-DD): ").strip()
                    result = evaluator.evaluate_date(date_str)
                    
                elif choice == '2':
                    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                    results = evaluator.evaluate_date_range(start_date, end_date)
                    
                    if results:
                        summary = evaluator.generate_summary_report(results)
                        evaluator.display_summary_report(summary)
                        evaluator.create_evaluation_plots(results)
                
                elif choice == '3':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.date:
        # Single date evaluation
        evaluator.evaluate_date(args.date)
    
    elif args.start_date and args.end_date:
        # Date range evaluation
        results = evaluator.evaluate_date_range(args.start_date, args.end_date)
        
        if results:
            summary = evaluator.generate_summary_report(results)
            evaluator.display_summary_report(summary)
            evaluator.create_evaluation_plots(results)
    
    else:
        print("❌ Please provide either --date or both --start-date and --end-date")
        print("   Or use --interactive for interactive mode")

if __name__ == "__main__":
    main()
