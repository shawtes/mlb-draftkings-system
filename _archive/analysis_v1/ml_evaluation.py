#!/usr/bin/env python3
"""
Comprehensive ML Model Evaluation Script
Evaluates trained price prediction models with detailed metrics
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our modules
from maybe import get_coinbase_data, calculate_indicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model directories
MODELS_DIR = os.path.join(current_dir, 'models')

class MLModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        self.models_evaluated = 0
        
    def calculate_regression_metrics(self, y_true, y_pred, symbol, timeframe):
        """Calculate comprehensive regression metrics"""
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return None
            
            # Basic regression metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            # Percentage-based metrics
            mae_percent = np.mean(np.abs((y_true_clean - y_pred_clean) / np.abs(y_true_clean))) * 100
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
            
            # Directional accuracy (did we predict the right direction?)
            direction_actual = np.sign(y_true_clean)
            direction_predicted = np.sign(y_pred_clean)
            directional_accuracy = np.mean(direction_actual == direction_predicted) * 100
            
            # Hit rate (predictions within certain thresholds)
            within_05_pct = np.mean(np.abs(y_true_clean - y_pred_clean) <= 0.5) * 100  # Within 0.5%
            within_1_pct = np.mean(np.abs(y_true_clean - y_pred_clean) <= 1.0) * 100   # Within 1%
            within_2_pct = np.mean(np.abs(y_true_clean - y_pred_clean) <= 2.0) * 100   # Within 2%
            
            # Bias metrics
            mean_error = np.mean(y_pred_clean - y_true_clean)  # Average bias
            median_error = np.median(y_pred_clean - y_true_clean)  # Median bias
            
            # Volatility metrics
            pred_volatility = np.std(y_pred_clean)
            actual_volatility = np.std(y_true_clean)
            volatility_ratio = pred_volatility / actual_volatility if actual_volatility != 0 else np.inf
            
            # Extreme prediction analysis
            extreme_threshold = 2 * np.std(y_true_clean)
            extreme_predictions = np.sum(np.abs(y_pred_clean) > extreme_threshold)
            extreme_actual = np.sum(np.abs(y_true_clean) > extreme_threshold)
            
            # Return distribution comparison
            pred_mean = np.mean(y_pred_clean)
            actual_mean = np.mean(y_true_clean)
            pred_std = np.std(y_pred_clean)
            actual_std = np.std(y_true_clean)
            
            metrics = {
                'symbol': symbol,
                'timeframe': timeframe,
                'samples': len(y_true_clean),
                
                # Core regression metrics
                'r2_score': round(r2, 4),
                'mae': round(mae, 4),
                'rmse': round(rmse, 4),
                'mae_percent': round(mae_percent, 2),
                'mape': round(mape, 2),
                
                # Directional and hit rate metrics
                'directional_accuracy': round(directional_accuracy, 2),
                'within_05_pct': round(within_05_pct, 2),
                'within_1_pct': round(within_1_pct, 2),
                'within_2_pct': round(within_2_pct, 2),
                
                # Bias and error distribution
                'mean_bias': round(mean_error, 4),
                'median_bias': round(median_error, 4),
                'volatility_ratio': round(volatility_ratio, 4),
                
                # Distribution comparison
                'pred_mean': round(pred_mean, 4),
                'actual_mean': round(actual_mean, 4),
                'pred_std': round(pred_std, 4),
                'actual_std': round(actual_std, 4),
                
                # Extreme predictions
                'extreme_predictions': extreme_predictions,
                'extreme_actual': extreme_actual,
                
                # Model quality indicators
                'model_quality': self._assess_model_quality(r2, mae_percent, directional_accuracy),
                'trading_viability': self._assess_trading_viability(directional_accuracy, within_1_pct, r2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol} {timeframe}: {str(e)}")
            return None
    
    def _assess_model_quality(self, r2, mae_percent, directional_accuracy):
        """Assess overall model quality"""
        if r2 > 0.3 and mae_percent < 15 and directional_accuracy > 60:
            return "EXCELLENT"
        elif r2 > 0.1 and mae_percent < 25 and directional_accuracy > 55:
            return "GOOD"
        elif r2 > 0.05 and mae_percent < 35 and directional_accuracy > 52:
            return "FAIR"
        else:
            return "POOR"
    
    def _assess_trading_viability(self, directional_accuracy, within_1_pct, r2):
        """Assess if model is viable for trading"""
        if directional_accuracy > 58 and within_1_pct > 40 and r2 > 0.08:
            return "HIGHLY_VIABLE"
        elif directional_accuracy > 55 and within_1_pct > 30 and r2 > 0.05:
            return "VIABLE"
        elif directional_accuracy > 52 and within_1_pct > 20:
            return "MARGINAL"
        else:
            return "NOT_VIABLE"
    
    def evaluate_model(self, symbol, granularity=3600):
        """Evaluate all timeframe models for a symbol"""
        logger.info(f"🔍 Evaluating models for {symbol} (granularity: {granularity})")
        
        model_prefix = f"{symbol.replace('-', '')}_{granularity}"
        metadata_path = os.path.join(MODELS_DIR, f"{model_prefix}_price_metadata.pkl")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"No models found for {symbol}")
            return None
        
        try:
            # Load model components
            metadata = joblib.load(metadata_path)
            scaler = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_scaler.pkl"))
            features = joblib.load(os.path.join(MODELS_DIR, f"{model_prefix}_price_features.pkl"))
            
            # Get test data
            df = get_coinbase_data(symbol, granularity, days=60)  # 60 days for evaluation
            if df is None or df.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Calculate indicators (same as training)
            df = calculate_indicators(df, symbol=symbol)
            
            # Add price-specific features (same as training)
            df['price_ma_5'] = df['close'].rolling(window=5).mean()
            df['price_ma_10'] = df['close'].rolling(window=10).mean()
            df['price_ma_20'] = df['close'].rolling(window=20).mean()
            df['price_volatility'] = df['close'].rolling(window=20).std()
            df['price_momentum'] = df['close'].pct_change(periods=5)
            df['high_low_ratio'] = df['high'] / df['low']
            df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
            df['price_momentum_3'] = df['close'].pct_change(periods=3)
            df['price_acceleration'] = df['price_momentum'].diff()
            df['intraday_range'] = (df['high'] - df['low']) / df['close'] * 100
            df['volume_momentum'] = df['volume'].pct_change(periods=3)
            
            # Calculate target variables (same as training)
            if granularity <= 60:  # 1 minute or less
                periods_15min = 15
                periods_30min = 30
                periods_1h = 60
                periods_4h = 240
                periods_24h = 1440
            elif granularity <= 900:  # 15 minutes or less
                periods_15min = 1
                periods_30min = 2
                periods_1h = 4
                periods_4h = 16
                periods_24h = 96
            elif granularity <= 3600:  # 1 hour or less
                periods_15min = 1
                periods_30min = 1
                periods_1h = 1
                periods_4h = 4
                periods_24h = 24
            else:  # Daily or larger
                periods_15min = 1
                periods_30min = 1
                periods_1h = 1
                periods_4h = 1
                periods_24h = 1
            
            # Create targets
            df['price_15min'] = df['close'].shift(-periods_15min)
            df['price_30min'] = df['close'].shift(-periods_30min)
            df['price_1h'] = df['close'].shift(-periods_1h)
            df['price_4h'] = df['close'].shift(-periods_4h)
            df['price_24h'] = df['close'].shift(-periods_24h)
            
            df['price_change_15min'] = (df['price_15min'] - df['close']) / df['close'] * 100
            df['price_change_30min'] = (df['price_30min'] - df['close']) / df['close'] * 100
            df['price_change_1h'] = (df['price_1h'] - df['close']) / df['close'] * 100
            df['price_change_4h'] = (df['price_4h'] - df['close']) / df['close'] * 100
            df['price_change_24h'] = (df['price_24h'] - df['close']) / df['close'] * 100
            
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < 50:
                logger.warning(f"Insufficient evaluation data for {symbol}: {len(df)} rows")
                return None
            
            # Use last 20% of data for evaluation (out-of-sample)
            eval_start = int(len(df) * 0.8)
            X_eval = df[features][eval_start:]
            
            # Ensure all features are numeric and available
            available_features = [f for f in features if f in X_eval.columns]
            if len(available_features) < len(features) * 0.8:  # Need at least 80% of features
                logger.warning(f"Missing too many features for {symbol}: {len(available_features)}/{len(features)}")
                return None
            
            X_eval = X_eval[available_features].select_dtypes(include=[np.number])
            X_eval_scaled = scaler.transform(X_eval)
            
            symbol_results = {'symbol': symbol, 'granularity': granularity, 'timeframes': {}}
            
            # Evaluate each timeframe model
            for horizon in ['15min', '30min', '1h', '4h', '24h']:
                model_path = os.path.join(MODELS_DIR, f"{model_prefix}_{horizon}_price_regressor.pkl")
                target_col = f'price_change_{horizon}'
                
                if os.path.exists(model_path) and target_col in df.columns:
                    try:
                        model = joblib.load(model_path)
                        y_true = df[target_col][eval_start:].values
                        y_pred = model.predict(X_eval_scaled)
                        
                        metrics = self.calculate_regression_metrics(y_true, y_pred, symbol, horizon)
                        if metrics:
                            symbol_results['timeframes'][horizon] = metrics
                            logger.info(f"✅ {horizon}: R²={metrics['r2_score']:.3f}, MAE={metrics['mae']:.3f}%, Dir.Acc={metrics['directional_accuracy']:.1f}%")
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {horizon} model: {str(e)}")
            
            if symbol_results['timeframes']:
                self.evaluation_results[symbol] = symbol_results
                self.models_evaluated += 1
                return symbol_results
            else:
                logger.warning(f"No valid timeframe evaluations for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {str(e)}")
            return None
    
    def create_evaluation_report(self):
        """Create comprehensive evaluation report"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
        
        logger.info(f"📊 Creating evaluation report for {self.models_evaluated} symbols")
        
        # Collect all metrics into a DataFrame
        all_metrics = []
        for symbol, results in self.evaluation_results.items():
            for timeframe, metrics in results['timeframes'].items():
                all_metrics.append(metrics)
        
        if not all_metrics:
            logger.warning("No metrics to report")
            return
        
        df_metrics = pd.DataFrame(all_metrics)
        
        # Create summary statistics
        summary_stats = {
            'Total Models Evaluated': len(all_metrics),
            'Symbols Evaluated': len(self.evaluation_results),
            'Average R²': df_metrics['r2_score'].mean(),
            'Average MAE': df_metrics['mae'].mean(),
            'Average MAE%': df_metrics['mae_percent'].mean(),
            'Average Directional Accuracy': df_metrics['directional_accuracy'].mean(),
            'Models with R² > 0.1': (df_metrics['r2_score'] > 0.1).sum(),
            'Models with Dir.Acc > 55%': (df_metrics['directional_accuracy'] > 55).sum(),
            'Highly Viable Models': (df_metrics['trading_viability'] == 'HIGHLY_VIABLE').sum(),
            'Viable Models': (df_metrics['trading_viability'] == 'VIABLE').sum(),
        }
        
        # Print detailed report
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE ML MODEL EVALUATION REPORT")
        print("="*80)
        
        print("\n📊 SUMMARY STATISTICS:")
        for key, value in summary_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n📈 PERFORMANCE BY TIMEFRAME:")
        timeframe_summary = df_metrics.groupby('timeframe').agg({
            'r2_score': ['mean', 'std', 'min', 'max'],
            'mae_percent': ['mean', 'std', 'min', 'max'],
            'directional_accuracy': ['mean', 'std', 'min', 'max'],
            'within_1_pct': ['mean', 'std']
        }).round(4)
        print(timeframe_summary)
        
        print(f"\n💰 TRADING VIABILITY BY TIMEFRAME:")
        viability_summary = pd.crosstab(df_metrics['timeframe'], df_metrics['trading_viability'])
        print(viability_summary)
        
        print(f"\n🏆 TOP PERFORMING MODELS:")
        # Sort by a composite score
        df_metrics['composite_score'] = (
            df_metrics['r2_score'] * 0.3 + 
            (df_metrics['directional_accuracy'] / 100) * 0.4 + 
            (df_metrics['within_1_pct'] / 100) * 0.3
        )
        top_models = df_metrics.nlargest(10, 'composite_score')[
            ['symbol', 'timeframe', 'r2_score', 'mae_percent', 'directional_accuracy', 
             'trading_viability', 'composite_score']
        ]
        print(top_models.to_string(index=False))
        
        print(f"\n⚠️  MODELS NEEDING ATTENTION:")
        poor_models = df_metrics[df_metrics['trading_viability'] == 'NOT_VIABLE'][
            ['symbol', 'timeframe', 'r2_score', 'mae_percent', 'directional_accuracy', 'model_quality']
        ]
        if len(poor_models) > 0:
            print(poor_models.to_string(index=False))
        else:
            print("   All models show acceptable performance!")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(current_dir, f'ml_evaluation_results_{timestamp}.csv')
        df_metrics.to_csv(results_path, index=False)
        logger.info(f"💾 Detailed results saved to: {results_path}")
        
        print(f"\n💾 Detailed results saved to: {results_path}")
        print("="*80)
    
    def plot_evaluation_results(self):
        """Create evaluation plots"""
        if not self.evaluation_results:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Collect metrics
            all_metrics = []
            for symbol, results in self.evaluation_results.items():
                for timeframe, metrics in results['timeframes'].items():
                    all_metrics.append(metrics)
            
            df_metrics = pd.DataFrame(all_metrics)
            
            # Create plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('ML Model Evaluation Results', fontsize=16)
            
            # R² distribution
            sns.boxplot(data=df_metrics, x='timeframe', y='r2_score', ax=axes[0,0])
            axes[0,0].set_title('R² Score by Timeframe')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # MAE% distribution
            sns.boxplot(data=df_metrics, x='timeframe', y='mae_percent', ax=axes[0,1])
            axes[0,1].set_title('MAE% by Timeframe')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Directional accuracy
            sns.boxplot(data=df_metrics, x='timeframe', y='directional_accuracy', ax=axes[0,2])
            axes[0,2].set_title('Directional Accuracy by Timeframe')
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # R² vs Directional Accuracy scatter
            sns.scatterplot(data=df_metrics, x='r2_score', y='directional_accuracy', 
                           hue='timeframe', ax=axes[1,0])
            axes[1,0].set_title('R² vs Directional Accuracy')
            
            # Trading viability distribution
            viability_counts = df_metrics['trading_viability'].value_counts()
            axes[1,1].pie(viability_counts.values, labels=viability_counts.index, autopct='%1.1f%%')
            axes[1,1].set_title('Trading Viability Distribution')
            
            # Model quality by symbol
            quality_by_symbol = df_metrics.groupby('symbol')['r2_score'].mean().sort_values(ascending=False)
            quality_by_symbol.head(10).plot(kind='bar', ax=axes[1,2])
            axes[1,2].set_title('Top 10 Symbols by Avg R²')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(current_dir, f'ml_evaluation_plots_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Evaluation plots saved to: {plot_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping plots")
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")

def main():
    """Main evaluation function"""
    evaluator = MLModelEvaluator()
    
    # List of symbols to evaluate
    symbols_to_evaluate = [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD",
        "MATIC-USD", "LINK-USD", "UNI-USD", "AVAX-USD", "LTC-USD"
    ]
    
    logger.info(f"🚀 Starting comprehensive ML evaluation for {len(symbols_to_evaluate)} symbols")
    
    # Evaluate each symbol
    successful_evaluations = 0
    for symbol in symbols_to_evaluate:
        logger.info(f"\n📊 Evaluating {symbol}...")
        result = evaluator.evaluate_model(symbol, granularity=3600)
        if result:
            successful_evaluations += 1
        
        # Small delay to avoid overwhelming the system
        import time
        time.sleep(1)
    
    logger.info(f"\n✅ Completed evaluation: {successful_evaluations}/{len(symbols_to_evaluate)} symbols")
    
    # Create comprehensive report
    evaluator.create_evaluation_report()
    
    # Create plots if possible
    evaluator.plot_evaluation_results()
    
    return evaluator.evaluation_results

if __name__ == "__main__":
    results = main() 