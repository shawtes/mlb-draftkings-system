import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from rl_team_selector import MLBRLTeamSelector, MLBTeamSelectionEnvironment, DQNAgent
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardRLValidator:
    """
    Walk-forward validation for RL-based team selection
    
    This class implements a time-series walk-forward validation approach where:
    1. Train on historical data up to date T
    2. Predict optimal lineup for date T+1
    3. Evaluate against actual performance
    4. Move forward and repeat
    """
    
    def __init__(self, data_path: str, initial_train_days: int = 365,
                 validation_window: int = 1, retrain_frequency: int = 7):
        """
        Initialize walk-forward validator
        
        Args:
            data_path: Path to MLB data CSV
            initial_train_days: Number of days for initial training
            validation_window: Number of days to validate on each step
            retrain_frequency: How often to retrain the model (in days)
        """
        self.data_path = data_path
        self.initial_train_days = initial_train_days
        self.validation_window = validation_window
        self.retrain_frequency = retrain_frequency
        
        self.data_df = None
        self.validation_results = []
        self.performance_history = []
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the MLB data"""
        logger.info("Loading MLB data for walk-forward validation...")
        
        # Load data
        self.data_df = pd.read_csv(self.data_path, low_memory=False)
        
        # Convert date column
        self.data_df['date'] = pd.to_datetime(self.data_df['date'], errors='coerce')
        self.data_df = self.data_df.dropna(subset=['date'])
        
        # Calculate fantasy points if not present
        if 'calculated_dk_fpts' not in self.data_df.columns:
            self.data_df['calculated_dk_fpts'] = self._calculate_dk_fpts(self.data_df)
        
        # Remove rows with missing critical data
        self.data_df = self.data_df.dropna(subset=['Name', 'calculated_dk_fpts'])
        
        # Sort by date
        self.data_df = self.data_df.sort_values('date')
        
        # Get unique dates for validation
        self.unique_dates = sorted(self.data_df['date'].unique())
        
        logger.info(f"Data loaded: {len(self.data_df)} rows, {len(self.data_df['Name'].unique())} unique players")
        logger.info(f"Date range: {self.data_df['date'].min()} to {self.data_df['date'].max()}")
        logger.info(f"Unique dates: {len(self.unique_dates)}")
        
    def _calculate_dk_fpts(self, df):
        """Calculate DraftKings fantasy points"""
        return (df.get('1B', 0) * 3 + df.get('2B', 0) * 5 + df.get('3B', 0) * 8 + 
                df.get('HR', 0) * 10 + df.get('RBI', 0) * 2 + df.get('R', 0) * 2 + 
                df.get('BB', 0) * 2 + df.get('HBP', 0) * 2 + df.get('SB', 0) * 5)
    
    def get_baseline_performance(self, validation_date: datetime, method: str = 'random') -> Dict:
        """
        Get baseline performance for comparison
        
        Args:
            validation_date: Date to validate on
            method: 'random', 'top_salary', 'top_avg_points'
        """
        available_players = self.data_df[self.data_df['date'] == validation_date].copy()
        
        if len(available_players) == 0:
            return {'total_points': 0, 'total_salary': 0, 'players': []}
        
        # Add salary if not present
        if 'salary' not in available_players.columns:
            available_players['salary'] = np.random.randint(3000, 12000, len(available_players))
        
        salary_cap = 50000
        lineup_size = 8
        
        if method == 'random':
            # Random selection within salary cap
            selected_players = []
            current_salary = 0
            available_indices = available_players.index.tolist()
            np.random.shuffle(available_indices)
            
            for idx in available_indices:
                if len(selected_players) >= lineup_size:
                    break
                player = available_players.loc[idx]
                if current_salary + player['salary'] <= salary_cap:
                    selected_players.append(player)
                    current_salary += player['salary']
                    
        elif method == 'top_salary':
            # Select highest salary players within cap
            available_players = available_players.sort_values('salary', ascending=False)
            selected_players = []
            current_salary = 0
            
            for _, player in available_players.iterrows():
                if len(selected_players) >= lineup_size:
                    break
                if current_salary + player['salary'] <= salary_cap:
                    selected_players.append(player)
                    current_salary += player['salary']
                    
        elif method == 'top_avg_points':
            # Select players with highest historical average points
            player_avg_points = self.data_df[self.data_df['date'] < validation_date].groupby('Name')['calculated_dk_fpts'].mean()
            available_players['avg_points'] = available_players['Name'].map(player_avg_points).fillna(0)
            available_players = available_players.sort_values('avg_points', ascending=False)
            
            selected_players = []
            current_salary = 0
            
            for _, player in available_players.iterrows():
                if len(selected_players) >= lineup_size:
                    break
                if current_salary + player['salary'] <= salary_cap:
                    selected_players.append(player)
                    current_salary += player['salary']
        
        # Calculate total performance
        total_points = sum(player['calculated_dk_fpts'] for player in selected_players)
        total_salary = sum(player['salary'] for player in selected_players)
        
        return {
            'total_points': total_points,
            'total_salary': total_salary,
            'players': [player['Name'] for player in selected_players],
            'method': method
        }
    
    def train_model(self, train_data: pd.DataFrame, episodes: int = 1000) -> MLBRLTeamSelector:
        """Train RL model on given training data"""
        logger.info(f"Training RL model on {len(train_data)} rows of data...")
        
        # Create temporary data file
        temp_path = 'temp_train_data.csv'
        train_data.to_csv(temp_path, index=False)
        
        # Initialize and train model
        selector = MLBRLTeamSelector(temp_path, model_save_path=None)
        selector.data_df = train_data.copy()  # Use provided data directly
        selector.setup_environment()
        
        # Train with fewer episodes for faster validation
        selector.train(episodes=episodes, save_freq=episodes//5)
        
        # Clean up temp file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return selector
    
    def validate_single_date(self, train_data: pd.DataFrame, validation_date: datetime,
                           model: MLBRLTeamSelector = None) -> Dict:
        """Validate model performance on a single date"""
        
        # Get available players for validation date
        validation_data = self.data_df[self.data_df['date'] == validation_date].copy()
        
        if len(validation_data) == 0:
            logger.warning(f"No players available for validation on {validation_date}")
            return None
        
        # Train model if not provided
        if model is None:
            model = self.train_model(train_data, episodes=500)
        
        # Get RL prediction
        rl_lineup = model.predict_optimal_lineup(validation_date)
        
        if rl_lineup is None or len(rl_lineup) == 0:
            logger.warning(f"RL model failed to generate lineup for {validation_date}")
            rl_performance = {'total_points': 0, 'total_salary': 0, 'players': []}
        else:
            rl_performance = {
                'total_points': rl_lineup['Predicted_Points'].sum(),
                'total_salary': rl_lineup['Salary'].sum(),
                'players': rl_lineup['Name'].tolist()
            }
        
        # Get baseline performances
        random_performance = self.get_baseline_performance(validation_date, 'random')
        top_salary_performance = self.get_baseline_performance(validation_date, 'top_salary')
        top_avg_performance = self.get_baseline_performance(validation_date, 'top_avg_points')
        
        # Calculate actual performance for RL lineup
        actual_rl_points = 0
        if rl_lineup is not None and len(rl_lineup) > 0:
            for player_name in rl_lineup['Name']:
                player_actual = validation_data[validation_data['Name'] == player_name]
                if len(player_actual) > 0:
                    actual_rl_points += player_actual['calculated_dk_fpts'].iloc[0]
        
        return {
            'date': validation_date,
            'rl_performance': rl_performance,
            'rl_actual_points': actual_rl_points,
            'random_performance': random_performance,
            'top_salary_performance': top_salary_performance,
            'top_avg_performance': top_avg_performance,
            'available_players': len(validation_data)
        }
    
    def run_walk_forward_validation(self, start_date: str = None, end_date: str = None,
                                   max_validations: int = 50) -> List[Dict]:
        """
        Run walk-forward validation
        
        Args:
            start_date: Start date for validation (if None, uses initial_train_days from start)
            end_date: End date for validation (if None, uses all available data)
            max_validations: Maximum number of validation steps
        """
        
        if start_date is None:
            start_idx = self.initial_train_days
        else:
            start_date = pd.to_datetime(start_date)
            start_idx = next(i for i, date in enumerate(self.unique_dates) if date >= start_date)
        
        if end_date is None:
            end_idx = len(self.unique_dates)
        else:
            end_date = pd.to_datetime(end_date)
            end_idx = next(i for i, date in enumerate(self.unique_dates) if date > end_date)
        
        logger.info(f"Starting walk-forward validation from index {start_idx} to {end_idx}")
        logger.info(f"Validation date range: {self.unique_dates[start_idx]} to {self.unique_dates[end_idx-1]}")
        
        validation_results = []
        current_model = None
        last_retrain_idx = 0
        
        for idx in range(start_idx, min(end_idx, start_idx + max_validations)):
            validation_date = self.unique_dates[idx]
            
            logger.info(f"Validating on date {validation_date} ({idx - start_idx + 1}/{min(end_idx - start_idx, max_validations)})")
            
            # Get training data (all data up to validation date)
            train_data = self.data_df[self.data_df['date'] < validation_date].copy()
            
            if len(train_data) == 0:
                logger.warning(f"No training data available for {validation_date}")
                continue
            
            # Retrain model if needed
            if (current_model is None or 
                idx - last_retrain_idx >= self.retrain_frequency):
                logger.info(f"Retraining model with {len(train_data)} rows of training data")
                current_model = self.train_model(train_data, episodes=800)
                last_retrain_idx = idx
            
            # Validate on current date
            result = self.validate_single_date(train_data, validation_date, current_model)
            
            if result is not None:
                validation_results.append(result)
                
                # Log performance
                rl_points = result['rl_actual_points']
                random_points = result['random_performance']['total_points']
                top_avg_points = result['top_avg_performance']['total_points']
                
                logger.info(f"  RL Actual: {rl_points:.1f} pts")
                logger.info(f"  Random: {random_points:.1f} pts")
                logger.info(f"  Top Avg: {top_avg_points:.1f} pts")
                logger.info(f"  RL vs Random: {rl_points - random_points:+.1f} pts")
                logger.info(f"  RL vs Top Avg: {rl_points - top_avg_points:+.1f} pts")
        
        self.validation_results = validation_results
        return validation_results
    
    def analyze_results(self) -> Dict:
        """Analyze validation results"""
        if not self.validation_results:
            logger.warning("No validation results to analyze")
            return {}
        
        # Extract performance metrics
        rl_actual_points = [r['rl_actual_points'] for r in self.validation_results]
        random_points = [r['random_performance']['total_points'] for r in self.validation_results]
        top_avg_points = [r['top_avg_performance']['total_points'] for r in self.validation_results]
        
        # Calculate performance metrics
        analysis = {
            'num_validations': len(self.validation_results),
            'rl_mean_points': np.mean(rl_actual_points),
            'rl_std_points': np.std(rl_actual_points),
            'random_mean_points': np.mean(random_points),
            'top_avg_mean_points': np.mean(top_avg_points),
            'rl_vs_random_mean_diff': np.mean(rl_actual_points) - np.mean(random_points),
            'rl_vs_top_avg_mean_diff': np.mean(rl_actual_points) - np.mean(top_avg_points),
            'rl_win_rate_vs_random': np.mean([rl > rand for rl, rand in zip(rl_actual_points, random_points)]),
            'rl_win_rate_vs_top_avg': np.mean([rl > top for rl, top in zip(rl_actual_points, top_avg_points)]),
        }
        
        logger.info("Walk-Forward Validation Analysis:")
        logger.info(f"Number of validations: {analysis['num_validations']}")
        logger.info(f"RL Mean Points: {analysis['rl_mean_points']:.2f} ± {analysis['rl_std_points']:.2f}")
        logger.info(f"Random Mean Points: {analysis['random_mean_points']:.2f}")
        logger.info(f"Top Average Mean Points: {analysis['top_avg_mean_points']:.2f}")
        logger.info(f"RL vs Random Difference: {analysis['rl_vs_random_mean_diff']:+.2f} pts")
        logger.info(f"RL vs Top Average Difference: {analysis['rl_vs_top_avg_mean_diff']:+.2f} pts")
        logger.info(f"RL Win Rate vs Random: {analysis['rl_win_rate_vs_random']:.1%}")
        logger.info(f"RL Win Rate vs Top Average: {analysis['rl_win_rate_vs_top_avg']:.1%}")
        
        return analysis
    
    def plot_results(self, save_path: str = None):
        """Plot validation results"""
        if not self.validation_results:
            logger.warning("No validation results to plot")
            return
        
        # Extract data for plotting
        dates = [r['date'] for r in self.validation_results]
        rl_points = [r['rl_actual_points'] for r in self.validation_results]
        random_points = [r['random_performance']['total_points'] for r in self.validation_results]
        top_avg_points = [r['top_avg_performance']['total_points'] for r in self.validation_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Performance over time
        axes[0, 0].plot(dates, rl_points, label='RL Model', marker='o', alpha=0.7)
        axes[0, 0].plot(dates, random_points, label='Random Selection', marker='s', alpha=0.7)
        axes[0, 0].plot(dates, top_avg_points, label='Top Average', marker='^', alpha=0.7)
        axes[0, 0].set_title('Performance Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Fantasy Points')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Performance differences
        rl_vs_random = [rl - rand for rl, rand in zip(rl_points, random_points)]
        rl_vs_top_avg = [rl - top for rl, top in zip(rl_points, top_avg_points)]
        
        axes[0, 1].bar(range(len(dates)), rl_vs_random, alpha=0.7, label='RL vs Random')
        axes[0, 1].bar(range(len(dates)), rl_vs_top_avg, alpha=0.7, label='RL vs Top Avg')
        axes[0, 1].set_title('Performance Differences')
        axes[0, 1].set_xlabel('Validation Index')
        axes[0, 1].set_ylabel('Point Difference')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 3: Distribution comparison
        axes[1, 0].hist(rl_points, bins=20, alpha=0.7, label='RL Model', density=True)
        axes[1, 0].hist(random_points, bins=20, alpha=0.7, label='Random Selection', density=True)
        axes[1, 0].hist(top_avg_points, bins=20, alpha=0.7, label='Top Average', density=True)
        axes[1, 0].set_title('Performance Distribution')
        axes[1, 0].set_xlabel('Fantasy Points')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative performance
        cumulative_rl = np.cumsum(rl_points)
        cumulative_random = np.cumsum(random_points)
        cumulative_top_avg = np.cumsum(top_avg_points)
        
        axes[1, 1].plot(cumulative_rl, label='RL Model', marker='o')
        axes[1, 1].plot(cumulative_random, label='Random Selection', marker='s')
        axes[1, 1].plot(cumulative_top_avg, label='Top Average', marker='^')
        axes[1, 1].set_title('Cumulative Performance')
        axes[1, 1].set_xlabel('Validation Index')
        axes[1, 1].set_ylabel('Cumulative Fantasy Points')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save validation results to CSV"""
        if not self.validation_results:
            logger.warning("No validation results to save")
            return
        
        # Flatten results for CSV
        flattened_results = []
        for result in self.validation_results:
            row = {
                'date': result['date'],
                'rl_actual_points': result['rl_actual_points'],
                'rl_predicted_points': result['rl_performance']['total_points'],
                'rl_salary': result['rl_performance']['total_salary'],
                'random_points': result['random_performance']['total_points'],
                'random_salary': result['random_performance']['total_salary'],
                'top_avg_points': result['top_avg_performance']['total_points'],
                'top_avg_salary': result['top_avg_performance']['total_salary'],
                'available_players': result['available_players'],
                'rl_vs_random_diff': result['rl_actual_points'] - result['random_performance']['total_points'],
                'rl_vs_top_avg_diff': result['rl_actual_points'] - result['top_avg_performance']['total_points']
            }
            flattened_results.append(row)
        
        results_df = pd.DataFrame(flattened_results)
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

def main():
    """Main function to run walk-forward validation"""
    
    # Configuration
    DATA_PATH = 'C:/Users/smtes/FangraphsData/merged_fangraphs_data.csv'
    RESULTS_PATH = '4_DATA/walkforward_validation_results.csv'
    PLOT_PATH = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/walkforward_validation_plots.png'
    
    # Initialize validator
    validator = WalkForwardRLValidator(
        data_path=DATA_PATH,
        initial_train_days=365,  # Train on 1 year of data initially
        validation_window=1,     # Validate on 1 day at a time
        retrain_frequency=7      # Retrain every 7 days
    )
    
    # Run validation
    results = validator.run_walk_forward_validation(
        start_date='2024-01-01',  # Start validation from this date
        end_date='2024-12-31',    # End validation at this date
        max_validations=30        # Limit to 30 validations for demo
    )
    
    # Analyze results
    analysis = validator.analyze_results()
    
    # Plot results
    validator.plot_results(save_path=PLOT_PATH)
    
    # Save results
    validator.save_results(RESULTS_PATH)
    
    logger.info("Walk-forward validation completed!")

if __name__ == "__main__":
    main()
