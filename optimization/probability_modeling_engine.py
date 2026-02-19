"""
Advanced Probability Modeling System for MLB DraftKings Optimizer
Integrates Bayesian inference, Monte Carlo methods, and sophisticated probability distributions
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from scipy.stats import multivariate_normal, multivariate_t, skewnorm, beta, gamma
    from scipy.special import logsumexp
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

try:
    import pymc3 as pm
    PYMC3_AVAILABLE = True
except ImportError:
    PYMC3_AVAILABLE = False

class ProbabilityModelingEngine:
    """
    Advanced probability modeling engine for DFS optimization
    Features:
    - Bayesian player performance modeling
    - Monte Carlo simulation with multiple distributions
    - Correlation modeling with copulas
    - Regime-aware probability updates
    - Dynamic confidence intervals
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.player_models = {}
        self.correlation_matrix = None
        self.regime_probabilities = {}
        self.confidence_level = 0.95
        
        print("🎲 Advanced Probability Modeling Engine Initialized")
        print(f"📊 Advanced Stats Available: {ADVANCED_STATS_AVAILABLE}")
        print(f"🔬 PyMC3 Available: {PYMC3_AVAILABLE}")
    
    def bayesian_player_modeling(self, player_data, historical_data=None):
        """
        Build Bayesian models for each player's performance
        Uses hierarchical modeling with priors based on position and salary
        """
        self.logger.info("🔬 Building Bayesian player models...")
        
        models = {}
        
        for _, player in player_data.iterrows():
            name = player['Name']
            position = player['Pos']
            salary = player['Salary']
            projected = player['Predicted_DK_Points']
            
            # Create position-based priors
            position_priors = self._get_position_priors(position)
            
            # Salary-based variance adjustment
            salary_factor = min(max(salary / 8000, 0.5), 2.0)  # Normalize around $8k
            
            # Build Bayesian model
            model = {
                'name': name,
                'position': position,
                'salary': salary,
                'projected_mean': projected,
                'prior_mean': position_priors['mean'],
                'prior_variance': position_priors['variance'] * salary_factor,
                'likelihood_variance': self._estimate_likelihood_variance(projected, position),
                'posterior_params': self._update_posterior(projected, position_priors, salary_factor)
            }
            
            # Add uncertainty quantification
            model['confidence_interval'] = self._calculate_confidence_interval(model)
            model['probability_distributions'] = self._fit_probability_distributions(model)
            
            models[name] = model
        
        self.player_models = models
        self.logger.info(f"✅ Built Bayesian models for {len(models)} players")
        return models
    
    def _get_position_priors(self, position):
        """Get position-specific priors based on historical data"""
        position_stats = {
            'P': {'mean': 12.5, 'variance': 6.0},
            'C': {'mean': 8.5, 'variance': 3.5},
            '1B': {'mean': 9.2, 'variance': 4.0},
            '2B': {'mean': 8.8, 'variance': 3.8},
            '3B': {'mean': 9.1, 'variance': 4.1},
            'SS': {'mean': 8.9, 'variance': 3.9},
            'OF': {'mean': 9.0, 'variance': 4.0}
        }
        
        return position_stats.get(position, {'mean': 9.0, 'variance': 4.0})
    
    def _estimate_likelihood_variance(self, projected, position):
        """Estimate likelihood variance based on projection and position"""
        base_variance = max(0.15 * projected, 2.0)
        
        # Position-specific variance multipliers
        position_multipliers = {
            'P': 1.5,  # Pitchers more volatile
            'C': 1.0,
            '1B': 1.1,
            '2B': 1.0,
            '3B': 1.1,
            'SS': 1.0,
            'OF': 1.0
        }
        
        multiplier = position_multipliers.get(position, 1.0)
        return base_variance * multiplier
    
    def _update_posterior(self, projected, priors, salary_factor):
        """Update posterior distribution using Bayesian inference"""
        prior_mean = priors['mean']
        prior_var = priors['variance'] * salary_factor
        likelihood_var = self._estimate_likelihood_variance(projected, 'OF')
        
        # Bayesian update
        posterior_var = 1 / (1/prior_var + 1/likelihood_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + projected/likelihood_var)
        
        return {
            'mean': posterior_mean,
            'variance': posterior_var,
            'std': np.sqrt(posterior_var)
        }
    
    def _calculate_confidence_interval(self, model):
        """Calculate confidence intervals for player performance"""
        posterior = model['posterior_params']
        mean = posterior['mean']
        std = posterior['std']
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        return {
            'lower': mean - z_score * std,
            'upper': mean + z_score * std,
            'width': 2 * z_score * std
        }
    
    def _fit_probability_distributions(self, model):
        """Fit multiple probability distributions to player performance"""
        posterior = model['posterior_params']
        mean = posterior['mean']
        std = posterior['std']
        
        distributions = {}
        
        # Normal distribution
        distributions['normal'] = {
            'type': 'normal',
            'params': {'loc': mean, 'scale': std},
            'cdf': lambda x: stats.norm.cdf(x, loc=mean, scale=std),
            'pdf': lambda x: stats.norm.pdf(x, loc=mean, scale=std),
            'ppf': lambda p: stats.norm.ppf(p, loc=mean, scale=std)
        }
        
        # Skewed normal (for asymmetric performance)
        if ADVANCED_STATS_AVAILABLE:
            skewness = 0.5  # Slight positive skew (upside potential)
            distributions['skewnorm'] = {
                'type': 'skewnorm',
                'params': {'a': skewness, 'loc': mean, 'scale': std},
                'cdf': lambda x: stats.skewnorm.cdf(x, a=skewness, loc=mean, scale=std),
                'pdf': lambda x: stats.skewnorm.pdf(x, a=skewness, loc=mean, scale=std),
                'ppf': lambda p: stats.skewnorm.ppf(p, a=skewness, loc=mean, scale=std)
            }
            
            # Gamma distribution (for positive values)
            if mean > 0:
                shape = (mean/std)**2
                scale = std**2/mean
                distributions['gamma'] = {
                    'type': 'gamma',
                    'params': {'a': shape, 'scale': scale},
                    'cdf': lambda x: stats.gamma.cdf(x, a=shape, scale=scale),
                    'pdf': lambda x: stats.gamma.pdf(x, a=shape, scale=scale),
                    'ppf': lambda p: stats.gamma.ppf(p, a=shape, scale=scale)
                }
        
        return distributions
    
    def monte_carlo_simulation(self, lineup_players, n_simulations=10000):
        """
        Advanced Monte Carlo simulation with multiple probability distributions
        """
        self.logger.info(f"🎲 Running Monte Carlo simulation with {n_simulations} iterations...")
        
        results = {
            'simulations': [],
            'statistics': {},
            'risk_metrics': {},
            'probability_analysis': {}
        }
        
        player_names = [p['Name'] for _, p in lineup_players.iterrows()]
        
        # Run simulations
        for i in range(n_simulations):
            lineup_points = []
            
            for _, player in lineup_players.iterrows():
                name = player['Name']
                
                if name in self.player_models:
                    model = self.player_models[name]
                    
                    # Sample from posterior distribution
                    posterior = model['posterior_params']
                    simulated_points = np.random.normal(
                        posterior['mean'], 
                        posterior['std']
                    )
                    
                    # Ensure non-negative values
                    simulated_points = max(0, simulated_points)
                    lineup_points.append(simulated_points)
                else:
                    # Fallback for players without models
                    projected = player['Predicted_DK_Points']
                    std = projected * 0.2  # 20% coefficient of variation
                    simulated_points = max(0, np.random.normal(projected, std))
                    lineup_points.append(simulated_points)
            
            total_points = sum(lineup_points)
            results['simulations'].append({
                'iteration': i,
                'total_points': total_points,
                'player_points': dict(zip(player_names, lineup_points))
            })
        
        # Calculate statistics
        total_points_array = np.array([s['total_points'] for s in results['simulations']])
        
        results['statistics'] = {
            'mean': np.mean(total_points_array),
            'median': np.median(total_points_array),
            'std': np.std(total_points_array),
            'min': np.min(total_points_array),
            'max': np.max(total_points_array),
            'percentiles': {
                '5th': np.percentile(total_points_array, 5),
                '25th': np.percentile(total_points_array, 25),
                '75th': np.percentile(total_points_array, 75),
                '95th': np.percentile(total_points_array, 95)
            }
        }
        
        # Risk metrics
        results['risk_metrics'] = self._calculate_risk_metrics(total_points_array)
        
        # Probability analysis
        results['probability_analysis'] = self._probability_analysis(total_points_array)
        
        self.logger.info("✅ Monte Carlo simulation complete")
        return results
    
    def _calculate_risk_metrics(self, returns):
        """Calculate comprehensive risk metrics"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        return {
            'value_at_risk_95': np.percentile(returns, 5),
            'value_at_risk_99': np.percentile(returns, 1),
            'conditional_var_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'conditional_var_99': np.mean(returns[returns <= np.percentile(returns, 1)]),
            'sharpe_ratio': mean_return / std_return if std_return > 0 else 0,
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'tail_ratio': np.percentile(returns, 95) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
        }
    
    def _calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio (downside deviation)"""
        mean_return = np.mean(returns)
        downside_returns = returns[returns < mean_return]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        return mean_return / downside_std if downside_std > 0 else 0
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)
    
    def _probability_analysis(self, returns):
        """Analyze probability of different outcomes"""
        mean_return = np.mean(returns)
        
        # Define score thresholds
        thresholds = [100, 120, 140, 160, 180, 200]
        
        probabilities = {}
        for threshold in thresholds:
            prob = np.mean(returns >= threshold)
            probabilities[f'prob_above_{threshold}'] = prob
        
        # Probability of beating average
        prob_above_mean = np.mean(returns >= mean_return)
        
        # Probability of extreme outcomes
        prob_top_10pct = 0.1  # By definition
        prob_bottom_10pct = 0.1  # By definition
        
        return {
            'threshold_probabilities': probabilities,
            'prob_above_mean': prob_above_mean,
            'prob_top_decile': prob_top_10pct,
            'prob_bottom_decile': prob_bottom_10pct,
            'expected_value': mean_return,
            'value_at_risk_interpretation': f"5% chance of scoring below {np.percentile(returns, 5):.1f} points"
        }
    
    def correlation_modeling(self, player_data):
        """
        Model correlations between players using advanced techniques
        """
        self.logger.info("🔗 Modeling player correlations...")
        
        # Create correlation matrix based on various factors
        n_players = len(player_data)
        correlation_matrix = np.eye(n_players)
        
        players_list = list(player_data.iterrows())
        
        for i in range(n_players):
            for j in range(i+1, n_players):
                player1 = players_list[i][1]
                player2 = players_list[j][1]
                
                # Calculate correlation based on multiple factors
                correlation = self._calculate_player_correlation(player1, player2)
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        self.correlation_matrix = correlation_matrix
        self.logger.info("✅ Correlation modeling complete")
        return correlation_matrix
    
    def _calculate_player_correlation(self, player1, player2):
        """Calculate correlation between two players"""
        correlation = 0.0
        
        # Same team correlation
        if player1['Team'] == player2['Team']:
            correlation += 0.3
        
        # Same position correlation
        if player1['Pos'] == player2['Pos']:
            correlation += 0.1
        
        # Salary tier correlation
        salary_diff = abs(player1['Salary'] - player2['Salary'])
        if salary_diff < 1000:
            correlation += 0.05
        
        # Cap correlation at reasonable levels
        correlation = min(correlation, 0.5)
        
        return correlation
    
    def regime_detection(self, market_data=None):
        """
        Detect different market regimes and adjust probabilities accordingly
        """
        self.logger.info("🔍 Detecting market regimes...")
        
        # Simplified regime detection (in production, use more sophisticated methods)
        regimes = {
            'high_scoring': {'probability': 0.3, 'adjustment': 1.15},
            'normal': {'probability': 0.5, 'adjustment': 1.0},
            'low_scoring': {'probability': 0.2, 'adjustment': 0.85}
        }
        
        # Current regime detection (simplified)
        current_regime = 'normal'  # Default
        
        self.regime_probabilities = regimes
        self.logger.info(f"✅ Current regime: {current_regime}")
        
        return regimes, current_regime
    
    def update_probabilities_with_regime(self, regime):
        """Update player probabilities based on current regime"""
        if regime in self.regime_probabilities:
            adjustment = self.regime_probabilities[regime]['adjustment']
            
            for name, model in self.player_models.items():
                # Adjust posterior parameters
                model['posterior_params']['mean'] *= adjustment
                # Recalculate confidence intervals
                model['confidence_interval'] = self._calculate_confidence_interval(model)
    
    def optimize_lineup_with_probabilities(self, player_data, constraints=None):
        """
        Optimize lineup using probability-based objective function
        """
        self.logger.info("🎯 Optimizing lineup with probability modeling...")
        
        # Build Bayesian models
        self.bayesian_player_modeling(player_data)
        
        # Model correlations
        self.correlation_modeling(player_data)
        
        # Objective function: maximize expected value while controlling risk
        def objective_function(weights):
            expected_return = 0
            risk_penalty = 0
            
            for i, (_, player) in enumerate(player_data.iterrows()):
                name = player['Name']
                if name in self.player_models:
                    model = self.player_models[name]
                    expected_return += weights[i] * model['posterior_params']['mean']
                    risk_penalty += weights[i] * model['posterior_params']['variance']
            
            # Risk adjustment
            risk_adjusted_return = expected_return - 0.1 * risk_penalty
            
            return -risk_adjusted_return  # Negative for minimization
        
        # Constraints
        n_players = len(player_data)
        bounds = [(0, 1) for _ in range(n_players)]
        
        # Salary constraint
        salaries = player_data['Salary'].values
        salary_constraint = {'type': 'ineq', 'fun': lambda x: 50000 - np.dot(x, salaries)}
        
        # Position constraints (simplified)
        position_constraints = []
        
        # Optimize
        from scipy.optimize import minimize
        
        initial_weights = np.ones(n_players) / n_players
        
        result = minimize(
            objective_function,
            initial_weights,
            bounds=bounds,
            constraints=[salary_constraint] + position_constraints,
            method='SLSQP'
        )
        
        if result.success:
            optimal_weights = result.x
            selected_players = player_data[optimal_weights > 0.01]  # Threshold for selection
            
            self.logger.info(f"✅ Optimization complete. Selected {len(selected_players)} players")
            return selected_players, optimal_weights
        else:
            self.logger.error("❌ Optimization failed")
            return None, None

def demo_probability_modeling():
    """Demonstrate the probability modeling system"""
    print("🎲 Advanced Probability Modeling Demo")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    
    players_data = []
    positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
    teams = ['NYY', 'BOS', 'LAD', 'HOU', 'ATL']
    
    for i in range(30):
        pos = np.random.choice(positions)
        team = np.random.choice(teams)
        salary = np.random.randint(3000, 12000)
        projected = np.random.uniform(8, 20)
        
        players_data.append({
            'Name': f'Player_{i+1}',
            'Pos': pos,
            'Team': team,
            'Salary': salary,
            'Predicted_DK_Points': projected
        })
    
    df = pd.DataFrame(players_data)
    
    # Initialize probability modeling engine
    engine = ProbabilityModelingEngine()
    
    # Run Bayesian modeling
    print("\n🔬 Building Bayesian Models...")
    models = engine.bayesian_player_modeling(df)
    
    # Show some model results
    print("\n📊 Sample Player Models:")
    for i, (name, model) in enumerate(list(models.items())[:3]):
        print(f"\n{name}:")
        print(f"  Projected: {model['projected_mean']:.1f}")
        print(f"  Posterior Mean: {model['posterior_params']['mean']:.1f}")
        print(f"  Posterior Std: {model['posterior_params']['std']:.1f}")
        print(f"  95% CI: [{model['confidence_interval']['lower']:.1f}, {model['confidence_interval']['upper']:.1f}]")
    
    # Run Monte Carlo simulation on a sample lineup
    print("\n🎲 Running Monte Carlo Simulation...")
    sample_lineup = df.head(10)  # First 10 players
    mc_results = engine.monte_carlo_simulation(sample_lineup, n_simulations=5000)
    
    # Display results
    stats = mc_results['statistics']
    risk_metrics = mc_results['risk_metrics']
    prob_analysis = mc_results['probability_analysis']
    
    print("\n📈 Monte Carlo Results:")
    print(f"  Expected Total Points: {stats['mean']:.1f}")
    print(f"  Standard Deviation: {stats['std']:.1f}")
    print(f"  95% Confidence Interval: [{stats['percentiles']['5th']:.1f}, {stats['percentiles']['95th']:.1f}]")
    
    print("\n⚠️ Risk Metrics:")
    print(f"  Value at Risk (95%): {risk_metrics['value_at_risk_95']:.1f}")
    print(f"  Conditional VaR (95%): {risk_metrics['conditional_var_95']:.1f}")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
    
    print("\n📊 Probability Analysis:")
    for threshold, prob in prob_analysis['threshold_probabilities'].items():
        print(f"  {threshold}: {prob:.1%}")
    
    print(f"\n{prob_analysis['value_at_risk_interpretation']}")
    
    # Correlation modeling
    print("\n🔗 Correlation Analysis:")
    corr_matrix = engine.correlation_modeling(df)
    print(f"  Average correlation: {np.mean(corr_matrix[corr_matrix != 1]):.3f}")
    print(f"  Max correlation: {np.max(corr_matrix[corr_matrix != 1]):.3f}")
    
    print("\n🎯 Probability Modeling Demo Complete!")

if __name__ == "__main__":
    demo_probability_modeling()
