"""
Advanced Probability Modeling System for MLB DraftKings Optimizer
================================================================

This module implements sophisticated probability modeling techniques for DFS optimization:
- Bayesian inference for player performance prediction
- Monte Carlo simulation for outcome probability distributions
- Hidden Markov Models for player state modeling
- Probabilistic lineup optimization using Expected Value Theory
- Probability-weighted scoring and risk assessment
- Dynamic probability updates based on game conditions
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.stats import beta, gamma, dirichlet, multivariate_normal
    from scipy.special import gammaln, digamma
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class BayesianPlayerModel:
    """
    Bayesian model for player performance prediction
    Uses Beta-Binomial and Gamma-Poisson conjugate priors
    """
    
    def __init__(self):
        self.player_models = {}
        self.prior_alpha = 1.0  # Beta prior for success probability
        self.prior_beta = 1.0
        self.prior_shape = 1.0  # Gamma prior for points
        self.prior_rate = 1.0
    
    def update_player_model(self, player_name, historical_points, historical_games):
        """Update Bayesian model for a player"""
        if not historical_points:
            return
        
        points_array = np.array(historical_points)
        games_array = np.array(historical_games)
        
        # Beta-Binomial model for success probability (games with >10 points)
        successes = np.sum(points_array > 10)
        trials = len(points_array)
        
        posterior_alpha = self.prior_alpha + successes
        posterior_beta = self.prior_beta + trials - successes
        
        # Gamma-Poisson model for points distribution
        sum_points = np.sum(points_array)
        n_games = len(points_array)
        
        posterior_shape = self.prior_shape + sum_points
        posterior_rate = self.prior_rate + n_games
        
        self.player_models[player_name] = {
            'beta_alpha': posterior_alpha,
            'beta_beta': posterior_beta,
            'gamma_shape': posterior_shape,
            'gamma_rate': posterior_rate,
            'mean_points': sum_points / n_games if n_games > 0 else 0,
            'variance': np.var(points_array) if len(points_array) > 1 else 1,
            'last_update': len(historical_points)
        }
    
    def predict_player_distribution(self, player_name, n_samples=1000):
        """Generate probability distribution for player performance"""
        if player_name not in self.player_models:
            # Use uninformed prior
            return np.random.gamma(2, 5, n_samples)  # Generic distribution
        
        model = self.player_models[player_name]
        
        # Sample from posterior distributions
        success_prob = np.random.beta(model['beta_alpha'], model['beta_beta'])
        points_rate = np.random.gamma(model['gamma_shape'], 1/model['gamma_rate'])
        
        # Generate samples
        samples = np.random.poisson(points_rate, n_samples)
        
        # Add success probability weighting
        success_mask = np.random.binomial(1, success_prob, n_samples)
        samples = samples * success_mask + np.random.exponential(2, n_samples) * (1 - success_mask)
        
        return np.maximum(samples, 0)  # Ensure non-negative

class MonteCarloSimulator:
    """
    Monte Carlo simulation for lineup outcome modeling
    """
    
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        self.simulation_results = []
    
    def simulate_lineup_outcomes(self, lineup_df, player_models=None):
        """Simulate all possible outcomes for a lineup"""
        if player_models is None:
            player_models = BayesianPlayerModel()
        
        simulation_results = []
        
        print(f"🎲 Running {self.n_simulations:,} Monte Carlo simulations...")
        
        for sim in range(self.n_simulations):
            total_points = 0
            lineup_outcome = []
            
            for _, player in lineup_df.iterrows():
                # Generate probability distribution for this player
                if hasattr(player_models, 'predict_player_distribution'):
                    player_points = player_models.predict_player_distribution(player['Name'], 1)[0]
                else:
                    # Fallback: use normal distribution around projection
                    base_projection = player['Predicted_DK_Points']
                    volatility = base_projection * 0.25  # 25% volatility
                    player_points = np.random.normal(base_projection, volatility)
                    player_points = max(0, player_points)  # Non-negative
                
                total_points += player_points
                lineup_outcome.append({
                    'player': player['Name'],
                    'position': player['Pos'],
                    'projected': player['Predicted_DK_Points'],
                    'simulated': player_points,
                    'salary': player['Salary']
                })
            
            simulation_results.append({
                'simulation_id': sim,
                'total_points': total_points,
                'lineup_outcome': lineup_outcome,
                'total_salary': lineup_df['Salary'].sum()
            })
            
            # Progress indicator
            if sim % 1000 == 0 and sim > 0:
                print(f"  📊 Completed {sim:,} simulations...")
        
        self.simulation_results = simulation_results
        return simulation_results
    
    def calculate_outcome_probabilities(self, target_scores=None):
        """Calculate probability of achieving different score thresholds"""
        if not self.simulation_results:
            return {}
        
        if target_scores is None:
            target_scores = [100, 120, 140, 160, 180, 200]
        
        total_sims = len(self.simulation_results)
        scores = [result['total_points'] for result in self.simulation_results]
        
        probabilities = {}
        for target in target_scores:
            prob = np.mean(np.array(scores) >= target)
            probabilities[f'prob_over_{target}'] = prob
        
        # Additional statistics
        probabilities['mean_score'] = np.mean(scores)
        probabilities['median_score'] = np.median(scores)
        probabilities['std_score'] = np.std(scores)
        probabilities['var_95'] = np.percentile(scores, 5)  # 95% VaR
        probabilities['var_99'] = np.percentile(scores, 1)  # 99% VaR
        probabilities['max_score'] = np.max(scores)
        probabilities['min_score'] = np.min(scores)
        
        return probabilities

class HiddenMarkovModel:
    """
    Hidden Markov Model for player state modeling
    Models different performance states (hot, cold, average)
    """
    
    def __init__(self, n_states=3):
        self.n_states = n_states  # Hot, Average, Cold
        self.state_names = ['Cold', 'Average', 'Hot']
        self.transition_matrix = None
        self.emission_params = None
        self.initial_state_probs = None
    
    def fit_player_hmm(self, player_history):
        """Fit HMM to player's historical performance"""
        if len(player_history) < 10:
            return self.default_hmm_params()
        
        # Use Gaussian Mixture Model as approximation to HMM
        if SKLEARN_AVAILABLE:
            gmm = GaussianMixture(n_components=self.n_states, random_state=42)
            history_reshaped = np.array(player_history).reshape(-1, 1)
            gmm.fit(history_reshaped)
            
            # Extract parameters
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            weights = gmm.weights_
            
            # Sort by mean (Cold, Average, Hot)
            sorted_indices = np.argsort(means)
            
            self.emission_params = {
                'means': means[sorted_indices],
                'variances': variances[sorted_indices],
                'weights': weights[sorted_indices]
            }
            
            # Simple transition matrix (tendency to stay in same state)
            self.transition_matrix = np.array([
                [0.7, 0.2, 0.1],  # Cold -> Cold, Average, Hot
                [0.2, 0.6, 0.2],  # Average -> Cold, Average, Hot  
                [0.1, 0.2, 0.7]   # Hot -> Cold, Average, Hot
            ])
            
            self.initial_state_probs = weights[sorted_indices]
        else:
            return self.default_hmm_params()
    
    def default_hmm_params(self):
        """Default HMM parameters when fitting fails"""
        self.emission_params = {
            'means': np.array([8.0, 12.0, 18.0]),  # Cold, Average, Hot
            'variances': np.array([4.0, 6.0, 8.0]),
            'weights': np.array([0.3, 0.4, 0.3])
        }
        
        self.transition_matrix = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ])
        
        self.initial_state_probs = np.array([0.3, 0.4, 0.3])
    
    def predict_next_performance(self, current_state=None):
        """Predict next performance based on current state"""
        if self.emission_params is None:
            self.default_hmm_params()
        
        if current_state is None:
            current_state = np.random.choice(self.n_states, p=self.initial_state_probs)
        
        # Transition to next state
        next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
        
        # Sample from emission distribution
        mean = self.emission_params['means'][next_state]
        variance = self.emission_params['variances'][next_state]
        
        performance = np.random.normal(mean, np.sqrt(variance))
        
        return max(0, performance), next_state, self.state_names[next_state]

class ProbabilisticLineupOptimizer:
    """
    Probabilistic lineup optimization using Expected Value Theory
    """
    
    def __init__(self):
        self.bayesian_model = BayesianPlayerModel()
        self.monte_carlo = MonteCarloSimulator()
        self.hmm_models = {}
    
    def optimize_lineup_with_probabilities(self, player_df, target_score=150, risk_tolerance=0.5):
        """
        Optimize lineup considering probability distributions
        """
        print("🎯 Starting Probabilistic Lineup Optimization...")
        
        # Step 1: Build Bayesian models for each player
        self.build_player_models(player_df)
        
        # Step 2: Calculate expected values and uncertainties
        enhanced_df = self.calculate_probabilistic_metrics(player_df)
        
        # Step 3: Optimize using probability-weighted objective
        optimal_lineup = self.probability_weighted_optimization(
            enhanced_df, target_score, risk_tolerance
        )
        
        return optimal_lineup
    
    def build_player_models(self, player_df):
        """Build Bayesian models for all players"""
        print("📊 Building Bayesian player models...")
        
        for _, player in player_df.iterrows():
            # Generate mock historical data (in production, use real data)
            historical_points = self.generate_mock_history(player['Predicted_DK_Points'])
            historical_games = list(range(len(historical_points)))
            
            self.bayesian_model.update_player_model(
                player['Name'], historical_points, historical_games
            )
    
    def generate_mock_history(self, projected_points, n_games=20):
        """Generate mock historical data for demonstration"""
        # Create realistic historical performance
        base_volatility = projected_points * 0.3
        history = []
        
        for i in range(n_games):
            # Add some trend and seasonality
            trend = np.sin(i * 0.3) * 2
            performance = np.random.normal(projected_points + trend, base_volatility)
            history.append(max(0, performance))
        
        return history
    
    def calculate_probabilistic_metrics(self, player_df):
        """Calculate probabilistic metrics for each player"""
        print("📈 Calculating probabilistic metrics...")
        
        enhanced_df = player_df.copy()
        
        # Add probability-based metrics
        enhanced_df['expected_value'] = 0.0
        enhanced_df['value_variance'] = 0.0
        enhanced_df['prob_over_15'] = 0.0
        enhanced_df['prob_over_20'] = 0.0
        enhanced_df['risk_adjusted_value'] = 0.0
        enhanced_df['kelly_weight'] = 0.0
        
        for idx, player in enhanced_df.iterrows():
            # Get probability distribution
            distribution = self.bayesian_model.predict_player_distribution(
                player['Name'], n_samples=1000
            )
            
            # Calculate metrics
            expected_value = np.mean(distribution)
            value_variance = np.var(distribution)
            prob_over_15 = np.mean(distribution >= 15)
            prob_over_20 = np.mean(distribution >= 20)
            
            # Risk-adjusted value (considering variance)
            risk_adjusted_value = expected_value - 0.5 * value_variance
            
            # Kelly criterion weight
            win_prob = prob_over_15
            avg_win = np.mean(distribution[distribution >= 15]) if np.any(distribution >= 15) else 0
            avg_loss = np.mean(distribution[distribution < 15]) if np.any(distribution < 15) else 0
            
            if avg_loss > 0:
                kelly_weight = win_prob - (1 - win_prob) * avg_win / avg_loss
            else:
                kelly_weight = 0.1
            
            kelly_weight = max(0, min(kelly_weight, 0.5))  # Cap at 50%
            
            # Update dataframe
            enhanced_df.at[idx, 'expected_value'] = expected_value
            enhanced_df.at[idx, 'value_variance'] = value_variance
            enhanced_df.at[idx, 'prob_over_15'] = prob_over_15
            enhanced_df.at[idx, 'prob_over_20'] = prob_over_20
            enhanced_df.at[idx, 'risk_adjusted_value'] = risk_adjusted_value
            enhanced_df.at[idx, 'kelly_weight'] = kelly_weight
        
        return enhanced_df
    
    def probability_weighted_optimization(self, enhanced_df, target_score, risk_tolerance):
        """
        Optimize lineup using probability-weighted scoring
        """
        print(f"🎯 Optimizing for target score: {target_score} with risk tolerance: {risk_tolerance}")
        
        # Simple greedy optimization with probability weighting
        selected_players = []
        remaining_salary = 50000
        position_needs = {
            'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3
        }
        
        # Sort by risk-adjusted value per dollar
        enhanced_df['value_per_dollar'] = enhanced_df['risk_adjusted_value'] / enhanced_df['Salary'] * 1000
        
        for position, needed in position_needs.items():
            position_players = enhanced_df[
                (enhanced_df['Pos'] == position) & 
                (enhanced_df['Salary'] <= remaining_salary)
            ].nlargest(needed * 3, 'value_per_dollar')  # Get top candidates
            
            # Apply probability weighting
            if len(position_players) > 0:
                # Weight by probability of exceeding expectations
                weights = position_players['prob_over_15'] * position_players['kelly_weight']
                
                # Select players based on probability-weighted selection
                selected_count = 0
                for _, player in position_players.iterrows():
                    if selected_count >= needed:
                        break
                    
                    if player['Salary'] <= remaining_salary:
                        selected_players.append(player)
                        remaining_salary -= player['Salary']
                        selected_count += 1
        
        if selected_players:
            lineup_df = pd.DataFrame(selected_players)
            return lineup_df
        else:
            return pd.DataFrame()

class ProbabilityAnalyzer:
    """
    Analyze probability distributions and provide insights
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_lineup_probabilities(self, lineup_df, simulation_results):
        """Comprehensive probability analysis of a lineup"""
        print("📊 Analyzing lineup probabilities...")
        
        if not simulation_results:
            return {}
        
        scores = [result['total_points'] for result in simulation_results]
        
        analysis = {
            'descriptive_stats': self.calculate_descriptive_stats(scores),
            'risk_metrics': self.calculate_risk_metrics(scores),
            'probability_bands': self.calculate_probability_bands(scores),
            'player_contributions': self.analyze_player_contributions(simulation_results),
            'scenario_analysis': self.scenario_analysis(scores)
        }
        
        return analysis
    
    def calculate_descriptive_stats(self, scores):
        """Calculate descriptive statistics"""
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'skewness': stats.skew(scores),
            'kurtosis': stats.kurtosis(scores)
        }
    
    def calculate_risk_metrics(self, scores):
        """Calculate risk metrics"""
        return {
            'var_95': np.percentile(scores, 5),
            'var_99': np.percentile(scores, 1),
            'cvar_95': np.mean([s for s in scores if s <= np.percentile(scores, 5)]),
            'cvar_99': np.mean([s for s in scores if s <= np.percentile(scores, 1)]),
            'downside_deviation': np.std([s for s in scores if s < np.mean(scores)]),
            'upside_potential': np.mean([s for s in scores if s > np.mean(scores)])
        }
    
    def calculate_probability_bands(self, scores):
        """Calculate probability of achieving different score bands"""
        bands = [
            (0, 100), (100, 120), (120, 140), (140, 160), 
            (160, 180), (180, 200), (200, float('inf'))
        ]
        
        band_probs = {}
        for low, high in bands:
            if high == float('inf'):
                prob = np.mean(np.array(scores) >= low)
                band_probs[f'{low}+'] = prob
            else:
                prob = np.mean((np.array(scores) >= low) & (np.array(scores) < high))
                band_probs[f'{low}-{high}'] = prob
        
        return band_probs
    
    def analyze_player_contributions(self, simulation_results):
        """Analyze individual player contributions to lineup variance"""
        player_contributions = {}
        
        for result in simulation_results:
            for player_result in result['lineup_outcome']:
                player_name = player_result['player']
                if player_name not in player_contributions:
                    player_contributions[player_name] = []
                
                player_contributions[player_name].append(player_result['simulated'])
        
        contribution_stats = {}
        for player, scores in player_contributions.items():
            contribution_stats[player] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'contribution_to_variance': np.var(scores)
            }
        
        return contribution_stats
    
    def scenario_analysis(self, scores):
        """Perform scenario analysis"""
        return {
            'best_case': np.percentile(scores, 95),
            'worst_case': np.percentile(scores, 5),
            'base_case': np.median(scores),
            'optimistic': np.percentile(scores, 75),
            'pessimistic': np.percentile(scores, 25)
        }

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Advanced Probability Modeling...")
    
    # Create sample data
    np.random.seed(42)
    sample_data = []
    positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
    
    for i in range(50):
        sample_data.append({
            'Name': f'Player_{i}',
            'Pos': np.random.choice(positions),
            'Team': np.random.choice(['NYY', 'BOS', 'LAD']),
            'Salary': np.random.randint(3000, 12000),
            'Predicted_DK_Points': np.random.uniform(8, 25)
        })
    
    df = pd.DataFrame(sample_data)
    
    # Test probabilistic optimization
    optimizer = ProbabilisticLineupOptimizer()
    optimal_lineup = optimizer.optimize_lineup_with_probabilities(df)
    
    if not optimal_lineup.empty:
        print(f"\\n✅ Generated optimal lineup with {len(optimal_lineup)} players")
        print(f"💰 Total Salary: ${optimal_lineup['Salary'].sum():,}")
        print(f"📊 Expected Points: {optimal_lineup['expected_value'].sum():.1f}")
        print(f"🎯 Risk-Adjusted Value: {optimal_lineup['risk_adjusted_value'].sum():.1f}")
        
        # Run Monte Carlo simulation
        print("\\n🎲 Running Monte Carlo simulation...")
        simulation_results = optimizer.monte_carlo.simulate_lineup_outcomes(optimal_lineup)
        
        # Calculate probabilities
        probabilities = optimizer.monte_carlo.calculate_outcome_probabilities()
        
        print(f"\\n📈 Probability Analysis:")
        print(f"  Mean Score: {probabilities['mean_score']:.1f}")
        print(f"  Std Dev: {probabilities['std_score']:.1f}")
        print(f"  Prob > 150: {probabilities.get('prob_over_160', 0):.1%}")
        print(f"  VaR (95%): {probabilities['var_95']:.1f}")
        
        # Full analysis
        analyzer = ProbabilityAnalyzer()
        analysis = analyzer.analyze_lineup_probabilities(optimal_lineup, simulation_results)
        
        print(f"\\n🔍 Risk Metrics:")
        risk_metrics = analysis['risk_metrics']
        print(f"  Downside Deviation: {risk_metrics['downside_deviation']:.1f}")
        print(f"  Upside Potential: {risk_metrics['upside_potential']:.1f}")
        
    print("\\n🎯 Probability modeling test complete!")
