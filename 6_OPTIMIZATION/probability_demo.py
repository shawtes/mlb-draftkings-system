"""
Comprehensive Probability Modeling Demo
=======================================

This demo showcases all the advanced probability modeling features:
- Bayesian player performance modeling
- Monte Carlo simulation
- Hidden Markov Models for player states
- Probabilistic lineup optimization
- Risk analysis and scenario modeling
"""

import numpy as np
import pandas as pd
from probability_modeling import (
    BayesianPlayerModel, 
    MonteCarloSimulator, 
    HiddenMarkovModel,
    ProbabilisticLineupOptimizer, 
    ProbabilityAnalyzer
)

def create_realistic_player_data():
    """Create realistic player data for demonstration"""
    np.random.seed(42)
    
    # Define realistic position constraints and salary ranges
    position_configs = {
        'P': {'count': 20, 'salary_range': (3000, 12000), 'points_range': (8, 25)},
        'C': {'count': 12, 'salary_range': (2500, 6000), 'points_range': (6, 18)},
        '1B': {'count': 12, 'salary_range': (3000, 8000), 'points_range': (8, 22)},
        '2B': {'count': 12, 'salary_range': (2500, 7000), 'points_range': (7, 20)},
        '3B': {'count': 12, 'salary_range': (3000, 8500), 'points_range': (8, 22)},
        'SS': {'count': 12, 'salary_range': (2800, 7500), 'points_range': (7, 21)},
        'OF': {'count': 20, 'salary_range': (2500, 8000), 'points_range': (6, 24)}
    }
    
    teams = ['NYY', 'BOS', 'HOU', 'LAD', 'ATL', 'CLE', 'TB', 'SD', 'TOR', 'PHI']
    
    players = []
    player_id = 1
    
    for pos, config in position_configs.items():
        for i in range(config['count']):
            # Generate correlated salary and points (higher salary = higher projection)
            salary = np.random.randint(*config['salary_range'])
            salary_percentile = (salary - config['salary_range'][0]) / (config['salary_range'][1] - config['salary_range'][0])
            
            # Points correlated with salary but with noise
            base_points = config['points_range'][0] + salary_percentile * (config['points_range'][1] - config['points_range'][0])
            projected_points = base_points + np.random.normal(0, 2)
            projected_points = max(config['points_range'][0], min(config['points_range'][1], projected_points))
            
            players.append({
                'Name': f'{pos}_{player_id:03d}',
                'Pos': pos,
                'Team': np.random.choice(teams),
                'Salary': salary,
                'Predicted_DK_Points': round(projected_points, 1),
                'Value': round(projected_points / (salary / 1000), 2)
            })
            player_id += 1
    
    return pd.DataFrame(players)

def demo_bayesian_modeling():
    """Demonstrate Bayesian player modeling"""
    print("\\n" + "="*60)
    print("🧠 BAYESIAN PLAYER MODELING DEMO")
    print("="*60)
    
    bayesian_model = BayesianPlayerModel()
    
    # Create sample player with realistic history
    player_name = "Mike_Trout"
    
    # Generate realistic historical performance (with streaks and slumps)
    np.random.seed(123)
    n_games = 50
    historical_points = []
    
    # Simulate hot/cold streaks
    current_state = 1  # 0=cold, 1=average, 2=hot
    state_duration = 0
    
    for game in range(n_games):
        # Change state occasionally
        if state_duration > np.random.poisson(8):  # Average streak length
            current_state = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
            state_duration = 0
        
        # Generate points based on state
        if current_state == 0:  # Cold
            points = np.random.normal(8, 3)
        elif current_state == 1:  # Average
            points = np.random.normal(15, 4)
        else:  # Hot
            points = np.random.normal(22, 5)
        
        points = max(0, points)
        historical_points.append(points)
        state_duration += 1
    
    # Update Bayesian model
    bayesian_model.update_player_model(player_name, historical_points, list(range(n_games)))
    
    print(f"📊 Analyzing {player_name} with {n_games} games of history...")
    print(f"   Historical Average: {np.mean(historical_points):.1f} points")
    print(f"   Historical Std Dev: {np.std(historical_points):.1f}")
    
    # Generate predictions
    predictions = bayesian_model.predict_player_distribution(player_name, 10000)
    
    print(f"\\n🔮 Bayesian Predictions:")
    print(f"   Expected Points: {np.mean(predictions):.1f}")
    print(f"   Prediction Std Dev: {np.std(predictions):.1f}")
    print(f"   Prob > 15 points: {np.mean(predictions >= 15):.1%}")
    print(f"   Prob > 20 points: {np.mean(predictions >= 20):.1%}")
    print(f"   95% Confidence Interval: [{np.percentile(predictions, 2.5):.1f}, {np.percentile(predictions, 97.5):.1f}]")
    
    return bayesian_model

def demo_hidden_markov_model():
    """Demonstrate Hidden Markov Model for player states"""
    print("\\n" + "="*60)
    print("🔄 HIDDEN MARKOV MODEL DEMO")
    print("="*60)
    
    hmm = HiddenMarkovModel()
    
    # Generate sample performance history with clear states
    np.random.seed(456)
    n_games = 30
    
    # Simulate realistic player states over time
    true_states = []
    performance_history = []
    state_names = ['Cold', 'Average', 'Hot']
    
    current_state = 1  # Start average
    
    for game in range(n_games):
        # State transition (tendency to stay in same state)
        if np.random.random() < 0.15:  # 15% chance to change state
            current_state = np.random.choice([0, 1, 2])
        
        true_states.append(current_state)
        
        # Generate performance based on state
        if current_state == 0:  # Cold
            performance = np.random.normal(8, 2)
        elif current_state == 1:  # Average
            performance = np.random.normal(13, 3)
        else:  # Hot
            performance = np.random.normal(19, 3)
        
        performance = max(0, performance)
        performance_history.append(performance)
    
    print(f"📈 Generated {n_games} games of performance data")
    print(f"   Average Performance: {np.mean(performance_history):.1f}")
    
    # Fit HMM
    hmm.fit_player_hmm(performance_history)
    
    print(f"\\n🧩 HMM Analysis:")
    print(f"   State Means: {hmm.emission_params['means']}")
    print(f"   State Std Devs: {np.sqrt(hmm.emission_params['variances'])}")
    print(f"   State Weights: {hmm.emission_params['weights']}")
    
    # Predict next performance from different states
    print(f"\\n🔮 Next Game Predictions:")
    for state in range(3):
        next_perf, next_state, state_name = hmm.predict_next_performance(state)
        print(f"   From {state_names[state]} state: {next_perf:.1f} points → {state_name}")
    
    return hmm

def demo_monte_carlo_simulation():
    """Demonstrate Monte Carlo simulation"""
    print("\\n" + "="*60)
    print("🎲 MONTE CARLO SIMULATION DEMO")
    print("="*60)
    
    # Create a sample lineup
    sample_lineup = pd.DataFrame([
        {'Name': 'P_001', 'Pos': 'P', 'Team': 'NYY', 'Salary': 8500, 'Predicted_DK_Points': 16.5},
        {'Name': 'P_002', 'Pos': 'P', 'Team': 'BOS', 'Salary': 6200, 'Predicted_DK_Points': 12.8},
        {'Name': 'C_001', 'Pos': 'C', 'Team': 'HOU', 'Salary': 4800, 'Predicted_DK_Points': 11.2},
        {'Name': '1B_001', 'Pos': '1B', 'Team': 'LAD', 'Salary': 6500, 'Predicted_DK_Points': 14.7},
        {'Name': '2B_001', 'Pos': '2B', 'Team': 'ATL', 'Salary': 5200, 'Predicted_DK_Points': 12.3},
        {'Name': '3B_001', 'Pos': '3B', 'Team': 'CLE', 'Salary': 5800, 'Predicted_DK_Points': 13.8},
        {'Name': 'SS_001', 'Pos': 'SS', 'Team': 'TB', 'Salary': 4900, 'Predicted_DK_Points': 11.9},
        {'Name': 'OF_001', 'Pos': 'OF', 'Team': 'SD', 'Salary': 5500, 'Predicted_DK_Points': 13.1},
        {'Name': 'OF_002', 'Pos': 'OF', 'Team': 'TOR', 'Salary': 4200, 'Predicted_DK_Points': 10.4},
        {'Name': 'OF_003', 'Pos': 'OF', 'Team': 'PHI', 'Salary': 3400, 'Predicted_DK_Points': 8.7}
    ])
    
    print(f"💰 Sample Lineup (Total Salary: ${sample_lineup['Salary'].sum():,}):")
    for _, player in sample_lineup.iterrows():
        print(f"   {player['Name']:8} {player['Pos']:3} ${player['Salary']:,} → {player['Predicted_DK_Points']:5.1f} pts")
    
    print(f"\\n📊 Projected Total: {sample_lineup['Predicted_DK_Points'].sum():.1f} points")
    
    # Run Monte Carlo simulation
    mc_simulator = MonteCarloSimulator(n_simulations=25000)
    simulation_results = mc_simulator.simulate_lineup_outcomes(sample_lineup)
    
    # Calculate probabilities
    probabilities = mc_simulator.calculate_outcome_probabilities()
    
    print(f"\\n🎲 Monte Carlo Results ({len(simulation_results):,} simulations):")
    print(f"   Mean Score: {probabilities['mean_score']:.1f}")
    print(f"   Median Score: {probabilities['median_score']:.1f}")
    print(f"   Std Deviation: {probabilities['std_score']:.1f}")
    print(f"   Min Score: {probabilities['min_score']:.1f}")
    print(f"   Max Score: {probabilities['max_score']:.1f}")
    
    print(f"\\n📈 Probability Analysis:")
    target_scores = [120, 140, 160, 180, 200]
    for target in target_scores:
        prob_key = f'prob_over_{target}'
        if prob_key in probabilities:
            print(f"   Prob > {target}: {probabilities[prob_key]:.1%}")
    
    print(f"\\n📉 Risk Metrics:")
    print(f"   VaR (95%): {probabilities['var_95']:.1f}")
    print(f"   VaR (99%): {probabilities['var_99']:.1f}")
    
    return simulation_results, probabilities

def demo_probabilistic_optimization():
    """Demonstrate probabilistic lineup optimization"""
    print("\\n" + "="*60)
    print("🎯 PROBABILISTIC OPTIMIZATION DEMO")
    print("="*60)
    
    # Create player data
    df_players = create_realistic_player_data()
    print(f"📊 Created {len(df_players)} players across all positions")
    
    # Initialize probabilistic optimizer
    prob_optimizer = ProbabilisticLineupOptimizer()
    
    # Run optimization
    optimal_lineup = prob_optimizer.optimize_lineup_with_probabilities(
        df_players, target_score=160, risk_tolerance=0.7
    )
    
    if not optimal_lineup.empty:
        print(f"\\n🏆 OPTIMAL LINEUP FOUND:")
        print(f"   Players: {len(optimal_lineup)}")
        print(f"   Total Salary: ${optimal_lineup['Salary'].sum():,}")
        print(f"   Expected Points: {optimal_lineup['expected_value'].sum():.1f}")
        print(f"   Risk-Adjusted Value: {optimal_lineup['risk_adjusted_value'].sum():.1f}")
        
        print(f"\\n👥 Lineup Composition:")
        for _, player in optimal_lineup.iterrows():
            print(f"   {player['Name']:12} {player['Pos']:3} {player['Team']:3} "
                  f"${player['Salary']:,} → {player['expected_value']:5.1f} pts "
                  f"(P>{player['prob_over_15']:.0%})")
        
        # Position and team breakdown
        pos_counts = optimal_lineup['Pos'].value_counts()
        team_counts = optimal_lineup['Team'].value_counts()
        
        print(f"\\n🎯 Position Breakdown: {dict(pos_counts)}")
        print(f"🏟️  Team Breakdown: {dict(team_counts)}")
        
        # Run Monte Carlo on optimal lineup
        print(f"\\n🎲 Running Monte Carlo simulation on optimal lineup...")
        mc_simulator = MonteCarloSimulator(n_simulations=10000)
        simulation_results = mc_simulator.simulate_lineup_outcomes(optimal_lineup)
        probabilities = mc_simulator.calculate_outcome_probabilities()
        
        print(f"\\n📈 Simulation Results:")
        print(f"   Mean Score: {probabilities['mean_score']:.1f}")
        print(f"   Std Deviation: {probabilities['std_score']:.1f}")
        print(f"   Prob > 140: {probabilities.get('prob_over_140', 0):.1%}")
        print(f"   Prob > 160: {probabilities.get('prob_over_160', 0):.1%}")
        print(f"   VaR (95%): {probabilities['var_95']:.1f}")
        
        return optimal_lineup, simulation_results
    
    else:
        print("❌ No optimal lineup found")
        return None, None

def demo_probability_analysis():
    """Demonstrate comprehensive probability analysis"""
    print("\\n" + "="*60)
    print("📊 COMPREHENSIVE PROBABILITY ANALYSIS")
    print("="*60)
    
    # Get lineup from previous demo
    df_players = create_realistic_player_data()
    prob_optimizer = ProbabilisticLineupOptimizer()
    optimal_lineup = prob_optimizer.optimize_lineup_with_probabilities(df_players)
    
    if optimal_lineup.empty:
        print("❌ No lineup available for analysis")
        return
    
    # Run simulation
    mc_simulator = MonteCarloSimulator(n_simulations=15000)
    simulation_results = mc_simulator.simulate_lineup_outcomes(optimal_lineup)
    
    # Full analysis
    analyzer = ProbabilityAnalyzer()
    analysis = analyzer.analyze_lineup_probabilities(optimal_lineup, simulation_results)
    
    # Display results
    print(f"\\n📈 DESCRIPTIVE STATISTICS:")
    desc_stats = analysis['descriptive_stats']
    print(f"   Mean: {desc_stats['mean']:.1f}")
    print(f"   Median: {desc_stats['median']:.1f}")
    print(f"   Std Dev: {desc_stats['std']:.1f}")
    print(f"   Skewness: {desc_stats['skewness']:.3f}")
    print(f"   Kurtosis: {desc_stats['kurtosis']:.3f}")
    
    print(f"\\n📉 RISK METRICS:")
    risk_metrics = analysis['risk_metrics']
    print(f"   VaR (95%): {risk_metrics['var_95']:.1f}")
    print(f"   CVaR (95%): {risk_metrics['cvar_95']:.1f}")
    print(f"   Downside Deviation: {risk_metrics['downside_deviation']:.1f}")
    print(f"   Upside Potential: {risk_metrics['upside_potential']:.1f}")
    
    print(f"\\n🎯 PROBABILITY BANDS:")
    prob_bands = analysis['probability_bands']
    for band, prob in prob_bands.items():
        print(f"   Score {band}: {prob:.1%}")
    
    print(f"\\n🎭 SCENARIO ANALYSIS:")
    scenarios = analysis['scenario_analysis']
    print(f"   Best Case (95th %ile): {scenarios['best_case']:.1f}")
    print(f"   Optimistic (75th %ile): {scenarios['optimistic']:.1f}")
    print(f"   Base Case (median): {scenarios['base_case']:.1f}")
    print(f"   Pessimistic (25th %ile): {scenarios['pessimistic']:.1f}")
    print(f"   Worst Case (5th %ile): {scenarios['worst_case']:.1f}")
    
    print(f"\\n👥 PLAYER CONTRIBUTION ANALYSIS:")
    contributions = analysis['player_contributions']
    sorted_contributions = sorted(contributions.items(), 
                                key=lambda x: x[1]['contribution_to_variance'], 
                                reverse=True)
    
    print(f"   Top Risk Contributors:")
    for i, (player, stats) in enumerate(sorted_contributions[:5]):
        print(f"   {i+1}. {player}: {stats['mean']:.1f} ± {stats['std']:.1f} "
              f"(variance: {stats['contribution_to_variance']:.1f})")

def main():
    """Run all probability modeling demos"""
    print("🎲 ADVANCED PROBABILITY MODELING DEMONSTRATION")
    print("=" * 70)
    
    # Run all demos
    bayesian_model = demo_bayesian_modeling()
    hmm_model = demo_hidden_markov_model()
    simulation_results, probabilities = demo_monte_carlo_simulation()
    optimal_lineup, lineup_simulation = demo_probabilistic_optimization()
    demo_probability_analysis()
    
    print("\\n" + "="*70)
    print("🎯 PROBABILITY MODELING DEMONSTRATION COMPLETE!")
    print("="*70)
    
    print("\\n📋 FEATURES DEMONSTRATED:")
    print("✅ Bayesian Player Performance Modeling")
    print("✅ Hidden Markov Model State Detection")  
    print("✅ Monte Carlo Simulation (25,000 iterations)")
    print("✅ Probabilistic Lineup Optimization")
    print("✅ Comprehensive Risk Analysis")
    print("✅ Scenario Analysis & Stress Testing")
    print("✅ Player Contribution Analysis")
    print("✅ VaR and CVaR Risk Metrics")
    
    print("\\n🚀 NEXT STEPS:")
    print("• Integrate with real historical player data")
    print("• Add dynamic probability updates during games")
    print("• Implement ensemble modeling techniques")
    print("• Add correlation-aware portfolio optimization")
    print("• Create real-time probability monitoring")

if __name__ == "__main__":
    main()
