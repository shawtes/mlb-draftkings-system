"""
Comprehensive Demo of Advanced Probability Modeling for MLB DFS
Shows all the probability modeling features in action
"""

import numpy as np
import pandas as pd
import sys
import os

def create_realistic_player_data():
    """Create realistic MLB player data for demonstration"""
    np.random.seed(42)
    
    # Define realistic player pools by position
    position_configs = {
        'P': {'count': 20, 'salary_range': (4000, 12000), 'points_range': (8, 25)},
        'C': {'count': 12, 'salary_range': (2500, 6000), 'points_range': (6, 16)},
        '1B': {'count': 15, 'salary_range': (3000, 8000), 'points_range': (7, 18)},
        '2B': {'count': 15, 'salary_range': (2500, 7000), 'points_range': (6, 17)},
        '3B': {'count': 15, 'salary_range': (3000, 8500), 'points_range': (7, 19)},
        'SS': {'count': 15, 'salary_range': (2800, 7500), 'points_range': (6, 18)},
        'OF': {'count': 25, 'salary_range': (2500, 8000), 'points_range': (6, 20)}
    }
    
    teams = ['NYY', 'BOS', 'HOU', 'LAD', 'ATL', 'CLE', 'TB', 'SD', 'TOR', 'PHI',
             'CHC', 'STL', 'MIL', 'WSH', 'NYM', 'SF', 'COL', 'ARI', 'SEA', 'TEX']
    
    players = []
    player_id = 1
    
    # Create realistic player names
    first_names = ['Mike', 'Juan', 'Jose', 'Carlos', 'Alex', 'David', 'Luis', 'Aaron', 
                   'Tyler', 'Jacob', 'Ryan', 'Matt', 'Chris', 'Nick', 'Andrew']
    last_names = ['Smith', 'Johnson', 'Garcia', 'Rodriguez', 'Martinez', 'Anderson',
                  'Taylor', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson']
    
    for position, config in position_configs.items():
        for i in range(config['count']):
            # Generate realistic salary
            salary = np.random.randint(config['salary_range'][0], config['salary_range'][1])
            
            # Generate realistic projected points based on salary
            salary_factor = (salary - config['salary_range'][0]) / (config['salary_range'][1] - config['salary_range'][0])
            
            base_points = config['points_range'][0] + salary_factor * (config['points_range'][1] - config['points_range'][0])
            
            # Add realistic variance
            projected_points = base_points + np.random.normal(0, 1.5)
            projected_points = max(config['points_range'][0], projected_points)
            
            # Create realistic name
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            
            players.append({
                'Name': f'{first_name} {last_name}',
                'Pos': position,
                'Team': np.random.choice(teams),
                'Salary': salary,
                'Predicted_DK_Points': round(projected_points, 1),
                'Value': round(projected_points / (salary / 1000), 2)
            })
    
    return pd.DataFrame(players)

def demo_comprehensive_probability_modeling():
    """Comprehensive demonstration of all probability modeling features"""
    print("🎲 COMPREHENSIVE PROBABILITY MODELING DEMO")
    print("=" * 70)
    
    # Import the probability modeling engine
    try:
        from probability_modeling_engine import ProbabilityModelingEngine
        from probability_integration_new import ProbabilityModelingIntegrator
        print("✅ All probability modeling modules loaded successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Create realistic player data
    print("\\n📊 Creating realistic MLB player dataset...")
    df_players = create_realistic_player_data()
    print(f"✅ Created dataset with {len(df_players)} players")
    
    # Show dataset summary
    print("\\n📋 Dataset Summary:")
    position_counts = df_players['Pos'].value_counts()
    for pos, count in position_counts.items():
        avg_salary = df_players[df_players['Pos'] == pos]['Salary'].mean()
        avg_points = df_players[df_players['Pos'] == pos]['Predicted_DK_Points'].mean()
        print(f"  {pos:3}: {count:2} players | Avg Salary: ${avg_salary:,.0f} | Avg Points: {avg_points:.1f}")
    
    # Initialize probability modeling engine
    print("\\n🔬 Initializing Advanced Probability Modeling Engine...")
    engine = ProbabilityModelingEngine()
    
    # Step 1: Bayesian Player Modeling
    print("\\n🎯 STEP 1: BAYESIAN PLAYER MODELING")
    print("-" * 50)
    models = engine.bayesian_player_modeling(df_players)
    
    # Show detailed model results for different positions
    print("\\n📊 Detailed Bayesian Model Results:")
    
    sample_players = []
    for pos in ['P', 'C', '1B', 'OF']:
        pos_players = df_players[df_players['Pos'] == pos]
        if not pos_players.empty:
            sample_players.append(pos_players.iloc[0])
    
    for player in sample_players:
        name = player['Name']
        if name in models:
            model = models[name]
            print(f"\\n  {name} ({player['Pos']}):")
            print(f"    Salary: ${player['Salary']:,}")
            print(f"    Projected: {model['projected_mean']:.1f} pts")
            print(f"    Bayesian Mean: {model['posterior_params']['mean']:.1f} pts")
            print(f"    Uncertainty (σ): {model['posterior_params']['std']:.1f}")
            print(f"    95% CI: [{model['confidence_interval']['lower']:.1f}, {model['confidence_interval']['upper']:.1f}]")
            print(f"    CI Width: {model['confidence_interval']['width']:.1f}")
    
    # Step 2: Correlation Modeling
    print("\\n🔗 STEP 2: CORRELATION MODELING")
    print("-" * 50)
    correlation_matrix = engine.correlation_modeling(df_players)
    
    # Analyze correlations
    non_diagonal = correlation_matrix[correlation_matrix != 1]
    print(f"  Correlation Matrix: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
    print(f"  Average Correlation: {np.mean(non_diagonal):.3f}")
    print(f"  Max Correlation: {np.max(non_diagonal):.3f}")
    print(f"  Min Correlation: {np.min(non_diagonal):.3f}")
    print(f"  Std Deviation: {np.std(non_diagonal):.3f}")
    
    # Find highly correlated pairs
    high_corr_threshold = 0.3
    high_corr_pairs = []
    n_players = len(df_players)
    
    for i in range(n_players):
        for j in range(i+1, n_players):
            if correlation_matrix[i, j] > high_corr_threshold:
                player1 = df_players.iloc[i]
                player2 = df_players.iloc[j]
                high_corr_pairs.append((player1['Name'], player2['Name'], correlation_matrix[i, j]))
    
    print(f"\\n  High Correlation Pairs (>{high_corr_threshold}):")
    for name1, name2, corr in high_corr_pairs[:5]:  # Show top 5
        print(f"    {name1} ↔ {name2}: {corr:.3f}")
    
    # Step 3: Monte Carlo Simulation
    print("\\n🎲 STEP 3: MONTE CARLO SIMULATION")
    print("-" * 50)
    
    # Create optimal lineup based on value
    df_players['value_score'] = df_players['Predicted_DK_Points'] / (df_players['Salary'] / 1000)
    
    # Build a balanced lineup (simplified position requirements)
    lineup_positions = {'P': 2, 'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 3}
    optimal_lineup = []
    
    for pos, count in lineup_positions.items():
        pos_players = df_players[df_players['Pos'] == pos].nlargest(count, 'value_score')
        optimal_lineup.extend(pos_players.to_dict('records'))
    
    optimal_lineup_df = pd.DataFrame(optimal_lineup)
    
    print(f"\\n📋 Optimal Lineup (Value-Based):")
    total_salary = optimal_lineup_df['Salary'].sum()
    total_projected = optimal_lineup_df['Predicted_DK_Points'].sum()
    
    for _, player in optimal_lineup_df.iterrows():
        print(f"  {player['Name']:20} {player['Pos']:3} ${player['Salary']:5,} → {player['Predicted_DK_Points']:5.1f} pts (Value: {player['value_score']:.2f})")
    
    print(f"\\n  Total Salary: ${total_salary:,} ({total_salary/50000*100:.1f}% of cap)")
    print(f"  Total Projected: {total_projected:.1f} points")
    
    # Run Monte Carlo simulation
    print("\\n🎲 Running Monte Carlo Simulation (10,000 iterations)...")
    mc_results = engine.monte_carlo_simulation(optimal_lineup_df, n_simulations=10000)
    
    # Display comprehensive Monte Carlo results
    stats = mc_results['statistics']
    risk_metrics = mc_results['risk_metrics']
    prob_analysis = mc_results['probability_analysis']
    
    print("\\n📈 MONTE CARLO RESULTS:")
    print(f"  Expected Points: {stats['mean']:.1f}")
    print(f"  Median Points: {stats['median']:.1f}")
    print(f"  Standard Deviation: {stats['std']:.1f}")
    print(f"  Min Simulated: {stats['min']:.1f}")
    print(f"  Max Simulated: {stats['max']:.1f}")
    
    print("\\n📊 PERCENTILE ANALYSIS:")
    for pct, value in stats['percentiles'].items():
        print(f"  {pct:>4} Percentile: {value:.1f} points")
    
    print("\\n⚠️ RISK METRICS:")
    print(f"  Value at Risk (95%): {risk_metrics['value_at_risk_95']:.1f} points")
    print(f"  Value at Risk (99%): {risk_metrics['value_at_risk_99']:.1f} points")
    print(f"  Conditional VaR (95%): {risk_metrics['conditional_var_95']:.1f} points")
    print(f"  Conditional VaR (99%): {risk_metrics['conditional_var_99']:.1f} points")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {risk_metrics['max_drawdown']:.1f} points")
    print(f"  Tail Ratio: {risk_metrics['tail_ratio']:.2f}")
    
    print("\\n📈 PROBABILITY ANALYSIS:")
    for threshold, prob in prob_analysis['threshold_probabilities'].items():
        threshold_value = int(threshold.split('_')[-1])
        print(f"  Probability of ≥{threshold_value:3} points: {prob:6.1%}")
    
    print(f"\\n💡 INTERPRETATION:")
    print(f"  • {prob_analysis['value_at_risk_interpretation']}")
    print(f"  • Expected value: {prob_analysis['expected_value']:.1f} points")
    print(f"  • Probability of above-average performance: {prob_analysis['prob_above_mean']:.1%}")
    
    # Step 4: Regime Detection
    print("\\n🔍 STEP 4: REGIME DETECTION")
    print("-" * 50)
    regimes, current_regime = engine.regime_detection()
    
    print("  Market Regimes Identified:")
    for regime_name, regime_data in regimes.items():
        prob = regime_data['probability']
        adj = regime_data['adjustment']
        status = "← CURRENT" if regime_name == current_regime else ""
        print(f"    {regime_name:12}: {prob:.0%} probability, {adj:.0%} adjustment {status}")
    
    # Update probabilities based on regime
    print(f"\\n  Updating player probabilities for '{current_regime}' regime...")
    engine.update_probabilities_with_regime(current_regime)
    
    # Show regime-adjusted results for sample players
    print("\\n  Regime-Adjusted Player Projections:")
    for i, player in enumerate(optimal_lineup_df.head(3).itertuples()):
        name = player.Name
        if name in engine.player_models:
            model = engine.player_models[name]
            original = model['projected_mean']
            adjusted = model['posterior_params']['mean']
            change = ((adjusted - original) / original) * 100
            print(f"    {name:20}: {original:.1f} → {adjusted:.1f} pts ({change:+.1f}%)")
    
    # Step 5: Advanced Portfolio Optimization
    print("\\n🎯 STEP 5: PROBABILITY-BASED OPTIMIZATION")
    print("-" * 50)
    
    print("  Running probability-based lineup optimization...")
    selected_players, optimal_weights = engine.optimize_lineup_with_probabilities(df_players)
    
    if selected_players is not None and len(selected_players) > 0:
        print(f"\\n  Optimized Lineup ({len(selected_players)} players):")
        
        # Calculate portfolio metrics
        total_salary_opt = selected_players['Salary'].sum()
        total_points_opt = selected_players['Predicted_DK_Points'].sum()
        
        for _, player in selected_players.iterrows():
            name = player['Name']
            weight = optimal_weights[df_players[df_players['Name'] == name].index[0]] if len(optimal_weights) > 0 else 0
            print(f"    {name:20} {player['Pos']:3} ${player['Salary']:5,} → {player['Predicted_DK_Points']:5.1f} pts (Weight: {weight:.3f})")
        
        print(f"\\n  Optimized Portfolio Metrics:")
        print(f"    Total Salary: ${total_salary_opt:,}")
        print(f"    Total Points: {total_points_opt:.1f}")
        print(f"    Salary Efficiency: {total_salary_opt/50000*100:.1f}%")
    else:
        print("  ⚠️ Optimization failed - using value-based approach")
    
    # Summary and Insights
    print("\\n🎯 SUMMARY AND INSIGHTS")
    print("=" * 70)
    
    print("\\n✅ ACCOMPLISHED:")
    print("  • Built Bayesian models for all players with uncertainty quantification")
    print("  • Modeled correlations between players based on team, position, and salary")
    print("  • Ran 10,000 Monte Carlo simulations for robust risk assessment")
    print("  • Calculated comprehensive risk metrics (VaR, CVaR, Sharpe, Sortino)")
    print("  • Detected market regimes and adjusted probabilities accordingly")
    print("  • Performed probability-based portfolio optimization")
    
    print("\\n🔬 KEY INSIGHTS:")
    print(f"  • Expected lineup performance: {stats['mean']:.1f} ± {stats['std']:.1f} points")
    print(f"  • 95% confidence range: {stats['percentiles']['5th']:.1f} to {stats['percentiles']['95th']:.1f} points")
    print(f"  • Risk-adjusted performance (Sharpe): {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  • Downside risk (VaR 95%): {risk_metrics['value_at_risk_95']:.1f} points")
    print(f"  • Average player correlation: {np.mean(non_diagonal):.3f}")
    
    print("\\n💡 PRACTICAL APPLICATIONS:")
    print("  • Bankroll management using Kelly Criterion and VaR")
    print("  • Risk-aware lineup construction with correlation modeling")
    print("  • Confidence intervals for performance expectations")
    print("  • Regime-adaptive strategy adjustment")
    print("  • Portfolio diversification across player types")
    
    print("\\n🎯 PROBABILITY MODELING DEMO COMPLETE!")
    print("\\nThis advanced system provides institutional-grade")
    print("probability modeling for DFS optimization! 🚀")

if __name__ == "__main__":
    demo_comprehensive_probability_modeling()
