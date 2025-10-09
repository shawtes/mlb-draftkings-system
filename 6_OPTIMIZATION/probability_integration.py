"""
Probability Modeling Integration for MLB DFS Optimizer
====================================================

This module integrates advanced probability modeling with the existing optimizer:
- Bayesian player performance modeling
- Monte Carlo simulation for lineup analysis
- Probabilistic optimization strategies
- Risk-aware lineup construction
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from probability_modeling import (
    BayesianPlayerModel, 
    MonteCarloSimulator, 
    ProbabilisticLineupOptimizer, 
    ProbabilityAnalyzer
)

class ProbabilityModelingIntegrator:
    """
    Integration layer for probability modeling in the main optimizer
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.bayesian_model = BayesianPlayerModel()
        self.monte_carlo = MonteCarloSimulator()
        self.probabilistic_optimizer = ProbabilisticLineupOptimizer()
        self.probability_analyzer = ProbabilityAnalyzer()
        self.use_probability_modeling = False
        
        # Configuration
        self.monte_carlo_sims = 10000
        self.target_score = 150
        self.risk_tolerance = 0.5
        self.confidence_levels = [0.90, 0.95, 0.99]
        
        print("✅ Probability Modeling Integrator initialized")
    
    def add_probability_modeling_tab(self, tabs_widget):
        """Add probability modeling tab to the main GUI"""
        from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                                   QGroupBox, QGridLayout, QLabel, 
                                   QCheckBox, QSpinBox, QDoubleSpinBox,
                                   QComboBox, QPushButton, QTextEdit)
        
        prob_tab = QWidget()
        tabs_widget.addTab(prob_tab, "🎲 Probability Modeling")
        
        layout = QVBoxLayout(prob_tab)
        
        # Enable/Disable toggle
        self.probability_enabled = QCheckBox("Enable Advanced Probability Modeling")
        self.probability_enabled.setToolTip("Use Bayesian inference, Monte Carlo simulation, and probabilistic optimization")
        self.probability_enabled.setChecked(False)
        self.probability_enabled.stateChanged.connect(self.toggle_probability_modeling)
        layout.addWidget(self.probability_enabled)
        
        # Status label
        self.prob_status_label = QLabel("Probability modeling disabled")
        self.prob_status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.prob_status_label)
        
        # Monte Carlo Parameters
        mc_group = QGroupBox("Monte Carlo Simulation")
        mc_layout = QGridLayout(mc_group)
        
        mc_layout.addWidget(QLabel("Simulations:"), 0, 0)
        self.mc_simulations = QSpinBox()
        self.mc_simulations.setRange(1000, 100000)
        self.mc_simulations.setValue(10000)
        self.mc_simulations.setSingleStep(1000)
        mc_layout.addWidget(self.mc_simulations, 0, 1)
        
        mc_layout.addWidget(QLabel("Target Score:"), 1, 0)
        self.target_score_spin = QSpinBox()
        self.target_score_spin.setRange(100, 300)
        self.target_score_spin.setValue(150)
        mc_layout.addWidget(self.target_score_spin, 1, 1)
        
        mc_layout.addWidget(QLabel("Risk Tolerance:"), 2, 0)
        self.risk_tolerance_spin = QDoubleSpinBox()
        self.risk_tolerance_spin.setRange(0.1, 2.0)
        self.risk_tolerance_spin.setValue(0.5)
        self.risk_tolerance_spin.setSingleStep(0.1)
        mc_layout.addWidget(self.risk_tolerance_spin, 2, 1)
        
        layout.addWidget(mc_group)
        
        # Bayesian Parameters
        bayesian_group = QGroupBox("Bayesian Modeling")
        bayesian_layout = QGridLayout(bayesian_group)
        
        bayesian_layout.addWidget(QLabel("Prior Strength:"), 0, 0)
        self.prior_strength = QDoubleSpinBox()
        self.prior_strength.setRange(0.1, 10.0)
        self.prior_strength.setValue(1.0)
        self.prior_strength.setSingleStep(0.1)
        bayesian_layout.addWidget(self.prior_strength, 0, 1)
        
        bayesian_layout.addWidget(QLabel("Update Method:"), 1, 0)
        self.update_method = QComboBox()
        self.update_method.addItems(["Conjugate Prior", "Variational Bayes", "MCMC"])
        bayesian_layout.addWidget(self.update_method, 1, 1)
        
        layout.addWidget(bayesian_group)
        
        # Probability Analysis
        analysis_group = QGroupBox("Probability Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("🎲 Run Probability Analysis")
        self.run_analysis_btn.clicked.connect(self.run_probability_analysis)
        self.run_analysis_btn.setEnabled(False)
        analysis_layout.addWidget(self.run_analysis_btn)
        
        # Results display
        self.analysis_results = QTextEdit()
        self.analysis_results.setMaximumHeight(200)
        self.analysis_results.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_results)
        
        layout.addWidget(analysis_group)
        
        return prob_tab
    
    def toggle_probability_modeling(self, enabled):
        """Toggle probability modeling on/off"""
        self.use_probability_modeling = enabled
        
        if enabled:
            self.prob_status_label.setText("✅ Probability modeling ENABLED")
            self.prob_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.run_analysis_btn.setEnabled(True)
        else:
            self.prob_status_label.setText("❌ Probability modeling DISABLED")
            self.prob_status_label.setStyleSheet("color: gray;")
            self.run_analysis_btn.setEnabled(False)
    
    def run_probability_analysis(self):
        """Run probability analysis on current player pool"""
        if not hasattr(self.parent_app, 'df_players') or self.parent_app.df_players is None:
            self.analysis_results.setText("❌ No player data available. Please load players first.")
            return
        
        try:
            # Get current parameters
            self.monte_carlo_sims = self.mc_simulations.value()
            self.target_score = self.target_score_spin.value()
            self.risk_tolerance = self.risk_tolerance_spin.value()
            
            # Run analysis
            results = self.analyze_player_probabilities(self.parent_app.df_players)
            
            # Display results
            self.display_analysis_results(results)
            
        except Exception as e:
            self.analysis_results.setText(f"❌ Error during analysis: {str(e)}")
            logging.error(f"Probability analysis error: {e}")
    
    def analyze_player_probabilities(self, df_players):
        """Analyze probability distributions for all players"""
        print("🎲 Running probability analysis...")
        
        # Build Bayesian models
        self.build_bayesian_models(df_players)
        
        # Calculate probabilistic metrics
        enhanced_df = self.calculate_player_probabilities(df_players)
        
        # Generate optimal lineup using probabilistic methods
        optimal_lineup = self.probabilistic_optimizer.optimize_lineup_with_probabilities(
            enhanced_df, self.target_score, self.risk_tolerance
        )
        
        # Run Monte Carlo simulation on optimal lineup
        simulation_results = []
        if not optimal_lineup.empty:
            simulation_results = self.monte_carlo.simulate_lineup_outcomes(
                optimal_lineup, self.bayesian_model
            )
        
        # Calculate outcome probabilities
        probabilities = {}
        if simulation_results:
            probabilities = self.monte_carlo.calculate_outcome_probabilities()
        
        # Full probability analysis
        analysis = {}
        if simulation_results:
            analysis = self.probability_analyzer.analyze_lineup_probabilities(
                optimal_lineup, simulation_results
            )
        
        return {
            'enhanced_df': enhanced_df,
            'optimal_lineup': optimal_lineup,
            'simulation_results': simulation_results,
            'probabilities': probabilities,
            'analysis': analysis
        }
    
    def build_bayesian_models(self, df_players):
        """Build Bayesian models for all players"""
        for _, player in df_players.iterrows():
            # Generate mock historical data (in production, use real data)
            projected = player['Predicted_DK_Points']
            n_games = 20
            
            # Create realistic historical performance with trends
            historical_points = []
            for i in range(n_games):
                # Add seasonality and noise
                base_performance = projected + np.sin(i * 0.3) * 2
                performance = np.random.normal(base_performance, projected * 0.25)
                historical_points.append(max(0, performance))
            
            # Update Bayesian model
            self.bayesian_model.update_player_model(
                player['Name'], historical_points, list(range(n_games))
            )
    
    def calculate_player_probabilities(self, df_players):
        """Calculate probabilistic metrics for each player"""
        enhanced_df = df_players.copy()
        
        # Add probability columns
        prob_columns = [
            'prob_over_10', 'prob_over_15', 'prob_over_20', 'prob_over_25',
            'expected_value', 'value_variance', 'risk_adjusted_value',
            'kelly_weight', 'sharpe_ratio', 'downside_risk'
        ]
        
        for col in prob_columns:
            enhanced_df[col] = 0.0
        
        for idx, player in enhanced_df.iterrows():
            # Get probability distribution
            distribution = self.bayesian_model.predict_player_distribution(
                player['Name'], n_samples=1000
            )
            
            # Calculate probabilities
            enhanced_df.at[idx, 'prob_over_10'] = np.mean(distribution >= 10)
            enhanced_df.at[idx, 'prob_over_15'] = np.mean(distribution >= 15)
            enhanced_df.at[idx, 'prob_over_20'] = np.mean(distribution >= 20)
            enhanced_df.at[idx, 'prob_over_25'] = np.mean(distribution >= 25)
            
            # Expected value and variance
            expected_value = np.mean(distribution)
            value_variance = np.var(distribution)
            enhanced_df.at[idx, 'expected_value'] = expected_value
            enhanced_df.at[idx, 'value_variance'] = value_variance
            
            # Risk-adjusted metrics
            enhanced_df.at[idx, 'risk_adjusted_value'] = expected_value - 0.5 * value_variance
            
            # Kelly criterion weight
            win_prob = np.mean(distribution >= player['Predicted_DK_Points'])
            if win_prob > 0:
                kelly_weight = min(0.25, max(0.01, win_prob - 0.1))
            else:
                kelly_weight = 0.01
            enhanced_df.at[idx, 'kelly_weight'] = kelly_weight
            
            # Sharpe ratio (risk-adjusted return)
            if value_variance > 0:
                sharpe_ratio = expected_value / np.sqrt(value_variance)
            else:
                sharpe_ratio = 0
            enhanced_df.at[idx, 'sharpe_ratio'] = sharpe_ratio
            
            # Downside risk
            downside_outcomes = distribution[distribution < expected_value]
            if len(downside_outcomes) > 0:
                downside_risk = np.std(downside_outcomes)
            else:
                downside_risk = 0
            enhanced_df.at[idx, 'downside_risk'] = downside_risk
        
        return enhanced_df
    
    def display_analysis_results(self, results):
        """Display analysis results in the GUI"""
        if not results:
            self.analysis_results.setText("❌ No analysis results available")
            return
        
        output = []
        output.append("🎲 PROBABILITY ANALYSIS RESULTS")
        output.append("=" * 50)
        
        # Player probabilities summary
        enhanced_df = results.get('enhanced_df', pd.DataFrame())
        if not enhanced_df.empty:
            output.append(f"\\n📊 PLAYER ANALYSIS ({len(enhanced_df)} players):")
            output.append(f"  Average Expected Value: {enhanced_df['expected_value'].mean():.2f}")
            output.append(f"  Average Prob > 15 pts: {enhanced_df['prob_over_15'].mean():.1%}")
            output.append(f"  Average Sharpe Ratio: {enhanced_df['sharpe_ratio'].mean():.3f}")
            
            # Top players by probability metrics
            top_expected = enhanced_df.nlargest(3, 'expected_value')
            output.append(f"\\n  🏆 Top Expected Value Players:")
            for _, player in top_expected.iterrows():
                output.append(f"    {player['Name']}: {player['expected_value']:.1f} pts")
        
        # Optimal lineup analysis
        optimal_lineup = results.get('optimal_lineup', pd.DataFrame())
        if not optimal_lineup.empty:
            output.append(f"\\n🎯 OPTIMAL LINEUP ANALYSIS:")
            output.append(f"  Total Salary: ${optimal_lineup['Salary'].sum():,}")
            output.append(f"  Expected Points: {optimal_lineup['expected_value'].sum():.1f}")
            output.append(f"  Risk-Adjusted Value: {optimal_lineup['risk_adjusted_value'].sum():.1f}")
            
            # Position breakdown
            pos_counts = optimal_lineup['Pos'].value_counts()
            output.append(f"  Position Breakdown: {dict(pos_counts)}")
        
        # Monte Carlo simulation results
        probabilities = results.get('probabilities', {})
        if probabilities:
            output.append(f"\\n🎲 MONTE CARLO SIMULATION:")
            output.append(f"  Mean Score: {probabilities.get('mean_score', 0):.1f}")
            output.append(f"  Standard Deviation: {probabilities.get('std_score', 0):.1f}")
            output.append(f"  Probability > 120: {probabilities.get('prob_over_120', 0):.1%}")
            output.append(f"  Probability > 150: {probabilities.get('prob_over_150', 0):.1%}")
            output.append(f"  VaR (95%): {probabilities.get('var_95', 0):.1f}")
            output.append(f"  Max Potential: {probabilities.get('max_score', 0):.1f}")
        
        # Risk analysis
        analysis = results.get('analysis', {})
        if analysis:
            risk_metrics = analysis.get('risk_metrics', {})
            if risk_metrics:
                output.append(f"\\n📈 RISK ANALYSIS:")
                output.append(f"  Downside Deviation: {risk_metrics.get('downside_deviation', 0):.1f}")
                output.append(f"  Upside Potential: {risk_metrics.get('upside_potential', 0):.1f}")
                output.append(f"  CVaR (95%): {risk_metrics.get('cvar_95', 0):.1f}")
        
        # Display in GUI
        self.analysis_results.setText("\\n".join(output))
        
        # Log to console
        print("\\n".join(output))
    
    def optimize_lineups_with_probabilities(self, df_filtered, num_lineups=10):
        """
        Main method to optimize lineups using probability modeling
        """
        if not self.use_probability_modeling:
            return []
        
        logging.info("🎲 Starting probability-based lineup optimization")
        
        try:
            # Build Bayesian models
            self.build_bayesian_models(df_filtered)
            
            # Calculate probabilistic metrics
            enhanced_df = self.calculate_player_probabilities(df_filtered)
            
            # Generate multiple lineups using probabilistic methods
            lineups = []
            for i in range(num_lineups):
                # Vary parameters slightly for diversity
                risk_tolerance = self.risk_tolerance + np.random.normal(0, 0.1)
                risk_tolerance = max(0.1, min(2.0, risk_tolerance))
                
                lineup = self.probabilistic_optimizer.optimize_lineup_with_probabilities(
                    enhanced_df, self.target_score, risk_tolerance
                )
                
                if not lineup.empty:
                    # Run Monte Carlo simulation
                    simulation_results = self.monte_carlo.simulate_lineup_outcomes(
                        lineup, self.bayesian_model
                    )
                    
                    # Calculate probabilities
                    probabilities = self.monte_carlo.calculate_outcome_probabilities()
                    
                    lineup_result = {
                        'lineup_id': i,
                        'lineup': lineup,
                        'total_points': lineup['expected_value'].sum(),
                        'total_salary': lineup['Salary'].sum(),
                        'risk_metrics': {
                            'expected_value': lineup['expected_value'].sum(),
                            'total_variance': lineup['value_variance'].sum(),
                            'mean_score': probabilities.get('mean_score', 0),
                            'std_score': probabilities.get('std_score', 0),
                            'var_95': probabilities.get('var_95', 0),
                            'prob_over_150': probabilities.get('prob_over_150', 0)
                        },
                        'probabilities': probabilities,
                        'simulation_results': simulation_results
                    }
                    
                    lineups.append(lineup_result)
            
            logging.info(f"🎯 Generated {len(lineups)} probability-optimized lineups")
            return lineups
            
        except Exception as e:
            logging.error(f"❌ Error in probability-based optimization: {str(e)}")
            return []

def integrate_probability_modeling(main_app):
    """
    Main function to integrate probability modeling into the MLB optimizer
    """
    print("🎲 Integrating Probability Modeling...")
    
    # Create the integrator
    integrator = ProbabilityModelingIntegrator(main_app)
    
    # Add the probability modeling tab to the main GUI
    if hasattr(main_app, 'tabs'):
        integrator.add_probability_modeling_tab(main_app.tabs)
    
    # Store the integrator in the main app
    main_app.probability_integrator = integrator
    
    # Add method to check if probability modeling should be used
    def should_use_probability_modeling():
        return (hasattr(main_app, 'probability_integrator') and 
                main_app.probability_integrator.use_probability_modeling)
    
    main_app.should_use_probability_modeling = should_use_probability_modeling
    
    # Add method to run probability optimization
    def run_probability_optimization(df_filtered, num_lineups=10):
        if hasattr(main_app, 'probability_integrator'):
            return main_app.probability_integrator.optimize_lineups_with_probabilities(
                df_filtered, num_lineups
            )
        return []
    
    main_app.run_probability_optimization = run_probability_optimization
    
    print("✅ Probability Modeling integration complete!")
    
    return integrator

# Test the integration
if __name__ == "__main__":
    print("🧪 Testing Probability Modeling Integration...")
    
    # Create mock app
    class MockApp:
        def __init__(self):
            from PyQt5.QtWidgets import QTabWidget
            self.tabs = QTabWidget()
            self.df_players = None
    
    # Test integration
    mock_app = MockApp()
    integrator = integrate_probability_modeling(mock_app)
    
    print("🎯 Probability Modeling Integration test complete!")
