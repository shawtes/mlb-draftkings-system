"""
Integration of Advanced Probability Modeling with the Main Optimizer
This file provides a clean integration layer for adding probability modeling
to the existing optimizer without breaking the current functionality.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

# Import the probability modeling engine
try:
    from probability_modeling_engine import ProbabilityModelingEngine
    PROBABILITY_MODELING_AVAILABLE = True
    print("✅ Probability Modeling Engine loaded successfully!")
except ImportError as e:
    print(f"⚠️ Probability Modeling Engine not available: {e}")
    PROBABILITY_MODELING_AVAILABLE = False

class ProbabilityModelingIntegrator:
    """
    Integration layer for probability modeling in the main optimizer
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.probability_engine = None
        self.use_probability_modeling = False
        
        # Initialize the probability modeling engine if available
        if PROBABILITY_MODELING_AVAILABLE:
            try:
                self.probability_engine = ProbabilityModelingEngine()
                print("✅ Probability Modeling Engine initialized")
            except Exception as e:
                print(f"❌ Failed to initialize probability engine: {e}")
                self.probability_engine = None
        
        # Default parameters
        self.default_params = {
            'confidence_level': 0.95,
            'mc_simulations': 10000,
            'risk_tolerance': 1.0,
            'bayesian_updates': True,
            'correlation_modeling': True,
            'regime_detection': True
        }
    
    def add_probability_modeling_tab(self, tabs_widget):
        """Add the probability modeling tab to the main GUI"""
        prob_tab = QWidget()
        tabs_widget.addTab(prob_tab, "🎲 Probability Modeling")
        
        layout = QVBoxLayout(prob_tab)
        
        # Main toggle
        self.probability_enabled = QCheckBox("Enable Advanced Probability Modeling")
        self.probability_enabled.setToolTip("Use Bayesian inference, Monte Carlo simulation, and correlation modeling")
        self.probability_enabled.setChecked(False)
        self.probability_enabled.stateChanged.connect(self.toggle_probability_modeling)
        layout.addWidget(self.probability_enabled)
        
        # Status label
        self.status_label = QLabel("Probability modeling disabled")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # Parameters section
        params_group = QGroupBox("Probability Parameters")
        params_layout = QGridLayout(params_group)
        
        # Confidence Level
        params_layout.addWidget(QLabel("Confidence Level:"), 0, 0)
        self.confidence_level = QDoubleSpinBox()
        self.confidence_level.setRange(0.90, 0.99)
        self.confidence_level.setValue(0.95)
        self.confidence_level.setSingleStep(0.01)
        self.confidence_level.setDecimals(2)
        params_layout.addWidget(self.confidence_level, 0, 1)
        
        # Monte Carlo Simulations
        params_layout.addWidget(QLabel("MC Simulations:"), 1, 0)
        self.mc_simulations = QSpinBox()
        self.mc_simulations.setRange(1000, 50000)
        self.mc_simulations.setValue(10000)
        self.mc_simulations.setSingleStep(1000)
        params_layout.addWidget(self.mc_simulations, 1, 1)
        
        # Risk Tolerance
        params_layout.addWidget(QLabel("Risk Tolerance:"), 2, 0)
        self.risk_tolerance = QDoubleSpinBox()
        self.risk_tolerance.setRange(0.1, 2.0)
        self.risk_tolerance.setValue(1.0)
        self.risk_tolerance.setSingleStep(0.1)
        params_layout.addWidget(self.risk_tolerance, 2, 1)
        
        # Feature toggles
        self.bayesian_updates = QCheckBox("Bayesian Player Updates")
        self.bayesian_updates.setChecked(True)
        params_layout.addWidget(self.bayesian_updates, 3, 0, 1, 2)
        
        self.correlation_modeling = QCheckBox("Correlation Modeling")
        self.correlation_modeling.setChecked(True)
        params_layout.addWidget(self.correlation_modeling, 4, 0, 1, 2)
        
        self.regime_detection = QCheckBox("Regime Detection")
        self.regime_detection.setChecked(True)
        params_layout.addWidget(self.regime_detection, 5, 0, 1, 2)
        
        layout.addWidget(params_group)
        
        # Analysis section
        analysis_group = QGroupBox("Probability Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Probability Analysis")
        self.run_analysis_btn.clicked.connect(self.run_probability_analysis)
        analysis_layout.addWidget(self.run_analysis_btn)
        
        # Results display
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setMaximumHeight(200)
        analysis_layout.addWidget(self.analysis_results)
        
        layout.addWidget(analysis_group)
        
        # Library status
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        self.library_status = QLabel()
        self.update_library_status()
        status_layout.addWidget(self.library_status)
        
        layout.addWidget(status_group)
        
        # Initially disable controls
        self.toggle_probability_modeling(False)
        
        return prob_tab
    
    def toggle_probability_modeling(self, enabled):
        """Toggle probability modeling on/off"""
        self.use_probability_modeling = enabled and PROBABILITY_MODELING_AVAILABLE
        
        if enabled and PROBABILITY_MODELING_AVAILABLE:
            self.status_label.setText("✅ Probability modeling ENABLED")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif enabled and not PROBABILITY_MODELING_AVAILABLE:
            self.status_label.setText("⚠️ Probability modeling UNAVAILABLE - missing libraries")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.probability_enabled.setChecked(False)
            self.use_probability_modeling = False
        else:
            self.status_label.setText("❌ Probability modeling DISABLED")
            self.status_label.setStyleSheet("color: gray;")
    
    def update_library_status(self):
        """Update library availability status"""
        status_text = "Library Status:\\n"
        
        if PROBABILITY_MODELING_AVAILABLE:
            status_text += "✅ Probability Modeling Engine: Available\\n"
        else:
            status_text += "❌ Probability Modeling Engine: Missing\\n"
        
        # Check individual libraries
        libraries = [
            ("scipy", "SciPy (Statistical Functions)"),
            ("numpy", "NumPy (Numerical Computing)"),
            ("pandas", "Pandas (Data Analysis)"),
            ("sklearn", "Scikit-learn (Machine Learning)")
        ]
        
        for lib_name, lib_desc in libraries:
            try:
                __import__(lib_name)
                status_text += f"✅ {lib_desc}: Available\\n"
            except ImportError:
                status_text += f"❌ {lib_desc}: Missing\\n"
        
        self.library_status.setText(status_text)
        self.library_status.setStyleSheet("font-family: monospace; font-size: 10px;")
    
    def get_probability_parameters(self):
        """Get parameters from UI or use defaults"""
        if hasattr(self, 'confidence_level'):
            return {
                'confidence_level': self.confidence_level.value(),
                'mc_simulations': self.mc_simulations.value(),
                'risk_tolerance': self.risk_tolerance.value(),
                'bayesian_updates': self.bayesian_updates.isChecked(),
                'correlation_modeling': self.correlation_modeling.isChecked(),
                'regime_detection': self.regime_detection.isChecked()
            }
        else:
            return self.default_params
    
    def run_probability_analysis(self):
        """Run probability analysis on current player data"""
        if not self.use_probability_modeling or not self.probability_engine:
            self.analysis_results.setText("❌ Probability modeling not available")
            return
        
        # Get player data from parent app
        if not hasattr(self.parent_app, 'df_players') or self.parent_app.df_players is None:
            self.analysis_results.setText("❌ No player data available. Please load player data first.")
            return
        
        try:
            self.analysis_results.setText("🔄 Running probability analysis...")
            
            # Get parameters
            params = self.get_probability_parameters()
            
            # Set confidence level in engine
            self.probability_engine.confidence_level = params['confidence_level']
            
            # Build Bayesian models
            df_players = self.parent_app.df_players
            models = self.probability_engine.bayesian_player_modeling(df_players)
            
            # Run correlation modeling if enabled
            if params['correlation_modeling']:
                corr_matrix = self.probability_engine.correlation_modeling(df_players)
                avg_correlation = np.mean(corr_matrix[corr_matrix != 1])
                max_correlation = np.max(corr_matrix[corr_matrix != 1])
            else:
                avg_correlation = 0
                max_correlation = 0
            
            # Create sample lineup for Monte Carlo
            sample_lineup = df_players.head(10)  # First 10 players
            mc_results = self.probability_engine.monte_carlo_simulation(
                sample_lineup, 
                n_simulations=params['mc_simulations']
            )
            
            # Format results
            stats = mc_results['statistics']
            risk_metrics = mc_results['risk_metrics']
            prob_analysis = mc_results['probability_analysis']
            
            results_text = f"""
✅ PROBABILITY ANALYSIS COMPLETE

📊 PLAYER MODELS:
  • Built Bayesian models for {len(models)} players
  • Confidence Level: {params['confidence_level']:.0%}
  • Monte Carlo Simulations: {params['mc_simulations']:,}

🎲 MONTE CARLO RESULTS (Sample 10-Player Lineup):
  • Expected Total Points: {stats['mean']:.1f}
  • Standard Deviation: {stats['std']:.1f}
  • 95% Confidence Interval: [{stats['percentiles']['5th']:.1f}, {stats['percentiles']['95th']:.1f}]

⚠️ RISK METRICS:
  • Value at Risk (95%): {risk_metrics['value_at_risk_95']:.1f}
  • Conditional VaR (95%): {risk_metrics['conditional_var_95']:.1f}
  • Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}
  • Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}

🔗 CORRELATION ANALYSIS:
  • Average Correlation: {avg_correlation:.3f}
  • Maximum Correlation: {max_correlation:.3f}

📈 PROBABILITY THRESHOLDS:
  • Above 100 points: {prob_analysis['threshold_probabilities']['prob_above_100']:.1%}
  • Above 120 points: {prob_analysis['threshold_probabilities']['prob_above_120']:.1%}
  • Above 140 points: {prob_analysis['threshold_probabilities']['prob_above_140']:.1%}

💡 INTERPRETATION:
{prob_analysis['value_at_risk_interpretation']}
"""
            
            self.analysis_results.setText(results_text)
            
        except Exception as e:
            error_text = f"❌ Error in probability analysis: {str(e)}"
            self.analysis_results.setText(error_text)
            logging.error(f"Probability analysis error: {e}")
    
    def optimize_lineup_with_probabilities(self, df_filtered, num_lineups=10):
        """
        Optimize lineup using probability modeling
        """
        if not self.use_probability_modeling or not self.probability_engine:
            return []
        
        logging.info("🎲 Starting probability-based optimization")
        
        try:
            # Get parameters
            params = self.get_probability_parameters()
            
            # Set confidence level
            self.probability_engine.confidence_level = params['confidence_level']
            
            # Build Bayesian models
            models = self.probability_engine.bayesian_player_modeling(df_filtered)
            
            # Model correlations if enabled
            if params['correlation_modeling']:
                corr_matrix = self.probability_engine.correlation_modeling(df_filtered)
            
            # Detect regimes if enabled
            if params['regime_detection']:
                regimes, current_regime = self.probability_engine.regime_detection()
                self.probability_engine.update_probabilities_with_regime(current_regime)
            
            # Run Monte Carlo simulation for multiple lineups
            results = []
            
            for lineup_idx in range(num_lineups):
                # Create different lineup variations
                lineup_players = self._create_lineup_variation(df_filtered, lineup_idx)
                
                # Run Monte Carlo simulation
                mc_results = self.probability_engine.monte_carlo_simulation(
                    lineup_players, 
                    n_simulations=params['mc_simulations']
                )
                
                # Calculate lineup metrics
                stats = mc_results['statistics']
                risk_metrics = mc_results['risk_metrics']
                
                result = {
                    'lineup_id': lineup_idx,
                    'lineup': lineup_players,
                    'total_points': lineup_players['Predicted_DK_Points'].sum(),
                    'total_salary': lineup_players['Salary'].sum(),
                    'probability_metrics': {
                        'expected_points': stats['mean'],
                        'std_dev': stats['std'],
                        'confidence_interval': [stats['percentiles']['5th'], stats['percentiles']['95th']],
                        'value_at_risk': risk_metrics['value_at_risk_95'],
                        'conditional_var': risk_metrics['conditional_var_95'],
                        'sharpe_ratio': risk_metrics['sharpe_ratio'],
                        'sortino_ratio': risk_metrics['sortino_ratio']
                    }
                }
                
                results.append(result)
            
            # Sort by risk-adjusted score
            results.sort(key=lambda x: x['probability_metrics']['sharpe_ratio'], reverse=True)
            
            logging.info(f"🎯 Probability-based optimization complete. Generated {len(results)} lineups")
            return results
            
        except Exception as e:
            logging.error(f"❌ Error in probability-based optimization: {str(e)}")
            return []
    
    def _create_lineup_variation(self, df_filtered, variation_idx):
        """Create different lineup variations for testing"""
        # Simple variation: take different slices of players
        start_idx = variation_idx * 2
        end_idx = start_idx + 10
        
        if end_idx > len(df_filtered):
            # Wrap around if needed
            lineup_players = df_filtered.iloc[start_idx:].copy()
            remaining = 10 - len(lineup_players)
            lineup_players = pd.concat([lineup_players, df_filtered.iloc[:remaining]])
        else:
            lineup_players = df_filtered.iloc[start_idx:end_idx].copy()
        
        return lineup_players

def integrate_probability_modeling(main_app):
    """
    Main function to integrate probability modeling into the MLB optimizer
    """
    print("🎲 Integrating Probability Modeling...")
    
    # Create the integrator
    integrator = ProbabilityModelingIntegrator(main_app)
    
    # Add the probability tab to the main GUI
    if hasattr(main_app, 'tabs'):
        integrator.add_probability_modeling_tab(main_app.tabs)
    
    # Store the integrator in the main app
    main_app.probability_integrator = integrator
    
    # Add method to check if probability modeling should be used
    def should_use_probability_modeling():
        return (hasattr(main_app, 'probability_integrator') and 
                main_app.probability_integrator.use_probability_modeling)
    
    main_app.should_use_probability_modeling = should_use_probability_modeling
    
    # Add method to run probability-based optimization
    def run_probability_optimization(df_filtered, num_lineups=10):
        if hasattr(main_app, 'probability_integrator'):
            return main_app.probability_integrator.optimize_lineup_with_probabilities(
                df_filtered, num_lineups
            )
        return []
    
    main_app.run_probability_optimization = run_probability_optimization
    
    print("✅ Probability Modeling integration complete!")
    
    return integrator
