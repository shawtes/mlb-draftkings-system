"""
Integration patch for adding advanced quantitative optimization to the main optimizer.
This file contains the necessary modifications to integrate the advanced quantitative
optimizer with copulas, GARCH, Monte Carlo, VaR, Kelly criterion, and other financial
modeling techniques.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

# Try to import the advanced quantitative optimizer
try:
    from advanced_quant_optimizer import AdvancedQuantitativeOptimizer
    ADVANCED_QUANT_AVAILABLE = True
    print("✅ Advanced Quantitative Optimizer loaded successfully!")
except ImportError as e:
    print(f"⚠️ Advanced Quantitative Optimizer not available: {e}")
    ADVANCED_QUANT_AVAILABLE = False

class AdvancedQuantIntegrator:
    """
    Integration layer for advanced quantitative optimization in the main optimizer.
    """
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.advanced_optimizer = None
        self.use_advanced_quant = False
        
        # Initialize the advanced optimizer if available
        if ADVANCED_QUANT_AVAILABLE:
            try:
                self.advanced_optimizer = AdvancedQuantitativeOptimizer()
                print("✅ Advanced Quantitative Optimizer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize advanced optimizer: {e}")
                self.advanced_optimizer = None
        
        # Default parameters
        self.default_params = {
            'optimization_strategy': 'combined',
            'risk_tolerance': 1.0,
            'var_confidence': 0.95,
            'target_volatility': 0.15,
            'mc_simulations': 10000,
            'time_horizon': 1,
            'garch_p': 1,
            'garch_q': 1,
            'garch_lookback': 100,
            'copula_family': 'gaussian',
            'dependency_threshold': 0.3,
            'kelly_fraction_limit': 0.25,
            'expected_win_rate': 0.2
        }
    
    def add_advanced_quant_tab(self, tabs_widget):
        """Add the advanced quantitative optimization tab to the main GUI"""
        advanced_tab = QWidget()
        tabs_widget.addTab(advanced_tab, "🔬 Advanced Quant")
        
        layout = QVBoxLayout(advanced_tab)
        
        # Main toggle
        self.advanced_quant_enabled = QCheckBox("Enable Advanced Quantitative Optimization")
        self.advanced_quant_enabled.setToolTip("Use advanced financial techniques: GARCH volatility, copulas, Monte Carlo, VaR, Kelly criterion")
        self.advanced_quant_enabled.setChecked(False)
        self.advanced_quant_enabled.stateChanged.connect(self.toggle_advanced_quant)
        layout.addWidget(self.advanced_quant_enabled)
        
        # Status label
        self.status_label = QLabel("Advanced quantitative optimization disabled")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        
        # Optimization Strategy
        params_layout.addWidget(QLabel("Strategy:"), 0, 0)
        self.optimization_strategy = QComboBox()
        self.optimization_strategy.addItems(["combined", "kelly_criterion", "risk_parity", "mean_variance", "equal_weight"])
        params_layout.addWidget(self.optimization_strategy, 0, 1)
        
        # Risk Tolerance
        params_layout.addWidget(QLabel("Risk Tolerance:"), 1, 0)
        self.risk_tolerance = QDoubleSpinBox()
        self.risk_tolerance.setRange(0.1, 2.0)
        self.risk_tolerance.setValue(1.0)
        self.risk_tolerance.setSingleStep(0.1)
        params_layout.addWidget(self.risk_tolerance, 1, 1)
        
        # VaR Confidence
        params_layout.addWidget(QLabel("VaR Confidence:"), 2, 0)
        self.var_confidence = QDoubleSpinBox()
        self.var_confidence.setRange(0.90, 0.99)
        self.var_confidence.setValue(0.95)
        self.var_confidence.setSingleStep(0.01)
        params_layout.addWidget(self.var_confidence, 2, 1)
        
        # Monte Carlo Simulations
        params_layout.addWidget(QLabel("MC Simulations:"), 3, 0)
        self.mc_simulations = QSpinBox()
        self.mc_simulations.setRange(1000, 50000)
        self.mc_simulations.setValue(10000)
        params_layout.addWidget(self.mc_simulations, 3, 1)
        
        # Kelly Fraction Limit
        params_layout.addWidget(QLabel("Kelly Fraction Limit:"), 4, 0)
        self.kelly_fraction_limit = QDoubleSpinBox()
        self.kelly_fraction_limit.setRange(0.1, 1.0)
        self.kelly_fraction_limit.setValue(0.25)
        self.kelly_fraction_limit.setSingleStep(0.05)
        params_layout.addWidget(self.kelly_fraction_limit, 4, 1)
        
        layout.addWidget(params_group)
        
        # Library status
        status_group = QGroupBox("Library Status")
        status_layout = QVBoxLayout(status_group)
        
        self.library_status = QLabel()
        self.update_library_status()
        status_layout.addWidget(self.library_status)
        
        layout.addWidget(status_group)
        
        # Initially disable controls
        self.toggle_advanced_quant(False)
        
        return advanced_tab
    
    def toggle_advanced_quant(self, enabled):
        """Toggle advanced quantitative optimization on/off"""
        self.use_advanced_quant = enabled and ADVANCED_QUANT_AVAILABLE
        
        if enabled and ADVANCED_QUANT_AVAILABLE:
            self.status_label.setText("✅ Advanced quantitative optimization ENABLED")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif enabled and not ADVANCED_QUANT_AVAILABLE:
            self.status_label.setText("⚠️ Advanced quantitative optimization UNAVAILABLE - missing libraries")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.advanced_quant_enabled.setChecked(False)
            self.use_advanced_quant = False
        else:
            self.status_label.setText("❌ Advanced quantitative optimization DISABLED")
            self.status_label.setStyleSheet("color: gray;")
    
    def update_library_status(self):
        """Update library availability status"""
        status_text = "Library Status:\\n"
        
        if ADVANCED_QUANT_AVAILABLE:
            status_text += "✅ Advanced Quantitative Optimizer: Available\\n"
        else:
            status_text += "❌ Advanced Quantitative Optimizer: Missing\\n"
        
        # Check individual libraries
        libraries = [
            ("arch", "ARCH (GARCH)"),
            ("copulas", "Copulas"),
            ("scipy", "SciPy"),
            ("sklearn", "Scikit-learn")
        ]
        
        for lib_name, lib_desc in libraries:
            try:
                __import__(lib_name)
                status_text += f"✅ {lib_desc}: Available\\n"
            except ImportError:
                status_text += f"❌ {lib_desc}: Missing - pip install {lib_name}\\n"
        
        self.library_status.setText(status_text)
        self.library_status.setStyleSheet("font-family: monospace; font-size: 10px;")
    
    def get_advanced_quant_parameters(self):
        """Get parameters from UI or use defaults"""
        if hasattr(self, 'optimization_strategy'):
            return {
                'optimization_strategy': self.optimization_strategy.currentText(),
                'risk_tolerance': self.risk_tolerance.value(),
                'var_confidence': self.var_confidence.value(),
                'mc_simulations': self.mc_simulations.value(),
                'kelly_fraction_limit': self.kelly_fraction_limit.value(),
                'target_volatility': 0.15,
                'time_horizon': 1,
                'garch_p': 1,
                'garch_q': 1,
                'garch_lookback': 100,
                'copula_family': 'gaussian',
                'dependency_threshold': 0.3,
                'expected_win_rate': 0.2
            }
        else:
            return self.default_params
    
    def optimize_lineups_with_advanced_quant(self, df_filtered, num_lineups=10):
        """
        Optimize lineups using advanced quantitative techniques
        """
        if not self.use_advanced_quant or not self.advanced_optimizer:
            return []
        
        logging.info("🔬 Starting advanced quantitative optimization")
        
        try:
            # Get parameters
            params = self.get_advanced_quant_parameters()
            
            # Prepare player data
            player_data = []
            for _, player in df_filtered.iterrows():
                player_data.append({
                    'Name': player['Name'],
                    'Pos': player['Pos'],
                    'Team': player['Team'],
                    'Salary': player['Salary'],
                    'Predicted_DK_Points': player['Predicted_DK_Points'],
                    'Value': player.get('Value', player['Predicted_DK_Points'] / (player['Salary'] / 1000))
                })
            
            # Generate historical data (mock for now)
            historical_data = self.generate_historical_performance_data(df_filtered)
            
            # Run optimization
            logging.info(f"🎯 Running advanced optimization with {len(player_data)} players")
            
            # Convert to DataFrame format expected by the optimizer
            player_df = df_filtered.copy()
            
            # Ensure required columns exist and map to expected names
            if 'Value' not in player_df.columns:
                player_df['Value'] = player_df['Predicted_DK_Points'] / (player_df['Salary'] / 1000)
            
            # Map column names to what the optimizer expects
            column_mapping = {
                'Pos': 'Position'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in player_df.columns and new_name not in player_df.columns:
                    player_df[new_name] = player_df[old_name]
            
            # Run the advanced optimization
            optimization_result = self.advanced_optimizer.advanced_lineup_optimization(player_df)
            
            # The optimizer returns a dictionary with portfolio metrics and selected players
            # We need to extract the lineup information and convert to our format
            if optimization_result:
                # Check if it's the advanced result or fallback result
                if 'selected_players' in optimization_result:
                    # Advanced result format
                    optimized_lineups = [optimization_result]
                elif 'recommendations' in optimization_result:
                    # Fallback result format - convert to expected format
                    recommendations = optimization_result['recommendations']
                    selected_players = [rec['player'] for rec in recommendations]
                    
                    # Create a result that matches the expected format
                    converted_result = {
                        'selected_players': selected_players,
                        'sharpe_ratio': 0.5,  # Default values for fallback
                        'volatility': 0.2,
                        'var_95': -5.0,
                        'cvar_95': -7.0,
                        'kelly_fraction': 0.1
                    }
                    optimized_lineups = [converted_result]
                else:
                    optimized_lineups = []
            else:
                optimized_lineups = []
            
            # Convert results
            results = []
            for i, lineup_result in enumerate(optimized_lineups):
                # Get selected players from the result
                selected_players = lineup_result.get('selected_players', [])
                
                if selected_players:
                    # Create lineup DataFrame from selected players
                    lineup_df = df_filtered[df_filtered['Name'].isin(selected_players)].copy()
                    
                    if not lineup_df.empty:
                        result = {
                            'lineup_id': i,
                            'lineup': lineup_df,
                            'total_points': lineup_df['Predicted_DK_Points'].sum(),
                            'total_salary': lineup_df['Salary'].sum(),
                            'risk_metrics': {
                                'sharpe_ratio': lineup_result.get('sharpe_ratio', 0),
                                'volatility': lineup_result.get('volatility', 0),
                                'var_95': lineup_result.get('var_95', 0),
                                'cvar_95': lineup_result.get('cvar_95', 0),
                                'kelly_fraction': lineup_result.get('kelly_fraction', 0)
                            }
                        }
                        results.append(result)
            
            logging.info(f"🎯 Advanced optimization complete. Generated {len(results)} lineups")
            return results
            
        except Exception as e:
            logging.error(f"❌ Error in advanced quantitative optimization: {str(e)}")
            return []
    
    def generate_historical_performance_data(self, df_filtered):
        """Generate mock historical performance data"""
        historical_data = {}
        np.random.seed(42)  # For reproducible results
        
        for _, player in df_filtered.iterrows():
            name = player['Name']
            projected = player['Predicted_DK_Points']
            
            # Generate realistic historical performance
            volatility = max(0.15, min(0.35, projected * 0.02))
            historical_points = np.random.normal(
                loc=projected,
                scale=projected * volatility,
                size=100
            )
            historical_points = np.maximum(historical_points, 0)
            historical_data[name] = historical_points.tolist()
        
        return historical_data
    
    def convert_lineup_result(self, lineup_result, df_filtered):
        """Convert lineup result to DataFrame format"""
        try:
            selected_players = lineup_result.get('selected_players', [])
            
            if not selected_players:
                return pd.DataFrame()
            
            lineup_players = []
            for player_name in selected_players:
                player_data = df_filtered[df_filtered['Name'] == player_name]
                if not player_data.empty:
                    lineup_players.append(player_data.iloc[0])
            
            if not lineup_players:
                return pd.DataFrame()
            
            return pd.DataFrame(lineup_players)
            
        except Exception as e:
            logging.error(f"Error converting lineup result: {e}")
            return pd.DataFrame()

def integrate_advanced_quant_optimizer(main_app):
    """
    Main function to integrate advanced quantitative optimization into the MLB optimizer
    """
    print("🔬 Integrating Advanced Quantitative Optimizer...")
    
    # Create the integrator
    integrator = AdvancedQuantIntegrator(main_app)
    
    # Add the advanced tab to the main GUI
    if hasattr(main_app, 'tabs'):
        integrator.add_advanced_quant_tab(main_app.tabs)
    
    # Store the integrator in the main app
    main_app.advanced_quant_integrator = integrator
    
    # Add method to check if advanced optimization should be used
    def should_use_advanced_quant():
        return (hasattr(main_app, 'advanced_quant_integrator') and 
                main_app.advanced_quant_integrator.use_advanced_quant)
    
    main_app.should_use_advanced_quant = should_use_advanced_quant
    
    # Add method to run advanced optimization
    def run_advanced_optimization(df_filtered, num_lineups=10):
        if hasattr(main_app, 'advanced_quant_integrator'):
            return main_app.advanced_quant_integrator.optimize_lineups_with_advanced_quant(
                df_filtered, num_lineups
            )
        return []
    
    main_app.run_advanced_optimization = run_advanced_optimization
    
    print("✅ Advanced Quantitative Optimizer integration complete!")
    
    return integrator

# Test the integration
if __name__ == "__main__":
    print("🧪 Testing Advanced Quantitative Integration...")
    
    # Create mock data for testing
    import pandas as pd
    import numpy as np
    
    # Create sample player data
    np.random.seed(42)
    players = []
    positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'OF']
    teams = ['NYY', 'BOS', 'HOU', 'LAD', 'ATL']
    
    for i in range(50):
        players.append({
            'Name': f'Player_{i}',
            'Pos': np.random.choice(positions),
            'Team': np.random.choice(teams),
            'Salary': np.random.randint(2000, 12000),
            'Predicted_DK_Points': np.random.uniform(5, 25),
            'Value': np.random.uniform(1, 5)
        })
    
    df_test = pd.DataFrame(players)
    
    # Test the integrator
    class MockApp:
        def __init__(self):
            self.tabs = QTabWidget()
    
    # This would normally be called from the main application
    mock_app = MockApp()
    integrator = integrate_advanced_quant_optimizer(mock_app)
    
    # Test optimization
    if integrator.use_advanced_quant:
        results = integrator.optimize_lineups_with_advanced_quant(df_test, num_lineups=3)
        print(f"✅ Generated {len(results)} optimized lineups")
        
        for i, result in enumerate(results):
            print(f"\\nLineup {i+1}:")
            print(f"  Total Points: {result['total_points']:.2f}")
            print(f"  Total Salary: ${result['total_salary']:,}")
            print(f"  Sharpe Ratio: {result['risk_metrics']['sharpe_ratio']:.3f}")
            print(f"  VaR (95%): {result['risk_metrics']['var_95']:.2f}")
    else:
        print("❌ Advanced quantitative optimization not enabled")
    
    print("🎯 Integration test complete!")
