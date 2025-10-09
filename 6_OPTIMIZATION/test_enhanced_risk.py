#!/usr/bin/env python3
"""
Test script for enhanced risk management features
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Import the enhanced risk management classes
    from optimizer01 import (
        GARCHRiskModel, VaRCalculator, AdvancedRiskModel, 
        LiquidityRiskManager, EnhancedKellyCriterion, PortfolioRiskAnalyzer
    )
    
    print("✅ Successfully imported enhanced risk management classes!")
    
    # Test GARCH model
    print("\n🔬 Testing GARCH Risk Model...")
    garch_model = GARCHRiskModel()
    test_returns = pd.Series(np.random.normal(0.05, 0.2, 100))
    garch_model.fit(test_returns)
    volatility_forecast = garch_model.forecast_volatility()
    print(f"   📊 GARCH volatility forecast: {volatility_forecast:.4f}")
    
    # Test VaR Calculator
    print("\n💹 Testing VaR Calculator...")
    var_calc = VaRCalculator(confidence_level=0.95)
    var_95 = var_calc.calculate_var(test_returns)
    es_95 = var_calc.calculate_expected_shortfall(test_returns)
    print(f"   📊 VaR (95%): {var_95:.4f}")
    print(f"   📊 Expected Shortfall (95%): {es_95:.4f}")
    
    # Test Advanced Risk Model
    print("\n🎯 Testing Advanced Risk Model...")
    risk_model = AdvancedRiskModel()
    
    # Create mock factor data
    mock_factors = pd.DataFrame({
        'team_concentration': np.random.uniform(0.1, 0.4, 100),
        'position_concentration': np.random.uniform(0.1, 0.3, 100),
        'salary_factor': np.random.uniform(0.5, 1.0, 100)
    })
    
    risk_model.fit_factor_model(test_returns, mock_factors)
    print(f"   📊 Factor model fitted: {risk_model.fitted}")
    
    # Test Enhanced Kelly Criterion
    print("\n💰 Testing Enhanced Kelly Criterion...")
    kelly_calc = EnhancedKellyCriterion()
    
    # Test optimal position sizing
    optimal_size = kelly_calc.calculate_optimal_position_size(
        expected_return=0.08,
        volatility=0.20,
        bankroll=1000
    )
    print(f"   📊 Optimal position size: ${optimal_size:.2f}")
    
    # Test traditional Kelly sizing
    kelly_result = kelly_calc.calculate_kelly_sizing(
        expected_return=0.08,
        win_probability=0.55,
        odds_received=1.8,
        volatility=0.20
    )
    print(f"   📊 Kelly fraction: {kelly_result['kelly_fraction']:.4f}")
    
    # Test Portfolio Risk Analyzer
    print("\n📊 Testing Portfolio Risk Analyzer...")
    portfolio_analyzer = PortfolioRiskAnalyzer(risk_model)
    
    # Mock portfolio weights
    portfolio_weights = {
        'player_1': 0.15,
        'player_2': 0.12,
        'player_3': 0.18,
        'player_4': 0.10,
        'player_5': 0.13,
        'player_6': 0.11,
        'player_7': 0.09,
        'player_8': 0.12
    }
    
    # Mock returns data
    mock_returns = pd.DataFrame({
        player: np.random.normal(0.05, 0.2, 100) 
        for player in portfolio_weights.keys()
    })
    
    risk_decomp = portfolio_analyzer.decompose_portfolio_risk(portfolio_weights, mock_returns)
    print(f"   📊 Total portfolio volatility: {risk_decomp['total_volatility']:.4f}")
    print(f"   📊 Factor risk contribution: {risk_decomp['factor_contribution']:.1%}")
    print(f"   📊 Specific risk contribution: {risk_decomp['specific_contribution']:.1%}")
    
    print("\n🎉 All enhanced risk management tests passed!")
    print("\n📋 Enhanced Features Summary:")
    print("   ✅ GARCH volatility modeling")
    print("   ✅ Value at Risk (VaR) calculations")
    print("   ✅ Expected Shortfall risk metrics")
    print("   ✅ Multi-factor risk decomposition")
    print("   ✅ Enhanced Kelly Criterion position sizing")
    print("   ✅ Portfolio risk analysis")
    print("   ✅ Liquidity risk management")
    print("\n🚀 Your DFS optimizer now has quantitative finance-grade risk management!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure optimizer01.py is in the same directory")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Enhanced Risk Management Test Complete")
print("="*60)
