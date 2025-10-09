"""
Demo script showing how to use the Advanced Quantitative Optimizer
with copulas, GARCH, Monte Carlo, VaR, Kelly criterion, and other
advanced financial modeling techniques for DraftKings MLB optimization.
"""

import numpy as np
import pandas as pd
import sys
import os
from optimizer_integration_patch import AdvancedQuantIntegrator

def create_sample_player_data():
    """Create sample player data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Player positions and typical salary ranges
    positions = {
        'P': (2000, 12000),    # Pitchers
        'C': (2500, 6000),     # Catchers  
        '1B': (3000, 8000),    # First Base
        '2B': (2500, 7000),    # Second Base
        '3B': (3000, 8500),    # Third Base
        'SS': (2800, 7500),    # Shortstop
        'OF': (2500, 8000),    # Outfield
    }
    
    teams = ['NYY', 'BOS', 'HOU', 'LAD', 'ATL', 'CLE', 'TB', 'SD', 'TOR', 'PHI']
    
    players = []
    player_id = 1
    
    # Create realistic player distribution
    for pos, (min_sal, max_sal) in positions.items():
        num_players = 15 if pos == 'P' else 10 if pos == 'OF' else 8
        
        for i in range(num_players):
            salary = np.random.randint(min_sal, max_sal)
            
            # Realistic projected points based on salary (higher salary = higher projection)
            salary_factor = (salary - min_sal) / (max_sal - min_sal)
            base_points = 8 + salary_factor * 12  # 8-20 point range
            
            # Add some noise
            projected_points = base_points + np.random.normal(0, 1.5)
            projected_points = max(5, projected_points)  # Minimum 5 points
            
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

def demo_advanced_quantitative_optimization():
    """Demonstrate the advanced quantitative optimizer"""
    print("🚀 Advanced Quantitative Optimizer Demo")
    print("=" * 60)
    
    # Create sample data
    print("📊 Creating sample player data...")
    df_players = create_sample_player_data()
    print(f"✅ Created {len(df_players)} players")
    
    # Show sample data
    print("\\n📋 Sample Player Data:")
    print(df_players.head(10).to_string(index=False))
    
    # Initialize the integrator
    print("\\n🔬 Initializing Advanced Quantitative Optimizer...")
    
    class MockApp:
        def __init__(self):
            self.name = "Mock DFS App"
    
    mock_app = MockApp()
    integrator = AdvancedQuantIntegrator(mock_app)
    
    if not integrator.advanced_optimizer:
        print("❌ Advanced optimizer not available")
        return
    
    print("✅ Advanced Quantitative Optimizer initialized")
    
    # Enable advanced optimization
    integrator.use_advanced_quant = True
    
    print("\\n🎯 Running Advanced Quantitative Optimization...")
    print("Features being used:")
    print("  📈 GARCH volatility estimation")
    print("  🔗 Copula dependency modeling")  
    print("  🎲 Monte Carlo simulation")
    print("  📊 Value at Risk (VaR) calculation")
    print("  💰 Kelly Criterion position sizing")
    print("  🎯 Sharpe ratio optimization")
    print("  🔄 Risk parity weighting")
    
    # Run optimization
    try:
        results = integrator.optimize_lineups_with_advanced_quant(df_players, num_lineups=3)
        
        if results:
            print(f"\\n✅ Generated {len(results)} optimized lineups")
            print("=" * 60)
            
            for i, result in enumerate(results):
                print(f"\\n🏆 LINEUP {i+1}:")
                print("-" * 40)
                
                lineup = result['lineup']
                risk_metrics = result['risk_metrics']
                
                print(f"💰 Total Salary: ${result['total_salary']:,}")
                print(f"📊 Total Points: {result['total_points']:.1f}")
                print(f"💡 Salary Usage: {result['total_salary']/50000*100:.1f}%")
                
                print("\\n📈 Risk Metrics:")
                print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
                print(f"  Volatility: {risk_metrics['volatility']:.3f}")
                print(f"  VaR (95%): {risk_metrics['var_95']:.2f}")
                print(f"  CVaR (95%): {risk_metrics['cvar_95']:.2f}")
                print(f"  Kelly Fraction: {risk_metrics['kelly_fraction']:.3f}")
                
                print("\\n👥 Lineup Composition:")
                for _, player in lineup.iterrows():
                    print(f"  {player['Name']:12} {player['Pos']:3} {player['Team']:3} "
                          f"${player['Salary']:,} → {player['Predicted_DK_Points']:5.1f} pts")
                
                # Position breakdown
                pos_counts = lineup['Pos'].value_counts()
                print(f"\\n🎯 Position Breakdown: {dict(pos_counts)}")
                
                # Team breakdown
                team_counts = lineup['Team'].value_counts()
                print(f"🏟️  Team Breakdown: {dict(team_counts)}")
        else:
            print("❌ No optimized lineups generated")
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n🎯 Demo Complete!")
    print("\\nThe Advanced Quantitative Optimizer uses:")
    print("• GARCH models for volatility estimation")
    print("• Copulas for modeling player dependencies")
    print("• Monte Carlo simulation for risk assessment")
    print("• VaR and CVaR for downside risk measurement")
    print("• Kelly Criterion for optimal position sizing")
    print("• Sharpe ratio optimization for risk-adjusted returns")
    print("• Risk parity for balanced portfolio allocation")
    print("• Regime detection for market state analysis")

if __name__ == "__main__":
    demo_advanced_quantitative_optimization()
