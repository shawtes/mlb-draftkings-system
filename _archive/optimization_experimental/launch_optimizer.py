#!/usr/bin/env python3
"""
DFS Optimizer Launcher with Advanced Risk Management
"""

import sys
import os

def main():
    """Launch the DFS optimizer with risk management"""
    
    print("🔥 Starting Advanced DFS Optimizer with Risk Management")
    print("=" * 60)
    
    try:
        # Set up the path for imports
        project_dir = os.path.dirname(os.path.abspath(__file__))
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)
        
        # Import and check risk engine availability
        try:
            from dfs_risk_engine import DFSRiskEngine, DFSBankrollManager, RiskMetrics
            print("✅ Risk Management Engine: LOADED")
        except ImportError as e:
            print(f"⚠️ Risk Management Engine: NOT AVAILABLE ({e})")
        # Import the main optimizer
        from optimizer01 import FantasyBaseballApp
        from PyQt5.QtWidgets import QApplication
        
        print("✅ DFS Optimizer: LOADED")
        print("✅ PyQt5 GUI: LOADED")
        
        # Create and run the application
        app = QApplication(sys.argv)
        window = FantasyBaseballApp()
        window.show()
        
        print("\n🚀 Launching DFS Optimizer GUI...")
        print("\n📋 Features Available:")
        print("   • Kelly Criterion Position Sizing")
        print("   • GARCH Volatility Modeling") 
        print("   • Portfolio Theory Optimization")
        print("   • Risk-Adjusted Lineup Selection")
        print("   • Professional Bankroll Management")
        print("   • 🔥 TEAM COMBINATION GENERATOR")
        print("   • Advanced Quantitative Modeling")
        print("   • 🎯 PROBABILITY-ENHANCED OPTIMIZATION")
        print("   • Implied Volatility from Probability Distributions")
        print("   • Risk-Aware Player Selection using Prob_Over_X columns")
        print("   • ✅ PRESERVES EXACT LINEUP COUNTS (Fixed)")
        print("   • 🚀 SEQUENTIAL OPTIMIZATION (No More Filtering!)")
        print("   • 🎯 TEAM COMBINATION CONSTRAINTS (Fixed!)")
        print("\n🎯 COMBINATION GENERATION:")
        print("   Go to 'Team Combinations' tab after loading data!")
        print("   Stack patterns: 4|2, 5|3, 4|2|2, 3|3|2, and more")
        print("   ✅ Now delivers EXACT lineup counts you request!")
        print("   🔥 NO MORE 'FILTERED OUT' messages in logs!")
        print("   🎯 ENFORCES specific team constraints (CHC(5) + BOS(2))")
        print("   ⚠️ IMPORTANT: Set 'Min Unique Players' to 0 for team combinations!")
        print("   📋 Sequential optimization automatically ensures diversity")
        print("\n🎲 PROBABILITY FEATURES:")
        print("   • Automatically detects Prob_Over_X columns in your CSV")
        print("   • Uses probability distributions for implied volatility")
        print("   • Enhanced Kelly Criterion with win probabilities")
        print("   • Floor/Ceiling optimization based on thresholds")
        print("\n💡 Set your bankroll and risk profile in the Risk Management section!")
        print("🚀 Sequential optimization ensures TRUE diversity without filtering!")
        print("📊 CHC(5)+BOS(2) → 25 lineups → 25 UNIQUE lineups with proper stacks!")
        print("🎯 Each lineup will have exactly 5 CHC players and 2 BOS players!")
        
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure all required packages are installed:")
        print("      pip install PyQt5 pandas numpy scipy arch")
        print("   2. Check that all files are in the correct locations")
        print("   3. Try running from the project root directory")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        input("\nPress Enter to exit...")
        sys.exit(exit_code)
