#!/usr/bin/env python3
"""
NFL Week 8 GPP Tournament Optimizer
==================================

This script runs the genetic algorithm optimizer for Week 8 NFL DFS
with GPP (Guaranteed Prize Pool) tournament settings.

Features:
- Loads Week 8 data with DraftKings salaries and projections
- Configures GPP-optimized settings
- Generates diverse lineups for tournaments
- Exports lineups in DraftKings format
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the genetic algorithm optimizer
from genetic_algo_nfl_optimizer import FantasyFootballApp, OptimizationWorker
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread

def load_week8_data():
    """
    Load the Week 8 NFL data we fetched
    """
    print("="*80)
    print("🏈 LOADING WEEK 8 NFL DATA")
    print("="*80)
    
    # Load the CSV file
    csv_file = "nfl_week8_draftkings_13game_slate.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        print("   Please run fetch_nfl_week8_data.py first")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} players from {csv_file}")
    
    # Rename columns to match optimizer expectations
    df = df.rename(columns={
        'FantasyPoints': 'Predicted_DK_Points',
        'Value': 'PointsPerK'
    })
    
    # Add required columns for NFL optimizer
    df['Opponent'] = 'TBD'  # Placeholder
    df['InjuryStatus'] = 'Active'  # Placeholder
    df['GameInfo'] = 'TBD'  # Placeholder
    
    # Show data summary
    print(f"\n📊 Data Summary:")
    print(f"   Total Players: {len(df)}")
    print(f"   Positions: {df['Position'].value_counts().to_dict()}")
    print(f"   Salary Range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
    print(f"   Avg Salary: ${df['Salary'].mean():,.0f}")
    print(f"   Players with Projections: {df['Predicted_DK_Points'].gt(0).sum()}")
    
    # Show top projected players
    print(f"\n🏆 Top 10 Projected Players:")
    top_proj = df.nlargest(10, 'Predicted_DK_Points')[['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points', 'PointsPerK']]
    print(top_proj.to_string(index=False))
    
    return df

def configure_gpp_settings():
    """
    Configure GPP-optimized settings
    """
    print(f"\n🎯 CONFIGURING GPP SETTINGS")
    print("="*50)
    
    settings = {
        # Basic settings
        'num_lineups': 150,  # Good for GPP tournaments
        'salary_cap': 50000,
        'min_salary': 49000,  # Use most of the salary cap
        
        # GPP-specific settings
        'stack_type': 'game_stack',  # Game stacking for GPP
        'min_exposure': 0.05,  # 5% minimum exposure
        'max_exposure': 0.25,  # 25% maximum exposure (diversity)
        
        # Risk management
        'risk_tolerance': 'high',  # High risk for GPP
        'bankroll': 1000,
        'disable_kelly': False,
        
        # Advanced settings
        'use_advanced_quant': False,
        'min_unique': 0,  # No uniqueness constraint for GPP
        
        # Team selections (empty for now, will be set by user)
        'team_selections': {},
        'stack_settings': {},
        'included_players': [],  # Will be populated by user selections
    }
    
    print(f"✅ GPP Settings Configured:")
    print(f"   Lineups: {settings['num_lineups']}")
    print(f"   Salary Cap: ${settings['salary_cap']:,}")
    print(f"   Min Salary: ${settings['min_salary']:,}")
    print(f"   Stack Type: {settings['stack_type']}")
    print(f"   Exposure: {settings['min_exposure']*100:.0f}% - {settings['max_exposure']*100:.0f}%")
    print(f"   Risk Tolerance: {settings['risk_tolerance']}")
    
    return settings

def run_optimization(df, settings):
    """
    Run the genetic algorithm optimization
    """
    print(f"\n🚀 RUNNING GENETIC ALGORITHM OPTIMIZATION")
    print("="*60)
    
    # Create the optimization worker
    worker = OptimizationWorker(
        df_players=df,
        salary_cap=settings['salary_cap'],
        position_limits={'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1},
        included_players=settings['included_players'],
        stack_settings=settings['stack_settings'],
        min_exposure=settings['min_exposure'],
        max_exposure=settings['max_exposure'],
        min_points=0,
        monte_carlo_iterations=1000,
        num_lineups=settings['num_lineups'],
        team_selections=settings['team_selections'],
        min_unique=settings['min_unique'],
        bankroll=settings['bankroll'],
        risk_tolerance=settings['risk_tolerance'],
        disable_kelly=settings['disable_kelly'],
        min_salary=settings['min_salary'],
        use_advanced_quant=settings['use_advanced_quant']
    )
    
    print(f"✅ Optimization worker created")
    print(f"   Target lineups: {settings['num_lineups']}")
    print(f"   Position limits: QB(1), RB(2), WR(3), TE(1), DST(1)")
    
    # Run optimization
    print(f"\n🔄 Starting optimization...")
    start_time = datetime.now()
    
    try:
        # Run the optimization
        worker.run()
        
        # Get results
        results = worker.results if hasattr(worker, 'results') else {}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Optimization completed in {duration:.1f} seconds")
        
        if results:
            print(f"   Generated lineups: {len(results.get('lineups', []))}")
            return results
        else:
            print("❌ No results generated")
            return None
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return None

def analyze_results(results, df):
    """
    Analyze the generated lineups
    """
    if not results or 'lineups' not in results:
        print("❌ No lineups to analyze")
        return
    
    print(f"\n📊 LINEUP ANALYSIS")
    print("="*50)
    
    lineups = results['lineups']
    print(f"✅ Generated {len(lineups)} lineups")
    
    # Calculate lineup statistics
    lineup_stats = []
    
    for i, lineup in enumerate(lineups):
        if isinstance(lineup, dict) and 'lineup' in lineup:
            lineup_df = lineup['lineup']
        else:
            lineup_df = lineup
        
        if isinstance(lineup_df, pd.DataFrame):
            total_salary = lineup_df['Salary'].sum()
            total_projection = lineup_df['Predicted_DK_Points'].sum()
            
            lineup_stats.append({
                'lineup_id': i + 1,
                'total_salary': total_salary,
                'total_projection': total_projection,
                'value': total_projection / (total_salary / 1000) if total_salary > 0 else 0
            })
    
    if lineup_stats:
        stats_df = pd.DataFrame(lineup_stats)
        
        print(f"\n📈 Lineup Statistics:")
        print(f"   Avg Salary: ${stats_df['total_salary'].mean():,.0f}")
        print(f"   Avg Projection: {stats_df['total_projection'].mean():.1f} pts")
        print(f"   Avg Value: {stats_df['value'].mean():.3f}")
        print(f"   Salary Range: ${stats_df['total_salary'].min():,} - ${stats_df['total_salary'].max():,}")
        print(f"   Projection Range: {stats_df['total_projection'].min():.1f} - {stats_df['total_projection'].max():.1f} pts")
        
        # Show top lineups
        print(f"\n🏆 Top 5 Lineups by Projection:")
        top_lineups = stats_df.nlargest(5, 'total_projection')
        for _, lineup in top_lineups.iterrows():
            print(f"   Lineup {lineup['lineup_id']}: ${lineup['total_salary']:,} salary, {lineup['total_projection']:.1f} pts, {lineup['value']:.3f} value")

def export_lineups(results, df, output_filename="nfl_week8_gpp_lineups.csv"):
    """
    Export lineups in DraftKings format
    """
    if not results or 'lineups' not in results:
        print("❌ No lineups to export")
        return
    
    print(f"\n💾 EXPORTING LINEUPS")
    print("="*50)
    
    lineups = results['lineups']
    
    # Create DraftKings format
    dk_lineups = []
    
    for i, lineup in enumerate(lineups):
        if isinstance(lineup, dict) and 'lineup' in lineup:
            lineup_df = lineup['lineup']
        else:
            lineup_df = lineup
        
        if isinstance(lineup_df, pd.DataFrame):
            # Create DraftKings lineup format
            dk_lineup = {
                'Entry ID': f"week8_gpp_{i+1:03d}",
                'Contest Name': 'NFL Week 8 GPP Tournament',
                'Contest ID': '183787056',
                'Entry Fee': '$1',
                'QB': '',
                'RB': '',
                'RB': '',
                'WR': '',
                'WR': '',
                'WR': '',
                'TE': '',
                'FLEX': '',
                'DST': ''
            }
            
            # Fill positions
            for _, player in lineup_df.iterrows():
                pos = player['Position']
                player_id = player['OperatorPlayerID']
                
                if pos == 'QB':
                    dk_lineup['QB'] = player_id
                elif pos == 'RB':
                    if not dk_lineup['RB']:
                        dk_lineup['RB'] = player_id
                    else:
                        dk_lineup['RB'] = dk_lineup['RB'] + ',' + player_id
                elif pos == 'WR':
                    if not dk_lineup['WR']:
                        dk_lineup['WR'] = player_id
                    else:
                        dk_lineup['WR'] = dk_lineup['WR'] + ',' + player_id
                elif pos == 'TE':
                    dk_lineup['TE'] = player_id
                elif pos == 'DST':
                    dk_lineup['DST'] = player_id
            
            dk_lineups.append(dk_lineup)
    
    # Save to CSV
    if dk_lineups:
        export_df = pd.DataFrame(dk_lineups)
        export_df.to_csv(output_filename, index=False)
        print(f"✅ Exported {len(dk_lineups)} lineups to {output_filename}")
        
        # Show sample
        print(f"\n📋 Sample Lineup:")
        sample = export_df.iloc[0]
        for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
            if sample[pos]:
                print(f"   {pos}: {sample[pos]}")
    else:
        print("❌ No valid lineups to export")

def main():
    """
    Main function to run the Week 8 GPP optimizer
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           NFL WEEK 8 GPP TOURNAMENT OPTIMIZER           ║
    ║              Genetic Algorithm + DraftKings              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Load data
    df = load_week8_data()
    if df is None:
        return
    
    # Step 2: Configure settings
    settings = configure_gpp_settings()
    
    # Step 3: Run optimization
    results = run_optimization(df, settings)
    if results is None:
        return
    
    # Step 4: Analyze results
    analyze_results(results, df)
    
    # Step 5: Export lineups
    export_lineups(results, df)
    
    print(f"\n" + "="*80)
    print("✅ WEEK 8 GPP OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"📁 Generated: nfl_week8_gpp_lineups.csv")
    print(f"🎯 Ready for DraftKings upload")
    print(f"💰 Good luck in your tournaments!")

if __name__ == "__main__":
    main()
