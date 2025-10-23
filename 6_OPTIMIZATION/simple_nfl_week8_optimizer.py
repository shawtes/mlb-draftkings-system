#!/usr/bin/env python3
"""
Simple NFL Week 8 Optimizer
===========================

This script creates a simple optimizer that generates lineups
without the complex risk management that might be causing issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

def create_simple_lineups(df, num_lineups=150):
    """
    Create simple lineups using basic optimization
    """
    print(f"\n🎯 CREATING {num_lineups} SIMPLE LINEUPS")
    print("="*50)
    
    # Filter to players with projections
    df_with_proj = df[df['Predicted_DK_Points'] > 0].copy()
    print(f"✅ Using {len(df_with_proj)} players with projections")
    
    # Position limits
    position_limits = {
        'QB': 1,
        'RB': 2, 
        'WR': 3,
        'TE': 1,
        'DST': 1
    }
    
    lineups = []
    
    for i in range(num_lineups):
        lineup = create_single_lineup(df_with_proj, position_limits, i)
        if lineup is not None:
            lineups.append(lineup)
    
    print(f"✅ Generated {len(lineups)} valid lineups")
    return lineups

def create_single_lineup(df, position_limits, lineup_id):
    """
    Create a single lineup using greedy selection
    """
    salary_cap = 50000
    min_salary = 49000
    
    # Create lineup DataFrame
    lineup_df = pd.DataFrame()
    used_players = set()
    total_salary = 0
    
    # Add noise to projections for diversity
    noise_factor = 0.1 + (lineup_id % 10) * 0.05  # 0.1 to 0.6
    df_copy = df.copy()
    df_copy['Adjusted_Points'] = df_copy['Predicted_DK_Points'] * (1 + np.random.normal(0, noise_factor))
    
    # Sort by value (adjusted points per $1000)
    df_copy['Value'] = df_copy['Adjusted_Points'] / (df_copy['Salary'] / 1000)
    df_copy = df_copy.sort_values('Value', ascending=False)
    
    # Select QB
    qb_candidates = df_copy[(df_copy['Position'] == 'QB') & (~df_copy['PlayerID'].isin(used_players))]
    if qb_candidates.empty:
        return None
    
    qb = qb_candidates.iloc[0]
    lineup_df = pd.concat([lineup_df, qb.to_frame().T])
    used_players.add(qb['PlayerID'])
    total_salary += qb['Salary']
    
    # Select RB (2)
    rb_candidates = df_copy[(df_copy['Position'] == 'RB') & (~df_copy['PlayerID'].isin(used_players))]
    if len(rb_candidates) < 2:
        return None
    
    for _ in range(2):
        rb = rb_candidates.iloc[0]
        lineup_df = pd.concat([lineup_df, rb.to_frame().T])
        used_players.add(rb['PlayerID'])
        total_salary += rb['Salary']
        rb_candidates = rb_candidates[~rb_candidates['PlayerID'].isin(used_players)]
    
    # Select WR (3)
    wr_candidates = df_copy[(df_copy['Position'] == 'WR') & (~df_copy['PlayerID'].isin(used_players))]
    if len(wr_candidates) < 3:
        return None
    
    for _ in range(3):
        wr = wr_candidates.iloc[0]
        lineup_df = pd.concat([lineup_df, wr.to_frame().T])
        used_players.add(wr['PlayerID'])
        total_salary += wr['Salary']
        wr_candidates = wr_candidates[~wr_candidates['PlayerID'].isin(used_players)]
    
    # Select TE
    te_candidates = df_copy[(df_copy['Position'] == 'TE') & (~df_copy['PlayerID'].isin(used_players))]
    if te_candidates.empty:
        return None
    
    te = te_candidates.iloc[0]
    lineup_df = pd.concat([lineup_df, te.to_frame().T])
    used_players.add(te['PlayerID'])
    total_salary += te['Salary']
    
    # Select DST
    dst_candidates = df_copy[(df_copy['Position'] == 'DST') & (~df_copy['PlayerID'].isin(used_players))]
    if dst_candidates.empty:
        return None
    
    dst = dst_candidates.iloc[0]
    lineup_df = pd.concat([lineup_df, dst.to_frame().T])
    used_players.add(dst['PlayerID'])
    total_salary += dst['Salary']
    
    # Check salary constraints
    if total_salary < min_salary or total_salary > salary_cap:
        return None
    
    # Calculate total projection
    total_projection = lineup_df['Predicted_DK_Points'].sum()
    
    return {
        'lineup_id': lineup_id + 1,
        'lineup': lineup_df,
        'total_salary': total_salary,
        'total_projection': total_projection,
        'value': total_projection / (total_salary / 1000)
    }

def analyze_lineups(lineups):
    """
    Analyze the generated lineups
    """
    if not lineups:
        print("❌ No lineups to analyze")
        return
    
    print(f"\n📊 LINEUP ANALYSIS")
    print("="*50)
    
    # Calculate statistics
    salaries = [l['total_salary'] for l in lineups]
    projections = [l['total_projection'] for l in lineups]
    values = [l['value'] for l in lineups]
    
    print(f"✅ Generated {len(lineups)} lineups")
    print(f"   Avg Salary: ${np.mean(salaries):,.0f}")
    print(f"   Avg Projection: {np.mean(projections):.1f} pts")
    print(f"   Avg Value: {np.mean(values):.3f}")
    print(f"   Salary Range: ${min(salaries):,} - ${max(salaries):,}")
    print(f"   Projection Range: {min(projections):.1f} - {max(projections):.1f} pts")
    
    # Show top lineups
    print(f"\n🏆 Top 10 Lineups by Projection:")
    sorted_lineups = sorted(lineups, key=lambda x: x['total_projection'], reverse=True)
    
    for i, lineup in enumerate(sorted_lineups[:10]):
        print(f"   #{i+1}: ${lineup['total_salary']:,} salary, {lineup['total_projection']:.1f} pts, {lineup['value']:.3f} value")
        
        # Show players
        players = lineup['lineup']
        for _, player in players.iterrows():
            print(f"      {player['Position']}: {player['Name']} (${player['Salary']:,}, {player['Predicted_DK_Points']:.1f} pts)")

def export_lineups(lineups, output_filename="nfl_week8_simple_lineups.csv"):
    """
    Export lineups in DraftKings format
    """
    if not lineups:
        print("❌ No lineups to export")
        return
    
    print(f"\n💾 EXPORTING LINEUPS")
    print("="*50)
    
    # Create DraftKings format
    dk_lineups = []
    
    for lineup in lineups:
        lineup_df = lineup['lineup']
        
        # Create DraftKings lineup format
        dk_lineup = {
            'Entry ID': f"week8_simple_{lineup['lineup_id']:03d}",
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
        qb_players = []
        rb_players = []
        wr_players = []
        te_players = []
        dst_players = []
        
        for _, player in lineup_df.iterrows():
            pos = player['Position']
            player_id = player['OperatorPlayerID']
            
            if pos == 'QB':
                qb_players.append(player_id)
            elif pos == 'RB':
                rb_players.append(player_id)
            elif pos == 'WR':
                wr_players.append(player_id)
            elif pos == 'TE':
                te_players.append(player_id)
            elif pos == 'DST':
                dst_players.append(player_id)
        
        # Assign to DraftKings format
        if qb_players:
            dk_lineup['QB'] = qb_players[0]
        if len(rb_players) >= 2:
            dk_lineup['RB'] = ','.join(rb_players[:2])
        if len(wr_players) >= 3:
            dk_lineup['WR'] = ','.join(wr_players[:3])
        if te_players:
            dk_lineup['TE'] = te_players[0]
        if dst_players:
            dk_lineup['DST'] = dst_players[0]
        
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
    Main function to run the simple Week 8 optimizer
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              NFL WEEK 8 SIMPLE OPTIMIZER                ║
    ║              Basic Greedy Algorithm                      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    csv_file = "nfl_week8_draftkings_13game_slate.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        print("   Please run fetch_nfl_week8_data.py first")
        return
    
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} players from {csv_file}")
    
    # Rename columns
    df = df.rename(columns={
        'FantasyPoints': 'Predicted_DK_Points',
        'Value': 'PointsPerK'
    })
    
    # Add required columns
    df['Opponent'] = 'TBD'
    df['InjuryStatus'] = 'Active'
    df['GameInfo'] = 'TBD'
    
    print(f"📊 Data Summary:")
    print(f"   Total Players: {len(df)}")
    print(f"   Players with Projections: {df['Predicted_DK_Points'].gt(0).sum()}")
    print(f"   Positions: {df['Position'].value_counts().to_dict()}")
    
    # Generate lineups
    lineups = create_simple_lineups(df, num_lineups=150)
    
    if lineups:
        # Analyze lineups
        analyze_lineups(lineups)
        
        # Export lineups
        export_lineups(lineups)
        
        print(f"\n" + "="*80)
        print("✅ WEEK 8 SIMPLE OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"📁 Generated: nfl_week8_simple_lineups.csv")
        print(f"🎯 Ready for DraftKings upload")
    else:
        print("❌ Failed to generate any lineups")

if __name__ == "__main__":
    main()
