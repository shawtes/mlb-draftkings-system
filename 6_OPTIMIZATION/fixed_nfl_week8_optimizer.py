#!/usr/bin/env python3
"""
Fixed NFL Week 8 Optimizer
==========================

This script creates lineups with proper handling of DST players
who may not have projections.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

def create_fixed_lineups(df, num_lineups=150):
    """
    Create lineups with proper DST handling
    """
    print(f"\n🎯 CREATING {num_lineups} FIXED LINEUPS")
    print("="*50)
    
    # For DST, use all players (even without projections)
    # For other positions, use players with projections
    df_qb = df[(df['Position'] == 'QB') & (df['FantasyPoints'] > 0)].copy()
    df_rb = df[(df['Position'] == 'RB') & (df['FantasyPoints'] > 0)].copy()
    df_wr = df[(df['Position'] == 'WR') & (df['FantasyPoints'] > 0)].copy()
    df_te = df[(df['Position'] == 'TE') & (df['FantasyPoints'] > 0)].copy()
    df_dst = df[df['Position'] == 'DST'].copy()  # Include all DST players
    
    print(f"✅ Position counts:")
    print(f"   QB: {len(df_qb)}")
    print(f"   RB: {len(df_rb)}")
    print(f"   WR: {len(df_wr)}")
    print(f"   TE: {len(df_te)}")
    print(f"   DST: {len(df_dst)}")
    
    # Add default projections for DST if missing
    df_dst['FantasyPoints'] = df_dst['FantasyPoints'].fillna(5.0)  # Default 5 points for DST
    
    lineups = []
    
    for i in range(num_lineups):
        lineup = create_single_fixed_lineup(df_qb, df_rb, df_wr, df_te, df_dst, i)
        if lineup is not None:
            lineups.append(lineup)
    
    print(f"✅ Generated {len(lineups)} valid lineups")
    return lineups

def create_single_fixed_lineup(df_qb, df_rb, df_wr, df_te, df_dst, lineup_id):
    """
    Create a single lineup with proper position handling
    """
    salary_cap = 50000
    min_salary = 49000
    
    # Add noise for diversity
    noise_factor = 0.1 + (lineup_id % 10) * 0.05
    
    # Select QB
    if df_qb.empty:
        return None
    
    qb_copy = df_qb.copy()
    qb_copy['Adjusted_Points'] = qb_copy['FantasyPoints'] * (1 + np.random.normal(0, noise_factor))
    qb_copy['Value'] = qb_copy['Adjusted_Points'] / (qb_copy['Salary'] / 1000)
    qb = qb_copy.sort_values('Value', ascending=False).iloc[0]
    
    # Select RB (2)
    if len(df_rb) < 2:
        return None
    
    rb_copy = df_rb.copy()
    rb_copy['Adjusted_Points'] = rb_copy['FantasyPoints'] * (1 + np.random.normal(0, noise_factor))
    rb_copy['Value'] = rb_copy['Adjusted_Points'] / (rb_copy['Salary'] / 1000)
    rb_copy = rb_copy.sort_values('Value', ascending=False)
    
    rb1 = rb_copy.iloc[0]
    rb2 = rb_copy.iloc[1]
    
    # Select WR (3)
    if len(df_wr) < 3:
        return None
    
    wr_copy = df_wr.copy()
    wr_copy['Adjusted_Points'] = wr_copy['FantasyPoints'] * (1 + np.random.normal(0, noise_factor))
    wr_copy['Value'] = wr_copy['Adjusted_Points'] / (wr_copy['Salary'] / 1000)
    wr_copy = wr_copy.sort_values('Value', ascending=False)
    
    wr1 = wr_copy.iloc[0]
    wr2 = wr_copy.iloc[1]
    wr3 = wr_copy.iloc[2]
    
    # Select TE
    if df_te.empty:
        return None
    
    te_copy = df_te.copy()
    te_copy['Adjusted_Points'] = te_copy['FantasyPoints'] * (1 + np.random.normal(0, noise_factor))
    te_copy['Value'] = te_copy['Adjusted_Points'] / (te_copy['Salary'] / 1000)
    te = te_copy.sort_values('Value', ascending=False).iloc[0]
    
    # Select DST (use cheapest available)
    if df_dst.empty:
        return None
    
    dst_copy = df_dst.copy()
    dst = dst_copy.sort_values('Salary').iloc[0]  # Cheapest DST
    
    # Calculate totals
    total_salary = qb['Salary'] + rb1['Salary'] + rb2['Salary'] + wr1['Salary'] + wr2['Salary'] + wr3['Salary'] + te['Salary'] + dst['Salary']
    total_projection = qb['FantasyPoints'] + rb1['FantasyPoints'] + rb2['FantasyPoints'] + wr1['FantasyPoints'] + wr2['FantasyPoints'] + wr3['FantasyPoints'] + te['FantasyPoints'] + dst['FantasyPoints']
    
    # Check salary constraints
    if total_salary < min_salary or total_salary > salary_cap:
        return None
    
    # Create lineup DataFrame
    lineup_data = [
        {'Name': qb['Name'], 'Position': 'QB', 'Team': qb['Team'], 'Salary': qb['Salary'], 'FantasyPoints': qb['FantasyPoints'], 'OperatorPlayerID': qb['OperatorPlayerID']},
        {'Name': rb1['Name'], 'Position': 'RB', 'Team': rb1['Team'], 'Salary': rb1['Salary'], 'FantasyPoints': rb1['FantasyPoints'], 'OperatorPlayerID': rb1['OperatorPlayerID']},
        {'Name': rb2['Name'], 'Position': 'RB', 'Team': rb2['Team'], 'Salary': rb2['Salary'], 'FantasyPoints': rb2['FantasyPoints'], 'OperatorPlayerID': rb2['OperatorPlayerID']},
        {'Name': wr1['Name'], 'Position': 'WR', 'Team': wr1['Team'], 'Salary': wr1['Salary'], 'FantasyPoints': wr1['FantasyPoints'], 'OperatorPlayerID': wr1['OperatorPlayerID']},
        {'Name': wr2['Name'], 'Position': 'WR', 'Team': wr2['Team'], 'Salary': wr2['Salary'], 'FantasyPoints': wr2['FantasyPoints'], 'OperatorPlayerID': wr2['OperatorPlayerID']},
        {'Name': wr3['Name'], 'Position': 'WR', 'Team': wr3['Team'], 'Salary': wr3['Salary'], 'FantasyPoints': wr3['FantasyPoints'], 'OperatorPlayerID': wr3['OperatorPlayerID']},
        {'Name': te['Name'], 'Position': 'TE', 'Team': te['Team'], 'Salary': te['Salary'], 'FantasyPoints': te['FantasyPoints'], 'OperatorPlayerID': te['OperatorPlayerID']},
        {'Name': dst['Name'], 'Position': 'DST', 'Team': dst['Team'], 'Salary': dst['Salary'], 'FantasyPoints': dst['FantasyPoints'], 'OperatorPlayerID': dst['OperatorPlayerID']}
    ]
    
    lineup_df = pd.DataFrame(lineup_data)
    
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

def export_lineups(lineups, output_filename="nfl_week8_fixed_lineups.csv"):
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
            'Entry ID': f"week8_fixed_{lineup['lineup_id']:03d}",
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
    Main function to run the fixed Week 8 optimizer
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║              NFL WEEK 8 FIXED OPTIMIZER                 ║
    ║              Handles DST without projections             ║
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
        'FantasyPoints': 'FantasyPoints',  # Keep original name
        'Value': 'PointsPerK'
    })
    
    print(f"📊 Data Summary:")
    print(f"   Total Players: {len(df)}")
    print(f"   Players with Projections: {df['FantasyPoints'].gt(0).sum()}")
    print(f"   Positions: {df['Position'].value_counts().to_dict()}")
    
    # Generate lineups
    lineups = create_fixed_lineups(df, num_lineups=150)
    
    if lineups:
        # Analyze lineups
        analyze_lineups(lineups)
        
        # Export lineups
        export_lineups(lineups)
        
        print(f"\n" + "="*80)
        print("✅ WEEK 8 FIXED OPTIMIZATION COMPLETE!")
        print("="*80)
        print(f"📁 Generated: nfl_week8_fixed_lineups.csv")
        print(f"🎯 Ready for DraftKings upload")
        print(f"💰 Good luck in your tournaments!")
    else:
        print("❌ Failed to generate any lineups")

if __name__ == "__main__":
    main()
