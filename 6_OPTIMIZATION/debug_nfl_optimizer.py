#!/usr/bin/env python3
"""
Debug NFL Week 8 Optimizer
==========================

This script debugs the lineup creation process to identify issues.
"""

import sys
import os
import pandas as pd
import numpy as np

def debug_lineup_creation():
    """
    Debug the lineup creation process
    """
    print("🔍 DEBUGGING NFL LINEUP CREATION")
    print("="*50)
    
    # Load data
    csv_file = "nfl_week8_draftkings_13game_slate.csv"
    df = pd.read_csv(csv_file)
    
    # Rename columns
    df = df.rename(columns={
        'FantasyPoints': 'Predicted_DK_Points',
        'Value': 'PointsPerK'
    })
    
    # Filter to players with projections
    df_with_proj = df[df['Predicted_DK_Points'] > 0].copy()
    print(f"✅ Players with projections: {len(df_with_proj)}")
    
    # Check position distribution
    print(f"\n📊 Position Distribution:")
    pos_counts = df_with_proj['Position'].value_counts()
    print(pos_counts)
    
    # Check salary distribution
    print(f"\n💰 Salary Distribution:")
    print(f"   Min: ${df_with_proj['Salary'].min():,}")
    print(f"   Max: ${df_with_proj['Salary'].max():,}")
    print(f"   Mean: ${df_with_proj['Salary'].mean():,.0f}")
    print(f"   Median: ${df_with_proj['Salary'].median():,.0f}")
    
    # Check projections
    print(f"\n📈 Projection Distribution:")
    print(f"   Min: {df_with_proj['Predicted_DK_Points'].min():.1f}")
    print(f"   Max: {df_with_proj['Predicted_DK_Points'].max():.1f}")
    print(f"   Mean: {df_with_proj['Predicted_DK_Points'].mean():.1f}")
    
    # Try to create one lineup manually
    print(f"\n🎯 CREATING MANUAL LINEUP")
    print("="*30)
    
    salary_cap = 50000
    min_salary = 49000
    
    # Select QB
    qb_candidates = df_with_proj[df_with_proj['Position'] == 'QB'].copy()
    qb_candidates = qb_candidates.sort_values('Predicted_DK_Points', ascending=False)
    print(f"QB candidates: {len(qb_candidates)}")
    
    if qb_candidates.empty:
        print("❌ No QB candidates!")
        return
    
    qb = qb_candidates.iloc[0]
    print(f"Selected QB: {qb['Name']} (${qb['Salary']:,}, {qb['Predicted_DK_Points']:.1f} pts)")
    
    # Select RB (2)
    rb_candidates = df_with_proj[df_with_proj['Position'] == 'RB'].copy()
    rb_candidates = rb_candidates.sort_values('Predicted_DK_Points', ascending=False)
    print(f"RB candidates: {len(rb_candidates)}")
    
    if len(rb_candidates) < 2:
        print("❌ Not enough RB candidates!")
        return
    
    rb1 = rb_candidates.iloc[0]
    rb2 = rb_candidates.iloc[1]
    print(f"Selected RB1: {rb1['Name']} (${rb1['Salary']:,}, {rb1['Predicted_DK_Points']:.1f} pts)")
    print(f"Selected RB2: {rb2['Name']} (${rb2['Salary']:,}, {rb2['Predicted_DK_Points']:.1f} pts)")
    
    # Select WR (3)
    wr_candidates = df_with_proj[df_with_proj['Position'] == 'WR'].copy()
    wr_candidates = wr_candidates.sort_values('Predicted_DK_Points', ascending=False)
    print(f"WR candidates: {len(wr_candidates)}")
    
    if len(wr_candidates) < 3:
        print("❌ Not enough WR candidates!")
        return
    
    wr1 = wr_candidates.iloc[0]
    wr2 = wr_candidates.iloc[1]
    wr3 = wr_candidates.iloc[2]
    print(f"Selected WR1: {wr1['Name']} (${wr1['Salary']:,}, {wr1['Predicted_DK_Points']:.1f} pts)")
    print(f"Selected WR2: {wr2['Name']} (${wr2['Salary']:,}, {wr2['Predicted_DK_Points']:.1f} pts)")
    print(f"Selected WR3: {wr3['Name']} (${wr3['Salary']:,}, {wr3['Predicted_DK_Points']:.1f} pts)")
    
    # Select TE
    te_candidates = df_with_proj[df_with_proj['Position'] == 'TE'].copy()
    te_candidates = te_candidates.sort_values('Predicted_DK_Points', ascending=False)
    print(f"TE candidates: {len(te_candidates)}")
    
    if te_candidates.empty:
        print("❌ No TE candidates!")
        return
    
    te = te_candidates.iloc[0]
    print(f"Selected TE: {te['Name']} (${te['Salary']:,}, {te['Predicted_DK_Points']:.1f} pts)")
    
    # Select DST
    dst_candidates = df_with_proj[df_with_proj['Position'] == 'DST'].copy()
    dst_candidates = dst_candidates.sort_values('Predicted_DK_Points', ascending=False)
    print(f"DST candidates: {len(dst_candidates)}")
    
    if dst_candidates.empty:
        print("❌ No DST candidates!")
        return
    
    dst = dst_candidates.iloc[0]
    print(f"Selected DST: {dst['Name']} (${dst['Salary']:,}, {dst['Predicted_DK_Points']:.1f} pts)")
    
    # Calculate totals
    total_salary = qb['Salary'] + rb1['Salary'] + rb2['Salary'] + wr1['Salary'] + wr2['Salary'] + wr3['Salary'] + te['Salary'] + dst['Salary']
    total_projection = qb['Predicted_DK_Points'] + rb1['Predicted_DK_Points'] + rb2['Predicted_DK_Points'] + wr1['Predicted_DK_Points'] + wr2['Predicted_DK_Points'] + wr3['Predicted_DK_Points'] + te['Predicted_DK_Points'] + dst['Predicted_DK_Points']
    
    print(f"\n📊 LINEUP TOTALS:")
    print(f"   Total Salary: ${total_salary:,}")
    print(f"   Total Projection: {total_projection:.1f} pts")
    print(f"   Value: {total_projection / (total_salary / 1000):.3f}")
    print(f"   Salary Cap: ${salary_cap:,}")
    print(f"   Min Salary: ${min_salary:,}")
    print(f"   Within constraints: {min_salary <= total_salary <= salary_cap}")
    
    # Check if this lineup would be valid
    if min_salary <= total_salary <= salary_cap:
        print("✅ This lineup is VALID!")
        
        # Create the lineup DataFrame
        lineup_data = [
            {'Name': qb['Name'], 'Position': 'QB', 'Team': qb['Team'], 'Salary': qb['Salary'], 'Predicted_DK_Points': qb['Predicted_DK_Points'], 'OperatorPlayerID': qb['OperatorPlayerID']},
            {'Name': rb1['Name'], 'Position': 'RB', 'Team': rb1['Team'], 'Salary': rb1['Salary'], 'Predicted_DK_Points': rb1['Predicted_DK_Points'], 'OperatorPlayerID': rb1['OperatorPlayerID']},
            {'Name': rb2['Name'], 'Position': 'RB', 'Team': rb2['Team'], 'Salary': rb2['Salary'], 'Predicted_DK_Points': rb2['Predicted_DK_Points'], 'OperatorPlayerID': rb2['OperatorPlayerID']},
            {'Name': wr1['Name'], 'Position': 'WR', 'Team': wr1['Team'], 'Salary': wr1['Salary'], 'Predicted_DK_Points': wr1['Predicted_DK_Points'], 'OperatorPlayerID': wr1['OperatorPlayerID']},
            {'Name': wr2['Name'], 'Position': 'WR', 'Team': wr2['Team'], 'Salary': wr2['Salary'], 'Predicted_DK_Points': wr2['Predicted_DK_Points'], 'OperatorPlayerID': wr2['OperatorPlayerID']},
            {'Name': wr3['Name'], 'Position': 'WR', 'Team': wr3['Team'], 'Salary': wr3['Salary'], 'Predicted_DK_Points': wr3['Predicted_DK_Points'], 'OperatorPlayerID': wr3['OperatorPlayerID']},
            {'Name': te['Name'], 'Position': 'TE', 'Team': te['Team'], 'Salary': te['Salary'], 'Predicted_DK_Points': te['Predicted_DK_Points'], 'OperatorPlayerID': te['OperatorPlayerID']},
            {'Name': dst['Name'], 'Position': 'DST', 'Team': dst['Team'], 'Salary': dst['Salary'], 'Predicted_DK_Points': dst['Predicted_DK_Points'], 'OperatorPlayerID': dst['OperatorPlayerID']}
        ]
        
        lineup_df = pd.DataFrame(lineup_data)
        
        # Export this single lineup
        dk_lineup = {
            'Entry ID': 'week8_debug_001',
            'Contest Name': 'NFL Week 8 GPP Tournament',
            'Contest ID': '183787056',
            'Entry Fee': '$1',
            'QB': qb['OperatorPlayerID'],
            'RB': f"{rb1['OperatorPlayerID']},{rb2['OperatorPlayerID']}",
            'WR': f"{wr1['OperatorPlayerID']},{wr2['OperatorPlayerID']},{wr3['OperatorPlayerID']}",
            'TE': te['OperatorPlayerID'],
            'FLEX': '',
            'DST': dst['OperatorPlayerID']
        }
        
        export_df = pd.DataFrame([dk_lineup])
        export_df.to_csv('nfl_week8_debug_lineup.csv', index=False)
        print(f"\n💾 Exported debug lineup to: nfl_week8_debug_lineup.csv")
        
    else:
        print("❌ This lineup is INVALID due to salary constraints!")

if __name__ == "__main__":
    debug_lineup_creation()
