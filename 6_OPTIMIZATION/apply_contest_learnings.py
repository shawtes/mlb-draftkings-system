#!/usr/bin/env python3
"""
Apply learnings from contest analysis to improve player projections
Based on real contest results from Contest 183480502
"""

import pandas as pd
import sys

def apply_contest_learnings(player_file, output_file=None):
    """
    Adjust player projections based on contest performance patterns
    """
    
    print("="*100)
    print("APPLYING CONTEST LEARNINGS TO PLAYER PROJECTIONS")
    print("="*100)
    
    # Load player data
    df = pd.read_csv(player_file)
    print(f"\nLoaded {len(df)} players from {player_file}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create a copy for modifications
    df_improved = df.copy()
    
    # Add a new column for adjusted projections if not exists
    if 'AvgPointsPerGame' in df.columns:
        base_proj_col = 'AvgPointsPerGame'
    elif 'Projection' in df.columns:
        base_proj_col = 'Projection'
    elif 'FPPG' in df.columns:
        base_proj_col = 'FPPG'
    else:
        print("⚠️  No projection column found. Using Salary/1000 as baseline.")
        df_improved['Projection'] = df['Salary'] / 1000 * 3
        base_proj_col = 'Projection'
    
    df_improved['Adjusted_Projection'] = df_improved[base_proj_col]
    df_improved['Confidence'] = 1.0
    df_improved['Notes'] = ''
    
    changes = []
    
    # 1. BOOST ELITE PERFORMERS
    elite_boosts = {
        # QBs
        'Jalen Hurts': 1.15,
        'Patrick Mahomes': 1.12,
        'Drake Maye': 1.10,
        
        # RBs
        'Quinshon Judkins': 1.20,
        "D'Andre Swift": 1.18,
        'Rhamondre Stevenson': 1.15,
        'Kyle Monangai': 1.12,
        
        # WRs - ELITE TIER
        'DeVonta Smith': 1.25,
        'A.J. Brown': 1.22,
        'Chris Olave': 1.18,
        'Jordan Addison': 1.15,
        'Xavier Legette': 1.20,
        'Rashee Rice': 1.12,
        
        # TEs
        'Juwan Johnson': 1.18,
        'T.J. Hockenson': 1.12,
        'Austin Hooper': 1.15
    }
    
    for player, boost in elite_boosts.items():
        mask = df_improved['Name'].str.contains(player, case=False, na=False)
        if mask.any():
            old_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            df_improved.loc[mask, 'Adjusted_Projection'] *= boost
            df_improved.loc[mask, 'Confidence'] = 1.2
            df_improved.loc[mask, 'Notes'] = 'Elite performer - boosted'
            new_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            changes.append(f"✅ BOOSTED {player}: {old_val:.1f} → {new_val:.1f} ({boost}x)")
    
    # 2. PENALIZE KNOWN BUSTS
    bust_penalties = {
        # QBs
        'Caleb Williams': 0.60,
        'Justin Fields': 0.55,
        'Tua Tagovailoa': 0.50,
        
        # RBs - EXPENSIVE DISAPPOINTMENTS
        'Ashton Jeanty': 0.50,
        'Saquon Barkley': 0.55,
        'Alvin Kamara': 0.60,
        'Breece Hall': 0.65,
        'TreVeyon Henderson': 0.40,
        
        # WRs - LOW CEILING BUSTS
        'Rome Odunze': 0.55,
        'Jaylen Waddle': 0.50,
        'Elic Ayomanor': 0.50,
        'Jerry Jeudy': 0.55,
        'Garrett Wilson': 0.45,
        'Jakobi Meyers': 0.45,
        'Luther Burden III': 0.50,
        
        # TEs - ZERO POINT TRAPS
        'Darren Waller': 0.30,
        'Brock Bowers': 0.50,
        'Michael Mayer': 0.40,
        'Chig Okonkwo': 0.45,
        'David Njoku': 0.45
    }
    
    for player, penalty in bust_penalties.items():
        mask = df_improved['Name'].str.contains(player, case=False, na=False)
        if mask.any():
            old_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            df_improved.loc[mask, 'Adjusted_Projection'] *= penalty
            df_improved.loc[mask, 'Confidence'] = 0.5
            df_improved.loc[mask, 'Notes'] = 'Contest bust - penalized'
            new_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            changes.append(f"❌ PENALIZED {player}: {old_val:.1f} → {new_val:.1f} ({penalty}x)")
    
    # 3. BOOST DEFENSES THAT PERFORMED
    dst_boosts = {
        'Browns': 1.30,
        'Patriots': 1.25,
        'Panthers': 1.22,
        'Chiefs': 1.10,
        'Bears': 1.10,
        'Eagles': 1.08
    }
    
    for dst, boost in dst_boosts.items():
        mask = df_improved['Name'].str.contains(dst, case=False, na=False) & (df_improved['Position'] == 'DST')
        if mask.any():
            old_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            df_improved.loc[mask, 'Adjusted_Projection'] *= boost
            df_improved.loc[mask, 'Notes'] = 'Strong DST performance'
            new_val = df_improved.loc[mask, 'Adjusted_Projection'].values[0]
            changes.append(f"🛡️  BOOSTED {dst} DST: {old_val:.1f} → {new_val:.1f} ({boost}x)")
    
    # 4. TEAM-BASED ADJUSTMENTS
    team_multipliers = {
        'KC': 1.08,   # Chiefs - best offense
        'PHI': 1.10,  # Eagles - elite
        'NE': 1.05,   # Patriots - solid
        'MIN': 1.02,  # Vikings - good
        'MIA': 0.85,  # Dolphins - struggled
        'LV': 0.70,   # Raiders - worst
        'CHI': 0.90   # Bears - risky
    }
    
    for team, mult in team_multipliers.items():
        mask = (df_improved['Team'] == team) & (df_improved['Position'] != 'DST')
        count = mask.sum()
        if count > 0:
            df_improved.loc[mask, 'Adjusted_Projection'] *= mult
            if mult > 1.0:
                changes.append(f"📈 TEAM BOOST {team}: {count} players boosted by {mult}x")
            else:
                changes.append(f"📉 TEAM PENALTY {team}: {count} players penalized by {mult}x")
    
    # Print summary of changes
    print("\n" + "="*100)
    print("ADJUSTMENTS APPLIED")
    print("="*100)
    for change in changes:
        print(change)
    
    # Calculate impact
    original_avg = df[base_proj_col].mean()
    adjusted_avg = df_improved['Adjusted_Projection'].mean()
    
    print("\n" + "="*100)
    print("IMPACT SUMMARY")
    print("="*100)
    print(f"Original Average Projection: {original_avg:.2f}")
    print(f"Adjusted Average Projection: {adjusted_avg:.2f}")
    print(f"Total Players Modified: {len([c for c in changes if 'BOOSTED' in c or 'PENALIZED' in c])}")
    
    # Show top players by adjusted projection
    print("\n" + "="*100)
    print("TOP 30 PLAYERS BY ADJUSTED PROJECTION")
    print("="*100)
    
    top_players = df_improved.nlargest(30, 'Adjusted_Projection')
    for i, (idx, row) in enumerate(top_players.iterrows(), 1):
        conf_emoji = "🔥" if row['Confidence'] > 1.0 else "⚠️" if row['Confidence'] < 1.0 else "📊"
        print(f"{i:2d}. {conf_emoji} {row['Name']:25s} ({row['Position']:3s}-{row['Team']:3s}) "
              f"${row['Salary']:5.0f} → {row['Adjusted_Projection']:5.1f} pts")
    
    # Save to file
    if output_file is None:
        output_file = player_file.replace('.csv', '_OPTIMIZED.csv')
    
    df_improved.to_csv(output_file, index=False)
    print(f"\n✅ Saved optimized player pool to: {output_file}")
    
    return df_improved

if __name__ == "__main__":
    # Apply to current week's data (you can change this for future weeks)
    input_file = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nfl_week7_DK_PLAYER_POOL_COMPLETE.csv"
    output_file = "/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nfl_week7_DK_PLAYER_POOL_OPTIMIZED.csv"
    
    apply_contest_learnings(input_file, output_file)

