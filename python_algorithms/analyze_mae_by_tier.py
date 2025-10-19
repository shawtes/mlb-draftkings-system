#!/usr/bin/env python3
"""
Analyze MAE by Player Tier (No API Calls)
==========================================

This script reads the already-fetched data and segments players into tiers
based on their actual fantasy points scored, then calculates MAE for each tier.

Tiers:
- Elite (20+ points): Star players with huge games
- High (15-19.99 points): Strong performers
- Medium-High (10-14.99 points): Solid contributors
- Medium (5-9.99 points): Bench/flex players
- Low (0-4.99 points): Minimal production
"""

import pandas as pd
import numpy as np
import sys


def analyze_mae_by_tier(csv_filename: str):
    """
    Analyze MAE across different player performance tiers
    
    Args:
        csv_filename: Path to the detailed comparison CSV file
    """
    print("="*70)
    print("MAE ANALYSIS BY PLAYER TIER")
    print("="*70)
    print(f"\n📂 Reading data from: {csv_filename}\n")
    
    # Read the data
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_filename}")
        print("   Run option 3 in sportsdata_nfl_api.py first to generate the data")
        return
    
    print(f"✅ Loaded {len(df)} players\n")
    
    # Define tier boundaries based on ACTUAL fantasy points scored
    tiers = [
        {'name': 'Elite (20+ pts)', 'min': 20, 'max': float('inf'), 'emoji': '🌟'},
        {'name': 'High (15-19.99 pts)', 'min': 15, 'max': 19.99, 'emoji': '🔥'},
        {'name': 'Medium-High (10-14.99 pts)', 'min': 10, 'max': 14.99, 'emoji': '💪'},
        {'name': 'Medium (5-9.99 pts)', 'min': 5, 'max': 9.99, 'emoji': '📊'},
        {'name': 'Low (0-4.99 pts)', 'min': 0, 'max': 4.99, 'emoji': '📉'},
    ]
    
    # Stats to analyze
    stat_columns = [
        'FantasyPoints',
        'PassingYards',
        'PassingTouchdowns',
        'RushingYards',
        'RushingTouchdowns',
        'ReceivingYards',
        'ReceivingTouchdowns',
        'Receptions'
    ]
    
    # Results storage
    tier_results = []
    
    print("="*70)
    print("TIER BREAKDOWN")
    print("="*70)
    
    # Analyze each tier
    for tier in tiers:
        # Filter data for this tier
        tier_df = df[
            (df['FantasyPoints_actual'] >= tier['min']) & 
            (df['FantasyPoints_actual'] < tier['max'])
        ]
        
        n_players = len(tier_df)
        
        if n_players == 0:
            print(f"\n{tier['emoji']} {tier['name']}: No players")
            continue
        
        print(f"\n{tier['emoji']} {tier['name']}")
        print(f"   Players: {n_players}")
        
        # Calculate stats for this tier
        tier_stats = {
            'Tier': tier['name'],
            'Players': n_players,
            'Emoji': tier['emoji']
        }
        
        # Calculate MAE for each stat
        for stat in stat_columns:
            proj_col = f"{stat}_proj"
            actual_col = f"{stat}_actual"
            
            if proj_col in tier_df.columns and actual_col in tier_df.columns:
                # Remove NaN values
                valid = tier_df[[proj_col, actual_col]].dropna()
                
                if len(valid) > 0:
                    # MAE
                    mae = abs(valid[proj_col] - valid[actual_col]).mean()
                    
                    # Mean values
                    mean_proj = valid[proj_col].mean()
                    mean_actual = valid[actual_col].mean()
                    
                    tier_stats[f'{stat}_MAE'] = mae
                    tier_stats[f'{stat}_Proj_Avg'] = mean_proj
                    tier_stats[f'{stat}_Actual_Avg'] = mean_actual
                    
                    if stat == 'FantasyPoints':
                        print(f"   Fantasy Points MAE: {mae:.2f}")
                        print(f"   Avg Projected: {mean_proj:.2f} | Avg Actual: {mean_actual:.2f}")
                        print(f"   Bias: {mean_actual - mean_proj:+.2f} points")
        
        tier_results.append(tier_stats)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(tier_results)
    
    # Display fantasy points comparison
    print("\n" + "="*70)
    print("FANTASY POINTS MAE BY TIER")
    print("="*70)
    
    fp_summary = results_df[['Emoji', 'Tier', 'Players', 'FantasyPoints_MAE', 
                              'FantasyPoints_Proj_Avg', 'FantasyPoints_Actual_Avg']].copy()
    fp_summary['Bias'] = fp_summary['FantasyPoints_Actual_Avg'] - fp_summary['FantasyPoints_Proj_Avg']
    fp_summary.columns = ['', 'Tier', 'N', 'MAE', 'Proj Avg', 'Actual Avg', 'Bias']
    
    print(fp_summary.to_string(index=False))
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Find tier with best/worst MAE
    fp_summary_sorted = fp_summary.sort_values('MAE')
    best_tier = fp_summary_sorted.iloc[0]
    worst_tier = fp_summary_sorted.iloc[-1]
    
    print(f"\n✅ BEST ACCURACY:  {best_tier['Tier']}")
    print(f"   MAE: {best_tier['MAE']:.2f} points")
    print(f"   This tier has the most reliable projections")
    
    print(f"\n❌ WORST ACCURACY: {worst_tier['Tier']}")
    print(f"   MAE: {worst_tier['MAE']:.2f} points")
    print(f"   This tier has the least reliable projections")
    
    # Analyze bias (over/under projection)
    print(f"\n📊 PROJECTION BIAS:")
    for _, row in fp_summary.iterrows():
        bias = row['Bias']
        tier_name = row['Tier']
        if bias > 1:
            print(f"   ⬆️  {tier_name}: UNDER-projected by {bias:.2f} points")
        elif bias < -1:
            print(f"   ⬇️  {tier_name}: OVER-projected by {abs(bias):.2f} points")
        else:
            print(f"   ✅ {tier_name}: Well calibrated (bias: {bias:+.2f})")
    
    # Save detailed breakdown
    output_file = "mae_by_tier_2025REG_week6.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved detailed tier analysis to: {output_file}")
    
    # Show top players from elite tier
    if len(df[df['FantasyPoints_actual'] >= 20]) > 0:
        print("\n" + "="*70)
        print("ELITE TIER PLAYERS (20+ points)")
        print("="*70)
        elite_players = df[df['FantasyPoints_actual'] >= 20].sort_values(
            'FantasyPoints_actual', ascending=False
        ).head(15)
        
        elite_display = elite_players[['Name_proj', 'Position_proj', 
                                       'FantasyPoints_proj', 'FantasyPoints_actual',
                                       'FantasyPoints_error']].copy()
        elite_display.columns = ['Name', 'Pos', 'Projected', 'Actual', 'Error']
        print(elite_display.to_string(index=False))
    
    return results_df


def main():
    """
    Main function
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        MAE Analysis by Player Tier (Offline)            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Use the already-downloaded data
    csv_file = "detailed_comparison_2025REG_week6.csv"
    
    print(f"Analyzing: {csv_file}")
    print("This uses previously downloaded data (no API calls)\n")
    
    results = analyze_mae_by_tier(csv_file)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Files created:")
    print("   ✓ mae_by_tier_2025REG_week6.csv")
    print("\n💡 Use this analysis to:")
    print("   1. Adjust projections based on player tier")
    print("   2. Identify which fantasy point ranges are most predictable")
    print("   3. Build better DFS lineups by understanding projection reliability")


if __name__ == "__main__":
    main()

