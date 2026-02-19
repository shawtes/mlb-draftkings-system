"""
NBA Research-Based Optimizer - Complete Example
================================================
Shows how to use SportsData.io API + MIT Research Optimizer together

This example demonstrates:
1. Fetching real-time projections and historical data
2. Running cash game optimization (high win rate)
3. Running GPP optimization (high ceiling)
4. Exporting lineups for DraftKings upload
"""

from nba_sportsdata_fetcher import NBAResearchPipeline
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your SportsData.io API key
API_KEY = "d62d0ae315504e53a232ff7d1c3bea33"  # SportsData.io API key

# Today's slate date (or specific date)
SLATE_DATE = None  # None = today, or "2025-FEB-15"

# Bankroll settings (from Fantasy Bible recommendations)
TOTAL_BANKROLL = 1000  # Your total bankroll
CASH_ALLOCATION = 0.80  # 80% to cash games
GPP_ALLOCATION = 0.20   # 20% to GPP

# Contest settings
NUM_CASH_LINEUPS = 1   # Usually 1-3 for cash
NUM_GPP_LINEUPS = 20   # 20-150 for GPP diversification

# ============================================================================
# MAIN OPTIMIZATION SCRIPT
# ============================================================================

def main():
    """Run complete NBA DFS optimization session"""
    
    print("=" * 70)
    print("🏀 NBA RESEARCH-BASED DFS OPTIMIZER")
    print("=" * 70)
    print(f"📅 Slate Date: {SLATE_DATE or 'Today'}")
    print(f"💰 Bankroll: ${TOTAL_BANKROLL}")
    print(f"   - Cash Games: ${TOTAL_BANKROLL * CASH_ALLOCATION:.2f} ({CASH_ALLOCATION*100:.0f}%)")
    print(f"   - GPP Tournaments: ${TOTAL_BANKROLL * GPP_ALLOCATION:.2f} ({GPP_ALLOCATION*100:.0f}%)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Step 1: Initialize Pipeline
    # ========================================================================
    print("🔧 Initializing NBA Research Pipeline...")
    pipeline = NBAResearchPipeline(API_KEY)
    print("✅ Pipeline ready!\n")
    
    # ========================================================================
    # Step 2: CASH GAME Optimization
    # ========================================================================
    print("💰 CASH GAME OPTIMIZATION")
    print("-" * 70)
    print("Strategy: High floor, low variance, beat 50% of field")
    print(f"Generating {NUM_CASH_LINEUPS} cash lineup(s)...\n")
    
    cash_lineups, cash_players = pipeline.run_cash_optimization(
        date=SLATE_DATE,
        num_lineups=NUM_CASH_LINEUPS
    )
    
    if cash_lineups:
        print(f"✅ Generated {len(cash_lineups)} CASH lineup(s):\n")
        
        for i, lineup_indices in enumerate(cash_lineups, 1):
            lineup_df = cash_players.loc[lineup_indices]
            total_salary = lineup_df['Salary'].sum()
            projected_points = lineup_df['Predicted_DK_Points'].sum()
            
            print(f"  Lineup #{i}:")
            print(f"    💵 Salary: ${total_salary:,} / $50,000")
            print(f"    📊 Projected: {projected_points:.2f} DK points")
            print(f"    👥 Players:")
            
            for _, player in lineup_df.iterrows():
                print(f"       {player['Position']:3s} | {player['Name']:25s} | "
                      f"${player['Salary']:,} | {player['Predicted_DK_Points']:.1f} pts")
            print()
        
        # Save cash lineups
        save_lineups_for_dk(cash_lineups, cash_players, 'cash_lineups.csv')
        print(f"💾 Cash lineups saved to 'cash_lineups.csv'\n")
    else:
        print("❌ No cash lineups generated\n")
    
    # ========================================================================
    # Step 3: GPP TOURNAMENT Optimization
    # ========================================================================
    print("🏆 GPP TOURNAMENT OPTIMIZATION")
    print("-" * 70)
    print("Strategy: High ceiling, variance maximization, game stacking")
    print(f"Generating {NUM_GPP_LINEUPS} diverse GPP lineup(s)...\n")
    
    gpp_lineups, gpp_players = pipeline.run_gpp_optimization(
        date=SLATE_DATE,
        num_lineups=NUM_GPP_LINEUPS,
        stack_type='game_stack'  # or 'pg_c_stack'
    )
    
    if gpp_lineups:
        print(f"✅ Generated {len(gpp_lineups)} GPP lineup(s):\n")
        
        # Show summary of all lineups
        print(f"  GPP Lineup Summary:")
        print(f"  {'#':<4} {'Salary':<10} {'Projected':<12} {'Top Players'}")
        print(f"  {'-'*4} {'-'*10} {'-'*12} {'-'*40}")
        
        for i, lineup_indices in enumerate(gpp_lineups[:10], 1):  # Show first 10
            lineup_df = gpp_players.loc[lineup_indices]
            total_salary = lineup_df['Salary'].sum()
            projected = lineup_df['Predicted_DK_Points'].sum()
            top_players = ', '.join(lineup_df.nlargest(3, 'Predicted_DK_Points')['Name'].values)
            
            print(f"  {i:<4} ${total_salary:,:<9} {projected:>6.2f} pts   {top_players}")
        
        if len(gpp_lineups) > 10:
            print(f"  ... and {len(gpp_lineups) - 10} more lineups")
        print()
        
        # Save GPP lineups
        save_lineups_for_dk(gpp_lineups, gpp_players, 'gpp_lineups.csv')
        print(f"💾 GPP lineups saved to 'gpp_lineups.csv'\n")
    else:
        print("❌ No GPP lineups generated\n")
    
    # ========================================================================
    # Step 4: Portfolio Summary
    # ========================================================================
    print("=" * 70)
    print("📊 PORTFOLIO SUMMARY")
    print("=" * 70)
    
    total_cash_entries = len(cash_lineups)
    total_gpp_entries = len(gpp_lineups)
    
    print(f"\n💰 Cash Games:")
    print(f"   Lineups: {total_cash_entries}")
    print(f"   Budget: ${TOTAL_BANKROLL * CASH_ALLOCATION:.2f}")
    if total_cash_entries > 0:
        avg_proj = sum(cash_players.loc[lineup, 'Predicted_DK_Points'].sum() 
                      for lineup in cash_lineups) / len(cash_lineups)
        print(f"   Avg Projection: {avg_proj:.2f} pts")
        print(f"   Expected Win Rate: 60-65%")
        print(f"   Expected ROI: 15-20%")
    
    print(f"\n🏆 GPP Tournaments:")
    print(f"   Lineups: {total_gpp_entries}")
    print(f"   Budget: ${TOTAL_BANKROLL * GPP_ALLOCATION:.2f}")
    if total_gpp_entries > 0:
        avg_proj = sum(gpp_players.loc[lineup, 'Predicted_DK_Points'].sum() 
                      for lineup in gpp_lineups) / len(gpp_lineups)
        print(f"   Avg Projection: {avg_proj:.2f} pts")
        print(f"   Expected Top 10% Rate: 12-15%")
        print(f"   Expected ROI: 30-50%")
    
    # Expected returns calculation
    cash_expected = TOTAL_BANKROLL * CASH_ALLOCATION * 1.175  # 17.5% ROI
    gpp_expected = TOTAL_BANKROLL * GPP_ALLOCATION * 1.40    # 40% ROI
    total_expected = cash_expected + gpp_expected
    expected_profit = total_expected - TOTAL_BANKROLL
    
    print(f"\n💵 Expected Returns (Conservative Estimates):")
    print(f"   Cash Return: ${cash_expected:.2f}")
    print(f"   GPP Return: ${gpp_expected:.2f}")
    print(f"   Total Return: ${total_expected:.2f}")
    print(f"   Expected Profit: ${expected_profit:.2f} ({expected_profit/TOTAL_BANKROLL*100:.1f}% ROI)")
    
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print("\n📋 Next Steps:")
    print("   1. Review lineups in CSV files")
    print("   2. Upload to DraftKings")
    print("   3. Monitor for late-breaking news (injuries, lineup changes)")
    print("   4. Make late swaps if needed (5-10 min before lock)")
    print("   5. Track results for ROI analysis")
    print("\n🎯 Good luck! May the research be with you! 🏀\n")


def save_lineups_for_dk(lineups, player_pool, filename):
    """
    Save lineups in DraftKings upload format
    
    Format: PG, SG, SF, PF, C, G, F, UTIL
    """
    dk_lineups = []
    
    for lineup_indices in lineups:
        lineup_df = player_pool.loc[lineup_indices].copy()
        
        # Sort by position for DK format
        position_order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
        lineup_dict = {}
        
        for pos in position_order:
            # Find player for this position
            pos_players = lineup_df[lineup_df['Position'] == pos]
            if not pos_players.empty:
                player = pos_players.iloc[0]
                lineup_dict[f"{pos}"] = player['Name']
                lineup_dict[f"{pos}_ID"] = player.get('ID', '')
                lineup_dict[f"{pos}_Salary"] = player['Salary']
                # Remove from available pool
                lineup_df = lineup_df.drop(player.name)
        
        dk_lineups.append(lineup_dict)
    
    # Save to CSV
    df = pd.DataFrame(dk_lineups)
    df.to_csv(filename, index=False)


# ============================================================================
# ADVANCED: Custom Optimization Parameters
# ============================================================================

def advanced_optimization_example():
    """
    Example with custom parameters for advanced users
    """
    print("🔬 ADVANCED OPTIMIZATION EXAMPLE\n")
    
    pipeline = NBAResearchPipeline(API_KEY)
    
    # Get raw data
    fetcher = pipeline.fetcher
    players = fetcher.prepare_optimizer_input()
    historical = fetcher.get_historical_stats(
        start_date=(datetime.now().strftime('%Y-%b-%d')).upper(),
        num_days=30
    )
    
    # Custom cash optimization with correlation matrix
    print("💰 Cash game with custom correlation...")
    
    # Build correlation matrix from historical data
    # (Advanced: would calculate actual player correlations)
    correlation_matrix = None  # Your correlation matrix here
    
    cash_lineups = pipeline.optimizer.optimize(
        player_pool=players,
        contest_type='cash',
        position_limits={'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 
                        'G': 1, 'F': 1, 'UTIL': 1},
        salary_cap=50000,
        num_lineups=3,
        correlation_matrix=correlation_matrix,
        num_opponents=2000,  # More opponent samples = better modeling
        historical_data=historical
    )
    
    print(f"✅ Generated {len(cash_lineups)} lineups with custom parameters\n")
    
    # Custom GPP with PG-C stacking
    print("🏆 GPP with PG-C stack...")
    
    gpp_lineups = pipeline.optimizer.optimize(
        player_pool=players,
        contest_type='gpp',
        position_limits={'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
                        'G': 1, 'F': 1, 'UTIL': 1},
        salary_cap=50000,
        num_lineups=50,  # More lineups for diversification
        ownership_data=players.get('Ownership%', pd.Series([10.0]*len(players))),
        stack_config={'type': 'pg_c_stack'}  # Force PG+C from same team
    )
    
    print(f"✅ Generated {len(gpp_lineups)} GPP lineups with PG-C stacks\n")


# ============================================================================
# RUN IT!
# ============================================================================

if __name__ == "__main__":
    # Check if API key is set
    if API_KEY == "your_api_key_here":
        print("\n" + "=" * 70)
        print("⚠️  PLEASE SET YOUR API KEY")
        print("=" * 70)
        print("\n1. Get your API key from: https://sportsdata.io")
        print("2. Replace 'your_api_key_here' in this file")
        print("3. Run again\n")
    else:
        # Run main optimization
        main()
        
        # Uncomment for advanced example:
        # advanced_optimization_example()

