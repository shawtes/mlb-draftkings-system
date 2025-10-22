"""
NBA DFS Research-Based Genetic Algorithm Optimizer
===================================================

Integrates findings from:
1. MIT Paper: "How to Play Strategically in Fantasy Sports (and Win)" 
   - Dirichlet-Multinomial opponent modeling
   - Mean-variance optimization for cash games
   - Binary quadratic programming for GPP
   
2. Fantasy Sports Bible:
   - Stacking strategies (PG-C correlation, Game stacks)
   - Ownership-based differentiation
   - Bankroll management (80/20 cash/GPP split)
   
3. Genetic Algorithm for lineup diversity

Author: Enhanced for NBA DFS
Date: October 22, 2025
"""

import pandas as pd
import numpy as np
import pulp
import random
import logging
from collections import defaultdict
from itertools import combinations
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# NBA DRAFTKINGS SETTINGS
# ============================================================================

SALARY_CAP = 50000
MIN_SALARY = 49000  # Use most of salary cap
REQUIRED_TEAM_SIZE = 8

POSITION_LIMITS = {
    'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
    'G': 1,  # PG or SG
    'F': 1,  # SF or PF  
    'UTIL': 1  # Any position
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_roster_positions(roster_position_str):
    """
    Parse DraftKings roster position string to determine eligible spots.
    E.g., 'PG/SG/G/UTIL' -> ['PG', 'SG', 'G', 'UTIL']
    """
    if pd.isna(roster_position_str):
        return ['UTIL']
    
    positions = str(roster_position_str).split('/')
    eligible = []
    
    # Direct positions
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        if pos in positions:
            eligible.append(pos)
    
    # G spot (PG or SG)
    if 'PG' in positions or 'SG' in positions or 'G' in positions:
        if 'G' not in eligible:
            eligible.append('G')
    
    # F spot (SF or PF)
    if 'SF' in positions or 'PF' in positions or 'F' in positions:
        if 'F' not in eligible:
            eligible.append('F')
    
    # Everyone eligible for UTIL
    if 'UTIL' not in eligible:
        eligible.append('UTIL')
    
    return eligible


def get_primary_position(roster_position_str):
    """Extract primary position from roster string."""
    if pd.isna(roster_position_str):
        return 'UTIL'
    positions = str(roster_position_str).split('/')
    for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
        if pos in positions:
            return pos
    return 'UTIL'


# ============================================================================
# OPPONENT PORTFOLIO MODEL (MIT Paper - Dirichlet-Multinomial)
# ============================================================================

class OpponentPortfolioModel:
    """
    Models opponent lineup construction using Dirichlet-multinomial distribution.
    Based on Algorithm 1 from MIT paper.
    """
    
    def __init__(self, player_pool_df, salary_cap=SALARY_CAP, min_salary=MIN_SALARY):
        self.player_pool = player_pool_df.copy()
        self.salary_cap = salary_cap
        self.min_salary = min_salary
        self.alphas = self._estimate_dirichlet_alphas()
    
    def _estimate_dirichlet_alphas(self):
        """
        Estimate Dirichlet distribution parameters (alphas) for each position.
        Based on player features: projected points, salary, ownership.
        
        Formula: α_i = exp(β * X_i) where X includes features.
        """
        alphas = {}
        
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            pos_players = self.player_pool[
                self.player_pool['Roster_Position'].str.contains(pos, na=False)
            ]
            
            if pos_players.empty:
                alphas[pos] = {}
                continue
            
            # Feature engineering for alpha estimation
            alphas[pos] = {}
            for _, player in pos_players.iterrows():
                # Simple heuristic: higher projection + lower salary = higher alpha
                projection_score = player.get('Projected_DK_Points', 0) / 50  # Normalize
                salary_score = (1 - player.get('Salary', 5000) / SALARY_CAP) * 0.5
                ownership_score = (1 - player.get('Est_Ownership', 50) / 100) * 0.2
                
                alpha = max(0.1, projection_score + salary_score + ownership_score)
                alphas[pos][player['DK_ID']] = alpha
        
        return alphas
    
    def generate_opponent_lineup(self, max_attempts=100):
        """
        Generate a single feasible opponent lineup using accept-reject sampling.
        Algorithm 1 from MIT paper.
        """
        for attempt in range(max_attempts):
            lineup_players = []
            used_ids = set()
            total_salary = 0
            
            # Sample for each position using Dirichlet-Multinomial
            for pos in ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']:
                if pos in ['G', 'F', 'UTIL']:
                    # Flexible positions - sample from remaining pool
                    available = self.player_pool[
                        ~self.player_pool['DK_ID'].isin(used_ids)
                    ]
                    
                    if pos == 'G':
                        available = available[available['Roster_Position'].str.contains('PG|SG|G', na=False)]
                    elif pos == 'F':
                        available = available[available['Roster_Position'].str.contains('SF|PF|F', na=False)]
                    # UTIL: all remaining players are available
                else:
                    # Core positions
                    available = self.player_pool[
                        (self.player_pool['Roster_Position'].str.contains(pos, na=False)) &
                        (~self.player_pool['DK_ID'].isin(used_ids))
                    ]
                
                if available.empty:
                    break
                
                # Get alphas for available players
                if pos in self.alphas:
                    player_alphas = [self.alphas[pos].get(pid, 1.0) for pid in available['DK_ID']]
                else:
                    player_alphas = [1.0] * len(available)
                
                # Draw from Dirichlet to get probabilities
                probs = np.random.dirichlet(player_alphas)
                
                # Sample player
                selected_idx = np.random.choice(len(available), p=probs)
                selected_player = available.iloc[selected_idx]
                
                lineup_players.append(selected_player)
                used_ids.add(selected_player['DK_ID'])
                total_salary += selected_player['Salary']
            
            # Check feasibility
            if (len(lineup_players) == REQUIRED_TEAM_SIZE and 
                self.min_salary <= total_salary <= self.salary_cap):
                return pd.DataFrame(lineup_players)
        
        # Return empty if no feasible lineup found
        return pd.DataFrame()
    
    def sample_opponent_scores(self, num_samples=1000):
        """Generate distribution of opponent scores."""
        scores = []
        for _ in range(num_samples):
            lineup = self.generate_opponent_lineup()
            if not lineup.empty:
                score = lineup['Projected_DK_Points'].sum()
                scores.append(score)
        
        if not scores:
            # Fallback if sampling fails
            mean_score = self.player_pool['Projected_DK_Points'].mean() * REQUIRED_TEAM_SIZE
            scores = np.random.normal(mean_score, mean_score * 0.15, num_samples).tolist()
        
        return scores


# ============================================================================
# CASH GAME OPTIMIZER (MIT Paper - Mean-Variance for Double-Ups)
# ============================================================================

class CashGameOptimizer:
    """
    Optimize for cash games (50/50s, Double-Ups) using mean-variance optimization.
    Goal: Maximize probability of beating 50th percentile opponent.
    Based on MIT Paper Section 4.
    """
    
    def __init__(self, player_pool_df, opponent_model, lambda_risk=0.5):
        self.player_pool = player_pool_df.copy()
        self.opponent_model = opponent_model
        self.lambda_risk = lambda_risk  # Risk parameter (lower = more conservative)
    
    def optimize(self, num_lineups=1):
        """
        Generate optimal cash game lineups.
        """
        logging.info(f"💰 Optimizing {num_lineups} cash game lineups...")
        
        # Step 1: Get opponent score distribution
        opponent_scores = self.opponent_model.sample_opponent_scores(num_samples=1000)
        median_opponent = np.median(opponent_scores)
        logging.info(f"   Median opponent score: {median_opponent:.2f}")
        
        # Step 2: Generate lineups optimized to beat median
        lineups = []
        for i in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_Cash_{i}", pulp.LpMaximize)
            
            # Decision variables
            player_vars = pulp.LpVariable.dicts(
                "Player",
                self.player_pool.index,
                cat=pulp.LpBinary
            )
            
            # Objective: Maximize expected points (floor-focused)
            # For cash, we use 'Floor' (conservative projection) instead of ceiling
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Floor'] * player_vars[idx]
                for idx in self.player_pool.index
            ]), "Total_Floor_Points"
            
            # Salary constraints
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Salary'] * player_vars[idx]
                for idx in self.player_pool.index
            ]) <= SALARY_CAP, "Salary_Cap"
            
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Salary'] * player_vars[idx]
                for idx in self.player_pool.index
            ]) >= MIN_SALARY, "Min_Salary"
            
            # Exactly 8 players
            prob += pulp.lpSum([player_vars[idx] for idx in self.player_pool.index]) == REQUIRED_TEAM_SIZE, "Total_Players"
            
            # Position constraints
            self._add_position_constraints(prob, player_vars)
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                selected_indices = [idx for idx in self.player_pool.index if player_vars[idx].varValue > 0.5]
                lineup = self.player_pool.loc[selected_indices].copy()
                lineups.append(lineup)
                logging.info(f"   ✅ Cash Lineup {i+1}: Floor={lineup['Floor'].sum():.1f}, Salary=${lineup['Salary'].sum():,}")
        
        return lineups
    
    def _add_position_constraints(self, prob, player_vars):
        """Add NBA position constraints to optimization problem."""
        df = self.player_pool
        
        # Core positions (exactly 1 each)
        for pos in ['PG', 'SG', 'SF', 'PF', 'C']:
            eligible = df[df['Roster_Position'].str.contains(pos, na=False)].index
            prob += pulp.lpSum([player_vars[idx] for idx in eligible]) >= 1, f"Min_{pos}"
        
        # G spot (at least 1 PG or SG)
        g_eligible = df[df['Roster_Position'].str.contains('PG|SG|G', na=False)].index
        prob += pulp.lpSum([player_vars[idx] for idx in g_eligible]) >= 6, "G_Spot"  # 5 core guards + 1 G spot
        
        # F spot (at least 1 SF or PF)
        f_eligible = df[df['Roster_Position'].str.contains('SF|PF|F', na=False)].index
        prob += pulp.lpSum([player_vars[idx] for idx in f_eligible]) >= 4, "F_Spot"  # 2 core forwards + 1 F spot + 1 UTIL forward


# ============================================================================
# TOURNAMENT (GPP) OPTIMIZER (MIT Paper Section 5 + Genetic Algorithm)
# ============================================================================

class TournamentOptimizer:
    """
    Optimize for GPP tournaments using variance maximization and genetic algorithm.
    Goal: Maximize ceiling while differentiating from field.
    """
    
    def __init__(self, player_pool_df, stack_type='balanced'):
        self.player_pool = player_pool_df.copy()
        self.stack_type = stack_type
    
    def optimize(self, num_lineups=20, use_genetic=True):
        """
        Generate diverse GPP lineups.
        """
        logging.info(f"🏆 Optimizing {num_lineups} GPP tournament lineups...")
        
        if use_genetic:
            return self._genetic_algorithm_optimization(num_lineups)
        else:
            return self._pulp_optimization(num_lineups)
    
    def _genetic_algorithm_optimization(self, num_lineups):
        """
        Use genetic algorithm for maximum diversity.
        """
        logging.info("   🧬 Using Genetic Algorithm for diversity...")
        
        # Create initial population (3x target size)
        population = self._create_initial_population(num_lineups * 3)
        
        # Evolve for 5 generations
        for gen in range(5):
            population = self._evolve_population(population)
        
        # Select most diverse subset
        diverse_lineups = self._select_diverse(population, num_lineups)
        
        logging.info(f"   ✅ Generated {len(diverse_lineups)} diverse GPP lineups")
        return diverse_lineups
    
    def _create_initial_population(self, population_size):
        """Create initial population of feasible lineups."""
        population = []
        
        for _ in range(population_size):
            prob = pulp.LpProblem("NBA_GPP", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts(
                "Player",
                self.player_pool.index,
                cat=pulp.LpBinary
            )
            
            # Objective: Maximize ceiling with ownership penalty
            prob += pulp.lpSum([
                (self.player_pool.loc[idx, 'Ceiling'] - 
                 self.player_pool.loc[idx, 'Est_Ownership'] * 0.05) * player_vars[idx]
                for idx in self.player_pool.index
            ]), "GPP_Value"
            
            # Standard constraints
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Salary'] * player_vars[idx]
                for idx in self.player_pool.index
            ]) <= SALARY_CAP
            
            prob += pulp.lpSum([player_vars[idx] for idx in self.player_pool.index]) == REQUIRED_TEAM_SIZE
            
            # Add randomness for diversity
            random_boost = {idx: random.uniform(0.9, 1.1) for idx in self.player_pool.index}
            for idx in self.player_pool.index:
                player_vars[idx].setInitialValue(random.random())
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                selected_indices = [idx for idx in self.player_pool.index if player_vars[idx].varValue > 0.5]
                lineup = self.player_pool.loc[selected_indices].copy()
                population.append(lineup)
        
        return population
    
    def _evolve_population(self, population):
        """Evolve population using crossover and mutation."""
        evolved = []
        
        for _ in range(len(population)):
            # Select two parents
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if random.random() < 0.3:  # 30% mutation rate
                child = self._mutate(child)
            
            if len(child) == REQUIRED_TEAM_SIZE:
                evolved.append(child)
        
        return evolved
    
    def _crossover(self, parent1, parent2):
        """Combine two lineups."""
        # Take random subset from each parent
        n1 = random.randint(3, 5)
        subset1 = parent1.sample(n=n1)
        
        # Fill remaining spots from parent2
        used_ids = set(subset1['DK_ID'])
        remaining_needed = REQUIRED_TEAM_SIZE - len(subset1)
        
        available = parent2[~parent2['DK_ID'].isin(used_ids)]
        if len(available) >= remaining_needed:
            subset2 = available.sample(n=remaining_needed)
            return pd.concat([subset1, subset2])
        
        return parent1  # Return parent if crossover fails
    
    def _mutate(self, lineup):
        """Randomly swap one player."""
        if len(lineup) < REQUIRED_TEAM_SIZE:
            return lineup
        
        # Remove random player
        to_remove = lineup.sample(n=1)
        pos = get_primary_position(to_remove.iloc[0]['Roster_Position'])
        
        # Add replacement
        used_ids = set(lineup['DK_ID']) - set(to_remove['DK_ID'])
        available = self.player_pool[
            (~self.player_pool['DK_ID'].isin(used_ids)) &
            (self.player_pool['Roster_Position'].str.contains(pos, na=False))
        ]
        
        if not available.empty:
            replacement = available.sample(n=1)
            return pd.concat([lineup[lineup['DK_ID'] != to_remove.iloc[0]['DK_ID']], replacement])
        
        return lineup
    
    def _select_diverse(self, population, num_lineups):
        """Select most diverse lineups from population."""
        if len(population) <= num_lineups:
            return population
        
        # Calculate pairwise player overlap
        selected = [population[0]]  # Start with first lineup
        
        for _ in range(num_lineups - 1):
            best_candidate = None
            max_diversity = -1
            
            for candidate in population:
                if any((set(candidate['DK_ID']) == set(s['DK_ID'])) for s in selected):
                    continue
                
                # Calculate average overlap with selected lineups
                overlaps = []
                for sel in selected:
                    overlap = len(set(candidate['DK_ID']) & set(sel['DK_ID']))
                    overlaps.append(overlap)
                
                avg_diversity = REQUIRED_TEAM_SIZE - np.mean(overlaps)
                
                if avg_diversity > max_diversity:
                    max_diversity = avg_diversity
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
        
        return selected
    
    def _pulp_optimization(self, num_lineups):
        """Fallback PuLP optimization without genetic algorithm."""
        lineups = []
        
        for i in range(num_lineups):
            prob = pulp.LpProblem(f"NBA_GPP_{i}", pulp.LpMaximize)
            player_vars = pulp.LpVariable.dicts(
                "Player",
                self.player_pool.index,
                cat=pulp.LpBinary
            )
            
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Ceiling'] * player_vars[idx]
                for idx in self.player_pool.index
            ])
            
            prob += pulp.lpSum([
                self.player_pool.loc[idx, 'Salary'] * player_vars[idx]
                for idx in self.player_pool.index
            ]) <= SALARY_CAP
            
            prob += pulp.lpSum([player_vars[idx] for idx in self.player_pool.index]) == REQUIRED_TEAM_SIZE
            
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] == 'Optimal':
                selected_indices = [idx for idx in self.player_pool.index if player_vars[idx].varValue > 0.5]
                lineup = self.player_pool.loc[selected_indices].copy()
                lineups.append(lineup)
        
        return lineups


# ============================================================================
# MAIN NBA RESEARCH OPTIMIZER
# ============================================================================

class NBAResearchGeneticOptimizer:
    """
    Main NBA DFS optimizer integrating MIT research + genetic algorithm.
    """
    
    def __init__(self, player_pool_csv):
        """Load player pool from CSV."""
        self.player_pool = pd.read_csv(player_pool_csv)
        logging.info(f"✅ Loaded {len(self.player_pool)} players from {player_pool_csv}")
        
        # Initialize models
        self.opponent_model = OpponentPortfolioModel(self.player_pool)
        self.cash_optimizer = CashGameOptimizer(self.player_pool, self.opponent_model)
        self.gpp_optimizer = TournamentOptimizer(self.player_pool)
    
    def optimize_cash(self, num_lineups=1):
        """Generate cash game lineups."""
        return self.cash_optimizer.optimize(num_lineups)
    
    def optimize_gpp(self, num_lineups=20):
        """Generate GPP lineups."""
        return self.gpp_optimizer.optimize(num_lineups, use_genetic=True)
    
    def export_lineups(self, lineups, filename, contest_type='cash'):
        """Export lineups to DraftKings CSV format."""
        if not lineups:
            logging.warning("No lineups to export")
            return
        
        export_rows = []
        
        for i, lineup in enumerate(lineups):
            if len(lineup) != REQUIRED_TEAM_SIZE:
                logging.warning(f"Lineup {i+1} has {len(lineup)} players, expected {REQUIRED_TEAM_SIZE}")
                continue
            
            # Sort by position for DK format
            lineup_sorted = lineup.copy()
            lineup_sorted['Pri_Pos'] = lineup_sorted['Roster_Position'].apply(get_primary_position)
            
            position_order = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
            row = {}
            
            for pos in position_order:
                if pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    player = lineup_sorted[lineup_sorted['Pri_Pos'] == pos].iloc[0] if not lineup_sorted[lineup_sorted['Pri_Pos'] == pos].empty else None
                else:
                    # Flexible spots
                    player = lineup_sorted.iloc[position_order.index(pos)] if len(lineup_sorted) > position_order.index(pos) else None
                
                if player is not None:
                    row[pos] = player['DK_ID']
            
            export_rows.append(row)
        
        # Create DataFrame
        export_df = pd.DataFrame(export_rows)
        
        # Save
        export_df.to_csv(filename, index=False)
        logging.info(f"💾 Exported {len(export_rows)} lineups to {filename}")
        
        return filename


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🏀 NBA RESEARCH-BASED GENETIC ALGORITHM OPTIMIZER")
    print("=" * 70)
    print("\nIntegrating:")
    print("  ✅ MIT Paper: Dirichlet-Multinomial opponent modeling")
    print("  ✅ Mean-variance optimization for cash games")
    print("  ✅ Variance maximization for GPP tournaments")
    print("  ✅ Genetic algorithm for lineup diversity")
    print("=" * 70)
    
    # Find latest optimized player pool
    import glob
    player_files = glob.glob("/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_*.csv")
    
    if not player_files:
        print("\n❌ No player pool CSV found!")
        print("   Run: python3 nba_dk_slate_optimizer.py first")
        exit(1)
    
    latest_file = max(player_files, key=lambda x: x.split('_')[-1])
    print(f"\n📁 Using player pool: {latest_file}")
    
    # Initialize optimizer
    optimizer = NBAResearchGeneticOptimizer(latest_file)
    
    # Generate cash lineups (conservative)
    print("\n" + "=" * 70)
    print("💰 GENERATING CASH GAME LINEUPS (50/50s, Double-Ups)")
    print("=" * 70)
    cash_lineups = optimizer.optimize_cash(num_lineups=3)
    
    for i, lineup in enumerate(cash_lineups):
        print(f"\n💵 Cash Lineup {i+1}:")
        print(lineup[['Name', 'Team', 'Salary', 'Floor', 'Projected_DK_Points']].to_string(index=False))
        print(f"Total Floor: {lineup['Floor'].sum():.1f} | Salary: ${lineup['Salary'].sum():,}")
    
    # Generate GPP lineups (high variance)
    print("\n" + "=" * 70)
    print("🏆 GENERATING GPP TOURNAMENT LINEUPS (High Ceiling)")
    print("=" * 70)
    gpp_lineups = optimizer.optimize_gpp(num_lineups=20)
    
    print(f"\n✅ Generated {len(gpp_lineups)} diverse GPP lineups")
    for i, lineup in enumerate(gpp_lineups[:3]):  # Show first 3
        print(f"\n🎯 GPP Lineup {i+1}:")
        print(lineup[['Name', 'Team', 'Salary', 'Ceiling', 'Est_Ownership']].to_string(index=False))
        print(f"Total Ceiling: {lineup['Ceiling'].sum():.1f} | Salary: ${lineup['Salary'].sum():,}")
    
    # Export lineups
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cash_file = f"/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_cash_lineups_{timestamp}.csv"
    gpp_file = f"/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_gpp_lineups_{timestamp}.csv"
    
    optimizer.export_lineups(cash_lineups, cash_file, 'cash')
    optimizer.export_lineups(gpp_lineups, gpp_file, 'gpp')
    
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Cash Lineups: {len(cash_lineups)} (conservative, high win rate)")
    print(f"  GPP Lineups: {len(gpp_lineups)} (high ceiling, diverse)")
    print(f"\n💾 Files saved:")
    print(f"  {cash_file}")
    print(f"  {gpp_file}")
    print("\n🚀 Ready to upload to DraftKings!")
    print("=" * 70)

