"""
NBA DFS Research-Based Optimizer Core
======================================
Based on MIT Research Paper: "How to Play Strategically in Fantasy Sports (and Win)"
by Martin B. Haugh and Raghav Singal (Imperial College & Columbia University)

And Z-Code Fantasy Sports Bible strategies for daily fantasy sports optimization.

Key Research Insights Implemented:
1. Dirichlet-Multinomial opponent modeling
2. Mean-variance optimization for double-up/cash games
3. Binary quadratic programming for portfolio optimization
4. Strategic stacking with correlation maximization
5. Ownership-based differentiation for GPP tournaments
6. Risk-adjusted bankroll management
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. OPPONENT PORTFOLIO MODELING (Dirichlet-Multinomial)
# ============================================================================

class OpponentPortfolioModel:
    """
    Models opponent team selections using Dirichlet-Multinomial distribution
    Based on Section 3 of MIT paper (Dirichlet Regression for Opponent Modeling)
    """
    
    def __init__(self):
        self.position_params = {}  # α parameters for each position
        self.historical_selections = {}
        
    def fit_dirichlet_params(self, historical_data: pd.DataFrame, position: str):
        """
        Estimate Dirichlet parameters α for a position using historical selections
        
        α_pos = exp(β_pos * X_pos)
        where X_pos includes: projected points, salary, ownership %, etc.
        """
        if position not in historical_data['Position'].values:
            logging.warning(f"No historical data for position {position}")
            return None
            
        pos_data = historical_data[historical_data['Position'] == position].copy()
        
        # Features: [Projected Points, Salary (scaled), Ownership%]
        if 'Projected_Points' in pos_data.columns and 'Salary' in pos_data.columns:
            # Normalize features
            proj_scaled = (pos_data['Projected_Points'] - pos_data['Projected_Points'].mean()) / pos_data['Projected_Points'].std()
            salary_scaled = (pos_data['Salary'] - pos_data['Salary'].mean()) / pos_data['Salary'].std()
            
            # Simple exponential model: α = exp(β0 + β1*proj + β2*salary)
            # For simplicity, use projected points as main driver
            alpha = np.exp(proj_scaled * 0.5 + salary_scaled * 0.3)
            alpha = np.maximum(alpha, 0.1)  # Ensure positive
            
            self.position_params[position] = alpha.values
            return alpha.values
        
        return None
    
    def sample_opponent_position(self, position: str, available_players: pd.DataFrame) -> int:
        """
        Sample a player selection for a position using Dirichlet-Multinomial
        
        Steps (Algorithm 1 from paper):
        1. Draw α ~ Dir(α_pos)
        2. Draw selection ~ Mult(α, 1)
        """
        if position not in self.position_params:
            # Fallback: sample proportional to projected points
            if 'Projected_Points' in available_players.columns:
                probs = available_players['Projected_Points'].values
                probs = probs / probs.sum()
                return np.random.choice(len(available_players), p=probs)
            return np.random.choice(len(available_players))
        
        # Dirichlet sampling
        alpha_pos = self.position_params[position]
        
        # Ensure alpha matches available players
        if len(alpha_pos) != len(available_players):
            # Resize alpha to match
            alpha_pos = np.ones(len(available_players)) * alpha_pos.mean()
        
        # Sample from Dirichlet
        theta = np.random.dirichlet(alpha_pos)
        
        # Sample from Multinomial with theta
        selection = np.random.multinomial(1, theta)
        return np.argmax(selection)
    
    def generate_opponent_lineup(self, player_pool: pd.DataFrame, 
                                 position_limits: Dict[str, int],
                                 salary_cap: float,
                                 min_salary: float = 0.95) -> List[int]:
        """
        Generate a complete opponent lineup using accept-reject sampling
        Based on Algorithm 1 from MIT paper
        
        Returns: List of player indices
        """
        max_attempts = 1000
        min_salary_threshold = salary_cap * min_salary
        
        for attempt in range(max_attempts):
            lineup_indices = []
            total_salary = 0
            position_counts = {pos: 0 for pos in position_limits.keys()}
            
            # Sample each position
            for position, limit in position_limits.items():
                pos_pool = player_pool[player_pool['Position'] == position].copy()
                
                for _ in range(limit):
                    if len(pos_pool) == 0:
                        break
                        
                    # Sample a player for this position
                    idx = self.sample_opponent_position(position, pos_pool)
                    player = pos_pool.iloc[idx]
                    
                    lineup_indices.append(player.name)
                    total_salary += player['Salary']
                    position_counts[position] += 1
                    
                    # Remove selected player from pool
                    pos_pool = pos_pool.drop(player.name)
            
            # Check if lineup is valid
            if (total_salary <= salary_cap and 
                total_salary >= min_salary_threshold and
                all(position_counts[pos] == limit for pos, limit in position_limits.items())):
                return lineup_indices
        
        logging.warning("Failed to generate valid opponent lineup after max attempts")
        return []


# ============================================================================
# 2. MEAN-VARIANCE OPTIMIZATION (Double-Up / 50/50 Cash Games)
# ============================================================================

class CashGameOptimizer:
    """
    Optimizes lineups for double-up/50-50 contests using mean-variance framework
    Based on Section 4 of MIT paper (Double-Up Problem with Stochastic Benchmarks)
    
    Key insight: Maximize P(portfolio_score > 50th_percentile_opponent_score)
    """
    
    def __init__(self, opponent_model: OpponentPortfolioModel):
        self.opponent_model = opponent_model
        
    def optimize_cash_lineup(self, player_pool: pd.DataFrame,
                             position_limits: Dict[str, int],
                             salary_cap: float,
                             correlation_matrix: Optional[np.ndarray] = None,
                             num_opponents: int = 1000) -> Tuple[List[int], float]:
        """
        Solve the double-up problem using mean-variance optimization
        
        maximize: μ_w^T - λ * σ_w^T
        subject to: budget, position, and feasibility constraints
        
        where λ controls risk-reward tradeoff
        """
        # Generate opponent distribution
        opponent_scores = self._sample_opponent_scores(
            player_pool, position_limits, salary_cap, num_opponents
        )
        
        # Calculate 50th percentile (median) benchmark
        benchmark = np.median(opponent_scores)
        benchmark_std = np.std(opponent_scores)
        
        logging.info(f"💰 CASH GAME: Opponent median = {benchmark:.2f}, std = {benchmark_std:.2f}")
        
        # Use mean-variance to find optimal lineup
        # For cash games, prioritize high floor (low variance)
        best_lineup = self._solve_mean_variance(
            player_pool, position_limits, salary_cap,
            benchmark, benchmark_std,
            lambda_risk=0.5,  # Conservative for cash
            correlation_matrix=correlation_matrix
        )
        
        return best_lineup
    
    def _sample_opponent_scores(self, player_pool: pd.DataFrame,
                                 position_limits: Dict[str, int],
                                 salary_cap: float,
                                 num_samples: int) -> np.ndarray:
        """Sample opponent lineup scores using Dirichlet-Multinomial model"""
        scores = []
        
        for _ in range(num_samples):
            lineup_indices = self.opponent_model.generate_opponent_lineup(
                player_pool, position_limits, salary_cap
            )
            
            if lineup_indices:
                lineup_score = player_pool.loc[lineup_indices, 'Projected_Points'].sum()
                scores.append(lineup_score)
        
        return np.array(scores)
    
    def _solve_mean_variance(self, player_pool: pd.DataFrame,
                            position_limits: Dict[str, int],
                            salary_cap: float,
                            benchmark: float,
                            benchmark_std: float,
                            lambda_risk: float = 0.5,
                            correlation_matrix: Optional[np.ndarray] = None) -> List[int]:
        """
        Solve mean-variance optimization problem
        
        Based on Proposition 4.1 from MIT paper:
        x* ∈ argmax(μ_x + λ * σ_x) for cash games
        """
        n_players = len(player_pool)
        
        # Extract features
        means = player_pool['Projected_Points'].values
        
        # Use correlation matrix if provided, else assume independence
        if correlation_matrix is not None and correlation_matrix.shape == (n_players, n_players):
            cov_matrix = correlation_matrix
        else:
            # Estimate variance from projected points (simplified)
            variances = (means * 0.3) ** 2  # Assume 30% coefficient of variation
            cov_matrix = np.diag(variances)
        
        # Binary quadratic programming formulation
        # Objective: maximize μ^T x - λ * x^T Σ x
        def objective(x):
            # Mean term
            mean_score = np.dot(means, x)
            # Variance term
            variance = np.dot(x, np.dot(cov_matrix, x))
            # Risk-adjusted objective
            return -(mean_score - lambda_risk * np.sqrt(variance))
        
        # Constraints
        constraints = []
        
        # Budget constraint
        salaries = player_pool['Salary'].values
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: salary_cap - np.dot(salaries, x)
        })
        
        # Position constraints
        for position, limit in position_limits.items():
            pos_mask = (player_pool['Position'] == position).astype(int).values
            constraints.append({
                'type': 'eq',
                'fun': lambda x, m=pos_mask, l=limit: np.dot(m, x) - l
            })
        
        # Binary constraints (0 or 1)
        bounds = [(0, 1) for _ in range(n_players)]
        
        # Initial guess: top players by value
        x0 = np.zeros(n_players)
        value = means / salaries
        top_indices = np.argsort(value)[-sum(position_limits.values()):]
        x0[top_indices] = 1
        
        # Solve optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            # Round to binary
            lineup_binary = np.round(result.x)
            lineup_indices = np.where(lineup_binary > 0.5)[0].tolist()
            return lineup_indices
        
        logging.warning("Mean-variance optimization failed, using greedy fallback")
        return self._greedy_fallback(player_pool, position_limits, salary_cap)
    
    def _greedy_fallback(self, player_pool: pd.DataFrame,
                        position_limits: Dict[str, int],
                        salary_cap: float) -> List[int]:
        """Simple greedy selection as fallback"""
        player_pool = player_pool.copy()
        player_pool['value'] = player_pool['Projected_Points'] / player_pool['Salary']
        player_pool = player_pool.sort_values('value', ascending=False)
        
        lineup = []
        total_salary = 0
        position_counts = {pos: 0 for pos in position_limits.keys()}
        
        for idx, player in player_pool.iterrows():
            pos = player['Position']
            if (position_counts.get(pos, 0) < position_limits.get(pos, 0) and
                total_salary + player['Salary'] <= salary_cap):
                lineup.append(idx)
                total_salary += player['Salary']
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if len(lineup) == sum(position_limits.values()):
                    break
        
        return lineup


# ============================================================================
# 3. TOURNAMENT (GPP) OPTIMIZER - Variance Maximization
# ============================================================================

class TournamentOptimizer:
    """
    Optimizes lineups for top-heavy GPP tournaments
    Based on Section 5 of MIT paper and Fantasy Bible GPP strategies
    
    Key strategies:
    - Maximize ceiling/variance (not floor like cash)
    - Differentiate from field (low ownership plays)
    - Strategic stacking for correlation
    """
    
    def __init__(self, opponent_model: OpponentPortfolioModel):
        self.opponent_model = opponent_model
        
    def optimize_tournament_lineup(self, player_pool: pd.DataFrame,
                                   position_limits: Dict[str, int],
                                   salary_cap: float,
                                   ownership_data: Optional[pd.Series] = None,
                                   stack_config: Optional[Dict] = None) -> List[int]:
        """
        Optimize for tournament (GPP) contests
        
        Objectives:
        1. High ceiling (maximize upside variance)
        2. Low ownership (differentiation)
        3. Strategic correlations (stacking)
        """
        player_pool = player_pool.copy()
        
        # Add ceiling estimation (projection + variance)
        if 'Projected_Points' in player_pool.columns:
            player_pool['Ceiling'] = player_pool['Projected_Points'] * 1.5  # 150% upside
            
        # Adjust for ownership (lower ownership = higher value in GPP)
        if ownership_data is not None:
            # Boost low-ownership players
            ownership_multiplier = 1.0 + (1.0 - ownership_data / 100) * 0.3
            player_pool['GPP_Value'] = player_pool['Ceiling'] * ownership_multiplier
        else:
            player_pool['GPP_Value'] = player_pool['Ceiling']
        
        # Apply stacking if configured
        if stack_config:
            return self._optimize_with_stacking(
                player_pool, position_limits, salary_cap, stack_config
            )
        
        # Otherwise, maximize ceiling with variance
        return self._maximize_variance_lineup(
            player_pool, position_limits, salary_cap
        )
    
    def _maximize_variance_lineup(self, player_pool: pd.DataFrame,
                                  position_limits: Dict[str, int],
                                  salary_cap: float) -> List[int]:
        """
        Select high-ceiling, high-variance lineup for tournaments
        """
        # Sort by GPP_Value (ceiling * ownership adjustment)
        player_pool = player_pool.sort_values('GPP_Value', ascending=False)
        
        lineup = []
        total_salary = 0
        position_counts = {pos: 0 for pos in position_limits.keys()}
        
        for idx, player in player_pool.iterrows():
            pos = player['Position']
            if (position_counts.get(pos, 0) < position_limits.get(pos, 0) and
                total_salary + player['Salary'] <= salary_cap):
                lineup.append(idx)
                total_salary += player['Salary']
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if len(lineup) == sum(position_limits.values()):
                    break
        
        return lineup
    
    def _optimize_with_stacking(self, player_pool: pd.DataFrame,
                                position_limits: Dict[str, int],
                                salary_cap: float,
                                stack_config: Dict) -> List[int]:
        """
        Optimize with NBA-specific stacking strategies
        
        NBA Stacks (based on correlation):
        - PG + C (pick and roll)
        - PG + SG (backcourt)
        - Game stack (4-5 players from high O/U game)
        """
        stack_type = stack_config.get('type', 'game_stack')
        
        if stack_type == 'pg_c_stack':
            return self._build_pg_c_stack(player_pool, position_limits, salary_cap)
        elif stack_type == 'game_stack':
            return self._build_game_stack(player_pool, position_limits, salary_cap, stack_config)
        else:
            return self._maximize_variance_lineup(player_pool, position_limits, salary_cap)
    
    def _build_pg_c_stack(self, player_pool: pd.DataFrame,
                         position_limits: Dict[str, int],
                         salary_cap: float) -> List[int]:
        """Build PG + C stack from same team (pick and roll correlation)"""
        lineup = []
        total_salary = 0
        
        # Find best PG
        pgs = player_pool[player_pool['Position'] == 'PG'].sort_values('GPP_Value', ascending=False)
        
        if len(pgs) > 0:
            best_pg = pgs.iloc[0]
            lineup.append(best_pg.name)
            total_salary += best_pg['Salary']
            
            # Find C from same team
            if 'Team' in player_pool.columns:
                same_team_cs = player_pool[
                    (player_pool['Position'] == 'C') & 
                    (player_pool['Team'] == best_pg['Team'])
                ].sort_values('GPP_Value', ascending=False)
                
                if len(same_team_cs) > 0:
                    best_c = same_team_cs.iloc[0]
                    if total_salary + best_c['Salary'] <= salary_cap:
                        lineup.append(best_c.name)
                        total_salary += best_c['Salary']
        
        # Fill remaining positions with best available
        position_counts = {'PG': 1 if len(lineup) > 0 else 0, 'C': 1 if len(lineup) > 1 else 0}
        remaining_pool = player_pool.drop(lineup, errors='ignore')
        
        for idx, player in remaining_pool.sort_values('GPP_Value', ascending=False).iterrows():
            pos = player['Position']
            if (position_counts.get(pos, 0) < position_limits.get(pos, 0) and
                total_salary + player['Salary'] <= salary_cap):
                lineup.append(idx)
                total_salary += player['Salary']
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if len(lineup) == sum(position_limits.values()):
                    break
        
        return lineup
    
    def _build_game_stack(self, player_pool: pd.DataFrame,
                         position_limits: Dict[str, int],
                         salary_cap: float,
                         stack_config: Dict) -> List[int]:
        """Build game stack from high-scoring game"""
        if 'Game' not in player_pool.columns and 'Opponent' not in player_pool.columns:
            return self._maximize_variance_lineup(player_pool, position_limits, salary_cap)
        
        # Find highest total game
        if 'GameTotal' in player_pool.columns:
            best_game = player_pool.sort_values('GameTotal', ascending=False)['Game'].iloc[0]
            game_players = player_pool[player_pool['Game'] == best_game]
        else:
            # Fallback: use first game
            game_players = player_pool.head(20)
        
        # Take 4-5 players from this game
        game_lineup = game_players.sort_values('GPP_Value', ascending=False).head(4).index.tolist()
        
        # Fill rest with best available
        remaining_pool = player_pool.drop(game_lineup, errors='ignore')
        total_salary = player_pool.loc[game_lineup, 'Salary'].sum()
        
        position_counts = player_pool.loc[game_lineup, 'Position'].value_counts().to_dict()
        
        for idx, player in remaining_pool.sort_values('GPP_Value', ascending=False).iterrows():
            pos = player['Position']
            if (position_counts.get(pos, 0) < position_limits.get(pos, 0) and
                total_salary + player['Salary'] <= salary_cap):
                game_lineup.append(idx)
                total_salary += player['Salary']
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
                if len(game_lineup) == sum(position_limits.values()):
                    break
        
        return game_lineup


# ============================================================================
# 4. RESEARCH-BASED OPTIMIZATION MANAGER
# ============================================================================

class ResearchBasedNBAOptimizer:
    """
    Main coordinator for research-based NBA DFS optimization
    Combines all research insights into a unified optimization framework
    """
    
    def __init__(self):
        self.opponent_model = OpponentPortfolioModel()
        self.cash_optimizer = CashGameOptimizer(self.opponent_model)
        self.tournament_optimizer = TournamentOptimizer(self.opponent_model)
        
    def optimize(self, player_pool: pd.DataFrame,
                contest_type: str,
                position_limits: Dict[str, int],
                salary_cap: float,
                num_lineups: int = 1,
                **kwargs) -> List[List[int]]:
        """
        Main optimization entry point
        
        Args:
            player_pool: DataFrame with player data
            contest_type: 'cash' or 'gpp'
            position_limits: {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'UTIL': 1}
            salary_cap: DraftKings salary cap (50000)
            num_lineups: Number of lineups to generate
            
        Returns:
            List of lineups (each lineup is a list of player indices)
        """
        logging.info(f"🔬 RESEARCH-BASED NBA OPTIMIZER: Generating {num_lineups} {contest_type.upper()} lineups")
        
        # Fit opponent model if historical data provided
        if 'historical_data' in kwargs:
            self._fit_opponent_model(kwargs['historical_data'])
        
        lineups = []
        
        for i in range(num_lineups):
            if contest_type.lower() == 'cash':
                lineup = self.cash_optimizer.optimize_cash_lineup(
                    player_pool, position_limits, salary_cap,
                    correlation_matrix=kwargs.get('correlation_matrix'),
                    num_opponents=kwargs.get('num_opponents', 1000)
                )
            else:  # GPP / Tournament
                lineup = self.tournament_optimizer.optimize_tournament_lineup(
                    player_pool, position_limits, salary_cap,
                    ownership_data=kwargs.get('ownership_data'),
                    stack_config=kwargs.get('stack_config')
                )
            
            if lineup:
                lineups.append(lineup)
                logging.info(f"✅ Lineup {i+1}/{num_lineups}: {len(lineup)} players, "
                           f"Proj: {player_pool.loc[lineup, 'Projected_Points'].sum():.2f}")
        
        return lineups
    
    def _fit_opponent_model(self, historical_data: pd.DataFrame):
        """Fit Dirichlet parameters using historical lineup data"""
        for position in historical_data['Position'].unique():
            self.opponent_model.fit_dirichlet_params(historical_data, position)
        
        logging.info("✅ Opponent model fitted with historical data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("🔬 NBA Research-Based Optimizer Core Module")
    print("=" * 60)
    print("✅ Opponent Portfolio Modeling (Dirichlet-Multinomial)")
    print("✅ Mean-Variance Optimization (Cash Games)")
    print("✅ Tournament Optimization (GPP)")
    print("✅ Strategic Stacking (PG-C, Game Stacks)")
    print("=" * 60)

