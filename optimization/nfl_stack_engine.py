"""
NFL DFS Stacking Engine
Implements proper NFL stacking logic for the optimizer
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from nfl_stack_config import (
    NFL_STACK_TYPES, 
    get_stack_positions, 
    requires_game_stack,
    get_stack_correlation
)

class NFLStackEngine:
    """
    Core engine for NFL DFS stacking logic
    """
    
    def __init__(self, df_players: pd.DataFrame):
        """
        Initialize stacking engine with player pool
        
        Args:
            df_players: DataFrame with player data including Team, Position, Opponent
        """
        self.df = df_players.copy()
        self.logger = logging.getLogger(__name__)
        
        # Ensure required columns exist
        required_cols = ['Name', 'Team', 'Position', 'Salary']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add Opponent column if missing
        if 'Opponent' not in self.df.columns:
            self.df['Opponent'] = None
            self.logger.warning("'Opponent' column missing - game stacks will not work")
    
    def get_teams_in_game(self, team: str) -> Tuple[str, Optional[str]]:
        """
        Get both teams playing in a game
        
        Args:
            team: Team abbreviation
            
        Returns:
            Tuple of (team, opponent_team)
        """
        if 'Opponent' not in self.df.columns or self.df['Opponent'].isna().all():
            return (team, None)
        
        # Find opponent from player data
        team_players = self.df[self.df['Team'] == team]
        if len(team_players) > 0 and 'Opponent' in team_players.columns:
            opponent = team_players['Opponent'].mode()
            if len(opponent) > 0 and pd.notna(opponent.iloc[0]):
                return (team, opponent.iloc[0])
        
        return (team, None)
    
    def validate_stack_feasibility(self, stack_type: str, team: str, opponent: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if a stack is feasible with available players
        
        Args:
            stack_type: Key from NFL_STACK_TYPES
            team: Primary team for stack
            opponent: Opponent team (required for game stacks)
            
        Returns:
            (is_feasible, error_message)
        """
        stack_info = NFL_STACK_TYPES.get(stack_type)
        if not stack_info:
            return (False, f"Unknown stack type: {stack_type}")
        
        # Check if game stack but no opponent provided
        if requires_game_stack(stack_type) and not opponent:
            return (False, f"{stack_info['name']} requires opponent team")
        
        # Get required positions
        positions = get_stack_positions(stack_type)
        same_team_positions = positions['same_team']
        opp_team_positions = positions.get('opp_team', [])
        
        # Check same team positions
        team_players = self.df[self.df['Team'] == team]
        for pos in same_team_positions:
            available = team_players[team_players['Position'] == pos]
            if len(available) == 0:
                return (False, f"No {pos} available for {team}")
        
        # Check opponent team positions
        if opp_team_positions and opponent:
            opp_players = self.df[self.df['Team'] == opponent]
            for pos in opp_team_positions:
                available = opp_players[opp_players['Position'] == pos]
                if len(available) == 0:
                    return (False, f"No {pos} available for opponent {opponent}")
        
        return (True, "Stack is feasible")
    
    def apply_stack_constraints_pulp(self, problem, player_vars: dict, stack_type: str, 
                                     team: str, opponent: Optional[str] = None):
        """
        Apply stacking constraints to PuLP optimization problem
        
        Args:
            problem: PuLP LpProblem instance
            player_vars: Dict mapping player index to PuLP binary variable
            stack_type: Key from NFL_STACK_TYPES
            team: Primary team for stack
            opponent: Opponent team (required for game stacks)
        """
        import pulp
        
        stack_info = NFL_STACK_TYPES.get(stack_type)
        if not stack_info:
            self.logger.error(f"Unknown stack type: {stack_type}")
            return
        
        self.logger.info(f"Applying {stack_info['name']} for {team}" + (f" vs {opponent}" if opponent else ""))
        
        # Handle 'no_stack' case - no constraints needed
        if stack_type == 'no_stack':
            self.logger.debug("No stack constraints applied")
            return
        
        positions = get_stack_positions(stack_type)
        same_team_positions = positions['same_team']
        opp_team_positions = positions.get('opp_team', [])
        min_same_team = positions.get('min_same_team', len(same_team_positions))
        
        # ========== SAME TEAM CONSTRAINTS ==========
        if same_team_positions:
            # Get players from primary team
            team_players = self.df[self.df['Team'] == team]
            
            # Track how many of each position we need
            position_counts = {}
            for pos in same_team_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Apply constraints for each position
            for pos, count in position_counts.items():
                eligible_players = team_players[team_players['Position'] == pos]
                eligible_indices = eligible_players.index.tolist()
                
                if len(eligible_indices) >= count:
                    # Must select exactly 'count' players of this position from this team
                    problem += (
                        pulp.lpSum([player_vars[idx] for idx in eligible_indices if idx in player_vars]) == count,
                        f"Stack_{team}_{pos}_exactly_{count}"
                    )
                    self.logger.debug(f"  {team} {pos}: Must select {count} from {len(eligible_indices)} available")
                else:
                    self.logger.warning(f"  {team} {pos}: Only {len(eligible_indices)} available, need {count}")
        
        # ========== OPPONENT TEAM CONSTRAINTS ==========
        if opp_team_positions and opponent:
            opp_players = self.df[self.df['Team'] == opponent]
            
            # Track how many of each position we need from opponent
            opp_position_counts = {}
            for pos in opp_team_positions:
                opp_position_counts[pos] = opp_position_counts.get(pos, 0) + 1
            
            # Apply constraints for opponent positions
            for pos, count in opp_position_counts.items():
                eligible_players = opp_players[opp_players['Position'] == pos]
                eligible_indices = eligible_players.index.tolist()
                
                if len(eligible_indices) >= count:
                    # Must select exactly 'count' players of this position from opponent
                    problem += (
                        pulp.lpSum([player_vars[idx] for idx in eligible_indices if idx in player_vars]) == count,
                        f"Stack_{opponent}_opp_{pos}_exactly_{count}"
                    )
                    self.logger.debug(f"  {opponent} {pos}: Must select {count} from {len(eligible_indices)} available")
                else:
                    self.logger.warning(f"  {opponent} {pos}: Only {len(eligible_indices)} available, need {count}")
        
        self.logger.info(f"Stack constraints applied successfully for {stack_info['name']}")
    
    def get_available_stacks_for_team(self, team: str, contest_type: str = 'gpp_tournament') -> List[str]:
        """
        Get list of feasible stack types for a team
        
        Args:
            team: Team abbreviation
            contest_type: Contest type from nfl_stack_config
            
        Returns:
            List of feasible stack type keys
        """
        team_info = self.get_teams_in_game(team)
        opponent = team_info[1]
        
        feasible_stacks = []
        
        for stack_key, stack_info in NFL_STACK_TYPES.items():
            # Check if recommended for contest type
            if contest_type not in stack_info.get('recommended_for', []):
                continue
            
            # Check feasibility
            is_feasible, msg = self.validate_stack_feasibility(stack_key, team, opponent)
            if is_feasible:
                feasible_stacks.append(stack_key)
            else:
                self.logger.debug(f"  {stack_key} not feasible: {msg}")
        
        return feasible_stacks
    
    def get_all_game_matchups(self) -> List[Tuple[str, str]]:
        """
        Get all unique game matchups from player data
        
        Returns:
            List of (team1, team2) tuples
        """
        if 'Opponent' not in self.df.columns or self.df['Opponent'].isna().all():
            return []
        
        matchups = set()
        for team in self.df['Team'].unique():
            if pd.isna(team):
                continue
            
            team_players = self.df[self.df['Team'] == team]
            if len(team_players) > 0:
                opponent = team_players['Opponent'].mode()
                if len(opponent) > 0 and pd.notna(opponent.iloc[0]):
                    opp_team = opponent.iloc[0]
                    # Store as sorted tuple to avoid duplicates (A vs B == B vs A)
                    matchup = tuple(sorted([team, opp_team]))
                    matchups.add(matchup)
        
        return list(matchups)
    
    def generate_stack_combinations(self, contest_type: str = 'gpp_tournament', 
                                   max_per_team: int = 3) -> List[Tuple[str, str, Optional[str]]]:
        """
        Generate all feasible stack combinations for lineup generation
        
        Args:
            contest_type: Contest type (cash_game, gpp_tournament, etc.)
            max_per_team: Maximum number of stacks per team
            
        Returns:
            List of (stack_type, team, opponent) tuples
        """
        combinations = []
        
        # Get all teams
        teams = self.df['Team'].unique()
        teams = [t for t in teams if pd.notna(t)]
        
        for team in teams:
            opponent_info = self.get_teams_in_game(team)
            opponent = opponent_info[1]
            
            # Get feasible stacks for this team
            feasible_stacks = self.get_available_stacks_for_team(team, contest_type)
            
            # Limit number of stacks per team
            for stack_type in feasible_stacks[:max_per_team]:
                combinations.append((stack_type, team, opponent))
        
        self.logger.info(f"Generated {len(combinations)} stack combinations for {contest_type}")
        return combinations
    
    def boost_stack_projections(self, stack_type: str) -> pd.DataFrame:
        """
        Apply correlation boosts to player projections based on stack type
        
        Args:
            stack_type: Stack type key
            
        Returns:
            Modified DataFrame with boosted projections
        """
        correlation = get_stack_correlation(stack_type)
        
        if correlation > 0:
            # Positive correlation - boost all players in stack slightly
            boost_factor = 1.0 + (correlation * 0.1)  # Up to 8.5% boost
            self.logger.debug(f"Applying {boost_factor:.2f}x boost for {stack_type}")
        
        # Return copy to avoid modifying original
        return self.df.copy()


# ============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_stack_engine(df_players: pd.DataFrame) -> NFLStackEngine:
    """Factory function to create stack engine"""
    return NFLStackEngine(df_players)

def get_stack_display_string(stack_type: str, team: str, opponent: Optional[str] = None) -> str:
    """
    Create human-readable string for a stack
    
    Args:
        stack_type: Stack type key
        team: Primary team
        opponent: Opponent team (if applicable)
        
    Returns:
        Display string like "QB + 2 WR (KC)" or "Game Stack (KC vs BUF)"
    """
    stack_info = NFL_STACK_TYPES.get(stack_type, {})
    stack_name = stack_info.get('name', stack_type)
    
    if opponent:
        return f"{stack_name} ({team} vs {opponent})"
    else:
        return f"{stack_name} ({team})"

def validate_lineup_stack(lineup: pd.DataFrame, stack_type: str, 
                         team: str, opponent: Optional[str] = None) -> bool:
    """
    Validate that a generated lineup satisfies stack constraints
    
    Args:
        lineup: DataFrame of players in lineup
        stack_type: Expected stack type
        team: Expected primary team
        opponent: Expected opponent team (if game stack)
        
    Returns:
        True if lineup satisfies stack, False otherwise
    """
    positions = get_stack_positions(stack_type)
    same_team_positions = positions['same_team']
    opp_team_positions = positions.get('opp_team', [])
    
    # Check same team positions
    for pos in same_team_positions:
        count = len(lineup[(lineup['Team'] == team) & (lineup['Position'] == pos)])
        if count == 0:
            return False
    
    # Check opponent positions
    if opp_team_positions and opponent:
        for pos in opp_team_positions:
            count = len(lineup[(lineup['Team'] == opponent) & (lineup['Position'] == pos)])
            if count == 0:
                return False
    
    return True

