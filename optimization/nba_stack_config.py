"""
NBA DFS Stacking Configuration
Based on proper Daily Fantasy Sports theory for NBA

Stacking in NBA DFS:
- Focus on team correlation (same team players)
- Game environment matters (high-scoring games)
- Position-based correlations (PG-C, PG-Wing, etc.)
"""

# ============================================================================
# NBA STACK TYPES - PROPER DFS THEORY
# ============================================================================

NBA_STACK_TYPES = {
    'no_stack': {
        'name': 'No Stack',
        'description': 'Independent player selection, no correlation enforcement',
        'positions_same_team': [],
        'recommended_for': ['cash_game'],
        'ownership_impact': 'neutral',
        'correlation': 0.0,
        'leverage': 1.0
    },
    
    'pg_c_stack': {
        'name': 'PG + C Stack',
        'description': 'Point Guard + Center correlation (highest correlation - PnR)',
        'positions_same_team': ['PG', 'C'],
        'min_players_same_team': 2,
        'recommended_for': ['cash_game', 'gpp_tournament'],
        'ownership_impact': 'high',
        'correlation': 0.48,
        'leverage': 1.15
    },
    
    'pg_wing_stack': {
        'name': 'PG + Wing Stack',
        'description': 'Point Guard + Wing (SF/SG) - assist correlation',
        'positions_same_team': ['PG', 'SG', 'SF'],
        'min_players_same_team': 2,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'medium',
        'correlation': 0.42,
        'leverage': 1.20
    },
    
    'game_stack': {
        'name': 'Game Stack',
        'description': '4+ players from one high-scoring game',
        'positions_same_team': [],
        'min_players_same_game': 4,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'medium',
        'correlation': 0.60,
        'leverage': 1.35,
        'requires_game': True
    },
    
    'balanced': {
        'name': 'Balanced',
        'description': 'No specific stacking, balanced approach',
        'positions_same_team': [],
        'recommended_for': ['cash_game'],
        'ownership_impact': 'neutral',
        'correlation': 0.0,
        'leverage': 1.0
    },
}


def get_stack_positions(stack_key: str) -> dict:
    """
    Get position requirements for a given stack type.
    
    Args:
        stack_key: Stack type identifier (e.g., 'pg_c_stack', 'game_stack')
        
    Returns:
        Dictionary with 'same_team' and optionally 'opp_team' position lists
    """
    stack_info = NBA_STACK_TYPES.get(stack_key, {})
    
    result = {
        'same_team': stack_info.get('positions_same_team', []),
        'opp_team': stack_info.get('positions_opp_team', []),
        'min_players_same_team': stack_info.get('min_players_same_team', 0),
        'min_players_same_game': stack_info.get('min_players_same_game', 0),
        'requires_game': stack_info.get('requires_game', False)
    }
    
    return result





