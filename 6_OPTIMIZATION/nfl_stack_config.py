"""
NFL DFS Stacking Configuration
Based on proper Daily Fantasy Sports theory for NFL

Stacking in NFL DFS is different from MLB:
- Focus on QB correlation with pass catchers
- Game environment matters (high-scoring games)
- Opponent "bring-back" creates hedge opportunities
"""

# ============================================================================
# NFL STACK TYPES - PROPER DFS THEORY
# ============================================================================

NFL_STACK_TYPES = {
    'no_stack': {
        'name': 'No Stack',
        'description': 'Independent player selection, no correlation enforcement',
        'positions_same_team': [],
        'recommended_for': ['cash_game'],
        'ownership_impact': 'neutral',
        'correlation': 0.0,
        'leverage': 1.0
    },
    
    'qb_wr': {
        'name': 'QB + WR (Primary)',
        'description': 'QB paired with his top WR target - most common stack',
        'positions_same_team': ['QB', 'WR'],
        'min_players_same_team': 2,
        'recommended_for': ['cash_game', 'gpp_tournament'],
        'ownership_impact': 'high',
        'correlation': 0.75,
        'leverage': 1.15
    },
    
    'qb_2wr': {
        'name': 'QB + 2 WR (Double)',
        'description': 'QB paired with two WRs - aggressive team stack',
        'positions_same_team': ['QB', 'WR', 'WR'],
        'min_players_same_team': 3,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_high',
        'correlation': 0.85,
        'leverage': 1.30
    },
    
    'qb_wr_te': {
        'name': 'QB + WR + TE (Triple)',
        'description': 'QB with WR and TE - contrarian stack, lower ownership',
        'positions_same_team': ['QB', 'WR', 'TE'],
        'min_players_same_team': 3,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'low',
        'correlation': 0.70,
        'leverage': 1.25
    },
    
    'qb_2wr_te': {
        'name': 'QB + 2 WR + TE (Full Passing)',
        'description': 'QB with 2 WRs and TE - ultra aggressive passing game stack',
        'positions_same_team': ['QB', 'WR', 'WR', 'TE'],
        'min_players_same_team': 4,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.80,
        'leverage': 1.45
    },
    
    'qb_wr_rb': {
        'name': 'QB + WR + RB (Run/Pass Balance)',
        'description': 'QB with WR and RB - total offense stack',
        'positions_same_team': ['QB', 'WR', 'RB'],
        'min_players_same_team': 3,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'medium',
        'correlation': 0.65,
        'leverage': 1.20
    },
    
    # GAME STACKS - Multi-team same game
    'game_qb_wr_opp_wr': {
        'name': 'Game Stack (QB + WR + Opp WR)',
        'description': 'Your QB+WR plus opponent WR - bets on high-scoring game',
        'positions_same_team': ['QB', 'WR'],
        'positions_opp_team': ['WR'],
        'min_players_same_game': 3,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'medium',
        'correlation': 0.60,
        'leverage': 1.35,
        'requires_game': True
    },
    
    'game_qb_wr_opp_rb': {
        'name': 'Bring-Back (QB + WR + Opp RB)',
        'description': 'Your QB+WR plus opponent RB - hedge strategy',
        'positions_same_team': ['QB', 'WR'],
        'positions_opp_team': ['RB'],
        'min_players_same_game': 3,
        'recommended_for': ['gpp_tournament', 'single_entry_gpp'],
        'ownership_impact': 'low',
        'correlation': 0.50,
        'leverage': 1.25,
        'requires_game': True
    },
    
    'game_qb_2wr_opp_wr': {
        'name': 'Full Game Stack (QB + 2 WR + Opp WR)',
        'description': 'Double stack your QB side plus opponent WR',
        'positions_same_team': ['QB', 'WR', 'WR'],
        'positions_opp_team': ['WR'],
        'min_players_same_game': 4,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'low',
        'correlation': 0.70,
        'leverage': 1.50,
        'requires_game': True
    },
    
    'game_qb_2wr_opp_qb': {
        'name': 'QB Showdown (QB + 2 WR + Opp QB)',
        'description': 'Both QBs from same game - shootout play',
        'positions_same_team': ['QB', 'WR', 'WR'],
        'positions_opp_team': ['QB'],
        'min_players_same_game': 4,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.55,
        'leverage': 1.40,
        'requires_game': True
    },
    
    # RUN-HEAVY STACKS
    'rb_dst_opp': {
        'name': 'RB + Opp DST',
        'description': 'RB with opposing defense - contrarian negative correlation',
        'positions_same_team': ['RB'],
        'positions_opp_team': ['DST'],
        'min_players_same_game': 2,
        'recommended_for': ['cash_game'],
        'ownership_impact': 'low',
        'correlation': -0.30,  # Negative correlation
        'leverage': 1.10,
        'requires_game': True
    },
    
    '2rb_same_team': {
        'name': '2 RB Same Team',
        'description': 'Both RBs from same team - rare but can work with RBBC',
        'positions_same_team': ['RB', 'RB'],
        'min_players_same_team': 2,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.20,
        'leverage': 1.15
    },
    
    # TEAM HEAVY STACKS - 3, 4, 5 Players from Same Team
    'qb_plus_2': {
        'name': 'QB + 2 (3 Team)',
        'description': 'QB with 2 other players from same team - moderate team exposure',
        'positions_same_team': ['QB', 'any', 'any'],
        'min_players_same_team': 3,
        'recommended_for': ['gpp_tournament', 'cash_game'],
        'ownership_impact': 'medium',
        'correlation': 0.70,
        'leverage': 1.25
    },
    
    'qb_plus_3': {
        'name': 'QB + 3 (4 Team)',
        'description': 'QB with 3 other players from same team - heavy team exposure',
        'positions_same_team': ['QB', 'any', 'any', 'any'],
        'min_players_same_team': 4,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'low',
        'correlation': 0.80,
        'leverage': 1.40
    },
    
    'qb_plus_4': {
        'name': 'QB + 4 (5 Team)',
        'description': 'QB with 4 other players from same team - ultra heavy team exposure',
        'positions_same_team': ['QB', 'any', 'any', 'any', 'any'],
        'min_players_same_team': 5,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.85,
        'leverage': 1.60
    },
    
    # SPECIFIC 4/2 STACKS - 4 total players, 2 from each position group
    'qb_2wr_2rb': {
        'name': '4/2: QB + 2 WR + 2 RB',
        'description': 'QB with 2 WRs and 2 RBs from same team - balanced offense',
        'positions_same_team': ['QB', 'WR', 'WR', 'RB', 'RB'],
        'min_players_same_team': 5,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.75,
        'leverage': 1.55
    },
    
    'qb_2wr_rb_te': {
        'name': '4/2: QB + 2 WR + RB + TE',
        'description': 'QB with 2 WRs, RB and TE from same team - full offense stack',
        'positions_same_team': ['QB', 'WR', 'WR', 'RB', 'TE'],
        'min_players_same_team': 5,
        'recommended_for': ['gpp_tournament'],
        'ownership_impact': 'very_low',
        'correlation': 0.78,
        'leverage': 1.58
    }
}

# ============================================================================
# STACK SELECTION BY CONTEST TYPE
# ============================================================================

RECOMMENDED_STACKS_BY_CONTEST = {
    'cash_game': [
        'qb_wr',           # Most common, reliable
        'no_stack',        # Sometimes best plays don't stack
        'qb_wr_rb',        # Total offense correlation
        'qb_plus_2',       # Moderate team exposure
    ],
    
    'gpp_tournament': [
        'qb_2wr',          # Aggressive but popular
        'qb_wr_te',        # Contrarian, lower ownership
        'game_qb_wr_opp_wr',  # Game stack for shootouts
        'game_qb_wr_opp_rb',  # Bring-back hedge
        'qb_2wr_te',       # Ultra aggressive
        'qb_plus_3',       # Heavy team exposure
        'qb_plus_4',       # Ultra heavy team exposure
        'qb_2wr_2rb',      # Full team stack with RBs
        'qb_2wr_rb_te',    # Full offense stack
    ],
    
    'single_entry_gpp': [
        'qb_wr_te',        # Lower ownership
        'game_qb_wr_opp_rb',  # Hedge with bring-back
        'qb_wr_rb',        # Balanced correlation
        'qb_plus_3',       # Heavy team exposure
    ],
    
    '3_max': [
        'qb_2wr',          # Need ceiling
        'game_qb_2wr_opp_wr', # Full game stack
        'qb_2wr_te',       # All pass catchers
        'qb_plus_4',       # Ultra heavy team exposure
        'qb_2wr_rb_te',    # Full offense stack
    ]
}

# ============================================================================
# GAME ENVIRONMENT STACK BOOSTS
# ============================================================================

GAME_ENVIRONMENT_MULTIPLIERS = {
    'high_total': {
        # Over/Under >= 50 points
        'boost_stacks': ['game_qb_wr_opp_wr', 'game_qb_2wr_opp_wr', 'qb_2wr', 'qb_plus_3', 'qb_plus_4'],
        'multiplier': 1.25,
        'description': 'High-scoring game expected'
    },
    
    'close_spread': {
        # Spread <= 3 points
        'boost_stacks': ['game_qb_2wr_opp_qb', 'game_qb_wr_opp_wr'],
        'multiplier': 1.15,
        'description': 'Competitive game, both teams will score'
    },
    
    'big_favorite': {
        # Spread >= 7 points
        'boost_stacks': ['qb_2wr', 'qb_wr_rb', 'qb_plus_3', 'qb_plus_4', 'qb_2wr_2rb', 'qb_2wr_rb_te'],
        'multiplier': 1.10,
        'description': 'Favorite should dominate, stack their offense'
    },
    
    'dome_game': {
        # Indoor stadium
        'boost_stacks': ['qb_2wr', 'qb_2wr_te', 'qb_plus_3', 'qb_2wr_rb_te'],
        'multiplier': 1.08,
        'description': 'Passing conditions perfect'
    },
    
    'bad_weather': {
        # Wind/rain/snow
        'boost_stacks': ['2rb_same_team', 'qb_wr_rb', 'qb_2wr_2rb'],
        'multiplier': 1.05,
        'description': 'Run-heavy game script likely'
    }
}

# ============================================================================
# OWNERSHIP-BASED STACK SELECTION
# ============================================================================

OWNERSHIP_STRATEGY = {
    'chalk': {
        # High ownership (>25%)
        'prefer_stacks': ['qb_wr', 'qb_2wr', 'qb_plus_2'],
        'description': 'Popular stacks, good for cash games'
    },
    
    'contrarian': {
        # Low ownership (<10%)
        'prefer_stacks': ['qb_wr_te', 'game_qb_wr_opp_rb', '2rb_same_team', 'qb_plus_4', 'qb_2wr_2rb', 'qb_2wr_rb_te'],
        'description': 'Unique stacks for GPP leverage'
    },
    
    'balanced': {
        # Medium ownership (10-25%)
        'prefer_stacks': ['qb_wr_rb', 'game_qb_wr_opp_wr', 'qb_plus_3'],
        'description': 'Mix of correlation and uniqueness'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_stack_type_info(stack_key: str) -> dict:
    """Get information about a specific stack type"""
    return NFL_STACK_TYPES.get(stack_key, {})

def get_recommended_stacks(contest_type: str) -> list:
    """Get list of recommended stack types for a contest"""
    return RECOMMENDED_STACKS_BY_CONTEST.get(contest_type, ['qb_wr'])

def get_stack_correlation(stack_key: str) -> float:
    """Get the correlation coefficient for a stack type"""
    stack_info = NFL_STACK_TYPES.get(stack_key, {})
    return stack_info.get('correlation', 0.0)

def requires_game_stack(stack_key: str) -> bool:
    """Check if stack requires opponent team (game stack)"""
    stack_info = NFL_STACK_TYPES.get(stack_key, {})
    return stack_info.get('requires_game', False)

def get_stack_positions(stack_key: str) -> dict:
    """Get positions required for a stack"""
    stack_info = NFL_STACK_TYPES.get(stack_key, {})
    return {
        'same_team': stack_info.get('positions_same_team', []),
        'opp_team': stack_info.get('positions_opp_team', []),
        'min_same_team': stack_info.get('min_players_same_team', 0),
        'min_same_game': stack_info.get('min_players_same_game', 0)
    }

def get_all_stack_names() -> list:
    """Get list of all available stack type names"""
    return list(NFL_STACK_TYPES.keys())

def get_stack_display_names() -> dict:
    """Get mapping of stack keys to display names"""
    return {key: info['name'] for key, info in NFL_STACK_TYPES.items()}

