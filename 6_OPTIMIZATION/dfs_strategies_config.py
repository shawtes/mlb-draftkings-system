#!/usr/bin/env python3
"""
DFS Strategy Configuration
Based on "Fantasy Football For Dummies" strategies
"""

# CONTEST TYPE CONFIGURATIONS
CONTEST_MODES = {
    'cash_game': {
        'name': 'Cash Game (50/50, H2H)',
        'objective': 'consistency',
        'player_selection': 'high_floor',
        'diversity_weight': 0.3,  # Low diversity needed
        'contrarian_enabled': False,
        'ownership_consideration': False,
        'exposure_max_default': 0.40,  # Can use popular players more
        'stack_aggressiveness': 'moderate',
        'value_focus': 'floor',
        'description': 'Safe, consistent lineups. Top 50% finish = profit.'
    },
    'gpp_tournament': {
        'name': 'GPP Tournament',
        'objective': 'upside',
        'player_selection': 'high_ceiling',
        'diversity_weight': 0.8,  # High diversity needed
        'contrarian_enabled': True,
        'ownership_consideration': True,
        'exposure_max_default': 0.20,  # Need unique lineups
        'stack_aggressiveness': 'aggressive',
        'value_focus': 'ceiling',
        'description': 'High-upside plays. Need top 1-10% to profit big.'
    },
    'single_entry_gpp': {
        'name': 'Single Entry GPP',
        'objective': 'balanced_upside',
        'player_selection': 'balanced_ceiling',
        'diversity_weight': 0.5,
        'contrarian_enabled': True,
        'ownership_consideration': True,
        'exposure_max_default': 1.0,  # Only 1 lineup
        'stack_aggressiveness': 'moderate',
        'value_focus': 'balanced',
        'description': 'One shot to win. Balance upside with safety.'
    }
}

# STACKING STRATEGIES
NFL_STACK_TYPES = {
    'primary_stack': {
        'name': 'Primary Stack (QB + 2 WRs)',
        'positions': ['QB', 'WR', 'WR'],
        'same_team': True,
        'correlation': 0.85,
        'description': 'High correlation - QB TDs help both WRs',
        'recommended_for': ['gpp_tournament', 'cash_game']
    },
    'tight_stack': {
        'name': 'Tight Stack (QB + WR + TE)',
        'positions': ['QB', 'WR', 'TE'],
        'same_team': True,
        'correlation': 0.80,
        'description': 'Less common, good contrarian play',
        'recommended_for': ['gpp_tournament']
    },
    'double_stack': {
        'name': 'Double Stack (QB + 2 WRs + RB)',
        'positions': ['QB', 'WR', 'WR', 'RB'],
        'same_team': True,
        'correlation': 0.75,
        'description': 'Ultra-aggressive team stack',
        'recommended_for': ['gpp_tournament']
    },
    'game_stack': {
        'name': 'Game Stack (QB + WR + Opp WR)',
        'positions': ['QB', 'WR', 'WR_OPP'],
        'same_game': True,
        'correlation': 0.65,
        'description': 'Bet on high-scoring game',
        'recommended_for': ['gpp_tournament', 'single_entry_gpp']
    },
    'bring_back': {
        'name': 'Bring Back (QB + 2 WRs + Opp RB)',
        'positions': ['QB', 'WR', 'WR', 'RB_OPP'],
        'same_game': True,
        'correlation': 0.55,
        'description': 'Hedge with opponent RB',
        'recommended_for': ['gpp_tournament']
    },
    'rb_stack': {
        'name': 'RB Stack (RB + DST_OPP)',
        'positions': ['RB', 'DST_OPP'],
        'opponent_defense': True,
        'correlation': -0.40,  # Negative correlation
        'description': 'RB does well when defense struggles',
        'recommended_for': ['cash_game']
    }
}

# POSITION-SPECIFIC EXPOSURE LIMITS
POSITION_EXPOSURE_LIMITS = {
    'cash_game': {
        'QB': {'min': 0.10, 'max': 0.40, 'reason': 'QBs are consistent'},
        'RB': {'min': 0.15, 'max': 0.50, 'reason': 'Workhorses get volume'},
        'WR': {'min': 0.10, 'max': 0.35, 'reason': 'More variance than RB'},
        'TE': {'min': 0.15, 'max': 0.45, 'reason': 'Fewer elite options'},
        'DST': {'min': 0.20, 'max': 0.60, 'reason': 'Punt position, streamable'}
    },
    'gpp_tournament': {
        'QB': {'min': 0.05, 'max': 0.25, 'reason': 'Diversify QB pool'},
        'RB': {'min': 0.08, 'max': 0.35, 'reason': 'Top RBs can be chalk'},
        'WR': {'min': 0.05, 'max': 0.20, 'reason': 'Most variance, spread out'},
        'TE': {'min': 0.10, 'max': 0.30, 'reason': 'Limited options, ok to repeat'},
        'DST': {'min': 0.15, 'max': 0.40, 'reason': 'Save salary, diversify'}
    }
}

# TEAM EXPOSURE LIMITS
TEAM_EXPOSURE_LIMITS = {
    'cash_game': {
        'max_per_team': 0.40,  # Max 40% of lineups from one team
        'max_stacks_per_team': 0.30  # Max 30% of lineups stack same team
    },
    'gpp_tournament': {
        'max_per_team': 0.30,  # Max 30% of lineups from one team
        'max_stacks_per_team': 0.20  # Max 20% of lineups stack same team
    }
}

# CONTRARIAN STRATEGY THRESHOLDS
CONTRARIAN_THRESHOLDS = {
    'high_owned': 0.25,      # >25% ownership = chalk
    'medium_owned': 0.15,    # 15-25% = popular
    'low_owned': 0.08,       # <8% = contrarian
    'leverage_owned': 0.05,  # <5% = extreme contrarian
}

CONTRARIAN_ADJUSTMENTS = {
    'gpp_tournament': {
        'high_owned_penalty': 0.85,      # Reduce projection by 15%
        'medium_owned_neutral': 1.0,     # No change
        'low_owned_boost': 1.15,         # Increase by 15%
        'leverage_owned_boost': 1.25     # Increase by 25%
    },
    'cash_game': {
        'high_owned_penalty': 1.0,       # Don't penalize in cash
        'medium_owned_neutral': 1.0,
        'low_owned_boost': 1.0,
        'leverage_owned_boost': 1.0
    }
}

# FADE CRITERIA
FADE_CRITERIA = {
    'injury_risk': {
        'tags': ['Questionable', 'Doubtful', 'Out', 'GTD'],
        'action': 'exclude',
        'description': 'Injury concerns'
    },
    'weather': {
        'wind_threshold': 20,      # mph
        'precip_threshold': 0.3,   # inches
        'action': 'penalize_0.75',
        'description': 'Bad weather hurts passing'
    },
    'tough_matchup': {
        'def_rank_threshold': 5,   # Top 5 defense
        'action': 'penalize_0.85',
        'description': 'Elite defense matchup'
    },
    'overpriced': {
        'salary_increase': 500,    # $500+ increase
        'projection_flat': True,
        'action': 'penalize_0.90',
        'description': 'Price went up, value went down'
    },
    'public_chalk': {
        'ownership_threshold': 0.30,
        'ceiling_percentile': 60,  # Below 60th percentile ceiling
        'action': 'fade_in_gpp',
        'description': 'High-owned without upside'
    }
}

# GAME ENVIRONMENT FACTORS
GAME_ENVIRONMENT_BOOSTS = {
    'high_total': {
        'threshold': 48.0,         # Game total over 48
        'offensive_boost': 1.15,   # 15% boost for offense
        'description': 'High-scoring game expected'
    },
    'low_total': {
        'threshold': 40.0,         # Game total under 40
        'offensive_penalty': 0.85, # 15% penalty for offense
        'dst_boost': 1.20,         # 20% boost for defense
        'description': 'Low-scoring game expected'
    },
    'large_favorite': {
        'spread_threshold': -7.0,  # Favored by 7+
        'rb_boost': 1.12,          # 12% boost for RBs (run clock)
        'qb_penalty': 0.95,        # 5% penalty for QB (less passing)
        'description': 'Blowout script favors RBs'
    },
    'large_underdog': {
        'spread_threshold': 7.0,   # Underdog by 7+
        'qb_boost': 1.15,          # 15% boost for QB (passing to catch up)
        'rb_penalty': 0.85,        # 15% penalty for RB (less running)
        'wr_boost': 1.10,          # 10% boost for WRs
        'description': 'Negative game script, pass-heavy'
    },
    'bad_weather': {
        'wind_threshold': 20,
        'precip_threshold': 0.3,
        'qb_penalty': 0.75,
        'wr_penalty': 0.80,
        'rb_boost': 1.15,
        'te_boost': 1.10,
        'description': 'Run game benefits from bad weather'
    }
}

# HEDGE LINEUP STRATEGIES
HEDGE_STRATEGIES = {
    'fade_top_qb': {
        'description': 'Use cheap QB, load up elsewhere',
        'qb_max_salary': 6000,
        'rb_wr_emphasis': 'heavy'
    },
    'rb_heavy': {
        'description': '3 RBs via FLEX, fade WR3',
        'rb_count': 3,
        'wr_count': 2
    },
    'wr_heavy': {
        'description': '4 WRs via FLEX, fade RB2',
        'rb_count': 1,
        'wr_count': 4
    },
    'contrarian_stack': {
        'description': 'Stack unpopular game',
        'ownership_max': 0.15
    },
    'stars_and_scrubs': {
        'description': 'Load up 3 studs, punt rest',
        'studs_count': 3,
        'studs_min_salary': 8000,
        'scrubs_max_salary': 4000
    },
    'balanced': {
        'description': 'No extreme positions, balanced build',
        'salary_variance_max': 2000
    }
}

# MULTI-ENTRY ALLOCATION
MULTI_ENTRY_STRATEGIES = {
    '1_entry': {
        'balanced': 1.0
    },
    '3_entries': {
        'balanced': 0.34,
        'contrarian': 0.33,
        'high_upside': 0.33
    },
    '5_entries': {
        'core': 0.40,
        'hedge': 0.40,
        'longshot': 0.20
    },
    '20_entries': {
        'core_builds': 0.50,      # 10 lineups
        'hedge_builds': 0.25,      # 5 lineups
        'longshot_builds': 0.25    # 5 lineups
    },
    '150_entries': {
        'core': 0.40,
        'variance': 0.40,
        'extreme_contrarian': 0.20
    }
}

# BANKROLL MANAGEMENT
BANKROLL_RULES = {
    'cash_game': {
        'max_risk_pct': 0.05,  # 5% of bankroll per slate
        'kelly_fraction': 0.25,
        'description': 'Conservative - high win rate'
    },
    'gpp_tournament': {
        'max_risk_pct': 0.02,  # 2% of bankroll per slate
        'kelly_fraction': 0.10,
        'description': 'Aggressive - low win rate, high payouts'
    },
    'single_entry_gpp': {
        'max_risk_pct': 0.03,
        'kelly_fraction': 0.15,
        'description': 'Moderate - one chance to shine'
    }
}

# VALUE CALCULATION SETTINGS
VALUE_SETTINGS = {
    'cash_game': {
        'projection_type': 'floor',
        'formula': 'floor / (salary / 1000)',
        'min_threshold': 2.5,  # Min 2.5x value
        'description': 'Use floor projections for safety'
    },
    'gpp_tournament': {
        'projection_type': 'ceiling',
        'formula': 'ceiling / (salary / 1000)',
        'min_threshold': 3.5,  # Min 3.5x ceiling value
        'description': 'Use ceiling projections for upside'
    },
    'balanced': {
        'projection_type': 'median',
        'formula': 'median / (salary / 1000)',
        'min_threshold': 3.0,
        'description': 'Balanced approach'
    }
}

# CORRELATION MATRIX FOR POSITION COMBINATIONS
POSITION_CORRELATION = {
    ('QB', 'WR_SAME_TEAM'): 0.42,
    ('QB', 'TE_SAME_TEAM'): 0.35,
    ('QB', 'RB_SAME_TEAM'): 0.18,
    ('QB', 'WR_OPP_TEAM'): 0.25,
    ('WR', 'WR_SAME_TEAM'): 0.15,
    ('RB', 'WR_SAME_TEAM'): 0.08,
    ('RB', 'DST_OPP_TEAM'): -0.30,
    ('DST', 'QB_OPP_TEAM'): -0.40,
    ('DST', 'RB_OPP_TEAM'): -0.25,
}

# OWNERSHIP PROJECTION SOURCES
OWNERSHIP_SOURCES = {
    'rotogrinDers': 'https://rotogrinders.com/projected-stats/nfl',
    'dfsondemand': 'https://www.dfsondemand.com/dfs-ownership-projections/',
    'fantasy_labs': 'https://www.fantasylabs.com/nfl/ownership/',
    'rts': 'https://www.rtsports.com/dfs',
    'manual': 'User-provided ownership percentages'
}

# OPTIMIZATION WEIGHTS BY CONTEST TYPE
OPTIMIZATION_WEIGHTS = {
    'cash_game': {
        'projection': 0.70,
        'value': 0.20,
        'consistency': 0.10,
        'ownership': 0.00
    },
    'gpp_tournament': {
        'projection': 0.40,
        'value': 0.20,
        'ceiling': 0.25,
        'ownership': 0.15  # Contrarian factor
    },
    'single_entry_gpp': {
        'projection': 0.50,
        'value': 0.20,
        'ceiling': 0.20,
        'ownership': 0.10
    }
}

if __name__ == "__main__":
    print("DFS Strategies Configuration Loaded")
    print(f"\nContest Modes: {len(CONTEST_MODES)}")
    print(f"Stack Types: {len(NFL_STACK_TYPES)}")
    print(f"Game Environment Factors: {len(GAME_ENVIRONMENT_BOOSTS)}")
    print(f"Hedge Strategies: {len(HEDGE_STRATEGIES)}")

