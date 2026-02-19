#!/usr/bin/env python3
"""
DFS Strategy Helper Functions
Implements advanced DFS strategies from Fantasy Football For Dummies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dfs_strategies_config import *


class ContestarianEngine:
    """Implements contrarian player selection for GPP tournaments"""
    
    def __init__(self, contest_mode: str = 'gpp_tournament'):
        self.contest_mode = contest_mode
        self.thresholds = CONTRARIAN_THRESHOLDS
        self.adjustments = CONTRARIAN_ADJUSTMENTS.get(contest_mode, {})
    
    def apply_ownership_adjustments(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust player projections based on ownership
        
        Args:
            df_players: DataFrame with 'ownership' column (0-1 scale)
        
        Returns:
            DataFrame with adjusted projections
        """
        if 'ownership' not in df_players.columns:
            logging.warning("No ownership data available, skipping contrarian adjustments")
            return df_players
        
        df = df_players.copy()
        
        # Classify ownership levels
        df['ownership_tier'] = pd.cut(
            df['ownership'],
            bins=[0, self.thresholds['leverage_owned'], self.thresholds['low_owned'], 
                  self.thresholds['medium_owned'], self.thresholds['high_owned'], 1.0],
            labels=['leverage', 'low', 'medium', 'high', 'chalk']
        )
        
        # Apply adjustments based on contest mode
        if self.contest_mode == 'gpp_tournament':
            # Penalize chalk (high-owned players)
            high_owned = df['ownership'] > self.thresholds['high_owned']
            df.loc[high_owned, 'FantasyPoints'] *= self.adjustments.get('high_owned_penalty', 1.0)
            
            # Boost low-owned players
            low_owned = df['ownership'] < self.thresholds['low_owned']
            df.loc[low_owned, 'FantasyPoints'] *= self.adjustments.get('low_owned_boost', 1.0)
            
            # Extra boost for extreme contrarian
            leverage = df['ownership'] < self.thresholds['leverage_owned']
            df.loc[leverage, 'FantasyPoints'] *= self.adjustments.get('leverage_owned_boost', 1.0)
            
            logging.info(f"Applied contrarian adjustments: {high_owned.sum()} penalized, {low_owned.sum()} boosted")
        
        return df
    
    def identify_contrarian_plays(self, df_players: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Find the best contrarian plays"""
        if 'ownership' not in df_players.columns:
            return df_players.nlargest(top_n, 'Value')
        
        # Calculate contrarian score: high ceiling, low ownership
        df = df_players.copy()
        df['contrarian_score'] = (df.get('ceiling', df['FantasyPoints']) / df['ownership'].clip(lower=0.01)) * df['Value']
        
        return df.nlargest(top_n, 'contrarian_score')[['Name', 'Position', 'Salary', 'FantasyPoints', 'ownership', 'contrarian_score']]


class GameEnvironmentAnalyzer:
    """Analyzes game environment factors and adjusts projections"""
    
    def __init__(self):
        self.boosts = GAME_ENVIRONMENT_BOOSTS
    
    def apply_game_environment_boosts(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Apply boosts/penalties based on game environment
        
        Expected columns: game_total, spread, wind, precip
        """
        df = df_players.copy()
        
        # High-scoring game boost
        if 'game_total' in df.columns:
            high_total = df['game_total'] > self.boosts['high_total']['threshold']
            offensive = df['Position'] != 'DST'
            df.loc[high_total & offensive, 'FantasyPoints'] *= self.boosts['high_total']['offensive_boost']
            
            # Low-scoring game
            low_total = df['game_total'] < self.boosts['low_total']['threshold']
            df.loc[low_total & offensive, 'FantasyPoints'] *= self.boosts['low_total']['offensive_penalty']
            df.loc[low_total & (df['Position'] == 'DST'), 'FantasyPoints'] *= self.boosts['low_total']['dst_boost']
        
        # Game script adjustments based on spread
        if 'spread' in df.columns:
            # Large favorites (run more)
            big_fav = df['spread'] < self.boosts['large_favorite']['spread_threshold']
            df.loc[big_fav & (df['Position'] == 'RB'), 'FantasyPoints'] *= self.boosts['large_favorite']['rb_boost']
            df.loc[big_fav & (df['Position'] == 'QB'), 'FantasyPoints'] *= self.boosts['large_favorite']['qb_penalty']
            
            # Large underdogs (pass more)
            big_dog = df['spread'] > self.boosts['large_underdog']['spread_threshold']
            df.loc[big_dog & (df['Position'] == 'QB'), 'FantasyPoints'] *= self.boosts['large_underdog']['qb_boost']
            df.loc[big_dog & (df['Position'] == 'WR'), 'FantasyPoints'] *= self.boosts['large_underdog']['wr_boost']
            df.loc[big_dog & (df['Position'] == 'RB'), 'FantasyPoints'] *= self.boosts['large_underdog']['rb_penalty']
        
        # Weather adjustments
        if 'wind' in df.columns and 'precip' in df.columns:
            bad_weather = ((df['wind'] > self.boosts['bad_weather']['wind_threshold']) | 
                          (df['precip'] > self.boosts['bad_weather']['precip_threshold']))
            
            df.loc[bad_weather & (df['Position'] == 'QB'), 'FantasyPoints'] *= self.boosts['bad_weather']['qb_penalty']
            df.loc[bad_weather & (df['Position'] == 'WR'), 'FantasyPoints'] *= self.boosts['bad_weather']['wr_penalty']
            df.loc[bad_weather & (df['Position'] == 'RB'), 'FantasyPoints'] *= self.boosts['bad_weather']['rb_boost']
            df.loc[bad_weather & (df['Position'] == 'TE'), 'FantasyPoints'] *= self.boosts['bad_weather']['te_boost']
        
        return df


class StackingOptimizer:
    """Advanced stacking strategies for NFL DFS"""
    
    def __init__(self, stack_type: str = 'primary_stack'):
        self.stack_type = stack_type
        self.stack_config = NFL_STACK_TYPES.get(stack_type, {})
    
    def build_primary_stack(self, df_players: pd.DataFrame, team: str) -> Dict[str, List[str]]:
        """Build QB + 2 WRs stack"""
        stack = {}
        team_players = df_players[df_players['Team'] == team]
        
        qb = team_players[team_players['Position'] == 'QB'].nlargest(1, 'FantasyPoints')
        wrs = team_players[team_players['Position'] == 'WR'].nlargest(2, 'FantasyPoints')
        
        if not qb.empty and len(wrs) >= 2:
            stack['QB'] = qb['Name'].tolist()
            stack['WR'] = wrs['Name'].tolist()
            return stack
        return None
    
    def build_game_stack(self, df_players: pd.DataFrame, team1: str, team2: str) -> Dict[str, List[str]]:
        """Build QB + WR + Opponent WR stack"""
        stack = {}
        
        team1_players = df_players[df_players['Team'] == team1]
        team2_players = df_players[df_players['Team'] == team2]
        
        qb = team1_players[team1_players['Position'] == 'QB'].nlargest(1, 'FantasyPoints')
        wr1 = team1_players[team1_players['Position'] == 'WR'].nlargest(1, 'FantasyPoints')
        wr2_opp = team2_players[team2_players['Position'] == 'WR'].nlargest(1, 'FantasyPoints')
        
        if not qb.empty and not wr1.empty and not wr2_opp.empty:
            stack['QB'] = qb['Name'].tolist()
            stack['WR'] = wr1['Name'].tolist() + wr2_opp['Name'].tolist()
            return stack
        return None
    
    def calculate_stack_correlation(self, positions: List[str]) -> float:
        """Calculate expected correlation for a stack"""
        correlation = 1.0
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                key = (pos1, f"{pos2}_SAME_TEAM")
                correlation *= POSITION_CORRELATION.get(key, 0.1)
        return correlation


class ValueAnalyzer:
    """Enhanced value calculations with ceiling/floor"""
    
    def __init__(self, contest_mode: str = 'gpp_tournament'):
        self.contest_mode = contest_mode
        self.settings = VALUE_SETTINGS.get(contest_mode, VALUE_SETTINGS['balanced'])
    
    def calculate_enhanced_value(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate value based on contest type
        
        Expected columns: FantasyPoints, Salary, ceiling (optional), floor (optional)
        """
        df = df_players.copy()
        
        # Generate ceiling/floor if not present
        if 'ceiling' not in df.columns:
            df['ceiling'] = df['FantasyPoints'] * 1.4  # 40% upside
        
        if 'floor' not in df.columns:
            df['floor'] = df['FantasyPoints'] * 0.7  # 30% downside
        
        # Calculate different value types
        df['value_projection'] = df['FantasyPoints'] / (df['Salary'] / 1000)
        df['value_ceiling'] = df['ceiling'] / (df['Salary'] / 1000)
        df['value_floor'] = df['floor'] / (df['Salary'] / 1000)
        
        # Set primary value based on contest mode
        if self.settings['projection_type'] == 'ceiling':
            df['Value'] = df['value_ceiling']
        elif self.settings['projection_type'] == 'floor':
            df['Value'] = df['value_floor']
        else:
            df['Value'] = df['value_projection']
        
        # Add value rating (handle small datasets)
        try:
            df['value_rating'] = pd.qcut(df['Value'], q=5, labels=['Poor', 'Fair', 'Good', 'Great', 'Elite'], duplicates='drop')
        except ValueError:
            # If not enough unique values for 5 bins, use simpler categorization
            df['value_rating'] = pd.cut(df['Value'], bins=[0, 2, 2.5, 3, 3.5, 100], 
                                       labels=['Poor', 'Fair', 'Good', 'Great', 'Elite'])
        
        return df


class BankrollManager:
    """Manages bankroll and calculates safe entry amounts"""
    
    def __init__(self, total_bankroll: float):
        self.bankroll = total_bankroll
    
    def calculate_max_entries(self, contest_type: str, entry_fee: float) -> Tuple[int, str]:
        """
        Calculate maximum safe entries using Kelly Criterion principles
        
        Returns:
            (max_entries, recommendation_message)
        """
        rules = BANKROLL_RULES.get(contest_type, BANKROLL_RULES['gpp_tournament'])
        
        max_risk = self.bankroll * rules['max_risk_pct']
        max_entries = int(max_risk / entry_fee)
        
        risk_pct = (max_entries * entry_fee) / self.bankroll * 100
        
        message = f"""
        💰 Bankroll Analysis ({contest_type})
        ═══════════════════════════════════════
        Total Bankroll: ${self.bankroll:,.2f}
        Entry Fee: ${entry_fee:.2f}
        Max Safe Entries: {max_entries}
        Total Risk: ${max_entries * entry_fee:,.2f} ({risk_pct:.1f}%)
        
        Kelly Fraction: {rules['kelly_fraction']}
        Max Risk %: {rules['max_risk_pct']*100:.1f}%
        
        {rules['description']}
        """
        
        return max_entries, message
    
    def calculate_roi_needed(self, contest_type: str, entry_fee: float, num_entries: int) -> str:
        """Calculate ROI needed to break even"""
        total_investment = entry_fee * num_entries
        
        # Typical cash rates
        cash_rates = {
            'cash_game': 0.50,      # 50% cash
            'gpp_tournament': 0.20,  # 20% cash
            'single_entry_gpp': 0.20
        }
        
        cash_rate = cash_rates.get(contest_type, 0.20)
        needed_roi = 1.0 / cash_rate
        
        message = f"""
        📊 ROI Analysis
        ═══════════════════════════════════════
        Total Investment: ${total_investment:,.2f}
        Entries: {num_entries} × ${entry_fee:.2f}
        
        Cash Rate: {cash_rate*100:.0f}%
        ROI Needed to Break Even: {needed_roi:.1f}x
        
        To profit, you need:
        - Avg finish: Top {cash_rate*100:.0f}%
        - Avg return: ${total_investment * needed_roi:,.2f}+
        """
        
        return message


class FadeManager:
    """Manages player fades based on various criteria"""
    
    def __init__(self):
        self.fade_list = []
        self.fade_reasons = {}
    
    def add_fade(self, player_name: str, reason: str):
        """Add a player to the fade list"""
        self.fade_list.append(player_name)
        self.fade_reasons[player_name] = reason
    
    def apply_fades(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """Remove or heavily penalize faded players"""
        df = df_players.copy()
        
        for player in self.fade_list:
            if player in df['Name'].values:
                reason = self.fade_reasons.get(player, 'Manual fade')
                logging.info(f"Fading {player}: {reason}")
                df = df[df['Name'] != player]
        
        return df
    
    def apply_auto_fades(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """Automatically fade players based on criteria"""
        df = df_players.copy()
        
        # Injury risk
        if 'injury_status' in df.columns:
            risky = df['injury_status'].isin(FADE_CRITERIA['injury_risk']['tags'])
            logging.info(f"Auto-fading {risky.sum()} players due to injury risk")
            df = df[~risky]
        
        # Weather
        if 'wind' in df.columns:
            bad_wind = df['wind'] > FADE_CRITERIA['weather']['wind_threshold']
            passing_positions = df['Position'].isin(['QB', 'WR'])
            df.loc[bad_wind & passing_positions, 'FantasyPoints'] *= 0.75
        
        # Public chalk without upside (GPP only)
        if 'ownership' in df.columns and 'ceiling' in df.columns:
            ceiling_pct = df['ceiling'] / df['FantasyPoints']
            chalk = df['ownership'] > FADE_CRITERIA['public_chalk']['ownership_threshold']
            low_ceiling = ceiling_pct < 1.6  # Less than 60% upside
            
            auto_fade = chalk & low_ceiling
            logging.info(f"Auto-fading {auto_fade.sum()} chalk players without upside")
            df = df[~auto_fade]
        
        return df


class HedgeLineupBuilder:
    """Builds hedge lineups with different construction philosophies"""
    
    def __init__(self, base_lineups: List[pd.DataFrame]):
        self.base_lineups = base_lineups
    
    def generate_hedge_lineups(self, df_players: pd.DataFrame, num_hedge: int) -> List[Dict]:
        """Generate hedge lineups with different strategies"""
        hedge_lineups = []
        strategies = list(HEDGE_STRATEGIES.keys())
        
        for i in range(num_hedge):
            strategy = strategies[i % len(strategies)]
            config = HEDGE_STRATEGIES[strategy]
            
            hedge = {
                'strategy': strategy,
                'description': config['description'],
                'constraints': config
            }
            hedge_lineups.append(hedge)
        
        return hedge_lineups


def generate_ownership_projections_manual(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple ownership projections based on salary and value
    (To be replaced with scraping or manual input)
    """
    df = df_players.copy()
    
    # Simple model: high salary + high value = high ownership
    df['salary_percentile'] = df['Salary'].rank(pct=True)
    df['value_percentile'] = df.get('Value', df['FantasyPoints'] / (df['Salary'] / 1000)).rank(pct=True)
    
    # Weighted average
    df['ownership'] = (df['salary_percentile'] * 0.6 + df['value_percentile'] * 0.4)
    
    # Normalize to 0-100 scale
    df['ownership'] = df['ownership'] * 0.30  # Max 30% ownership
    
    return df


if __name__ == "__main__":
    print("DFS Strategy Helpers Loaded")
    print("\nAvailable Classes:")
    print("  - ContestarianEngine")
    print("  - GameEnvironmentAnalyzer")
    print("  - StackingOptimizer")
    print("  - ValueAnalyzer")
    print("  - BankrollManager")
    print("  - FadeManager")
    print("  - HedgeLineupBuilder")

