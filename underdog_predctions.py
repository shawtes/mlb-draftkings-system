import pandas as pd
import numpy as np
import joblib
import concurrent.futures
import time
import warnings
import logging
import os
import json
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, QuantileRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

# Suppress specific pandas warnings related to runtime operations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
warnings.filterwarnings('ignore', message='invalid value encountered in subtract')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# Also suppress XGBoost device warnings for cleaner output
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')
# Suppress sklearn version warnings for pickled objects
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

class UnderdogProbabilityPredictor:
    """
    Advanced Underdog Fantasy prediction system with Power Play and Insurance optimization.
    Handles individual stat predictions (hits, runs, RBIs, strikeouts) and parlay construction.
    """
    
    def __init__(self):
        self.power_play_multipliers = {
            '3x': 3.0,
            '6x': 6.0, 
            '10x': 10.0,
            '20x': 20.0
        }
        self.insurance_cost_factor = 0.15  # 15% cost for insurance
        self.stat_models = {}
        self.probability_thresholds = {
            'conservative': 0.65,
            'moderate': 0.55,
            'aggressive': 0.45
        }
    
    def predict_individual_stats(self, player_data):
        """Predict individual statistics for Underdog Fantasy props"""
        predictions = {}
        
        # Predict hits (based on batting average and recent form)
        if 'batting_avg' in player_data and 'at_bats' in player_data:
            base_hit_prob = player_data['batting_avg']
            recent_form = player_data.get('recent_avg', base_hit_prob)
            hits_prob = (base_hit_prob * 0.7 + recent_form * 0.3)
            predictions['hits'] = {
                'probability': min(hits_prob, 0.85),
                'expected_value': hits_prob * player_data.get('at_bats', 4),
                'confidence': self._calculate_confidence(hits_prob, player_data.get('games_played', 50))
            }
        
        # Predict runs (based on OBP, team scoring, lineup position)
        if 'obp' in player_data:
            base_run_prob = player_data['obp'] * 0.3  # Base probability
            lineup_boost = player_data.get('lineup_position_factor', 1.0)
            team_scoring = player_data.get('team_runs_per_game', 4.5) / 4.5
            runs_prob = base_run_prob * lineup_boost * team_scoring
            predictions['runs'] = {
                'probability': min(runs_prob, 0.75),
                'expected_value': runs_prob,
                'confidence': self._calculate_confidence(runs_prob, player_data.get('games_played', 50))
            }
        
        # Predict RBIs (based on RBI opportunities and clutch performance)
        if 'rbi_rate' in player_data:
            base_rbi_prob = player_data['rbi_rate']
            risp_performance = player_data.get('risp_avg', player_data.get('batting_avg', 0.250))
            team_baserunners = player_data.get('team_obp', 0.320) * 1.5
            rbi_prob = base_rbi_prob * (risp_performance / 0.250) * min(team_baserunners, 1.8)
            predictions['rbis'] = {
                'probability': min(rbi_prob, 0.70),
                'expected_value': rbi_prob,
                'confidence': self._calculate_confidence(rbi_prob, player_data.get('games_played', 50))
            }
        
        # Predict strikeouts (for pitchers)
        if 'strikeout_rate' in player_data:
            base_k_rate = player_data['strikeout_rate']
            opponent_k_rate = player_data.get('opponent_contact_rate', 0.80)
            matchup_factor = (1 - opponent_k_rate) + 0.2  # Opposing team's strikeout tendency
            k_prob = base_k_rate * matchup_factor
            predictions['strikeouts'] = {
                'probability': min(k_prob, 0.90),
                'expected_value': k_prob * player_data.get('expected_innings', 6.0),
                'confidence': self._calculate_confidence(k_prob, player_data.get('games_started', 20))
            }
        
        return predictions
    
    def _calculate_confidence(self, probability, sample_size):
        """Calculate confidence level based on probability and sample size"""
        base_confidence = min(sample_size / 100, 1.0)  # More games = higher confidence
        prob_confidence = 1 - abs(0.5 - probability) * 2  # Extreme probabilities get lower confidence
        return (base_confidence + prob_confidence) / 2
    
    def optimize_power_play(self, player_predictions, risk_tolerance='moderate'):
        """Optimize Power Play selections based on risk tolerance"""
        power_plays = []
        threshold = self.probability_thresholds[risk_tolerance]
        
        # Debug: Print threshold and available predictions
        print(f"Debug: Using threshold {threshold} for {risk_tolerance} risk tolerance")
        
        for player, stats in player_predictions.items():
            for stat, prediction in stats.items():
                prob = prediction['probability']
                conf = prediction['confidence']
                
                # More lenient filtering for demo purposes
                if prob >= threshold * 0.7 and conf > 0.5:  # Reduced thresholds
                    # Calculate expected value for each multiplier
                    for multiplier_name, multiplier in self.power_play_multipliers.items():
                        expected_return = prob * multiplier
                        kelly_fraction = self._kelly_criterion(prob, multiplier)
                        
                        # Only include plays with positive expected value
                        if expected_return > 1.0:
                            power_plays.append({
                                'player': player,
                                'stat': stat,
                                'multiplier': multiplier_name,
                                'probability': prob,
                                'expected_return': expected_return,
                                'kelly_fraction': kelly_fraction,
                                'confidence': conf,
                                'risk_score': self._calculate_risk_score(prediction, multiplier)
                            })
        
        # Sort by expected return and risk tolerance
        sorted_plays = sorted(power_plays, key=lambda x: (x['expected_return'], -x['risk_score']), reverse=True)
        print(f"Debug: Generated {len(sorted_plays)} power plays")
        return sorted_plays
    
    def _kelly_criterion(self, prob, multiplier):
        """Calculate Kelly criterion for optimal bet sizing"""
        if prob <= 0 or prob >= 1:
            return 0
        
        odds = multiplier - 1  # Convert multiplier to odds
        kelly = (prob * odds - (1 - prob)) / odds if odds > 0 else 0
        return max(0, min(kelly, 0.25))  # Cap at 25% of bankroll
    
    def _calculate_risk_score(self, prediction, multiplier):
        """Calculate risk score (lower is better)"""
        variance = prediction['probability'] * (1 - prediction['probability'])
        multiplier_risk = multiplier - 1  # Higher multipliers = more risk
        confidence_risk = 1 - prediction['confidence']
        return variance * multiplier_risk * (1 + confidence_risk)
    
    def create_insurance_plays(self, primary_plays, insurance_budget_pct=0.15):
        """Create insurance plays to hedge against primary play failures"""
        insurance_plays = []
        
        for play in primary_plays[:5]:  # Insure top 5 plays
            # Create opposite bet as insurance
            fail_probability = 1 - play['probability']
            insurance_multiplier = 1 / fail_probability if fail_probability > 0 else 1
            
            # Calculate insurance stake
            primary_stake = play.get('stake', 1.0)
            insurance_stake = primary_stake * insurance_budget_pct
            
            insurance_plays.append({
                'type': 'insurance',
                'primary_play': play,
                'probability': fail_probability,
                'stake': insurance_stake,
                'payout_multiplier': insurance_multiplier,
                'net_protection': insurance_stake * insurance_multiplier - insurance_stake
            })
        
        return insurance_plays
    
    def build_optimal_parlay(self, player_predictions, max_legs=4, min_probability=0.15):
        """Build optimal parlay combinations"""
        eligible_plays = []
        
        # Collect all eligible individual plays
        for player, stats in player_predictions.items():
            for stat, prediction in stats.items():
                if prediction['probability'] >= min_probability:
                    eligible_plays.append({
                        'player': player,
                        'stat': stat,
                        'probability': prediction['probability'],
                        'confidence': prediction['confidence']
                    })
        
        print(f"Debug: Found {len(eligible_plays)} eligible plays for parlays")
        
        # Sort by probability * confidence
        eligible_plays.sort(key=lambda x: x['probability'] * x['confidence'], reverse=True)
        
        # Build parlays of different sizes
        parlays = []
        from itertools import combinations
        
        # Generate all possible combinations for each leg count
        for leg_count in range(2, min(max_legs + 1, len(eligible_plays) + 1)):
            # Get all combinations of this size
            for parlay_combo in combinations(eligible_plays, leg_count):
                parlay_legs = list(parlay_combo)
                
                # Calculate combined probability (assuming independence)
                combined_prob = 1.0
                for leg in parlay_legs:
                    combined_prob *= leg['probability']
                
                # Calculate payout multiplier based on fair odds
                payout_multiplier = 1.0
                for leg in parlay_legs:
                    # Estimate individual odds from probability
                    implied_odds = 1 / leg['probability'] if leg['probability'] > 0 else 1
                    payout_multiplier *= implied_odds
                
                # Calculate expected value
                expected_value = combined_prob * payout_multiplier
                
                # Only include profitable parlays (expected value > 1)
                if expected_value > 1.0:
                    parlays.append({
                        'legs': parlay_legs,
                        'leg_count': leg_count,
                        'combined_probability': combined_prob,
                        'payout_multiplier': payout_multiplier,
                        'expected_value': expected_value,
                        'risk_reward_ratio': expected_value / leg_count
                    })
        
        # Sort by expected value and return top parlays
        sorted_parlays = sorted(parlays, key=lambda x: x['expected_value'], reverse=True)
        print(f"Debug: Generated {len(sorted_parlays)} profitable parlays")
        return sorted_parlays[:20]  # Return top 20 instead of just returning all
    
    def generate_underdog_strategy(self, player_data_dict, bankroll=1000, risk_profile='moderate'):
        """Generate comprehensive Underdog Fantasy strategy"""
        strategy = {
            'total_bankroll': bankroll,
            'risk_profile': risk_profile,
            'individual_predictions': {},
            'power_plays': [],
            'insurance_plays': [],
            'parlay_recommendations': [],
            'bankroll_allocation': {}
        }
        
        # Generate predictions for all players
        for player, data in player_data_dict.items():
            strategy['individual_predictions'][player] = self.predict_individual_stats(data)
        
        # Optimize Power Plays
        strategy['power_plays'] = self.optimize_power_play(
            strategy['individual_predictions'], 
            risk_profile
        )[:10]  # Top 10 power plays
        
        # Create insurance plays
        strategy['insurance_plays'] = self.create_insurance_plays(strategy['power_plays'])
        
        # Build optimal parlays
        strategy['parlay_recommendations'] = self.build_optimal_parlay(
            strategy['individual_predictions']
        )[:5]  # Top 5 parlays
        
        # Allocate bankroll
        strategy['bankroll_allocation'] = self._allocate_bankroll(
            strategy, bankroll, risk_profile
        )
        
        return strategy
    
    def _allocate_bankroll(self, strategy, bankroll, risk_profile):
        """Allocate bankroll across different play types"""
        allocation = {
            'conservative': {'power_plays': 0.4, 'insurance': 0.3, 'parlays': 0.2, 'reserve': 0.1},
            'moderate': {'power_plays': 0.5, 'insurance': 0.2, 'parlays': 0.25, 'reserve': 0.05},
            'aggressive': {'power_plays': 0.6, 'insurance': 0.1, 'parlays': 0.3, 'reserve': 0.0}
        }
        
        profile_allocation = allocation[risk_profile]
        
        return {
            'power_plays_budget': bankroll * profile_allocation['power_plays'],
            'insurance_budget': bankroll * profile_allocation['insurance'],
            'parlay_budget': bankroll * profile_allocation['parlays'],
            'reserve_budget': bankroll * profile_allocation['reserve'],
            'recommended_unit_size': bankroll * 0.02  # 2% per unit
        }

# Reintroduce necessary definitions for league_avg_wOBA, league_avg_HR_FlyBall, and wOBA_weights
league_avg_wOBA = {
    2020: 0.320,
    2021: 0.318,
    2022: 0.317,
    2023: 0.316,
    2024: 0.315
}

league_avg_HR_FlyBall = {
    2020: 0.145,
    2021: 0.144,
    2022: 0.143,
    2023: 0.142,
    2024: 0.141
}

wOBA_weights = {
    2020: {'BB': 0.69, 'HBP': 0.72, '1B': 0.88, '2B': 1.24, '3B': 1.56, 'HR': 2.08},
    2021: {'BB': 0.68, 'HBP': 0.71, '1B': 0.87, '2B': 1.23, '3B': 1.55, 'HR': 2.07},
    2022: {'BB': 0.67, 'HBP': 0.70, '1B': 0.86, '2B': 1.22, '3B': 1.54, 'HR': 2.06},
    2023: {'BB': 0.66, 'HBP': 0.69, '1B': 0.85, '2B': 1.21, '3B': 1.53, 'HR': 2.05},
    2024: {'BB': 0.65, 'HBP': 0.68, '1B': 0.84, '2B': 1.20, '3B': 1.52, 'HR': 2.04}
}

class EnhancedMLBFinancialStyleEngine:
    def __init__(self, stat_cols=None, rolling_windows=None):
        if stat_cols is None:
            self.stat_cols = ['HR', 'RBI', 'BB', 'SB', 'H', '1B', '2B', '3B', 'R', 'calculated_dk_fpts']
        else:
            self.stat_cols = stat_cols
        if rolling_windows is None:
            self.rolling_windows = [3, 7, 14, 28, 45]
        else:
            self.rolling_windows = rolling_windows

    def calculate_features(self, df):
        df = df.copy()
        
        # --- Preprocessing ---
        # Ensure date is datetime and sort
        date_col = 'game_date' if 'game_date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(['Name', date_col])

        # Standardize opportunity columns
        if 'PA' not in df.columns and 'PA.1' in df.columns:
            df['PA'] = df['PA.1']
        if 'AB' not in df.columns and 'AB.1' in df.columns:
            df['AB'] = df['AB.1']
            
        # Ensure base columns exist
        required_cols = self.stat_cols + ['PA', 'AB']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
                print(f"Warning: Column '{col}' not found. Initialized with 0.")

        # Group by player
        all_players_data = []
        for name, group in df.groupby('Name'):
            new_features = {}
            
            # --- Momentum Features (like RSI, MACD) ---
            for col in self.stat_cols:
                for window in self.rolling_windows:
                    # Rolling means (SMA)
                    new_features[f'{col}_sma_{window}'] = group[col].rolling(window).mean()
                    # Exponential rolling means (EMA)
                    new_features[f'{col}_ema_{window}'] = group[col].ewm(span=window, adjust=False).mean()
                    # Rate of Change (Momentum)
                    new_features[f'{col}_roc_{window}'] = group[col].pct_change(periods=window)
                # Performance vs moving average
                if f'{col}_sma_28' in new_features:
                    new_features[f'{col}_vs_sma_28'] = (group[col] / new_features[f'{col}_sma_28']) - 1
            
            # --- Volatility Features (like Bollinger Bands) ---
            for window in self.rolling_windows:
                mean = group['calculated_dk_fpts'].rolling(window).mean()
                std = group['calculated_dk_fpts'].rolling(window).std()
                new_features[f'dk_fpts_upper_band_{window}'] = mean + (2 * std)
                new_features[f'dk_fpts_lower_band_{window}'] = mean - (2 * std)
                if mean is not None and not mean.empty:
                    new_features[f'dk_fpts_band_width_{window}'] = (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}']) / mean
                    new_features[f'dk_fpts_band_position_{window}'] = (group['calculated_dk_fpts'] - new_features[f'dk_fpts_lower_band_{window}']) / (new_features[f'dk_fpts_upper_band_{window}'] - new_features[f'dk_fpts_lower_band_{window}'])

            # --- "Volume" (PA/AB) based Features ---
            for vol_col in ['PA', 'AB']:
                if vol_col in group.columns:
                    new_features[f'{vol_col}_roll_mean_28'] = group[vol_col].rolling(28).mean()
                    new_features[f'{vol_col}_ratio'] = group[vol_col] / new_features[f'{vol_col}_roll_mean_28']
                    new_features[f'dk_fpts_{vol_col}_corr_28'] = group['calculated_dk_fpts'].rolling(28).corr(group[vol_col])

            # --- Interaction / Ratio Features ---
            for col in ['HR', 'RBI', 'BB', 'H', 'SO', 'R']:
                if col in group.columns and 'PA' in group.columns and group['PA'].sum() > 0:
                    new_features[f'{col}_per_pa'] = group[col] / group['PA']
            
            # --- Temporal Features ---
            new_features['day_of_week'] = group[date_col].dt.dayofweek
            new_features['month'] = group[date_col].dt.month
            new_features['is_weekend'] = (new_features['day_of_week'] >= 5).astype(int)
            new_features['day_of_week_sin'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
            new_features['day_of_week_cos'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)

            all_players_data.append(pd.concat([group, pd.DataFrame(new_features, index=group.index)], axis=1))
            
        enhanced_df = pd.concat(all_players_data, ignore_index=True)
        # Final cleanup
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.ffill()
        enhanced_df = enhanced_df.fillna(0)
        return enhanced_df

def clean_infinite_values(df):
    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, replace NaN with the mean of the column
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # For non-numeric columns, replace NaN with a placeholder value (e.g., 'Unknown')
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def calculate_dk_fpts(row):
    return (row['1B'] * 3 + row['2B'] * 5 + row['3B'] * 8 + row['HR'] * 10 +
            row['RBI'] * 2 + row['R'] * 2 + row['BB'] * 2 + row['HBP'] * 2 + row['SB'] * 5)

def engineer_features(df, date_series=None):
    if date_series is None:
        date_series = df['date']
    
    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors='coerce')

    # Extract date features
    df['year'] = date_series.dt.year
    df['month'] = date_series.dt.month
    df['day'] = date_series.dt.day
    df['day_of_week'] = date_series.dt.dayofweek
    df['day_of_season'] = (date_series - date_series.min()).dt.days
    df['week_of_season'] = (date_series - date_series.min()).dt.days // 7
    df['day_of_year'] = date_series.dt.dayofyear

    # Calculate key statistics
    df['wOBA'] = (df['BB']*0.69 + df['HBP']*0.72 + (df['H'] - df['2B'] - df['3B'] - df['HR'])*0.88 + df['2B']*1.24 + df['3B']*1.56 + df['HR']*2.08) / (df['AB'] + df['BB'] - df['IBB'] + df['SF'] + df['HBP'])
    df['BABIP'] = df.apply(lambda x: (x['H'] - x['HR']) / (x['AB'] - x['SO'] - x['HR'] + x['SF']) if (x['AB'] - x['SO'] - x['HR'] + x['SF']) > 0 else 0, axis=1)
    df['ISO'] = df['SLG'] - df['AVG']

    # Advanced Sabermetric Metrics
    logging.info(f"Year range in data: {df['year'].min()} to {df['year'].max()}")
    
    def safe_wRAA(row):
        year = row['year']
        if year not in league_avg_wOBA:
            logging.warning(f"Year {year} not found in league_avg_wOBA. Using {max(league_avg_wOBA.keys())} instead.")
            year = max(league_avg_wOBA.keys())
        return ((row['wOBA'] - league_avg_wOBA[year]) / 1.15) * row['AB'] if row['AB'] > 0 else 0

    df['wRAA'] = df.apply(safe_wRAA, axis=1)
    df['wRC'] = df['wRAA'] + (df['AB'] * 0.1)  # Assuming league_runs/PA = 0.1
    df['wRC+'] = df.apply(lambda x: (x['wRC'] / x['AB'] / league_avg_wOBA.get(x['year'], league_avg_wOBA[2020]) * 100) if x['AB'] > 0 and league_avg_wOBA.get(x['year'], league_avg_wOBA[2020]) > 0 else 0, axis=1)

    df['flyBalls'] = df.apply(lambda x: x['HR'] / league_avg_HR_FlyBall.get(x['year'], league_avg_HR_FlyBall[2020]) if league_avg_HR_FlyBall.get(x['year'], league_avg_HR_FlyBall[2020]) > 0 else 0, axis=1)

    # Calculate singles
    df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']

    # Calculate wOBA using year-specific weights
    def safe_wOBA_Statcast(row):
        year = row['year']
        if year not in wOBA_weights:
            logging.warning(f"Year {year} not found in wOBA_weights. Using {max(wOBA_weights.keys())} instead.")
            year = max(wOBA_weights.keys())
        weights = wOBA_weights[year]
        numerator = (
            weights['BB'] * row['BB'] +
            weights['HBP'] * row['HBP'] +
            weights['1B'] * row['1B'] +
            weights['2B'] * row['2B'] +
            weights['3B'] * row['3B'] +
            weights['HR'] * row['HR']
        )
        denominator = row['AB'] + row['BB'] - row['IBB'] + row['SF'] + row['HBP']
        return numerator / denominator if denominator > 0 else 0

    df['wOBA_Statcast'] = df.apply(safe_wOBA_Statcast, axis=1)

    # Calculate SLG
    df['SLG_Statcast'] = df.apply(lambda x: (
        x['1B'] + (2 * x['2B']) + (3 * x['3B']) + (4 * x['HR'])
    ) / x['AB'] if x['AB'] > 0 else 0, axis=1)

    # Calculate RAR_Statcast (Runs Above Replacement)
    df['RAR_Statcast'] = df['WAR'] * 10 if 'WAR' in df.columns else 0

    # Calculate Offense_Statcast
    df['Offense_Statcast'] = df['wRAA'] + df['BsR'] if 'BsR' in df.columns else df['wRAA']

    # Calculate Dollars_Statcast
    WAR_conversion_factor = 8.0  # Example conversion factor, can be adjusted
    df['Dollars_Statcast'] = df['WAR'] * WAR_conversion_factor if 'WAR' in df.columns else 0

    # Calculate WPA/LI_Statcast
    df['WPA/LI_Statcast'] = df['WPA/LI'] if 'WPA/LI' in df.columns else 0

    # Calculate rolling statistics if 'calculated_dk_fpts' is present
    if 'calculated_dk_fpts' in df.columns:
        for window in [7, 10, 49]:  # Added window 10 for constraint purposes
            df[f'rolling_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min())
            df[f'rolling_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max())
            df[f'rolling_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean())

        for window in [3, 7, 14, 28]:
            df[f'lag_mean_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            df[f'lag_max_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).max().shift(1))
            df[f'lag_min_fpts_{window}'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(window, min_periods=1).min().shift(1))

    # Remove 5-game average calculation - not in training script
    # df['5_game_avg'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    
    # Player consistency features (only add if they exist in training)
    df['fpts_std'] = df.groupby('Name')['calculated_dk_fpts'].transform(lambda x: x.rolling(10, min_periods=1).std())
    df['fpts_volatility'] = df['fpts_std'] / df['rolling_mean_fpts_7']
    
    return df

def concurrent_feature_engineering(df, chunksize):
    print("Starting concurrent feature engineering...")
    
    # Apply financial-style features first
    print("Applying financial-style feature engineering...")
    financial_engine = EnhancedMLBFinancialStyleEngine()
    df = financial_engine.calculate_features(df)
    
    # Then use parallel processing for traditional features
    chunks = [df[i:i+chunksize].copy() for i in range(0, df.shape[0], chunksize)]
    date_series = df['date']
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        processed_chunks = list(executor.map(engineer_features, chunks, 
                                           [date_series[i:i+chunksize] for i in range(0, df.shape[0], chunksize)]))
    
    result_df = pd.concat(processed_chunks)
    
    # Final cleanup of infinite values
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.fillna(0)
    
    print("Concurrent feature engineering completed.")
    return result_df

def create_synthetic_rows_for_all_players(df, all_players, prediction_date):
    print(f"Creating synthetic rows for all players for date: {prediction_date}...")
    synthetic_rows = []
    
    # Calculate league averages for realistic defaults
    league_averages = df.select_dtypes(include=[np.number]).mean()
    league_std = df.select_dtypes(include=[np.number]).std()
    
    for player in all_players:
        player_df = df[df['Name'] == player].sort_values('date', ascending=False)
        if player_df.empty:
            print(f"No historical data found for player {player}. Using league average values.")
            # Create default row with league averages instead of random values
            default_row = pd.DataFrame([league_averages]).copy()
            default_row['date'] = prediction_date
            default_row['Name'] = player
            # Use conservative estimate for unknown players
            default_row['calculated_dk_fpts'] = max(2.0, league_averages.get('calculated_dk_fpts', 5.0))
            default_row['has_historical_data'] = False
            synthetic_rows.append(default_row)
        else:
            print(f"Using {len(player_df)} rows of data for {player}. Date range: {player_df['date'].min()} to {player_df['date'].max()}")
            
            # Use recent data for player averages (up to 5 most recent games)
            player_df = player_df.head(5)
            
            numeric_columns = player_df.select_dtypes(include=[np.number]).columns
            numeric_averages = player_df[numeric_columns].mean()
            
            synthetic_row = pd.DataFrame([numeric_averages], columns=numeric_columns)
            synthetic_row['date'] = prediction_date
            synthetic_row['Name'] = player
            synthetic_row['has_historical_data'] = True
            
            # Ensure 'calculated_dk_fpts' is included and realistic
            if 'calculated_dk_fpts' in player_df.columns:
                dk_fpts_avg = player_df['calculated_dk_fpts'].mean()
                # Apply some variance but keep reasonable
                synthetic_row['calculated_dk_fpts'] = max(0, min(dk_fpts_avg, 25))  # Cap at 25 for baseline
            else:
                synthetic_row['calculated_dk_fpts'] = 5.0  # Conservative default
            
            # Handle categorical columns
            for col in player_df.select_dtypes(include=['object']).columns:
                if col not in ['date', 'Name']:
                    synthetic_row[col] = player_df[col].mode().iloc[0] if not player_df[col].mode().empty else player_df[col].iloc[0]
            
            synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    
    # Final validation and cleanup
    synthetic_df = validate_synthetic_data(synthetic_df)
    
    print(f"Created {len(synthetic_rows)} synthetic rows for date: {prediction_date}.")
    print(f"Number of players with historical data: {sum(synthetic_df['has_historical_data'])}")
    print(f"Synthetic data stats - calculated_dk_fpts range: {synthetic_df['calculated_dk_fpts'].min():.2f} to {synthetic_df['calculated_dk_fpts'].max():.2f}")
    
    return synthetic_df

def validate_synthetic_data(df):
    """Validate and clean synthetic data to ensure realistic values"""
    # Cap extreme values in key statistics
    df['calculated_dk_fpts'] = np.clip(df['calculated_dk_fpts'], 0, 30)
    
    # Remove 5_game_avg reference - not in training script
    # if '5_game_avg' in df.columns:
    #     df['5_game_avg'] = np.clip(df['5_game_avg'], 0, 30)
    
    # Ensure batting stats are within realistic ranges
    if 'AVG' in df.columns:
        df['AVG'] = np.clip(df['AVG'], 0, 0.500)
    if 'OBP' in df.columns:
        df['OBP'] = np.clip(df['OBP'], 0, 0.600)
    if 'SLG' in df.columns:
        df['SLG'] = np.clip(df['SLG'], 0, 1.000)
    
    # Clean infinite and NaN values
    df = clean_infinite_values(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def process_predictions(chunk, pipeline, player_adjustments):
    # Prepare features exactly as in training
    features = chunk.drop(columns=['calculated_dk_fpts'])

    # Clean the features to ensure no infinite or excessively large values
    features = clean_infinite_values(features)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    # Use the complete pipeline for prediction (this handles preprocessing and feature selection)
    raw_predictions = pipeline.predict(features)

    # Debug: Print raw prediction statistics
    print(f"Raw predictions - Min: {raw_predictions.min():.2f}, Max: {raw_predictions.max():.2f}, Mean: {raw_predictions.mean():.2f}")

    # Apply smart outlier handling instead of hard clipping
    chunk.loc[:, 'predicted_dk_fpts'] = apply_smart_prediction_constraints(raw_predictions, chunk)
    chunk.loc[:, 'predicted_dk_fpts'] = chunk.apply(lambda row: adjust_predictions(row, player_adjustments), axis=1)

    # Remove outliers with predictions > 35
    outlier_count = chunk[chunk['predicted_dk_fpts'] > 35].shape[0]
    if outlier_count > 0:
        print(f"Removing {outlier_count} outliers with predictions > 35.")
        chunk = chunk[chunk['predicted_dk_fpts'] <= 35]

    # Debug: Print statistics after outlier removal
    print(f"After outlier removal: {chunk.shape[0]} players")

    return chunk

def apply_smart_prediction_constraints(raw_predictions, chunk):
    """Apply strict constraints based on player's 10-day rolling range"""
    constrained_predictions = raw_predictions.copy()
    
    # Calculate realistic bounds based on player's 10-day rolling range
    for i, (idx, row) in enumerate(chunk.iterrows()):
        raw_pred = raw_predictions[i]
        
        # Get 10-day rolling min and max if available
        rolling_min_10 = row.get('rolling_min_fpts_10', None)
        rolling_max_10 = row.get('rolling_max_fpts_10', None)
        
        # If we have 10-day rolling data, use it as primary constraint
        if rolling_min_10 is not None and rolling_max_10 is not None and not (np.isnan(rolling_min_10) or np.isnan(rolling_max_10)):
            # Use 10-day range with minimal expansion (10% or 2 points max)
            range_expansion = min(2.0, (rolling_max_10 - rolling_min_10) * 0.1)
            lower_bound = max(0, rolling_min_10 - range_expansion)
            upper_bound = rolling_max_10 + range_expansion
            
            # Ensure minimum reasonable range of 3 points
            current_range = upper_bound - lower_bound
            if current_range < 3.0:
                center = (upper_bound + lower_bound) / 2
                lower_bound = max(0, center - 1.5)
                upper_bound = center + 1.5
            
            # Hard constraint within this range
            constrained_predictions[i] = np.clip(raw_pred, lower_bound, upper_bound)
            
        else:
            # For players without 10-day data, use 7-day range if available
            rolling_min_7 = row.get('rolling_min_fpts_7', None)
            rolling_max_7 = row.get('rolling_max_fpts_7', None)
            
            if rolling_min_7 is not None and rolling_max_7 is not None and not (np.isnan(rolling_min_7) or np.isnan(rolling_max_7)):
                # Use 7-day range with slightly more expansion
                range_expansion = min(3.0, (rolling_max_7 - rolling_min_7) * 0.2)
                lower_bound = max(0, rolling_min_7 - range_expansion)
                upper_bound = rolling_max_7 + range_expansion
                
                # Ensure minimum reasonable range of 4 points
                current_range = upper_bound - lower_bound
                if current_range < 4.0:
                    center = (upper_bound + lower_bound) / 2
                    lower_bound = max(0, center - 2.0)
                    upper_bound = center + 2.0
                
                constrained_predictions[i] = np.clip(raw_pred, lower_bound, upper_bound)
            else:
                # Fallback for players with no recent history - use conservative league-wide ranges
                # Use position-based reasonable ranges
                position = row.get('Position', 'OF')  # Default to OF
                if position in ['C']:
                    # Catchers tend to score less
                    constrained_predictions[i] = np.clip(raw_pred, 0, 12)
                elif position in ['1B', '3B', 'OF']:
                    # Power positions
                    constrained_predictions[i] = np.clip(raw_pred, 0, 18)
                elif position in ['SS', '2B']:
                    # Middle infield
                    constrained_predictions[i] = np.clip(raw_pred, 0, 15)
                else:
                    # Default fallback
                    constrained_predictions[i] = np.clip(raw_pred, 0, 15)
    
    # Final safety check - no prediction should exceed absolute maximum
    constrained_predictions = np.clip(constrained_predictions, 0, 35)
    
    print(f"Constrained predictions - Min: {constrained_predictions.min():.2f}, Max: {constrained_predictions.max():.2f}, Mean: {constrained_predictions.mean():.2f}")
    
    return constrained_predictions

def adjust_predictions(row, player_adjustments):
    """Adjust predictions based on player-specific average differences."""
    prediction = row['predicted_dk_fpts']
    player = row['Name']
    
    # Check if player_adjustments is empty or if player is not in adjustments
    if player_adjustments.empty or player not in player_adjustments.index:
        # No adjustments available, return original prediction
        return max(0, prediction)
    
    try:
        if prediction > row.get('calculated_dk_fpts', 0):
            adjustment = player_adjustments.loc[player, 'avg_positive_diff'] / 4  # Reduced adjustment factor
        else:
            adjustment = player_adjustments.loc[player, 'avg_negative_diff'] / 4  # Reduced adjustment factor
        
        adjusted_prediction = prediction - adjustment
    except (KeyError, TypeError):
        # If adjustment fails, use a small default adjustment
        if prediction > row.get('calculated_dk_fpts', 0):
            adjusted_prediction = prediction - 0.5
        else:
            adjusted_prediction = prediction + 0.5
    
    return max(0, adjusted_prediction)  # Ensure non-negative prediction

def rolling_predictions(train_data, model_pipeline, test_dates, chunksize):
    print("Starting rolling predictions...")
    results = []
    for current_date in test_dates:
        print(f"Processing date: {current_date}")
        synthetic_rows = create_synthetic_rows_for_all_players(train_data, train_data['Name'].unique(), current_date)
        if synthetic_rows.empty:
            print(f"No synthetic rows generated for date: {current_date}")
            continue
        print(f"Synthetic rows generated for date: {current_date}")
        chunks = [synthetic_rows[i:i+chunksize].copy() for i in range(0, synthetic_rows.shape[0], chunksize)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            processed_chunks = list(executor.map(process_predictions, chunks, [model_pipeline]*len(chunks)))
        results.extend(processed_chunks)
    print(f"Generated rolling predictions for {len(results)} days.")
    return pd.concat(results)

def predict_unseen_data(input_file, model_file, prediction_date):
    print("Loading dataset...")
    df = pd.read_csv(input_file,
                     dtype={'inheritedRunners': 'float64', 'inheritedRunnersScored': 'float64', 'catchersInterference': 'int64', 'salary': 'int64'},
                     low_memory=False)
    
    # Debug: Check the first few date values and their format
    print(f"Sample date values from CSV:")
    print(df['date'].head(10).tolist())
    print(f"Date column data type: {df['date'].dtype}")
    
    # Try multiple approaches to parse dates
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            # Try with infer_datetime_format
            df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # Fallback: just use errors='coerce' to convert what we can
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Check for any failed conversions
    null_dates = df['date'].isnull().sum()
    if null_dates > 0:
        print(f"Warning: {null_dates} dates could not be parsed and were set to NaT")
        # Drop rows with null dates
        df = df.dropna(subset=['date'])
        print(f"Dropped rows with invalid dates. New shape: {df.shape}")
    
    prediction_date = pd.to_datetime(prediction_date)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range in dataset: {df['date'].min()} to {df['date'].max()}")
    print(f"Year range in dataset: {df['date'].dt.year.min()} to {df['date'].dt.year.max()}")
    print(f"Number of unique players: {df['Name'].nunique()}")
    
    # Get all unique players from the entire dataset
    all_players = df['Name'].unique()
      # No need to filter data up to the prediction date
    df.sort_values(by=['Name', 'date'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # Calculate calculated_dk_fpts if not present
    if 'calculated_dk_fpts' not in df.columns:
        print("Calculating DK Fantasy Points...")
        df['calculated_dk_fpts'] = df.apply(calculate_dk_fpts, axis=1)

    df.fillna(0, inplace=True)
    print("Dataset loaded and preprocessed.")    # Load or create LabelEncoders - Updated paths to match training script
    name_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/label_encoder_name_sep2.pkl'
    team_encoder_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/label_encoder_team_sep2.pkl'
    scaler_path = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/scaler_sep2.pkl'    # Load or create LabelEncoders (handle version compatibility)
    try:
        if os.path.exists(name_encoder_path):
            le_name = joblib.load(name_encoder_path)
        else:
            le_name = LabelEncoder()
            le_name.fit(df['Name'])
            joblib.dump(le_name, name_encoder_path)
    except (FileNotFoundError, Exception) as e:
        print("Creating new name encoder due to compatibility issues...")
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)

    try:
        if os.path.exists(team_encoder_path):
            le_team = joblib.load(team_encoder_path)
        else:
            le_team = LabelEncoder()
            le_team.fit(df['Team'])
            joblib.dump(le_team, team_encoder_path)
    except (FileNotFoundError, Exception) as e:
        print("Creating new team encoder due to compatibility issues...")
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)

    # Update LabelEncoders with new players/teams - Fix this to match training approach
    # Instead of manually expanding encoder classes, use the training approach
    # that recreates encoders when needed
    
    # This ensures compatibility with the training pipeline
    try:
        # Test if encoders work with current data
        df['Name_encoded'] = le_name.transform(df['Name'])
        df['Team_encoded'] = le_team.transform(df['Team'])
    except ValueError as e:
        print(f"Encoder compatibility issue: {e}")
        print("Recreating encoders with current data...")
        # Recreate encoders with all data (training + new)
        le_name = LabelEncoder()
        le_name.fit(df['Name'])
        joblib.dump(le_name, name_encoder_path)
        
        le_team = LabelEncoder()
        le_team.fit(df['Team'])
        joblib.dump(le_team, team_encoder_path)
        
        # Now encode with new encoders
        df['Name_encoded'] = le_name.transform(df['Name'])
        df['Team_encoded'] = le_team.transform(df['Team'])    # Remove this scaler loading - the pipeline handles scaling internally
    # if os.path.exists(scaler_path):
    #     scaler = joblib.load(scaler_path)
    # else:
    #     raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # The pipeline includes all preprocessing steps

    chunksize = 20000  # Increased for better performance
    
    # --- New Financial-Style Feature Engineering Step ---
    print("Starting enhanced feature engineering with financial-style features...")
    df = concurrent_feature_engineering(df, chunksize)
    
    # --- Centralized Data Cleaning ---
    print("Cleaning final dataset of any infinite or NaN values...")
    df = clean_infinite_values(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    print("Enhanced feature engineering complete.")

    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # Fix the categorical variable encoding to match training exactly
    # The training script uses 'Name' and 'Team' as categorical features
    # Remove these lines that override the proper encoding:
    # df['team_encoded'] = df['Team']
    # df['Name_encoded'] = df['Name']
    
    # The pipeline will handle the categorical encoding properly

    if df.empty:
        raise ValueError(f"No data available up to {prediction_date}")
    
    print(f"Data available up to {df['date'].max()}")

    print("Loading Underdog Fantasy models...")
    
    # Load all available underdog models
    underdog_models = {}
    models_dir = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/'
    
    # List of all possible underdog targets
    underdog_targets = [
        'hits_over_0.5', 'hits_over_1.5', 'runs_over_0.5', 'runs_over_1.5',
        'rbis_over_0.5', 'rbis_over_1.5', 'home_runs_over_0.5', 'stolen_bases_over_0.5',
        'total_bases_over_1.5', 'total_bases_over_2.5'
    ]
    
    for target in underdog_targets:
        model_path = os.path.join(models_dir, f'underdog_{target}_model.pkl')
        if os.path.exists(model_path):
            try:
                underdog_models[target] = joblib.load(model_path)
                print(f"✓ Loaded model for {target}")
            except Exception as e:
                print(f"✗ Failed to load model for {target}: {e}")
        else:
            print(f"✗ Model not found for {target}")
    
    if not underdog_models:
        print("No Underdog Fantasy models found! Falling back to DraftKings model...")
        pipeline = joblib.load(model_file)
        print("Model pipeline steps:")
        for step_name, step in pipeline.named_steps.items():
            print(f"- {step_name}: {type(step).__name__}")
    else:
        print(f"Successfully loaded {len(underdog_models)} Underdog Fantasy models")
        pipeline = None  # We'll use underdog_models instead
    
    # Function to make underdog predictions
    def make_underdog_predictions(df_chunk, models_dict):
        """Make binary predictions using trained underdog models"""
        predictions = {}
        
        # Prepare features (similar to training)
        feature_cols = [col for col in df_chunk.columns if col not in ['Name', 'date', 'calculated_dk_fpts']]
        X = df_chunk[feature_cols]
        
        # Clean features
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        for target, model in models_dict.items():
            try:
                # Get probabilities for the positive class (over the threshold)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]  # Probability of positive class
                else:
                    proba = model.decision_function(X)  # For SVM or other classifiers
                    # Convert to probabilities using sigmoid
                    proba = 1 / (1 + np.exp(-proba))
                
                predictions[target] = proba
            except Exception as e:
                print(f"Error predicting {target}: {e}")
                predictions[target] = np.zeros(len(X))
        
        return predictions
    
    print(f"Processing date: {prediction_date}")
    
    # Create synthetic rows for all players for the prediction date
    current_df = create_synthetic_rows_for_all_players(df, all_players, prediction_date)
    
    if current_df.empty:
        print(f"No data available for date: {prediction_date}")
        return
    
    # Make underdog predictions if models are available
    if underdog_models:
        print("Making Underdog Fantasy predictions...")
        underdog_predictions = make_underdog_predictions(current_df, underdog_models)
        
        # Create underdog predictions dataframe
        underdog_df = pd.DataFrame(underdog_predictions, index=current_df.index)
        underdog_df['Name'] = current_df['Name']
        underdog_df['date'] = prediction_date
        
        # Save underdog predictions
        output_dir = '7_ANALYSIS'
        os.makedirs(output_dir, exist_ok=True)
        underdog_output_file = os.path.join(output_dir, f'underdog_predictions_{prediction_date}.csv')
        underdog_df.to_csv(underdog_output_file, index=False)
        print(f"Underdog predictions saved to: {underdog_output_file}")
        
        # Generate underdog strategy
        print("Generating Underdog Fantasy strategy...")
        
        # Convert predictions to player_data format for strategy generation
        player_data_dict = {}
        for idx, row in current_df.iterrows():
            player = row['Name']
            player_data_dict[player] = {
                'hits_over_0.5': underdog_predictions.get('hits_over_0.5', [0.5])[idx] if 'hits_over_0.5' in underdog_predictions else 0.5,
                'hits_over_1.5': underdog_predictions.get('hits_over_1.5', [0.3])[idx] if 'hits_over_1.5' in underdog_predictions else 0.3,
                'runs_over_0.5': underdog_predictions.get('runs_over_0.5', [0.4])[idx] if 'runs_over_0.5' in underdog_predictions else 0.4,
                'rbis_over_0.5': underdog_predictions.get('rbis_over_0.5', [0.4])[idx] if 'rbis_over_0.5' in underdog_predictions else 0.4,
                'home_runs_over_0.5': underdog_predictions.get('home_runs_over_0.5', [0.2])[idx] if 'home_runs_over_0.5' in underdog_predictions else 0.2,
                'stolen_bases_over_0.5': underdog_predictions.get('stolen_bases_over_0.5', [0.1])[idx] if 'stolen_bases_over_0.5' in underdog_predictions else 0.1,
                'total_bases_over_1.5': underdog_predictions.get('total_bases_over_1.5', [0.6])[idx] if 'total_bases_over_1.5' in underdog_predictions else 0.6,
            }
        
        # Generate strategy using the UnderdogPredictor class
        predictor = UnderdogPredictor()
        strategy = predictor.generate_underdog_strategy(player_data_dict, bankroll=1000, risk_profile='moderate')
        
        # Save strategy
        strategy_output_file = os.path.join(output_dir, f'underdog_strategy_{prediction_date}.json')
        with open(strategy_output_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            strategy_json = {}
            for key, value in strategy.items():
                if isinstance(value, dict):
                    strategy_json[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in value.items()}
                elif hasattr(value, 'tolist'):
                    strategy_json[key] = value.tolist()
                else:
                    strategy_json[key] = value
            
            json.dump(strategy_json, f, indent=2, default=str)
        print(f"Underdog strategy saved to: {strategy_output_file}")
        
        print("\n=== UNDERDOG FANTASY ANALYSIS COMPLETE ===")
        print(f"Models used: {list(underdog_models.keys())}")
        print(f"Players analyzed: {len(player_data_dict)}")
        print(f"Top 5 power plays:")
        for i, play in enumerate(strategy['power_plays'][:5]):
            print(f"  {i+1}. {play}")
        print(f"Check {output_dir} for detailed results")
        return
    
      # Load player adjustments
    player_adjustments_path = '4_DATA/player_adjustments.csv'
    if os.path.exists(player_adjustments_path):
        player_adjustments = pd.read_csv(player_adjustments_path, index_col='Name')
    else:
        print("Player adjustments file not found. Using default adjustments.")
        player_adjustments = pd.DataFrame(columns=['avg_positive_diff', 'avg_negative_diff'])
      # Process predictions in chunks
    chunks = [current_df[i:i+chunksize] for i in range(0, current_df.shape[0], chunksize)]
    chunk_predictions = []
    all_features_for_prob = []
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_pred = process_predictions(chunk, pipeline, player_adjustments)
        chunk_predictions.append(chunk_pred)
        
        # Collect features for probability prediction
        features_for_prob = chunk.drop(columns=['calculated_dk_fpts'])
        features_for_prob = clean_infinite_values(features_for_prob)
        all_features_for_prob.append(features_for_prob)
    
    # Combine chunk predictions
    predictions = pd.concat(chunk_predictions)
    all_features = pd.concat(all_features_for_prob)
      # Create enhanced predictions with probabilities
    print("Creating enhanced DraftKings predictions with probability analysis...")
    enhanced_predictions, probability_results, prob_summary = create_enhanced_predictions_with_probabilities(
        pipeline, all_features, predictions['Name'], prediction_date
    )
    
    # Update predictions with enhanced values
    predictions['predicted_dk_fpts'] = enhanced_predictions
    
    print("Prediction statistics:")
    if 'predicted_dk_fpts' in predictions.columns:
        print(predictions['predicted_dk_fpts'].describe())
        print(f"Prediction range: {predictions['predicted_dk_fpts'].min():.2f} to {predictions['predicted_dk_fpts'].max():.2f}")
    else:
        print("Error: 'predicted_dk_fpts' column not found in predictions.")
        print("Available columns:", predictions.columns.tolist())    # Save main predictions
    output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/batters_predictions_{prediction_date.strftime("%Y%m%d")}.csv'
    predictions.to_csv(output_file, index=False)
    print(f"Main predictions saved to {output_file}")
    
    # Save probability predictions
    prob_output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/batters_probability_predictions_{prediction_date.strftime("%Y%m%d")}.csv'
    prob_summary.to_csv(prob_output_file, index=False)
    print(f"Probability predictions saved to {prob_output_file}")
    
    # CREATE COMPREHENSIVE UNDERDOG PLAYER DATASET
    print("\nCreating comprehensive Underdog player dataset from DraftKings predictions...")
    underdog_player_dataset = create_underdog_player_dataset_from_predictions(predictions, prediction_date)
    
    # Display sample DraftKings probability predictions
    print("\nSample DraftKings MLB Probability Predictions:")
    print("="*80)
    sample_df = prob_summary.head(10)
    for _, row in sample_df.iterrows():
        print(f"\nPlayer: {row['Name']}")
        print(f"Predicted DraftKings Points: {row['Predicted_DK_Points']:.1f}")
        print("Probability of exceeding thresholds:")
        for threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            prob_col = f'Prob_Over_{threshold}'
            if prob_col in row:
                print(f"  > {threshold} points: {row[prob_col]}")
    print("="*80)
    
    # Print sample of main predictions
    print("\nSample main predictions:")
    print(predictions[['Name', 'predicted_dk_fpts', 'has_historical_data']].head(10))
    
    # === UNDERDOG FANTASY INTEGRATION ===
    print("\n" + "="*80)
    print("GENERATING UNDERDOG FANTASY PREDICTIONS AND STRATEGIES")
    print("="*80)
    
    # Initialize Underdog predictor
    underdog_predictor = UnderdogProbabilityPredictor()
    
    # Convert DraftKings predictions to Underdog player data format
    player_data_dict = {}
    for _, row in predictions.head(20).iterrows():  # Top 20 players for strategy
        player_name = row['Name']
        
        # Convert DK predictions to Underdog-compatible format
        player_data_dict[player_name] = {
            'batting_avg': max(0.200, row.get('calculated_dk_fpts', 0) / 40),  # Approximate BA from DK points
            'at_bats': 4,  # Typical game at-bats
            'obp': min(0.450, max(0.280, row.get('calculated_dk_fpts', 0) / 35)),  # Approximate OBP
            'lineup_position_factor': 1.0 + (row.get('calculated_dk_fpts', 0) - 10) / 50,  # Position boost
            'team_runs_per_game': 4.5,  # League average
            'rbi_rate': max(0.15, row.get('calculated_dk_fpts', 0) / 60),  # RBI opportunity rate
            'risp_avg': max(0.200, row.get('calculated_dk_fpts', 0) / 45),  # RISP performance
            'team_obp': 0.320,  # League average team OBP
            'strikeout_rate': 0.25 if 'pitcher' in str(row.get('position', '')).lower() else 0.20,
            'opponent_contact_rate': 0.80,
            'expected_innings': 6.0 if 'pitcher' in str(row.get('position', '')).lower() else 0,
            'games_played': 50,  # Assume half season sample
            'recent_avg': max(0.200, row.get('calculated_dk_fpts', 0) / 42)  # Recent form
        }
    
    # Generate comprehensive strategy for different risk profiles
    risk_profiles = ['conservative', 'moderate', 'aggressive']
    
    for risk_profile in risk_profiles:
        print(f"\n{'-'*20} {risk_profile.upper()} STRATEGY {'-'*20}")
        
        strategy = underdog_predictor.generate_underdog_strategy(
            player_data_dict, 
            bankroll=1000, 
            risk_profile=risk_profile
        )
        
        print(f"\nBankroll Allocation (${strategy['total_bankroll']}):")
        allocation = strategy['bankroll_allocation']
        print(f"  Power Plays: ${allocation['power_plays_budget']:.2f}")
        print(f"  Insurance: ${allocation['insurance_budget']:.2f}")
        print(f"  Parlays: ${allocation['parlay_budget']:.2f}")
        print(f"  Reserve: ${allocation['reserve_budget']:.2f}")
        print(f"  Unit Size: ${allocation['recommended_unit_size']:.2f}")
        
        print(f"\nTop 5 Power Play Recommendations:")
        for i, play in enumerate(strategy['power_plays'][:5], 1):
            print(f"  {i}. {play['player']} - {play['stat']} ({play['multiplier']})")
            print(f"     Probability: {play['probability']:.1%} | Expected Return: {play['expected_return']:.2f}")
            print(f"     Kelly Fraction: {play['kelly_fraction']:.1%} | Confidence: {play['confidence']:.1%}")
        
        print(f"\nTop 3 Parlay Recommendations:")
        for i, parlay in enumerate(strategy['parlay_recommendations'][:3], 1):
            print(f"  {i}. {parlay['leg_count']}-leg parlay:")
            for leg in parlay['legs']:
                print(f"     - {leg['player']} {leg['stat']} ({leg['probability']:.1%})")
            print(f"     Combined Probability: {parlay['combined_probability']:.1%}")
            print(f"     Payout: {parlay['payout_multiplier']:.1f}x | Expected Value: {parlay['expected_value']:.2f}")
        
        print(f"\nInsurance Plays: {len(strategy['insurance_plays'])} available")
        
        # Save strategy to CSV
        strategy_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_strategy_{risk_profile}_{prediction_date.strftime("%Y%m%d")}.csv'
        
        # Convert strategy to DataFrame for saving
        strategy_df_data = []
        
        # Add power plays
        for play in strategy['power_plays']:
            strategy_df_data.append({
                'type': 'power_play',
                'player': play['player'],
                'stat': play['stat'],
                'probability': play['probability'],
                'multiplier': play['multiplier'],
                'expected_return': play['expected_return'],
                'kelly_fraction': play['kelly_fraction'],
                'confidence': play['confidence'],
                'risk_score': play['risk_score'],
                'legs': 1,
                'leg_details': f"{play['player']} {play['stat']}"
            })
        
        # Add top 10 parlays
        for i, parlay in enumerate(strategy['parlay_recommendations'][:10], 1):
            leg_details = " + ".join([f"{leg['player']} {leg['stat']}" for leg in parlay['legs']])
            strategy_df_data.append({
                'type': 'parlay',
                'player': f"Parlay_{i}",
                'stat': f"{parlay['leg_count']}_leg_parlay",
                'probability': parlay['combined_probability'],
                'multiplier': f"{parlay['payout_multiplier']:.1f}x",
                'expected_return': parlay['expected_value'],
                'kelly_fraction': 0,  # Kelly not calculated for parlays yet
                'confidence': sum([leg['confidence'] for leg in parlay['legs']]) / len(parlay['legs']),
                'risk_score': 1 / parlay['combined_probability'],  # Inverse probability as risk
                'legs': parlay['leg_count'],
                'leg_details': leg_details
            })
        
        strategy_df = pd.DataFrame(strategy_df_data)
        strategy_df.to_csv(strategy_file, index=False)
        print(f"Strategy saved to: {strategy_file}")
        
        # Save separate parlay file for detailed analysis
        parlay_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_parlays_{risk_profile}_{prediction_date.strftime("%Y%m%d")}.csv'
        
        parlay_df_data = []
        for i, parlay in enumerate(strategy['parlay_recommendations'][:10], 1):
            parlay_row = {
                'parlay_id': f"Parlay_{i}",
                'leg_count': parlay['leg_count'],
                'combined_probability': parlay['combined_probability'],
                'payout_multiplier': parlay['payout_multiplier'],
                'expected_value': parlay['expected_value'],
                'risk_reward_ratio': parlay['risk_reward_ratio'],
                'expected_profit_pct': (parlay['expected_value'] - 1) * 100
            }
            
            # Add individual leg details
            for j, leg in enumerate(parlay['legs'], 1):
                parlay_row[f'leg_{j}_player'] = leg['player']
                parlay_row[f'leg_{j}_stat'] = leg['stat']
                parlay_row[f'leg_{j}_probability'] = leg['probability']
                parlay_row[f'leg_{j}_confidence'] = leg['confidence']
            
            parlay_df_data.append(parlay_row)
        
        if parlay_df_data:
            parlay_df = pd.DataFrame(parlay_df_data)
            parlay_df.to_csv(parlay_file, index=False)
            print(f"Top 10 parlays saved to: {parlay_file}")
    
    print("\n" + "="*80)
    print("UNDERDOG FANTASY ANALYSIS COMPLETE")
    print("="*80)

    return predictions

class ProbabilityPredictor:
    """
    A class to predict probabilities of achieving different point thresholds
    using quantile regression and distribution modeling.
    """
    def __init__(self, point_thresholds=None, quantiles=None):
        if point_thresholds is None:
            # DFS point thresholds in 5-point increments for realistic DraftKings analysis
            self.point_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        else:
            self.point_thresholds = point_thresholds
            
        if quantiles is None:
            # Quantiles for distribution modeling
            self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        else:
            self.quantiles = quantiles
            
        self.quantile_models = {}
        self.distribution_params = None
        
    def predict_probabilities(self, X, main_predictions, preprocessor, selector):
        """
        Predict probabilities of achieving different point thresholds
        """
        results = []
          # Simple probability estimation based on prediction value and typical MLB variance
        typical_std = 8.0  # Typical standard deviation for MLB DraftKings points
        
        for i, main_pred in enumerate(main_predictions):
            player_probs = {'main_prediction': main_pred}
            
            for threshold in self.point_thresholds:
                if typical_std > 0:
                    # Probability of exceeding threshold using normal approximation
                    z_score = (threshold - main_pred) / typical_std
                    prob_exceed = 1 - stats.norm.cdf(z_score)
                    player_probs[f'prob_over_{threshold}'] = max(0, min(1, prob_exceed))
                else:
                    # If no variance, use deterministic approach
                    player_probs[f'prob_over_{threshold}'] = 1.0 if main_pred > threshold else 0.0
                    
            results.append(player_probs)
            
        return results
        
    def create_probability_summary(self, probability_results, player_names, prediction_date=None):
        """
        Create a summary DataFrame with probability predictions
        """
        summary_data = []
        
        for i, (name, probs) in enumerate(zip(player_names, probability_results)):
            row = {
                'Name': name, 
                'Date': prediction_date if prediction_date else pd.Timestamp.now().date(),
                'Predicted_DK_Points': probs['main_prediction']
            }
            
            # Add probability columns
            for threshold in self.point_thresholds:
                if f'prob_over_{threshold}' in probs:
                    row[f'Prob_Over_{threshold}'] = f"{probs[f'prob_over_{threshold}']:.1%}"
                    
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)

def create_enhanced_predictions_with_probabilities(pipeline, features, player_names, prediction_date=None):
    """
    Create predictions with probability estimates - Updated to work with complete pipeline
    """
    print("Creating enhanced DraftKings predictions with probabilities...")
    
    # Get main predictions using the complete pipeline
    main_predictions = pipeline.predict(features)
    
    # Debug: Print raw prediction statistics
    print(f"Raw main predictions - Min: {main_predictions.min():.2f}, Max: {main_predictions.max():.2f}, Mean: {main_predictions.mean():.2f}")
    
    # Apply realistic constraints for MLB DraftKings fantasy points
    # Most MLB games result in 0-30 points, with exceptional games up to 45
    main_predictions = np.clip(main_predictions, 0, 45)
    
    # Additional outlier handling - if prediction is extremely high, apply log scaling
    for i in range(len(main_predictions)):
        if main_predictions[i] > 30:
            excess = main_predictions[i] - 30
            # Apply logarithmic scaling to reduce extreme values
            scaled_excess = np.log1p(excess) * 3
            main_predictions[i] = 30 + scaled_excess
    
    # Final safety constraint
    main_predictions = np.clip(main_predictions, 0, 40)
    
    print(f"Constrained main predictions - Min: {main_predictions.min():.2f}, Max: {main_predictions.max():.2f}, Mean: {main_predictions.mean():.2f}")
    
    # Get probability predictions - No need to extract preprocessor/selector separately
    # The pipeline handles everything internally
    probability_predictor = ProbabilityPredictor()
    probability_results = probability_predictor.predict_probabilities(
        features, main_predictions, None, None
    )
    
    # Create summary DataFrame with date
    prob_summary = probability_predictor.create_probability_summary(
        probability_results, player_names, prediction_date
    )
    
    return main_predictions, probability_results, prob_summary

def create_underdog_player_dataset_from_predictions(predictions_df, prediction_date):
    """
    Create a comprehensive Underdog Fantasy player dataset from DraftKings predictions.
    This converts DK predictions into Underdog-compatible format with all necessary fields.
    """
    print("Converting DraftKings predictions to Underdog Fantasy format...")
    
    underdog_rows = []
    
    for _, row in predictions_df.iterrows():
        player_name = row['Name']
        dk_points = row.get('predicted_dk_fpts', 0)
        
        # Convert DraftKings stats to Underdog Fantasy format
        underdog_row = {
            'Name': player_name,
            'Date': prediction_date.strftime('%Y-%m-%d'),
            'Team': row.get('Team', 'UNK'),
            'Position': row.get('Position', 'UNK'),
            'predicted_dk_fpts': dk_points,
            'has_historical_data': row.get('has_historical_data', True),
            
            # Convert DK points to batting stats (approximations)
            'batting_avg': max(0.180, min(0.400, dk_points / 40)),
            'at_bats': 4,  # Standard at-bats per game
            'obp': max(0.250, min(0.500, dk_points / 35)),
            'lineup_position_factor': 1.0 + (dk_points - 8) / 40,  # Position in lineup boost
            'team_runs_per_game': 4.5 + (dk_points - 8) / 20,  # Team offensive strength
            'rbi_rate': max(0.12, min(0.45, dk_points / 60)),
            'risp_avg': max(0.180, min(0.400, dk_points / 45)),
            'team_obp': 0.320 + (dk_points - 8) / 100,  # Team OBP
            'games_played': row.get('games_played', 100),
            'recent_avg': max(0.180, min(0.400, dk_points / 42)),
            
            # Pitcher-specific stats (if applicable)
            'strikeout_rate': 0.25 if row.get('Position', '') == 'P' else 0.20,
            'opponent_contact_rate': 0.80 - (dk_points - 8) / 200 if row.get('Position', '') == 'P' else 0.80,
            'expected_innings': max(5.0, min(8.0, dk_points / 2.5)) if row.get('Position', '') == 'P' else 0,
            'games_started': row.get('games_started', 25) if row.get('Position', '') == 'P' else 0,
            
            # Underdog Fantasy specific fields
            'underdog_eligible': 'Yes',
            'status': 'Available',
            'confidence_score': min(0.95, max(0.50, row.get('has_historical_data', True) * 0.8 + 0.2)),
            
            # Statistical projections for different outcomes
            'hits_probability': max(0.15, min(0.50, dk_points / 35)),
            'runs_probability': max(0.08, min(0.35, dk_points / 45)),
            'rbis_probability': max(0.10, min(0.40, dk_points / 50)),
            'strikeouts_probability': max(0.15, min(0.90, dk_points / 20)) if row.get('Position', '') == 'P' else 0,
            
            # Risk assessment
            'volatility_score': abs(dk_points - 10) / 20,  # Distance from average
            'upside_potential': max(0, dk_points - 12) / 8,  # Potential for big games
            'floor_score': max(2, dk_points - 5),  # Likely minimum performance
            'ceiling_score': dk_points + 8,  # Potential maximum performance
        }
        
        underdog_rows.append(underdog_row)
    
    # Convert to DataFrame
    underdog_df = pd.DataFrame(underdog_rows)
    
    # Clean and validate data
    underdog_df = underdog_df.fillna(0)
    
    # Cap extreme values
    underdog_df['batting_avg'] = underdog_df['batting_avg'].clip(0.150, 0.450)
    underdog_df['obp'] = underdog_df['obp'].clip(0.250, 0.550)
    underdog_df['lineup_position_factor'] = underdog_df['lineup_position_factor'].clip(0.7, 1.5)
    
    # Sort by predicted DK points (highest first)
    underdog_df = underdog_df.sort_values('predicted_dk_fpts', ascending=False)
    
    # Save the comprehensive dataset
    timestamp = prediction_date.strftime("%Y%m%d")
    output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_player_dataset_from_dk_{timestamp}.csv'
    underdog_df.to_csv(output_file, index=False)
    print(f"💾 Underdog player dataset created from DK predictions: {output_file}")
    
    # Also save a simplified version for quick GUI loading
    simple_columns = ['Name', 'Team', 'Position', 'predicted_dk_fpts', 'batting_avg', 'obp', 
                     'hits_probability', 'runs_probability', 'rbis_probability', 'confidence_score',
                     'underdog_eligible', 'status']
    simple_df = underdog_df[simple_columns]
    simple_output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_players_simple_from_dk.csv'
    simple_df.to_csv(simple_output_file, index=False)
    print(f"💾 Simple Underdog dataset created: {simple_output_file}")
    
    print(f"📊 Created dataset with {len(underdog_df)} players")
    print(f"🎯 Top 5 players by DK points: {underdog_df['Name'].head().tolist()}")
    
    return output_file

def create_comprehensive_player_dataset():
    """
    Create a comprehensive player dataset for Underdog Fantasy analysis.
    This includes realistic MLB player data that can be used in the GUI.
    """
    print("Creating comprehensive player dataset...")
    
    # Expanded player dataset with more realistic MLB players
    comprehensive_players = {
        # AL East
        'Ronald Acuna Jr.': {
            'Team': 'ATL', 'Position': 'OF', 'batting_avg': 0.337, 'at_bats': 4, 'obp': 0.416, 
            'lineup_position_factor': 1.2, 'team_runs_per_game': 5.2, 'rbi_rate': 0.35, 
            'risp_avg': 0.310, 'team_obp': 0.340, 'games_played': 140, 'recent_avg': 0.345,
            'predicted_dk_fpts': 12.5
        },
        'Mike Trout': {
            'Team': 'LAA', 'Position': 'OF', 'batting_avg': 0.283, 'at_bats': 4, 'obp': 0.369, 
            'lineup_position_factor': 1.15, 'team_runs_per_game': 4.8, 'rbi_rate': 0.32, 
            'risp_avg': 0.295, 'team_obp': 0.325, 'games_played': 120, 'recent_avg': 0.290,
            'predicted_dk_fpts': 11.8
        },
        'Mookie Betts': {
            'Team': 'LAD', 'Position': 'OF', 'batting_avg': 0.307, 'at_bats': 4, 'obp': 0.390, 
            'lineup_position_factor': 1.1, 'team_runs_per_game': 5.0, 'rbi_rate': 0.30, 
            'risp_avg': 0.315, 'team_obp': 0.335, 'games_played': 135, 'recent_avg': 0.312,
            'predicted_dk_fpts': 12.1
        },
        'Aaron Judge': {
            'Team': 'NYY', 'Position': 'OF', 'batting_avg': 0.311, 'at_bats': 4, 'obp': 0.404, 
            'lineup_position_factor': 1.25, 'team_runs_per_game': 5.1, 'rbi_rate': 0.38, 
            'risp_avg': 0.325, 'team_obp': 0.342, 'games_played': 148, 'recent_avg': 0.315,
            'predicted_dk_fpts': 13.2
        },
        'Vladimir Guerrero Jr.': {
            'Team': 'TOR', 'Position': '1B', 'batting_avg': 0.264, 'at_bats': 4, 'obp': 0.342, 
            'lineup_position_factor': 1.05, 'team_runs_per_game': 4.5, 'rbi_rate': 0.29, 
            'risp_avg': 0.275, 'team_obp': 0.318, 'games_played': 142, 'recent_avg': 0.268,
            'predicted_dk_fpts': 10.3
        },
        'Rafael Devers': {
            'Team': 'BOS', 'Position': '3B', 'batting_avg': 0.272, 'at_bats': 4, 'obp': 0.354, 
            'lineup_position_factor': 1.08, 'team_runs_per_game': 4.7, 'rbi_rate': 0.31, 
            'risp_avg': 0.285, 'team_obp': 0.325, 'games_played': 138, 'recent_avg': 0.278,
            'predicted_dk_fpts': 10.8
        },
        'Wander Franco': {
            'Team': 'TB', 'Position': 'SS', 'batting_avg': 0.281, 'at_bats': 4, 'obp': 0.347, 
            'lineup_position_factor': 1.0, 'team_runs_per_game': 4.4, 'rbi_rate': 0.25, 
            'risp_avg': 0.292, 'team_obp': 0.320, 'games_played': 125, 'recent_avg': 0.285,
            'predicted_dk_fpts': 9.8
        },
        'Adley Rutschman': {
            'Team': 'BAL', 'Position': 'C', 'batting_avg': 0.254, 'at_bats': 4, 'obp': 0.362, 
            'lineup_position_factor': 0.95, 'team_runs_per_game': 4.6, 'rbi_rate': 0.27, 
            'risp_avg': 0.268, 'team_obp': 0.330, 'games_played': 132, 'recent_avg': 0.258,
            'predicted_dk_fpts': 9.2
        },
        # AL Central
        'Jose Ramirez': {
            'Team': 'CLE', 'Position': '3B', 'batting_avg': 0.280, 'at_bats': 4, 'obp': 0.355, 
            'lineup_position_factor': 1.12, 'team_runs_per_game': 4.8, 'rbi_rate': 0.33, 
            'risp_avg': 0.295, 'team_obp': 0.328, 'games_played': 145, 'recent_avg': 0.283,
            'predicted_dk_fpts': 11.5
        },
        'Byron Buxton': {
            'Team': 'MIN', 'Position': 'OF', 'batting_avg': 0.230, 'at_bats': 4, 'obp': 0.306, 
            'lineup_position_factor': 1.05, 'team_runs_per_game': 4.3, 'rbi_rate': 0.28, 
            'risp_avg': 0.245, 'team_obp': 0.315, 'games_played': 92, 'recent_avg': 0.235,
            'predicted_dk_fpts': 8.7
        },
        'Salvador Perez': {
            'Team': 'KC', 'Position': 'C', 'batting_avg': 0.250, 'at_bats': 4, 'obp': 0.289, 
            'lineup_position_factor': 0.85, 'team_runs_per_game': 4.1, 'rbi_rate': 0.32, 
            'risp_avg': 0.265, 'team_obp': 0.305, 'games_played': 138, 'recent_avg': 0.253,
            'predicted_dk_fpts': 8.9
        },
        # AL West
        'Corey Seager': {
            'Team': 'TEX', 'Position': 'SS', 'batting_avg': 0.327, 'at_bats': 4, 'obp': 0.390, 
            'lineup_position_factor': 1.15, 'team_runs_per_game': 5.0, 'rbi_rate': 0.35, 
            'risp_avg': 0.345, 'team_obp': 0.338, 'games_played': 119, 'recent_avg': 0.332,
            'predicted_dk_fpts': 12.3
        },
        'Julio Rodriguez': {
            'Team': 'SEA', 'Position': 'OF', 'batting_avg': 0.275, 'at_bats': 4, 'obp': 0.331, 
            'lineup_position_factor': 1.08, 'team_runs_per_game': 4.5, 'rbi_rate': 0.28, 
            'risp_avg': 0.285, 'team_obp': 0.318, 'games_played': 155, 'recent_avg': 0.280,
            'predicted_dk_fpts': 10.4
        },
        'Shohei Ohtani': {
            'Team': 'LAD', 'Position': 'DH', 'batting_avg': 0.354, 'at_bats': 4, 'obp': 0.412, 
            'lineup_position_factor': 1.3, 'team_runs_per_game': 5.2, 'rbi_rate': 0.40, 
            'risp_avg': 0.368, 'team_obp': 0.342, 'games_played': 159, 'recent_avg': 0.358,
            'predicted_dk_fpts': 14.8
        },
        # NL East
        'Bryce Harper': {
            'Team': 'PHI', 'Position': '1B', 'batting_avg': 0.293, 'at_bats': 4, 'obp': 0.373, 
            'lineup_position_factor': 1.18, 'team_runs_per_game': 4.9, 'rbi_rate': 0.34, 
            'risp_avg': 0.308, 'team_obp': 0.335, 'games_played': 145, 'recent_avg': 0.297,
            'predicted_dk_fpts': 11.9
        },
        'Juan Soto': {
            'Team': 'NYY', 'Position': 'OF', 'batting_avg': 0.288, 'at_bats': 4, 'obp': 0.421, 
            'lineup_position_factor': 1.22, 'team_runs_per_game': 5.1, 'rbi_rate': 0.31, 
            'risp_avg': 0.295, 'team_obp': 0.342, 'games_played': 157, 'recent_avg': 0.291,
            'predicted_dk_fpts': 12.6
        },
        'Francisco Lindor': {
            'Team': 'NYM', 'Position': 'SS', 'batting_avg': 0.273, 'at_bats': 4, 'obp': 0.344, 
            'lineup_position_factor': 1.05, 'team_runs_per_game': 4.6, 'rbi_rate': 0.29, 
            'risp_avg': 0.285, 'team_obp': 0.322, 'games_played': 152, 'recent_avg': 0.278,
            'predicted_dk_fpts': 10.6
        },
        # Pitchers
        'Gerrit Cole': {
            'Team': 'NYY', 'Position': 'P', 'strikeout_rate': 0.285, 'opponent_contact_rate': 0.75, 
            'expected_innings': 7.0, 'games_started': 30, 'batting_avg': 0.050, 'at_bats': 2,
            'predicted_dk_fpts': 15.2
        },
        'Jacob deGrom': {
            'Team': 'TEX', 'Position': 'P', 'strikeout_rate': 0.305, 'opponent_contact_rate': 0.72, 
            'expected_innings': 6.5, 'games_started': 25, 'batting_avg': 0.045, 'at_bats': 2,
            'predicted_dk_fpts': 16.8
        },
        'Shane Bieber': {
            'Team': 'CLE', 'Position': 'P', 'strikeout_rate': 0.298, 'opponent_contact_rate': 0.73, 
            'expected_innings': 6.8, 'games_started': 28, 'batting_avg': 0.048, 'at_bats': 2,
            'predicted_dk_fpts': 16.1
        },
        'Spencer Strider': {
            'Team': 'ATL', 'Position': 'P', 'strikeout_rate': 0.320, 'opponent_contact_rate': 0.70, 
            'expected_innings': 6.2, 'games_started': 26, 'batting_avg': 0.042, 'at_bats': 2,
            'predicted_dk_fpts': 17.3
        },
        'Sandy Alcantara': {
            'Team': 'MIA', 'Position': 'P', 'strikeout_rate': 0.245, 'opponent_contact_rate': 0.78, 
            'expected_innings': 7.5, 'games_started': 32, 'batting_avg': 0.055, 'at_bats': 2,
            'predicted_dk_fpts': 14.8
        }
    }
    
    # Convert to DataFrame for CSV export
    player_rows = []
    for name, stats in comprehensive_players.items():
        row = {'Name': name}
        row.update(stats)
        # Add some calculated fields for Underdog analysis
        row['Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        row['Status'] = 'Available'
        row['underdog_eligible'] = 'Yes'
        player_rows.append(row)
    
    players_df = pd.DataFrame(player_rows)
    
    # Save the comprehensive player dataset
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_player_dataset_{timestamp}.csv'
    players_df.to_csv(output_file, index=False)
    print(f"💾 Comprehensive player dataset saved to: {output_file}")
    
    # Also save a simplified version for quick loading
    simple_output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_players_simple.csv'
    players_df.to_csv(simple_output_file, index=False)
    print(f"💾 Simple player dataset saved to: {simple_output_file}")
    
    return comprehensive_players, output_file

def run_underdog_analysis_standalone(prediction_date='2025-09-21', bankroll=1000):
    """
    Run standalone Underdog Fantasy analysis with comprehensive player data.
    This demonstrates the Underdog functionality and saves complete player datasets.
    """
    print("="*80)
    print("STANDALONE UNDERDOG FANTASY ANALYSIS")
    print("="*80)
    
    # Initialize Underdog predictor
    underdog_predictor = UnderdogProbabilityPredictor()
    
    # Create comprehensive player dataset
    print("Creating comprehensive player dataset for GUI usage...")
    comprehensive_players, player_dataset_file = create_comprehensive_player_dataset()
    
    # SAVE INDIVIDUAL PLAYER PREDICTIONS AS SEPARATE CSV
    print("Saving individual player predictions for GUI...")
    player_predictions_data = []
    
    for player_name, player_data in comprehensive_players.items():
        # Get individual stat predictions
        predictions = underdog_predictor.predict_individual_stats(player_data)
        
        for stat, prediction in predictions.items():
            player_predictions_data.append({
                'Name': player_name,
                'Team': player_data.get('Team', 'UNK'),
                'Position': player_data.get('Position', 'UNK'),
                'Stat': stat,
                'Probability': prediction['probability'],
                'Expected_Value': prediction['expected_value'],
                'Confidence': prediction['confidence'],
                'DK_Points_Estimate': player_data.get('predicted_dk_fpts', 10.0),
                'Batting_Avg': player_data.get('batting_avg', 0.250),
                'OBP': player_data.get('obp', 0.320),
                'Team_Runs_Per_Game': player_data.get('team_runs_per_game', 4.5),
                'RBI_Rate': player_data.get('rbi_rate', 0.25),
                'Status': 'Available',
                'Date': prediction_date
            })
    
    # Save individual predictions CSV for GUI
    predictions_df = pd.DataFrame(player_predictions_data)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    predictions_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_player_predictions_{timestamp}.csv'
    predictions_df.to_csv(predictions_file, index=False)
    print(f"💾 Individual player predictions saved to: {predictions_file}")
    
    # Use a subset for strategy analysis demonstration (first 8 players)
    sample_players = dict(list(comprehensive_players.items())[:8])
    
    # Generate strategies for all risk profiles
    risk_profiles = ['conservative', 'moderate', 'aggressive']
    all_strategies = {}
    
    for risk_profile in risk_profiles:
        print(f"\n{'-'*25} {risk_profile.upper()} STRATEGY {'-'*25}")
        
        strategy = underdog_predictor.generate_underdog_strategy(
            sample_players, bankroll=bankroll, risk_profile=risk_profile
        )
        all_strategies[risk_profile] = strategy
        
        # Display detailed analysis
        print(f"\nBankroll: ${bankroll}")
        allocation = strategy['bankroll_allocation']
        print(f"Allocation - Power Plays: ${allocation['power_plays_budget']:.0f} | "
              f"Insurance: ${allocation['insurance_budget']:.0f} | "
              f"Parlays: ${allocation['parlay_budget']:.0f} | "
              f"Reserve: ${allocation['reserve_budget']:.0f}")
        print(f"Recommended Unit Size: ${allocation['recommended_unit_size']:.2f}")
        
        print(f"\n🎯 TOP POWER PLAY OPPORTUNITIES:")
        for i, play in enumerate(strategy['power_plays'][:8], 1):
            roi = (play['expected_return'] - 1) * 100
            kelly_pct = play['kelly_fraction'] * 100
            print(f"  {i}. {play['player'][:15]:<15} | {play['stat']:<10} | {play['multiplier']:<3} | "
                  f"Prob: {play['probability']:>5.1%} | ROI: {roi:>6.1f}% | Kelly: {kelly_pct:>4.1f}%")
        
        print(f"\n🔗 OPTIMAL PARLAY COMBINATIONS:")
        for i, parlay in enumerate(strategy['parlay_recommendations'][:5], 1):
            expected_profit = (parlay['expected_value'] - 1) * 100
            print(f"  {i}. {parlay['leg_count']}-Leg Parlay | Prob: {parlay['combined_probability']:>5.1%} | "
                  f"Payout: {parlay['payout_multiplier']:>5.1f}x | Expected Profit: {expected_profit:>6.1f}%")
            for j, leg in enumerate(parlay['legs'][:3], 1):  # Show first 3 legs
                print(f"     {j}) {leg['player'][:12]} {leg['stat']} ({leg['probability']:.1%})")
        
        print(f"\n🛡️  INSURANCE COVERAGE: {len(strategy['insurance_plays'])} plays protected")
        
        # Individual player predictions sample
        print(f"\n📊 INDIVIDUAL STAT PREDICTIONS (Sample):")
        sample_predictions = list(strategy['individual_predictions'].items())[:3]
        for player, predictions in sample_predictions:
            print(f"\n  {player}:")
            for stat, pred in predictions.items():
                expected = pred['expected_value']
                confidence = pred['confidence']
                print(f"    {stat.capitalize():<12}: {pred['probability']:>5.1%} prob | "
                      f"Expected: {expected:>4.2f} | Confidence: {confidence:>5.1%}")
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    comparison_data = []
    for profile, strategy in all_strategies.items():
        power_plays = strategy['power_plays'][:5]
        avg_prob = np.mean([p['probability'] for p in power_plays]) if power_plays else 0
        avg_expected_return = np.mean([p['expected_return'] for p in power_plays]) if power_plays else 0
        max_kelly = max([p['kelly_fraction'] for p in power_plays]) if power_plays else 0
        
        comparison_data.append({
            'Profile': profile.title(),
            'Avg Probability': f"{avg_prob:.1%}",
            'Avg Expected Return': f"{avg_expected_return:.2f}x",
            'Max Kelly %': f"{max_kelly*100:.1f}%",
            'Power Play Budget': f"${strategy['bankroll_allocation']['power_plays_budget']:.0f}",
            'Total Plays': len(power_plays)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comprehensive analysis
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_comprehensive_analysis_{timestamp}.csv'
    
    # Create detailed output DataFrame
    output_data = []
    for profile, strategy in all_strategies.items():
        # Add power plays
        for play in strategy['power_plays']:
            output_data.append({
                'risk_profile': profile,
                'type': 'power_play',
                'player': play['player'],
                'stat': play['stat'],
                'probability': play['probability'],
                'multiplier': play['multiplier'],
                'expected_return': play['expected_return'],
                'kelly_fraction': play['kelly_fraction'],
                'confidence': play['confidence'],
                'risk_score': play['risk_score'],
                'roi_percent': (play['expected_return'] - 1) * 100,
                'legs': 1,
                'leg_details': f"{play['player']} {play['stat']}"
            })
        
        # Add top 10 parlays
        for i, parlay in enumerate(strategy['parlay_recommendations'][:10], 1):
            leg_details = " + ".join([f"{leg['player']} {leg['stat']}" for leg in parlay['legs']])
            output_data.append({
                'risk_profile': profile,
                'type': 'parlay',
                'player': f"Parlay_{i}",
                'stat': f"{parlay['leg_count']}_leg_parlay",
                'probability': parlay['combined_probability'],
                'multiplier': f"{parlay['payout_multiplier']:.1f}x",
                'expected_return': parlay['expected_value'],
                'kelly_fraction': 0,  # Kelly not calculated for parlays
                'confidence': sum([leg['confidence'] for leg in parlay['legs']]) / len(parlay['legs']),
                'risk_score': 1 / parlay['combined_probability'],
                'roi_percent': (parlay['expected_value'] - 1) * 100,
                'legs': parlay['leg_count'],
                'leg_details': leg_details
            })
    
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)
    
    # Save separate detailed parlay file
    parlay_output_file = f'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/7_ANALYSIS/underdog_top_parlays_{timestamp}.csv'
    
    parlay_data = []
    for profile, strategy in all_strategies.items():
        for i, parlay in enumerate(strategy['parlay_recommendations'][:10], 1):
            parlay_row = {
                'risk_profile': profile,
                'parlay_rank': i,
                'leg_count': parlay['leg_count'],
                'combined_probability': parlay['combined_probability'],
                'payout_multiplier': parlay['payout_multiplier'],
                'expected_value': parlay['expected_value'],
                'expected_profit_pct': (parlay['expected_value'] - 1) * 100,
                'risk_reward_ratio': parlay['risk_reward_ratio']
            }
            
            # Add individual leg details
            for j, leg in enumerate(parlay['legs'], 1):
                parlay_row[f'leg_{j}_player'] = leg['player']
                parlay_row[f'leg_{j}_stat'] = leg['stat']
                parlay_row[f'leg_{j}_probability'] = leg['probability']
                parlay_row[f'leg_{j}_confidence'] = leg['confidence']
            
            parlay_data.append(parlay_row)
    
    if parlay_data:
        parlay_df = pd.DataFrame(parlay_data)
        parlay_df.to_csv(parlay_output_file, index=False)
        print(f"\n💾 Top 10 parlays saved to: {parlay_output_file}")
    
    print(f"\n💾 Comprehensive analysis saved to: {output_file}")
    print(f"📈 Total plays analyzed: {len(output_data)}")
    
    if len(output_data) > 0:
        print(f"🎯 Best overall ROI: {output_df['roi_percent'].max():.1f}%")
        print(f"🛡️  Most conservative play: {output_df['probability'].max():.1%} probability")
    else:
        print("⚠️  No qualifying plays found with current thresholds")
        print("💡 Consider adjusting risk tolerance or using more aggressive settings")
    
    # SUMMARY OF CSV FILES CREATED FOR GUI
    print(f"\n📁 CSV FILES CREATED FOR GUI USAGE:")
    print(f"   1. Player Dataset: {player_dataset_file}")
    print(f"   2. Individual Predictions: {predictions_file}")
    print(f"   3. Strategy Analysis: {output_file}")
    print(f"   4. Parlay Details: {parlay_output_file}")
    print(f"\n💡 Use these CSV files in the GUI for team/player selection and analysis!")
    
    return all_strategies

if __name__ == "__main__":
    import sys
    
    print("Underdog Fantasy Prediction System")
    print("Choose analysis type:")
    print("1. Underdog Fantasy Analysis (uses trained underdog models)")
    print("2. Standalone Underdog Analysis (demo with sample data)")
    
    choice = input("Enter choice (1 or 2, or press Enter for demo): ").strip()
    
    if choice == "1":
        # Full analysis with Underdog models
        input_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/1_CORE_TRAINING/battersfinal_dataset_with_features.csv'
        model_file = 'c:/Users/smtes/OneDrive/Documents/draftkings project/MLB_DRAFTKINGS_SYSTEM/3_MODELS/batters_final_ensemble_model_pipeline011.pkl'  # Fallback for DraftKings
        prediction_date = '2025-01-21'  # Updated to current date
        
        print(f"\nRunning Underdog Fantasy analysis for {prediction_date}...")
        predict_unseen_data(input_file, model_file, prediction_date)
        
    else:
        # Standalone Underdog analysis
        bankroll = 1000
        if len(sys.argv) > 1:
            try:
                bankroll = int(sys.argv[1])
            except ValueError:
                print(f"Invalid bankroll amount: {sys.argv[1]}, using default $1000")
        
        print(f"\nRunning standalone Underdog analysis with ${bankroll} bankroll...")
        run_underdog_analysis_standalone(bankroll=bankroll)
    
    print("\nAnalysis complete! Check the 7_ANALYSIS folder for output files.")