#!/usr/bin/env python3
"""
Probability-Enhanced Optimizer
Advanced DFS optimization using probability statistics from CSV projections
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
import re

class ProbabilityEnhancedOptimizer:
    """
    Enhanced optimizer that leverages probability statistics from CSV projections
    to improve lineup construction and risk management
    """
    
    def __init__(self):
        self.probability_columns = {}
        self.confidence_intervals = {}
        self.volatility_metrics = {}
        
    def detect_probability_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect probability-related columns in the CSV
        Returns a dictionary categorizing different types of probability columns
        """
        prob_columns = {
            'threshold_probabilities': [],  # Prob_Over_X columns
            'confidence_intervals': [],     # Lower/Upper bounds
            'uncertainty_metrics': [],      # Std, variance, etc.
            'percentiles': []              # Q10, Q25, Q75, Q90, etc.
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Detect threshold probability columns (Prob_Over_X, Prob_Above_X)
            if any(x in col_lower for x in ['prob_over_', 'prob_above_', 'prob_under_', 'prob_below_']):
                prob_columns['threshold_probabilities'].append(col)
            
            # Detect confidence interval columns
            elif any(x in col_lower for x in ['lower', 'upper', 'ci_']):
                prob_columns['confidence_intervals'].append(col)
            
            # Detect uncertainty metrics
            elif any(x in col_lower for x in ['std', 'variance', 'var', 'volatility']):
                prob_columns['uncertainty_metrics'].append(col)
            
            # Detect percentile columns
            elif re.search(r'[qp]\d+', col_lower) or 'percentile' in col_lower:
                prob_columns['percentiles'].append(col)
        
        self.probability_columns = prob_columns
        
        # Log detected columns
        total_prob_cols = sum(len(cols) for cols in prob_columns.values())
        if total_prob_cols > 0:
            logging.info(f"🎲 Detected {total_prob_cols} probability-related columns:")
            for category, columns in prob_columns.items():
                if columns:
                    logging.info(f"   {category}: {columns}")
        else:
            logging.info("⚠️ No probability columns detected in CSV")
        
        return prob_columns
    
    def clean_probability_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean probability data by converting percentage text to decimals and handling NaN values
        """
        print("🧹 Starting probability data cleaning...")
        df_cleaned = df.copy()
        
        # Detect probability columns directly for cleaning
        print("🧹 Detecting probability columns...")
        prob_columns_detected = self.detect_probability_columns(df)
        prob_cols = prob_columns_detected.get('threshold_probabilities', [])
        print(f"🧹 Found {len(prob_cols)} probability columns: {prob_cols}")
        
        if not prob_cols:
            print("🧹 No probability columns found to clean")
            return df_cleaned
        
        print(f"🧹 Cleaning {len(prob_cols)} probability columns...")
        
        for i, col in enumerate(prob_cols):
            print(f"🧹 Processing column {i+1}/{len(prob_cols)}: {col}")
            if col in df_cleaned.columns:
                # Check original data type and sample
                print(f"   Original dtype: {df_cleaned[col].dtype}")
                print(f"   Sample values: {df_cleaned[col].head(3).tolist()}")
                
                # Convert percentage text to decimal
                if df_cleaned[col].dtype == 'object':
                    # Handle percentage strings like "97.1%" -> 0.971
                    df_cleaned[col] = df_cleaned[col].astype(str).str.replace('%', '').str.replace('nan', '')
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce') / 100.0
                    print(f"   Converted {col} from percentage text to decimal")
                
                # Fill NaN values with 0 for probability columns
                nan_count = df_cleaned[col].isnull().sum()
                if nan_count > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(0.0)
                    print(f"   Filled {nan_count} NaN values in {col} with 0.0")
                
                # Ensure values are between 0 and 1
                df_cleaned[col] = df_cleaned[col].clip(0.0, 1.0)
                print(f"   Final sample values: {df_cleaned[col].head(3).tolist()}")
        
        print("✅ Probability data cleaning complete")
        
        logging.info(f"✅ Probability data cleaning complete")
        return df_cleaned
    
    def extract_threshold_probabilities(self, df: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """
        Extract threshold probabilities for each player
        Returns dict: {player_name: {threshold: probability}}
        """
        threshold_probs = {}
        
        if not self.probability_columns['threshold_probabilities']:
            return threshold_probs
        
        for _, row in df.iterrows():
            player_name = row['Name']
            player_probs = {}
            
            for col in self.probability_columns['threshold_probabilities']:
                # Extract threshold value from column name
                threshold_match = re.search(r'(\d+)', col)
                if threshold_match:
                    threshold = float(threshold_match.group(1))
                    prob_value = row[col]
                    
                    # Since data is now pre-cleaned, just ensure it's a valid probability
                    if pd.notna(prob_value) and 0 <= prob_value <= 1:
                        player_probs[threshold] = prob_value
            
            if player_probs:
                threshold_probs[player_name] = player_probs
        
        return threshold_probs
    
    def extract_probability_thresholds(self, df: pd.DataFrame) -> List[float]:
        """
        Extract the probability thresholds from column names
        """
        thresholds = []
        
        for col in df.columns:
            col_lower = col.lower()
            if 'prob_over_' in col_lower:
                # Extract threshold value (e.g., "Prob_Over_15" -> 15.0)
                try:
                    threshold_str = col.split('_')[-1]
                    threshold = float(threshold_str)
                    thresholds.append(threshold)
                except (ValueError, IndexError):
                    continue
        
        return sorted(thresholds)
    
    def calculate_implied_volatility(self, threshold_probs: Dict[float, float], 
                                   projection: float) -> float:
        """
        Calculate implied volatility from threshold probabilities
        Uses normal distribution approximation to back out volatility
        """
        if not threshold_probs or len(threshold_probs) < 2:
            return projection * 0.3  # Default 30% volatility
        
        volatilities = []
        
        for threshold, prob in threshold_probs.items():
            if 0.01 <= prob <= 0.99:  # Valid probability range
                # Back out implied volatility using normal distribution
                z_score = stats.norm.ppf(1 - prob)  # Z-score for exceedance probability
                if z_score != 0:
                    implied_vol = (threshold - projection) / z_score
                    if implied_vol > 0:  # Only positive volatilities
                        volatilities.append(implied_vol)
        
        if volatilities:
            # Use median volatility to avoid outliers
            return np.median(volatilities)
        else:
            return projection * 0.3
    
    def calculate_expected_utility(self, projection: float, volatility: float, 
                                 risk_aversion: float = 0.5) -> float:
        """
        Calculate expected utility considering both return and risk
        Higher utility = better risk-adjusted value
        """
        return projection - (risk_aversion * volatility**2)
    
    def calculate_kelly_fraction(self, threshold_probs: Dict[float, float], 
                                contest_threshold: float = 15.0) -> float:
        """
        Calculate Kelly Criterion fraction using probability of exceeding contest threshold
        """
        if contest_threshold in threshold_probs:
            win_prob = threshold_probs[contest_threshold]
        else:
            # Interpolate probability for the threshold
            thresholds = sorted(threshold_probs.keys())
            if not thresholds:
                return 0.1  # Default conservative allocation
            
            if contest_threshold <= min(thresholds):
                win_prob = threshold_probs[min(thresholds)]
            elif contest_threshold >= max(thresholds):
                win_prob = threshold_probs[max(thresholds)]
            else:
                # Linear interpolation
                lower_thresh = max([t for t in thresholds if t <= contest_threshold])
                upper_thresh = min([t for t in thresholds if t >= contest_threshold])
                
                if lower_thresh == upper_thresh:
                    win_prob = threshold_probs[lower_thresh]
                else:
                    weight = (contest_threshold - lower_thresh) / (upper_thresh - lower_thresh)
                    win_prob = (threshold_probs[lower_thresh] * (1 - weight) + 
                              threshold_probs[upper_thresh] * weight)
        
        # Kelly formula: f = (bp - q) / b
        # Where b = odds, p = win probability, q = loss probability
        # For DFS, we approximate odds based on player pricing efficiency
        if win_prob <= 0 or win_prob >= 1:
            return 0.1
        
        # Conservative Kelly with 25% fraction for DFS
        kelly_fraction = min(0.25, max(0.01, (win_prob - 0.5) * 0.5))
        return kelly_fraction
    
    def enhance_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance player data with probability-derived metrics
        """
        print("🔍 Starting player data enhancement...")
        
        # Detect probability columns
        print("🔍 Detecting probability columns...")
        prob_cols = self.detect_probability_columns(df)
        self.probability_columns = prob_cols  # Store for later use
        print(f"🔍 Detected columns: {prob_cols}")
        
        if not any(prob_cols.values()):
            print("⚠️ No probability columns found - using basic projections only")
            logging.warning("No probability columns found - using basic projections only")
            return df
        
        # Clean probability data (convert percentages, handle NaN)
        df_cleaned = self.clean_probability_data(df)
        logging.info("🧹 Applied probability data cleaning")
        
        # Extract threshold probabilities from cleaned data
        threshold_probs = self.extract_threshold_probabilities(df_cleaned)
        
        # Calculate enhanced metrics for each player
        enhanced_metrics = []
        
        # Find the projection column
        possible_projection_columns = [
            'Predicted_DK_Points', 'My_Proj', 'ML_Prediction', 'PPG_Projection',
            'Projection', 'Points', 'DK_Points', 'Fantasy_Points'
        ]
        
        projection_col = None
        for col in possible_projection_columns:
            if col in df.columns:
                projection_col = col
                break
        
        if projection_col is None:
            logging.warning("No projection column found, using first numeric column")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            projection_col = numeric_cols[0] if len(numeric_cols) > 0 else 'My_Proj'
        
        for _, row in df.iterrows():
            player_name = row['Name']
            projection = row[projection_col]
            
            player_threshold_probs = threshold_probs.get(player_name, {})
            
            # Calculate implied volatility
            implied_vol = self.calculate_implied_volatility(player_threshold_probs, projection)
            
            # Calculate expected utility
            expected_utility = self.calculate_expected_utility(projection, implied_vol)
            
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_fraction(player_threshold_probs)
            
            # Calculate floor and ceiling estimates
            floor_prob = max([prob for thresh, prob in player_threshold_probs.items() 
                            if thresh <= 10], default=0.5)
            ceiling_prob = max([prob for thresh, prob in player_threshold_probs.items() 
                              if thresh >= 25], default=0.1)
            
            enhanced_metrics.append({
                'Name': player_name,
                'Implied_Volatility': implied_vol,
                'Expected_Utility': expected_utility,
                'Kelly_Fraction': kelly_fraction,
                'Floor_Probability': floor_prob,
                'Ceiling_Probability': ceiling_prob,
                'Risk_Adjusted_Points': projection / max(1, (1 + implied_vol/max(1, projection))),
                'Upside_Potential': ceiling_prob * projection,
                'Safety_Score': floor_prob * max(0, (1 - implied_vol/max(1, projection)))
            })
        
        # Merge enhanced metrics with cleaned original data
        enhanced_df = pd.DataFrame(enhanced_metrics)
        result_df = df_cleaned.merge(enhanced_df, on='Name', how='left')
        
        # Fill NaN values with defaults using the flexible projection column
        result_df = result_df.fillna({
            'Implied_Volatility': result_df[projection_col] * 0.3,
            'Expected_Utility': result_df[projection_col],
            'Kelly_Fraction': 0.1,
            'Floor_Probability': 0.5,
            'Ceiling_Probability': 0.1,
            'Risk_Adjusted_Points': result_df[projection_col],
            'Upside_Potential': result_df[projection_col] * 0.1,
            'Safety_Score': result_df[projection_col] * 0.5
        })
        
        logging.info(f"✅ Enhanced {len(result_df)} players with probability-derived metrics")
        return result_df
    
    def optimize_for_contest_type(self, df: pd.DataFrame, contest_type: str = 'gpp') -> pd.DataFrame:
        """
        Optimize player rankings based on contest type and probability metrics
        """
        # Find the projection column
        possible_projection_columns = [
            'Predicted_DK_Points', 'My_Proj', 'ML_Prediction', 'PPG_Projection',
            'Projection', 'Points', 'DK_Points', 'Fantasy_Points'
        ]
        
        projection_col = None
        for col in possible_projection_columns:
            if col in df.columns:
                projection_col = col
                break
        
        if projection_col is None:
            logging.warning("No projection column found for contest optimization")
            return df
        
        if contest_type.lower() in ['gpp', 'tournament', 'milly']:
            # Tournament strategy: favor ceiling and upside
            df['Contest_Score'] = (
                df[projection_col] * 0.4 +
                df.get('Upside_Potential', df[projection_col] * 0.1) * 0.3 +
                df.get('Expected_Utility', df[projection_col]) * 0.2 +
                df.get('Ceiling_Probability', 0.1) * df[projection_col] * 0.1
            )
        elif contest_type.lower() in ['cash', 'double_up', 'h2h']:
            # Cash game strategy: favor floor and consistency
            df['Contest_Score'] = (
                df.get('Risk_Adjusted_Points', df[projection_col]) * 0.4 +
                df.get('Safety_Score', df[projection_col] * 0.5) * 0.3 +
                df.get('Expected_Utility', df[projection_col]) * 0.2 +
                df.get('Floor_Probability', 0.5) * df[projection_col] * 0.1
            )
        else:
            # Balanced strategy
            df['Contest_Score'] = (
                df.get('Expected_Utility', df[projection_col]) * 0.5 +
                df.get('Risk_Adjusted_Points', df[projection_col]) * 0.3 +
                df.get('Upside_Potential', df[projection_col] * 0.1) * 0.2
            )
        
        return df.sort_values('Contest_Score', ascending=False)
    
    def create_probability_summary(self, df: pd.DataFrame) -> Dict:
        """
        Create a summary of probability metrics for the loaded data
        """
        if 'Implied_Volatility' not in df.columns:
            return {"message": "No probability enhancements applied"}
        
        summary = {
            "total_players": len(df),
            "probability_columns_detected": sum(len(cols) for cols in self.probability_columns.values()),
            "avg_implied_volatility": df['Implied_Volatility'].mean(),
            "avg_kelly_fraction": df['Kelly_Fraction'].mean(),
            "high_floor_players": len(df[df['Floor_Probability'] > 0.7]),
            "high_ceiling_players": len(df[df['Ceiling_Probability'] > 0.3]),
            "top_utility_players": df.nlargest(5, 'Expected_Utility')['Name'].tolist()
        }
        
        return summary


def demo_probability_enhancement():
    """Demo function to test probability enhancement"""
    # Create sample data with probability columns
    sample_data = {
        'Name': ['Player A', 'Player B', 'Player C'],
        'Team': ['NYY', 'BOS', 'LAD'],
        'Pos': ['OF', 'SS', '3B'],
        'Salary': [8000, 9500, 7200],
        'Predicted_DK_Points': [15.2, 18.7, 12.8],
        'Prob_Over_5': ['85%', '92%', '78%'],
        'Prob_Over_10': ['65%', '78%', '55%'],
        'Prob_Over_15': ['35%', '52%', '25%'],
        'Prob_Over_20': ['15%', '28%', '8%'],
        'Prob_Over_25': ['5%', '12%', '2%']
    }
    
    df = pd.DataFrame(sample_data)
    
    optimizer = ProbabilityEnhancedOptimizer()
    enhanced_df = optimizer.enhance_player_data(df)
    
    print("🎲 Probability Enhancement Demo")
    print("="*50)
    print("\nOriginal Data:")
    print(df[['Name', 'Predicted_DK_Points', 'Prob_Over_15']].to_string(index=False))
    
    print("\nEnhanced Data:")
    cols_to_show = ['Name', 'Predicted_DK_Points', 'Implied_Volatility', 'Expected_Utility', 'Kelly_Fraction']
    print(enhanced_df[cols_to_show].round(3).to_string(index=False))
    
    summary = optimizer.create_probability_summary(enhanced_df)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    demo_probability_enhancement()
