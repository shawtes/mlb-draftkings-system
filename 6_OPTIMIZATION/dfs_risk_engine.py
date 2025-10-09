#!/usr/bin/env python3
"""
Advanced DFS Risk Management Engine
Implements financial risk management concepts for DFS optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# OPTIONAL ARCH IMPORT - Position sizing still works without it
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch package not available - using simplified volatility models")

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    sharpe_ratio: float
    volatility: float
    var_95: float  # Value at Risk 95%
    expected_return: float
    max_drawdown: float
    kelly_fraction: float

class DFSRiskEngine:
    """
    Advanced risk management engine for DFS optimization
    Implements financial concepts like Kelly Criterion, GARCH, and portfolio theory
    """
    
    def __init__(self):
        self.player_volatility_cache = {}
        self.correlation_matrix = None
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_kelly_fraction(self, edge: float, variance: float, bankroll: float = 1000) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            edge: Expected advantage (probability of winning - probability of losing)
            variance: Variance of returns
            bankroll: Current bankroll size
            
        Returns:
            Optimal fraction of bankroll to risk
        """
        if variance <= 0:
            return 0.0
            
        # Kelly formula: f = (bp - q) / b
        # Where b = odds, p = win probability, q = lose probability
        # Simplified for DFS: f = edge / variance
        kelly_fraction = edge / variance
        
        # Apply safety constraints
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
        
        logging.debug(f"Kelly fraction calculated: {kelly_fraction:.4f} for edge={edge:.4f}, variance={variance:.4f}")
        return kelly_fraction
    
    def garch_volatility_forecast(self, returns: pd.Series, forecast_horizon: int = 1) -> float:
        """
        GARCH(1,1) volatility forecasting for player performance
        
        Args:
            returns: Historical returns/performance data
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Forecasted volatility
        """
        if not ARCH_AVAILABLE:
            logging.debug("ARCH package not available, using standard deviation")
            return returns.std()
            
        try:
            if len(returns) < 10:  # Need minimum data for GARCH
                return returns.std()
            
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Forecast volatility
            forecast = fitted_model.forecast(horizon=forecast_horizon)
            volatility = np.sqrt(forecast.variance.iloc[-1, 0])
            
            logging.debug(f"GARCH volatility forecast: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logging.warning(f"GARCH modeling failed: {e}, using standard deviation")
            return returns.std()
    
    def build_correlation_matrix(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build player correlation matrix for portfolio optimization
        
        Args:
            player_data: DataFrame with player performance history
            
        Returns:
            Correlation matrix
        """
        try:
            # Create returns matrix (players x time periods)
            returns_matrix = player_data.pivot_table(
                index='date', 
                columns='player_name', 
                values='fantasy_points'
            ).pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_matrix.corr()
            
            # Handle NaN values
            correlation_matrix = correlation_matrix.fillna(0)
            
            # Ensure positive semi-definite
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.001)  # Floor eigenvalues
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            self.correlation_matrix = correlation_matrix
            logging.info(f"Built correlation matrix for {len(correlation_matrix)} players")
            return correlation_matrix
            
        except Exception as e:
            logging.error(f"Error building correlation matrix: {e}")
            # Return identity matrix as fallback
            n_players = len(player_data['player_name'].unique())
            return pd.DataFrame(np.eye(n_players))
    
    def calculate_portfolio_risk(self, weights: np.ndarray, covariance_matrix: np.ndarray) -> float:
        """
        Calculate portfolio risk (volatility)
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Player covariance matrix
            
        Returns:
            Portfolio volatility
        """
        portfolio_variance = weights.T @ covariance_matrix @ weights
        return np.sqrt(portfolio_variance)
    
    def calculate_sharpe_ratio(self, returns: float, volatility: float) -> float:
        """
        Calculate risk-adjusted Sharpe ratio
        
        Args:
            returns: Expected returns
            volatility: Portfolio volatility
            
        Returns:
            Sharpe ratio
        """
        if volatility <= 0:
            return 0.0
        return (returns - self.risk_free_rate) / volatility
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            cumulative_returns: Cumulative returns series
            
        Returns:
            Maximum drawdown percentage
        """
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    def optimize_lineup_allocation(self, lineups: List[Dict], target_risk: float = None) -> Dict:
        """
        Optimize allocation across multiple lineups using Modern Portfolio Theory
        
        Args:
            lineups: List of lineup dictionaries with expected returns and risks
            target_risk: Target risk level (if None, maximize Sharpe ratio)
            
        Returns:
            Optimal allocation weights and metrics
        """
        n_lineups = len(lineups)
        if n_lineups == 0:
            return {}
        
        # Extract expected returns and build covariance matrix
        expected_returns = np.array([lineup.get('expected_return', 0) for lineup in lineups])
        
        # Simple risk model if no correlation data available
        variances = np.array([lineup.get('variance', 1) for lineup in lineups])
        covariance_matrix = np.diag(variances)
        
        # Optimization objective function
        def objective(weights):
            portfolio_return = weights @ expected_returns
            portfolio_risk = self.calculate_portfolio_risk(weights, covariance_matrix)
            
            if target_risk is not None:
                # Minimize tracking error to target risk
                return (portfolio_risk - target_risk) ** 2
            else:
                # Maximize Sharpe ratio (minimize negative Sharpe)
                sharpe = self.calculate_sharpe_ratio(portfolio_return, portfolio_risk)
                return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (no short selling, max 50% in any single lineup)
        bounds = [(0, 0.5) for _ in range(n_lineups)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_lineups) / n_lineups
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = optimal_weights @ expected_returns
                portfolio_risk = self.calculate_portfolio_risk(optimal_weights, covariance_matrix)
                sharpe = self.calculate_sharpe_ratio(portfolio_return, portfolio_risk)
                
                return {
                    'weights': optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_risk,
                    'sharpe_ratio': sharpe,
                    'success': True
                }
            else:
                logging.warning("Portfolio optimization failed, using equal weights")
                return self._equal_weight_fallback(lineups)
                
        except Exception as e:
            logging.error(f"Portfolio optimization error: {e}")
            return self._equal_weight_fallback(lineups)
    
    def _equal_weight_fallback(self, lineups: List[Dict]) -> Dict:
        """Fallback to equal weights if optimization fails"""
        n_lineups = len(lineups)
        weights = np.ones(n_lineups) / n_lineups
        expected_returns = np.array([lineup.get('expected_return', 0) for lineup in lineups])
        
        return {
            'weights': weights,
            'expected_return': weights @ expected_returns,
            'volatility': 0.1,  # Default risk
            'sharpe_ratio': 0.0,
            'success': False
        }
    
    def dynamic_position_sizing(self, bankroll: float, current_edge: float, 
                              recent_volatility: float, drawdown_factor: float = 1.0) -> Dict:
        """
        Dynamic position sizing based on bankroll, edge, and market conditions
        
        Args:
            bankroll: Current bankroll
            current_edge: Current estimated edge
            recent_volatility: Recent performance volatility
            drawdown_factor: Reduction factor during drawdowns (0-1)
            
        Returns:
            Position sizing recommendations
        """
        # Base Kelly calculation
        base_kelly = self.calculate_kelly_fraction(current_edge, recent_volatility**2)
        
        # Apply drawdown adjustment
        adjusted_kelly = base_kelly * drawdown_factor
        
        # Calculate position sizes
        optimal_position_size = bankroll * adjusted_kelly
        
        # Conservative and aggressive alternatives
        conservative_size = optimal_position_size * 0.5
        aggressive_size = optimal_position_size * 1.5
        
        return {
            'optimal_position_size': optimal_position_size,
            'conservative_size': conservative_size,
            'aggressive_size': aggressive_size,
            'kelly_fraction': adjusted_kelly,
            'base_kelly': base_kelly,
            'drawdown_adjustment': drawdown_factor,
            'recommended_lineups': max(1, int(optimal_position_size / 25))  # Assuming $25 per lineup
        }
    
    def calculate_lineup_risk_metrics(self, lineup_data: Dict, 
                                    historical_performance: pd.DataFrame = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a lineup
        
        Args:
            lineup_data: Dictionary containing lineup information
            historical_performance: Historical performance data for backtesting
            
        Returns:
            RiskMetrics object with all calculated metrics
        """
        try:
            # Extract basic metrics
            expected_return = lineup_data.get('expected_points', 0)
            projected_salary = lineup_data.get('total_salary', 50000)
            
            # Calculate volatility from player projections
            if 'players' in lineup_data:
                player_variances = [p.get('variance', 1) for p in lineup_data['players']]
                portfolio_variance = np.sum(player_variances)  # Assuming independence
                volatility = np.sqrt(portfolio_variance)
            else:
                volatility = expected_return * 0.3  # Default 30% volatility
            
            # Calculate Sharpe ratio
            sharpe_ratio = self.calculate_sharpe_ratio(expected_return, volatility)
            
            # Calculate VaR (95% confidence)
            var_95 = expected_return - 1.645 * volatility  # Normal distribution approximation
            
            # Kelly fraction
            edge = expected_return / projected_salary if projected_salary > 0 else 0
            kelly_fraction = self.calculate_kelly_fraction(edge, volatility**2)
            
            # Max drawdown (simplified calculation)
            max_drawdown = -2 * volatility  # Approximation
            
            return RiskMetrics(
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                var_95=var_95,
                expected_return=expected_return,
                max_drawdown=max_drawdown,
                kelly_fraction=kelly_fraction
            )
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0)

class DFSBankrollManager:
    """
    Bankroll management for DFS optimization
    """
    
    def __init__(self, initial_bankroll: float = 1000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.risk_engine = DFSRiskEngine()
        
    def update_bankroll(self, pnl: float):
        """Update bankroll with profit/loss"""
        self.current_bankroll += pnl
        
    def get_drawdown_factor(self) -> float:
        """Calculate drawdown factor for position sizing"""
        drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if drawdown <= 0:
            return 1.0  # No drawdown
        elif drawdown < 0.1:
            return 1.0  # Less than 10% drawdown
        elif drawdown < 0.2:
            return 0.8  # 10-20% drawdown
        elif drawdown < 0.3:
            return 0.6  # 20-30% drawdown
        else:
            return 0.4  # More than 30% drawdown
    
    def calculate_position_limits(self, edge: float, volatility: float) -> Dict:
        """Calculate position limits based on current bankroll state"""
        drawdown_factor = self.get_drawdown_factor()
        
        return self.risk_engine.dynamic_position_sizing(
            self.current_bankroll, edge, volatility, drawdown_factor
        )

# Example usage functions
def demonstrate_risk_engine():
    """Demonstrate the risk engine functionality"""
    print("🔥 DFS Risk Engine Demonstration")
    print("=" * 50)
    
    # Initialize risk engine
    risk_engine = DFSRiskEngine()
    
    # Example lineup data
    lineup_data = {
        'expected_points': 150,
        'total_salary': 49500,
        'players': [
            {'name': 'Player1', 'variance': 25},
            {'name': 'Player2', 'variance': 30},
            # ... more players
        ]
    }
    
    # Calculate risk metrics
    risk_metrics = risk_engine.calculate_lineup_risk_metrics(lineup_data)
    
    print(f"📊 Risk Metrics:")
    print(f"   Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
    print(f"   Volatility: {risk_metrics.volatility:.2f}")
    print(f"   VaR (95%): {risk_metrics.var_95:.2f}")
    print(f"   Kelly Fraction: {risk_metrics.kelly_fraction:.3f}")
    
    # Bankroll management
    bankroll_manager = DFSBankrollManager(1000)
    position_sizing = bankroll_manager.calculate_position_limits(0.05, 0.3)
    
    print(f"\n💰 Position Sizing:")
    print(f"   Recommended Lineups: {position_sizing['recommended_lineups']}")
    print(f"   Optimal Position: ${position_sizing['optimal_position_size']:.2f}")
    print(f"   Kelly Fraction: {position_sizing['kelly_fraction']:.3f}")

if __name__ == "__main__":
    demonstrate_risk_engine()
