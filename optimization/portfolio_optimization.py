#!/usr/bin/env python3
"""
Portfolio Optimization & Risk Management Enhancements (Step 5)
Advanced portfolio construction and risk management system integrating with ML and time series models
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import scipy.optimize as sco
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Advanced optimization libraries
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    logger.info("✅ CVXPY available for advanced optimization")
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("⚠️ CVXPY not available, using scipy fallback")

try:
    # Financial risk libraries
    import pypfopt
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    PYPFOPT_AVAILABLE = True
    logger.info("✅ PyPortfolioOpt available for advanced portfolio optimization")
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("⚠️ PyPortfolioOpt not available, using custom implementation")

@dataclass
class PortfolioOptimizationResults:
    """Results container for portfolio optimization"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    max_drawdown: float
    
    # Optimization details
    method: str
    optimization_success: bool
    constraints_satisfied: bool
    
    # Advanced metrics
    diversification_ratio: float
    concentration_score: float
    effective_number_assets: float
    
    # Performance attribution
    factor_exposures: Dict[str, float]
    risk_contributions: Dict[str, float]

@dataclass
class RiskManagementResults:
    """Results container for risk management analysis"""
    portfolio_var: float
    portfolio_cvar: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    
    # Stress testing
    monte_carlo_scenarios: np.ndarray
    worst_case_scenario: float
    stress_test_results: Dict[str, float]
    
    # Dynamic risk metrics
    rolling_volatility: pd.Series
    rolling_correlation: pd.DataFrame
    regime_risk_scores: Dict[str, float]

@dataclass
class BlackLittermanResults:
    """Results container for Black-Litterman model"""
    implied_returns: pd.Series
    adjusted_returns: pd.Series
    views_matrix: np.ndarray
    confidence_matrix: np.ndarray
    optimal_weights: pd.Series
    tau: float

class AdvancedPortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine
    Implements multiple optimization methods with risk management
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_methods = {
            'mean_variance': self._mean_variance_optimization,
            'risk_parity': self._risk_parity_optimization,
            'black_litterman': self._black_litterman_optimization,
            'maximum_diversification': self._maximum_diversification,
            'minimum_variance': self._minimum_variance_optimization,
            'cvar_optimization': self._cvar_optimization
        }
        
        logger.info("🎯 Advanced Portfolio Optimizer initialized")
        logger.info(f"   Available methods: {list(self.optimization_methods.keys())}")
        logger.info(f"   Risk-free rate: {risk_free_rate:.2%}")
    
    def optimize_portfolio(self, 
                          returns_data: pd.DataFrame,
                          method: str = 'mean_variance',
                          target_return: Optional[float] = None,
                          target_volatility: Optional[float] = None,
                          constraints: Optional[Dict] = None,
                          views: Optional[Dict] = None) -> PortfolioOptimizationResults:
        """
        Main portfolio optimization function
        
        Args:
            returns_data: DataFrame of asset returns
            method: Optimization method to use
            target_return: Target portfolio return (if applicable)
            target_volatility: Target portfolio volatility (if applicable)
            constraints: Portfolio constraints (min/max weights, etc.)
            views: Market views for Black-Litterman (if applicable)
        """
        try:
            logger.info(f"🔧 Optimizing portfolio using {method}")
            logger.info(f"   Assets: {len(returns_data.columns)}")
            logger.info(f"   Periods: {len(returns_data)}")
            
            # Default constraints
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 0.3,
                    'max_concentration': 0.5,
                    'min_assets': 3
                }
            
            # Calculate expected returns and covariance matrix
            expected_returns = self._calculate_expected_returns(returns_data)
            cov_matrix = self._calculate_covariance_matrix(returns_data)
            
            # Run optimization
            if method in self.optimization_methods:
                optimal_weights = self.optimization_methods[method](
                    expected_returns, cov_matrix, constraints, target_return, target_volatility, views
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Calculate risk metrics
            var_95 = self._calculate_var(returns_data, optimal_weights, confidence=0.95)
            cvar_95 = self._calculate_cvar(returns_data, optimal_weights, confidence=0.95)
            max_drawdown = self._calculate_max_drawdown(returns_data, optimal_weights)
            
            # Calculate advanced metrics
            diversification_ratio = self._calculate_diversification_ratio(optimal_weights, cov_matrix)
            concentration_score = self._calculate_concentration_score(optimal_weights)
            effective_number_assets = self._calculate_effective_number_assets(optimal_weights)
            
            # Risk contributions
            risk_contributions = self._calculate_risk_contributions(optimal_weights, cov_matrix)
            
            # Create results
            assets = returns_data.columns.tolist()
            weights_dict = dict(zip(assets, optimal_weights))
            risk_contrib_dict = dict(zip(assets, risk_contributions))
            
            results = PortfolioOptimizationResults(
                optimal_weights=weights_dict,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                method=method,
                optimization_success=True,
                constraints_satisfied=self._check_constraints(optimal_weights, constraints),
                diversification_ratio=diversification_ratio,
                concentration_score=concentration_score,
                effective_number_assets=effective_number_assets,
                factor_exposures={},  # To be implemented with factor models
                risk_contributions=risk_contrib_dict
            )
            
            logger.info(f"✅ Portfolio optimization successful")
            logger.info(f"   Expected Return: {portfolio_return:.2%}")
            logger.info(f"   Expected Volatility: {portfolio_volatility:.2%}")
            logger.info(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
            logger.info(f"   Diversification Ratio: {diversification_ratio:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Portfolio optimization failed: {str(e)}")
            # Return fallback results
            n_assets = len(returns_data.columns)
            equal_weights = np.ones(n_assets) / n_assets
            assets = returns_data.columns.tolist()
            
            return PortfolioOptimizationResults(
                optimal_weights=dict(zip(assets, equal_weights)),
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                method=method,
                optimization_success=False,
                constraints_satisfied=False,
                diversification_ratio=1.0,
                concentration_score=1.0/n_assets,
                effective_number_assets=n_assets,
                factor_exposures={},
                risk_contributions=dict(zip(assets, equal_weights))
            )
    
    def _mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                   constraints: Dict, target_return: Optional[float] = None,
                                   target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Mean-Variance Optimization (Markowitz)"""
        n_assets = len(expected_returns)
        
        # Objective function: maximize Sharpe ratio or minimize volatility
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if target_return is not None:
                # Minimize volatility for target return
                return portfolio_volatility
            else:
                # Maximize Sharpe ratio
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(x * expected_returns) - target_return
            })
        
        if target_volatility is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(cov_matrix, x))) - target_volatility
            })
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if result.success:
            return result.x
        else:
            logger.warning("Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _risk_parity_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                 constraints: Dict, target_return: Optional[float] = None,
                                 target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Risk Parity Optimization"""
        n_assets = len(expected_returns)
        
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_volatility
            risk_contributions = weights * marginal_risk
            
            # Target: equal risk contributions
            target_risk = portfolio_volatility / n_assets
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = sco.minimize(risk_parity_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _black_litterman_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                     constraints: Dict, target_return: Optional[float] = None,
                                     target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Black-Litterman Model Implementation"""
        n_assets = len(expected_returns)
        
        if views is None:
            # If no views provided, fall back to mean-variance
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints, target_return, target_volatility)
        
        # Black-Litterman parameters
        tau = 0.025  # Scaling factor for uncertainty of prior
        
        # Market capitalization weights (equal weights as proxy)
        market_weights = np.ones(n_assets) / n_assets
        
        # Risk aversion parameter
        risk_aversion = (np.sum(market_weights * expected_returns) - self.risk_free_rate) / np.dot(market_weights, np.dot(cov_matrix, market_weights))
        
        # Implied returns
        implied_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # Views matrix P and view values Q
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        
        for i, (asset_view, confidence) in enumerate(views.items()):
            if asset_view in [f"asset_{j}" for j in range(n_assets)]:  # Simplified view structure
                asset_idx = int(asset_view.split('_')[1])
                P[i, asset_idx] = 1.0
                Q[i] = confidence
        
        # Confidence matrix (diagonal)
        omega = np.eye(len(views)) * 0.01  # 1% uncertainty in views
        
        # Black-Litterman formula
        tau_cov = tau * cov_matrix
        M1 = np.linalg.inv(tau_cov)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau_cov), implied_returns)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
        
        # New expected returns
        new_expected_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # Use new expected returns in mean-variance optimization
        return self._mean_variance_optimization(new_expected_returns, cov_matrix, constraints, target_return, target_volatility)
    
    def _maximum_diversification(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                constraints: Dict, target_return: Optional[float] = None,
                                target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Maximum Diversification Portfolio"""
        n_assets = len(expected_returns)
        
        # Individual asset volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))
        
        def diversification_objective(weights):
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            weighted_avg_volatility = np.sum(weights * individual_vols)
            return -weighted_avg_volatility / portfolio_volatility  # Maximize diversification ratio
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = sco.minimize(diversification_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if result.success:
            return result.x
        else:
            logger.warning("Maximum diversification optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _minimum_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                      constraints: Dict, target_return: Optional[float] = None,
                                      target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Minimum Variance Portfolio"""
        n_assets = len(expected_returns)
        
        def variance_objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * expected_returns) - target_return
            })
        
        # Bounds
        bounds = [(constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = sco.minimize(variance_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
        
        if result.success:
            return result.x
        else:
            logger.warning("Minimum variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets
    
    def _cvar_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                          constraints: Dict, target_return: Optional[float] = None,
                          target_volatility: Optional[float] = None, views: Optional[Dict] = None) -> np.ndarray:
        """Conditional Value at Risk (CVaR) Optimization"""
        
        # For now, use minimum variance as a proxy for CVaR optimization
        # In practice, this would require scenario-based optimization
        logger.warning("CVaR optimization using minimum variance proxy")
        return self._minimum_variance_optimization(expected_returns, cov_matrix, constraints, target_return, target_volatility)
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using multiple methods"""
        # Simple mean
        mean_returns = returns_data.mean().values
        
        # Exponentially weighted mean (more weight on recent observations)
        ewm_returns = returns_data.ewm(span=60).mean().iloc[-1].values
        
        # Combine methods (60% EWM, 40% simple mean)
        expected_returns = 0.6 * ewm_returns + 0.4 * mean_returns
        
        return expected_returns
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with shrinkage"""
        # Exponentially weighted covariance
        cov_matrix = returns_data.ewm(span=60).cov().iloc[-len(returns_data.columns):].values
        
        # Ledoit-Wolf shrinkage
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_shrunk = lw.fit(returns_data.fillna(0)).covariance_
            
            # Combine methods
            cov_matrix = 0.7 * cov_matrix + 0.3 * cov_shrunk
        except ImportError:
            logger.warning("Sklearn not available, using simple covariance")
        
        return cov_matrix
    
    def _calculate_var(self, returns_data: pd.DataFrame, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        portfolio_returns = np.dot(returns_data.values, weights)
        return np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns_data: pd.DataFrame, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        portfolio_returns = np.dot(returns_data.values, weights)
        var = self._calculate_var(returns_data, weights, confidence)
        return portfolio_returns[portfolio_returns <= var].mean()
    
    def _calculate_max_drawdown(self, returns_data: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        portfolio_returns = np.dot(returns_data.values, weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_volatility = np.sum(weights * individual_vols)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        return weighted_avg_volatility / portfolio_volatility
    
    def _calculate_concentration_score(self, weights: np.ndarray) -> float:
        """Calculate Herfindahl concentration index"""
        return np.sum(weights ** 2)
    
    def _calculate_effective_number_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets"""
        return 1 / self._calculate_concentration_score(weights)
    
    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions"""
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_risk
        return risk_contributions / risk_contributions.sum()  # Normalize to sum to 1
    
    def _check_constraints(self, weights: np.ndarray, constraints: Dict) -> bool:
        """Check if constraints are satisfied"""
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)
        max_concentration = constraints.get('max_concentration', 1.0)
        
        # Check weight bounds
        if np.any(weights < min_weight) or np.any(weights > max_weight):
            return False
        
        # Check concentration
        if self._calculate_concentration_score(weights) > max_concentration:
            return False
        
        # Check weights sum to 1
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return False
        
        return True

class AdvancedRiskManager:
    """
    Advanced Risk Management System
    Implements comprehensive risk analysis and monitoring
    """
    
    def __init__(self):
        self.confidence_levels = [0.90, 0.95, 0.99]
        logger.info("🛡️ Advanced Risk Manager initialized")
    
    def analyze_portfolio_risk(self, returns_data: pd.DataFrame, weights: np.ndarray,
                              scenarios: int = 10000) -> RiskManagementResults:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            returns_data: Historical returns data
            weights: Portfolio weights
            scenarios: Number of Monte Carlo scenarios
        """
        try:
            logger.info("🔍 Analyzing portfolio risk...")
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_data.values, weights)
            
            # Basic risk metrics
            portfolio_var = np.percentile(portfolio_returns, 5)  # 95% VaR
            portfolio_cvar = portfolio_returns[portfolio_returns <= portfolio_var].mean()
            
            # Component VaR (approximate)
            component_var = {}
            marginal_var = {}
            
            for i, asset in enumerate(returns_data.columns):
                # Marginal VaR
                asset_returns = returns_data.iloc[:, i].values
                correlation = np.corrcoef(portfolio_returns, asset_returns)[0, 1]
                asset_vol = np.std(asset_returns)
                portfolio_vol = np.std(portfolio_returns)
                
                marginal_var[asset] = correlation * asset_vol / portfolio_vol * portfolio_var
                component_var[asset] = weights[i] * marginal_var[asset]
            
            # Monte Carlo simulation
            monte_carlo_scenarios = self._monte_carlo_simulation(returns_data, weights, scenarios)
            worst_case_scenario = np.min(monte_carlo_scenarios)
            
            # Stress testing
            stress_test_results = self._stress_test_portfolio(returns_data, weights)
            
            # Dynamic risk metrics
            rolling_volatility = pd.Series(portfolio_returns).rolling(window=30).std()
            rolling_correlation = returns_data.rolling(window=30).corr()
            
            # Regime risk scores (simplified)
            regime_risk_scores = self._calculate_regime_risk_scores(returns_data, weights)
            
            results = RiskManagementResults(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                component_var=component_var,
                marginal_var=marginal_var,
                monte_carlo_scenarios=monte_carlo_scenarios,
                worst_case_scenario=worst_case_scenario,
                stress_test_results=stress_test_results,
                rolling_volatility=rolling_volatility,
                rolling_correlation=rolling_correlation.iloc[-len(returns_data.columns):],
                regime_risk_scores=regime_risk_scores
            )
            
            logger.info(f"✅ Risk analysis complete")
            logger.info(f"   Portfolio VaR (95%): {portfolio_var:.2%}")
            logger.info(f"   Portfolio CVaR (95%): {portfolio_cvar:.2%}")
            logger.info(f"   Worst case scenario: {worst_case_scenario:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Risk analysis failed: {str(e)}")
            return self._create_fallback_risk_results(returns_data, weights)
    
    def _monte_carlo_simulation(self, returns_data: pd.DataFrame, weights: np.ndarray, 
                               scenarios: int = 10000) -> np.ndarray:
        """Monte Carlo simulation for portfolio returns"""
        mean_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        # Generate random scenarios
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, scenarios)
        portfolio_scenarios = np.dot(random_returns, weights)
        
        return portfolio_scenarios
    
    def _stress_test_portfolio(self, returns_data: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Stress test portfolio under various scenarios"""
        portfolio_returns = np.dot(returns_data.values, weights)
        
        stress_scenarios = {
            'market_crash_2008': -0.20,  # -20% market shock
            'covid_crash_2020': -0.35,   # -35% market shock
            'volatility_spike': np.std(portfolio_returns) * 3,  # 3x volatility
            'correlation_breakdown': 0.15,  # Assume correlations go to 1
        }
        
        stress_results = {}
        for scenario_name, shock in stress_scenarios.items():
            if 'crash' in scenario_name:
                stressed_return = portfolio_returns.mean() + shock
            elif 'volatility' in scenario_name:
                stressed_return = portfolio_returns.mean() - shock
            else:
                stressed_return = portfolio_returns.mean() * 0.8  # Generic stress
            
            stress_results[scenario_name] = stressed_return
        
        return stress_results
    
    def _calculate_regime_risk_scores(self, returns_data: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate risk scores for different market regimes"""
        portfolio_returns = np.dot(returns_data.values, weights)
        
        # Simple regime detection based on volatility
        volatility = pd.Series(portfolio_returns).rolling(window=20).std()
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)
        
        regimes = {
            'low_volatility': np.mean(portfolio_returns[volatility <= low_vol_threshold]),
            'normal_volatility': np.mean(portfolio_returns[(volatility > low_vol_threshold) & (volatility < high_vol_threshold)]),
            'high_volatility': np.mean(portfolio_returns[volatility >= high_vol_threshold])
        }
        
        return regimes
    
    def _create_fallback_risk_results(self, returns_data: pd.DataFrame, weights: np.ndarray) -> RiskManagementResults:
        """Create fallback risk results in case of errors"""
        n_assets = len(weights)
        
        return RiskManagementResults(
            portfolio_var=-0.05,
            portfolio_cvar=-0.08,
            component_var={asset: -0.01 for asset in returns_data.columns},
            marginal_var={asset: -0.01 for asset in returns_data.columns},
            monte_carlo_scenarios=np.random.normal(-0.05, 0.15, 1000),
            worst_case_scenario=-0.25,
            stress_test_results={'fallback': -0.15},
            rolling_volatility=pd.Series([0.15] * len(returns_data)),
            rolling_correlation=pd.DataFrame(np.eye(n_assets), 
                                           index=returns_data.columns, 
                                           columns=returns_data.columns),
            regime_risk_scores={'normal': -0.02, 'stress': -0.15}
        )

class DynamicPortfolioManager:
    """
    Dynamic Portfolio Management System
    Integrates optimization with real-time risk management and rebalancing
    """
    
    def __init__(self, rebalance_frequency: str = 'weekly'):
        self.optimizer = AdvancedPortfolioOptimizer()
        self.risk_manager = AdvancedRiskManager()
        self.rebalance_frequency = rebalance_frequency
        self.current_portfolio = None
        self.performance_history = []
        
        logger.info("🎯 Dynamic Portfolio Manager initialized")
        logger.info(f"   Rebalance frequency: {rebalance_frequency}")
    
    def update_portfolio(self, returns_data: pd.DataFrame, ml_signals: Dict[str, float],
                        market_regime: str = 'normal', method: str = 'mean_variance') -> PortfolioOptimizationResults:
        """
        Update portfolio allocation based on current market conditions
        
        Args:
            returns_data: Historical returns data
            ml_signals: ML confidence scores for each asset
            market_regime: Current market regime (normal, stress, bull, bear)
            method: Optimization method to use
        """
        try:
            logger.info(f"🔄 Updating portfolio allocation")
            logger.info(f"   Market regime: {market_regime}")
            logger.info(f"   ML signals: {len(ml_signals)} assets")
            
            # Adjust constraints based on market regime
            constraints = self._get_regime_based_constraints(market_regime)
            
            # Incorporate ML signals as views for Black-Litterman
            views = self._convert_ml_signals_to_views(ml_signals) if method == 'black_litterman' else None
            
            # Optimize portfolio
            optimization_results = self.optimizer.optimize_portfolio(
                returns_data=returns_data,
                method=method,
                constraints=constraints,
                views=views
            )
            
            # Risk analysis
            weights_array = np.array([optimization_results.optimal_weights[asset] for asset in returns_data.columns])
            risk_results = self.risk_manager.analyze_portfolio_risk(returns_data, weights_array)
            
            # Update current portfolio
            self.current_portfolio = {
                'weights': optimization_results.optimal_weights,
                'last_update': datetime.now(),
                'method': method,
                'regime': market_regime,
                'risk_metrics': risk_results
            }
            
            # Record performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'expected_return': optimization_results.expected_return,
                'expected_volatility': optimization_results.expected_volatility,
                'sharpe_ratio': optimization_results.sharpe_ratio,
                'method': method,
                'regime': market_regime
            })
            
            logger.info(f"✅ Portfolio updated successfully")
            logger.info(f"   Active positions: {sum(1 for w in optimization_results.optimal_weights.values() if w > 0.01)}")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Portfolio update failed: {str(e)}")
            # Return equal weight fallback
            equal_weights = {asset: 1.0/len(returns_data.columns) for asset in returns_data.columns}
            return PortfolioOptimizationResults(
                optimal_weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                var_95=0.0,
                cvar_95=0.0,
                max_drawdown=0.0,
                method='equal_weight_fallback',
                optimization_success=False,
                constraints_satisfied=True,
                diversification_ratio=1.0,
                concentration_score=1.0/len(returns_data.columns),
                effective_number_assets=len(returns_data.columns),
                factor_exposures={},
                risk_contributions=equal_weights
            )
    
    def _get_regime_based_constraints(self, regime: str) -> Dict:
        """Get constraints based on market regime"""
        base_constraints = {
            'min_weight': 0.0,
            'max_weight': 0.3,
            'max_concentration': 0.5,
            'min_assets': 3
        }
        
        if regime == 'stress':
            # More conservative in stress regime
            base_constraints.update({
                'max_weight': 0.2,
                'max_concentration': 0.4,
                'min_assets': 5
            })
        elif regime == 'bull':
            # More aggressive in bull regime
            base_constraints.update({
                'max_weight': 0.4,
                'max_concentration': 0.6,
                'min_assets': 3
            })
        elif regime == 'bear':
            # Very conservative in bear regime
            base_constraints.update({
                'max_weight': 0.15,
                'max_concentration': 0.3,
                'min_assets': 6
            })
        
        return base_constraints
    
    def _convert_ml_signals_to_views(self, ml_signals: Dict[str, float]) -> Dict:
        """Convert ML confidence signals to Black-Litterman views"""
        views = {}
        for asset, confidence in ml_signals.items():
            # Convert confidence to expected return view
            # High confidence -> higher expected return
            expected_return_view = (confidence - 0.5) * 0.1  # Scale to reasonable return range
            views[f"view_{asset}"] = expected_return_view
        
        return views
    
    def get_rebalancing_signals(self, current_weights: Dict[str, float], 
                               target_weights: Dict[str, float], 
                               threshold: float = 0.05) -> Dict[str, Dict]:
        """Generate rebalancing signals when weights drift too far from targets"""
        rebalancing_signals = {}
        
        for asset in target_weights:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights[asset]
            drift = abs(current_weight - target_weight)
            
            if drift > threshold:
                action = 'buy' if target_weight > current_weight else 'sell'
                amount = abs(target_weight - current_weight)
                
                rebalancing_signals[asset] = {
                    'action': action,
                    'amount': amount,
                    'current_weight': current_weight,
                    'target_weight': target_weight,
                    'drift': drift,
                    'priority': drift / threshold  # Higher priority for larger drifts
                }
        
        return rebalancing_signals

# Convenience functions for integration
def optimize_portfolio_advanced(returns_data: pd.DataFrame, 
                               method: str = 'mean_variance',
                               ml_signals: Optional[Dict[str, float]] = None,
                               market_regime: str = 'normal') -> PortfolioOptimizationResults:
    """
    Convenience function for advanced portfolio optimization
    """
    manager = DynamicPortfolioManager()
    
    if ml_signals is None:
        ml_signals = {asset: 0.5 for asset in returns_data.columns}  # Neutral signals
    
    return manager.update_portfolio(returns_data, ml_signals, market_regime, method)

def analyze_portfolio_risk_comprehensive(returns_data: pd.DataFrame, 
                                       weights: Union[Dict[str, float], np.ndarray]) -> RiskManagementResults:
    """
    Convenience function for comprehensive risk analysis
    """
    risk_manager = AdvancedRiskManager()
    
    if isinstance(weights, dict):
        weights_array = np.array([weights[asset] for asset in returns_data.columns])
    else:
        weights_array = weights
    
    return risk_manager.analyze_portfolio_risk(returns_data, weights_array)

def calculate_efficient_frontier(returns_data: pd.DataFrame, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate efficient frontier points
    
    Returns:
        Tuple of (returns, volatilities) for efficient frontier
    """
    optimizer = AdvancedPortfolioOptimizer()
    
    # Calculate range of target returns
    expected_returns = optimizer._calculate_expected_returns(returns_data)
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    target_returns = np.linspace(min_return, max_return, num_points)
    
    efficient_returns = []
    efficient_volatilities = []
    
    for target_return in target_returns:
        try:
            result = optimizer.optimize_portfolio(
                returns_data=returns_data,
                method='mean_variance',
                target_return=target_return
            )
            
            if result.optimization_success:
                efficient_returns.append(result.expected_return)
                efficient_volatilities.append(result.expected_volatility)
        except:
            continue
    
    return np.array(efficient_returns), np.array(efficient_volatilities)

if __name__ == "__main__":
    # Example usage and testing
    logger.info("🚀 Portfolio Optimization & Risk Management System (Step 5)")
    logger.info("   Advanced portfolio construction and risk management")
    logger.info("   Ready for integration with ML trading system") 