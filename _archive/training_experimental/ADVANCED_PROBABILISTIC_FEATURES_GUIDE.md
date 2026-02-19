# Advanced Probabilistic and Copula Features Guide

## Overview
This guide details the comprehensive probabilistic and statistical feature engineering implemented for MLB fantasy point prediction. The features are designed to capture complex dependencies, extreme events, and statistical relationships that traditional features may miss.

## Feature Categories

### 1. GARCH Volatility Features
**Purpose**: Model time-varying volatility in fantasy point performance
- `garch_volatility`: GARCH(1,1) conditional volatility
- `garch_conditional_volatility`: Time-varying volatility estimates  
- `volatility_regime`: Binary indicator for high/low volatility periods

**Benefits**:
- Captures heteroskedasticity in player performance
- Identifies periods of high uncertainty
- Helps with risk-adjusted predictions

### 2. Distributional Features
**Purpose**: Capture higher-order statistical moments and tail behavior
- `skewness_7d/14d/30d`: Rolling skewness over different windows
- `kurtosis_7d/14d/30d`: Rolling kurtosis (tail heaviness)
- `var_95_7d/14d/30d`: Value at Risk (5th percentile)
- `var_99_7d/14d/30d`: Extreme Value at Risk (1st percentile)
- `expected_shortfall_7d/14d/30d`: Expected loss beyond VaR
- `tail_ratio`: Ratio of upper to lower tail thickness
- `prob_exceed_5/10/15/20`: Dynamic probabilities of exceeding thresholds

**Benefits**:
- Captures asymmetric risk profiles
- Identifies players with extreme upside/downside
- Enables probabilistic predictions

### 3. Correlation Features
**Purpose**: Model dynamic relationships between top players
- `avg_player_correlation`: Average correlation with other elite players
- `correlation_volatility`: Instability in correlation relationships

**Benefits**:
- Captures market-wide effects
- Identifies contrarian opportunities
- Models systematic risk factors

### 4. Regime Features
**Purpose**: Identify different performance states
- `bull_regime`: Binary indicator for uptrend periods
- `regime_strength`: Magnitude of trend
- `momentum_regime`: Categorical momentum state (0-2)
- `consistency_regime`: Binary indicator for low volatility periods

**Benefits**:
- Adapts to changing player conditions
- Identifies performance cycles
- Enables regime-dependent predictions

### 5. Advanced Information Features
**Purpose**: Capture complex statistical properties
- `entropy`: Information content of performance distribution
- `hurst_exponent`: Long-term memory and trend persistence
- `max_drawdown`: Maximum historical decline
- `current_drawdown`: Current decline from peak
- `drawdown_duration`: Length of current decline
- `rolling_sharpe`: Risk-adjusted returns

**Benefits**:
- Measures performance complexity
- Identifies trend persistence
- Captures risk-adjusted metrics

## New Advanced Features

### 6. Copula Dependencies
**Purpose**: Model non-linear dependencies between different statistics

#### Gaussian Copula Parameters
- `gaussian_copula_HR_RBI`: Gaussian copula between HR and RBI
- `gaussian_copula_HR_R`: Gaussian copula between HR and Runs
- `gaussian_copula_HR_H`: Gaussian copula between HR and Hits
- [Additional combinations for all key statistics]

#### Clayton Copula Parameters
- `clayton_copula_HR_RBI`: Clayton copula (lower tail dependence)
- `clayton_copula_HR_R`: Clayton copula between HR and Runs
- [Additional combinations for all key statistics]

#### Tail Dependence Coefficients
- `upper_tail_dep_HR_RBI`: Upper tail dependence between HR and RBI
- `lower_tail_dep_HR_RBI`: Lower tail dependence between HR and RBI
- [Additional combinations for all key statistics]

**Benefits**:
- Captures non-linear statistical relationships
- Models extreme co-movements
- Enables sophisticated dependency modeling

### 7. Extreme Value Theory Features
**Purpose**: Model extreme events and tail behavior
- `evt_location`: Location parameter of extreme value distribution
- `evt_scale`: Scale parameter (spread of extremes)
- `evt_shape`: Shape parameter (tail heaviness)
- `evt_return_level`: Expected maximum in future periods
- `exceedance_prob`: Probability of exceeding high thresholds
- `extreme_value_index`: Index of extreme behavior
- `pot_threshold`: Peak-over-threshold level
- `pot_excess_mean`: Mean of excesses over threshold
- `pot_excess_std`: Standard deviation of excesses

**Benefits**:
- Predicts extreme performances
- Models tail risks and opportunities
- Enables extreme event forecasting

### 8. Network Features
**Purpose**: Model player interactions and market dynamics
- `network_centrality`: Player's importance in performance network
- `network_clustering`: Local clustering in performance space
- `network_volatility`: Instability in network position
- `network_efficiency`: Efficiency of network connections

**Benefits**:
- Captures market dynamics
- Identifies influential players
- Models systemic relationships

### 9. Spectral Features
**Purpose**: Frequency domain analysis of performance patterns
- `dominant_frequency`: Most important cyclical pattern
- `spectral_entropy`: Complexity of frequency components
- `spectral_centroid`: Center of frequency distribution
- `spectral_rolloff`: Frequency rolloff characteristics
- `rolling_spectral_entropy`: Time-varying spectral complexity

**Benefits**:
- Identifies cyclical patterns
- Captures rhythm and seasonality
- Enables frequency-based predictions

## Implementation Details

### Data Requirements
- Minimum 10-15 observations per player for basic features
- Minimum 20+ observations for advanced features
- Requires historical performance data across multiple statistics

### Computational Considerations
- Features are calculated using rolling windows
- Computationally intensive for large datasets
- Benefits from parallel processing

### Risk Management
- All features are cleaned for infinite values
- Forward-fill and zero-fill strategies for missing data
- Robust to outliers and missing observations

## Model Integration

### Feature Selection
- Total features: 100+ probabilistic and copula features
- Automatic feature selection reduces dimensionality
- Top 550 features selected for final model

### Prediction Enhancement
- Enables probabilistic predictions
- Supports confidence intervals
- Allows for risk-adjusted forecasts

### Performance Benefits
- Captures non-linear relationships
- Improves extreme event prediction
- Enhances overall accuracy

## Usage Recommendations

### For Daily Predictions
1. Use shorter windows (7-14 days) for recent patterns
2. Focus on regime and correlation features
3. Monitor extreme value indicators

### For Tournament Selection
1. Emphasize tail dependence and extreme value features
2. Use network centrality for contrarian plays
3. Leverage spectral features for timing

### For Risk Management
1. Monitor drawdown and volatility features
2. Use copula features for portfolio construction
3. Apply extreme value theory for position sizing

## Future Enhancements

### Potential Additions
1. Markov regime switching models
2. Multivariate GARCH models
3. Machine learning-based copulas
4. Dynamic factor models
5. Wavelets analysis

### Data Enhancements
1. Intraday performance data
2. Weather and contextual factors
3. Opponent-specific adjustments
4. Injury and roster information

This comprehensive feature set transforms traditional fantasy sports prediction into a sophisticated probabilistic modeling framework, enabling more accurate and nuanced predictions.
