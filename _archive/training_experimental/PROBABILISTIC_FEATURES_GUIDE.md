# Probabilistic Feature Engineering for MLB Fantasy Points

## Overview
Added advanced probabilistic feature engineering to capture complex statistical relationships and improve prediction accuracy for MLB fantasy point prediction.

## New Features Added

### 1. GARCH Volatility Features
- **garch_volatility**: GARCH(1,1) conditional volatility for each player
- **garch_conditional_volatility**: Time-varying volatility estimates
- **volatility_regime**: Binary indicator for high/low volatility periods

### 2. Distributional Features
- **skewness_7d/14d/30d**: Rolling skewness of fantasy points (tail asymmetry)
- **kurtosis_7d/14d/30d**: Rolling kurtosis of fantasy points (tail thickness)
- **var_95_7d/14d/30d**: Value at Risk (95th percentile) - downside risk measure
- **var_99_7d/14d/30d**: Value at Risk (99th percentile) - extreme downside risk
- **expected_shortfall_7d/14d/30d**: Expected loss beyond VaR threshold
- **tail_ratio**: Ratio of upside to downside tail risks

### 3. Dynamic Probability Features
- **prob_exceed_5/10/15/20**: Rolling probability of exceeding fantasy point thresholds
- These adapt to each player's recent performance patterns

### 4. Correlation Features (Copula-inspired)
- **avg_player_correlation**: Average correlation with other top players
- **correlation_volatility**: Volatility of correlations (dependency instability)

### 5. Regime Features
- **bull_regime**: Binary indicator for performance above long-term average
- **regime_strength**: Strength of current performance regime
- **momentum_regime**: Categorical momentum state (low/medium/high)
- **consistency_regime**: Binary indicator for low volatility periods

### 6. Advanced Statistical Features
- **entropy**: Information entropy of recent performance distribution
- **hurst_exponent**: Measure of long-term memory and trend persistence
- **max_drawdown**: Maximum peak-to-trough decline in rolling window
- **current_drawdown**: Current drawdown from recent peak
- **drawdown_duration**: Length of current drawdown period
- **rolling_sharpe**: Risk-adjusted return ratio

## Benefits

### 1. Risk Modeling
- GARCH features capture volatility clustering (periods of high/low variance)
- VaR and Expected Shortfall provide downside risk measures
- Drawdown features identify players in slumps

### 2. Regime Detection
- Identifies when players are in hot/cold streaks
- Captures momentum and mean reversion patterns
- Adapts to changing player performance states

### 3. Dependency Modeling
- Correlation features capture how player performances move together
- Helps identify lineup construction risks and opportunities
- Models contagion effects between players

### 4. Distributional Intelligence
- Skewness/kurtosis capture non-normal performance patterns
- Entropy measures performance predictability
- Hurst exponent identifies trend vs. random walk behavior

### 5. Dynamic Thresholds
- Probability features adapt to each player's baseline performance
- More realistic than static thresholds across all players
- Captures performance distribution evolution over time

## Implementation Notes

### Computational Efficiency
- Uses rolling windows to balance statistical power with computational speed
- Fallback to simplified features when insufficient data
- Handles missing data gracefully

### Statistical Robustness
- Bounds extreme values to prevent outlier contamination
- Uses robust statistical measures where possible
- Handles edge cases (zero variance, insufficient observations)

### Model Integration
- All features are standardized and preprocessed through the pipeline
- Compatible with existing ensemble model architecture
- Feature selection will automatically identify most predictive features

## Expected Performance Impact

### Improved Accuracy
- Better capture of player performance cycles
- More nuanced risk assessment
- Enhanced regime change detection

### Better Risk Management
- Identify high-variance players
- Detect correlation risks in lineups
- Predict performance distribution tails

### Enhanced Insights
- Understand player performance patterns
- Identify arbitrage opportunities
- Better timing of player selection

## Installation Requirements
For full GARCH functionality, install:
```bash
pip install arch
```

If not available, simplified volatility features are used automatically.

## Next Steps
1. Train model with new features
2. Analyze feature importance
3. Monitor prediction accuracy improvements
4. Consider additional features like:
   - Copula dependency parameters
   - Markov regime switching
   - Spectral analysis features
   - Network centrality measures
