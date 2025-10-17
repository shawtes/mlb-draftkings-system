# Advanced Quant Tab Design
## Financial-Grade Quantitative Optimization Settings

---

## Purpose

The **Advanced Quant Tab** provides access to sophisticated financial modeling techniques for lineup optimization. This includes GARCH volatility modeling, Monte Carlo simulations, Kelly Criterion position sizing, and risk-adjusted portfolio optimization.

**Target Users:** Advanced DFS professionals familiar with financial concepts

---

## Layout Structure

```
┌────────────────────────────────────────────────────────────────┐
│ 🔬 Advanced Quant Tab                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [✓] Enable Advanced Quantitative Optimization                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ OPTIMIZATION STRATEGY                                    │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Strategy: [Combined ▼]                                   │ │
│  │   • combined       ← Selected                            │ │
│  │   • kelly_criterion                                      │ │
│  │   • risk_parity                                          │ │
│  │   • mean_variance                                        │ │
│  │   • equal_weight                                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ RISK PARAMETERS                                          │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Risk Tolerance:          [1.0_____] (0.1 - 2.0)         │ │
│  │ VaR Confidence Level:    [0.95____] (0.90 - 0.99)       │ │
│  │ Target Volatility:       [0.15____] (0.05 - 0.50)       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ MONTE CARLO SIMULATION                                   │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Simulations:             [10000___] (1K - 50K)           │ │
│  │ Time Horizon (days):     [1_______] (1 - 30)            │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ GARCH VOLATILITY MODELING                                │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ GARCH p:                 [1_______] (1 - 5)              │ │
│  │ GARCH q:                 [1_______] (1 - 5)              │ │
│  │ Lookback Period:         [100_____] (30 - 365 days)     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ COPULA DEPENDENCY MODELING                               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Copula Family: [Gaussian ▼]                              │ │
│  │ Dependency Threshold:    [0.30____] (0.1 - 0.9)         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ KELLY CRITERION POSITION SIZING                          │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Max Kelly Fraction:      [0.25____] (0.1 - 1.0)         │ │
│  │ Expected Win Rate:       [0.20____] (0.1 - 0.9)         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ STATUS & INFORMATION                                     │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Status: ✓ Advanced quantitative optimization ENABLED    │ │
│  │                                                          │ │
│  │ Library Status:                                          │ │
│  │ ✓ ARCH (GARCH): Available                                │ │
│  │ ⚠ Copulas: Optional - limited dependency modeling       │ │
│  │ ✓ SciPy: Available                                       │ │
│  │ ✓ Scikit-learn: Available                                │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ✓ Tip: These settings enable financial-grade risk modeling │
│         for professional DFS portfolio management            │
└────────────────────────────────────────────────────────────────┘
```

---

## Master Control

### Enable Advanced Quantitative Optimization Checkbox

```
[✓] Enable Advanced Quantitative Optimization
```

**Function:**
- Master switch for all advanced features
- When checked: Enables all controls below
- When unchecked: Disables all controls, uses standard optimization

**Status Indicator:**
- **Enabled + Libraries Available:** "✅ Advanced quantitative optimization ENABLED"
- **Enabled + Missing Libraries:** "⚠️ Advanced quantitative optimization UNAVAILABLE - missing libraries"
- **Disabled:** "❌ Advanced quantitative optimization DISABLED"

---

## Optimization Strategy Section

### Strategy Dropdown

**Options:**

#### 1. Combined (Default)
```
Combines multiple optimization techniques:
• Kelly Criterion for position sizing
• Risk parity for balance
• Mean-variance for efficiency
• Copula modeling for dependencies

Best for: Balanced professional approach
```

#### 2. Kelly Criterion
```
Pure Kelly optimal betting strategy
Maximizes long-term growth rate

Formula: f* = (bp - q) / b
Where:
  f* = fraction of bankroll
  b = odds (payout ratio)
  p = win probability
  q = loss probability (1-p)

Best for: Aggressive growth, high confidence
```

#### 3. Risk Parity
```
Equal risk contribution from each position
Balances volatility across lineup

Formula: w_i ∝ 1/σ_i
Where:
  w_i = weight of player i
  σ_i = volatility of player i

Best for: Defensive, consistent returns
```

#### 4. Mean-Variance
```
Classic Markowitz portfolio optimization
Maximizes return for given risk level

Formula: max(μ'w - λw'Σw)
Where:
  μ = expected returns
  w = weights
  λ = risk aversion
  Σ = covariance matrix

Best for: Academic approach, balanced risk/return
```

#### 5. Equal Weight
```
Simple equal allocation
Baseline strategy for comparison

All positions weighted equally
No advanced modeling

Best for: Simplicity, benchmark comparison
```

---

## Risk Parameters Section

### Risk Tolerance Slider
- **Range:** 0.1 - 2.0
- **Default:** 1.0 (neutral)
- **Meaning:**
  - < 1.0: Conservative (prefer safety)
  - 1.0: Neutral (balanced)
  - > 1.0: Aggressive (prefer upside)

**Impact:**
```python
risk_adjusted_score = base_score + (risk_tolerance * upside_potential)
                                  - ((2 - risk_tolerance) * downside_risk)
```

### VaR Confidence Level
- **Range:** 0.90 - 0.99
- **Default:** 0.95 (95% confidence)
- **Meaning:** Probability level for Value-at-Risk calculation

**Interpretation:**
```
VaR(0.95) = Maximum expected loss in 95% of scenarios
Example: VaR(0.95) = -15 points
→ 95% of time, won't lose more than 15 points
→ 5% of time, could lose more
```

### Target Volatility
- **Range:** 0.05 - 0.50
- **Default:** 0.15 (15% volatility)
- **Meaning:** Target standard deviation of returns

**Application:**
```python
# Scale positions to match target volatility
portfolio_volatility = sqrt(w' Σ w)
scale_factor = target_volatility / portfolio_volatility
adjusted_weights = weights * scale_factor
```

---

## Monte Carlo Simulation Section

### Simulations Count
- **Range:** 1,000 - 50,000
- **Default:** 10,000
- **Purpose:** Number of random scenarios to simulate

**Trade-off:**
- **More simulations:** More accurate, slower
- **Fewer simulations:** Less accurate, faster

**Recommended:**
- Quick test: 1,000
- Normal use: 10,000
- High stakes: 25,000+

### Time Horizon
- **Range:** 1 - 30 days
- **Default:** 1 (single day/contest)
- **Purpose:** Forecast period for simulations

**Usage:**
- 1 day: Single slate
- 7 days: Weekly tournaments
- 30 days: Monthly optimization

---

## GARCH Volatility Modeling Section

**GARCH:** Generalized Autoregressive Conditional Heteroskedasticity
**Purpose:** Model time-varying volatility

### GARCH p Parameter
- **Range:** 1 - 5
- **Default:** 1
- **Meaning:** Number of lagged squared errors (ARCH terms)

**Formula:**
```
σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
       └─────────┬─────────┘
              ARCH(p)
```

### GARCH q Parameter
- **Range:** 1 - 5
- **Default:** 1
- **Meaning:** Number of lagged conditional variances

**Common Models:**
- GARCH(1,1): Standard (most common)
- GARCH(1,2): More persistent volatility
- GARCH(2,1): More reactive to shocks

### Lookback Period
- **Range:** 30 - 365 days
- **Default:** 100 days
- **Purpose:** Historical window for volatility estimation

**Trade-off:**
- Longer: More stable, less responsive
- Shorter: More responsive, less stable

---

## Copula Dependency Modeling Section

**Copula:** Mathematical tool for modeling dependencies

### Copula Family Dropdown

**Options:**

#### Gaussian Copula (Default)
```
Assumes normal distribution of dependencies
Symmetric tail behavior

Best for: General use, well-behaved data
```

#### t-Copula
```
Student's t-distribution
Allows for heavier tails (extreme events)

Best for: High-variance sports, GPP tournaments
```

#### Clayton Copula
```
Lower tail dependence
Players fail together

Best for: Defensive stacks, pitcher + defense
```

#### Frank Copula
```
Weak tail dependence
More independent extreme events

Best for: Diversified lineups
```

#### Gumbel Copula
```
Upper tail dependence
Players succeed together

Best for: Offensive stacks, positive correlation
```

### Dependency Threshold
- **Range:** 0.1 - 0.9
- **Default:** 0.3 (30%)
- **Meaning:** Minimum correlation to model

**Impact:**
```python
if correlation(player_i, player_j) > threshold:
    model_dependency(player_i, player_j, copula_family)
else:
    treat_as_independent()
```

---

## Kelly Criterion Section

### Max Kelly Fraction
- **Range:** 0.1 - 1.0
- **Default:** 0.25 (fractional Kelly)
- **Meaning:** Maximum fraction of bankroll to risk

**Kelly Variants:**
- Full Kelly (1.0): Optimal but volatile
- Half Kelly (0.5): Safer, smoother growth
- **Quarter Kelly (0.25):** Recommended balance
- Tenth Kelly (0.1): Very conservative

**Formula:**
```python
kelly_fraction = (win_prob * payout - loss_prob) / payout
capped_fraction = min(kelly_fraction, max_kelly_fraction)
lineup_allocation = bankroll * capped_fraction
```

### Expected Win Rate
- **Range:** 0.1 - 0.9
- **Default:** 0.2 (20% win rate)
- **Meaning:** Estimated probability of lineup cashing

**Contest Type Estimates:**
- 50/50: 0.50 (50% win)
- Double-Up: 0.45 (45% win)
- **GPP Top 20%:** 0.20 (20% win)
- GPP Top 10%: 0.10 (10% win)
- GPP Top 1%: 0.01 (1% win)

---

## Status & Information Section

### Library Status Display

Shows availability of required Python packages:

```
Library Status:
✓ ARCH (GARCH): Available
⚠ Copulas: Optional - dependency modeling limited
✓ SciPy: Available
✓ Scikit-learn: Available
```

**States:**
- ✓ Available: Library installed and working
- ⚠ Optional: Feature available but limited
- ❌ Missing: Library not installed, feature disabled

**Installation Commands:**
```bash
pip install arch          # GARCH modeling
pip install copulas       # Copula modeling
pip install scipy         # Statistical functions
pip install scikit-learn  # Machine learning utilities
```

---

## Probability Enhancement Section

### Probability Detection
When CSV contains probability columns (`Prob_Over_5`, `Prob_Over_10`, etc.):

```
┌──────────────────────────────────────┐
│ ✓ Probability data detected          │
│ Enhanced metrics enabled              │
│                                       │
│ Columns found:                        │
│ • Prob_Over_5 (199 players)          │
│ • Prob_Over_10 (199 players)         │
│ • Prob_Over_15 (199 players)         │
│ • Prob_Over_20 (180 players)         │
│                                       │
│ Enhanced Metrics:                     │
│ • Expected_Utility: 199 players      │
│ • Risk_Adjusted_Points: 199 players  │
│ • Kelly_Fraction: 199 players        │
│ • Implied_Volatility: 199 players    │
└──────────────────────────────────────┘
```

### Contest Strategy
Auto-detected or user-selected:

```
Contest Strategy: [GPP ▼]
  • Cash Game (conservative)
  • Balanced (default)
  • GPP (aggressive)
```

**Impact:**
```python
if contest_strategy == 'Cash Game':
    optimize_for_floor()  # High-floor players
    min_volatility()
elif contest_strategy == 'GPP':
    optimize_for_ceiling()  # High-ceiling players
    accept_volatility()
```

---

## Integration with Optimization

### How Advanced Quant Affects Optimization

**Standard Optimization:**
```python
objective = maximize(sum(predicted_points))
```

**Advanced Quant Optimization:**
```python
# Multi-objective with risk adjustment
objective = maximize(
    expected_utility
    - (risk_penalty * volatility)
    + (kelly_bonus * win_probability)
    - (var_penalty * worst_case_loss)
)

# Portfolio constraints
constraints = [
    portfolio_volatility <= target_volatility,
    individual_weights <= kelly_max_fraction,
    correlation_limits(copula_model),
    ...
]
```

### Optimization Flow

```
1. Load player data
2. If advanced quant enabled:
   a. Estimate volatilities (GARCH)
   b. Model dependencies (Copula)
   c. Run Monte Carlo simulations
   d. Calculate risk metrics
   e. Apply Kelly sizing
3. Generate lineups with risk constraints
4. Select optimal portfolio
5. Display with risk metrics
```

---

## Risk Metrics Display

### After Optimization

When advanced quant is enabled, results include:

```
┌────────────────────────────────────────────┐
│ Lineup 1:                                  │
│ Total Points: 125.3                        │
│ Total Salary: $49,500                      │
│                                            │
│ Risk Metrics:                              │
│ • Sharpe Ratio: 1.45 ✓ Excellent          │
│ • Volatility: 12.8%                        │
│ • VaR(95%): -8.2 points                    │
│ • Kelly Fraction: 18.5%                    │
│ • Position Size: $185 (of $1000 bankroll)  │
│ • Win Probability: 22.3%                   │
└────────────────────────────────────────────┘
```

### Portfolio Summary

```
Portfolio Risk Analysis (100 lineups):
┌─────────────────────────────────────┐
│ Expected Return:    124.5 pts/LU   │
│ Portfolio Volatility:    11.2%      │
│ Sharpe Ratio:           1.38        │
│ Max Drawdown:          -15.8 pts    │
│ VaR(95%):              -9.3 pts     │
│ CVaR(95%):            -12.7 pts     │
│                                     │
│ Recommended Allocation:             │
│ • Total Entry Fees: $245            │
│ • Per Lineup: $2.45 avg             │
│ • Risk: 24.5% of $1000 bankroll     │
└─────────────────────────────────────┘
```

---

## Best Practices

### For Cash Games
```
Strategy: Risk Parity or Mean-Variance
Risk Tolerance: 0.6 - 0.8 (conservative)
Target Volatility: 0.08 - 0.12 (low)
Kelly Fraction: 0.15 - 0.25 (fractional)
Expected Win Rate: 0.45 - 0.50
```

### For GPP Tournaments
```
Strategy: Combined or Kelly Criterion
Risk Tolerance: 1.2 - 1.8 (aggressive)
Target Volatility: 0.15 - 0.25 (higher)
Kelly Fraction: 0.20 - 0.30
Expected Win Rate: 0.10 - 0.25
```

### For Large Bankrolls
```
Enable all features
Use full Monte Carlo (50K simulations)
GARCH(2,2) for stability
Copula modeling for all dependencies
Conservative Kelly (0.10 - 0.15)
```

---

## Error Handling

### Missing Libraries
```
⚠ Warning: ARCH library not installed

GARCH volatility modeling unavailable

Install with: pip install arch

Advanced optimization will continue with
estimated volatilities instead.
```

### Incompatible Data
```
⚠ Error: Insufficient historical data

GARCH requires 30+ days of data
Current data: 5 days

Options:
• Increase lookback period
• Use simpler volatility model
• Disable GARCH modeling
```

### Unstable Results
```
⚠ Warning: Kelly fractions exceed 100%

Estimated win rates may be too optimistic
Kelly recommended: 145% of bankroll (impossible!)

Suggestions:
• Reduce expected win rate
• Increase max Kelly fraction cap
• Review probability estimates
```

---

## Technical Notes

### Performance Impact
- **GARCH modeling:** +5-10 seconds
- **Copula modeling:** +10-20 seconds
- **Monte Carlo (10K):** +15-30 seconds
- **Total overhead:** ~30-60 seconds per optimization

### Memory Usage
- **Standard:** ~200 MB
- **With advanced quant:** ~500-800 MB
- **Large Monte Carlo:** Up to 1.5 GB

### Accuracy vs Speed
```
Configuration          Speed    Accuracy
─────────────────────  ───────  ────────
Disabled               Fast     Good
Basic (default)        Medium   Better
Full Advanced          Slow     Best
Monte Carlo 50K        Slowest  Optimal
```

---

Next: [My Entries Tab](07_MY_ENTRIES_TAB.md)

