# NFL DFS Strategies Implementation Guide
## Based on "Fantasy Football For Dummies"

---

## 🎯 What's Been Added

Your optimizer now includes **ALL** advanced DFS strategies from the book:

### ✅ Completed Implementations

1. **Contest Mode Optimization** - Different strategies for Cash vs GPP
2. **Advanced Stacking** - 6 stack types (QB+2WR, Game Stack, Bring Back, etc.)
3. **Contrarian Player Selection** - Ownership-based adjustments
4. **Ceiling/Floor Projections** - Not just averages
5. **Game Environment Factors** - Vegas lines, weather, game script
6. **Position-Specific Exposure** - Different limits per position
7. **Value Analysis** - Enhanced calculations for each contest type
8. **Bankroll Management** - Risk calculation and Kelly Criterion
9. **Fade Manager** - Auto-fade injury/weather risks
10. **Hedge Lineup Generation** - Multiple construction strategies

---

## 📁 New Files Created

### 1. `dfs_strategies_config.py`
**Configuration file for all DFS strategies**

Contains:
- Contest mode settings (Cash vs GPP)
- Stack type definitions
- Exposure limits by position
- Contrarian thresholds
- Game environment boost factors
- Bankroll rules
- Hedge strategies

### 2. `dfs_strategy_helpers.py`
**Helper classes that implement the strategies**

Classes:
- `ContestarianEngine` - Ownership-based player selection
- `GameEnvironmentAnalyzer` - Vegas/weather adjustments
- `StackingOptimizer` - Advanced stacking logic
- `ValueAnalyzer` - Enhanced value calculations
- `BankrollManager` - Risk management
- `FadeManager` - Player exclusion logic
- `HedgeLineupBuilder` - Diversified lineup construction

### 3. `load_nfl_data_enhanced.py`
**Enhanced data loader with ALL metrics**

Adds to your player data:
- **Ceiling** - Maximum realistic projection (upside)
- **Floor** - Minimum realistic projection (safety)
- **Ownership** - Projected ownership % (for contrarian plays)
- **Game Total** - Vegas over/under (high-scoring games)
- **Spread** - Vegas spread (game script implications)
- **Implied Points** - Team's expected points
- **Weather** - Wind, precipitation, temperature
- **Opponent** - Matchup data
- **Recent Form** - Trending up/down/stable

---

## 🚀 Quick Start Guide

### Step 1: Load Enhanced Data

```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION

# For GPP Tournaments
python3 load_nfl_data_enhanced.py
```

This creates TWO files:
- `nfl_week7_gpp_enhanced.csv` - Optimized for tournaments
- `nfl_week7_cash_enhanced.csv` - Optimized for cash games

### Step 2: Use the Enhanced Data

**Option A: In Your Existing Optimizer**
```python
# Load the enhanced CSV
df = pd.read_csv('nfl_week7_gpp_enhanced.csv')

# It now has all these columns:
# - ceiling, floor (for upside/safety)
# - ownership (for contrarian plays)
# - game_total, spread (for game environment)
# - wind, precip (for weather fades)
# - opponent, opp_def_rank (for matchups)
```

**Option B: Using the Helper Classes**
```python
from dfs_strategy_helpers import *

# Apply contrarian strategy
contrarian = ContestarianEngine('gpp_tournament')
df = contrarian.apply_ownership_adjustments(df)

# Apply game environment boosts
game_env = GameEnvironmentAnalyzer()
df = game_env.apply_game_environment_boosts(df)

# Calculate bankroll risk
bank_manager = BankrollManager(total_bankroll=1000)
max_entries, msg = bank_manager.calculate_max_entries('gpp_tournament', entry_fee=5)
print(msg)
```

---

## 📊 Understanding the New Columns

### Ceiling & Floor
```
FantasyPoints: 15.0 (average projection)
Ceiling: 21.0 (if everything goes right)
Floor: 10.5 (if things go wrong)
```

**Use for:**
- **GPP**: Target high ceiling players
- **Cash**: Target high floor players

**Position Variance:**
- QB: ±25% (most consistent)
- RB: ±35%
- WR: ±45% (highest variance)
- TE: ±40%
- DST: ±50%

### Ownership
```
ownership: 0.25 = 25% projected ownership
```

**Ownership Tiers:**
- **Leverage** (<5%): Extreme contrarian
- **Low** (5-8%): Contrarian play
- **Medium** (8-15%): Normal
- **High** (15-25%): Popular
- **Chalk** (>25%): Highly owned

**GPP Strategy:**
- Fade chalk (>25% owned) with mediocre upside
- Target low-owned (<8%) players with high ceiling

**Cash Strategy:**
- Ownership doesn't matter - play best projections

### Game Environment

#### Game Total
```
game_total: 48.5 (high-scoring game expected)
```

**Boosts Applied:**
- >48 total: +15% to all offense
- <40 total: -15% to offense, +20% to DST

#### Spread
```
spread: -7.0 (team favored by 7)
```

**Game Script Implications:**
- **Big Favorite** (<-7): RBs get boost (run clock), QB gets penalty
- **Big Underdog** (>+7): QB/WR get boost (passing to catch up), RB gets penalty

#### Weather
```
wind: 22 mph
precip: 0.4 inches
```

**Penalties Applied:**
- High wind (>20 mph): QB -25%, WR -20%, RB +15%
- Rain/snow: Similar to wind

### Opponent Data
```
opponent: TB
opp_def_rank: 15 (mid-tier defense)
```

**Matchup Quality:**
- 1-10: Elite defense (fade opposing offense)
- 11-22: Average defense
- 23-32: Weak defense (target opposing offense)

### Recent Form
```
recent_form: 1.12 (trending up)
trend: Up
```

**Momentum Factor:**
- >1.05: Hot (trending up)
- 0.95-1.05: Stable
- <0.95: Cold (trending down)

---

## 🎮 Contest-Specific Strategies

### GPP Tournament Strategy

**Goal:** Top 1-10% finish for big payouts

**Settings in enhanced file:**
```python
contest_mode = 'gpp_tournament'
```

**What it does:**
1. ✅ Uses **ceiling** projections (upside)
2. ✅ Applies **contrarian** boosts to low-owned players
3. ✅ Penalizes **chalk** (>25% owned)
4. ✅ Values **high variance** positions (WR, TE)
5. ✅ Exposure limits: 5-20% per player
6. ✅ Emphasizes **stacking**

**Player Selection:**
- Target high ceiling, low ownership
- Fade popular players without upside
- Stack aggressively (QB+2WR)
- Use game stacks in high-scoring games
- Diversify across 20+ lineups

### Cash Game Strategy

**Goal:** Top 50% finish (consistent profit)

**Settings:**
```python
contest_mode = 'cash_game'
```

**What it does:**
1. ✅ Uses **floor** projections (safety)
2. ✅ NO contrarian adjustments (play the chalk)
3. ✅ Values **consistency** over upside
4. ✅ Prefers **low variance** positions (QB, RB)
5. ✅ Exposure limits: 10-40% per player
6. ✅ Moderate stacking

**Player Selection:**
- Target high floor, high ownership
- Play the chalk if it's good value
- Use primary stacks (QB+WR)
- Avoid high-risk plays
- Need 1-3 lineups max

---

## 🔧 Advanced Features Usage

### 1. Contrarian Engine

```python
from dfs_strategy_helpers import ContestarianEngine

# Initialize for GPP
contrarian = ContestarianEngine('gpp_tournament')

# Apply ownership adjustments
df = contrarian.apply_ownership_adjustments(df)

# Find best contrarian plays
top_contrarian = contrarian.identify_contrarian_plays(df, top_n=10)
print(top_contrarian)
```

**What it does:**
- Boosts low-owned players by 15-25%
- Penalizes chalk players by 15%
- Creates "contrarian_score" metric

### 2. Game Environment Analyzer

```python
from dfs_strategy_helpers import GameEnvironmentAnalyzer

game_env = GameEnvironmentAnalyzer()
df = game_env.apply_game_environment_boosts(df)
```

**Boosts Applied:**
- High totals → Offense +15%
- Low totals → DST +20%
- Big favorites → RB +12%, QB -5%
- Big underdogs → QB +15%, WR +10%
- Bad weather → RB +15%, QB/WR -20-25%

### 3. Value Analyzer

```python
from dfs_strategy_helpers import ValueAnalyzer

# For GPP (uses ceiling)
value_analyzer = ValueAnalyzer('gpp_tournament')
df = value_analyzer.calculate_enhanced_value(df)

# For Cash (uses floor)
value_analyzer = ValueAnalyzer('cash_game')
df = value_analyzer.calculate_enhanced_value(df)
```

**Calculates:**
- `value_projection` - Standard value
- `value_ceiling` - Upside value (GPP)
- `value_floor` - Safety value (Cash)
- `value_rating` - Poor/Fair/Good/Great/Elite

### 4. Bankroll Manager

```python
from dfs_strategy_helpers import BankrollManager

# Initialize with your bankroll
bank_manager = BankrollManager(total_bankroll=1000)

# Calculate safe entry amounts
max_entries, msg = bank_manager.calculate_max_entries(
    contest_type='gpp_tournament',
    entry_fee=5.00
)

print(msg)
# Output:
# 💰 Bankroll Analysis (gpp_tournament)
# Total Bankroll: $1,000.00
# Entry Fee: $5.00
# Max Safe Entries: 4
# Total Risk: $20.00 (2.0%)
```

**Bankroll Rules:**
- **Cash Games**: Risk max 5% of bankroll
- **GPP**: Risk max 2% of bankroll
- **Single Entry GPP**: Risk max 3%

### 5. Fade Manager

```python
from dfs_strategy_helpers import FadeManager

fade_mgr = FadeManager()

# Manual fades
fade_mgr.add_fade("Patrick Mahomes", "Too expensive for this slate")
fade_mgr.add_fade("Travis Kelce", "Injury concern")

# Apply fades
df = fade_mgr.apply_fades(df)

# Auto-fades (injury, weather, etc.)
df = fade_mgr.apply_auto_fades(df)
```

**Auto-Fade Criteria:**
- Injury status (Q, D, Out, GTD)
- Bad weather (wind >20 mph, precip >0.3")
- Top 5 defense matchup
- Overpriced ($500+ salary increase)
- High ownership without upside

### 6. Stacking Optimizer

```python
from dfs_strategy_helpers import StackingOptimizer

# Primary stack (QB + 2 WRs)
stacker = StackingOptimizer('primary_stack')
stack = stacker.build_primary_stack(df, team='KC')

# Game stack (QB + WR + Opp WR)
stacker = StackingOptimizer('game_stack')
stack = stacker.build_game_stack(df, team1='KC', team2='BUF')
```

**Stack Types:**
1. **Primary** (QB+2WR) - High correlation (0.85)
2. **Tight** (QB+WR+TE) - Contrarian (0.80)
3. **Double** (QB+2WR+RB) - Ultra-aggressive (0.75)
4. **Game** (QB+WR+Opp WR) - High-scoring game (0.65)
5. **Bring Back** (QB+2WR+Opp RB) - Hedge (0.55)
6. **RB** (RB+Opp DST) - Negative correlation (-0.40)

---

## 📈 Optimization Workflow

### For GPP Tournaments

```python
# 1. Load enhanced GPP data
df = pd.read_csv('nfl_week7_gpp_enhanced.csv')

# 2. Apply contrarian adjustments
from dfs_strategy_helpers import ContestarianEngine
contrarian = ContestarianEngine('gpp_tournament')
df = contrarian.apply_ownership_adjustments(df)

# 3. Find best contrarian plays
top_plays = contrarian.identify_contrarian_plays(df, top_n=20)

# 4. Build stacks
from dfs_strategy_helpers import StackingOptimizer
stacker = StackingOptimizer('game_stack')
stacks = []
for team1, team2 in high_total_games:
    stack = stacker.build_game_stack(df, team1, team2)
    if stack:
        stacks.append(stack)

# 5. Generate diverse lineups (20-150 entries)
# Use your existing genetic optimizer with:
# - Max exposure: 20% per player
# - Min exposure: 5% (force diversity)
# - Stack requirements: 80% of lineups have stacks

# 6. Check bankroll risk
from dfs_strategy_helpers import BankrollManager
bank_mgr = BankrollManager(1000)
max_entries, msg = bank_mgr.calculate_max_entries('gpp_tournament', 5)
print(msg)
```

### For Cash Games

```python
# 1. Load enhanced Cash data
df = pd.read_csv('nfl_week7_cash_enhanced.csv')

# 2. Filter to high-floor players
safe_plays = df[df['floor'] > 8.0]

# 3. Don't apply contrarian (play the chalk)

# 4. Build conservative stack
from dfs_strategy_helpers import StackingOptimizer
stacker = StackingOptimizer('primary_stack')
stack = stacker.build_primary_stack(df, team='KC')

# 5. Generate 1-3 lineups
# Use your existing genetic optimizer with:
# - Max exposure: 40% per player (ok to repeat)
# - Focus on floor projections
# - One clean stack

# 6. Check bankroll risk
from dfs_strategy_helpers import BankrollManager
bank_mgr = BankrollManager(1000)
max_entries, msg = bank_mgr.calculate_max_entries('cash_game', 5)
print(msg)
```

---

## 🎓 Strategy Cheat Sheet

### When to Use Each Feature

| Feature | Cash Game | GPP | Purpose |
|---------|-----------|-----|---------|
| **Ceiling Projections** | ❌ | ✅ | Need upside to win big |
| **Floor Projections** | ✅ | ❌ | Need consistency to cash |
| **Contrarian Adjustments** | ❌ | ✅ | Differentiate from field |
| **Ownership Data** | ❌ | ✅ | Fade chalk, target leverage |
| **Game Environment** | ✅ | ✅ | Both benefit from context |
| **Weather Fades** | ✅ | ✅ | Avoid disasters |
| **Aggressive Stacking** | ❌ | ✅ | High correlation for GPP |
| **Conservative Stacking** | ✅ | ❌ | Safe QB+WR for cash |
| **High Exposure (40%)** | ✅ | ❌ | Ok to repeat chalk |
| **Low Exposure (20%)** | ❌ | ✅ | Need diversity |

---

## 📚 Configuration Reference

### Edit These Files to Customize

#### `dfs_strategies_config.py`

**Contest Mode Settings:**
```python
CONTEST_MODES = {
    'gpp_tournament': {
        'exposure_max_default': 0.20,  # Change to 0.15 for more diversity
        'contrarian_enabled': True,     # Set False to disable
    }
}
```

**Stack Aggressiveness:**
```python
NFL_STACK_TYPES = {
    'primary_stack': {
        'correlation': 0.85,  # Adjust expected correlation
    }
}
```

**Contrarian Thresholds:**
```python
CONTRARIAN_THRESHOLDS = {
    'high_owned': 0.25,  # Change to 0.30 for more lenient
}
```

**Game Environment Boosts:**
```python
GAME_ENVIRONMENT_BOOSTS = {
    'high_total': {
        'threshold': 48.0,  # Lower to 46.0 to boost more games
        'offensive_boost': 1.15,  # Increase to 1.20 for bigger boost
    }
}
```

#### `load_nfl_data_enhanced.py`

**Ownership Model:**
```python
# Line ~120-140
# Adjust the ownership formula weights
df['ownership'] = (
    df['salary_pct'] * 0.30 +  # Increase for more salary-based
    df['value_pct'] * 0.40 +   # Increase for more value-based
    df['is_stud'] +
    df['is_value'] +
    df['is_punt']
)
```

**Ceiling/Floor Variance:**
```python
# Line ~30-40
variance_by_position = {
    'QB': 0.25,  # Increase for more variance
    'WR': 0.45,  # Decrease for less variance
}
```

---

## 🆘 Troubleshooting

### "No ownership data available"
**Solution:** The enhanced loader generates ownership. If you get this error, make sure you're loading the enhanced CSV files.

### "ValueError: Bin labels must be one fewer"
**Solution:** Fixed! The code now handles small datasets automatically.

### "Not enough players for stack"
**Solution:** Lower your stack requirements or use a slate with more teams.

### Projections seem off after adjustments
**Solution:** Check that game environment data is realistic. You may need to adjust boost/penalty factors in `dfs_strategies_config.py`.

---

## 🎯 Next Steps

1. ✅ **Load Enhanced Data** - Run `load_nfl_data_enhanced.py`
2. ✅ **Explore the Data** - Check the new CSV columns
3. ⏭️ **Integrate with Optimizer** - Load enhanced CSV in your GUI
4. ⏭️ **Add Contest Mode Toggle** - Let users select Cash vs GPP
5. ⏭️ **Implement Stack Builder** - Use `StackingOptimizer` class
6. ⏭️ **Add Bankroll Display** - Show risk calculation in GUI
7. ⏭️ **Test & Iterate** - Generate lineups and analyze results

---

## 📞 Support

All strategies are based on "Fantasy Football For Dummies" Chapters 13-19.

**Key Concepts Implemented:**
- ✅ Contest type differentiation (Ch 14, 19)
- ✅ Value-based selection (Ch 15)
- ✅ Contrarian play (Ch 15)
- ✅ Stacking strategies (Ch 15, 19)
- ✅ Exposure control (Ch 15)
- ✅ Hedge lineups (Ch 15)
- ✅ Bankroll management (Ch 13, 19)

**What Makes This Elite:**
Your optimizer now has the same features used by professional DFS players on DraftKings and FanDuel!

Good luck crushing your contests! 🏈💰

