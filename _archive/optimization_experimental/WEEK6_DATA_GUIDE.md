# Week 6 NFL DFS Data - GPP & Cash Games

## Files Created

### 1. **nfl_week6_gpp_enhanced.csv** (Tournament/GPP Optimization)
- **Strategy:** High ceiling, contrarian plays
- **Optimizations:**
  - Emphasizes ceiling projections (30% weight)
  - Boosts low-ownership players (10% bonus for < 40% ownership)
  - Targets high-upside, volatile players
  - Best for tournaments where you need to differentiate

### 2. **nfl_week6_cash_enhanced.csv** (Cash Game Optimization)
- **Strategy:** High floor, consistent plays
- **Optimizations:**
  - Emphasizes floor projections (40% weight)
  - Boosts high-ownership safe plays (5% bonus for > 60% ownership)
  - Targets consistent, reliable players
  - Best for 50/50s, head-to-heads, double-ups

## Data Summary

- **Total Players:** 98 (with DraftKings salaries)
- **Positions:** QB, RB, WR, TE, DST
- **Week:** NFL 2025 Regular Season, Week 6
- **Source:** SportsData.io API (projections + DraftKings salaries)

## Top Players

### GPP (Tournament) Top 5:
1. **Josh Allen** (QB, $7,700) - 26.55 pts proj, 27.32 ceiling
2. **Jayden Daniels** (QB, $6,900) - 23.91 pts proj, 24.61 ceiling
3. **Caleb Williams** (QB, $5,600) - 20.01 pts proj, 20.59 ceiling
4. **Bijan Robinson** (RB, $8,200) - 17.77 pts proj, 18.29 ceiling
5. **Michael Penix Jr.** (QB, $5,000) - 16.35 pts proj, 16.82 ceiling

### Cash Game Top 5:
1. **Josh Allen** (QB, $7,700) - 20.95 pts proj, 18.22 floor
2. **Jayden Daniels** (QB, $6,900) - 18.87 pts proj, 16.41 floor
3. **Caleb Williams** (QB, $5,600) - 15.79 pts proj, 13.73 floor
4. **Bijan Robinson** (RB, $8,200) - 14.02 pts proj, 12.19 floor
5. **Michael Penix Jr.** (QB, $5,000) - 12.90 pts proj, 11.22 floor

## How to Use

### In the Optimizer:

1. **Load the appropriate file:**
   - For GPP/Tournaments → Load `nfl_week6_gpp_enhanced.csv`
   - For Cash Games → Load `nfl_week6_cash_enhanced.csv`

2. **Set lineup count:**
   - GPP: Generate 20-150 lineups for max diversity
   - Cash: Generate 1-5 lineups for top consistency

3. **Configure settings:**
   - GPP: Use higher min_unique (3-5 different players)
   - Cash: Use lower min_unique (1-2 different players)

4. **Select stacks:**
   - GPP: Aggressive stacks (4-5 players from one team)
   - Cash: Conservative stacks (2-3 players from top teams)

## Columns Included

### Core Data:
- `Name` - Player name
- `Position` - QB, RB, WR, TE, DST
- `Team` - NFL team abbreviation
- `Salary` - DraftKings salary
- `Predicted_DK_Points` - Projected fantasy points (contest-optimized)

### Value Metrics:
- `Value` - Points per $1K salary
- `PointsPerK` - Alternative value calculation
- `value_projection` - Projected value
- `value_ceiling` - Best-case value
- `value_floor` - Worst-case value
- `value_rating` - Overall value rating

### Projection Range:
- `ceiling` - High-end projection (80th percentile)
- `floor` - Low-end projection (20th percentile)

### Contest Strategy:
- `ownership` - Estimated ownership % (mock data)
- `ownership_tier` - Ownership category (Very Low to Very High)

### Game Environment:
- `game_total` - Projected total points in game
- `spread` - Point spread
- `implied_points` - Team's projected points
- `is_dome` - Indoor game (True/False)
- `wind` - Wind speed (mph)
- `precip` - Precipitation chance
- `temperature` - Game temperature

### Matchup Data:
- `opponent` - Opponent team
- `opp_def_rank` - Opponent defense rank (1-32)
- `opp_def_rank_pos` - Opponent defense rank vs position

### Trend Data:
- `recent_form` - Recent performance trend
- `trend` - Performance trajectory

## Week 6 vs Week 7

| Feature | Week 6 | Week 7 |
|---------|--------|--------|
| Games Completed | ✅ All games finished | ⏳ Only 1 game (CIN vs PIT) |
| Actual Results | ✅ Available | ⏳ Pending |
| Lineup Scoring | ✅ Can calculate | ❌ Incomplete |
| Best Use | Historical analysis, backtesting | Forward-looking projections |

## Tips

### For GPP:
- Focus on players with high ceiling/projection ratio
- Target low-ownership plays from high-scoring games
- Use diverse stacks to differentiate from field
- Don't be afraid of contrarian plays

### For Cash:
- Prioritize high floor/salary ratio
- Stick with chalk (high-ownership) safe plays
- Use proven studs even at high prices
- Avoid ultra-risky low-priced plays

## File Locations

- **Week 6 GPP:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nfl_week6_gpp_enhanced.csv`
- **Week 6 Cash:** `/Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nfl_week6_cash_enhanced.csv`

## Created

October 18, 2025

---

**Ready to optimize!** Load either file into the genetic algorithm optimizer and generate your lineups! 🏈🚀

