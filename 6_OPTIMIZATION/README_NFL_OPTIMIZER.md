# NFL DFS Genetic Algorithm Optimizer

## Overview

This is a **NFL DraftKings optimizer** using genetic algorithms to create diverse, high-quality lineups. The system has been fully converted from MLB to NFL, with real DraftKings salaries and SportsData.io projections.

---

## 🏈 What Changed from MLB to NFL

### Position Requirements
**MLB (10 players):**
- P (2), C (1), 1B (1), 2B (1), 3B (1), SS (1), OF (3)

**NFL (9 players):**
- QB (1), RB (2), WR (3), TE (1), FLEX (1), DST (1)
- FLEX can be RB/WR/TE

### Salary Cap & Constraints
- **Salary Cap:** $50,000 (same for both sports)
- **Min Salary (MLB):** $45,000
- **Min Salary (NFL):** $48,000 (NFL requires higher minimum spending)

### Data Sources
**MLB:**
- Manual CSV uploads with player stats

**NFL:**
- **DFS Slates API** - Real DraftKings salaries
- **Projections API** - Weekly fantasy point projections
- Automated data fetching and merging

---

## 📁 Key Files

### 1. `genetic_algo_nfl_optimizer.py`
The main optimizer GUI application. **Changes made:**
- ✅ Updated `POSITION_LIMITS` to NFL positions (QB, RB, WR, TE, FLEX, DST)
- ✅ Updated `REQUIRED_TEAM_SIZE` from 10 to 9
- ✅ Changed `MIN_SALARY_DEFAULT` to $48,000
- ✅ Renamed class from `FantasyBaseballApp` to `FantasyFootballApp`
- ✅ Updated window title to "Advanced NFL DFS Optimizer"
- ✅ Changed default contest name to "NFL Main Slate"

### 2. `load_nfl_data_from_api.py`
**NEW FILE** - Fetches and prepares NFL data for the optimizer.

**Features:**
- Fetches DFS slates by date (for real DraftKings salaries)
- Fetches weekly projections (for fantasy point estimates)
- Merges salary + projection data
- Filters to valid DraftKings positions
- Calculates value metrics (Points per $1K)
- Outputs optimizer-ready CSV files

**Usage:**
```python
python3 load_nfl_data_from_api.py
```

**Output:**
- `nfl_week7_draftkings_optimizer.csv` - Ready to load in the optimizer
- `nfl_dfs_slates_2025-10-20.json` - Raw slate data for reference

---

## 🚀 How to Use

### Step 1: Fetch Current Week Data
```bash
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION
python3 load_nfl_data_from_api.py
```

This will:
1. Fetch DraftKings slate for the specified date
2. Fetch fantasy projections for the week
3. Merge and clean the data
4. Create `nfl_week7_draftkings_optimizer.csv`

### Step 2: Run the Optimizer
```bash
python3 genetic_algo_nfl_optimizer.py
```

### Step 3: Load Your Data
1. Click **"Load Players"** in the GUI
2. Select `nfl_week7_draftkings_optimizer.csv`
3. The optimizer will validate NFL positions and salary cap

### Step 4: Configure Settings
- **Number of Lineups:** How many unique lineups to generate
- **Stack Settings:** Team stacking strategies (e.g., QB + WR from same team)
- **Exposure Limits:** Max % of lineups a player can appear in
- **Salary Constraints:** Min/max spending requirements

### Step 5: Generate Lineups
Click **"Generate Lineups"** and the genetic algorithm will:
- Create diverse, optimized lineups
- Respect position requirements (QB, RB, WR, TE, FLEX, DST)
- Stay under $50,000 salary cap
- Maximize projected fantasy points
- Ensure lineup diversity

### Step 6: Export for DraftKings
- **Export CSV** - Upload directly to DraftKings
- Includes all required fields (player names, positions, salaries)

---

## 📊 Data Structure

### Input CSV Format
The optimizer expects this format (created by `load_nfl_data_from_api.py`):

```csv
Name,PlayerID,Position,Team,Salary,FantasyPoints,Value
Baker Mayfield,19790,QB,TB,6600,18.63,2.82
Jared Goff,17922,QB,DET,6000,18.02,3.00
Jahmyr Gibbs,23200,RB,DET,7500,14.26,1.90
Amon-Ra St. Brown,22587,WR,DET,8200,12.60,1.54
Sam LaPorta,23141,TE,DET,4300,7.62,1.77
Seahawks,90,DST,SEA,3000,6.50,2.17
```

### Required Columns
- **Name** - Player full name
- **Position** - Must be: QB, RB, WR, TE, or DST
- **Team** - Team abbreviation (e.g., DET, TB, SEA)
- **Salary** - DraftKings salary ($3,000 - $10,000 range)
- **FantasyPoints** - Projected fantasy points
- **Value** - Points per $1,000 (FantasyPoints / Salary * 1000)

---

## 🔧 Configuration

### Updating for Different Weeks
Edit `load_nfl_data_from_api.py`:

```python
# Settings for upcoming week
DATE = "2025-10-27"  # Sunday games - DFS slate date
SEASON = "2025REG"   # Regular season
WEEK = 8             # Week number
```

### Position Limits (in `genetic_algo_nfl_optimizer.py`)
```python
POSITION_LIMITS = {
    'QB': 1,   # Quarterback
    'RB': 2,   # Running Back
    'WR': 3,   # Wide Receiver
    'TE': 1,   # Tight End
    'FLEX': 1, # RB/WR/TE
    'DST': 1   # Defense/Special Teams
}

FLEX_POSITIONS = ['RB', 'WR', 'TE']
```

---

## 🎯 NFL-Specific Features

### 1. FLEX Position Handling
The optimizer understands that FLEX can be filled by RB, WR, or TE players. The genetic algorithm will try different combinations to maximize points.

### 2. Team Stacking
NFL DFS benefits from stacking (e.g., QB + WR from same team). The optimizer has built-in stack detection and management:
- **QB-WR stacks** - Most common and effective
- **RB-DST anti-correlation** - Avoids pairing a team's RB with their opponent's DST
- **Team exposure limits** - Prevents over-concentration

### 3. Realistic Salary Ranges
- **QB:** $4,000 - $8,000
- **RB:** $3,500 - $10,000
- **WR:** $3,000 - $10,000
- **TE:** $2,500 - $7,000
- **DST:** $2,000 - $4,500

### 4. Game Theory Optimization
The genetic algorithm creates diverse lineups to:
- Maximize your chances in GPP (tournament) play
- Avoid having all your lineups busted by one bad player
- Differentiate from the field (unique player combinations)

---

## 📈 Genetic Algorithm Features

### Diversity Engine
- Creates unique lineups with controlled overlap
- Prevents duplicate or near-duplicate lineups
- Uses player exposure limits across the full lineup set

### Multi-Objective Optimization
1. **Maximize Projected Points**
2. **Maximize Salary Utilization** (reach $48K minimum)
3. **Maximize Lineup Diversity**
4. **Respect Position Requirements**

### Evolution Process
1. **Initial Population** - Generate random valid lineups
2. **Fitness Evaluation** - Score each lineup
3. **Selection** - Keep the best lineups
4. **Crossover** - Combine good lineups
5. **Mutation** - Introduce random changes for diversity
6. **Repeat** - Evolve over multiple generations

---

## 🔑 API Keys

The system uses **SportsData.io** NFL Fantasy API.

Current API key is in `load_nfl_data_from_api.py`:
```python
API_KEY = "1dd5e646265649af87e0d9cdb80d1c8c"
```

**API Endpoints Used:**
1. `DfsSlatesByDate` - Gets DraftKings salaries
2. `PlayerGameProjectionStatsByWeek` - Gets fantasy projections

---

## 📝 Example Workflow

### Sunday Slate Setup (Every Week)
```bash
# 1. Update the date/week in load_nfl_data_from_api.py
# 2. Fetch fresh data
python3 load_nfl_data_from_api.py

# Output:
# ✅ 104 NFL players loaded
# ✅ Real DraftKings salaries from 2025-10-20
# ✅ Fantasy projections for Week 7
# 💾 Saved to: nfl_week7_draftkings_optimizer.csv

# 3. Run optimizer
python3 genetic_algo_nfl_optimizer.py

# 4. In the GUI:
#    - Load nfl_week7_draftkings_optimizer.csv
#    - Set 20 lineups, 30% max exposure
#    - Enable QB-WR stacking
#    - Click "Generate Lineups"

# 5. Export and upload to DraftKings!
```

---

## ✅ Testing

### Verify NFL Positions
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('nfl_week7_draftkings_optimizer.csv')
print(df['Position'].value_counts())
print(f'Total players: {len(df)}')
print(f'Salary range: ${df.Salary.min()} - ${df.Salary.max()}')
"
```

Expected output:
```
WR     39
RB     26
TE     23
QB     12
DST     4
Total players: 104
Salary range: $2800 - $8200
```

---

## 🐛 Troubleshooting

### "No players loaded"
- Check that the CSV has the correct columns
- Verify positions are: QB, RB, WR, TE, DST (case-sensitive)

### "No valid lineups generated"
- Ensure you have enough players at each position
- Check salary cap settings (should be $50,000)
- Verify min salary is set correctly ($48,000 for NFL)

### "API Error: 404 Not Found"
- The date might not have a DFS slate yet
- Try a date closer to Sunday (slates usually post Thursday/Friday)
- Check that the week number matches the season

---

## 🎓 NFL DFS Strategy Tips

1. **Stack QB + WR** - When a QB throws TDs, their WRs also score
2. **Fade RB+DST same game** - If RB scores well, opponent's defense usually doesn't
3. **Target game environments** - High over/under totals = more scoring
4. **Leverage DST pricing** - Defenses are cheap and can boom/bust
5. **FLEX optimization** - Use FLEX for your best value play (RB/WR/TE)
6. **Diversify lineups** - In GPP, you want unique combinations
7. **Weather matters** - Wind/rain hurts passing, helps running

---

## 📦 Dependencies

```bash
pip install requests pandas numpy PyQt5
```

---

## 🎉 Summary

✅ **Converted** genetic_algo_nfl_optimizer.py from MLB to NFL
✅ **Created** automated data loader with real DraftKings salaries
✅ **Integrated** SportsData.io API for projections
✅ **Updated** all position limits and constraints for NFL
✅ **Tested** with Week 7 data (104 players, 5 positions)
✅ **Ready** to generate optimized DraftKings lineups!

---

## 📞 Support

For issues or questions:
1. Check that your CSV has the correct format
2. Verify API key is valid
3. Ensure you're using dates with available DFS slates
4. Check that all positions have enough players

Good luck with your NFL DFS lineups! 🏈💰

