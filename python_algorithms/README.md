# Python Algorithms & API Scripts

This folder contains Python utilities for data processing and API integrations.

## Files

### 1. `sportsdata_nfl_api.py`
SportsData.io NFL API integration for fetching player game statistics.

#### Setup

1. **Get your API key:**
   - Sign up at https://sportsdata.io/
   - Get your free API key

2. **Install requirements:**
   ```bash
   pip install requests pandas
   ```

3. **Set your API key:**
   - Edit the script and replace `YOUR_API_KEY_HERE` with your actual key
   - OR set environment variable: `export SPORTSDATA_API_KEY="your_key"`

#### Usage

**Interactive Mode (Easiest):**
```bash
python3 sportsdata_nfl_api.py
# Choose option 2, then enter your API key when prompted
```

**In Your Own Code:**
```python
from sportsdata_nfl_api import SportsDataNFLAPI

# Initialize
api = SportsDataNFLAPI("your_api_key_here")

# Get player stats for Week 7 of 2024 Regular Season
stats = api.get_player_stats_by_week("2024REG", 7)

# Get top 10 quarterbacks
top_qbs = api.get_top_scorers("2024REG", 7, position="QB", top_n=10)

# Save to CSV
api.save_to_csv("2024REG", 7, filename="week7_stats.csv")
```

#### Season Formats
- Regular Season: `"2024REG"`
- Preseason: `"2024PRE"`
- Postseason: `"2024POST"`

#### Week Numbers
- Preseason: 0-4
- Regular Season: 1-18
- Postseason: 1-4

---

### 2. `squares_sorted_array.py`
Algorithm for squaring and sorting array elements efficiently using two-pointer technique.

**Run it:**
```bash
python3 squares_sorted_array.py
```

Time Complexity: O(n)  
Space Complexity: O(n)

---

## Quick Start

```bash
# Navigate to this folder
cd /Users/sineshawmesfintesfaye/mlb-draftkings-system/python_algorithms

# Install dependencies
pip install requests pandas

# Run the NFL API script
python3 sportsdata_nfl_api.py
```

---

## API Response Example

The NFL API returns player stats like this:
```json
{
  "PlayerID": 12345,
  "Name": "Patrick Mahomes",
  "Position": "QB",
  "Team": "KC",
  "FantasyPoints": 28.5,
  "PassingYards": 352,
  "PassingTouchdowns": 3,
  "RushingYards": 22,
  "Interceptions": 0
}
```

Full list of fields: PassingYards, RushingYards, ReceivingYards, Touchdowns, FantasyPoints, and many more!

