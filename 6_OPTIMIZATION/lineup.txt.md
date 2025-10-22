IZATION/nba_dk_slate_optimizer.py
🏀 NBA DraftKings Slate Optimizer
======================================================================
API Key: d62d0ae315504e53a232...
DK Slate: /Users/sineshawmesfintesfaye/Downloads/DKEntries-3.csv
======================================================================

📋 Step 1: Loading DraftKings Slate Players...
✅ Loaded 420 players from DraftKings slate
   Salary range: $3,000 - $11,100
   Teams: 24 teams

📡 Step 2: Fetching Projections from SportsData.io API...
INFO:root:📥 Fetching NBA projections for 2025-OCT-22...
INFO:root:✅ Retrieved 418 player projections for 2025-OCT-22
✅ Fetched 418 player projections from API

🔗 Step 3: Matching API Projections to DK Slate...
   Available API columns: ['StatID', 'TeamID', 'ID', 'SeasonType', 'Season', 'Name', 'Team', 'Position', 'Started', 'InjuryStatus', 'GameID', 'OpponentID', 'Opponent', 'Day', 'DateTime', 'HomeOrAway', 'Games', 'FantasyPoints', 'Minutes', 'Seconds', 'FieldGoalsMade', 'FieldGoalsAttempted', 'FieldGoalsPercentage', 'TwoPointersMade', 'TwoPointersAttempted', 'TwoPointersPercentage', 'ThreePointersMade', 'ThreePointersAttempted', 'ThreePointersPercentage', 'FreeThrowsMade', 'FreeThrowsAttempted', 'FreeThrowsPercentage', 'OffensiveRebounds', 'DefensiveRebounds', 'ProjectedRebounds', 'ProjectedAssists', 'ProjectedSteals', 'ProjectedBlocks', 'ProjectedTurnovers', 'PersonalFouls', 'ProjectedPoints', 'FantasyPointsFanDuel', 'Predicted_DK_Points', 'PlusMinus', 'DoubleDoubles', 'TripleDoubles', 'Salary', 'Game', 'Name_Normalized']
✅ Matched 330/420 players to API projections
   Using DK averages for remaining 90 players

💰 Step 4: Calculating Value Metrics...
✅ Calculated value metrics for all players

💾 Step 5: Exporting Player Pool...
✅ Exported optimized player pool to:
   /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv

======================================================================
🌟 TOP PLAYERS BY PROJECTED POINTS
======================================================================

🏆 TOP 5 OVERALL:
----------------------------------------------------------------------
Giannis Antetokounmpo     | PF    | MIL | $11,100 |  57.6 pts | Value: 5.19
Victor Wembanyama         | C     | SAS | $10,900 |  54.3 pts | Value: 4.98
Cade Cunningham           | PG    | DET | $9,900 |  51.1 pts | Value: 5.16
Anthony Davis             | PF    | DAL | $9,700 |  48.5 pts | Value: 5.00
Karl-Anthony Towns        | PF    | NYK | $8,500 |  47.7 pts | Value: 5.61

PG:
----------------------------------------------------------------------
Cade Cunningham           | $9,900 |  51.1 pts | Value: 5.16
LaMelo Ball               | $9,500 |  47.1 pts | Value: 4.96
Trae Young                | $9,800 |  46.0 pts | Value: 4.69

SG:
----------------------------------------------------------------------
RJ Barrett                | $7,700 |  39.5 pts | Value: 5.12
Desmond Bane              | $7,100 |  38.2 pts | Value: 5.38
Cam Thomas                | $7,600 |  37.3 pts | Value: 4.91

SF:
----------------------------------------------------------------------
Jalen Johnson             | $8,800 |  41.4 pts | Value: 4.70
Deni Avdija               | $8,500 |  39.9 pts | Value: 4.69
Franz Wagner              | $7,600 |  39.7 pts | Value: 5.23

PF:
----------------------------------------------------------------------
Giannis Antetokounmpo     | $11,100 |  57.6 pts | Value: 5.19
Anthony Davis             | $9,700 |  48.5 pts | Value: 5.00
Karl-Anthony Towns        | $8,500 |  47.7 pts | Value: 5.61

C:
----------------------------------------------------------------------
Victor Wembanyama         | $10,900 |  54.3 pts | Value: 4.98
Joel Embiid               | $8,700 |  43.6 pts | Value: 5.01
Nikola Vucevic            | $8,800 |  40.5 pts | Value: 4.60

======================================================================
💎 TOP VALUE PLAYS (Best Points per $1000)
======================================================================
Dejounte Murray           | PG    | $3,000 |  41.0 pts | Value: 13.68
PJ Washington             | SF    | $3,000 |  26.4 pts | Value: 8.78
Tre Jones                 | PG    | $4,000 |  26.8 pts | Value: 6.71
Dereck Lively II          | C     | $5,300 |  34.0 pts | Value: 6.42
Guerschon Yabusele        | SF    | $3,800 |  23.7 pts | Value: 6.24
Jusuf Nurkic              | C     | $4,000 |  24.6 pts | Value: 6.14
Collin Sexton             | PG    | $5,300 |  30.6 pts | Value: 5.77
Jordan Clarkson           | PG    | $4,900 |  27.9 pts | Value: 5.70
Kelly Olynyk              | C     | $3,700 |  20.9 pts | Value: 5.65
Nick Richards             | C     | $4,100 |  22.9 pts | Value: 5.59

✅ No injured players in slate

======================================================================
📊 SLATE SUMMARY
======================================================================
Total Players: 420
Total Salary Cap: $50,000
Average Salary: $4,648
Average Projection: 14.5 pts
Highest Projection: 57.6 pts (Giannis Antetokounmpo)
API Projections: 330
DK Averages Used: 90

======================================================================
✅ OPTIMIZATION COMPLETE!
======================================================================

Next Steps:
1. Load the exported CSV into your lineup optimizer
2. Consider injury updates before finalizing lineups
3. Use value plays to free up salary for studs
4. Check game totals for high-scoring game stacks
======================================================================
sineshawmesfintesfaye@sineshaws-MacBook-Pro mlb-draftkings-system % /opt/hom
ebrew/bin/python3 /Users/sineshawmesfi
ntesfaye/mlb-draftkings-system/6_OPTIM
IZATION/nba_research_genetic_optimizer
.py
======================================================================
🏀 NBA RESEARCH-BASED GENETIC ALGORITHM OPTIMIZER
======================================================================

Integrating:
  ✅ MIT Paper: Dirichlet-Multinomial opponent modeling
  ✅ Mean-variance optimization for cash games
  ✅ Variance maximization for GPP tournaments
  ✅ Genetic algorithm for lineup diversity
======================================================================

📁 Using player pool: /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv
2025-10-22 12:48:00,270 - INFO - ✅ Loaded 420 players from /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv

======================================================================
💰 GENERATING CASH GAME LINEUPS (50/50s, Double-Ups)
======================================================================
2025-10-22 12:48:00,284 - INFO - 💰 Optimizing 3 cash game lineups...
2025-10-22 12:49:56,995 - INFO -    Median opponent score: 202.21
2025-10-22 12:49:57,770 - INFO -    ✅ Cash Lineup 1: Floor=213.7, Salary=$50,000
2025-10-22 12:49:57,837 - INFO -    ✅ Cash Lineup 2: Floor=213.7, Salary=$50,000
2025-10-22 12:49:57,902 - INFO -    ✅ Cash Lineup 3: Floor=213.7, Salary=$50,000

💵 Cash Lineup 1:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

💵 Cash Lineup 2:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

💵 Cash Lineup 3:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

======================================================================
🏆 GENERATING GPP TOURNAMENT LINEUPS (High Ceiling)
======================================================================
2025-10-22 12:49:57,907 - INFO - 🏆 Optimizing 20 GPP tournament lineups...
2025-10-22 12:49:57,907 - INFO -    🧬 Using Genetic Algorithm for diversity...
2025-10-22 12:50:00,763 - INFO -    ✅ Generated 20 diverse GPP lineups

✅ Generated 20 diverse GPP lineups

🎯 GPP Lineup 1:
                 Name Team  Salary  Ceiling  Est_Ownership
          Isaac Jones  SAC    3000    0.000       8.108108
     D'Angelo Russell  DAL    6400   41.678      56.232065
      Trentyn Flowers  CHI    3000    0.000       8.108108
        Miles Bridges  CHA    7800   54.353      71.856584
     Dereck Lively II  DAL    5300   44.252      55.663672
Giannis Antetokounmpo  MIL   11100   74.932     100.000000
      Kelly Oubre Jr.  PHI    6000   33.163      47.196438
         De'Aaron Fox  SAS    7200   55.289      71.109355
Total Ceiling: 303.7 | Salary: $49,800

🎯 GPP Lineup 2:
              Name Team  Salary  Ceiling  Est_Ownership
          Naz Reid  MIN    6000   36.153      49.989637
       Jalen Green  PHX    6400   44.096      58.490913
     PJ Washington  DAL    3000   34.255      40.108455
     Miles Bridges  CHA    7800   54.353      71.856584
     Bobi Klintman  DET    3000    0.000       8.108108
Karl-Anthony Towns  NYK    8500   62.036      80.925784
         Ben Saraf  BKN    4300   17.121      27.615723
    Isaiah Stevens  SAC    3000    0.000       8.108108
Total Ceiling: 248.0 | Salary: $42,000

🎯 GPP Lineup 3:
                 Name Team  Salary  Ceiling  Est_Ownership
      Jaden McDaniels  MIN    5600   35.919      48.689958
   Karl-Anthony Towns  NYK    8500   62.036      80.925784
Giannis Antetokounmpo  MIL   11100   74.932     100.000000
      Dejounte Murray  NOP    3000   53.339      57.936352
     D'Angelo Russell  DAL    6400   41.678      56.232065
     Brandon Williams  DAL    3500   17.173      25.502138
        Jarrett Allen  CLE    6600   39.962      55.169552
          Tyler Smith  MIL    3000    6.539      14.216713
Total Ceiling: 331.6 | Salary: $47,700
2025-10-22 12:50:00,773 - INFO - 💾 Exported 3 lineups to /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_cash_lineups_20251022_125000.csv
2025-10-22 12:50:00,795 - INFO - 💾 Exported 20 lineups to /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_gpp_lineups_20251022_125000.csv

======================================================================
✅ OPTIMIZATION COMPLETE!
======================================================================

📊 Summary:
  Cash Lineups: 3 (conservative, high win rate)
  GPP Lineups: 20 (high ceiling, diverse)

💾 Files saved:
  /Users/sineshawmesfintesfaye/mlb-draftkings-system/6IZATION/nba_dk_slate_optimizer.py
🏀 NBA DraftKings Slate Optimizer
======================================================================
API Key: d62d0ae315504e53a232...
DK Slate: /Users/sineshawmesfintesfaye/Downloads/DKEntries-3.csv
======================================================================

📋 Step 1: Loading DraftKings Slate Players...
✅ Loaded 420 players from DraftKings slate
   Salary range: $3,000 - $11,100
   Teams: 24 teams

📡 Step 2: Fetching Projections from SportsData.io API...
INFO:root:📥 Fetching NBA projections for 2025-OCT-22...
INFO:root:✅ Retrieved 418 player projections for 2025-OCT-22
✅ Fetched 418 player projections from API

🔗 Step 3: Matching API Projections to DK Slate...
   Available API columns: ['StatID', 'TeamID', 'ID', 'SeasonType', 'Season', 'Name', 'Team', 'Position', 'Started', 'InjuryStatus', 'GameID', 'OpponentID', 'Opponent', 'Day', 'DateTime', 'HomeOrAway', 'Games', 'FantasyPoints', 'Minutes', 'Seconds', 'FieldGoalsMade', 'FieldGoalsAttempted', 'FieldGoalsPercentage', 'TwoPointersMade', 'TwoPointersAttempted', 'TwoPointersPercentage', 'ThreePointersMade', 'ThreePointersAttempted', 'ThreePointersPercentage', 'FreeThrowsMade', 'FreeThrowsAttempted', 'FreeThrowsPercentage', 'OffensiveRebounds', 'DefensiveRebounds', 'ProjectedRebounds', 'ProjectedAssists', 'ProjectedSteals', 'ProjectedBlocks', 'ProjectedTurnovers', 'PersonalFouls', 'ProjectedPoints', 'FantasyPointsFanDuel', 'Predicted_DK_Points', 'PlusMinus', 'DoubleDoubles', 'TripleDoubles', 'Salary', 'Game', 'Name_Normalized']
✅ Matched 330/420 players to API projections
   Using DK averages for remaining 90 players

💰 Step 4: Calculating Value Metrics...
✅ Calculated value metrics for all players

💾 Step 5: Exporting Player Pool...
✅ Exported optimized player pool to:
   /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv

======================================================================
🌟 TOP PLAYERS BY PROJECTED POINTS
======================================================================

🏆 TOP 5 OVERALL:
----------------------------------------------------------------------
Giannis Antetokounmpo     | PF    | MIL | $11,100 |  57.6 pts | Value: 5.19
Victor Wembanyama         | C     | SAS | $10,900 |  54.3 pts | Value: 4.98
Cade Cunningham           | PG    | DET | $9,900 |  51.1 pts | Value: 5.16
Anthony Davis             | PF    | DAL | $9,700 |  48.5 pts | Value: 5.00
Karl-Anthony Towns        | PF    | NYK | $8,500 |  47.7 pts | Value: 5.61

PG:
----------------------------------------------------------------------
Cade Cunningham           | $9,900 |  51.1 pts | Value: 5.16
LaMelo Ball               | $9,500 |  47.1 pts | Value: 4.96
Trae Young                | $9,800 |  46.0 pts | Value: 4.69

SG:
----------------------------------------------------------------------
RJ Barrett                | $7,700 |  39.5 pts | Value: 5.12
Desmond Bane              | $7,100 |  38.2 pts | Value: 5.38
Cam Thomas                | $7,600 |  37.3 pts | Value: 4.91

SF:
----------------------------------------------------------------------
Jalen Johnson             | $8,800 |  41.4 pts | Value: 4.70
Deni Avdija               | $8,500 |  39.9 pts | Value: 4.69
Franz Wagner              | $7,600 |  39.7 pts | Value: 5.23

PF:
----------------------------------------------------------------------
Giannis Antetokounmpo     | $11,100 |  57.6 pts | Value: 5.19
Anthony Davis             | $9,700 |  48.5 pts | Value: 5.00
Karl-Anthony Towns        | $8,500 |  47.7 pts | Value: 5.61

C:
----------------------------------------------------------------------
Victor Wembanyama         | $10,900 |  54.3 pts | Value: 4.98
Joel Embiid               | $8,700 |  43.6 pts | Value: 5.01
Nikola Vucevic            | $8,800 |  40.5 pts | Value: 4.60

======================================================================
💎 TOP VALUE PLAYS (Best Points per $1000)
======================================================================
Dejounte Murray           | PG    | $3,000 |  41.0 pts | Value: 13.68
PJ Washington             | SF    | $3,000 |  26.4 pts | Value: 8.78
Tre Jones                 | PG    | $4,000 |  26.8 pts | Value: 6.71
Dereck Lively II          | C     | $5,300 |  34.0 pts | Value: 6.42
Guerschon Yabusele        | SF    | $3,800 |  23.7 pts | Value: 6.24
Jusuf Nurkic              | C     | $4,000 |  24.6 pts | Value: 6.14
Collin Sexton             | PG    | $5,300 |  30.6 pts | Value: 5.77
Jordan Clarkson           | PG    | $4,900 |  27.9 pts | Value: 5.70
Kelly Olynyk              | C     | $3,700 |  20.9 pts | Value: 5.65
Nick Richards             | C     | $4,100 |  22.9 pts | Value: 5.59

✅ No injured players in slate

======================================================================
📊 SLATE SUMMARY
======================================================================
Total Players: 420
Total Salary Cap: $50,000
Average Salary: $4,648
Average Projection: 14.5 pts
Highest Projection: 57.6 pts (Giannis Antetokounmpo)
API Projections: 330
DK Averages Used: 90

======================================================================
✅ OPTIMIZATION COMPLETE!
======================================================================

Next Steps:
1. Load the exported CSV into your lineup optimizer
2. Consider injury updates before finalizing lineups
3. Use value plays to free up salary for studs
4. Check game totals for high-scoring game stacks
======================================================================
sineshawmesfintesfaye@sineshaws-MacBook-Pro mlb-draftkings-system % /opt/hom
ebrew/bin/python3 /Users/sineshawmesfi
ntesfaye/mlb-draftkings-system/6_OPTIM
IZATION/nba_research_genetic_optimizer
.py
======================================================================
🏀 NBA RESEARCH-BASED GENETIC ALGORITHM OPTIMIZER
======================================================================

Integrating:
  ✅ MIT Paper: Dirichlet-Multinomial opponent modeling
  ✅ Mean-variance optimization for cash games
  ✅ Variance maximization for GPP tournaments
  ✅ Genetic algorithm for lineup diversity
======================================================================

📁 Using player pool: /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv
2025-10-22 12:48:00,270 - INFO - ✅ Loaded 420 players from /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_slate_optimized_20251022_123758.csv

======================================================================
💰 GENERATING CASH GAME LINEUPS (50/50s, Double-Ups)
======================================================================
2025-10-22 12:48:00,284 - INFO - 💰 Optimizing 3 cash game lineups...
2025-10-22 12:49:56,995 - INFO -    Median opponent score: 202.21
2025-10-22 12:49:57,770 - INFO -    ✅ Cash Lineup 1: Floor=213.7, Salary=$50,000
2025-10-22 12:49:57,837 - INFO -    ✅ Cash Lineup 2: Floor=213.7, Salary=$50,000
2025-10-22 12:49:57,902 - INFO -    ✅ Cash Lineup 3: Floor=213.7, Salary=$50,000

💵 Cash Lineup 1:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

💵 Cash Lineup 2:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

💵 Cash Lineup 3:
              Name Team  Salary  Floor  Projected_DK_Points
Karl-Anthony Towns  NYK    8500 33.404                47.72
      De'Aaron Fox  SAS    7200 29.771                42.53
    Scottie Barnes  TOR    8400 29.267                41.81
   Dejounte Murray  NOP    3000 28.721                41.03
      Desmond Bane  ORL    7100 26.733                38.19
   Trey Murphy III  NOP    7500 25.907                37.01
     Collin Sexton  CHA    5300 21.420                30.60
     PJ Washington  DAL    3000 18.445                26.35
Total Floor: 213.7 | Salary: $50,000

======================================================================
🏆 GENERATING GPP TOURNAMENT LINEUPS (High Ceiling)
======================================================================
2025-10-22 12:49:57,907 - INFO - 🏆 Optimizing 20 GPP tournament lineups...
2025-10-22 12:49:57,907 - INFO -    🧬 Using Genetic Algorithm for diversity...
2025-10-22 12:50:00,763 - INFO -    ✅ Generated 20 diverse GPP lineups

✅ Generated 20 diverse GPP lineups

🎯 GPP Lineup 1:
                 Name Team  Salary  Ceiling  Est_Ownership
          Isaac Jones  SAC    3000    0.000       8.108108
     D'Angelo Russell  DAL    6400   41.678      56.232065
      Trentyn Flowers  CHI    3000    0.000       8.108108
        Miles Bridges  CHA    7800   54.353      71.856584
     Dereck Lively II  DAL    5300   44.252      55.663672
Giannis Antetokounmpo  MIL   11100   74.932     100.000000
      Kelly Oubre Jr.  PHI    6000   33.163      47.196438
         De'Aaron Fox  SAS    7200   55.289      71.109355
Total Ceiling: 303.7 | Salary: $49,800

🎯 GPP Lineup 2:
              Name Team  Salary  Ceiling  Est_Ownership
          Naz Reid  MIN    6000   36.153      49.989637
       Jalen Green  PHX    6400   44.096      58.490913
     PJ Washington  DAL    3000   34.255      40.108455
     Miles Bridges  CHA    7800   54.353      71.856584
     Bobi Klintman  DET    3000    0.000       8.108108
Karl-Anthony Towns  NYK    8500   62.036      80.925784
         Ben Saraf  BKN    4300   17.121      27.615723
    Isaiah Stevens  SAC    3000    0.000       8.108108
Total Ceiling: 248.0 | Salary: $42,000

🎯 GPP Lineup 3:
                 Name Team  Salary  Ceiling  Est_Ownership
      Jaden McDaniels  MIN    5600   35.919      48.689958
   Karl-Anthony Towns  NYK    8500   62.036      80.925784
Giannis Antetokounmpo  MIL   11100   74.932     100.000000
      Dejounte Murray  NOP    3000   53.339      57.936352
     D'Angelo Russell  DAL    6400   41.678      56.232065
     Brandon Williams  DAL    3500   17.173      25.502138
        Jarrett Allen  CLE    6600   39.962      55.169552
          Tyler Smith  MIL    3000    6.539      14.216713
Total Ceiling: 331.6 | Salary: $47,700
2025-10-22 12:50:00,773 - INFO - 💾 Exported 3 lineups to /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_cash_lineups_20251022_125000.csv
2025-10-22 12:50:00,795 - INFO - 💾 Exported 20 lineups to /Users/sineshawmesfintesfaye/mlb-draftkings-system/6_OPTIMIZATION/nba_gpp_lineups_20251022_125000.csv

======================================================================
✅ OPTIMIZATION COMPLETE!
======================================================================

📊 Summary:
  Cash Lineups: 3 (conservative, high win rate)
  GPP Lineups: 20 (high ceiling, diverse)

💾 Files saved:
  /Users/sineshawmesfintesfaye/mlb-draftkings-system/6