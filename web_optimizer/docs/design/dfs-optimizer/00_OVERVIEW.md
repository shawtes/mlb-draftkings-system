# DFS MLB Optimizer GUI - Design Documentation
## Complete System Overview

**Version:** 1.0  
**Last Updated:** 2025-10-17  
**Purpose:** Comprehensive design specification for DFS MLB lineup optimization application

---

## Table of Contents

1. [Overview](#overview) - This Document
2. [Architecture & Layout](01_ARCHITECTURE.md)
3. [Players Tab](02_PLAYERS_TAB.md)
4. [Team Stacks Tab](03_TEAM_STACKS_TAB.md)
5. [Stack Exposure Tab](04_STACK_EXPOSURE_TAB.md)
6. [Team Combinations Tab](05_TEAM_COMBINATIONS_TAB.md)
7. [Advanced Quant Tab](06_ADVANCED_QUANT_TAB.md)
8. [My Entries Tab](07_MY_ENTRIES_TAB.md)
9. [Control Panel](08_CONTROL_PANEL.md)
10. [Data Flow & Interactions](09_DATA_FLOW.md)

---

## System Purpose

The DFS MLB Optimizer is a professional-grade daily fantasy sports lineup optimization application that uses advanced algorithms and financial modeling techniques to generate optimal MLB lineups for DraftKings contests.

### Core Capabilities

1. **Multi-Algorithm Optimization**
   - Linear Programming (PuLP)
   - Genetic Algorithm Diversity Engine
   - Monte Carlo Simulation
   - Risk Management (Kelly Criterion, Sharpe Ratio)
   - Probability-Based Optimization

2. **Advanced Features**
   - Team stacking strategies (2-5 player stacks)
   - Complex multi-stack combinations (e.g., 4|2|2)
   - Player exposure management
   - Salary cap optimization with minimum salary constraints
   - Lineup diversity controls (min unique players)
   - Contest-specific optimization (Cash games vs GPP)

3. **Professional Workflow**
   - Import player projections from CSV
   - Load DraftKings entry files
   - Generate 1-500+ unique lineups
   - Export to DraftKings-ready format
   - Favorites management for multi-session workflow
   - Real-time optimization with progress tracking

---

## Technology Stack

### Core Technologies
- **Python 3.12+**
- **PyQt5** - GUI Framework
- **PuLP 3.3.0** - Linear Programming Solver
- **Pandas 2.0+** - Data manipulation
- **NumPy 1.24+** - Numerical computations

### Optional Advanced Modules
- **ARCH** - GARCH volatility modeling
- **SciPy** - Statistical optimization
- **Copulas** - Dependency modeling (optional)

---

## User Personas

### Primary User: DFS Professional
- **Experience Level:** Advanced
- **Goals:**
  - Generate 50-500 lineups per slate
  - Maximize expected value while managing risk
  - Use custom projections and stacking strategies
  - Enter multiple contests efficiently

### Secondary User: Recreational Player
- **Experience Level:** Intermediate
- **Goals:**
  - Create 5-20 lineups for fun
  - Use simple stacking strategies
  - Understand basic optimization concepts
  - Quick contest entry

---

## Key Design Principles

1. **Professional First**
   - Advanced features prominently accessible
   - No hand-holding for basic concepts
   - Assumes user understands DFS fundamentals

2. **Efficiency Over Simplicity**
   - Batch operations preferred
   - Keyboard shortcuts for power users
   - Minimal clicks for common workflows

3. **Transparency**
   - Show all optimization parameters
   - Display detailed statistics and metrics
   - Log optimization decisions

4. **Flexibility**
   - Support multiple optimization strategies
   - Allow fine-grained control over all parameters
   - Enable custom workflows

5. **Performance**
   - Handle large player pools (200+ players)
   - Generate hundreds of lineups quickly
   - Responsive UI during optimization

---

## Application States

### 1. Initial State
- No data loaded
- All optimization controls disabled
- Tab structure visible but empty

### 2. Data Loaded State
- Player data imported
- Position tables populated
- Team stack tables populated
- Optimization controls enabled

### 3. Optimizing State
- Progress indicator active
- Optimization parameters locked
- Background worker thread running
- Cancel option available

### 4. Results State
- Optimized lineups displayed
- Exposure statistics calculated
- Export options enabled
- Ready for next optimization run

### 5. Favorites Management State
- Multiple optimization runs stored
- Lineups can be tagged/grouped
- Cross-session persistence

---

## File Formats

### Input: Player Projections CSV
**Required Columns:**
- `Name` - Player full name
- `Team` - Team abbreviation (e.g., "NYY", "LAD")
- `Position` - DraftKings position (P, C, 1B, 2B, 3B, SS, OF)
- `Salary` - DraftKings salary (integer)
- `Predicted_DK_Points` - Projected fantasy points (float)

**Optional Columns:**
- `ID` or `player_id` - DraftKings player ID
- `Prob_Over_X` - Probability columns for probability-based optimization
- `Value` - Points per $1000 salary
- Any custom metric columns

### Input: DraftKings Entries File
**Format:** DraftKings contest entry CSV
**Structure:**
```
Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
4766459650, MLB Main Slate, 162584429, $1, , , , , , , , , , 
```

### Output: Optimized Lineups CSV
**Two Formats Supported:**

1. **Simple Format** (10 columns)
   ```
   P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
   Player Name 1, Player Name 2, ...
   ```

2. **DraftKings Contest Format** (14 columns)
   ```
   Entry ID, Contest Name, Contest ID, Entry Fee, P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
   4766459650, MLB Main Slate, 162584429, $1, 12345678, 23456789, ...
   ```

---

## Optimization Constraints

### DraftKings MLB Rules
- **Salary Cap:** $50,000
- **Total Players:** 10
- **Position Requirements:**
  - Pitchers (P): 2
  - Catcher (C): 1
  - First Base (1B): 1
  - Second Base (2B): 1
  - Third Base (3B): 1
  - Shortstop (SS): 1
  - Outfielders (OF): 3

### Application Constraints
- **Minimum Salary:** Configurable (default: $45,000)
- **Maximum Salary:** $50,000 (fixed)
- **Min Unique Players:** 0-10 between lineups
- **Player Exposure:** 0-100% per player
- **Stack Requirements:** 2-5 players from same team
- **Lineup Count:** 1-500 lineups per optimization

---

## Performance Targets

### Speed
- Load CSV: < 2 seconds for 200 players
- Generate 100 lineups: < 30 seconds
- Generate 500 lineups: < 3 minutes
- Export CSV: < 1 second

### Memory
- Base application: < 200 MB
- With 500 lineups loaded: < 500 MB
- Peak during optimization: < 1 GB

### UI Responsiveness
- Table sorting: < 100ms
- Tab switching: < 50ms
- Control updates: Instant (< 16ms)

---

## Error Handling

### Common Errors
1. **Insufficient Players** - Not enough eligible players for position
2. **No Valid Lineups** - Constraints too restrictive
3. **File Format Error** - CSV doesn't match expected format
4. **Optimization Timeout** - No solution found in time limit
5. **Export Error** - Cannot write to file system

### Error Display
- **Modal Dialogs** for critical errors
- **Status Bar** for warnings
- **Console Log** for debugging information
- **Inline Validation** for user inputs

---


---

## Document Conventions

Throughout this documentation:

- **Bold** = Component names, UI elements, important terms
- `Code` = Variable names, file paths, code references
- *Italic* = User actions, states, emphasis
- > Quotes = User feedback, status messages
- • Bullets = Feature lists
- → Arrows = Process flows

