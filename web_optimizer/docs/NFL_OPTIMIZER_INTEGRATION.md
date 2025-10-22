# NFL DraftKings Optimizer - Frontend Integration

## Overview

This document describes the complete integration of the NFL DraftKings Optimizer into the web-based DFS Optimizer interface. The implementation supports both MLB and NFL sports with dynamic switching and sport-specific configurations.

## Implementation Summary

### Backend Components

#### 1. NFL Optimizer Module (`server/nfl-optimizer.js`)

**Purpose:** Node.js module that bridges the frontend with the Python NFL genetic algorithm optimizer.

**Key Features:**
- **Dual Operation Modes:**
  - Production: Calls Python NFL optimizer (`genetic_algo_nfl_optimizer.py`)
  - Fallback: JavaScript-based optimizer when Python unavailable
  
- **NFL-Specific Logic:**
  - Position requirements: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 DST
  - FLEX handling (RB/WR/TE eligible)
  - NFL stacking strategies (QB-centric stacks)
  
- **Stacking Implementations:**
  - QB + WR
  - QB + 2 WR
  - QB + WR + TE
  - QB + 2 WR + TE
  - QB + WR + RB
  - Game Stack (QB + WR from Team A + WR from Team B)
  - Bring-Back Stack

**Key Methods:**
```javascript
class NFLOptimizer {
  async optimize(config)                    // Main optimization entry point
  async optimizeWithPython(config)          // Python-based optimization
  async optimizeWithJavaScript(config)      // JS fallback optimization
  generateNFLLineup(...)                    // Generate single NFL lineup
  applyNFLStacking(...)                     // Apply NFL-specific stacks
  groupPlayersByPosition(players)           // Group players by position
  analyzeStacks(players)                    // Identify stack patterns
}
```

#### 2. Server Updates (`server/index.js`)

**New Endpoints:**

```javascript
// Set current sport mode
POST /api/set-sport
Body: { sport: "MLB" | "NFL" }

// Get current sport
GET /api/sport

// Run optimization (sport-aware)
POST /api/optimize
Body: {
  sport: "MLB" | "NFL",
  numLineups: number,
  minSalary: number,
  maxSalary: number,
  stackSettings: object,
  ...
}

// Export lineups (sport-aware)
GET /api/export/:format?sport=NFL
```

**Sport Detection:**
- Automatically selects optimizer based on sport parameter
- NFL: Uses `NFLOptimizer` class
- MLB: Uses `MLBOptimizer` class (existing)

### Frontend Components

#### 1. Sport Configuration (`components/sport-config.ts`)

**Purpose:** Centralized sport-specific configuration and utilities.

**Key Exports:**

```typescript
export type Sport = 'MLB' | 'NFL';

export interface SportConfig {
  positions: string[];
  positionLabels: Record<string, string>;
  positionCounts: Record<string, number>;
  stackTypes: string[];
  defaultMinSalary: number;
  maxSalary: number;
  lineupSize: number;
  salaryCapDescription: string;
}

export const SPORT_CONFIGS: Record<Sport, SportConfig>;
```

**Configurations:**

**MLB Configuration:**
```typescript
{
  positions: ['P', 'C', '1B', '2B', '3B', 'SS', 'OF'],
  positionCounts: { P: 2, C: 1, '1B': 1, '2B': 1, '3B': 1, SS: 1, OF: 3 },
  stackTypes: ['Same Team (2+)', 'Same Team (3+)', 'Same Team (4+)', ...],
  defaultMinSalary: 45000,
  lineupSize: 10,
  ...
}
```

**NFL Configuration:**
```typescript
{
  positions: ['QB', 'RB', 'WR', 'TE', 'FLEX', 'DST'],
  positionCounts: { QB: 1, RB: 2, WR: 3, TE: 1, FLEX: 1, DST: 1 },
  stackTypes: ['QB + WR', 'QB + 2 WR', 'QB + WR + TE', 'QB + 2 WR + TE', ...],
  defaultMinSalary: 48000,
  lineupSize: 9,
  ...
}
```

**Utility Functions:**
- `getPositionFilters(sport)` - Get position filter options
- `filterPlayersByPosition(players, position, sport)` - Filter with sport logic
- `getPositionCount(players, position, sport)` - Count players by position
- `validateLineup(players, sport)` - Validate lineup structure
- `formatLineupForExport(lineup, sport)` - Format for DK export

#### 2. Updated DFS Optimizer Component (`components/DFSOptimizer.tsx`)

**New State:**
```typescript
const [currentSport, setCurrentSport] = useState<Sport>('MLB');
const sportConfig = SPORT_CONFIGS[currentSport];
```

**Sport Selector UI:**
Located in Control Panel (top section):
```tsx
<Select value={currentSport} onValueChange={handleSportChange}>
  <SelectItem value="MLB">⚾ MLB (Baseball)</SelectItem>
  <SelectItem value="NFL">🏈 NFL (Football)</SelectItem>
</Select>
```

**Dynamic Adjustments:**
- Min Salary: Auto-adjusts based on sport ($45k MLB, $48k NFL)
- Stack Settings: Initializes with sport-specific stack types
- Position Filters: Shows relevant positions for selected sport
- Player Data: Clears when switching sports

**Sport Change Handler:**
```typescript
const handleSportChange = (newSport: Sport) => {
  setCurrentSport(newSport);
  setMinSalary(SPORT_CONFIGS[newSport].defaultMinSalary);
  setStackSettings(initializeStackSettings(newSport));
  setPlayerData([]);
  setSelectedPlayers([]);
};
```

#### 3. Updated Players Tab

**Props:**
```typescript
interface PlayersTabProps {
  playerData: Player[];
  selectedPlayers: string[];
  sport: Sport;  // NEW
  onPlayersChange: (players: string[]) => void;
  onPlayerDataChange: (players: Player[]) => void;
}
```

**Dynamic Position Tabs:**
- MLB: "All Batters", "C", "1B", "2B", "3B", "SS", "OF", "P"
- NFL: "All Offense", "QB", "RB", "WR", "TE", "FLEX", "DST"

**Position Filtering:**
```typescript
const filteredPlayers = filterPlayersByPosition(playerData, positionFilter, sport);
```

#### 4. Updated Stack Exposure Tab

**Props:**
```typescript
interface StackExposureTabProps {
  stackSettings: StackType[];
  sport: Sport;  // NEW
  onStackSettingsChange: (settings: StackType[]) => void;
}
```

**Dynamic Stack Types:**
- Displays sport-specific stack options
- MLB: Same team stacks, Pitcher + Batter
- NFL: QB-centric stacks, Game stacks, Bring-back

## Usage Guide

### For Users

#### Switching Sports

1. Open DFS Optimizer
2. In Control Panel (right side), find "Sport" section at top
3. Select "⚾ MLB (Baseball)" or "🏈 NFL (Football)"
4. Interface automatically adjusts:
   - Position filters
   - Stack types
   - Minimum salary
   - Lineup size

#### Loading NFL Data

1. Select "NFL" sport
2. Click "Load CSV" in File Operations
3. Upload CSV with columns:
   ```
   Name, Position, Team, Salary, Predicted_DK_Points, Ownership
   ```
4. Positions must be: QB, RB, WR, TE, DST
5. System validates data against NFL requirements

#### NFL Stack Configuration

1. Go to "Stack Exposure" tab
2. Select desired NFL stacks:
   - **QB + WR**: Safe, high correlation
   - **QB + 2 WR**: Aggressive ceiling play
   - **QB + WR + TE**: Pass-heavy game stack
   - **QB + 2 WR + TE**: Full passing game exposure
   - **Game Stack**: QB + WR (Team A) + WR (Team B) for shootouts
3. Set exposure percentages (min/max)
4. Enable stacks with checkboxes

#### Running NFL Optimization

1. Ensure players selected (minimum 9 for NFL)
2. Configure settings:
   - Lineups: Number to generate (e.g., 20)
   - Min Salary: Default $48,000 (adjustable)
   - Min Unique: Players that must differ between lineups
3. Click "Optimize" button
4. Wait for completion (progress shown)
5. Review results in "My Entries" tab

#### Exporting NFL Lineups

1. After optimization complete
2. Click "Save CSV" button
3. Select export format:
   - **DraftKings**: DK-ready format with correct positions
   - **Standard**: Detailed analysis format
4. File downloads as `nfl_draftkings_lineups.csv`
5. Upload directly to DraftKings

### For Developers

#### Adding New Stack Types

**Backend** (`server/nfl-optimizer.js`):
```javascript
applyNFLStacking(playersByPosition, stackSettings, ...) {
  const { types } = stackSettings;
  const stackType = types[0]; // e.g., "QB + 2 WR + RB"
  
  // Get QB from team
  const qb = teamQBs[0];
  
  // Get stackplayers based on type
  if (stackType === 'YOUR_NEW_STACK') {
    // Add your logic here
  }
  
  return { players: stackPlayers, ... };
}
```

**Frontend** (`components/sport-config.ts`):
```typescript
NFL: {
  stackTypes: [
    'QB + WR',
    'QB + 2 WR',
    'YOUR_NEW_STACK_TYPE',  // Add here
    ...
  ],
  ...
}
```

#### Modifying Position Requirements

**Backend** (`server/nfl-optimizer.js`):
```javascript
const positionReqs = {
  'QB': 1,
  'RB': 2,  // Modify counts here
  'WR': 3,
  'TE': 1,
  'FLEX': 1,
  'DST': 1
};
```

**Frontend** (`components/sport-config.ts`):
```typescript
NFL: {
  positionCounts: {
    'QB': 1,
    'RB': 2,  // Modify counts here
    'WR': 3,
    'TE': 1,
    'FLEX': 1,
    'DST': 1
  },
  ...
}
```

#### API Integration

**Calling Optimization API:**
```typescript
const response = await fetch('/api/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sport: 'NFL',
    numLineups: 20,
    minSalary: 48000,
    maxSalary: 50000,
    stackSettings: {
      enabled: true,
      teams: ['KC', 'BUF', 'SF'],
      types: ['QB + 2 WR + TE']
    },
    ...
  })
});

const { lineups, summary } = await response.json();
```

## Technical Details

### Position Logic

**MLB vs NFL Differences:**

| Aspect | MLB | NFL |
|--------|-----|-----|
| Lineup Size | 10 players | 9 players |
| Multi-Position | Yes (e.g., "1B/OF") | No (single position) |
| Flex Logic | P or UTIL | RB/WR/TE only |
| Position Split | Pitchers & Batters | Offense & Defense |
| Filtering | String contains check | Exact match |

**NFL FLEX Handling:**
```javascript
// Fill FLEX with best remaining RB/WR/TE
const flexEligible = ['RB', 'WR', 'TE'];
let flexPlayer = null;

for (const pos of flexEligible) {
  const player = selectPlayerForPosition(...);
  if (player.value > bestValue) {
    flexPlayer = player;
  }
}
```

### Stack Correlation

**NFL Stack Correlations:**
- QB + WR (same team): +0.85 correlation
- QB + 2 WR (same team): +0.75 correlation
- QB + WR + TE (same team): +0.70 correlation
- Game Stack (opposing teams): +0.45 correlation

**Stack Selection Logic:**
```javascript
// Prioritize QB from selected teams
const teamQBs = playersByPosition['QB']
  .filter(p => selectedTeams.includes(p.team))
  .sort((a, b) => b.projection - a.projection);

const qb = teamQBs[0];

// Get correlated players from same team
const teamWRs = playersByPosition['WR']
  .filter(p => p.team === qb.team)
  .sort((a, b) => b.projection - a.projection);

// Build stack
const stack = [qb, ...teamWRs.slice(0, 2)];
```

### Export Format

**NFL DraftKings CSV Format:**
```csv
QB,RB,RB,WR,WR,WR,TE,FLEX,DST
Patrick Mahomes,Christian McCaffrey,Saquon Barkley,Tyreek Hill,Cooper Kupp,Davante Adams,Travis Kelce,Derrick Henry,49ers
```

**Position Order (Critical):**
1. QB (best QB)
2. RB (best RB)
3. RB (2nd best RB)
4. WR (best WR)
5. WR (2nd best WR)
6. WR (3rd best WR)
7. TE (best TE)
8. FLEX (best remaining RB/WR/TE)
9. DST (selected defense)

## Testing

### Unit Tests

**Test Sport Switching:**
```typescript
test('should update minSalary when sport changes', () => {
  handleSportChange('NFL');
  expect(minSalary).toBe(48000);
  
  handleSportChange('MLB');
  expect(minSalary).toBe(45000);
});
```

**Test Position Filtering:**
```typescript
test('NFL position filter shows correct positions', () => {
  const filters = getPositionFilters('NFL');
  expect(filters).toContain('QB');
  expect(filters).toContain('RB');
  expect(filters).not.toContain('P');
});
```

### Integration Tests

**Test End-to-End Optimization:**
```bash
# Start server
npm run server

# Send optimization request
curl -X POST http://localhost:5001/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "sport": "NFL",
    "numLineups": 5,
    "minSalary": 48000,
    "players": [...]
  }'

# Verify response
# Should return 5 NFL lineups with 9 players each
```

## Troubleshooting

### Common Issues

**Issue: "Not enough players for position QB"**
- **Cause:** Insufficient QB data loaded
- **Solution:** Load CSV with at least 5-10 QBs

**Issue: "Lineups have wrong position order"**
- **Cause:** Position sorting disabled
- **Solution:** Ensure `fix_lineup_position_order()` called before export

**Issue: "FLEX position has QB/DST"**
- **Cause:** FLEX eligibility not restricted
- **Solution:** Check `flexEligible` array contains only ['RB', 'WR', 'TE']

**Issue: "Stack not appearing in lineups"**
- **Cause:** Insufficient players from selected teams
- **Solution:** Select teams with more available players (5+ per team)

### Debug Mode

**Enable Backend Logging:**
```javascript
// In nfl-optimizer.js
console.log('🏈 NFL Optimizer Debug:', {
  players: players.length,
  stackType: stackSettings.types,
  positionCounts: this.groupPlayersByPosition(players)
});
```

**Frontend Debug:**
```typescript
// In DFSOptimizer.tsx
useEffect(() => {
  console.log('Sport changed:', currentSport);
  console.log('Stack settings:', stackSettings);
  console.log('Min salary:', minSalary);
}, [currentSport, stackSettings, minSalary]);
```

## Performance

### Optimization Speed

**Target Performance:**
- 20 lineups: < 10 seconds
- 100 lineups: < 30 seconds
- 500 lineups: < 2 minutes

**Optimization:**
- Python optimizer (production): Fastest, uses PuLP linear programming
- JavaScript optimizer (fallback): Slower, but no Python dependency
- Parallel processing: Uses ProcessPoolExecutor for multi-core

### Memory Usage

**Typical Usage:**
- 100 players loaded: ~5 MB
- 100 lineups generated: ~10 MB
- Total frontend: ~50 MB

**Optimization:**
- Player data cached in memory
- Lineups streamed to frontend
- Cleanup on sport switch

## Future Enhancements

### Planned Features

1. **Real-Time Ownership Data**
   - Integration with RotoGrinders API
   - Live ownership updates during optimization

2. **Multi-Slate Optimization**
   - Main slate + Showdown simultaneously
   - Cross-slate correlation analysis

3. **Live Swap Functionality**
   - Replace player in lineup
   - Maintain stack integrity
   - Stay under salary cap

4. **Advanced Analytics**
   - Win probability calculations
   - ROI projections
   - Historical performance tracking

5. **Mobile Optimization**
   - Responsive design for tablets/phones
   - Touch-optimized controls
   - Mobile-specific layouts

## API Reference

### Endpoints

#### POST /api/set-sport
Set current sport mode.

**Request:**
```json
{
  "sport": "NFL"
}
```

**Response:**
```json
{
  "success": true,
  "sport": "NFL"
}
```

#### GET /api/sport
Get current sport mode.

**Response:**
```json
{
  "sport": "NFL"
}
```

#### POST /api/optimize
Run lineup optimization.

**Request:**
```json
{
  "sport": "NFL",
  "numLineups": 20,
  "minSalary": 48000,
  "maxSalary": 50000,
  "stackSettings": {
    "enabled": true,
    "teams": ["KC", "BUF"],
    "types": ["QB + 2 WR"]
  },
  "uniquePlayers": 7,
  "maxExposure": 40,
  "contestMode": "gpp",
  "riskTolerance": "medium"
}
```

**Response:**
```json
{
  "success": true,
  "sport": "NFL",
  "optimizationId": "uuid-here",
  "lineups": [
    {
      "id": "lineup-1",
      "players": [...],
      "totalSalary": 49800,
      "totalProjection": 145.6,
      "value": 2.93,
      "stacks": [...]
    }
  ],
  "summary": {
    "totalLineups": 20,
    "avgProjection": 142.3,
    "avgSalary": 49600,
    "topProjection": 148.9
  }
}
```

#### GET /api/export/:format
Export lineups to CSV.

**Query Params:**
- `sport`: "MLB" | "NFL"

**Formats:**
- `draftkings`: DK-ready CSV
- `standard`: Detailed analysis CSV

**Response:**
- CSV file download

## Resources

### Documentation
- [Implementation Guide](./6_OPTIMIZATION/DOCUMENTATION/IMPLEMENTATION_GUIDE.md)
- [DFS Strategy Guide](./6_OPTIMIZATION/DOCUMENTATION/DFS_STRATEGY_GUIDE.md)
- [Python Optimizer](./6_OPTIMIZATION/genetic_algo_nfl_optimizer.py)

### External Links
- [DraftKings NFL Rules](https://www.draftkings.com/help/rules/nfl)
- [PuLP Documentation](https://coin-or.github.io/pulp/)
- [NFL DFS Strategy](https://rotogrinders.com/lessons/nfl-dfs-strategy)

## Changelog

### Version 2.0 (October 2025)
- ✅ Added NFL support
- ✅ Sport selector in UI
- ✅ Dynamic position filtering
- ✅ NFL-specific stacking
- ✅ Sport-aware export
- ✅ Backend optimizer integration

### Version 1.0 (September 2025)
- ✅ MLB optimizer
- ✅ Web interface
- ✅ Desktop-style UI
- ✅ Team combinations
- ✅ Advanced quant features

---

**Last Updated:** October 22, 2025
**Version:** 2.0
**Author:** DFS Optimizer Team


