const { v4: uuidv4 } = require('uuid');

/**
 * NBA DraftKings Optimizer
 * Generates optimized NBA lineups with DraftKings position requirements
 */
class NBAOptimizer {
  constructor() {
    // Prioritize strategies that focus on projections
    this.strategies = ['greedy', 'projection', 'balanced', 'value'];
  }

  async optimize(config) {
    const { 
      players, 
      numLineups, 
      minSalary = 48000, 
      maxSalary = 50000, 
      stackSettings = {},
      uniquePlayers = 7,
      maxExposure = 40,
      monteCarloIterations = 100,
      sortingMethod = 'Points',
      minUniquePlayersBetweenLineups = 3,
      enableRiskManagement = true,
      disableKellySizing = false,
      stackTypes = {},
      exposureSettings = {},
      riskTolerance = 'medium',
      bankroll = 1000,
      advancedQuantSettings = {},
      onProgress 
    } = config;

    // Log quant settings if enabled
    if (advancedQuantSettings && advancedQuantSettings.enabled) {
      console.log('📊 Advanced Quant Settings enabled:', {
        strategy: advancedQuantSettings.strategy,
        riskTolerance: advancedQuantSettings.riskTolerance,
        varConfidence: advancedQuantSettings.varConfidence,
        targetVolatility: advancedQuantSettings.targetVolatility,
        monteCarloSims: advancedQuantSettings.monteCarloSims,
      });
    }

    const results = [];
    
    // Position requirements for DraftKings NBA (Classic 8-man)
    const positionReqs = {
      'PG': 1,
      'SG': 1,
      'SF': 1,
      'PF': 1,
      'C': 1,
      'G': 1,   // Guard (PG or SG)
      'F': 1,   // Forward (SF or PF)
      'UTIL': 1  // Utility (any position)
    };

    // Pre-filter players by position and selection
    const playersByPosition = this.groupPlayersByPosition(players);
    
    // Validate we have enough players for core positions
    const corePositions = ['PG', 'SG', 'SF', 'PF', 'C'];
    for (const pos of corePositions) {
      if (!playersByPosition[pos] || playersByPosition[pos].length < 1) {
        throw new Error(`Not enough players available for position ${pos}. Make sure players are selected and have valid positions.`);
      }
    }

    // Generate lineups using different strategies
    const lineupPool = new Set();
    const exposureTracker = new Map();
    
    for (let i = 0; i < numLineups; i++) {
      if (onProgress && i % Math.max(1, Math.floor(numLineups / 10)) === 0) {
        onProgress(Math.round((i / numLineups) * 100));
      }

      // Use different strategies to create diversity
      const strategy = this.strategies[i % this.strategies.length];
      
      let lineup;
      let attempts = 0;
      const maxAttempts = 50;

      do {
        lineup = this.generateAdvancedLineup(
          playersByPosition, 
          positionReqs, 
          minSalary, 
          maxSalary, 
          strategy,
          stackSettings,
          stackTypes,
          exposureSettings,
          lineupPool,
          exposureTracker,
          maxExposure,
          minUniquePlayersBetweenLineups
        );
        attempts++;
      } while (
        attempts < maxAttempts && 
        (lineup === null || this.isDuplicateLineup(lineup, lineupPool, uniquePlayers))
      );

      if (lineup && lineup.players.length === 8) {
        lineupPool.add(this.getLineupKey(lineup));
        
        // Track exposure
        lineup.players.forEach(player => {
          exposureTracker.set(player.id, (exposureTracker.get(player.id) || 0) + 1);
        });

        results.push({
          id: uuidv4(),
          players: lineup.players,
          totalSalary: lineup.totalSalary,
          totalProjection: lineup.totalProjection,
          totalPoints: lineup.totalProjection, // Alias for compatibility
          strategy: lineup.strategy,
          timestamp: new Date().toISOString()
        });
      }
    }

    if (onProgress) {
      onProgress(100);
    }

    console.log(`✅ Generated ${results.length} NBA lineups`);
    return results;
  }

  groupPlayersByPosition(players) {
    const grouped = {
      'PG': [],
      'SG': [],
      'SF': [],
      'PF': [],
      'C': [],
      'G': [],   // Guard flex (PG or SG)
      'F': [],   // Forward flex (SF or PF)
      'UTIL': [] // Utility (any position)
    };
    
    players.forEach(player => {
      if (!player.position) return;
      
      // Handle multi-position players (e.g., "PG/SG")
      const positions = player.position.split('/');
      
      positions.forEach(pos => {
        pos = pos.trim().toUpperCase();
        
        // Add to specific position
        if (!grouped[pos]) {
          grouped[pos] = [];
        }
        grouped[pos].push(player);
        
        // Also add to flex positions
        if (pos === 'PG' || pos === 'SG') {
          grouped['G'].push(player);  // Guards
        }
        if (pos === 'SF' || pos === 'PF') {
          grouped['F'].push(player);  // Forwards
        }
        
        // All players can fill UTIL
        if (!grouped['UTIL'].some(p => p.id === player.id)) {
          grouped['UTIL'].push(player);
        }
      });
    });

    return grouped;
  }

  generateAdvancedLineup(playersByPosition, positionReqs, minSalary, maxSalary, strategy, stackSettings, stackTypes, exposureSettings, lineupPool, exposureTracker, maxExposure, minUniquePlayersBetweenLineups) {
    const lineup = [];
    let totalSalary = 0;
    let totalProjection = 0;
    const usedPlayerIds = new Set();

    // Fill core positions first (PG, SG, SF, PF, C)
    const corePositions = ['PG', 'SG', 'SF', 'PF', 'C'];
    
    for (const position of corePositions) {
      const player = this.selectPlayerForPosition(
        position,
        playersByPosition,
        usedPlayerIds,
        strategy,
        exposureTracker,
        maxExposure
      );

      if (!player) {
        return null; // Can't fill required position
      }

      lineup.push({ ...player, rosterPosition: position });
      usedPlayerIds.add(player.id);
      totalSalary += player.salary;
      totalProjection += player.projection || 0;
    }

    // Fill G position (any remaining guard)
    const guardPlayer = this.selectPlayerForPosition(
      'G',
      playersByPosition,
      usedPlayerIds,
      strategy,
      exposureTracker,
      maxExposure,
      ['PG', 'SG']
    );

    if (!guardPlayer) {
      return null;
    }

    lineup.push({ ...guardPlayer, rosterPosition: 'G' });
    usedPlayerIds.add(guardPlayer.id);
    totalSalary += guardPlayer.salary;
    totalProjection += guardPlayer.projection || 0;

    // Fill F position (any remaining forward)
    const forwardPlayer = this.selectPlayerForPosition(
      'F',
      playersByPosition,
      usedPlayerIds,
      strategy,
      exposureTracker,
      maxExposure,
      ['SF', 'PF']
    );

    if (!forwardPlayer) {
      return null;
    }

    lineup.push({ ...forwardPlayer, rosterPosition: 'F' });
    usedPlayerIds.add(forwardPlayer.id);
    totalSalary += forwardPlayer.salary;
    totalProjection += forwardPlayer.projection || 0;

    // Fill UTIL position (any remaining player)
    const utilPlayer = this.selectPlayerForPosition(
      'UTIL',
      playersByPosition,
      usedPlayerIds,
      strategy,
      exposureTracker,
      maxExposure,
      ['PG', 'SG', 'SF', 'PF', 'C']
    );

    if (!utilPlayer) {
      return null;
    }

    lineup.push({ ...utilPlayer, rosterPosition: 'UTIL' });
    usedPlayerIds.add(utilPlayer.id);
    totalSalary += utilPlayer.salary;
    totalProjection += utilPlayer.projection || 0;

    // Validate salary constraints
    if (totalSalary < minSalary || totalSalary > maxSalary) {
      return null;
    }

    return {
      players: lineup,
      totalSalary,
      totalProjection,
      strategy
    };
  }

  selectPlayerForPosition(rosterPosition, playersByPosition, usedPlayerIds, strategy, exposureTracker, maxExposure, eligiblePositions = null) {
    // Determine which position pools to look at
    const positionsToCheck = eligiblePositions || [rosterPosition];
    
    // Gather all eligible players from the position pools
    const playerMap = new Map();
    for (const pos of positionsToCheck) {
      if (playersByPosition[pos]) {
        playersByPosition[pos].forEach(player => {
          if (!playerMap.has(player.id)) {
            playerMap.set(player.id, player);
          }
        });
      }
    }
    
    // Convert to array and filter out used players
    let eligiblePlayers = Array.from(playerMap.values()).filter(p => 
      !usedPlayerIds.has(p.id) &&
      (exposureTracker.get(p.id) || 0) < maxExposure
    );

    if (eligiblePlayers.length === 0) {
      return null;
    }

    // Sort by strategy
    eligiblePlayers.sort((a, b) => {
      switch (strategy) {
        case 'greedy':
        case 'projection':
          // Pure projection-based
          return (b.projection || 0) - (a.projection || 0);
        case 'value':
          // Value-based (points per $1000)
          const valueA = (a.projection || 0) / a.salary * 1000;
          const valueB = (b.projection || 0) / b.salary * 1000;
          return valueB - valueA;
        case 'balanced':
          // Balanced approach
          const balanceA = ((a.projection || 0) * 0.7) + (((a.projection || 0) / a.salary * 1000) * 0.3);
          const balanceB = ((b.projection || 0) * 0.7) + (((b.projection || 0) / b.salary * 1000) * 0.3);
          return balanceB - balanceA;
        default:
          return (b.projection || 0) - (a.projection || 0);
      }
    });

    // Use weighted randomness - favor top players but still allow diversity
    // 70% chance to pick from top 2, 30% chance to pick from top 5
    const useTop2 = Math.random() < 0.7;
    const topN = useTop2 ? Math.min(2, eligiblePlayers.length) : Math.min(5, eligiblePlayers.length);
    const randomIndex = Math.floor(Math.random() * topN);
    
    return eligiblePlayers[randomIndex];
  }

  isDuplicateLineup(lineup, lineupPool, uniqueThreshold) {
    const key = this.getLineupKey(lineup);
    if (lineupPool.has(key)) {
      return true;
    }

    // Check for similar lineups (based on unique player threshold)
    for (const existingKey of lineupPool) {
      const existingPlayerIds = new Set(existingKey.split('|'));
      const newPlayerIds = new Set(lineup.players.map(p => p.id));
      
      let commonPlayers = 0;
      for (const id of newPlayerIds) {
        if (existingPlayerIds.has(id)) {
          commonPlayers++;
        }
      }

      if (commonPlayers > (8 - uniqueThreshold)) {
        return true;
      }
    }

    return false;
  }

  getLineupKey(lineup) {
    return lineup.players.map(p => p.id).sort().join('|');
  }
}

module.exports = NBAOptimizer;

