const { v4: uuidv4 } = require('uuid');
const QuantEngine = require('./quant-engine');

/**
 * NBA DraftKings Optimizer
 * Generates optimized NBA lineups with DraftKings position requirements
 * and institutional-grade quantitative optimization
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
      contestMode = 'gpp',
      onProgress 
    } = config;

    // Initialize quant engine
    const quantEnabled = advancedQuantSettings && advancedQuantSettings.enabled;
    const quant = quantEnabled ? new QuantEngine({ ...advancedQuantSettings, sport: 'NBA' }) : null;

    if (quantEnabled) {
      console.log('📊 NBA Quant Engine ACTIVE:', {
        strategy: advancedQuantSettings.strategy,
        riskTolerance: advancedQuantSettings.riskTolerance,
        varConfidence: advancedQuantSettings.varConfidence,
        monteCarloSims: advancedQuantSettings.monteCarloSims,
        maxKellyFraction: advancedQuantSettings.maxKellyFraction,
      });
    }

    // Pre-compute quant scores for all players when quant is enabled
    const quantScores = quantEnabled ? quant.scorePlayersQuant(players, contestMode) : null;
    
    // Compute Kelly-optimal exposure limits when enabled
    const kellyLimits = (quantEnabled && advancedQuantSettings.strategy !== 'equal_weight')
      ? quant.kellyExposureLimits(players, maxExposure)
      : null;

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

    // Generate 3x candidates when quant enabled for portfolio selection
    const candidateMultiplier = quantEnabled ? 3 : 1;
    const candidateCount = numLineups * candidateMultiplier;

    // Generate lineups using different strategies
    const lineupPool = new Set();
    const exposureTracker = new Map();

    for (let i = 0; i < candidateCount; i++) {
      if (onProgress && i % Math.max(1, Math.floor(candidateCount / 10)) === 0) {
        onProgress(Math.round((i / candidateCount) * 80)); // reserve 80-100% for portfolio selection
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
          minUniquePlayersBetweenLineups,
          quantEnabled,
          quant,
          quantScores,
          kellyLimits
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

        // Build result object
        const result = {
          id: uuidv4(),
          players: lineup.players,
          totalSalary: lineup.totalSalary,
          totalProjection: lineup.totalProjection,
          totalPoints: lineup.totalProjection, // Alias for compatibility
          value: lineup.totalSalary > 0 ? lineup.totalProjection / lineup.totalSalary * 1000 : 0,
          strategy: lineup.strategy,
          timestamp: new Date().toISOString()
        };

        results.push(result);
      }
    }

    // Portfolio optimization: MC + selection when quant enabled, else just sort
    let portfolioMetrics = null;
    if (quantEnabled && quant && results.length > 0) {
      if (candidateMultiplier > 1) {
        console.log(`📊 NBA portfolio selection: ${results.length} candidates → ${numLineups} optimal`);
      }
      const portfolio = quant.portfolioOptimize(results, numLineups, contestMode, players);
      const selected = portfolio.selected;
      portfolioMetrics = portfolio.portfolioMetrics;
      console.log('📈 NBA Portfolio Metrics:', portfolioMetrics);

      if (onProgress) onProgress(100);
      console.log(`✅ Generated ${selected.length} NBA lineups (quant-optimized from ${results.length} candidates)`);
      selected.portfolioMetrics = portfolioMetrics;
      return selected;
    }

    results.sort((a, b) => b.totalProjection - a.totalProjection);

    if (onProgress) onProgress(100);
    console.log(`✅ Generated ${results.length} NBA lineups`);

    results.portfolioMetrics = portfolioMetrics;
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

  /**
   * Compute the minimum possible salary to fill a set of remaining positions.
   * Used to ensure we don't overspend early and leave positions unfillable.
   */
  _getMinSalaryForPositions(positions, playersByPosition, usedPlayerIds) {
    let minTotal = 0;
    const tempUsed = new Set(usedPlayerIds);

    for (const { pos, eligible } of positions) {
      const positionsToCheck = eligible || [pos];
      let cheapest = null;

      for (const p of positionsToCheck) {
        if (!playersByPosition[p]) continue;
        for (const player of playersByPosition[p]) {
          if (tempUsed.has(player.id)) continue;
          if (!cheapest || player.salary < cheapest.salary) {
            cheapest = player;
          }
        }
      }

      if (cheapest) {
        minTotal += cheapest.salary;
        tempUsed.add(cheapest.id);
      } else {
        minTotal += 3000; // fallback minimum salary estimate
      }
    }
    return minTotal;
  }

  generateAdvancedLineup(playersByPosition, positionReqs, minSalary, maxSalary, strategy, stackSettings, stackTypes, exposureSettings, lineupPool, exposureTracker, maxExposure, minUniquePlayersBetweenLineups, quantEnabled, quant, quantScores, kellyLimits) {
    const lineup = [];
    let totalSalary = 0;
    let totalProjection = 0;
    const usedPlayerIds = new Set();

    // All 8 positions in fill order with their eligible position pools
    const allSlots = [
      { pos: 'PG', eligible: null },
      { pos: 'SG', eligible: null },
      { pos: 'SF', eligible: null },
      { pos: 'PF', eligible: null },
      { pos: 'C', eligible: null },
      { pos: 'G', eligible: ['PG', 'SG'] },
      { pos: 'F', eligible: ['SF', 'PF'] },
      { pos: 'UTIL', eligible: ['PG', 'SG', 'SF', 'PF', 'C'] }
    ];

    // Shuffle the first 5 core positions to eliminate position-order bias
    for (let i = 4; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allSlots[i], allSlots[j]] = [allSlots[j], allSlots[i]];
    }

    for (let slotIdx = 0; slotIdx < allSlots.length; slotIdx++) {
      const { pos, eligible } = allSlots[slotIdx];
      const remainingSlots = allSlots.slice(slotIdx + 1);
      const remainingBudget = maxSalary - totalSalary;

      // Compute min salary needed for all positions AFTER this one
      const minSalaryForRemaining = this._getMinSalaryForPositions(
        remainingSlots, playersByPosition, usedPlayerIds
      );

      // Max we can spend on this slot = remaining budget minus what we need for the rest
      const maxForThisSlot = remainingBudget - minSalaryForRemaining;

      // For the last slot, also enforce the salary floor
      const slotsLeft = allSlots.length - slotIdx - 1;
      let minForThisSlot = 0;
      if (slotsLeft === 0) {
        // Last slot: check if we need a minimum to reach the salary floor
        minForThisSlot = Math.max(0, minSalary - totalSalary);
      }

      const player = this.selectPlayerForPosition(
        pos,
        playersByPosition,
        usedPlayerIds,
        strategy,
        exposureTracker,
        maxExposure,
        eligible,
        quantEnabled,
        quant,
        quantScores,
        kellyLimits,
        maxForThisSlot,
        minForThisSlot
      );

      if (!player) {
        return null;
      }

      lineup.push({ ...player, rosterPosition: pos });
      usedPlayerIds.add(player.id);
      totalSalary += player.salary;
      totalProjection += player.projection || 0;
    }

    // Final salary validation
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

  selectPlayerForPosition(rosterPosition, playersByPosition, usedPlayerIds, strategy, exposureTracker, maxExposure, eligiblePositions = null, quantEnabled = false, quant = null, quantScores = null, kellyLimits = null, maxSalaryForSlot = Infinity, minSalaryForSlot = 0) {
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

    // Convert to array and filter: used players, exposure limits, AND salary budget
    let eligiblePlayers = Array.from(playerMap.values()).filter(p => {
      if (usedPlayerIds.has(p.id)) return false;

      // Check exposure limit (use Kelly limit if available, otherwise default)
      const playerMaxExposure = (kellyLimits && kellyLimits.has(p.id))
        ? kellyLimits.get(p.id)
        : maxExposure;
      if ((exposureTracker.get(p.id) || 0) >= playerMaxExposure) return false;

      // Salary budget constraints — can we afford this player?
      if (p.salary > maxSalaryForSlot) return false;
      if (p.salary < minSalaryForSlot) return false;

      return true;
    });

    if (eligiblePlayers.length === 0) {
      return null;
    }

    // When quant is enabled, use quant-scored selection (with salary-filtered pool)
    if (quantEnabled && quant && quantScores) {
      return quant.selectPlayerQuant(eligiblePlayers, quantScores, strategy);
    }

    // Strategy-based selection with salary-aware sorting
    eligiblePlayers.sort((a, b) => {
      switch (strategy) {
        case 'greedy':
        case 'projection':
          return (b.projection || 0) - (a.projection || 0);
        case 'value': {
          const valueA = a.salary > 0 ? (a.projection || 0) / a.salary * 1000 : 0;
          const valueB = b.salary > 0 ? (b.projection || 0) / b.salary * 1000 : 0;
          return valueB - valueA;
        }
        case 'balanced': {
          const balanceA = ((a.projection || 0) * 0.6) + (a.salary > 0 ? ((a.projection || 0) / a.salary * 1000) * 0.4 : 0);
          const balanceB = ((b.projection || 0) * 0.6) + (b.salary > 0 ? ((b.projection || 0) / b.salary * 1000) * 0.4 : 0);
          return balanceB - balanceA;
        }
        default:
          return (b.projection || 0) - (a.projection || 0);
      }
    });

    // Weighted random from top candidates for diversity
    const poolSize = Math.min(Math.max(3, Math.floor(eligiblePlayers.length * 0.2)), 8);
    const topPool = eligiblePlayers.slice(0, poolSize);

    // Weight by projection — higher projection = higher pick probability
    const totalProj = topPool.reduce((s, p) => s + Math.max(p.projection || 0, 0.1), 0);
    let rand = Math.random() * totalProj;
    for (const player of topPool) {
      rand -= Math.max(player.projection || 0, 0.1);
      if (rand <= 0) return player;
    }
    return topPool[0];
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

