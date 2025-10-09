const { v4: uuidv4 } = require('uuid');

// Enhanced optimization algorithm with multiple strategies
class MLBOptimizer {
  constructor() {
    this.strategies = ['greedy', 'balanced', 'value', 'projection'];
  }

  async optimize(config) {
    const { 
      players, 
      numLineups, 
      minSalary = 45000, 
      maxSalary = 50000, 
      stackSettings = {},
      uniquePlayers = 7,
      maxExposure = 40,
      // Advanced settings
      monteCarloIterations = 100,
      sortingMethod = 'Points',
      minUniquePlayersBetweenLineups = 3,
      enableRiskManagement = true,
      disableKellySizing = false,
      stackTypes = {},
      exposureSettings = {},
      riskTolerance = 'medium',
      bankroll = 1000,
      onProgress 
    } = config;

    const results = [];
    
    // Position requirements for DraftKings MLB
    const positionReqs = {
      'P': 2,
      'C': 1,
      '1B': 1,
      '2B': 1,
      '3B': 1,
      'SS': 1,
      'OF': 3
    };

    // Pre-filter players by position and selection
    const playersByPosition = this.groupPlayersByPosition(players);
    
    // Validate we have enough players for each position
    for (const [pos, count] of Object.entries(positionReqs)) {
      if (!playersByPosition[pos] || playersByPosition[pos].length < count) {
        throw new Error(`Not enough players available for position ${pos}`);
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

      if (lineup) {
        const lineupKey = this.getLineupKey(lineup.players);
        lineupPool.add(lineupKey);
        
        // Update exposure tracking
        lineup.players.forEach(player => {
          const count = exposureTracker.get(player.id) || 0;
          exposureTracker.set(player.id, count + 1);
        });

        results.push({
          id: uuidv4(),
          players: lineup.players,
          totalSalary: lineup.totalSalary,
          totalProjection: lineup.totalProjection,
          value: lineup.totalProjection / lineup.totalSalary * 1000,
          strategy,
          stacks: this.analyzeStacks(lineup.players),
          timestamp: new Date().toISOString()
        });
      }

      // Small delay to simulate processing
      await new Promise(resolve => setTimeout(resolve, 5));
    }

    // Sort results by projection (descending)
    results.sort((a, b) => b.totalProjection - a.totalProjection);

    if (onProgress) onProgress(100);
    return results;
  }

  groupPlayersByPosition(players) {
    const grouped = {};
    
    players.forEach(player => {
      if (!player.selected) return;
      
      const positions = player.position.split('/');
      positions.forEach(pos => {
        if (!grouped[pos]) {
          grouped[pos] = [];
        }
        grouped[pos].push(player);
      });
    });

    // Sort each position group by value (projection/salary)
    Object.keys(grouped).forEach(pos => {
      grouped[pos].sort((a, b) => {
        const aValue = a.projection / a.salary;
        const bValue = b.projection / b.salary;
        return bValue - aValue;
      });
    });

    return grouped;
  }

  generateLineup(playersByPosition, positionReqs, minSalary, maxSalary, strategy, stackSettings, exposureTracker, maxExposure) {
    const lineup = [];
    let totalSalary = 0;
    let totalProjection = 0;
    const usedPlayers = new Set();

    // Apply stacking logic first if specified
    if (stackSettings.enabled && stackSettings.teams && stackSettings.teams.length > 0) {
      const stackResult = this.applyStacking(
        playersByPosition, 
        stackSettings, 
        usedPlayers, 
        exposureTracker, 
        maxExposure
      );
      
      if (stackResult) {
        lineup.push(...stackResult.players);
        totalSalary += stackResult.totalSalary;
        totalProjection += stackResult.totalProjection;
        stackResult.players.forEach(p => usedPlayers.add(p.id));
      }
    }

    // Fill remaining positions
    for (const [position, count] of Object.entries(positionReqs)) {
      const currentPositionCount = lineup.filter(p => 
        p.position.includes(position)
      ).length;
      
      const needed = count - currentPositionCount;
      
      for (let i = 0; i < needed; i++) {
        const player = this.selectPlayerForPosition(
          playersByPosition[position], 
          usedPlayers, 
          totalSalary, 
          maxSalary, 
          strategy,
          exposureTracker,
          maxExposure
        );
        
        if (!player) return null;
        
        lineup.push(player);
        totalSalary += player.salary;
        totalProjection += player.projection;
        usedPlayers.add(player.id);
      }
    }

    // Validate lineup constraints
    if (totalSalary < minSalary || totalSalary > maxSalary) {
      return null;
    }

    if (lineup.length !== Object.values(positionReqs).reduce((a, b) => a + b, 0)) {
      return null;
    }

    return {
      players: lineup,
      totalSalary,
      totalProjection
    };
  }

  selectPlayerForPosition(positionPlayers, usedPlayers, currentSalary, maxSalary, strategy, exposureTracker, maxExposure) {
    if (!positionPlayers || positionPlayers.length === 0) return null;
    
    const availablePlayers = positionPlayers.filter(p => {
      if (usedPlayers.has(p.id)) return false;
      if (currentSalary + p.salary > maxSalary) return false;
      
      // Check exposure limits
      const currentExposure = exposureTracker.get(p.id) || 0;
      const maxAllowed = Math.ceil(maxExposure / 100 * 10); // Rough estimate
      if (currentExposure >= maxAllowed) return false;
      
      return true;
    });

    if (availablePlayers.length === 0) return null;

    // Different selection strategies
    switch (strategy) {
      case 'greedy':
        return availablePlayers[0]; // Already sorted by value
      
      case 'balanced':
        const midIndex = Math.floor(availablePlayers.length / 3);
        return availablePlayers[Math.floor(Math.random() * midIndex)];
      
      case 'value':
        return availablePlayers
          .sort((a, b) => (b.projection / b.salary) - (a.projection / a.salary))[0];
      
      case 'projection':
        return availablePlayers
          .sort((a, b) => b.projection - a.projection)[0];
      
      default:
        return availablePlayers[Math.floor(Math.random() * Math.min(5, availablePlayers.length))];
    }
  }

  applyStacking(playersByPosition, stackSettings, usedPlayers, exposureTracker, maxExposure) {
    const { teams, minPlayersPerTeam = 2 } = stackSettings;
    
    if (!teams || teams.length === 0) return null;
    
    const selectedTeam = teams[Math.floor(Math.random() * teams.length)];
    const teamPlayers = [];
    
    // Collect all players from the selected team
    Object.values(playersByPosition).flat().forEach(player => {
      if (player.team === selectedTeam && !usedPlayers.has(player.id)) {
        const currentExposure = exposureTracker.get(player.id) || 0;
        const maxAllowed = Math.ceil(maxExposure / 100 * 10);
        if (currentExposure < maxAllowed) {
          teamPlayers.push(player);
        }
      }
    });
    
    if (teamPlayers.length < minPlayersPerTeam) return null;
    
    // Sort by value and take the best players
    teamPlayers.sort((a, b) => (b.projection / b.salary) - (a.projection / a.salary));
    
    const stackPlayers = teamPlayers.slice(0, Math.min(minPlayersPerTeam, teamPlayers.length));
    
    return {
      players: stackPlayers,
      totalSalary: stackPlayers.reduce((sum, p) => sum + p.salary, 0),
      totalProjection: stackPlayers.reduce((sum, p) => sum + p.projection, 0)
    };
  }

  isDuplicateLineup(lineup, lineupPool, uniquePlayers) {
    const lineupKey = this.getLineupKey(lineup.players);
    return lineupPool.has(lineupKey);
  }

  getLineupKey(players) {
    return players
      .map(p => p.id)
      .sort()
      .join('-');
  }

  analyzeStacks(players) {
    const teams = {};
    
    players.forEach(player => {
      if (!teams[player.team]) {
        teams[player.team] = [];
      }
      teams[player.team].push(player);
    });

    const stacks = [];
    Object.entries(teams).forEach(([team, teamPlayers]) => {
      if (teamPlayers.length >= 2) {
        stacks.push({
          team,
          players: teamPlayers.length,
          positions: teamPlayers.map(p => p.position).join(', ')
        });
      }
    });

    return stacks;
  }

  // Advanced methods for enhanced optimization
  getRiskMultiplier(riskTolerance) {
    switch (riskTolerance) {
      case 'conservative': return 0.8;
      case 'medium': return 1.0;
      case 'aggressive': return 1.3;
      default: return 1.0;
    }
  }

  calculateKellyBet(bankroll, riskTolerance) {
    const basePercentage = riskTolerance === 'conservative' ? 0.02 : 
                          riskTolerance === 'aggressive' ? 0.10 : 0.05;
    return bankroll * basePercentage;
  }

  calculateLineupScore(lineup, riskMultiplier, maxBankrollPerLineup) {
    const baseScore = lineup.totalProjection;
    const salaryEfficiency = (lineup.totalSalary / 50000); // Normalize salary usage
    const riskAdjustedScore = baseScore * riskMultiplier;
    const valueScore = (lineup.totalProjection / lineup.totalSalary) * 1000000; // Value component
    
    return riskAdjustedScore + (valueScore * 0.1) - (salaryEfficiency * 2);
  }

  sortPlayersByMethod(playersByPosition, sortingMethod) {
    Object.keys(playersByPosition).forEach(position => {
      switch (sortingMethod) {
        case 'Points':
          playersByPosition[position].sort((a, b) => b.projection - a.projection);
          break;
        case 'Value':
          playersByPosition[position].sort((a, b) => 
            (b.projection / b.salary) - (a.projection / a.salary)
          );
          break;
        case 'Salary':
          playersByPosition[position].sort((a, b) => b.salary - a.salary);
          break;
        default:
          playersByPosition[position].sort((a, b) => b.projection - a.projection);
      }
    });
  }

  checkStackExposure(stackType, stackCount, totalLineups, exposureSettings) {
    if (!exposureSettings[stackType]) return true;
    
    const { min, max } = exposureSettings[stackType];
    const currentPercentage = (stackCount / totalLineups) * 100;
    
    return currentPercentage >= min && currentPercentage <= max;
  }

  validateStackTypes(lineup, stackTypes) {
    const lineupStacks = this.analyzeStacks(lineup.players);
    const stackSignature = this.getStackSignature(lineupStacks);
    
    // Check if this stack type is enabled
    return stackTypes[stackSignature] !== false;
  }

  getStackSignature(stacks) {
    if (stacks.length === 0) return 'No Stacks';
    
    const stackSizes = stacks.map(s => s.players).sort((a, b) => b - a);
    return stackSizes.join('|');
  }

  // Enhanced lineup generation with advanced features
  async generateAdvancedLineup(
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
  ) {
    const lineup = [];
    let totalSalary = 0;
    let totalProjection = 0;
    const usedPlayers = new Set();

    // Apply stacking if configured
    if (stackSettings.enabled && stackSettings.teams && stackSettings.teams.length > 0) {
      const stackResult = this.applyAdvancedStacking(
        playersByPosition,
        stackSettings,
        usedPlayers,
        exposureTracker,
        maxExposure,
        stackTypes
      );
      
      if (stackResult) {
        lineup.push(...stackResult.players);
        totalSalary += stackResult.totalSalary;
        totalProjection += stackResult.totalProjection;
        stackResult.players.forEach(p => usedPlayers.add(p.id));
      }
    }

    // Fill remaining positions with exposure and uniqueness constraints
    for (const [position, count] of Object.entries(positionReqs)) {
      const currentPositionCount = lineup.filter(p => 
        p.position.includes(position)
      ).length;
      
      const needed = count - currentPositionCount;
      
      for (let i = 0; i < needed; i++) {
        const player = this.selectAdvancedPlayer(
          playersByPosition[position], 
          usedPlayers, 
          totalSalary, 
          maxSalary, 
          strategy,
          exposureTracker,
          maxExposure,
          lineupPool,
          minUniquePlayersBetweenLineups
        );
        
        if (!player) return null;
        
        lineup.push(player);
        totalSalary += player.salary;
        totalProjection += player.projection;
        usedPlayers.add(player.id);
      }
    }

    // Validate all constraints
    if (totalSalary < minSalary || totalSalary > maxSalary) {
      return null;
    }

    if (lineup.length !== Object.values(positionReqs).reduce((a, b) => a + b, 0)) {
      return null;
    }

    const lineupObj = {
      players: lineup,
      totalSalary,
      totalProjection,
      strategy
    };

    // Validate stack types if configured
    if (stackTypes && Object.keys(stackTypes).length > 0) {
      if (!this.validateStackTypes(lineupObj, stackTypes)) {
        return null;
      }
    }

    return lineupObj;
  }

  selectAdvancedPlayer(positionPlayers, usedPlayers, currentSalary, maxSalary, strategy, exposureTracker, maxExposure, lineupPool, minUniquePlayersBetweenLineups) {
    if (!positionPlayers || positionPlayers.length === 0) return null;
    
    const availablePlayers = positionPlayers.filter(p => {
      if (usedPlayers.has(p.id)) return false;
      if (currentSalary + p.salary > maxSalary) return false;
      
      // Check exposure limits with better tracking
      const currentExposure = exposureTracker.get(p.id) || 0;
      const totalLineups = lineupPool.size + 1; // Include current lineup
      const exposurePercentage = (currentExposure / totalLineups) * 100;
      
      if (exposurePercentage >= maxExposure) return false;
      
      // Check min/max exposure for specific players if set
      if (p.minExposure > 0 && exposurePercentage < p.minExposure) return false;
      if (p.maxExposure < 100 && exposurePercentage >= p.maxExposure) return false;
      
      return true;
    });

    if (availablePlayers.length === 0) return null;

    // Enhanced selection strategies
    switch (strategy) {
      case 'greedy':
        return availablePlayers[0]; // Already sorted by primary metric
      
      case 'balanced':
        const midIndex = Math.floor(availablePlayers.length / 3);
        const balancedPool = availablePlayers.slice(0, Math.max(1, midIndex));
        return balancedPool[Math.floor(Math.random() * balancedPool.length)];
      
      case 'value':
        return availablePlayers
          .sort((a, b) => (b.projection / b.salary) - (a.projection / a.salary))[0];
      
      case 'projection':
        return availablePlayers
          .sort((a, b) => b.projection - a.projection)[0];
      
      default:
        const randomPool = availablePlayers.slice(0, Math.min(5, availablePlayers.length));
        return randomPool[Math.floor(Math.random() * randomPool.length)];
    }
  }

  applyAdvancedStacking(playersByPosition, stackSettings, usedPlayers, exposureTracker, maxExposure, stackTypes) {
    const { teams, minPlayersPerTeam = 2, maxPlayersPerTeam = 4 } = stackSettings;
    
    if (!teams || teams.length === 0) return null;
    
    const selectedTeam = teams[Math.floor(Math.random() * teams.length)];
    const teamPlayers = [];
    
    // Collect all available players from the selected team
    Object.values(playersByPosition).flat().forEach(player => {
      if (player.team === selectedTeam && !usedPlayers.has(player.id)) {
        const currentExposure = exposureTracker.get(player.id) || 0;
        const exposurePercentage = (currentExposure / Math.max(1, exposureTracker.size)) * 100;
        
        if (exposurePercentage < maxExposure) {
          teamPlayers.push(player);
        }
      }
    });
    
    if (teamPlayers.length < minPlayersPerTeam) return null;
    
    // Sort by value and projection
    teamPlayers.sort((a, b) => {
      const valueA = a.projection / a.salary;
      const valueB = b.projection / b.salary;
      return valueB - valueA;
    });
    
    // Select optimal number of players for stack
    const stackSize = Math.min(
      maxPlayersPerTeam,
      Math.max(minPlayersPerTeam, Math.floor(Math.random() * 2) + minPlayersPerTeam)
    );
    
    const stackPlayers = teamPlayers.slice(0, Math.min(stackSize, teamPlayers.length));
    
    return {
      players: stackPlayers,
      totalSalary: stackPlayers.reduce((sum, p) => sum + p.salary, 0),
      totalProjection: stackPlayers.reduce((sum, p) => sum + p.projection, 0),
      team: selectedTeam,
      size: stackPlayers.length
    };
  }
}

module.exports = MLBOptimizer;
