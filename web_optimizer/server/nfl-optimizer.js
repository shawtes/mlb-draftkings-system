const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

/**
 * NFL DraftKings Optimizer - Backend Integration
 * Bridges Node.js server with Python NFL genetic algorithm optimizer
 */
class NFLOptimizer {
  constructor() {
    this.strategies = ['greedy', 'balanced', 'value', 'projection'];
    this.pythonOptimizerPath = this.findPythonOptimizer();
  }

  /**
   * Find the Python NFL optimizer script
   */
  findPythonOptimizer() {
    const possiblePaths = [
      path.join(__dirname, '../../6_OPTIMIZATION/genetic_algo_nfl_optimizer.py'),
      path.join(__dirname, '../../../6_OPTIMIZATION/genetic_algo_nfl_optimizer.py'),
      path.join(process.cwd(), '6_OPTIMIZATION/genetic_algo_nfl_optimizer.py'),
    ];

    for (const testPath of possiblePaths) {
      if (fs.existsSync(testPath)) {
        console.log(`✅ Found NFL optimizer at: ${testPath}`);
        return testPath;
      }
    }

    console.warn('⚠️  Python NFL optimizer not found, using fallback JavaScript optimizer');
    return null;
  }

  /**
   * Main optimization entry point
   */
  async optimize(config) {
    const { 
      players, 
      numLineups, 
      minSalary = 48000, 
      maxSalary = 50000, 
      stackSettings = {},
      uniquePlayers = 7,
      maxExposure = 40,
      stackTypes = {},
      exposureSettings = {},
      riskTolerance = 'medium',
      contestMode = 'gpp',
      onProgress 
    } = config;

    // Try Python optimizer first (preferred for production)
    if (this.pythonOptimizerPath) {
      try {
        return await this.optimizeWithPython(config);
      } catch (error) {
        console.error('Python optimizer failed, falling back to JavaScript:', error.message);
      }
    }

    // Fallback to JavaScript optimizer
    return await this.optimizeWithJavaScript(config);
  }

  /**
   * Optimize using Python genetic algorithm (production method)
   */
  async optimizeWithPython(config) {
    const {
      players,
      numLineups,
      minSalary,
      maxSalary,
      stackSettings,
      stackTypes,
      onProgress
    } = config;

    return new Promise(async (resolve, reject) => {
      try {
        // Create temporary CSV file with player data
        const tempDir = os.tmpdir();
        const inputFile = path.join(tempDir, `nfl_players_${Date.now()}.csv`);
        const outputFile = path.join(tempDir, `nfl_lineups_${Date.now()}.csv`);

        // Write player data to CSV
        await this.writePlayerCSV(players, inputFile);

        // Build Python command arguments
        const pythonArgs = [
          this.pythonOptimizerPath,
          '--input', inputFile,
          '--output', outputFile,
          '--lineups', numLineups.toString(),
          '--min-salary', minSalary.toString(),
          '--max-salary', maxSalary.toString(),
        ];

        // Add stack settings if provided
        if (stackSettings.enabled && stackSettings.types && stackSettings.types.length > 0) {
          pythonArgs.push('--stacks', stackSettings.types.join(','));
        }

        // Execute Python optimizer
        const python = spawn('python', pythonArgs);
        
        let stdout = '';
        let stderr = '';
        let lastProgress = 0;

        python.stdout.on('data', (data) => {
          stdout += data.toString();
          
          // Parse progress updates
          const progressMatch = data.toString().match(/Progress: (\d+)%/);
          if (progressMatch && onProgress) {
            const progress = parseInt(progressMatch[1]);
            if (progress > lastProgress) {
              lastProgress = progress;
              onProgress(progress);
            }
          }
        });

        python.stderr.on('data', (data) => {
          stderr += data.toString();
          console.log('Python optimizer log:', data.toString());
        });

        python.on('close', async (code) => {
          try {
            // Clean up input file
            if (fs.existsSync(inputFile)) {
              fs.unlinkSync(inputFile);
            }

            if (code !== 0) {
              throw new Error(`Python optimizer exited with code ${code}: ${stderr}`);
            }

            // Read and parse output file
            if (!fs.existsSync(outputFile)) {
              throw new Error('Python optimizer did not generate output file');
            }

            const lineups = await this.parseLineupsCSV(outputFile);
            
            // Clean up output file
            fs.unlinkSync(outputFile);

            if (onProgress) onProgress(100);
            resolve(lineups);

          } catch (error) {
            reject(error);
          }
        });

        python.on('error', (error) => {
          reject(new Error(`Failed to start Python optimizer: ${error.message}`));
        });

      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Write player data to CSV format for Python optimizer
   */
  async writePlayerCSV(players, filepath) {
    const headers = ['Name', 'Position', 'Team', 'Salary', 'Predicted_DK_Points', 'Ownership'];
    const rows = players.map(p => [
      p.name,
      p.position,
      p.team,
      p.salary,
      p.projection || p.projectedPoints || 0,
      p.ownership || 0
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    return fs.promises.writeFile(filepath, csvContent, 'utf8');
  }

  /**
   * Parse lineups from Python optimizer output CSV
   */
  async parseLineupsCSV(filepath) {
    const content = await fs.promises.readFile(filepath, 'utf8');
    const lines = content.trim().split('\n');
    
    if (lines.length < 2) {
      throw new Error('Empty lineup output from Python optimizer');
    }

    const headers = lines[0].split(',');
    const lineups = [];
    let currentLineup = null;

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const row = {};
      
      headers.forEach((header, idx) => {
        row[header.trim()] = values[idx]?.trim() || '';
      });

      // New lineup detected
      const lineupNum = parseInt(row['Lineup'] || row['lineup_num'] || '1');
      
      if (!currentLineup || currentLineup.number !== lineupNum) {
        if (currentLineup) {
          lineups.push(this.formatLineup(currentLineup));
        }
        currentLineup = {
          number: lineupNum,
          players: [],
          totalSalary: 0,
          totalProjection: 0
        };
      }

      // Add player to current lineup
      const player = {
        id: row['Name'] + '_' + row['Position'],
        name: row['Name'],
        position: row['Position'],
        team: row['Team'],
        salary: parseInt(row['Salary'] || 0),
        projection: parseFloat(row['Predicted_DK_Points'] || row['Fantasy_Points'] || 0)
      };

      currentLineup.players.push(player);
      currentLineup.totalSalary += player.salary;
      currentLineup.totalProjection += player.projection;
    }

    // Add last lineup
    if (currentLineup && currentLineup.players.length > 0) {
      lineups.push(this.formatLineup(currentLineup));
    }

    return lineups;
  }

  /**
   * Format lineup for frontend consumption
   */
  formatLineup(rawLineup) {
    return {
      id: uuidv4(),
      players: rawLineup.players,
      totalSalary: rawLineup.totalSalary,
      totalProjection: rawLineup.totalProjection,
      value: rawLineup.totalProjection / rawLineup.totalSalary * 1000,
      stacks: this.analyzeStacks(rawLineup.players),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * JavaScript fallback optimizer (when Python unavailable)
   */
  async optimizeWithJavaScript(config) {
    const { 
      players, 
      numLineups, 
      minSalary, 
      maxSalary, 
      stackSettings,
      uniquePlayers,
      maxExposure,
      onProgress 
    } = config;

    const results = [];
    
    // NFL DraftKings position requirements
    const positionReqs = {
      'QB': 1,
      'RB': 2,
      'WR': 3,
      'TE': 1,
      'FLEX': 1, // RB/WR/TE
      'DST': 1
    };

    // Pre-filter and group players by position
    const playersByPosition = this.groupPlayersByPosition(players);
    
    // Validate we have enough players
    for (const [pos, count] of Object.entries(positionReqs)) {
      if (pos === 'FLEX') continue; // FLEX filled from RB/WR/TE pool
      
      // Special handling for DST - if no DST players, create fallback
      if (pos === 'DST' && (!playersByPosition[pos] || playersByPosition[pos].length < count)) {
        console.warn(`⚠️ No DST players found, creating fallback DST players`);
        
        // Create fallback DST players from available teams
        const availableTeams = [...new Set(players.map(p => p.team).filter(Boolean))];
        const fallbackDSTs = availableTeams.slice(0, Math.max(4, count)).map((team, index) => ({
          id: `dst_fallback_${index}`,
          name: `${team} DST`,
          position: 'DST',
          team: team,
          salary: 3000 + (index * 200), // Vary salary slightly
          projection: 6.0 + (index * 0.5), // Vary projection slightly
          selected: true
        }));
        
        playersByPosition[pos] = fallbackDSTs;
        console.log(`✅ Created ${fallbackDSTs.length} fallback DST players`);
      }
      
      if (!playersByPosition[pos] || playersByPosition[pos].length < count) {
        throw new Error(`Not enough players available for position ${pos}. Need ${count}, have ${playersByPosition[pos]?.length || 0}`);
      }
    }

    // Generate lineups with diversity
    const lineupPool = new Set();
    const exposureTracker = new Map();
    
    for (let i = 0; i < numLineups; i++) {
      if (onProgress && i % Math.max(1, Math.floor(numLineups / 10)) === 0) {
        onProgress(Math.round((i / numLineups) * 100));
      }

      // Rotate strategies for diversity
      const strategy = this.strategies[i % this.strategies.length];
      
      let lineup;
      let attempts = 0;
      const maxAttempts = 50;

      do {
        lineup = this.generateNFLLineup(
          playersByPosition, 
          positionReqs, 
          minSalary, 
          maxSalary, 
          strategy,
          stackSettings,
          exposureTracker,
          maxExposure
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

      // Small delay for real-time feel
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    // Sort by projection (descending)
    results.sort((a, b) => b.totalProjection - a.totalProjection);

    if (onProgress) onProgress(100);
    return results;
  }

  /**
   * Group players by position for NFL
   */
  groupPlayersByPosition(players) {
    const grouped = {};
    
    players.forEach(player => {
      // Always include DST players regardless of selection status
      // Other positions require selection
      const isDST = player.position === 'DST';
      if (!player.selected && player.selected !== undefined && !isDST) return;
      
      // NFL positions don't have multi-position eligibility like MLB
      const pos = player.position;
      
      if (!grouped[pos]) {
        grouped[pos] = [];
      }
      grouped[pos].push(player);
    });

    // Sort each position group by value
    Object.keys(grouped).forEach(pos => {
      grouped[pos].sort((a, b) => {
        const aValue = (a.projection || a.projectedPoints || 0) / a.salary;
        const bValue = (b.projection || b.projectedPoints || 0) / b.salary;
        return bValue - aValue;
      });
    });

    return grouped;
  }

  /**
   * Generate a single NFL lineup
   */
  generateNFLLineup(playersByPosition, positionReqs, minSalary, maxSalary, strategy, stackSettings, exposureTracker, maxExposure) {
    const lineup = [];
    let totalSalary = 0;
    let totalProjection = 0;
    const usedPlayers = new Set();

    // Apply NFL stacking logic first (QB-focused stacks)
    if (stackSettings.enabled && stackSettings.teams && stackSettings.teams.length > 0) {
      const stackResult = this.applyNFLStacking(
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

    // Fill core positions (QB, RB, WR, TE, DST)
    for (const [position, count] of Object.entries(positionReqs)) {
      if (position === 'FLEX') continue; // Handle FLEX last
      
      const currentPositionCount = lineup.filter(p => 
        p.position === position
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
        totalProjection += (player.projection || player.projectedPoints || 0);
        usedPlayers.add(player.id);
      }
    }

    // Fill FLEX position (best remaining RB/WR/TE)
    const flexEligible = ['RB', 'WR', 'TE'];
    let flexPlayer = null;
    let bestFlexValue = -1;

    for (const pos of flexEligible) {
      const player = this.selectPlayerForPosition(
        playersByPosition[pos],
        usedPlayers,
        totalSalary,
        maxSalary,
        strategy,
        exposureTracker,
        maxExposure
      );

      if (player) {
        const value = (player.projection || player.projectedPoints || 0) / player.salary;
        if (value > bestFlexValue) {
          bestFlexValue = value;
          flexPlayer = player;
        }
      }
    }

    if (!flexPlayer) return null;

    lineup.push(flexPlayer);
    totalSalary += flexPlayer.salary;
    totalProjection += (flexPlayer.projection || flexPlayer.projectedPoints || 0);
    usedPlayers.add(flexPlayer.id);

    // Validate lineup constraints
    if (totalSalary < minSalary || totalSalary > maxSalary) {
      return null;
    }

    if (lineup.length !== 9) { // NFL lineups have 9 players
      return null;
    }

    return {
      players: lineup,
      totalSalary,
      totalProjection
    };
  }

  /**
   * Apply NFL-specific stacking (QB-centric)
   */
  applyNFLStacking(playersByPosition, stackSettings, usedPlayers, exposureTracker, maxExposure) {
    const { teams, types, minPlayersPerTeam = 2 } = stackSettings;
    
    if (!teams || teams.length === 0) return null;

    const selectedTeam = teams[Math.floor(Math.random() * teams.length)];
    const stackType = types && types.length > 0 
      ? types[Math.floor(Math.random() * types.length)]
      : 'QB + WR';

    // Get QB from selected team
    const teamQBs = (playersByPosition['QB'] || []).filter(p => 
      p.team === selectedTeam && !usedPlayers.has(p.id)
    );

    if (teamQBs.length === 0) return null;
    const qb = teamQBs[0];

    const stackPlayers = [qb];
    let totalSalary = qb.salary;
    let totalProjection = qb.projection || qb.projectedPoints || 0;

    // Add receivers based on stack type
    if (stackType.includes('QB + WR') || stackType.includes('QB + 2 WR')) {
      const wrCount = stackType.includes('2 WR') ? 2 : 1;
      const teamWRs = (playersByPosition['WR'] || [])
        .filter(p => p.team === selectedTeam && !usedPlayers.has(p.id))
        .slice(0, wrCount);

      teamWRs.forEach(wr => {
        stackPlayers.push(wr);
        totalSalary += wr.salary;
        totalProjection += (wr.projection || wr.projectedPoints || 0);
      });
    }

    // Add TE if specified
    if (stackType.includes('TE')) {
      const teamTEs = (playersByPosition['TE'] || [])
        .filter(p => p.team === selectedTeam && !usedPlayers.has(p.id))
        .slice(0, 1);

      if (teamTEs.length > 0) {
        stackPlayers.push(teamTEs[0]);
        totalSalary += teamTEs[0].salary;
        totalProjection += (teamTEs[0].projection || teamTEs[0].projectedPoints || 0);
      }
    }

    // Add RB if specified
    if (stackType.includes('RB')) {
      const teamRBs = (playersByPosition['RB'] || [])
        .filter(p => p.team === selectedTeam && !usedPlayers.has(p.id))
        .slice(0, 1);

      if (teamRBs.length > 0) {
        stackPlayers.push(teamRBs[0]);
        totalSalary += teamRBs[0].salary;
        totalProjection += (teamRBs[0].projection || teamRBs[0].projectedPoints || 0);
      }
    }

    if (stackPlayers.length < minPlayersPerTeam) return null;

    return {
      players: stackPlayers,
      totalSalary,
      totalProjection,
      team: selectedTeam,
      type: stackType
    };
  }

  /**
   * Select player for position using strategy
   */
  selectPlayerForPosition(positionPlayers, usedPlayers, currentSalary, maxSalary, strategy, exposureTracker, maxExposure) {
    if (!positionPlayers || positionPlayers.length === 0) return null;
    
    const availablePlayers = positionPlayers.filter(p => {
      if (usedPlayers.has(p.id)) return false;
      if (currentSalary + p.salary > maxSalary) return false;
      
      // Check exposure limits
      const currentExposure = exposureTracker.get(p.id) || 0;
      const maxAllowed = Math.ceil(maxExposure / 100 * 10);
      if (currentExposure >= maxAllowed) return false;
      
      return true;
    });

    if (availablePlayers.length === 0) return null;

    // Selection strategies
    switch (strategy) {
      case 'greedy':
        return availablePlayers[0];
      
      case 'balanced':
        const midIndex = Math.floor(availablePlayers.length / 3);
        return availablePlayers[Math.floor(Math.random() * Math.max(1, midIndex))];
      
      case 'value':
        return availablePlayers
          .sort((a, b) => {
            const aVal = (a.projection || a.projectedPoints || 0) / a.salary;
            const bVal = (b.projection || b.projectedPoints || 0) / b.salary;
            return bVal - aVal;
          })[0];
      
      case 'projection':
        return availablePlayers
          .sort((a, b) => (b.projection || b.projectedPoints || 0) - (a.projection || a.projectedPoints || 0))[0];
      
      default:
        return availablePlayers[Math.floor(Math.random() * Math.min(5, availablePlayers.length))];
    }
  }

  /**
   * Check if lineup is duplicate
   */
  isDuplicateLineup(lineup, lineupPool, uniquePlayers) {
    const lineupKey = this.getLineupKey(lineup.players);
    return lineupPool.has(lineupKey);
  }

  /**
   * Get unique key for lineup
   */
  getLineupKey(players) {
    return players
      .map(p => p.id)
      .sort()
      .join('-');
  }

  /**
   * Analyze team stacks in lineup
   */
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
        // Identify stack type
        const hasQB = teamPlayers.some(p => p.position === 'QB');
        const wrCount = teamPlayers.filter(p => p.position === 'WR').length;
        const hasTE = teamPlayers.some(p => p.position === 'TE');
        const hasRB = teamPlayers.some(p => p.position === 'RB');

        let stackType = 'Team Stack';
        if (hasQB) {
          if (wrCount >= 2 && hasTE) stackType = 'QB + 2 WR + TE';
          else if (wrCount >= 2) stackType = 'QB + 2 WR';
          else if (wrCount >= 1 && hasTE) stackType = 'QB + WR + TE';
          else if (wrCount >= 1 && hasRB) stackType = 'QB + WR + RB';
          else if (wrCount >= 1) stackType = 'QB + WR';
        }

        stacks.push({
          team,
          players: teamPlayers.length,
          positions: teamPlayers.map(p => p.position).join(', '),
          type: stackType
        });
      }
    });

    return stacks;
  }
}

module.exports = NFLOptimizer;


