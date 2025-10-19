const express = require('express');
const cors = require('cors');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');
const { Parser } = require('json2csv');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const MLBOptimizer = require('./optimizer');

const app = express();
const PORT = process.env.PORT || 8080;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static files
app.use(express.static(path.join(__dirname, '../client/build')));

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ 
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'), false);
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: 8080 });

// Global state
let playersData = [];
let optimizationResults = [];
let activeConnections = new Set();

// WebSocket connection handler
wss.on('connection', (ws) => {
  activeConnections.add(ws);
  console.log('Client connected. Total connections:', activeConnections.size);
  
  ws.on('close', () => {
    activeConnections.delete(ws);
    console.log('Client disconnected. Total connections:', activeConnections.size);
  });
  
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
    activeConnections.delete(ws);
  });
});

// Broadcast to all connected clients
function broadcast(message) {
  const messageStr = JSON.stringify(message);
  activeConnections.forEach((ws) => {
    if (ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(messageStr);
      } catch (error) {
        console.error('Error sending message:', error);
        activeConnections.delete(ws);
      }
    }
  });
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    connections: activeConnections.size
  });
});

// Upload and parse CSV data
app.post('/api/upload-players', upload.single('playersFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const results = [];
  const filePath = req.file.path;

  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => {
      try {
        // Log available columns from first row
        if (results.length === 0) {
          console.log('Available CSV columns:', Object.keys(data));
        }
        
        // More comprehensive field mapping for projections with robust parsing
        let projectionValue = 0;
        let projectionSource = 'none';
        
        // Simple projection field checking
        if (data.My_Proj && !isNaN(parseFloat(data.My_Proj))) {
          projectionValue = parseFloat(data.My_Proj);
          projectionSource = 'My_Proj';
        } else if (data.PPG_Projection && !isNaN(parseFloat(data.PPG_Projection))) {
          projectionValue = parseFloat(data.PPG_Projection);
          projectionSource = 'PPG_Projection';
        } else if (data.projection && !isNaN(parseFloat(data.projection))) {
          projectionValue = parseFloat(data.projection);
          projectionSource = 'projection';
        } else if (data.Projection && !isNaN(parseFloat(data.Projection))) {
          projectionValue = parseFloat(data.Projection);
          projectionSource = 'Projection';
        }
        
        // More comprehensive field mapping for position
        const positionValue = (
          data.Pos || 
          data.Position || 
          data.position || 
          data.POSITION ||
          data.Roster_Position ||
          data.roster_position ||
          data.DK_Position ||
          data.dk_position ||
          data.FD_Position ||
          data.fd_position
        ) || '';
        
        // More comprehensive field mapping for salary
        const salaryValue = parseInt(
          data.Salary || 
          data.salary || 
          data.SALARY ||
          data.DK_Salary ||
          data.dk_salary ||
          data.FD_Salary ||
          data.fd_salary ||
          data.Cost ||
          data.cost
        ) || 0;
        
        // Process and validate player data
        const player = {
          id: uuidv4(),
          name: data.Name || data.name || data.NAME || data.Player || data.player || '',
          team: data.Team || data.TeamAbbrev || data.team || data.TEAM || data.Tm || data.tm || '',
          position: positionValue,
          salary: salaryValue,
          projection: projectionValue,
          source: data.Source || data.source || '',
          value: 0,
          ownership: parseFloat(data.Ownership || data.ownership || data.Own || data.own) || 0,
          selected: false,
          locked: false,
          excluded: false,
          favorite: false,
          minExposure: 0,
          maxExposure: 100
        };
        
        // Calculate value (points per $1000) as a number
        if (player.salary > 0 && player.projection > 0) {
          player.value = parseFloat((player.projection / player.salary * 1000).toFixed(2));
        }
        
        results.push(player);
        
        // Log first few players for debugging
        if (results.length <= 3) {
          console.log(`Player ${results.length} processed:`, {
            name: player.name,
            position: player.position,
            salary: player.salary,
            projection: player.projection,
            projectionSource: projectionSource,
            value: player.value,
            rawMyProj: data.My_Proj,
            rawPPGProjection: data.PPG_Projection,
            projectionType: typeof player.projection
          });
          if (results.length === 1) {
            console.log('Raw CSV data sample (first 5 fields):', {
              Name: data.Name,
              Pos: data.Pos,
              Team: data.Team,
              Salary: data.Salary,
              My_Proj: data.My_Proj
            });
          }
        }
      } catch (error) {
        console.error('Error processing player data:', error, data);
      }
    })
    .on('end', () => {
      playersData = results;
      
      // Debug: Show projection statistics
      const playersWithProjections = results.filter(p => p.projection > 0);
      const zeroProjections = results.filter(p => p.projection === 0);
      console.log('\n=== PROJECTION PARSING SUMMARY ===');
      console.log(`Total players processed: ${results.length}`);
      console.log(`Players with projections > 0: ${playersWithProjections.length}`);
      console.log(`Players with zero projections: ${zeroProjections.length}`);
      if (playersWithProjections.length > 0) {
        const projections = playersWithProjections.map(p => p.projection);
        console.log(`Projection range: ${Math.min(...projections).toFixed(1)} to ${Math.max(...projections).toFixed(1)}`);
        console.log(`Average projection: ${(projections.reduce((a, b) => a + b, 0) / projections.length).toFixed(1)}`);
        console.log('Sample players with projections:');
        playersWithProjections.slice(0, 3).forEach(p => {
          console.log(`  - ${p.name}: ${p.projection} pts, $${p.salary}, ${p.value.toFixed(2)} value`);
        });
      }
      if (zeroProjections.length > 0 && zeroProjections.length < 10) {
        console.log('Players with zero projections:', zeroProjections.map(p => p.name).join(', '));
      }
      console.log('=====================================\n');
      
      // Clean up uploaded file
      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
      
      // Broadcast update to connected clients
      broadcast({
        type: 'PLAYERS_LOADED',
        data: { count: results.length, timestamp: new Date().toISOString() }
      });
      
      res.json({
        success: true,
        message: `Loaded ${results.length} players`,
        playersCount: results.length,
        playersWithProjections: playersWithProjections.length,
        teams: [...new Set(results.map(p => p.team))].sort(),
        positions: [...new Set(results.map(p => p.position))].sort()
      });
    })
    .on('error', (error) => {
      console.error('CSV parsing error:', error);
      res.status(500).json({ error: 'Error parsing CSV file' });
    });
});

// Get players data
app.get('/api/players', (req, res) => {
  const { position, team, search } = req.query;
  
  let filteredPlayers = [...playersData];
  
  // Filter by position
  if (position && position !== 'ALL') {
    filteredPlayers = filteredPlayers.filter(p => 
      p.position.includes(position)
    );
  }
  
  // Filter by team
  if (team && team !== 'ALL') {
    filteredPlayers = filteredPlayers.filter(p => p.team === team);
  }
  
  // Search filter
  if (search) {
    const searchLower = search.toLowerCase();
    filteredPlayers = filteredPlayers.filter(p =>
      p.name.toLowerCase().includes(searchLower) ||
      p.team.toLowerCase().includes(searchLower)
    );
  }
  
  res.json({
    players: filteredPlayers,
    total: filteredPlayers.length,
    allTeams: [...new Set(playersData.map(p => p.team))].sort(),
    allPositions: [...new Set(playersData.map(p => p.position))].sort()
  });
});

// Update player settings
app.put('/api/players/:id', (req, res) => {
  const { id } = req.params;
  const updates = req.body;
  
  const playerIndex = playersData.findIndex(p => p.id === id);
  if (playerIndex === -1) {
    return res.status(404).json({ error: 'Player not found' });
  }
  
  // Update player with provided fields
  playersData[playerIndex] = { ...playersData[playerIndex], ...updates };
  
  res.json({ success: true, player: playersData[playerIndex] });
});

// Bulk update players
app.put('/api/players/bulk', (req, res) => {
  const { action, filters, settings } = req.body;
  
  let updatedCount = 0;
  
  playersData.forEach((player) => {
    let shouldUpdate = true;
    
    // Apply filters
    if (filters.position && filters.position !== 'ALL') {
      shouldUpdate = shouldUpdate && player.position.includes(filters.position);
    }
    if (filters.team && filters.team !== 'ALL') {
      shouldUpdate = shouldUpdate && player.team === filters.team;
    }
    if (filters.salaryMin) {
      shouldUpdate = shouldUpdate && player.salary >= filters.salaryMin;
    }
    if (filters.salaryMax) {
      shouldUpdate = shouldUpdate && player.salary <= filters.salaryMax;
    }
    
    if (shouldUpdate) {
      if (action === 'select') {
        player.selected = true;
      } else if (action === 'deselect') {
        player.selected = false;
      } else if (action === 'update_exposure' && settings) {
        if (settings.minExposure !== undefined) {
          player.minExposure = Math.max(0, Math.min(100, settings.minExposure));
        }
        if (settings.maxExposure !== undefined) {
          player.maxExposure = Math.max(0, Math.min(100, settings.maxExposure));
        }
      }
      updatedCount++;
    }
  });
  
  res.json({ success: true, updatedCount });
});

// Run optimization
app.post('/api/optimize', async (req, res) => {
  try {
    const {
      numLineups = 1,
      minSalary = 45000,
      maxSalary = 50000,
      stackSettings = {},
      uniquePlayers = 7,
      maxExposure = 40,
      // Advanced settings from the PyQt5 version
      monteCarloIterations = 100,
      sortingMethod = 'Points',
      minUniquePlayersBetweenLineups = 3,
      enableRiskManagement = true,
      disableKellySizing = false,
      stackTypes = {},
      exposureSettings = {},
      riskTolerance = 'medium',
      bankroll = 1000
    } = req.body;
    
    // Validate inputs
    if (playersData.length === 0) {
      return res.status(400).json({ error: 'No player data loaded' });
    }
    
    const selectedPlayers = playersData.filter(p => p.selected);
    if (selectedPlayers.length < 10) {
      return res.status(400).json({ error: 'Need at least 10 players selected' });
    }
    
    // Start optimization process
    const optimizationId = uuidv4();
    
    broadcast({
      type: 'OPTIMIZATION_STARTED',
      data: { 
        id: optimizationId, 
        numLineups,
        timestamp: new Date().toISOString()
      }
    });
    
    // Use enhanced optimizer
    const optimizer = new MLBOptimizer();
    const results = await optimizer.optimize({
      players: selectedPlayers,
      numLineups,
      minSalary,
      maxSalary,
      stackSettings,
      uniquePlayers,
      maxExposure,
      // Pass advanced settings
      monteCarloIterations,
      sortingMethod,
      minUniquePlayersBetweenLineups,
      enableRiskManagement,
      disableKellySizing,
      stackTypes,
      exposureSettings,
      riskTolerance,
      bankroll,
      onProgress: (progress) => {
        broadcast({
          type: 'OPTIMIZATION_PROGRESS',
          data: { id: optimizationId, progress, timestamp: new Date().toISOString() }
        });
      }
    });
    
    optimizationResults = results;
    
    broadcast({
      type: 'OPTIMIZATION_COMPLETED',
      data: { 
        id: optimizationId, 
        lineups: results.length,
        timestamp: new Date().toISOString()
      }
    });
    
    res.json({
      success: true,
      optimizationId,
      lineups: results,
      summary: {
        totalLineups: results.length,
        avgProjection: results.reduce((sum, l) => sum + l.totalProjection, 0) / results.length,
        avgSalary: results.reduce((sum, l) => sum + l.totalSalary, 0) / results.length,
        topProjection: results.length > 0 ? results[0].totalProjection : 0,
        strategies: [...new Set(results.map(l => l.strategy))]
      }
    });
    
  } catch (error) {
    console.error('Optimization error:', error);
    broadcast({
      type: 'OPTIMIZATION_ERROR',
      data: { error: error.message, timestamp: new Date().toISOString() }
    });
    res.status(500).json({ error: error.message });
  }
});

// Get optimization results
app.get('/api/results', (req, res) => {
  res.json({
    lineups: optimizationResults,
    count: optimizationResults.length
  });
});

// Export lineups to CSV
app.get('/api/export/:format', (req, res) => {
  const { format } = req.params;
  
  if (optimizationResults.length === 0) {
    return res.status(400).json({ error: 'No optimization results to export' });
  }
  
  try {
    if (format === 'draftkings') {
      // DraftKings format
      const dkData = optimizationResults.map((lineup, index) => {
        const players = lineup.players;
        return {
          'Entry ID': `Lineup_${index + 1}`,
          'Contest Name': 'MLB Optimizer',
          'Contest ID': '12345',
          'Entry Fee': '$1.00',
          'P': players.find(p => p.position.includes('P'))?.name || '',
          'P/UTIL': players.filter(p => p.position.includes('P'))[1]?.name || '',
          'C': players.find(p => p.position.includes('C'))?.name || '',
          '1B': players.find(p => p.position.includes('1B'))?.name || '',
          '2B': players.find(p => p.position.includes('2B'))?.name || '',
          '3B': players.find(p => p.position.includes('3B'))?.name || '',
          'SS': players.find(p => p.position.includes('SS'))?.name || '',
          'OF': players.filter(p => p.position.includes('OF'))[0]?.name || '',
          'OF ': players.filter(p => p.position.includes('OF'))[1]?.name || '',
          'OF  ': players.filter(p => p.position.includes('OF'))[2]?.name || '',
          'Total Salary': lineup.totalSalary,
          'Projected Points': lineup.totalProjection.toFixed(2)
        };
      });
      
      const parser = new Parser({ 
        fields: Object.keys(dkData[0]),
        delimiter: ','
      });
      const csv = parser.parse(dkData);
      
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename=draftkings_lineups.csv');
      res.send(csv);
      
    } else {
      // Standard format
      const standardData = [];
      optimizationResults.forEach((lineup, lineupIndex) => {
        lineup.players.forEach((player, playerIndex) => {
          standardData.push({
            'Lineup': lineupIndex + 1,
            'Position': playerIndex + 1,
            'Player': player.name,
            'Team': player.team,
            'Position_Type': player.position,
            'Salary': player.salary,
            'Projected_Points': player.projectedPoints,
            'Lineup_Total_Salary': lineup.totalSalary,
            'Lineup_Total_Points': lineup.totalProjection
          });
        });
      });
      
      const parser = new Parser();
      const csv = parser.parse(standardData);
      
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename=optimizer_lineups.csv');
      res.send(csv);
    }
    
  } catch (error) {
    console.error('Export error:', error);
    res.status(500).json({ error: 'Error generating export' });
  }
});

// Get teams and stacking options
app.get('/api/teams', (req, res) => {
  const teams = [...new Set(playersData.map(p => p.team))].sort();
  const teamStats = teams.map(team => {
    const teamPlayers = playersData.filter(p => p.team === team);
    return {
      name: team,
      playerCount: teamPlayers.length,
      avgProjection: teamPlayers.reduce((sum, p) => sum + p.projectedPoints, 0) / teamPlayers.length,
      avgSalary: teamPlayers.reduce((sum, p) => sum + p.salary, 0) / teamPlayers.length,
      positions: [...new Set(teamPlayers.map(p => p.position))].sort()
    };
  });
  
  res.json({ teams: teamStats });
});

// Save favorites lineup
app.post('/api/favorites', (req, res) => {
  try {
    const { lineup, name } = req.body;
    
    if (!lineup || !lineup.players || lineup.players.length === 0) {
      return res.status(400).json({ error: 'Invalid lineup data' });
    }

    const favorite = {
      id: uuidv4(),
      name: name || `Favorite ${Date.now()}`,
      lineup,
      timestamp: new Date().toISOString()
    };

    // In a real app, this would be saved to a database
    // For now, we'll just return success
    res.json({ success: true, favorite });

  } catch (error) {
    console.error('Save favorite error:', error);
    res.status(500).json({ error: 'Error saving favorite' });
  }
});

// Get contest formats for DraftKings integration
app.get('/api/contest-formats', (req, res) => {
  const formats = [
    {
      id: 'dk_classic',
      name: 'DraftKings Classic',
      positions: ['P', 'P/UTIL', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF'],
      salaryLimit: 50000,
      site: 'DraftKings'
    },
    {
      id: 'dk_showdown',
      name: 'DraftKings Showdown',
      positions: ['CPT', 'UTIL', 'UTIL', 'UTIL', 'UTIL', 'UTIL'],
      salaryLimit: 50000,
      site: 'DraftKings'
    },
    {
      id: 'fd_classic',
      name: 'FanDuel Classic',
      positions: ['P', 'C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL'],
      salaryLimit: 60000,
      site: 'FanDuel'
    }
  ];

  res.json({ formats });
});

// Advanced export with custom formatting
app.post('/api/export-advanced', (req, res) => {
  try {
    const { format, options = {} } = req.body;
    
    if (optimizationResults.length === 0) {
      return res.status(400).json({ error: 'No optimization results to export' });
    }

    let exportData;
    let filename;
    let contentType = 'text/csv';

    switch (format) {
      case 'contest_ready':
        // Contest-ready format with contest details
        exportData = optimizationResults.map((lineup, index) => {
          const contestEntry = {
            'Entry ID': options.entryPrefix ? `${options.entryPrefix}_${index + 1}` : `Entry_${index + 1}`,
            'Contest Name': options.contestName || 'MLB Optimizer Contest',
            'Contest ID': options.contestId || '12345',
            'Entry Fee': options.entryFee || '$1.00',
            'Total Salary': lineup.totalSalary,
            'Projected Points': lineup.totalProjection.toFixed(2),
            'Expected Value': ((lineup.totalProjection - 100) * (options.entryFee ? parseFloat(options.entryFee.replace('$', '')) : 1)).toFixed(2)
          };

          // Add player positions
          const positionMap = {
            'P': ['P', 'P/UTIL'],
            'C': ['C'],
            '1B': ['1B'],
            '2B': ['2B'],
            '3B': ['3B'],
            'SS': ['SS'],
            'OF': ['OF', 'OF ', 'OF  ']
          };

          Object.entries(positionMap).forEach(([pos, slots]) => {
            const posPlayers = lineup.players.filter(p => p.position.includes(pos));
            slots.forEach((slot, idx) => {
              contestEntry[slot] = posPlayers[idx]?.name || '';
            });
          });

          return contestEntry;
        });
        filename = 'contest_ready_lineups.csv';
        break;

      case 'detailed_analysis':
        // Detailed analysis format
        exportData = [];
        optimizationResults.forEach((lineup, lineupIndex) => {
          lineup.players.forEach((player, playerIndex) => {
            exportData.push({
              'Lineup': lineupIndex + 1,
              'Player_Order': playerIndex + 1,
              'Player': player.name,
              'Team': player.team,
              'Position': player.position,
              'Salary': player.salary,
              'Projection': player.projection,
              'Value': (player.projection / player.salary * 1000).toFixed(3),
              'Ownership': player.ownership || 0,
              'Stack_Size': lineup.stacks?.length > 0 ? lineup.stacks[0].players : 0,
              'Lineup_Salary': lineup.totalSalary,
              'Lineup_Projection': lineup.totalProjection,
              'Lineup_Value': (lineup.totalProjection / lineup.totalSalary * 1000).toFixed(3),
              'Strategy': lineup.strategy || 'Unknown',
              'Risk_Score': lineup.riskAdjustedValue || 0
            });
          });
        });
        filename = 'detailed_analysis.csv';
        break;

      case 'summary_stats':
        // Summary statistics
        const playerStats = new Map();
        const teamStats = new Map();
        
        optimizationResults.forEach(lineup => {
          lineup.players.forEach(player => {
            if (!playerStats.has(player.id)) {
              playerStats.set(player.id, {
                name: player.name,
                team: player.team,
                position: player.position,
                salary: player.salary,
                projection: player.projection,
                appearances: 0,
                totalValue: 0
              });
            }
            const stats = playerStats.get(player.id);
            stats.appearances++;
            stats.totalValue += (player.projection / player.salary * 1000);
          });
        });

        exportData = Array.from(playerStats.values()).map(stats => ({
          'Player': stats.name,
          'Team': stats.team,
          'Position': stats.position,
          'Salary': stats.salary,
          'Projection': stats.projection,
          'Appearances': stats.appearances,
          'Exposure_Percentage': ((stats.appearances / optimizationResults.length) * 100).toFixed(1),
          'Average_Value': (stats.totalValue / stats.appearances).toFixed(3)
        })).sort((a, b) => b.Appearances - a.Appearances);
        
        filename = 'player_exposure_summary.csv';
        break;

      default:
        return res.status(400).json({ error: 'Unknown export format' });
    }

    const parser = new Parser();
    const csv = parser.parse(exportData);
    
    res.setHeader('Content-Type', contentType);
    res.setHeader('Content-Disposition', `attachment; filename=${filename}`);
    res.send(csv);

  } catch (error) {
    console.error('Advanced export error:', error);
    res.status(500).json({ error: 'Error generating export' });
  }
});

// Stack analysis endpoint
app.get('/api/stack-analysis', (req, res) => {
  try {
    if (optimizationResults.length === 0) {
      return res.json({ stacks: [], analysis: null });
    }

    const stackAnalysis = {
      totalLineups: optimizationResults.length,
      stackTypes: {},
      teamStacks: {},
      averageStackSize: 0
    };

    let totalStackSize = 0;
    let stackedLineups = 0;

    optimizationResults.forEach(lineup => {
      if (lineup.stacks && lineup.stacks.length > 0) {
        stackedLineups++;
        lineup.stacks.forEach(stack => {
          totalStackSize += stack.players;
          
          // Track stack types
          const stackKey = `${stack.players}-player`;
          stackAnalysis.stackTypes[stackKey] = (stackAnalysis.stackTypes[stackKey] || 0) + 1;
          
          // Track team stacks
          if (stack.team) {
            stackAnalysis.teamStacks[stack.team] = (stackAnalysis.teamStacks[stack.team] || 0) + 1;
          }
        });
      }
    });

    stackAnalysis.averageStackSize = stackedLineups > 0 ? totalStackSize / stackedLineups : 0;
    stackAnalysis.stackedPercentage = (stackedLineups / optimizationResults.length) * 100;

    res.json({
      analysis: stackAnalysis,
      stacks: Object.entries(stackAnalysis.stackTypes).map(([type, count]) => ({
        type,
        count,
        percentage: ((count / optimizationResults.length) * 100).toFixed(1)
      }))
    });

  } catch (error) {
    console.error('Stack analysis error:', error);
    res.status(500).json({ error: 'Error analyzing stacks' });
  }
});

// Serve React app for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/build/index.html'));
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Server error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`🚀 DFS Optimizer Server running on port ${PORT}`);
  console.log(`📊 WebSocket server running on port 8080`);
  console.log(`🌐 Access the app at http://localhost:${PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

module.exports = app;
