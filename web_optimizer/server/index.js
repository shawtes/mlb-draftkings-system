const express = require('express');
const cors = require('cors');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');
const { Parser } = require('json2csv');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const { spawn } = require('child_process');
const MLBOptimizer = require('./optimizer');
const NFLOptimizer = require('./nfl-optimizer');
const NBAOptimizer = require('./nba-optimizer');

const app = express();
const PORT = process.env.PORT || 5001;

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

// WebSocket server for real-time updates (will be initialized after HTTP server)

// Global state
let playersData = [];
let optimizationResults = [];
let activeConnections = new Set();
let currentSport = 'MLB'; // Track current sport mode (MLB or NFL)
let favorites = []; // Store favorite lineups
let dkEntriesData = null; // Store parsed DK entries file data
let dkEntriesFilePath = null; // Store path to loaded DK entries file

// Favorites file path
const FAVORITES_FILE = path.join(__dirname, 'data', 'favorites.json');

// Load favorites from file on startup
function loadFavorites() {
  try {
    if (fs.existsSync(FAVORITES_FILE)) {
      const data = fs.readFileSync(FAVORITES_FILE, 'utf8');
      const parsed = JSON.parse(data);
      // Handle both array format and object with favorites property
      let loadedFavorites = [];
      if (Array.isArray(parsed)) {
        loadedFavorites = parsed;
      } else if (parsed.favorites && Array.isArray(parsed.favorites)) {
        loadedFavorites = parsed.favorites;
      } else {
        loadedFavorites = [];
      }
      
      // Normalize old favorites format to new format
      favorites = loadedFavorites.map(fav => {
        // Normalize timestamp field (old format uses createdAt)
        if (!fav.timestamp && fav.createdAt) {
          fav.timestamp = fav.createdAt;
        }
        // Normalize lineup structure (old format might have players directly)
        if (!fav.lineup && fav.players) {
          fav.lineup = {
            players: fav.players,
            totalProjection: fav.totalPoints || 0,
            totalSalary: fav.totalSalary || 0
          };
        }
        return fav;
      });
      
      console.log(`✅ Loaded ${favorites.length} favorites from file`);
    } else {
      // Create directory if it doesn't exist
      const dir = path.dirname(FAVORITES_FILE);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      favorites = [];
      console.log('📝 No favorites file found, starting with empty favorites');
    }
  } catch (error) {
    console.error('Error loading favorites:', error);
    favorites = [];
  }
}

// Save favorites to file
function saveFavorites() {
  try {
    const dir = path.dirname(FAVORITES_FILE);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(FAVORITES_FILE, JSON.stringify(favorites, null, 2), 'utf8');
    console.log(`💾 Saved ${favorites.length} favorites to file`);
  } catch (error) {
    console.error('Error saving favorites:', error);
  }
}

// Load favorites on startup
loadFavorites();

// Parse DK entries file robustly (matching desktop logic)
function parseDkEntriesRobustly(filePath) {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const lines = fileContent.split('\n').map(line => line.trim()).filter(line => line);
    
    if (lines.length === 0) {
      console.warn('DK entries file is empty');
      return null;
    }

    // Standard DK header (NBA FORMAT - 12 columns total)
    const headerLine = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
    const entryRows = [];
    
    // Check if first row is header
    let startRow = 0;
    if (lines[0].includes('Entry ID')) {
      startRow = 1;
      console.log('📋 Found header in first row');
    } else {
      console.log('📋 No header found, processing all rows');
    }
    
    for (let i = startRow; i < lines.length; i++) {
      const row = lines[i].split(',').map(cell => cell.trim().replace(/^"|"$/g, ''));
      
      if (!row || row.length === 0 || row.every(cell => !cell)) {
        continue;
      }
      
      // Check if this is an entry row (has Entry ID in first column)
      if (row.length >= 4 && row[0]) {
        const entryId = row[0].replace(/,/g, '').replace(/"/g, '').trim();
        
        // Valid entry ID should be numeric and reasonably long (8+ digits)
        if (/^\d{8,}$/.test(entryId)) {
          const contestName = row[1] || '';
          const contestId = row[2] || '';
          const entryFee = row[3] || '';
          
          // Build the clean row with exactly 12 columns (4 metadata + 8 NBA positions)
          const cleanRow = [entryId, contestName, contestId, entryFee];
          
          // Add the 8 player positions (PG, SG, SF, PF, C, G, F, UTIL)
          for (let j = 4; j < 12; j++) {
            if (j < row.length) {
              cleanRow.push(row[j] || '');
            } else {
              cleanRow.push('');
            }
          }
          
          entryRows.push(cleanRow);
        }
      }
    }
    
    console.log(`📊 Found ${entryRows.length} valid entries`);
    
    if (entryRows.length === 0) {
      console.warn('No valid entries found!');
      return null;
    }
    
    // Create array of objects
    const entries = entryRows.map(row => {
      const entry = {};
      headerLine.forEach((header, index) => {
        entry[header] = row[index] || '';
      });
      return entry;
    });
    
    return entries;
    
  } catch (error) {
    console.error('Error in robust parsing:', error);
    return null;
  }
}

// Extract player ID mapping from DK entries file (matching desktop logic)
function extractPlayerIdMappingFromDkFile(filePath) {
  const playerMap = {};
  
  if (!filePath || !fs.existsSync(filePath)) {
    return playerMap;
  }
  
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const lines = fileContent.split('\n');
    
    // Look for "Name (ID)" pattern in each line
    // Pattern: player name followed by (6+ digit number)
    const nameIdPattern = /([A-Za-z][A-Za-z\s\.\-\']+)\s*\((\d{6,})\)/g;
    
    for (const line of lines) {
      let match;
      while ((match = nameIdPattern.exec(line)) !== null) {
        const namePart = match[1].trim();
        const idPart = match[2].trim();
        
        // Additional validation for real player names
        const isLikelyDst = namePart.split(/\s+/).length === 1 && namePart.length > 3;
        const isLikelyPlayer = namePart.split(/\s+/).length >= 2;
        
        if (namePart.length > 3 && (isLikelyPlayer || isLikelyDst) && idPart.length >= 6) {
          // Don't overwrite existing mappings (first occurrence wins)
          if (!playerMap[namePart]) {
            playerMap[namePart] = idPart;
            if (Object.keys(playerMap).length <= 15) {
              console.log(`✅ Found mapping: ${namePart} -> ${idPart}`);
            }
          }
        }
      }
    }
    
    console.log(`🎯 Successfully extracted ${Object.keys(playerMap).length} player ID mappings from DK entries file`);
    if (Object.keys(playerMap).length > 0) {
      const sample = Object.entries(playerMap).slice(0, 3);
      console.log(`Sample mappings:`, sample);
    }
    
  } catch (error) {
    console.warn(`Error reading DK entries file for player mappings: ${error.message}`);
  }
  
  return playerMap;
}

// Extract contest info from DK entries (matching desktop logic)
function extractContestInfoFromDkEntries(entries) {
  if (!entries || entries.length === 0) {
    return null;
  }
  
  const contestInfoList = [];
  
  for (const entry of entries) {
    const contestInfo = {
      entry_id: entry['Entry ID'] || '1000000001',
      contest_name: entry['Contest Name'] || 'Generated Lineups',
      contest_id: entry['Contest ID'] || '999999999',
      entry_fee: entry['Entry Fee'] || '$1'
    };
    contestInfoList.push(contestInfo);
  }
  
  // Log unique contests found
  const uniqueContests = {};
  for (const info of contestInfoList) {
    const contestKey = `${info.contest_name}|${info.contest_id}`;
    if (!uniqueContests[contestKey]) {
      uniqueContests[contestKey] = info;
      console.log(`📋 Found contest: ${info.contest_name} (ID: ${info.contest_id}, Fee: ${info.entry_fee})`);
    }
  }
  
  return contestInfoList;
}

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

// Call Python Markov Chain Optimizer
async function callPythonOptimizer(config) {
  return new Promise((resolve, reject) => {
    const { onProgress, ...pythonConfig } = config;
    
    const pythonScript = path.join(__dirname, 'makrov_cli_adapter.py');
    const inputData = JSON.stringify(pythonConfig);
    
    console.log('🐍 Launching makrovchain_optimizer.py (EXACT desktop logic) with', pythonConfig.numLineups, 'lineups');
    
    const pythonProcess = spawn('python3', [pythonScript, inputData]);
    
    let outputData = '';
    let errorData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error('Python stderr:', data.toString());
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python optimizer failed:', errorData);
        reject(new Error(errorData || 'Python optimizer failed'));
        return;
      }
      
      try {
        const result = JSON.parse(outputData);
        
        if (result.error) {
          reject(new Error(result.error));
          return;
        }
        
        console.log(`✅ Python optimizer generated ${result.lineups.length} lineups`);
        console.log(`   Avg projection: ${result.summary.avgProjection}`);
        if (result.warnings && result.warnings.length > 0) {
          console.log(`   ⚠️ Warnings: ${result.warnings.join(', ')}`);
        }

        // Add IDs and timestamps to lineups
        const lineups = result.lineups.map(lineup => ({
          id: uuidv4(),
          ...lineup,
          timestamp: new Date().toISOString()
        }));

        resolve({ lineups, warnings: result.warnings || [] });
      } catch (error) {
        console.error('Failed to parse Python output:', outputData);
        reject(new Error('Failed to parse optimizer output: ' + error.message));
      }
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python process:', error);
      reject(new Error('Failed to start Python optimizer: ' + error.message));
    });
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
        
        // Enhanced projection field checking with more column names
        if (data.Predicted_DK_Points && !isNaN(parseFloat(data.Predicted_DK_Points))) {
          projectionValue = parseFloat(data.Predicted_DK_Points);
          projectionSource = 'Predicted_DK_Points';
        } else if (data.Adjusted_Projection && !isNaN(parseFloat(data.Adjusted_Projection))) {
          projectionValue = parseFloat(data.Adjusted_Projection);
          projectionSource = 'Adjusted_Projection';
        } else if (data.AvgPointsPerGame && !isNaN(parseFloat(data.AvgPointsPerGame))) {
          projectionValue = parseFloat(data.AvgPointsPerGame);
          projectionSource = 'AvgPointsPerGame';
        } else if (data.My_Proj && !isNaN(parseFloat(data.My_Proj))) {
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
          ceiling: parseFloat(data.Ceiling || data.ceiling || data.Ceil) || undefined,
          floor: parseFloat(data.Floor || data.floor) || undefined,
          stdDev: parseFloat(data.StdDev || data.stdDev || data.Std_Dev || data.SD) || undefined,
          opponent: data.Opponent || data.opponent || data.Opp || data.opp || undefined,
          selected: false,
          locked: false,
          excluded: false,
          favorite: false,
          minExposure: 0,
          maxExposure: 100
        };
        
        // Special handling for DST positions - ensure they're recognized and selected
        if (player.position && (player.position.includes('DST') || player.position.includes('Defense') || player.position.includes('D/ST'))) {
          player.position = 'DST';
          player.selected = true; // Auto-select DST players
          console.log(`✅ DST player auto-selected: ${player.name} (${player.position})`);
        }
        
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
      res.status(500).json({ error: 'Error parsing CSV file. Please check that your CSV has proper headers and data format.' });
    });
});

// Upload and parse DK entries file
app.post('/api/upload-dk-entries', upload.single('dkEntriesFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const filePath = req.file.path;
  
  try {
    // Parse DK entries file
    const entries = parseDkEntriesRobustly(filePath);
    
    if (!entries || entries.length === 0) {
      // Clean up uploaded file
      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
      return res.status(400).json({ error: 'No valid entries found in the file. Please ensure the file contains DraftKings entries with Entry ID, Contest Name, Contest ID, Entry Fee, and player positions.' });
    }
    
    // Store parsed data
    dkEntriesData = entries;
    dkEntriesFilePath = filePath; // Keep file for player ID extraction
    
    // Extract contest info
    const contestInfoList = extractContestInfoFromDkEntries(entries);
    
    // Extract player ID mappings
    const playerIdMap = extractPlayerIdMappingFromDkFile(filePath);
    
    console.log(`✅ DraftKings entries file loaded successfully!`);
    console.log(`   📁 File: ${req.file.originalname}`);
    console.log(`   📊 Number of entries: ${entries.length}`);
    console.log(`   🎯 Player ID mappings: ${Object.keys(playerIdMap).length}`);
    console.log(`   📋 Contest info entries: ${contestInfoList ? contestInfoList.length : 0}`);
    
    // Broadcast update to connected clients
    broadcast({
      type: 'DK_ENTRIES_LOADED',
      data: { 
        count: entries.length,
        playerMappings: Object.keys(playerIdMap).length,
        timestamp: new Date().toISOString() 
      }
    });
    
    res.json({
      success: true,
      message: `Loaded ${entries.length} DraftKings entries`,
      entriesCount: entries.length,
      playerMappingsCount: Object.keys(playerIdMap).length,
      contestInfo: contestInfoList ? contestInfoList[0] : null,
      uniqueContests: contestInfoList ? [...new Set(contestInfoList.map(c => `${c.contest_name}|${c.contest_id}`))].length : 0
    });
    
  } catch (error) {
    console.error('Error loading DK entries file:', error);
    
    // Clean up uploaded file
    fs.unlink(filePath, (err) => {
      if (err) console.error('Error deleting file:', err);
    });
    
    res.status(500).json({ 
      error: 'Error loading DraftKings entries file: ' + error.message,
      details: 'Please ensure the file is a valid CSV format with proper DraftKings headers (Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL)'
    });
  }
});

// Get loaded DK entries info
app.get('/api/dk-entries', (req, res) => {
  if (!dkEntriesData || dkEntriesData.length === 0) {
    return res.json({
      loaded: false,
      message: 'No DK entries file loaded'
    });
  }
  
  const playerIdMap = dkEntriesFilePath ? extractPlayerIdMappingFromDkFile(dkEntriesFilePath) : {};
  const contestInfoList = extractContestInfoFromDkEntries(dkEntriesData);
  
  res.json({
    loaded: true,
    entriesCount: dkEntriesData.length,
    playerMappingsCount: Object.keys(playerIdMap).length,
    contestInfo: contestInfoList ? contestInfoList[0] : null,
    uniqueContests: contestInfoList ? [...new Set(contestInfoList.map(c => `${c.contest_name}|${c.contest_id}`))].length : 0
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

// Bulk update players (MUST come before :id route to avoid matching "bulk" as an id)
app.put('/api/players/bulk', (req, res) => {
  const { action, filters, settings } = req.body;
  
  let updatedCount = 0;
  
  playersData.forEach((player) => {
    let shouldUpdate = true;
    
    // Apply filters
    if (filters && filters.position && filters.position !== 'ALL') {
      shouldUpdate = shouldUpdate && player.position.includes(filters.position);
    }
    if (filters && filters.team && filters.team !== 'ALL') {
      shouldUpdate = shouldUpdate && player.team === filters.team;
    }
    if (filters && filters.salaryMin) {
      shouldUpdate = shouldUpdate && player.salary >= filters.salaryMin;
    }
    if (filters && filters.salaryMax) {
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

// Set sport mode
app.post('/api/set-sport', (req, res) => {
  const { sport } = req.body;
  
  if (!sport || !['MLB', 'NFL', 'NBA'].includes(sport.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid sport. Must be MLB, NFL, or NBA' });
  }
  
  currentSport = sport.toUpperCase();
  const sportEmoji = currentSport === 'MLB' ? '⚾' : currentSport === 'NFL' ? '🏈' : '🏀';
  console.log(`${sportEmoji} Sport mode set to: ${currentSport}`);
  
  res.json({ success: true, sport: currentSport });
});

// Get current sport mode
app.get('/api/sport', (req, res) => {
  res.json({ sport: currentSport });
});

// Run optimization (supports both MLB and NFL)
app.post('/api/optimize', async (req, res) => {
  try {
    const {
      sport = currentSport,
      numLineups = 1,
      minSalary = 45000,
      maxSalary = 50000,
      stackSettings = {},
      teamSelections = {},
      teamExposures = {},
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
      contestMode = 'gpp',
      bankroll = 1000,
      advancedQuantSettings = {}
    } = req.body;
    
    // Validate inputs
    if (playersData.length === 0) {
      return res.status(400).json({ error: 'No player data loaded' });
    }
    
    const selectedPlayers = playersData.filter(p => p.selected);
    const minPlayersRequired = sport === 'NFL' ? 9 : sport === 'NBA' ? 8 : 10;
    
    if (selectedPlayers.length < minPlayersRequired) {
      return res.status(400).json({ 
        error: `Need at least ${minPlayersRequired} players selected for ${sport}` 
      });
    }
    
    // Start optimization process
    const optimizationId = uuidv4();
    
    broadcast({
      type: 'OPTIMIZATION_STARTED',
      data: { 
        id: optimizationId, 
        sport,
        numLineups,
        timestamp: new Date().toISOString()
      }
    });
    
    // Select optimizer based on sport
    let optimizer, results;
    
    if (sport === 'NFL') {
      console.log('🏈 Using NFL Optimizer');
      if (advancedQuantSettings && Object.keys(advancedQuantSettings).length > 0) {
        console.log('📊 Advanced Quant Settings:', JSON.stringify(advancedQuantSettings, null, 2));
      }
      optimizer = new NFLOptimizer();
      results = await optimizer.optimize({
        players: selectedPlayers,
        numLineups,
        minSalary: minSalary || 48000, // NFL default
        maxSalary,
        stackSettings,
        uniquePlayers,
        maxExposure,
        stackTypes,
        exposureSettings,
        riskTolerance,
        contestMode,
        bankroll,
        advancedQuantSettings,
        onProgress: (progress) => {
          broadcast({
            type: 'OPTIMIZATION_PROGRESS',
            data: { id: optimizationId, progress, timestamp: new Date().toISOString() }
          });
        }
      });
    } else if (sport === 'NBA') {
      console.log('🏀 Using Python Markov Chain NBA Optimizer');
      if (advancedQuantSettings && Object.keys(advancedQuantSettings).length > 0) {
        console.log('📊 Advanced Quant Settings:', JSON.stringify(advancedQuantSettings, null, 2));
      }
      
      // Debug: Log what we're sending to Python
      console.log('🔍 DEBUG - Stack Settings:', JSON.stringify(stackSettings, null, 2));
      console.log('🔍 DEBUG - Team Selections:', JSON.stringify(teamSelections, null, 2));
      console.log('🔍 DEBUG - Team Exposures:', JSON.stringify(teamExposures, null, 2));

      // Call Python optimizer
      const pythonResult = await callPythonOptimizer({
        players: selectedPlayers,
        numLineups,
        minSalary: minSalary || 48000,
        maxSalary,
        stackSettings,
        teamSelections,
        teamExposures,
        uniquePlayers,
        maxExposure,
        onProgress: (progress) => {
          broadcast({
            type: 'OPTIMIZATION_PROGRESS',
            data: { id: optimizationId, progress, timestamp: new Date().toISOString() }
          });
        }
      });
      results = pythonResult.lineups;
      var optimizerWarnings = pythonResult.warnings || [];
    } else {
      console.log('⚾ Using MLB Optimizer');
      if (advancedQuantSettings && Object.keys(advancedQuantSettings).length > 0) {
        console.log('📊 Advanced Quant Settings:', JSON.stringify(advancedQuantSettings, null, 2));
      }
      optimizer = new MLBOptimizer();
      results = await optimizer.optimize({
        players: selectedPlayers,
        numLineups,
        minSalary: minSalary || 45000, // MLB default
        maxSalary,
        stackSettings,
        uniquePlayers,
        maxExposure,
        monteCarloIterations,
        sortingMethod,
        minUniquePlayersBetweenLineups,
        enableRiskManagement,
        disableKellySizing,
        stackTypes,
        exposureSettings,
        riskTolerance,
        bankroll,
        advancedQuantSettings,
        onProgress: (progress) => {
          broadcast({
            type: 'OPTIMIZATION_PROGRESS',
            data: { id: optimizationId, progress, timestamp: new Date().toISOString() }
          });
        }
      });
    }
    
    optimizationResults = results;
    
    broadcast({
      type: 'OPTIMIZATION_COMPLETED',
      data: { 
        id: optimizationId,
        sport,
        lineups: results.length,
        timestamp: new Date().toISOString()
      }
    });
    
    res.json({
      success: true,
      sport,
      optimizationId,
      lineups: results,
      warnings: typeof optimizerWarnings !== 'undefined' ? optimizerWarnings : [],
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

// Get lineups by sport
app.get('/api/lineups/:sport', (req, res) => {
  const { sport } = req.params;
  const { format = 'full' } = req.query;
  
  if (!['MLB', 'NFL', 'NBA'].includes(sport.toUpperCase())) {
    return res.status(400).json({ error: 'Invalid sport. Must be MLB, NFL, or NBA' });
  }
  
  // Return optimization results if available
  let filteredLineups = optimizationResults || [];
  
  // Format lineups based on request
  if (format === 'summary') {
    filteredLineups = optimizationResults.map(lineup => ({
      id: lineup.id,
      totalSalary: lineup.totalSalary,
      totalProjection: lineup.totalProjection,
      value: lineup.totalProjection / lineup.totalSalary * 1000,
      strategy: lineup.strategy || 'Unknown',
      playerCount: lineup.players.length,
      stacks: lineup.stacks || [],
      timestamp: lineup.timestamp
    }));
  }
  
  res.json({
    sport: sport.toUpperCase(),
    lineups: filteredLineups,
    count: filteredLineups.length,
    summary: {
      avgProjection: filteredLineups.length > 0 
        ? filteredLineups.reduce((sum, l) => sum + l.totalProjection, 0) / filteredLineups.length 
        : 0,
      avgSalary: filteredLineups.length > 0 
        ? filteredLineups.reduce((sum, l) => sum + l.totalSalary, 0) / filteredLineups.length 
        : 0,
      avgValue: filteredLineups.length > 0 
        ? filteredLineups.reduce((sum, l) => sum + (l.totalProjection / l.totalSalary * 1000), 0) / filteredLineups.length 
        : 0
    }
  });
});

// Export lineups to CSV
app.get('/api/export/:format', (req, res) => {
  const { format } = req.params;
  const { sport = currentSport } = req.query;
  
  if (optimizationResults.length === 0) {
    return res.status(400).json({ error: 'No optimization results to export' });
  }
  
  try {
    if (format === 'draftkings') {
      // DraftKings format - different for MLB vs NFL
      const dkData = optimizationResults.map((lineup, index) => {
        const players = lineup.players;
        
        if (sport === 'NFL') {
          // NFL DraftKings format
          return {
            'QB': players.find(p => p.position === 'QB')?.name || '',
            'RB': players.filter(p => p.position === 'RB')[0]?.name || '',
            'RB ': players.filter(p => p.position === 'RB')[1]?.name || '',
            'WR': players.filter(p => p.position === 'WR')[0]?.name || '',
            'WR ': players.filter(p => p.position === 'WR')[1]?.name || '',
            'WR  ': players.filter(p => p.position === 'WR')[2]?.name || '',
            'TE': players.find(p => p.position === 'TE')?.name || '',
            'FLEX': players[7]?.name || '', // 8th player (0-indexed position 7)
            'DST': players.find(p => p.position === 'DST')?.name || ''
          };
        } else {
          // MLB DraftKings format
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
        }
      });
      
      const parser = new Parser({ 
        fields: Object.keys(dkData[0]),
        delimiter: ','
      });
      const csv = parser.parse(dkData);
      
      const filename = sport === 'NFL' ? 'nfl_draftkings_lineups.csv' : 'mlb_draftkings_lineups.csv';
      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', `attachment; filename=${filename}`);
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

// Get all favorites
app.get('/api/favorites', (req, res) => {
  try {
    res.json({ success: true, favorites });
  } catch (error) {
    console.error('Get favorites error:', error);
    res.status(500).json({ error: 'Error fetching favorites' });
  }
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

    // Add to favorites array
    favorites.push(favorite);
    console.log(`✅ Saved favorite: ${favorite.name} (Total: ${favorites.length})`);
    
    // Save to file
    saveFavorites();

    res.json({ success: true, favorite, totalFavorites: favorites.length });

  } catch (error) {
    console.error('Save favorite error:', error);
    res.status(500).json({ error: 'Error saving favorite' });
  }
});

// Delete a favorite
app.delete('/api/favorites/:id', (req, res) => {
  try {
    const { id } = req.params;
    const index = favorites.findIndex(f => f.id === id);
    
    if (index === -1) {
      return res.status(404).json({ error: 'Favorite not found' });
    }

    favorites.splice(index, 1);
    console.log(`✅ Deleted favorite: ${id} (Total: ${favorites.length})`);
    
    // Save to file
    saveFavorites();

    res.json({ success: true, totalFavorites: favorites.length });
  } catch (error) {
    console.error('Delete favorite error:', error);
    res.status(500).json({ error: 'Error deleting favorite' });
  }
});

// Export lineups/favorites to DraftKings format
// Format lineup positions using the EXACT desktop algorithm
function formatLineupPositionsOnly(players, playerNameToIdMap) {
  // Sort lineup by projection descending (matching desktop)
  const sortedPlayers = [...players].sort((a, b) => {
    const projA = a.projection || a.projectedPoints || a.Predicted_DK_Points || 0;
    const projB = b.projection || b.projectedPoints || b.Predicted_DK_Points || 0;
    return projB - projA;
  });

  // Create position pools (players can be in multiple pools)
  const positionPlayers = {
    PG: [], SG: [], SF: [], PF: [], C: [],
    G: [], F: [], UTIL: []
  };

  const missingIds = [];

  // Build position eligibility pools
  sortedPlayers.forEach(player => {
    const rosterPos = (player.rosterPosition || player.position || '').toUpperCase();
    const playerName = (player.name || player.Name || '').trim();
    
    // Get DK ID with case-insensitive matching
    let dkId = playerNameToIdMap[playerName];
    if (!dkId) {
      const nameLower = playerName.toLowerCase();
      for (const [name, id] of Object.entries(playerNameToIdMap)) {
        if (name.toLowerCase() === nameLower) {
          dkId = id;
          break;
        }
      }
    }

    if (!dkId) {
      missingIds.push(playerName);
      return;
    }

    // Build position eligibility (matching desktop logic)
    const posEligible = [];
    
    if (rosterPos.includes('PG')) posEligible.push('PG', 'G', 'UTIL');
    if (rosterPos.includes('SG')) posEligible.push('SG', 'G', 'UTIL');
    if (rosterPos.includes('SF')) posEligible.push('SF', 'F', 'UTIL');
    if (rosterPos.includes('PF')) posEligible.push('PF', 'F', 'UTIL');
    if (rosterPos.includes('C')) posEligible.push('C', 'UTIL');
    if (rosterPos.includes('G') && !rosterPos.includes('PG') && !rosterPos.includes('SG')) {
      posEligible.push('G', 'UTIL');
    }
    if (rosterPos.includes('F') && !rosterPos.includes('SF') && !rosterPos.includes('PF')) {
      posEligible.push('F', 'UTIL');
    }

    // Add player to ALL eligible position pools
    const uniquePositions = [...new Set(posEligible)];
    uniquePositions.forEach(pos => {
      if (positionPlayers[pos]) {
        positionPlayers[pos].push(dkId);
      }
    });
  });

  if (missingIds.length > 0) {
    throw new Error(`Missing DraftKings IDs for: ${missingIds.join(', ')}`);
  }

  // Initialize position assignments array [PG, SG, SF, PF, C, G, F, UTIL]
  const positionAssignments = ['', '', '', '', '', '', '', ''];
  const usedPlayerIds = new Set();

  // Helper to assign player to slot
  function assignPlayer(slotIndex, playerId) {
    if (playerId && !usedPlayerIds.has(playerId)) {
      positionAssignments[slotIndex] = playerId;
      usedPlayerIds.add(playerId);
      return true;
    }
    return false;
  }

  // Helper to check eligibility
  function isPlayerEligibleForSlot(playerId, slotIndex) {
    const slotNames = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
    const slotName = slotNames[slotIndex];
    
    // Find player's position
    let playerPos = '';
    for (const player of sortedPlayers) {
      const name = (player.name || player.Name || '').trim();
      let checkId = playerNameToIdMap[name];
      if (!checkId) {
        const nameLower = name.toLowerCase();
        for (const [n, id] of Object.entries(playerNameToIdMap)) {
          if (n.toLowerCase() === nameLower) {
            checkId = id;
            break;
          }
        }
      }
      if (checkId === playerId) {
        playerPos = (player.rosterPosition || player.position || '').toUpperCase();
        break;
      }
    }

    if (!playerPos) return false;

    // DraftKings eligibility rules
    if (slotName === 'PG') return playerPos.includes('PG');
    if (slotName === 'SG') return playerPos.includes('SG');
    if (slotName === 'SF') return playerPos.includes('SF');
    if (slotName === 'PF') return playerPos.includes('PF');
    if (slotName === 'C') return playerPos.includes('C');
    if (slotName === 'G') return playerPos.includes('PG') || playerPos.includes('SG') || playerPos.includes('G');
    if (slotName === 'F') return playerPos.includes('SF') || playerPos.includes('PF') || playerPos.includes('F');
    if (slotName === 'UTIL') return true;
    
    return false;
  }

  // Phase 1: Fill C (slot 4) - most constrained
  if (positionPlayers.C.length > 0) {
    for (const playerId of positionPlayers.C) {
      if (assignPlayer(4, playerId)) break;
    }
  }

  // Phase 2: Fill PG (slot 0) - prioritize pure PG, then dual-eligible
  const pgCandidates = positionPlayers.PG.filter(id => !usedPlayerIds.has(id));
  const purePGs = pgCandidates.filter(id => !positionPlayers.SG.includes(id));
  const dualPGs = pgCandidates.filter(id => positionPlayers.SG.includes(id));
  for (const playerId of [...purePGs, ...dualPGs]) {
    if (assignPlayer(0, playerId)) break;
  }

  // Phase 3: Fill SG (slot 1)
  const sgCandidates = positionPlayers.SG.filter(id => !usedPlayerIds.has(id));
  const pureSGs = sgCandidates.filter(id => !positionPlayers.PG.includes(id));
  const dualSGs = sgCandidates.filter(id => positionPlayers.PG.includes(id));
  for (const playerId of [...pureSGs, ...dualSGs]) {
    if (assignPlayer(1, playerId)) break;
  }

  // Phase 4: Fill SF (slot 2)
  const sfCandidates = positionPlayers.SF.filter(id => !usedPlayerIds.has(id));
  const pureSFs = sfCandidates.filter(id => !positionPlayers.PF.includes(id));
  const dualSFs = sfCandidates.filter(id => positionPlayers.PF.includes(id));
  for (const playerId of [...pureSFs, ...dualSFs]) {
    if (assignPlayer(2, playerId)) break;
  }

  // Phase 5: Fill PF (slot 3)
  const pfCandidates = positionPlayers.PF.filter(id => !usedPlayerIds.has(id));
  const purePFs = pfCandidates.filter(id => !positionPlayers.SF.includes(id));
  const dualPFs = pfCandidates.filter(id => positionPlayers.SF.includes(id));
  for (const playerId of [...purePFs, ...dualPFs]) {
    if (assignPlayer(3, playerId)) break;
  }

  // Phase 6: Fill G (slot 5) - any unused guard-eligible player
  for (const playerId of [...positionPlayers.PG, ...positionPlayers.SG]) {
    if (!usedPlayerIds.has(playerId)) {
      if (assignPlayer(5, playerId)) break;
    }
  }

  // Phase 7: Fill F (slot 6) - any unused forward-eligible player
  for (const playerId of [...positionPlayers.SF, ...positionPlayers.PF]) {
    if (!usedPlayerIds.has(playerId)) {
      if (assignPlayer(6, playerId)) break;
    }
  }

  // Phase 8: Fill UTIL (slot 7) - any remaining player
  for (const playerId of positionPlayers.UTIL) {
    if (assignPlayer(7, playerId)) break;
  }

  // Phase 9: Enhanced backfill - fill empty positions
  const slotNames = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
  
  // Strategy 1: Fill from position-specific pools
  for (let i = 0; i < 8; i++) {
    if (!positionAssignments[i]) {
      const slotName = slotNames[i];
      if (positionPlayers[slotName]) {
        for (const playerId of positionPlayers[slotName]) {
          if (!usedPlayerIds.has(playerId) && assignPlayer(i, playerId)) {
            break;
          }
        }
      }
    }
  }

  // Strategy 2: Fill from UTIL pool with eligibility checking
  for (let i = 0; i < 8; i++) {
    if (!positionAssignments[i]) {
      for (const playerId of positionPlayers.UTIL) {
        if (!usedPlayerIds.has(playerId) && isPlayerEligibleForSlot(playerId, i)) {
          if (assignPlayer(i, playerId)) break;
        }
      }
    }
  }

  // Strategy 3: Last resort - try any remaining eligible player
  for (let i = 0; i < 8; i++) {
    if (!positionAssignments[i]) {
      for (const player of sortedPlayers) {
        const name = (player.name || player.Name || '').trim();
        let playerId = playerNameToIdMap[name];
        if (!playerId) {
          const nameLower = name.toLowerCase();
          for (const [n, id] of Object.entries(playerNameToIdMap)) {
            if (n.toLowerCase() === nameLower) {
              playerId = id;
              break;
            }
          }
        }
        if (playerId && !usedPlayerIds.has(playerId) && isPlayerEligibleForSlot(playerId, i)) {
          if (assignPlayer(i, playerId)) break;
        }
      }
    }
  }

  return positionAssignments;
}

app.post('/api/export-dk-entries', (req, res) => {
  try {
    const { lineups, type = 'lineups', contestName, contestId, entryFee } = req.body;
    
    if (!lineups || !Array.isArray(lineups) || lineups.length === 0) {
      return res.status(400).json({ error: 'No lineups provided for export' });
    }

    // PRIORITY 1: Extract player ID mapping from DK entries file (matching desktop)
    let playerNameToIdMap = {};
    if (dkEntriesFilePath && fs.existsSync(dkEntriesFilePath)) {
      playerNameToIdMap = extractPlayerIdMappingFromDkFile(dkEntriesFilePath);
      console.log(`🎯 PRIORITY: Using ${Object.keys(playerNameToIdMap).length} player IDs from DK entries file`);
    }
    
    // PRIORITY 2: Create player name to DK ID mapping from loaded player data
    if (Object.keys(playerNameToIdMap).length === 0) {
      playersData.forEach(player => {
        const name = (player.name || player.Name || '').trim();
        // Try to find DK ID in various possible fields (matching desktop priority)
        const dkId = player.dkId || player.DK_ID || player.DraftKingsID || player.operatorPlayerId || 
                     player.draftkingsId || player.draftKingsId || player.id;
        if (name && dkId) {
          const cleanId = String(dkId).replace(/\.0$/, '').trim();
          if (cleanId && cleanId !== 'undefined' && cleanId !== 'null') {
            // Don't overwrite if already exists from DK file
            if (!playerNameToIdMap[name]) {
              playerNameToIdMap[name] = cleanId;
            }
          }
        }
      });
    }
    
    // PRIORITY 3: Also extract from lineup player data
    if (lineups.length > 0) {
      const firstLineup = lineups[0];
      const players = firstLineup.players || firstLineup.lineup?.players || [];
      players.forEach(player => {
        const name = (player.name || player.Name || '').trim();
        const dkId = player.dkId || player.DK_ID || player.DraftKingsID || player.id;
        if (name && dkId && !playerNameToIdMap[name]) {
          const cleanId = String(dkId).replace(/\.0$/, '').trim();
          if (cleanId && cleanId !== 'undefined' && cleanId !== 'null') {
            playerNameToIdMap[name] = cleanId;
          }
        }
      });
    }

    if (Object.keys(playerNameToIdMap).length === 0) {
      return res.status(400).json({ 
        error: 'No DraftKings player IDs found. Please load a DK entries file or ensure player data includes DK_ID or DraftKingsID fields.' 
      });
    }

    // PRIORITY 1: Extract contest info from DK entries file (matching desktop)
    let finalContestName = contestName;
    let finalContestId = contestId;
    let finalEntryFee = entryFee;
    let reservedEntryIds = [];
    
    if (dkEntriesData && dkEntriesData.length > 0) {
      const contestInfoList = extractContestInfoFromDkEntries(dkEntriesData);
      if (contestInfoList && contestInfoList.length > 0) {
        const firstContest = contestInfoList[0];
        // Use contest info from DK file if not provided in request
        if (!finalContestName) finalContestName = firstContest.contest_name;
        if (!finalContestId) finalContestId = firstContest.contest_id;
        if (!finalEntryFee) finalEntryFee = firstContest.entry_fee;
        
        // Extract reserved entry IDs from DK entries file
        dkEntriesData.forEach(entry => {
          const entryId = entry['Entry ID'];
          if (entryId && /^\d{8,}$/.test(String(entryId))) {
            reservedEntryIds.push(String(entryId));
          }
        });
        
        console.log(`🎯 Using contest info from DK entries file: ${finalContestName} (ID: ${finalContestId})`);
        console.log(`🎯 Found ${reservedEntryIds.length} reserved entry IDs from DK file`);
      }
    }
    
    // Fallback to defaults or request values
    if (!finalContestName) finalContestName = 'NBA Main Slate';
    if (!finalContestId) finalContestId = '999999999';
    if (!finalEntryFee) finalEntryFee = '$1';

    // Limit lineups to available reserved entry IDs if using DK file
    let lineupsToExport = lineups;
    if (reservedEntryIds.length > 0 && lineups.length > reservedEntryIds.length) {
      console.log(`⚠️ Limiting export to ${reservedEntryIds.length} lineups (available reserved entry IDs)`);
      lineupsToExport = lineups.slice(0, reservedEntryIds.length);
    }
    
    console.log(`📊 Exporting ${lineupsToExport.length} ${type} to DraftKings format`);
    console.log(`   Player ID mappings: ${Object.keys(playerNameToIdMap).length} players`);
    console.log(`   Contest: ${finalContestName} (ID: ${finalContestId}, Fee: ${finalEntryFee})`);

    // Format lineups for DraftKings using desktop algorithm
    const dkEntries = [];
    let baseEntryId = 4763920000; // Default base entry ID
    
    // Use reserved entry IDs if available
    if (reservedEntryIds.length > 0) {
      console.log(`🎯 Using reserved entry IDs from DK entries file`);
    }

    lineupsToExport.forEach((lineup, index) => {
      try {
        // Get players from lineup (handle both formats)
        let players = [];
        if (Array.isArray(lineup.players)) {
          players = lineup.players;
        } else if (lineup.lineup && Array.isArray(lineup.lineup.players)) {
          players = lineup.lineup.players;
        } else if (Array.isArray(lineup)) {
          players = lineup;
        }
        
        if (players.length !== 8) {
          console.warn(`⚠️ Lineup ${index + 1} has ${players.length} players, skipping`);
          return;
        }

        // Use desktop algorithm to format positions
        const positionAssignments = formatLineupPositionsOnly(players, playerNameToIdMap);

        // Validate all positions are filled
        const allFilled = positionAssignments.every(id => id && id.trim() !== '');
        if (!allFilled) {
          console.warn(`⚠️ Lineup ${index + 1} missing positions after formatting, skipping`);
          return;
        }

        // Create entry row [Entry ID, Contest Name, Contest ID, Entry Fee, PG, SG, SF, PF, C, G, F, UTIL]
        // Use reserved entry ID if available, otherwise generate one
        let entryId;
        if (reservedEntryIds.length > index) {
          entryId = reservedEntryIds[index];
          console.log(`✅ Using reserved entry ID: ${entryId} for lineup ${index + 1}`);
        } else {
          entryId = String(baseEntryId + index);
        }
        
        const entry = [
          entryId,
          finalContestName,
          finalContestId,
          finalEntryFee,
          positionAssignments[0], // PG
          positionAssignments[1], // SG
          positionAssignments[2], // SF
          positionAssignments[3], // PF
          positionAssignments[4], // C
          positionAssignments[5], // G
          positionAssignments[6], // F
          positionAssignments[7]  // UTIL
        ];

        dkEntries.push(entry);
      } catch (error) {
        console.error(`❌ Error formatting lineup ${index + 1}:`, error.message);
        // Continue with other lineups
      }
    });

    if (dkEntries.length === 0) {
      return res.status(400).json({ error: 'No valid lineups could be formatted for export' });
    }

    // Create CSV (matching desktop format exactly)
    const headers = ['Entry ID', 'Contest Name', 'Contest ID', 'Entry Fee', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'];
    const csvRows = [headers, ...dkEntries.map(entry => entry.map(val => `"${val}"`).join(','))];
    const csvContent = csvRows.join('\n');

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="dk_entries_${Date.now()}.csv"`);
    res.send(csvContent);

    console.log(`✅ Exported ${dkEntries.length} lineups to DraftKings format`);

  } catch (error) {
    console.error('Export DK entries error:', error);
    res.status(500).json({ error: 'Error exporting to DraftKings format: ' + error.message });
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

// Upload ownership CSV and merge into existing player pool
app.post('/api/upload-ownership', upload.single('ownershipFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const filePath = req.file.path;
  const ownershipData = [];

  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => {
      const name = data.Name || data.name || data.NAME || data.Player || data.player || '';
      const ownership = parseFloat(data.Ownership || data.ownership || data.Own || data.own || data['Own%'] || data['own%']) || 0;
      if (name && ownership > 0) {
        ownershipData.push({ name: name.trim(), ownership });
      }
    })
    .on('end', () => {
      let matched = 0;
      ownershipData.forEach(({ name, ownership }) => {
        const nameLower = name.toLowerCase();
        const player = playersData.find(p => p.name.toLowerCase() === nameLower);
        if (player) {
          player.ownership = ownership;
          matched++;
        }
      });

      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });

      console.log(`📊 Ownership upload: ${matched}/${ownershipData.length} players matched`);
      res.json({ success: true, matched, total: ownershipData.length });
    })
    .on('error', (error) => {
      console.error('Ownership CSV parsing error:', error);
      res.status(500).json({ error: 'Error parsing ownership CSV' });
    });
});

// Upload projection source (for blending) - returns parsed data without merging
app.post('/api/upload-projection-source', upload.single('projectionFile'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const filePath = req.file.path;
  const sourceName = req.body.sourceName || req.file.originalname.replace('.csv', '');
  const players = [];

  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => {
      const name = data.Name || data.name || data.NAME || data.Player || data.player || '';
      let projection = 0;
      if (data.Predicted_DK_Points) projection = parseFloat(data.Predicted_DK_Points);
      else if (data.Projection) projection = parseFloat(data.Projection);
      else if (data.projection) projection = parseFloat(data.projection);
      else if (data.My_Proj) projection = parseFloat(data.My_Proj);
      else if (data.AvgPointsPerGame) projection = parseFloat(data.AvgPointsPerGame);
      else if (data.PPG_Projection) projection = parseFloat(data.PPG_Projection);

      if (name && projection > 0) {
        players.push({ name: name.trim(), projection });
      }
    })
    .on('end', () => {
      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });

      console.log(`📊 Projection source "${sourceName}": ${players.length} players parsed`);
      res.json({ success: true, source: sourceName, players });
    })
    .on('error', (error) => {
      console.error('Projection source CSV parsing error:', error);
      res.status(500).json({ error: 'Error parsing projection CSV' });
    });
});

// Betting / Odds stub endpoints (return empty so client falls back to sample data)
app.get('/api/odds/games', (req, res) => {
  res.json([]);
});

app.get('/api/odds/props', (req, res) => {
  res.json([]);
});

app.post('/api/bets/calculate-payout', (req, res) => {
  // Let client-side fallback handle the calculation
  res.status(501).json({ error: 'Use client-side calculation' });
});

app.get('/api/odds/convert', (req, res) => {
  res.status(501).json({ error: 'Use client-side conversion' });
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
  console.log(`📊 WebSocket server running on port ${PORT}/ws`);
  console.log(`🌐 Access the app at http://localhost:${PORT}`);
});

// Initialize WebSocket server on the same port as HTTP server
const wss = new WebSocket.Server({ server });

// WebSocket connection handler
wss.on('connection', (ws) => {
  activeConnections.add(ws);
  console.log('✅ WebSocket client connected. Total connections:', activeConnections.size);
  
  ws.on('close', () => {
    activeConnections.delete(ws);
    console.log('❌ WebSocket client disconnected. Total connections:', activeConnections.size);
  });
  
  ws.on('error', (error) => {
    console.error('⚠️  WebSocket error:', error);
    activeConnections.delete(ws);
  });
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
