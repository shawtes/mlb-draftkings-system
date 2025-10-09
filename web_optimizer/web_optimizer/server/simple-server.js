// Minimal server to test CSV upload and projection parsing
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 5001; // Use different port

// Middleware
app.use(cors());
app.use(express.json());

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

const upload = multer({ storage });

// Global state
let playersData = [];

// Upload and parse CSV data
app.post('/api/upload-players', upload.single('playersFile'), (req, res) => {
  console.log('File upload received');
  
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const results = [];
  const filePath = req.file.path;
  
  console.log('Processing file:', filePath);

  fs.createReadStream(filePath)
    .pipe(csv())
    .on('data', (data) => {
      try {
        // Log available columns from first row
        if (results.length === 0) {
          console.log('Available CSV columns:', Object.keys(data));
        }
        
        // Simple projection field checking
        let projectionValue = 0;
        let projectionSource = 'none';
        
        if (data.My_Proj && !isNaN(parseFloat(data.My_Proj))) {
          projectionValue = parseFloat(data.My_Proj);
          projectionSource = 'My_Proj';
        } else if (data.PPG_Projection && !isNaN(parseFloat(data.PPG_Projection))) {
          projectionValue = parseFloat(data.PPG_Projection);
          projectionSource = 'PPG_Projection';
        }
        
        // Process player data
        const player = {
          id: Date.now() + Math.random(),
          name: data.Name || '',
          team: data.Team || '',
          position: data.Pos || '',
          salary: parseInt(data.Salary) || 0,
          projection: projectionValue,
          value: 0,
          selected: false,
          locked: false,
          excluded: false
        };
        
        // Calculate value
        if (player.salary > 0 && player.projection > 0) {
          player.value = parseFloat((player.projection / player.salary * 1000).toFixed(2));
        }
        
        results.push(player);
        
        // Log first few players
        if (results.length <= 3) {
          console.log(`Player ${results.length}:`, {
            name: player.name,
            projection: player.projection,
            projectionSource: projectionSource,
            rawMyProj: data.My_Proj
          });
        }
      } catch (error) {
        console.error('Error processing player:', error);
      }
    })
    .on('end', () => {
      playersData = results;
      
      console.log(`Total players processed: ${results.length}`);
      console.log('Players with projections:', results.filter(p => p.projection > 0).length);
      
      // Clean up file
      fs.unlink(filePath, (err) => {
        if (err) console.error('Error deleting file:', err);
      });
      
      res.json({
        success: true,
        message: `Loaded ${results.length} players`,
        playersCount: results.length,
        playersWithProjections: results.filter(p => p.projection > 0).length
      });
    })
    .on('error', (error) => {
      console.error('CSV parsing error:', error);
      res.status(500).json({ error: 'Error parsing CSV file' });
    });
});

// Get players data
app.get('/api/players', (req, res) => {
  res.json({
    players: playersData,
    total: playersData.length
  });
});

// Static files for client
app.use(express.static(path.join(__dirname, '../client/build')));

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`🚀 Simple server running on port ${PORT}`);
  console.log(`🌐 Access at http://localhost:${PORT}`);
});
