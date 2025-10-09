// Minimal server using only Node.js built-ins for CSV upload
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const PORT = 5000;

// Simple CORS headers
function setCORSHeaders(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

// Simple multipart parser for file uploads
function parseMultipart(data, boundary) {
  const parts = data.split('--' + boundary);
  for (const part of parts) {
    if (part.includes('filename=')) {
      const lines = part.split('\r\n');
      let csvContent = '';
      let foundEmpty = false;
      
      for (const line of lines) {
        if (foundEmpty && line.trim()) {
          csvContent += line + '\n';
        } else if (line.trim() === '') {
          foundEmpty = true;
        }
      }
      
      return csvContent.trim();
    }
  }
  return null;
}

// Simple CSV parser
function parseCSV(csvText) {
  const lines = csvText.split('\n').filter(line => line.trim());
  if (lines.length === 0) return [];
  
  const headers = lines[0].split(',').map(h => h.trim());
  const results = [];
  
  console.log('Available CSV columns:', headers);
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const data = {};
    
    headers.forEach((header, index) => {
      data[header] = values[index] ? values[index].trim() : '';
    });
    
    // Parse projection
    let projectionValue = 0;
    let projectionSource = 'none';
    
    if (data.My_Proj && !isNaN(parseFloat(data.My_Proj))) {
      projectionValue = parseFloat(data.My_Proj);
      projectionSource = 'My_Proj';
    } else if (data.PPG_Projection && !isNaN(parseFloat(data.PPG_Projection))) {
      projectionValue = parseFloat(data.PPG_Projection);
      projectionSource = 'PPG_Projection';
    }
    
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
      excluded: false,
      favorite: false,
      minExposure: 0,
      maxExposure: 100
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
        value: player.value
      });
    }
  }
  
  console.log(`Total players processed: ${results.length}`);
  console.log('Players with projections:', results.filter(p => p.projection > 0).length);
  
  return results;
}

// Global state
let playersData = [];

// Create server
const server = http.createServer((req, res) => {
  setCORSHeaders(res);
  
  // Handle OPTIONS requests
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  const parsedUrl = url.parse(req.url, true);
  const pathname = parsedUrl.pathname;
  
  // Upload endpoint
  if (pathname === '/api/upload-players' && req.method === 'POST') {
    let body = '';
    
    req.on('data', chunk => {
      body += chunk.toString();
    });
    
    req.on('end', () => {
      try {
        const contentType = req.headers['content-type'] || '';
        const boundary = contentType.split('boundary=')[1];
        
        if (!boundary) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'No boundary found' }));
          return;
        }
        
        const csvContent = parseMultipart(body, boundary);
        
        if (!csvContent) {
          res.writeHead(400);
          res.end(JSON.stringify({ error: 'No CSV content found' }));
          return;
        }
        
        playersData = parseCSV(csvContent);
        
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          success: true,
          message: `Loaded ${playersData.length} players`,
          playersCount: playersData.length,
          playersWithProjections: playersData.filter(p => p.projection > 0).length
        }));
        
      } catch (error) {
        console.error('Upload error:', error);
        res.writeHead(500);
        res.end(JSON.stringify({ error: 'Internal server error' }));
      }
    });
  }
  // Get players endpoint
  else if (pathname === '/api/players' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      players: playersData,
      total: playersData.length
    }));
  }
  // Serve static files
  else {
    const filePath = path.join(__dirname, '../client/build', pathname === '/' ? 'index.html' : pathname);
    
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end('File not found');
      } else {
        const ext = path.extname(filePath);
        const contentType = {
          '.html': 'text/html',
          '.js': 'application/javascript',
          '.css': 'text/css',
          '.json': 'application/json'
        }[ext] || 'text/plain';
        
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
      }
    });
  }
});

server.listen(PORT, () => {
  console.log(`🚀 Minimal server running on port ${PORT}`);
  console.log(`🌐 Access at http://localhost:${PORT}`);
  console.log('📁 Upload your CSV file to test projection parsing');
});
