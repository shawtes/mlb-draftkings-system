const express = require('express');

// Test basic Express setup
const app = express();
const PORT = 5001;

app.get('/test', (req, res) => {
  res.json({ message: 'Server is working!', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Test server running on http://localhost:${PORT}`);
  console.log('Visit http://localhost:5001/test to verify');
});
