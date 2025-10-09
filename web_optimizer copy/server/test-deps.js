// Test script to check what's causing the server error
console.log('Starting dependency test...');

try {
  console.log('Testing express...');
  const express = require('express');
  console.log('✅ express loaded');

  console.log('Testing cors...');
  const cors = require('cors');
  console.log('✅ cors loaded');

  console.log('Testing multer...');
  const multer = require('multer');
  console.log('✅ multer loaded');

  console.log('Testing csv-parser...');
  const csv = require('csv-parser');
  console.log('✅ csv-parser loaded');

  console.log('Testing built-ins...');
  const fs = require('fs');
  const path = require('path');
  console.log('✅ fs and path loaded');

  console.log('Testing json2csv...');
  const { Parser } = require('json2csv');
  console.log('✅ json2csv loaded');

  console.log('Testing ws...');
  const WebSocket = require('ws');
  console.log('✅ ws loaded');

  console.log('Testing uuid...');
  const { v4: uuidv4 } = require('uuid');
  console.log('✅ uuid loaded');

  console.log('All dependencies loaded successfully!');
  
  // Test basic server startup
  const app = express();
  const server = app.listen(3333, () => {
    console.log('✅ Basic server started on port 3333');
    server.close(() => {
      console.log('✅ Server closed successfully');
      process.exit(0);
    });
  });

} catch (error) {
  console.error('❌ Error:', error.message);
  console.error('Stack:', error.stack);
  process.exit(1);
}
