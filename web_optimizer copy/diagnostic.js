const http = require('http');

console.log('🔍 DFS Optimizer Diagnostic Tool');
console.log('================================');

function testServer(port, path = '', description) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: port,
      path: path,
      method: 'GET'
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        resolve({
          status: res.statusCode,
          data: data,
          headers: res.headers
        });
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.end();
  });
}

async function runDiagnostics() {
  console.log('\n1. Testing Backend Server (Port 5000)...');
  try {
    const backendResponse = await testServer(5000, '/api/health');
    console.log('✅ Backend server is responding');
    console.log('   Status:', backendResponse.status);
    console.log('   Response:', backendResponse.data.substring(0, 200));
  } catch (error) {
    console.log('❌ Backend server error:', error.message);
  }

  console.log('\n2. Testing Frontend Server (Port 3000)...');
  try {
    const frontendResponse = await testServer(3000, '/');
    console.log('✅ Frontend server is responding');
    console.log('   Status:', frontendResponse.status);
    console.log('   Content type:', frontendResponse.headers['content-type']);
    console.log('   Content length:', frontendResponse.data.length);
    
    // Check if it's the loading screen or actual React app
    if (frontendResponse.data.includes('Loading DFS Optimizer...')) {
      console.log('⚠️  Frontend is showing loading screen (React app may not be compiling)');
    } else if (frontendResponse.data.includes('root')) {
      console.log('✅ Frontend has proper HTML structure');
    }
    
    // Check for common React errors
    if (frontendResponse.data.includes('Module not found') || 
        frontendResponse.data.includes('Failed to compile') ||
        frontendResponse.data.includes('SyntaxError')) {
      console.log('❌ React compilation errors detected');
    }
    
  } catch (error) {
    console.log('❌ Frontend server error:', error.message);
  }

  console.log('\n3. Testing WebSocket Server (Port 8080)...');
  try {
    const wsResponse = await testServer(8080, '/');
    console.log('✅ WebSocket server is responding');
  } catch (error) {
    console.log('❌ WebSocket server error:', error.message);
  }

  console.log('\n4. Checking React Build Status...');
  const fs = require('fs');
  const path = require('path');
  
  const buildPath = path.join(__dirname, 'client', 'build');
  const srcPath = path.join(__dirname, 'client', 'src');
  
  if (fs.existsSync(buildPath)) {
    const buildFiles = fs.readdirSync(buildPath);
    console.log('✅ React build directory exists');
    console.log('   Build files:', buildFiles.slice(0, 5).join(', '));
  } else {
    console.log('❌ React build directory not found');
  }
  
  if (fs.existsSync(srcPath)) {
    console.log('✅ React source directory exists');
  } else {
    console.log('❌ React source directory not found');
  }

  console.log('\n📋 Next Steps:');
  console.log('1. Check the browser developer console for JavaScript errors');
  console.log('2. Try accessing http://localhost:3000 directly');
  console.log('3. If you see a loading screen, React may be failing to compile');
  console.log('4. Check the React development server console for error messages');
}

runDiagnostics().catch(console.error); 