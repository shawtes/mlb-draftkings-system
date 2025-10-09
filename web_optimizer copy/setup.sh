#!/bin/bash

# MLB DFS Optimizer Setup Script
echo "==================================="
echo "MLB DFS Optimizer Setup"
echo "==================================="

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js v16 or higher."
    echo "Download from: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check npm installation  
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed."
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Install server dependencies
echo ""
echo "📦 Installing server dependencies..."
cd server
if npm install; then
    echo "✅ Server dependencies installed successfully"
else
    echo "❌ Failed to install server dependencies"
    exit 1
fi

# Install client dependencies
echo ""
echo "📦 Installing client dependencies..."
cd ../client
if npm install; then
    echo "✅ Client dependencies installed successfully"
else
    echo "❌ Failed to install client dependencies"
    exit 1
fi

cd ..

echo ""
echo "==================================="
echo "🎉 Setup completed successfully!"
echo "==================================="
echo ""
echo "To start the application:"
echo "1. Backend:  cd server && npm start"
echo "2. Frontend: cd client && npm start"
echo ""
echo "Or use the start.bat file on Windows"
echo ""
echo "Application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend:  http://localhost:5000"
echo ""
