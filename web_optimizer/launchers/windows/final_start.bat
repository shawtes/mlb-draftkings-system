@echo off
title MLB DFS Web Optimizer Startup
color 0a

echo.
echo    =====================================
echo       🚀 MLB DFS WEB OPTIMIZER 🚀
echo    =====================================
echo.

REM Set Node.js path
set "NODE_PATH=C:\Program Files\nodejs"
set "PATH=%NODE_PATH%;%PATH%"

REM Check if Node.js is available
echo 🔍 Checking Node.js installation...
"%NODE_PATH%\node.exe" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js not found! Please install from https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Node.js found and ready
echo.

REM Kill any existing node processes
echo 🧹 Cleaning up any existing processes...
taskkill /f /im node.exe >nul 2>&1

echo 📦 Checking dependencies...
echo.

REM Check server dependencies
cd /d "%~dp0..\..\server"
if not exist "node_modules" (
    echo Installing server dependencies...
    "%NODE_PATH%\npm.cmd" install
) else (
    echo ✅ Server dependencies ready
)

REM Check client dependencies  
cd /d "%~dp0..\..\client"
if not exist "node_modules" (
    echo Installing client dependencies...
    "%NODE_PATH%\npm.cmd" install
) else (
    echo ✅ Client dependencies ready
)

echo.
echo 🚀 Starting servers...
echo.

REM Start backend server
echo Starting backend server on port 5000...
cd /d "%~dp0..\..\server"
start "MLB DFS Backend" /min "%NODE_PATH%\node.exe" index.js

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

REM Start frontend server
echo Starting frontend development server...
cd /d "%~dp0..\..\client"

REM Open browser after a delay
start "" timeout /t 8 /nobreak ^>nul ^& start "" http://localhost:5173

echo.
echo 🌐 Opening MLB DFS Optimizer...
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:5000
echo.
echo ⏳ Please wait while the development server starts...
echo    This may take 20-30 seconds on first run.
echo.

REM Start Vite development server
"%NODE_PATH%\npm.cmd" run dev

echo.
echo 📝 If you see any errors, try:
echo    1. Refresh the browser page
echo    2. Wait a bit longer for the server to start
echo    3. Check that both servers are running
echo.
pause
