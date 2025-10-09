@echo off
echo 🚀 Starting MLB DFS Web Optimizer...
echo.

REM Set Node.js path
set "NODE_PATH=C:\Program Files\nodejs"
set "PATH=%NODE_PATH%;%PATH%"

echo ✅ Node.js path set
echo.

REM Check if backend is already running
echo 📡 Starting backend server...
cd /d "%~dp0server"
start "MLB DFS Backend" "%NODE_PATH%\node.exe" index.js

REM Wait for backend to start
timeout /t 3 /nobreak > nul

echo 🌐 Starting frontend server...
cd /d "%~dp0client"

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies first...
    "%NODE_PATH%\npm.cmd" install
)

echo.
echo 🎯 Opening MLB DFS Optimizer...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo The browser should open automatically...
echo.

REM Start React development server
"%NODE_PATH%\npm.cmd" start

pause
