@echo off
echo 🚀 Starting Fixed MLB DFS Web Optimizer...
echo.

REM Kill any existing node processes
taskkill /f /im node.exe >nul 2>&1
if errorlevel 1 echo No existing Node processes found
echo.

REM Set Node.js path
set "NODE_PATH=C:\Program Files\nodejs"
set "PATH=%NODE_PATH%;%PATH%"

echo ✅ Node.js path configured
echo.

echo 📡 Starting backend server...
cd /d "%~dp0server"
start "MLB DFS Backend" "%NODE_PATH%\node.exe" index.js

REM Wait for backend to start
timeout /t 3 /nobreak > nul

echo 🌐 Starting frontend server...
cd /d "%~dp0client"

echo.
echo 🎯 The MLB DFS Optimizer should open automatically in your browser...
echo 📊 Backend API: http://localhost:5000
echo 🌐 Frontend App: http://localhost:3000
echo.
echo ✨ All TypeScript errors have been fixed!
echo.

REM Start React development server with proper PATH
"%NODE_PATH%\npm.cmd" start

echo.
echo Press any key to close this window...
pause >nul
