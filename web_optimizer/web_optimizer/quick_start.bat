@echo off
echo 🚀 Starting MLB DFS Web Optimizer...
echo.

REM Check if Node.js is accessible
set "NODE_PATH=C:\Program Files\nodejs"
if not exist "%NODE_PATH%\node.exe" (
    echo ❌ Node.js not found at %NODE_PATH%
    echo Please ensure Node.js is installed and accessible
    pause
    exit /b 1
)

echo ✅ Node.js found at %NODE_PATH%
echo.

REM Add Node.js to PATH for this session
set "PATH=%NODE_PATH%;%PATH%"

echo 📦 Installing dependencies...
echo.

REM Install server dependencies
echo Installing server dependencies...
cd /d "%~dp0server"
"%NODE_PATH%\npm.cmd" install
if errorlevel 1 (
    echo ❌ Failed to install server dependencies
    pause
    exit /b 1
)

echo ✅ Server dependencies installed
echo.

REM Install client dependencies
echo Installing client dependencies...
cd /d "%~dp0client"
"%NODE_PATH%\npm.cmd" install
if errorlevel 1 (
    echo ❌ Failed to install client dependencies
    pause
    exit /b 1
)

echo ✅ Client dependencies installed
echo.

echo 🚀 Starting servers...
echo.

REM Start backend server in background
echo Starting backend server on port 5000...
cd /d "%~dp0server"
start "DFS Backend Server" cmd /k "echo Backend Server Running on http://localhost:5000 && node index.js"

REM Wait a moment for server to start
timeout /t 3 /nobreak

REM Start frontend client
echo Starting frontend client on port 3000...
cd /d "%~dp0client"
echo Frontend will open automatically in your browser...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
echo Press Ctrl+C in either window to stop the servers
echo.
"%NODE_PATH%\npm.cmd" start
)

REM Install client dependencies
echo Installing client dependencies...
cd /d "%~dp0client"
"%NODE_PATH%\npm.cmd" install
if errorlevel 1 (
    echo ❌ Failed to install client dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.
echo 🌐 Starting servers...
echo.

REM Start backend server in background
echo Starting backend server on port 5000...
cd /d "%~dp0server"
start "MLB DFS Backend" "%NODE_PATH%\node.exe" index.js

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend server
echo Starting frontend server on port 3000...
cd /d "%~dp0client"
echo This will open your browser automatically...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo.
"%NODE_PATH%\npm.cmd" start

pause
