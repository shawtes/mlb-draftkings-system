@echo off
echo 🚀 Starting UrSim DFS Optimizer (Integrated Frontend)...
echo.

REM Set Node.js path
set "NODE_PATH=C:\Program Files\nodejs"
set "PATH=%NODE_PATH%;%PATH%"

echo ✅ Node.js path set
echo.

echo 📡 Starting backend server in new window...
start "🔧 UrSim Backend - Port 5000" cmd /k "cd /d "%~dp0server" && echo ════════════════════════════════════════ && echo 📡 BACKEND SERVER STARTING... && echo ════════════════════════════════════════ && echo. && "%NODE_PATH%\node.exe" index.js"

REM Wait for backend to initialize
echo ⏳ Waiting for backend to start...
timeout /t 5 /nobreak > nul

echo.
echo 🌐 Starting frontend server in new window...

REM Check if node_modules exists first
cd /d "%~dp0client\ursimfrontend"
if not exist "node_modules" (
    echo.
    echo ⚠️  First time setup detected!
    echo 📦 Installing dependencies in frontend window...
    echo    This may take 5-10 minutes...
    echo.
    start "📦 Installing Dependencies" cmd /k "cd /d "%~dp0client\ursimfrontend" && echo Installing UrSim Frontend Dependencies... && echo This will take 5-10 minutes on first run... && echo. && "%NODE_PATH%\npm.cmd" install && echo. && echo ✅ Installation complete! && echo. && echo 🌐 Starting Vite dev server... && echo. && "%NODE_PATH%\npm.cmd" run dev"
) else (
    start "🌐 UrSim Frontend - Port 3000" cmd /k "cd /d "%~dp0client\ursimfrontend" && echo ════════════════════════════════════════ && echo 🌐 FRONTEND SERVER STARTING... && echo ════════════════════════════════════════ && echo. && "%NODE_PATH%\npm.cmd" run dev"
)

echo.
echo ✅ Both servers are starting in separate windows!
echo.
echo ════════════════════════════════════════
echo 📡 Backend:  http://localhost:5000
echo 🌐 Frontend: http://localhost:3000
echo ════════════════════════════════════════
echo.
echo 🔥 Features Available:
echo    ✅ UrSim Dashboard
echo    ✅ DFS Optimizer (All Tabs)
echo    ✅ Lineup Builder
echo    ✅ Game Analysis
echo    ✅ Firebase Authentication
echo.
echo 💡 Tips:
echo    • Check each window for errors
echo    • Backend window shows API logs
echo    • Frontend window shows build errors
echo    • Login/Register to access dashboard
echo    • Click "DFS Optimizer" in sidebar
echo.
echo 🎯 The browser should open automatically...
echo.
echo Press any key to close this launcher window...
echo (The servers will keep running in their own windows)
echo.
pause > nul
