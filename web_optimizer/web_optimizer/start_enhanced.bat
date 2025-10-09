@echo off
cd /d "c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer"

echo ========================================
echo Starting Enhanced DFS Optimizer...
echo ========================================

REM Start backend server
echo Starting backend server...
start "DFS Backend" cmd /k "cd server && node index.js"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting frontend...
start "DFS Frontend" cmd /k "cd client && npm start"

echo.
echo ========================================
echo DFS Optimizer is starting up...
echo Backend: http://localhost:8080
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to exit this window...
pause >nul
