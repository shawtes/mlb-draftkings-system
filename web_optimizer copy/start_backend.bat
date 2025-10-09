@echo off
echo Starting DFS Optimizer Backend Server...
cd /d "%~dp0\server"
echo Current directory: %CD%
echo Starting server on port 5000...
node index.js
pause
