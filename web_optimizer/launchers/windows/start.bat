@echo off
echo Starting MLB DFS Optimizer Web Application...
echo ==============================================

REM Fix Node.js PATH if needed
echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js not in PATH. Searching for installation...
    
    if exist "C:\Program Files\nodejs\node.exe" (
        echo Found Node.js at: C:\Program Files\nodejs
        set "PATH=%PATH%;C:\Program Files\nodejs"
    ) else if exist "C:\Program Files (x86)\nodejs\node.exe" (
        echo Found Node.js at: C:\Program Files ^(x86^)\nodejs
        set "PATH=%PATH%;C:\Program Files (x86)\nodejs"
    ) else (
        echo ERROR: Node.js not found! Please install from https://nodejs.org/
        pause
        exit /b 1
    )
)

echo Node.js is ready!
node --version
npm --version

echo.
echo Installing server dependencies...
cd ..\..\server
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install server dependencies
    pause
    exit /b 1
)

echo.
echo Installing client dependencies...
cd ..\client
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install client dependencies
    pause
    exit /b 1
)

echo.
echo SUCCESS! All dependencies installed.
echo.
echo Starting backend server...
cd ..\server
start "MLB DFS Optimizer - Backend" cmd /k "npm start"

echo.
echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting frontend development server...
cd ..\client
start "MLB DFS Optimizer - Frontend" cmd /k "npm run dev"

echo.
echo ==============================================
echo MLB DFS Optimizer is starting!
echo ==============================================
echo Backend API: http://localhost:5000
echo Frontend UI: http://localhost:5173
echo.
echo The web application will open automatically in your browser.
echo Both server windows will remain open for monitoring.
echo.
echo Close this window when done, or press any key to continue...
pause
