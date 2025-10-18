@echo off
REM Windows wrapper for launch_optimizer.py
REM This allows Windows users to run the Python launcher without changing the Linux-compatible .py file

echo.
echo 🚀 Starting DFS Optimizer Launcher (Python)...
echo.

REM Try to find Python
where python >nul 2>&1
if %errorlevel% equ 0 (
    python "%~dp0launch_optimizer.py"
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        python3 "%~dp0launch_optimizer.py"
    ) else (
        where py >nul 2>&1
        if %errorlevel% equ 0 (
            py "%~dp0launch_optimizer.py"
        ) else (
            echo ❌ Python not found!
            echo.
            echo Please install Python from https://python.org
            echo Make sure to check "Add Python to PATH" during installation
            echo.
            pause
            exit /b 1
        )
    )
)

pause

