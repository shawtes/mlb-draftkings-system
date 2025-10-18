@echo off
REM Windows wrapper for launch_web_optimizer.py (GUI version)
REM This allows Windows users to run the Python launcher without changing the Linux-compatible .py file

REM Try to find Python and run the GUI launcher
where python >nul 2>&1
if %errorlevel% equ 0 (
    start "" pythonw "%~dp0launch_web_optimizer.py"
) else (
    where python3 >nul 2>&1
    if %errorlevel% equ 0 (
        start "" python3 "%~dp0launch_web_optimizer.py"
    ) else (
        where py >nul 2>&1
        if %errorlevel% equ 0 (
            start "" pythonw "%~dp0launch_web_optimizer.py"
        ) else (
            echo ❌ Python not found!
            echo.
            echo Please install Python from https://python.org
            echo Make sure to check "Add Python to PATH" during installation
            echo.
            echo Also make sure to install these Python packages:
            echo   pip install psutil
            echo.
            pause
            exit /b 1
        )
    )
)

