#!/bin/bash

echo "================================================================================"
echo "🏈 NFL DFS OPTIMIZER - STARTUP SCRIPT"
echo "================================================================================"

cd "$(dirname "$0")"

# Check if psutil is installed
echo ""
echo "1️⃣ Checking dependencies..."
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ⚠️  psutil not installed"
    echo "   Installing psutil..."
    pip3 install psutil
    if [ $? -ne 0 ]; then
        echo "   ❌ Failed to install psutil"
        echo "   Try manually: pip3 install psutil"
        exit 1
    fi
    echo "   ✅ psutil installed"
else
    echo "   ✅ psutil already installed"
fi

# Check if data file exists
echo ""
echo "2️⃣ Checking data file..."
if [ ! -f "nfl_week7_1PM_SLATE_READY_FOR_OPTIMIZER.csv" ]; then
    echo "   ❌ Data file not found!"
    echo "   Expected: nfl_week7_1PM_SLATE_READY_FOR_OPTIMIZER.csv"
    echo ""
    echo "   Available CSV files:"
    ls -1 *.csv 2>/dev/null | head -10
    exit 1
fi

# Test load the data
echo "   Testing data load..."
python3 test_load_data.py > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "   ⚠️  Data file has issues"
    echo "   Running full diagnostic..."
    python3 test_load_data.py
    exit 1
fi
echo "   ✅ Data file verified (246 players)"

# Start the optimizer
echo ""
echo "3️⃣ Starting NFL DFS Optimizer..."
echo ""
echo "================================================================================"
echo "🎯 INSTRUCTIONS:"
echo "================================================================================"
echo ""
echo "When the GUI opens:"
echo "  1. Click 'Load Players' button"
echo "  2. Select: nfl_week7_1PM_SLATE_READY_FOR_OPTIMIZER.csv"
echo "  3. Wait for tables to populate (246 players)"
echo "  4. Configure your settings:"
echo "     • Contest Type: GPP or Cash"
echo "     • Stack Type: QB + 2WR or Game Stack"
echo "     • Teams: GB, PHI, DET/TB"
echo "     • Number of Lineups: 15-20"
echo "  5. Click 'Generate Lineups'"
echo "  6. Click 'Save Lineups' when done"
echo ""
echo "================================================================================"
echo ""

# Launch the optimizer
python3 genetic_algo_nfl_optimizer.py

echo ""
echo "================================================================================"
echo "Optimizer closed."
echo "================================================================================"

