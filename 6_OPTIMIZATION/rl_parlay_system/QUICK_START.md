# 🚀 RL Parlay System - Quick Start

## 📁 What's in this folder?

This folder contains a complete **Reinforcement Learning system** that learns to generate optimal NFL parlays by analyzing 3 years of historical data.

## 🎯 What it does

The AI agent learns to:
- ✅ Select the best players for parlay legs
- ✅ Choose optimal prop types (yards, receptions, etc.)
- ✅ Set realistic lines (70% of projections for safety)
- ✅ Create diverse, high-value parlay combinations
- ✅ Balance hit rate with potential payout

## 🚀 How to use it

### Option 1: Quick Demo
```bash
python main.py demo
```
*Runs with sample data to show how it works*

### Option 2: Full System
```bash
# 1. Setup
python setup.py

# 2. Collect real data (needs API key)
python main.py collect --api-key YOUR_API_KEY

# 3. Train the AI
python main.py train --api-key YOUR_API_KEY

# 4. Generate parlays
python main.py gui
```

### Option 3: From parent directory
```bash
# From 6_OPTIMIZATION folder
python run_rl_parlay.py demo
python run_rl_parlay.py gui
```

## 📊 What you get

- **Smart Parlays**: AI learns from 3 years of data
- **Realistic Lines**: 70% of projections for higher hit rates
- **Diverse Combinations**: Mix of players, teams, prop types
- **Expected Value**: Calculates profit potential
- **GUI Interface**: Easy to use, no coding required

## 🎮 GUI Features

1. **Data Loading**: Load your NFL data files
2. **Model Management**: Load trained AI models
3. **Parlay Generation**: Create multiple parlays
4. **Results Analysis**: Analyze and export results

## 📈 Example Output

```
--- Parlay 1 ---
Legs: 3
Hit Rate: 68%
Odds: +150
Expected Value: $18.50

1. Josh Allen (BUF) - Passing Yards O245 (85%)
2. Stefon Diggs (BUF) - Receiving Yards O75 (80%)
3. Derrick Henry (TEN) - Rushing Yards O85 (75%)
```

## 🔧 Requirements

- Python 3.7+
- PyTorch
- Pandas, NumPy
- Matplotlib
- Tkinter (for GUI)

## 📚 Full Documentation

See `RL_PARLAY_README.md` for complete documentation.

---

**Ready to generate smarter parlays? Run `python main.py demo` to get started!** 🏈🤖










