# 🚀 Quick Start Guide - UrSim DFS Optimizer

## One-Click Startup

Simply double-click: **`simple_start.bat`**

This will automatically:
1. ✅ Start the backend server on port 5000
2. ✅ Start the UrSim frontend on port 3000
3. ✅ Install dependencies if needed (first time only)
4. ✅ Open your browser automatically

## What You'll See

### Terminal Output:
```
🚀 Starting UrSim DFS Optimizer (Integrated Frontend)...
✅ Node.js path set
📡 Starting backend server...
🌐 Starting UrSim frontend (Vite)...

🎯 Opening UrSim DFS Optimizer...
════════════════════════════════════════
📡 Backend:  http://localhost:5000
🌐 Frontend: http://localhost:3000
════════════════════════════════════════

🔥 Features Available:
   ✅ UrSim Dashboard
   ✅ DFS Optimizer (All Tabs)
   ✅ Lineup Builder
   ✅ Game Analysis
   ✅ Firebase Authentication
```

### Two Windows Will Open:
1. **Backend window** - Shows server logs
2. **Frontend window** - Shows Vite dev server

## First Time Setup

**If this is your first time:**
- Dependencies will install automatically (5-10 minutes)
- Be patient during the first run!
- Subsequent runs will be much faster

## Using the Application

### 1. Homepage
- You'll see the UrSim landing page
- Click **"Login"** or **"Sign Up"**

### 2. Dashboard
After logging in:
- **Games Overview** - View active games
- **Props & Stacks** - Prop bet finder
- **Game Matchups** - Game analysis
- **Lineup Builder** - Build lineups
- **DFS Optimizer** ← Click here for full DFS features!
- **Settings** - Account settings

### 3. DFS Optimizer
Click "DFS Optimizer" in the sidebar to access:
- **Players Tab** - Upload player CSV files
- **Team Stacks** - Configure team stacking
- **Stack Exposure** - Set exposure limits
- **Team Combos** - Team combinations
- **Control Panel** - Run optimizations
- **Favorites** - Manage favorites
- **Results** - View/export lineups

## Manual Startup (Alternative)

If you prefer to start manually:

### Terminal 1 - Backend:
```bash
cd mlb-draftkings-system/web_optimizer/server
node index.js
```

### Terminal 2 - Frontend:
```bash
cd mlb-draftkings-system/web_optimizer/client/ursimfrontend
npm run dev
```

## Stopping the Application

**To stop everything:**
1. Close the browser
2. In each terminal window, press **Ctrl+C**
3. Or just close the terminal windows

## Troubleshooting

### Port Already in Use
If you see port errors:
```bash
# Kill processes on ports 3000 or 5000
netstat -ano | findstr :3000
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

### Dependencies Not Installing
```bash
# Manually install in frontend directory
cd client/ursimfrontend
npm install
```

### Backend Won't Start
Check that Node.js is installed:
```bash
node --version
npm --version
```

### White Page in Browser
- Make sure npm install completed successfully
- Check the terminal for any error messages
- Try refreshing the browser (Ctrl+F5)

## Ports Used

- **Frontend**: http://localhost:3000 (Vite dev server)
- **Backend**: http://localhost:5000 (Node.js API)
- **WebSocket**: ws://localhost:5000/ws (Real-time updates)

## System Requirements

- **Node.js**: v16 or higher
- **npm**: v8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Browser**: Chrome, Firefox, or Edge (latest version)

## Project Structure

```
web_optimizer/
├── simple_start.bat          ← Double-click to start!
├── server/                   ← Backend (Node.js)
│   └── index.js
└── client/
    └── ursimfrontend/        ← Frontend (Vite + React)
        ├── src/
        │   ├── components/
        │   │   ├── dfs/      ← DFS Optimizer components
        │   │   ├── Dashboard.tsx
        │   │   └── ...
        │   └── App.tsx
        └── package.json
```

## Next Steps

1. **Upload Player Data** - Go to Players tab, upload CSV
2. **Configure Stacks** - Set team stacking strategies
3. **Set Exposures** - Define exposure limits
4. **Optimize** - Run optimization from Control Panel
5. **Export Lineups** - Download results in DraftKings format

---

**Need Help?** Check the console logs in both terminal windows for error messages.

**Enjoy!** 🎉

