# 🚀 MLB DFS Web Optimizer - Quick Start Guide

## ✅ Your optimizer is now working!

### 🌐 Access URLs
- **Frontend (React App)**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 📋 Easy Startup Methods

### Method 1: Simple Batch File (Recommended)
```bash
cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer
simple_start.bat
```

### Method 2: Manual Start
1. **Start Backend**:
   ```bash
   cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer\server
   "C:\Program Files\nodejs\node.exe" index.js
   ```

2. **Start Frontend** (in new terminal):
   ```bash
   cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer\client
   $env:PATH = "C:\Program Files\nodejs;$env:PATH"
   npm.cmd start
   ```

## 🔧 Troubleshooting

### If frontend won't start:
```bash
cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer\client
$env:PATH = "C:\Program Files\nodejs;$env:PATH"
npm.cmd install
npm.cmd start
```

### If backend won't start:
```bash
cd c:\Users\smtes\Downloads\coinbase_ml_trader\web_optimizer\server
"C:\Program Files\nodejs\npm.cmd" install
"C:\Program Files\nodejs\node.exe" index.js
```

### If ports are in use:
```bash
taskkill /f /im node.exe
```
Then restart both servers.

## 🎯 Test Your Optimizer

1. Open http://localhost:3000 in your browser
2. Upload the sample file: `web_optimizer/sample_players.csv`
3. Configure optimization settings
4. Run optimization and see real-time results!

## 📊 Features Available

✅ **Player Upload**: CSV file processing  
✅ **Team Stacking**: Multi-player team strategies  
✅ **Exposure Control**: Player exposure limits  
✅ **Real-time Progress**: Live optimization updates  
✅ **Export Results**: Download optimized lineups  
✅ **Modern UI**: Professional web interface  

## 💡 Success Indicators

- ✅ Backend shows: "DFS Optimizer Server running on port 5000"
- ✅ Frontend shows React development server starting
- ✅ Browser opens to the optimizer interface
- ✅ Sample CSV uploads successfully

Your modern MLB DFS web optimizer is ready to use! 🚀⚾
