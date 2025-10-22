# DFS Optimizer Launchers

This directory contains organized launcher scripts for the DFS Web Optimizer.

## 📁 Directory Structure

```
launchers/
├── windows/           # Windows batch file launchers
│   ├── final_start.bat
│   ├── quick_start.bat
│   ├── simple_start.bat
│   ├── start.bat
│   ├── start_backend.bat
│   ├── start_enhanced.bat
│   └── start_fixed.bat
└── python/            # Cross-platform Python launchers
    ├── launch_optimizer.py
    ├── launch_optimizer_py.bat (Windows wrapper)
    ├── launch_web_optimizer.py
    └── launch_web_optimizer_py.bat (Windows wrapper)
```

## 🚀 Usage

### Windows Users

#### Option 1: Windows Batch Launchers (Recommended for Windows)

Double-click any of these `.bat` files in the `windows/` folder:

- **`quick_start.bat`** - Simplest launcher, installs dependencies and starts both servers
- **`simple_start.bat`** - Enhanced UI with colored output and separate server windows
- **`final_start.bat`** - Production-ready with comprehensive checks and error handling
- **`start.bat`** - Standard launcher with dependency installation
- **`start_enhanced.bat`** - Enhanced version with better logging
- **`start_fixed.bat`** - Version with TypeScript error fixes
- **`start_backend.bat`** - Start backend server only (for debugging)

#### Option 2: Python Launchers (Cross-platform)

From the `python/` folder, double-click:

- **`launch_optimizer_py.bat`** - CLI launcher with colored output
- **`launch_web_optimizer_py.bat`** - GUI launcher with visual controls

**Requirements:**
- Python 3.x installed
- For GUI launcher: `pip install psutil`

### Linux/Mac Users

From the `python/` folder:

```bash
# Make executable (first time only)
chmod +x launch_optimizer.py
chmod +x launch_web_optimizer.py

# Run
./launch_optimizer.py           # CLI version
./launch_web_optimizer.py       # GUI version
```

**Requirements:**
- Python 3.x installed
- For GUI launcher: `pip install psutil`

## 🔧 How They Work

All launchers:
1. Check for Node.js installation
2. Install dependencies if needed (server & client)
3. Start backend server on port 5000
4. Start frontend server on port 5173 (Vite)
5. Open browser to http://localhost:5173

## 🌐 Access URLs

- **Frontend**: http://localhost:5173 (Vite dev server)
- **Backend API**: http://localhost:5000 (Node.js/Express)

## 📝 Choosing a Launcher

### For Windows Users:
- **Beginners**: Use `quick_start.bat` or `simple_start.bat`
- **Advanced**: Use `final_start.bat` for production-like setup
- **Debugging**: Use `start_backend.bat` to test backend only

### For Cross-Platform:
- **CLI preference**: Use `launch_optimizer.py` (via `.bat` wrapper on Windows)
- **GUI preference**: Use `launch_web_optimizer.py` (via `.bat` wrapper on Windows)
- **Sharing with Linux/Mac users**: Python launchers work on all platforms

## 🛑 Stopping Servers

- **Windows batch launchers**: Press `Ctrl+C` in the server windows, or close the windows
- **Python launchers**: Press `Ctrl+C` in the terminal, or close the launcher window
- **GUI launcher**: Click "Stop All Servers" button

## 💡 Troubleshooting

### "Node.js not found"
- Install Node.js from https://nodejs.org
- Make sure to check "Add to PATH" during installation
- Restart your terminal/launcher after installation

### "Python not found" (for Python launchers)
- Install Python from https://python.org
- Make sure to check "Add Python to PATH" during installation
- Try `python`, `python3`, or `py` command in terminal

### "Port already in use"
- Close any existing Node.js processes
- Windows: Open Task Manager and end `node.exe` processes
- Or run a launcher that kills existing processes first

### Servers won't start
1. Check that you're in the correct directory
2. Verify Node.js is installed: `node --version`
3. Try deleting `node_modules` folders and reinstalling
4. Check for errors in the terminal output

## 📚 Documentation

For more information, see:
- `web_optimizer/docs/guides/` - User guides and quick starts
- `web_optimizer/docs/frontend/` - Frontend development docs
- `web_optimizer/docs/deployment/` - Deployment guides

---

*Last updated: October 18, 2025*







