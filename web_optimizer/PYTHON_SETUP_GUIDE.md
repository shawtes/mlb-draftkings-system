# Python Setup Guide for Real-Time Data Feature

## The Problem
Your Node.js server can't find Python, even though `py` works in your terminal. This is because the server process doesn't have Python in its PATH.

## Solution: Install Python Properly

### Option 1: Install from python.org (Recommended)

1. **Download Python:**
   - Go to: https://www.python.org/downloads/
   - Click "Download Python 3.x.x" (latest version)
   - Save the installer

2. **Install Python:**
   - Run the installer
   - **IMPORTANT:** Check the box "Add Python to PATH" at the bottom
   - Click "Install Now"
   - Wait for installation to complete

3. **Verify Installation:**
   - Open a NEW terminal/PowerShell window
   - Run: `python --version`
   - You should see: `Python 3.x.x`

4. **Restart Your Server:**
   - Stop your Node.js server (Ctrl+C)
   - Start it again: `npm start` or `node server/index.js`
   - The server will now have Python in its PATH

### Option 2: Use Windows Store

1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Install"
4. Wait for installation
5. Restart your server

### Option 3: Add Python to PATH Manually

If Python is already installed but not in PATH:

1. Find where Python is installed:
   - Common locations:
     - `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python3xx\`
     - `C:\Program Files\Python3xx\`

2. Add to PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab
   - Click "Environment Variables"
   - Under "System variables", find "Path"
   - Click "Edit"
   - Click "New"
   - Add: `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python3xx\`
   - Add: `C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python3xx\Scripts\`
   - Click OK on all dialogs

3. **Restart your computer** (or at least restart your terminal and server)

## Verify It Works

After installing/restarting:

1. Open a NEW terminal
2. Run: `python --version`
3. Should show: `Python 3.x.x`
4. Restart your Node.js server
5. Try the Real-Time button again

## Troubleshooting

**Still not working?**
- Make sure you restarted your server after installing Python
- Try restarting your computer
- Check server console for Python detection logs
- The code will try to find Python automatically in common locations


